"""
Interactive Streamlit Platform for Neuroinflammation and Survival Analysis
=========================================================================

This Streamlit application provides an interactive interface to explore the effects
of various parameters on neuroinflammation trajectories, senescence trajectories,
survival curves, and statistical measures.

Run with: streamlit run interactive_perturbation_platform.py
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy import stats
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import seaborn as sns

# Import functions from mixed_population_trial.py
# Since both files are in the same directory, we can import directly
try:
    from mixed_population_trial import (
        simulate_mixed_population_trial,
        calculate_statistics
    )
except ImportError:
    # Fallback: add current directory to path
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from mixed_population_trial import (
        simulate_mixed_population_trial,
        calculate_statistics
    )

# Base parameters (APOE33) - matching mixed_population_trial.py
BASE_PARAMS = {
    'eta': 8.00e-03/(365.0**2),
    'beta': 3*4.93e-01/(365.0),
    'kappa': 1.00e+00,
    'Nc': 5.99e+00,
    'epsilon_p': 0*(1.13e-02)/(365.0),
    'b': 5*3.35e-02/(365.0),
    'beta_tag': 8.42e-02/(365.0),
    'kappa_tag': 1.00e+00,
    'c': 6.03e-02,
    'alpha': 3.38e-02/(365.0),
    'epsilon_n': 3.91e-02/(365.0)
}

# Set page config
st.set_page_config(
    page_title="AD Model Interactive Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

plt.style.use('default')

# Initialize session state for caching control
if 'simulation_results' not in st.session_state:
    st.session_state.simulation_results = None
if 'current_params' not in st.session_state:
    st.session_state.current_params = None

@st.cache_data
def run_simulation_cached(
    base_params, n_per_arm, mci_threshold, s_fold_change,
    interval_days, treatment_duration_days, follow_up_years,
    dt, base_seed, sample_every_days
):
    """
    Cached simulation function to avoid re-running expensive computations.
    """
    control_data, treatment_data = simulate_mixed_population_trial(
        base_params, n_per_arm, mci_threshold, s_fold_change,
        interval_days, treatment_duration_days, follow_up_years,
        dt, base_seed, sample_every_days
    )
    return control_data, treatment_data

def calculate_trajectory_statistics(control_data, treatment_data, mci_threshold, follow_up_years):
    """
    Calculate statistics from trajectories at follow-up time.
    """
    control_trajs = control_data['trajectories']
    treatment_trajs = treatment_data['trajectories']
    treatment_start_times = treatment_data.get('treatment_start_times', [])
    
    control_n_at_followup = []
    treatment_n_at_followup = []
    
    # Control arm: measure n at follow_up_years after MCI threshold crossing
    for traj in control_trajs:
        t, n = traj['t'], traj['n']
        mci_cross_idx = np.where(n >= mci_threshold)[0]
        if len(mci_cross_idx) > 0:
            mci_cross_time = t[mci_cross_idx[0]]
            follow_up_time = mci_cross_time + follow_up_years
            if follow_up_time <= t[-1]:
                f = interp1d(t, n, kind='linear', bounds_error=False, fill_value=np.nan)
                n_at_followup = f(follow_up_time)
                if np.isfinite(n_at_followup):
                    control_n_at_followup.append(n_at_followup)
    
    # Treatment arm: measure n at follow_up_years after treatment initiation
    for i, traj in enumerate(treatment_trajs):
        t, n = traj['t'], traj['n']
        if i < len(treatment_start_times) and treatment_start_times[i] > 0:
            treatment_start_time = treatment_start_times[i]
            follow_up_time = treatment_start_time + follow_up_years
            if follow_up_time <= t[-1]:
                f = interp1d(t, n, kind='linear', bounds_error=False, fill_value=np.nan)
                n_at_followup = f(follow_up_time)
                if np.isfinite(n_at_followup):
                    treatment_n_at_followup.append(n_at_followup)
    
    # Calculate statistics
    stats_results = None
    if len(control_n_at_followup) > 0 and len(treatment_n_at_followup) > 0:
        stats_results = calculate_statistics(
            np.array(control_n_at_followup),
            np.array(treatment_n_at_followup),
            control_data['onset_times'],
            treatment_data['onset_times']
        )
    
    return stats_results

def plot_neuroinflammation_trajectories_streamlit(control_data, treatment_data, age_range=(50, 90), stats_results=None, n_per_arm=None):
    """Plot neuroinflammation trajectories adapted for Streamlit."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Control arm
    control_trajs = control_data['trajectories']
    control_n_arrays = [traj['n'] for traj in control_trajs]
    control_t_arrays = [traj['t'] for traj in control_trajs]
    
    # Treatment arm
    treatment_trajs = treatment_data['trajectories']
    treatment_n_arrays = [traj['n'] for traj in treatment_trajs]
    treatment_t_arrays = [traj['t'] for traj in treatment_trajs]
    
    # Find common time range
    min_t_all = max([max([t[0] for t in control_t_arrays]), max([t[0] for t in treatment_t_arrays])])
    max_t_all = min([min([t[-1] for t in control_t_arrays]), min([t[-1] for t in treatment_t_arrays])])
    # Limit to specified age range
    min_t_all = max(min_t_all, age_range[0])
    max_t_all = min(max_t_all, age_range[1])
    t_common = np.linspace(min_t_all, max_t_all, 500)
    
    # Interpolate control trajectories
    n_interp_control = []
    for t, n in zip(control_t_arrays, control_n_arrays):
        f_n = interp1d(t, n, kind='linear', bounds_error=False, fill_value=np.nan)
        n_interp_control.append(f_n(t_common))
    
    n_interp_control = np.array(n_interp_control)
    n_mean_control = np.nanmean(n_interp_control, axis=0)
    n_std_control = np.nanstd(n_interp_control, axis=0)
    n_se_control = n_std_control / np.sqrt(len(control_trajs))
    
    # Interpolate treatment trajectories
    n_interp_treatment = []
    for t, n in zip(treatment_t_arrays, treatment_n_arrays):
        f_n = interp1d(t, n, kind='linear', bounds_error=False, fill_value=np.nan)
        n_interp_treatment.append(f_n(t_common))
    
    n_interp_treatment = np.array(n_interp_treatment)
    n_mean_treatment = np.nanmean(n_interp_treatment, axis=0)
    n_std_treatment = np.nanstd(n_interp_treatment, axis=0)
    n_se_treatment = n_std_treatment / np.sqrt(len(treatment_trajs))
    
    # Calculate difference at age 80
    age_80_idx = np.argmin(np.abs(t_common - 80))
    if age_80_idx < len(t_common):
        n_diff_at_80 = n_mean_control[age_80_idx] - n_mean_treatment[age_80_idx]
    else:
        n_diff_at_80 = np.nan
    
    # Plot trajectories
    ax.plot(t_common, n_mean_control, color='blue', linewidth=2.5, label='Control Mean')
    ax.fill_between(t_common, n_mean_control - n_se_control, 
                     n_mean_control + n_se_control, color='blue', alpha=0.2, label='Control ±1 SE')
    
    ax.plot(t_common, n_mean_treatment, color='red', linewidth=2.5, label='Treatment Mean')
    ax.fill_between(t_common, n_mean_treatment - n_se_treatment, 
                     n_mean_treatment + n_se_treatment, color='red', alpha=0.2, label='Treatment ±1 SE')
    
    # Add annotation
    annotation_text = []
    if np.isfinite(n_diff_at_80):
        annotation_text.append(f'Δn at age 80: {n_diff_at_80:.3f}')
    if stats_results is not None:
        p_value = stats_results.get('p_value_ttest', np.nan)
        cohens_d = stats_results.get('cohens_d', np.nan)
        if np.isfinite(p_value):
            annotation_text.append(f'p-value: {p_value:.4f}')
        if np.isfinite(cohens_d):
            annotation_text.append(f"Cohen's d: {cohens_d:.3f}")
    
    if len(annotation_text) > 0:
        ax.text(0.05, 0.95, '\n'.join(annotation_text), 
                transform=ax.transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel('Age (years)', fontsize=12)
    ax.set_ylabel('Neuroinflammation (n)', fontsize=12)
    title = 'Neuroinflammation Trajectories'
    if n_per_arm is not None:
        title += f' (n={n_per_arm} per arm)'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    ax.set_xlim(age_range[0], age_range[1])
    ax.set_ylim(1, 10)
    sns.despine(ax=ax)
    
    plt.tight_layout()
    return fig

def plot_senescence_trajectories_streamlit(control_data, treatment_data, age_range=(50, 90), n_per_arm=None):
    """Plot senescence trajectories adapted for Streamlit."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Control arm
    control_trajs = control_data['trajectories']
    control_s_arrays = [traj['s'] for traj in control_trajs]
    control_t_arrays = [traj['t'] for traj in control_trajs]
    
    # Treatment arm
    treatment_trajs = treatment_data['trajectories']
    treatment_s_arrays = [traj['s'] for traj in treatment_trajs]
    treatment_t_arrays = [traj['t'] for traj in treatment_trajs]
    
    # Find common time range
    min_t_all = max([max([t[0] for t in control_t_arrays]), max([t[0] for t in treatment_t_arrays])])
    max_t_all = min([min([t[-1] for t in control_t_arrays]), min([t[-1] for t in treatment_t_arrays])])
    # Limit to specified age range
    min_t_all = max(min_t_all, age_range[0])
    max_t_all = min(max_t_all, age_range[1])
    t_common = np.linspace(min_t_all, max_t_all, 500)
    
    # Interpolate control trajectories
    s_interp_control = []
    for t, s in zip(control_t_arrays, control_s_arrays):
        f_s = interp1d(t, s, kind='linear', bounds_error=False, fill_value=np.nan)
        s_interp_control.append(f_s(t_common))
    
    s_interp_control = np.array(s_interp_control)
    s_mean_control = np.nanmean(s_interp_control, axis=0)
    s_std_control = np.nanstd(s_interp_control, axis=0)
    s_se_control = s_std_control / np.sqrt(len(control_trajs))
    
    # Interpolate treatment trajectories
    s_interp_treatment = []
    for t, s in zip(treatment_t_arrays, treatment_s_arrays):
        f_s = interp1d(t, s, kind='linear', bounds_error=False, fill_value=np.nan)
        s_interp_treatment.append(f_s(t_common))
    
    s_interp_treatment = np.array(s_interp_treatment)
    s_mean_treatment = np.nanmean(s_interp_treatment, axis=0)
    s_std_treatment = np.nanstd(s_interp_treatment, axis=0)
    s_se_treatment = s_std_treatment / np.sqrt(len(treatment_trajs))
    
    # Plot trajectories
    ax.plot(t_common, s_mean_control, color='blue', linewidth=2.5, label='Control Mean')
    ax.fill_between(t_common, s_mean_control - s_se_control, 
                     s_mean_control + s_se_control, color='blue', alpha=0.2, label='Control ±1 SE')
    
    ax.plot(t_common, s_mean_treatment, color='red', linewidth=2.5, label='Treatment Mean')
    ax.fill_between(t_common, s_mean_treatment - s_se_treatment, 
                     s_mean_treatment + s_se_treatment, color='red', alpha=0.2, label='Treatment ±1 SE')
    
    ax.set_xlabel('Age (years)', fontsize=12)
    ax.set_ylabel('Senescence (s)', fontsize=12)
    title = 'Senescence Trajectories'
    if n_per_arm is not None:
        title += f' (n={n_per_arm} per arm)'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.set_xlim(age_range[0], age_range[1])
    sns.despine(ax=ax)
    
    plt.tight_layout()
    return fig

def plot_survival_curves_streamlit(control_data, treatment_data, stats_results=None, n_per_arm=None):
    """Plot survival curves adapted for Streamlit."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Prepare data for lifelines
    control_onset_times = control_data['onset_times']
    control_events = np.ones(len(control_onset_times))
    
    treatment_onset_times = treatment_data['onset_times']
    treatment_events = np.ones(len(treatment_onset_times))
    
    # Calculate delay
    control_onset_clean = control_onset_times[np.isfinite(control_onset_times) & (control_onset_times > 0)]
    treatment_onset_clean = treatment_onset_times[np.isfinite(treatment_onset_times) & (treatment_onset_times > 0)]
    
    if len(control_onset_clean) > 0 and len(treatment_onset_clean) > 0:
        median_control_onset = np.median(control_onset_clean)
        median_treatment_onset = np.median(treatment_onset_clean)
        delay = median_treatment_onset - median_control_onset
    else:
        delay = np.nan
    
    # Perform log-rank test
    p_value_logrank = np.nan
    if len(control_onset_clean) > 0 and len(treatment_onset_clean) > 0:
        try:
            results = logrank_test(control_onset_clean, treatment_onset_clean,
                                  event_observed_A=np.ones(len(control_onset_clean)),
                                  event_observed_B=np.ones(len(treatment_onset_clean)))
            p_value_logrank = results.p_value
        except:
            p_value_logrank = np.nan
    
    # Fit Kaplan-Meier
    kmf_control = KaplanMeierFitter()
    kmf_treatment = KaplanMeierFitter()
    
    kmf_control.fit(control_onset_times, control_events, label='Control')
    kmf_treatment.fit(treatment_onset_times, treatment_events, label='Treatment')
    
    # Plot
    kmf_control.plot_survival_function(ax=ax, color='blue', linewidth=2)
    kmf_treatment.plot_survival_function(ax=ax, color='red', linewidth=2)
    
    # Add annotation
    annotation_lines = []
    if np.isfinite(delay):
        annotation_lines.append(f'Delay: {delay:.2f} years\n(median age difference)')
    if np.isfinite(p_value_logrank):
        annotation_lines.append(f"p-value (log-rank): {p_value_logrank:.4f}")
    if stats_results is not None:
        cohens_d = stats_results.get('cohens_d', np.nan)
        if np.isfinite(cohens_d):
            annotation_lines.append(f"Cohen's d: {cohens_d:.3f}")
    
    if len(annotation_lines) > 0:
        ax.text(0.95, 0.95, '\n'.join(annotation_lines), 
                transform=ax.transAxes, fontsize=12, verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel('Age (years)', fontsize=12)
    ax.set_ylabel('Survival Probability', fontsize=12)
    title = 'Survival Curves: Control vs Treatment'
    if n_per_arm is not None:
        title += f' (n={n_per_arm} per arm)'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc="lower left")
    ax.set_xlim(50, 110)
    sns.despine(ax=ax)
    
    plt.tight_layout()
    return fig

def plot_neuroinflammation_difference_streamlit(control_data, treatment_data, age_range=(50, 90), n_per_arm=None):
    """Plot neuroinflammation difference adapted for Streamlit."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Control arm
    control_trajs = control_data['trajectories']
    control_n_arrays = [traj['n'] for traj in control_trajs]
    control_t_arrays = [traj['t'] for traj in control_trajs]
    
    # Treatment arm
    treatment_trajs = treatment_data['trajectories']
    treatment_n_arrays = [traj['n'] for traj in treatment_trajs]
    treatment_t_arrays = [traj['t'] for traj in treatment_trajs]
    
    # Find common time range
    min_t_all = max([max([t[0] for t in control_t_arrays]), max([t[0] for t in treatment_t_arrays])])
    max_t_all = min([min([t[-1] for t in control_t_arrays]), min([t[-1] for t in treatment_t_arrays])])
    # Limit to specified age range
    min_t_all = max(min_t_all, age_range[0])
    max_t_all = min(max_t_all, age_range[1])
    t_common = np.linspace(min_t_all, max_t_all, 500)
    
    # Interpolate control trajectories
    n_interp_control = []
    for t, n in zip(control_t_arrays, control_n_arrays):
        f_n = interp1d(t, n, kind='linear', bounds_error=False, fill_value=np.nan)
        n_interp_control.append(f_n(t_common))
    
    n_interp_control = np.array(n_interp_control)
    n_mean_control = np.nanmean(n_interp_control, axis=0)
    n_std_control = np.nanstd(n_interp_control, axis=0)
    n_se_control = n_std_control / np.sqrt(len(control_trajs))
    
    # Interpolate treatment trajectories
    n_interp_treatment = []
    for t, n in zip(treatment_t_arrays, treatment_n_arrays):
        f_n = interp1d(t, n, kind='linear', bounds_error=False, fill_value=np.nan)
        n_interp_treatment.append(f_n(t_common))
    
    n_interp_treatment = np.array(n_interp_treatment)
    n_mean_treatment = np.nanmean(n_interp_treatment, axis=0)
    n_std_treatment = np.nanstd(n_interp_treatment, axis=0)
    n_se_treatment = n_std_treatment / np.sqrt(len(treatment_trajs))
    
    # Calculate difference
    diff_n = n_mean_control - n_mean_treatment
    diff_se_n = np.sqrt(n_se_control**2 + n_se_treatment**2)
    
    # Plot difference
    ax.plot(t_common, diff_n, color='blue', linewidth=2.5, label='Δn (Control - Treatment)')
    ax.fill_between(t_common, diff_n - diff_se_n, diff_n + diff_se_n, 
                    color='blue', alpha=0.2, label='±1 SE')
    
    ax.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.3)
    ax.set_xlabel('Age (years)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Δn (Control - Treatment)', fontsize=12, fontweight='bold')
    title = 'Neuroinflammation Difference'
    if n_per_arm is not None:
        title += f' (n={n_per_arm} per arm)'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best', framealpha=0.9)
    ax.set_xlim(age_range[0], age_range[1])
    sns.despine(ax=ax)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    return fig

def main():
    """Main Streamlit application."""
    
    # Title
    st.title("🧠 AD Model Interactive Perturbation Platform")
    st.markdown("Explore the effects of treatment parameters on neuroinflammation, senescence, and survival curves")
    
    # Sidebar for parameters
    st.sidebar.header("Simulation Parameters")
    
    # Base parameters (APOE33)
    st.sidebar.subheader("Base Model Parameters")
    st.sidebar.info("Using APOE33 baseline parameters")
    
    # Treatment parameters
    st.sidebar.subheader("Treatment Parameters")
    
    s_fold_change = st.sidebar.slider(
        "s reduction factor (s_fold_change)",
        min_value=0.1,
        max_value=0.9,
        value=0.3,
        step=0.05,
        help="Factor multiplying s (0.3 = 70% reduction)"
    )
    
    interval_days = st.sidebar.slider(
        "Treatment frequency (days)",
        min_value=30,
        max_value=180,
        value=90,
        step=30,
        help="Interval between treatments in days"
    )
    
    treatment_duration_years = st.sidebar.slider(
        "Treatment duration (years)",
        min_value=0.5,
        max_value=3.0,
        value=1.5,
        step=0.5,
        help="Total duration of treatment"
    )
    treatment_duration_days = treatment_duration_years * 365.0
    
    # Trial parameters
    st.sidebar.subheader("Trial Parameters")
    
    n_per_arm = st.sidebar.slider(
        "Arm size (n per arm)",
        min_value=10,
        max_value=500,
        value=200,
        step=10,
        help="Number of trajectories per arm"
    )
    
    mci_threshold = st.sidebar.slider(
        "MCI threshold",
        min_value=4.0,
        max_value=6.5,
        value=5.2,
        step=0.1,
        help="Neuroinflammation threshold for MCI"
    )
    
    follow_up_years = st.sidebar.slider(
        "Follow-up time (years)",
        min_value=0.5,
        max_value=3.0,
        value=1.5,
        step=0.5,
        help="Time after treatment start to measure outcomes"
    )
    
    # Plotting parameters
    st.sidebar.subheader("Plotting Parameters")
    
    age_min = st.sidebar.slider(
        "Age range - Minimum (years)",
        min_value=40,
        max_value=80,
        value=50,
        step=5
    )
    
    age_max = st.sidebar.slider(
        "Age range - Maximum (years)",
        min_value=60,
        max_value=100,
        value=90,
        step=5
    )
    
    age_range = (age_min, age_max)
    
    # Random seed
    base_seed = st.sidebar.number_input(
        "Random seed",
        min_value=1,
        max_value=10000,
        value=2025,
        step=1,
        help="Seed for reproducibility"
    )
    
    # Fixed parameters
    dt = 1.0
    sample_every_days = 30.0
    
    # Run simulation button
    st.sidebar.markdown("---")
    run_button = st.sidebar.button("🚀 Run Simulation", type="primary", use_container_width=True)
    
    # Main content area
    if run_button or st.session_state.simulation_results is not None:
        # Check if parameters changed
        current_params = (
            n_per_arm, mci_threshold, s_fold_change, interval_days,
            treatment_duration_days, follow_up_years, base_seed
        )
        
        if st.session_state.current_params != current_params or run_button:
            # Run simulation with progress
            with st.spinner("Running simulation... This may take a moment."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("Simulating trajectories...")
                progress_bar.progress(25)
                
                control_data, treatment_data = run_simulation_cached(
                    BASE_PARAMS, n_per_arm, mci_threshold, s_fold_change,
                    interval_days, treatment_duration_days, follow_up_years,
                    dt, base_seed, sample_every_days
                )
                
                progress_bar.progress(50)
                status_text.text("Calculating statistics...")
                
                stats_results = calculate_trajectory_statistics(
                    control_data, treatment_data, mci_threshold, follow_up_years
                )
                
                progress_bar.progress(100)
                status_text.text("Simulation complete!")
                
                # Store results in session state
                st.session_state.simulation_results = {
                    'control_data': control_data,
                    'treatment_data': treatment_data,
                    'stats_results': stats_results
                }
                st.session_state.current_params = current_params
                
                # Clear progress
                progress_bar.empty()
                status_text.empty()
        
        # Get results
        results = st.session_state.simulation_results
        control_data = results['control_data']
        treatment_data = results['treatment_data']
        stats_results = results['stats_results']
        
        # Display statistics
        st.header("📊 Statistics Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        if stats_results is not None:
            with col1:
                st.metric(
                    "Mean Control n",
                    f"{stats_results.get('mean_control', np.nan):.3f}",
                    help="Mean neuroinflammation in control arm at follow-up"
                )
            
            with col2:
                st.metric(
                    "Mean Treatment n",
                    f"{stats_results.get('mean_treatment', np.nan):.3f}",
                    help="Mean neuroinflammation in treatment arm at follow-up"
                )
            
            with col3:
                delta_n = stats_results.get('mean_difference', np.nan)
                st.metric(
                    "Δn (Control - Treatment)",
                    f"{delta_n:.3f}",
                    help="Difference in neuroinflammation (positive = beneficial)"
                )
            
            with col4:
                cohens_d = stats_results.get('cohens_d', np.nan)
                st.metric(
                    "Cohen's d",
                    f"{cohens_d:.3f}",
                    help="Effect size"
                )
        
        # Additional statistics
        st.subheader("Additional Statistics")
        col1, col2, col3 = st.columns(3)
        
        if stats_results is not None:
            with col1:
                p_value = stats_results.get('p_value_ttest', np.nan)
                if np.isfinite(p_value):
                    st.write(f"**P-value (t-test):** {p_value:.4f}")
                else:
                    st.write("**P-value (t-test):** N/A")
            
            with col2:
                delay = stats_results.get('delay', np.nan)
                if np.isfinite(delay):
                    st.write(f"**Delay:** {delay:.2f} years")
                else:
                    st.write("**Delay:** N/A")
            
            with col3:
                st.write(f"**Control n:** {stats_results.get('n_control', 0)}")
                st.write(f"**Treatment n:** {stats_results.get('n_treatment', 0)}")
        
        # Plots
        st.header("📈 Visualizations")
        
        # Plot 1: Neuroinflammation trajectories
        st.subheader("Neuroinflammation Trajectories (n(t))")
        fig_n = plot_neuroinflammation_trajectories_streamlit(
            control_data, treatment_data, age_range, stats_results, n_per_arm
        )
        st.pyplot(fig_n)
        plt.close(fig_n)
        
        # Plot 2: Senescence trajectories
        st.subheader("Senescence Trajectories (s(t))")
        fig_s = plot_senescence_trajectories_streamlit(
            control_data, treatment_data, age_range, n_per_arm
        )
        st.pyplot(fig_s)
        plt.close(fig_s)
        
        # Plot 3: Survival curves
        st.subheader("Survival Curves")
        fig_surv = plot_survival_curves_streamlit(
            control_data, treatment_data, stats_results, n_per_arm
        )
        st.pyplot(fig_surv)
        plt.close(fig_surv)
        
        # Plot 4: Neuroinflammation difference
        st.subheader("Neuroinflammation Difference (Δn)")
        fig_diff = plot_neuroinflammation_difference_streamlit(
            control_data, treatment_data, age_range, n_per_arm
        )
        st.pyplot(fig_diff)
        plt.close(fig_diff)
        
    else:
        # Initial state - show instructions
        st.info("👈 Adjust parameters in the sidebar and click 'Run Simulation' to start exploring!")
        
        st.markdown("""
        ### How to Use This Platform
        
        1. **Adjust Parameters**: Use the sliders in the sidebar to set:
           - Treatment parameters (s reduction factor, frequency, duration)
           - Trial parameters (arm size, MCI threshold, follow-up time)
           - Plotting parameters (age range)
        
        2. **Run Simulation**: Click the "Run Simulation" button to generate results
        
        3. **View Results**: 
           - Statistics summary shows key metrics
           - Four plots show trajectories and differences
           - All plots update automatically when parameters change
        
        ### Parameters Explained
        
        - **s_fold_change**: Factor multiplying senescence (s) during treatment. Lower values = more reduction.
        - **Treatment frequency**: How often treatment is applied (in days)
        - **Treatment duration**: Total length of treatment period (in years)
        - **Arm size**: Number of simulated trajectories per arm
        - **MCI threshold**: Neuroinflammation level that triggers treatment
        - **Follow-up time**: Time after treatment start to measure outcomes
        """)

if __name__ == "__main__":
    main()
