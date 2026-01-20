"""
Mixed Population Clinical Trial Simulation
===========================================

Simulates a clinical trial with a mixed APOE genotype population:
- 20% APOE44
- 30% APOE34
- 10% APOE24
- 30% APOE33
- 10% APOE23

Each arm has 50 trajectories.
Treatment: s elimination every 3 months for 1.5 years (MCI-triggered).

Plots:
- Survival curves: control vs treatment
- Neuroinflammation trajectories (individual and mean) for each arm
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import os
from scipy.interpolate import interp1d
from scipy import stats
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import pandas as pd
import seaborn as sns

plt.style.use('default')

# Parameter indexing
IDX_ETA        = 0
IDX_KAPPA      = 1
IDX_XC         = 2
IDX_EPS_P      = 3
IDX_b          = 4
IDX_BETA       = 5
IDX_BETA_TAG   = 6
IDX_KAPPA_TAG  = 7
IDX_c          = 8
IDX_ALPHA      = 9
IDX_EPS_N      = 10

def pack_params(eta, kappa, Nc, epsilon_p, b, beta, beta_tag, kappa_tag, c, alpha, epsilon_n):
    """Pack parameters into a fixed-length vector for Numba use."""
    p = np.empty(11, dtype=np.float64)
    p[IDX_ETA]       = eta
    p[IDX_KAPPA]     = kappa
    p[IDX_XC]        = Nc
    p[IDX_EPS_P]     = epsilon_p
    p[IDX_b]         = b
    p[IDX_BETA]      = beta
    p[IDX_BETA_TAG]  = beta_tag
    p[IDX_KAPPA_TAG] = kappa_tag
    p[IDX_c]         = c
    p[IDX_ALPHA]     = alpha
    p[IDX_EPS_N]     = epsilon_n
    return p

@jit(nopython=True)
def simulate_baseline_full_trajectory(
    dt, params0, t_final, sample_every, random_seed
):
    """Simulate one baseline trajectory, returning full trajectory.
    
    Returns:
    --------
    t_vals, x_vals, n_vals, s_vals, onset_time
    """
    np.random.seed(random_seed)
    new_s = 0.0
    new_t = 0.0
    new_x = 0.0
    new_y = 0.0
    first_cross_time = 0.0
    got_first_cross = False
    
    sample_step = max(1, int(round(sample_every / dt)))
    t_vals = []
    x_vals = []
    n_vals = []
    s_vals = []
    
    eta0       = params0[IDX_ETA]
    kappa0     = params0[IDX_KAPPA]
    Nc0        = params0[IDX_XC]
    eps_p0     = params0[IDX_EPS_P]
    b0         = params0[IDX_b]
    beta0      = params0[IDX_BETA]
    beta_tag0  = params0[IDX_BETA_TAG]
    kappa_tag0 = params0[IDX_KAPPA_TAG]
    c0         = params0[IDX_c]
    alpha0     = params0[IDX_ALPHA]
    eps_n0     = params0[IDX_EPS_N]

    step = 0
    while new_t <= t_final:
        new_x = new_x + (dt * (new_t * eta0 - (beta0 * (new_x / (new_x + kappa0))))) + np.sqrt(2.0 * eps_p0 * dt) * np.random.randn()
        new_s = new_s + (dt * (1/1000)*((beta0 * (new_x / (new_x + kappa0))- new_s)))
        new_y = new_y + (dt * (b0 * new_x)) - (dt * (beta_tag0 * (new_y / (new_y + kappa_tag0)))) + (dt * c0 * (new_y**1) * new_s) + np.sqrt(2.0 * eps_n0 * dt) * np.random.randn()

        if new_x < 0.0:
            new_x = 0.0
        if new_y < 0.0:
            new_y = 0.0
        if new_s < 0.0:
            new_s = 0.0

        if step % sample_step == 0:
            t_vals.append(new_t)
            x_vals.append(new_x)
            n_vals.append(new_y)
            s_vals.append(new_s)

        if (not got_first_cross) and (new_y >= Nc0):
            first_cross_time = new_t
            got_first_cross = True
        if new_y >= 10*Nc0:
            t_vals_years = np.array(t_vals) / 365.0
            return t_vals_years, np.array(x_vals), np.array(n_vals), np.array(s_vals), first_cross_time / 365.0

        new_t += dt
        step += 1

    t_vals_years = np.array(t_vals) / 365.0
    return t_vals_years, np.array(x_vals), np.array(n_vals), np.array(s_vals), t_final / 365.0

@jit(nopython=True)
def simulate_mci_triggered_treatment_full_trajectory(
    dt, params0, mci_threshold, s_fold_change, t_final, sample_every, random_seed, interval_days, treatment_duration_days
):
    """Simulate one trajectory with MCI-triggered treatment, returning full trajectory.
    
    Treatment: s is multiplied by s_fold_change (e.g., 0.3 means 70% reduction)
    Applied every interval_days for treatment_duration_days after MCI threshold crossing.
    
    Returns:
    --------
    t_vals, x_vals, n_vals, s_vals, treatment_start_time, onset_time
    """
    np.random.seed(random_seed)
    new_s = 0.0
    new_t = 0.0
    new_x = 0.0
    new_y = 0.0
    first_cross_time = 0.0
    got_first_cross = False
    perturbation_started = False
    perturbation_end_time = 0.0
    next_perturbation_time = 0.0
    treatment_start_time = 0.0
    
    sample_step = max(1, int(round(sample_every / dt)))
    t_vals = []
    x_vals = []
    n_vals = []
    s_vals = []
    
    eta0       = params0[IDX_ETA]
    kappa0     = params0[IDX_KAPPA]
    Nc0        = params0[IDX_XC]
    eps_p0     = params0[IDX_EPS_P]
    b0         = params0[IDX_b]
    beta0      = params0[IDX_BETA]
    beta_tag0  = params0[IDX_BETA_TAG]
    kappa_tag0 = params0[IDX_KAPPA_TAG]
    c0         = params0[IDX_c]
    alpha0     = params0[IDX_ALPHA]
    eps_n0     = params0[IDX_EPS_N]

    step = 0
    while new_t <= t_final:
        # Start treatment when n crosses MCI threshold
        if not perturbation_started and new_y >= mci_threshold:
            perturbation_started = True
            treatment_start_time = new_t
            perturbation_end_time = new_t + treatment_duration_days
            next_perturbation_time = new_t
            new_s = new_s * s_fold_change
        
        # Apply periodic treatments (every interval_days, within treatment window)
        if perturbation_started and new_t < perturbation_end_time and new_t >= next_perturbation_time:
            new_s = new_s * s_fold_change
            next_perturbation_time += interval_days
        
        new_x = new_x + (dt * (new_t * eta0 - (beta0 * (new_x / (new_x + kappa0))))) + np.sqrt(2.0 * eps_p0 * dt) * np.random.randn()
        new_s = new_s + (dt * (1/1000)*((beta0 * (new_x / (new_x + kappa0))- new_s)))
        new_y = new_y + (dt * (b0 * new_x)) - (dt * (beta_tag0 * (new_y / (new_y + kappa_tag0)))) + (dt * c0 * (new_y**1) * new_s) + np.sqrt(2.0 * eps_n0 * dt) * np.random.randn()

        if new_x < 0.0:
            new_x = 0.0
        if new_y < 0.0:
            new_y = 0.0
        if new_s < 0.0:
            new_s = 0.0

        if step % sample_step == 0:
            t_vals.append(new_t)
            x_vals.append(new_x)
            n_vals.append(new_y)
            s_vals.append(new_s)

        if (not got_first_cross) and (new_y >= Nc0):
            first_cross_time = new_t
            got_first_cross = True
        if new_y >= 10*Nc0:
            t_vals_years = np.array(t_vals) / 365.0
            return t_vals_years, np.array(x_vals), np.array(n_vals), np.array(s_vals), treatment_start_time / 365.0, first_cross_time / 365.0

        new_t += dt
        step += 1

    t_vals_years = np.array(t_vals) / 365.0
    return t_vals_years, np.array(x_vals), np.array(n_vals), np.array(s_vals), treatment_start_time / 365.0, t_final / 365.0

def assign_genotypes(n_total):
    """
    Assign genotypes to n_total participants based on specified percentages:
    - 20% APOE44
    - 30% APOE34
    - 10% APOE24
    - 30% APOE33
    - 10% APOE23
    """
    genotypes = []
    n_apoe44 = int(0.20 * n_total)
    n_apoe34 = int(0.30 * n_total)
    n_apoe24 = int(0.10 * n_total)
    n_apoe33 = int(0.30 * n_total)
    n_apoe23 = n_total - n_apoe44 - n_apoe34 - n_apoe24 - n_apoe33  # Remaining goes to APOE23
    
    genotypes.extend(['APOE44'] * n_apoe44)
    genotypes.extend(['APOE34'] * n_apoe34)
    genotypes.extend(['APOE24'] * n_apoe24)
    genotypes.extend(['APOE33'] * n_apoe33)
    genotypes.extend(['APOE23'] * n_apoe23)
    
    # Shuffle to randomize order
    np.random.shuffle(genotypes)
    return genotypes

def get_params_for_genotype(genotype, base_params):
    """
    Get parameters for a specific APOE genotype.
    
    Beta multipliers:
    - APOE44: 0.7
    - APOE34: 0.8
    - APOE24: 0.8
    - APOE33: 1.0 (baseline)
    - APOE23: 1.1
    """
    beta_multipliers = {
        'APOE44': 0.7,
        'APOE34': 0.8,
        'APOE24': 0.8,
        'APOE33': 1.0,
        'APOE23': 1.1
    }
    
    params = base_params.copy()
    # Use the base_params beta value and multiply by genotype-specific multiplier
    params['beta'] = beta_multipliers[genotype] * base_params['beta']
    return params

def simulate_mixed_population_trial(
    base_params, n_per_arm=50, mci_threshold=5.2, 
    s_fold_change=0.3, interval_days=90.0, treatment_duration_days=1.5*365.0,
    follow_up_years=1.5, dt=1.0, base_seed=2025, sample_every_days=30.0
):
    """
    Simulate a clinical trial with mixed APOE genotype population.
    
    Returns:
    --------
    control_data : dict with keys 'trajectories', 'onset_times', 'genotypes'
    treatment_data : dict with keys 'trajectories', 'onset_times', 'genotypes', 'treatment_start_times'
    """
    t_final = 400000.0  # days
    
    # Assign genotypes
    control_genotypes = assign_genotypes(n_per_arm)
    treatment_genotypes = assign_genotypes(n_per_arm)
    
    # Control arm
    control_trajectories = []
    control_onset_times = []
    
    print(f"Simulating {n_per_arm} control participants...")
    for i, genotype in enumerate(control_genotypes):
        params = get_params_for_genotype(genotype, base_params)
        params_packed = pack_params(**params)
        
        seed = base_seed + i
        t, x, n, s, onset = simulate_baseline_full_trajectory(
            dt, params_packed, t_final, sample_every_days, seed
        )
        
        control_trajectories.append({
            't': t,
            'x': x,
            'n': n,
            's': s,
            'genotype': genotype
        })
        control_onset_times.append(onset)
    
    # Treatment arm
    treatment_trajectories = []
    treatment_onset_times = []
    treatment_start_times = []
    
    print(f"Simulating {n_per_arm} treatment participants...")
    for i, genotype in enumerate(treatment_genotypes):
        params = get_params_for_genotype(genotype, base_params)
        params_packed = pack_params(**params)
        
        seed = base_seed + 10000 + i
        t, x, n, s, treatment_start_time, onset = simulate_mci_triggered_treatment_full_trajectory(
            dt, params_packed, mci_threshold, s_fold_change, t_final, 
            sample_every_days, seed, interval_days, treatment_duration_days
        )
        
        treatment_trajectories.append({
            't': t,
            'x': x,
            'n': n,
            's': s,
            'genotype': genotype
        })
        treatment_onset_times.append(onset)
        treatment_start_times.append(treatment_start_time)
    
    return {
        'trajectories': control_trajectories,
        'onset_times': np.array(control_onset_times),
        'genotypes': control_genotypes
    }, {
        'trajectories': treatment_trajectories,
        'onset_times': np.array(treatment_onset_times),
        'genotypes': treatment_genotypes,
        'treatment_start_times': np.array(treatment_start_times)
    }

def calculate_statistics(control_values, treatment_values, control_onset_times=None, treatment_onset_times=None):
    """
    Calculate statistical tests appropriate for clinical trials.
    
    Parameters:
    -----------
    control_values : np.array
        Neuroinflammation values for control arm
    treatment_values : np.array
        Neuroinflammation values for treatment arm
    control_onset_times : np.array, optional
        Onset times (age at which n crosses Nc) for control arm
    treatment_onset_times : np.array, optional
        Onset times (age at which n crosses Nc) for treatment arm
    
    Returns:
    --------
    results : dict
        Dictionary containing statistical test results
    """
    results = {}
    
    # Remove any NaN or infinite values
    control_clean = control_values[np.isfinite(control_values)]
    treatment_clean = treatment_values[np.isfinite(treatment_values)]
    
    results['n_control'] = len(control_clean)
    results['n_treatment'] = len(treatment_clean)
    results['mean_control'] = np.mean(control_clean)
    results['mean_treatment'] = np.mean(treatment_clean)
    results['std_control'] = np.std(control_clean, ddof=1)
    results['std_treatment'] = np.std(treatment_clean, ddof=1)
    results['median_control'] = np.median(control_clean)
    results['median_treatment'] = np.median(treatment_clean)
    
    # Mean difference
    mean_diff = results['mean_treatment'] - results['mean_control']
    results['mean_difference'] = mean_diff
    
    # 95% Confidence interval for mean difference (t-test based)
    if len(control_clean) > 1 and len(treatment_clean) > 1:
        # Independent t-test
        t_stat, p_value_ttest = stats.ttest_ind(treatment_clean, control_clean)
        results['t_statistic'] = t_stat
        results['p_value_ttest'] = p_value_ttest
        
        # Calculate pooled standard error for CI
        se_pooled = np.sqrt((results['std_control']**2 / len(control_clean)) + 
                           (results['std_treatment']**2 / len(treatment_clean)))
        df = len(control_clean) + len(treatment_clean) - 2
        t_critical = stats.t.ppf(0.975, df)  # 95% CI
        results['ci_lower'] = mean_diff - t_critical * se_pooled
        results['ci_upper'] = mean_diff + t_critical * se_pooled
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(control_clean) - 1) * results['std_control']**2 + 
                              (len(treatment_clean) - 1) * results['std_treatment']**2) / 
                             (len(control_clean) + len(treatment_clean) - 2))
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0.0
        results['cohens_d'] = cohens_d
        
        # Calculate 95% CI for Cohen's d
        n1 = len(control_clean)
        n2 = len(treatment_clean)
        df = n1 + n2 - 2
        if df > 0 and pooled_std > 0:
            se_d = np.sqrt((n1 + n2) / (n1 * n2) + cohens_d**2 / (2 * df))
            t_critical = stats.t.ppf(0.975, df)
            results['cohens_d_ci_lower'] = cohens_d - t_critical * se_d
            results['cohens_d_ci_upper'] = cohens_d + t_critical * se_d
        else:
            results['cohens_d_ci_lower'] = np.nan
            results['cohens_d_ci_upper'] = np.nan
        
        # Mann-Whitney U test
        u_statistic, p_value_mannwhitney = stats.mannwhitneyu(treatment_clean, control_clean, 
                                                              alternative='two-sided')
        results['u_statistic'] = u_statistic
        results['p_value_mannwhitney'] = p_value_mannwhitney
    else:
        results['t_statistic'] = np.nan
        results['p_value_ttest'] = np.nan
        results['ci_lower'] = np.nan
        results['ci_upper'] = np.nan
        results['cohens_d'] = np.nan
        results['cohens_d_ci_lower'] = np.nan
        results['cohens_d_ci_upper'] = np.nan
        results['u_statistic'] = np.nan
        results['p_value_mannwhitney'] = np.nan
    
    # Calculate delay in survival curve (difference in median onset times)
    if control_onset_times is not None and treatment_onset_times is not None:
        control_onset_clean = control_onset_times[np.isfinite(control_onset_times) & (control_onset_times > 0)]
        treatment_onset_clean = treatment_onset_times[np.isfinite(treatment_onset_times) & (treatment_onset_times > 0)]
        
        if len(control_onset_clean) > 0 and len(treatment_onset_clean) > 0:
            median_control_onset = np.median(control_onset_clean)
            median_treatment_onset = np.median(treatment_onset_clean)
            delay = median_treatment_onset - median_control_onset
            results['delay'] = delay
        else:
            results['delay'] = np.nan
    else:
        results['delay'] = np.nan
    
    return results

def plot_survival_curves(control_data, treatment_data, output_dir, stats_results=None, n_per_arm=None, genotype=None):
    """Plot Kaplan-Meier survival curves for control and treatment arms."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Prepare data for lifelines
    # Control arm
    control_onset_times = control_data['onset_times']
    control_events = np.ones(len(control_onset_times))  # All events observed
    
    # Treatment arm
    treatment_onset_times = treatment_data['onset_times']
    treatment_events = np.ones(len(treatment_onset_times))  # All events observed
    
    # Calculate delay (median age difference)
    control_onset_clean = control_onset_times[np.isfinite(control_onset_times) & (control_onset_times > 0)]
    treatment_onset_clean = treatment_onset_times[np.isfinite(treatment_onset_times) & (treatment_onset_times > 0)]
    
    if len(control_onset_clean) > 0 and len(treatment_onset_clean) > 0:
        median_control_onset = np.median(control_onset_clean)
        median_treatment_onset = np.median(treatment_onset_clean)
        delay = median_treatment_onset - median_control_onset
    else:
        delay = np.nan
    
    # Perform log-rank test for survival curves
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
    
    # Add delay annotation and statistics
    annotation_lines = []
    if np.isfinite(delay):
        annotation_lines.append(f'Delay: {delay:.2f} years\n(median age difference)')
    
    # Use log-rank test p-value for survival curves
    if np.isfinite(p_value_logrank):
        annotation_lines.append(f"p-value (log-rank): {p_value_logrank:.4f}")
    
    # Also include Cohen's d from neuroinflammation if available
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
    if genotype is not None:
        title = f'Survival Curves: Control vs Treatment ({genotype})'
    else:
        title = 'Survival Curves: Control vs Treatment'
    if n_per_arm is not None:
        title += f' (n={n_per_arm} per arm)'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc="lower left")
    ax.set_xlim(50, 110)
    sns.despine(ax=ax)

    # ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if genotype is not None:
        filename = f'survival_curves_{genotype.lower()}.png'
    else:
        filename = 'survival_curves_mixed_population.png'
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved survival curves to {output_dir}/{filename}")

def plot_neuroinflammation_trajectories(control_data, treatment_data, output_dir, follow_up_years=1.5, mci_threshold=5.2, stats_results=None, n_per_arm=None, genotype=None):
    """Plot neuroinflammation trajectories (control and treatment on same plot)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Control arm
    control_trajs = control_data['trajectories']
    control_n_arrays = [traj['n'] for traj in control_trajs]
    control_t_arrays = [traj['t'] for traj in control_trajs]
    
    # Treatment arm
    treatment_trajs = treatment_data['trajectories']
    treatment_n_arrays = [traj['n'] for traj in treatment_trajs]
    treatment_t_arrays = [traj['t'] for traj in treatment_trajs]
    
    # Find common time range (overlap of both arms), limited to 50-90 years
    min_t_all = max([max([t[0] for t in control_t_arrays]), max([t[0] for t in treatment_t_arrays])])
    max_t_all = min([min([t[-1] for t in control_t_arrays]), min([t[-1] for t in treatment_t_arrays])])
    # Limit to 50-90 years
    min_t_all = max(min_t_all, 50.0)
    max_t_all = min(max_t_all, 90.0)
    t_common = np.linspace(min_t_all, max_t_all, 500)
    
    # Interpolate control trajectories to common time
    n_interp_control = []
    for t, n in zip(control_t_arrays, control_n_arrays):
        f_n = interp1d(t, n, kind='linear', bounds_error=False, fill_value=np.nan)
        n_interp_control.append(f_n(t_common))
    
    n_interp_control = np.array(n_interp_control)
    n_mean_control = np.nanmean(n_interp_control, axis=0)
    n_std_control = np.nanstd(n_interp_control, axis=0)
    # Calculate standard error (SE = SD / sqrt(n))
    n_se_control = n_std_control / np.sqrt(len(control_trajs))
    
    # Interpolate treatment trajectories to common time
    n_interp_treatment = []
    for t, n in zip(treatment_t_arrays, treatment_n_arrays):
        f_n = interp1d(t, n, kind='linear', bounds_error=False, fill_value=np.nan)
        n_interp_treatment.append(f_n(t_common))
    
    n_interp_treatment = np.array(n_interp_treatment)
    n_mean_treatment = np.nanmean(n_interp_treatment, axis=0)
    n_std_treatment = np.nanstd(n_interp_treatment, axis=0)
    # Calculate standard error (SE = SD / sqrt(n))
    n_se_treatment = n_std_treatment / np.sqrt(len(treatment_trajs))
    
    # Calculate differences at age 80 (control - treatment, so positive when treatment reduces neuroinflammation)
    age_80_idx = np.argmin(np.abs(t_common - 80))
    if age_80_idx < len(t_common):
        n_diff_at_80 = n_mean_control[age_80_idx] - n_mean_treatment[age_80_idx]
    else:
        n_diff_at_80 = np.nan
    
    # Extract statistics from pre-calculated results
    if stats_results is not None:
        p_value = stats_results.get('p_value_ttest', np.nan)
        cohens_d = stats_results.get('cohens_d', np.nan)
    else:
        p_value = np.nan
        cohens_d = np.nan
    
    # Plot neuroinflammation (n) trajectories - mean and SE only
    # Plot mean trajectories
    ax.plot(t_common, n_mean_control, color='blue', linewidth=2.5, label='Control Mean')
    ax.fill_between(t_common, n_mean_control - n_se_control, 
                     n_mean_control + n_se_control, color='blue', alpha=0.2, label='Control ±1 SE')
    
    ax.plot(t_common, n_mean_treatment, color='red', linewidth=2.5, label='Treatment Mean')
    ax.fill_between(t_common, n_mean_treatment - n_se_treatment, 
                     n_mean_treatment + n_se_treatment, color='red', alpha=0.2, label='Treatment ±1 SE')
    
    # Add annotation for difference at age 80 and statistics
    annotation_text = []
    if np.isfinite(n_diff_at_80):
        annotation_text.append(f'Δn at age 80: {n_diff_at_80:.3f}')
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
    if genotype is not None:
        title = f'Neuroinflammation Trajectories ({genotype})'
    else:
        title = 'Neuroinflammation Trajectories'
    if n_per_arm is not None:
        title += f' (n={n_per_arm} per arm)'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    ax.set_xlim(50, 90)
    ax.set_ylim(1, 10)
    sns.despine(ax=ax)
    
    # Add figure-level title with differences at age 80 and statistics
    title_parts = []
    if np.isfinite(n_diff_at_80):
        title_parts.append(f'Age 80: Δn={n_diff_at_80:.3f}')
    if np.isfinite(p_value):
        title_parts.append(f"p={p_value:.4f}")
    if np.isfinite(cohens_d):
        title_parts.append(f"d={cohens_d:.3f}")
    
    if len(title_parts) > 0:
        fig.suptitle(' | '.join(title_parts), fontsize=12, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    if genotype is not None:
        filename = f'neuroinflammation_trajectories_{genotype.lower()}.png'
    else:
        filename = 'neuroinflammation_trajectories_mixed_population.png'
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved neuroinflammation trajectories to {output_dir}/{filename}")

def plot_neuroinflammation_difference(control_data, treatment_data, output_dir, n_per_arm=None, genotype=None):
    """Plot difference in neuroinflammation (control - treatment) as a function of age."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Control arm
    control_trajs = control_data['trajectories']
    control_n_arrays = [traj['n'] for traj in control_trajs]
    control_t_arrays = [traj['t'] for traj in control_trajs]
    
    # Treatment arm
    treatment_trajs = treatment_data['trajectories']
    treatment_n_arrays = [traj['n'] for traj in treatment_trajs]
    treatment_t_arrays = [traj['t'] for traj in treatment_trajs]
    
    # Find common time range (overlap of both arms), limited to 50-90 years
    min_t_all = max([max([t[0] for t in control_t_arrays]), max([t[0] for t in treatment_t_arrays])])
    max_t_all = min([min([t[-1] for t in control_t_arrays]), min([t[-1] for t in treatment_t_arrays])])
    # Limit to 50-90 years
    min_t_all = max(min_t_all, 50.0)
    max_t_all = min(max_t_all, 90.0)
    t_common = np.linspace(min_t_all, max_t_all, 500)
    
    # Interpolate control trajectories to common time
    n_interp_control = []
    for t, n in zip(control_t_arrays, control_n_arrays):
        f_n = interp1d(t, n, kind='linear', bounds_error=False, fill_value=np.nan)
        n_interp_control.append(f_n(t_common))
    
    n_interp_control = np.array(n_interp_control)
    n_mean_control = np.nanmean(n_interp_control, axis=0)
    n_std_control = np.nanstd(n_interp_control, axis=0)
    n_se_control = n_std_control / np.sqrt(len(control_trajs))
    
    # Interpolate treatment trajectories to common time
    n_interp_treatment = []
    for t, n in zip(treatment_t_arrays, treatment_n_arrays):
        f_n = interp1d(t, n, kind='linear', bounds_error=False, fill_value=np.nan)
        n_interp_treatment.append(f_n(t_common))
    
    n_interp_treatment = np.array(n_interp_treatment)
    n_mean_treatment = np.nanmean(n_interp_treatment, axis=0)
    n_std_treatment = np.nanstd(n_interp_treatment, axis=0)
    n_se_treatment = n_std_treatment / np.sqrt(len(treatment_trajs))
    
    # Calculate difference: control - treatment (positive means treatment reduces neuroinflammation)
    diff_n = n_mean_control - n_mean_treatment
    # SE of difference: sqrt(SE_control^2 + SE_treatment^2)
    diff_se_n = np.sqrt(n_se_control**2 + n_se_treatment**2)
    
    # Plot difference
    ax.plot(t_common, diff_n, color='blue', linewidth=2.5, label='Δn (Control - Treatment)')
    ax.fill_between(t_common, diff_n - diff_se_n, diff_n + diff_se_n, 
                    color='blue', alpha=0.2, label='±1 SE')
    
    ax.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.3)
    ax.set_xlabel('Age (years)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Δn (Control - Treatment)', fontsize=12, fontweight='bold')
    if genotype is not None:
        title = f'Neuroinflammation Difference ({genotype})'
    else:
        title = 'Neuroinflammation Difference'
    if n_per_arm is not None:
        title += f' (n={n_per_arm} per arm)'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best', framealpha=0.9)
    ax.set_xlim(50, 90)
    sns.despine(ax=ax)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    if genotype is not None:
        filename = f'neuroinflammation_difference_{genotype.lower()}.png'
    else:
        filename = 'neuroinflammation_difference_mixed_population.png'
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved neuroinflammation difference plot to {output_dir}/{filename}")

def plot_senescence_difference(control_data, treatment_data, output_dir, n_per_arm=None, genotype=None):
    """Plot difference in senescence (control - treatment) as a function of age."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Control arm
    control_trajs = control_data['trajectories']
    control_s_arrays = [traj['s'] for traj in control_trajs]
    control_t_arrays = [traj['t'] for traj in control_trajs]
    
    # Treatment arm
    treatment_trajs = treatment_data['trajectories']
    treatment_s_arrays = [traj['s'] for traj in treatment_trajs]
    treatment_t_arrays = [traj['t'] for traj in treatment_trajs]
    
    # Find common time range (overlap of both arms), limited to 50-90 years
    min_t_all = max([max([t[0] for t in control_t_arrays]), max([t[0] for t in treatment_t_arrays])])
    max_t_all = min([min([t[-1] for t in control_t_arrays]), min([t[-1] for t in treatment_t_arrays])])
    # Limit to 50-90 years
    min_t_all = max(min_t_all, 50.0)
    max_t_all = min(max_t_all, 90.0)
    t_common = np.linspace(min_t_all, max_t_all, 500)
    
    # Interpolate control trajectories to common time
    s_interp_control = []
    for t, s in zip(control_t_arrays, control_s_arrays):
        f_s = interp1d(t, s, kind='linear', bounds_error=False, fill_value=np.nan)
        s_interp_control.append(f_s(t_common))
    
    s_interp_control = np.array(s_interp_control)
    s_mean_control = np.nanmean(s_interp_control, axis=0)
    s_std_control = np.nanstd(s_interp_control, axis=0)
    s_se_control = s_std_control / np.sqrt(len(control_trajs))
    
    # Interpolate treatment trajectories to common time
    s_interp_treatment = []
    for t, s in zip(treatment_t_arrays, treatment_s_arrays):
        f_s = interp1d(t, s, kind='linear', bounds_error=False, fill_value=np.nan)
        s_interp_treatment.append(f_s(t_common))
    
    s_interp_treatment = np.array(s_interp_treatment)
    s_mean_treatment = np.nanmean(s_interp_treatment, axis=0)
    s_std_treatment = np.nanstd(s_interp_treatment, axis=0)
    s_se_treatment = s_std_treatment / np.sqrt(len(treatment_trajs))
    
    # Calculate difference: control - treatment (positive means treatment reduces senescence)
    diff_s = s_mean_control - s_mean_treatment
    # SE of difference: sqrt(SE_control^2 + SE_treatment^2)
    diff_se_s = np.sqrt(s_se_control**2 + s_se_treatment**2)
    
    # Plot difference
    ax.plot(t_common, diff_s, color='red', linewidth=2.5, label='Δs (Control - Treatment)')
    ax.fill_between(t_common, diff_s - diff_se_s, diff_s + diff_se_s, 
                    color='red', alpha=0.2, label='±1 SE')
    
    ax.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.3)
    ax.set_xlabel('Age (years)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Δs (Control - Treatment)', fontsize=12, fontweight='bold')
    if genotype is not None:
        title = f'Senescence Difference ({genotype})'
    else:
        title = 'Senescence Difference'
    if n_per_arm is not None:
        title += f' (n={n_per_arm} per arm)'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best', framealpha=0.9)
    ax.set_xlim(50, 90)
    sns.despine(ax=ax)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    if genotype is not None:
        filename = f'senescence_difference_{genotype.lower()}.png'
    else:
        filename = 'senescence_difference_mixed_population.png'
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved senescence difference plot to {output_dir}/{filename}")

def plot_neuroinflammation_equation_terms(control_data, treatment_data, base_params, output_dir, n_per_arm=None, genotype=None):
    """
    Plot neuroinflammation equation terms as a function of age.
    
    The neuroinflammation equation is:
    dy/dt = b*x - beta_tag*(y/(y+kappa_tag)) + c*y*s
    
    Terms (plotted):
    1. b*x (production from amyloid)
    2. beta_tag*(y/(y+kappa_tag)) (clearance, plotted as positive)
    3. c*y*s (senescence amplification)
    
    Each term is averaged over all APOE types in the mixed population.
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Extract parameters
    b = base_params['b']
    beta_tag = base_params['beta_tag']
    kappa_tag = base_params['kappa_tag']
    c = base_params['c']
    
    # Control arm
    control_trajs = control_data['trajectories']
    control_x_arrays = [traj['x'] for traj in control_trajs]
    control_n_arrays = [traj['n'] for traj in control_trajs]
    control_s_arrays = [traj['s'] for traj in control_trajs]
    control_t_arrays = [traj['t'] for traj in control_trajs]
    
    # Treatment arm
    treatment_trajs = treatment_data['trajectories']
    treatment_x_arrays = [traj['x'] for traj in treatment_trajs]
    treatment_n_arrays = [traj['n'] for traj in treatment_trajs]
    treatment_s_arrays = [traj['s'] for traj in treatment_trajs]
    treatment_t_arrays = [traj['t'] for traj in treatment_trajs]
    
    # Find common time range (overlap of both arms), limited to 50-90 years
    min_t_all = max([max([t[0] for t in control_t_arrays]), max([t[0] for t in treatment_t_arrays])])
    max_t_all = min([min([t[-1] for t in control_t_arrays]), min([t[-1] for t in treatment_t_arrays])])
    # Limit to 50-90 years
    min_t_all = max(min_t_all, 50.0)
    max_t_all = min(max_t_all, 90.0)
    t_common = np.linspace(min_t_all, max_t_all, 500)
    
    # Calculate terms for control arm
    term1_control_list = []  # b*x
    term2_control_list = []  # beta_tag*(y/(y+kappa_tag)) (plotted as positive)
    term3_control_list = []  # c*y*s
    
    for t, x, n, s in zip(control_t_arrays, control_x_arrays, control_n_arrays, control_s_arrays):
        f_x = interp1d(t, x, kind='linear', bounds_error=False, fill_value=np.nan)
        f_n = interp1d(t, n, kind='linear', bounds_error=False, fill_value=np.nan)
        f_s = interp1d(t, s, kind='linear', bounds_error=False, fill_value=np.nan)
        
        x_interp = f_x(t_common)
        n_interp = f_n(t_common)
        s_interp = f_s(t_common)
        
        # Calculate terms
        term1 = b * x_interp
        term2 = beta_tag * (n_interp / (n_interp + kappa_tag))  # Plotted as positive
        term3 = c * n_interp * s_interp
        
        term1_control_list.append(term1)
        term2_control_list.append(term2)
        term3_control_list.append(term3)
    
    # Calculate mean and SE for control
    term1_control = np.array(term1_control_list)
    term2_control = np.array(term2_control_list)
    term3_control = np.array(term3_control_list)
    
    term1_control_mean = np.nanmean(term1_control, axis=0)
    term1_control_se = np.nanstd(term1_control, axis=0) / np.sqrt(len(control_trajs))
    
    term2_control_mean = np.nanmean(term2_control, axis=0)
    term2_control_se = np.nanstd(term2_control, axis=0) / np.sqrt(len(control_trajs))
    
    term3_control_mean = np.nanmean(term3_control, axis=0)
    term3_control_se = np.nanstd(term3_control, axis=0) / np.sqrt(len(control_trajs))
    
    # Calculate terms for treatment arm
    term1_treatment_list = []  # b*x
    term2_treatment_list = []  # beta_tag*(y/(y+kappa_tag)) (plotted as positive)
    term3_treatment_list = []  # c*y*s
    
    for t, x, n, s in zip(treatment_t_arrays, treatment_x_arrays, treatment_n_arrays, treatment_s_arrays):
        f_x = interp1d(t, x, kind='linear', bounds_error=False, fill_value=np.nan)
        f_n = interp1d(t, n, kind='linear', bounds_error=False, fill_value=np.nan)
        f_s = interp1d(t, s, kind='linear', bounds_error=False, fill_value=np.nan)
        
        x_interp = f_x(t_common)
        n_interp = f_n(t_common)
        s_interp = f_s(t_common)
        
        # Calculate terms
        term1 = b * x_interp
        term2 = beta_tag * (n_interp / (n_interp + kappa_tag))  # Plotted as positive
        term3 = c * n_interp * s_interp
        
        term1_treatment_list.append(term1)
        term2_treatment_list.append(term2)
        term3_treatment_list.append(term3)
    
    # Calculate mean and SE for treatment
    term1_treatment = np.array(term1_treatment_list)
    term2_treatment = np.array(term2_treatment_list)
    term3_treatment = np.array(term3_treatment_list)
    
    term1_treatment_mean = np.nanmean(term1_treatment, axis=0)
    term1_treatment_se = np.nanstd(term1_treatment, axis=0) / np.sqrt(len(treatment_trajs))
    
    term2_treatment_mean = np.nanmean(term2_treatment, axis=0)
    term2_treatment_se = np.nanstd(term2_treatment, axis=0) / np.sqrt(len(treatment_trajs))
    
    term3_treatment_mean = np.nanmean(term3_treatment, axis=0)
    term3_treatment_se = np.nanstd(term3_treatment, axis=0) / np.sqrt(len(treatment_trajs))
    
    # Plot control terms
    valid_mask = ~np.isnan(term1_control_mean)
    ax.plot(t_common[valid_mask], term1_control_mean[valid_mask], 'b-', linewidth=2.5, 
            label='b·P (production, Control)', alpha=0.8)
    ax.fill_between(t_common[valid_mask], 
                    term1_control_mean[valid_mask] - term1_control_se[valid_mask],
                    term1_control_mean[valid_mask] + term1_control_se[valid_mask],
                    color='blue', alpha=0.15)
    
    valid_mask = ~np.isnan(term2_control_mean)
    ax.plot(t_common[valid_mask], term2_control_mean[valid_mask], 'r-', linewidth=2.5, 
            label='β\'·(N/(N+κ\')) (clearance, Control)', alpha=0.8)
    ax.fill_between(t_common[valid_mask], 
                    term2_control_mean[valid_mask] - term2_control_se[valid_mask],
                    term2_control_mean[valid_mask] + term2_control_se[valid_mask],
                    color='red', alpha=0.15)
    
    valid_mask = ~np.isnan(term3_control_mean)
    ax.plot(t_common[valid_mask], term3_control_mean[valid_mask], 'g-', linewidth=2.5, 
            label='c·N·s (amplification, Control)', alpha=0.8)
    ax.fill_between(t_common[valid_mask], 
                    term3_control_mean[valid_mask] - term3_control_se[valid_mask],
                    term3_control_mean[valid_mask] + term3_control_se[valid_mask],
                    color='green', alpha=0.15)
    
    # Plot treatment terms (dashed)
    valid_mask = ~np.isnan(term1_treatment_mean)
    ax.plot(t_common[valid_mask], term1_treatment_mean[valid_mask], 'b--', linewidth=2.5, 
            label='b·P (production, Treatment)', alpha=0.8)
    ax.fill_between(t_common[valid_mask], 
                    term1_treatment_mean[valid_mask] - term1_treatment_se[valid_mask],
                    term1_treatment_mean[valid_mask] + term1_treatment_se[valid_mask],
                    color='blue', alpha=0.1)
    
    valid_mask = ~np.isnan(term2_treatment_mean)
    ax.plot(t_common[valid_mask], term2_treatment_mean[valid_mask], 'r--', linewidth=2.5, 
            label='β\'·(N/(N+κ\')) (clearance, Treatment)', alpha=0.8)
    ax.fill_between(t_common[valid_mask], 
                    term2_treatment_mean[valid_mask] - term2_treatment_se[valid_mask],
                    term2_treatment_mean[valid_mask] + term2_treatment_se[valid_mask],
                    color='red', alpha=0.1)
    
    valid_mask = ~np.isnan(term3_treatment_mean)
    ax.plot(t_common[valid_mask], term3_treatment_mean[valid_mask], 'g--', linewidth=2.5, 
            label='c·N·s (amplification, Treatment)', alpha=0.8)
    ax.fill_between(t_common[valid_mask], 
                    term3_treatment_mean[valid_mask] - term3_treatment_se[valid_mask],
                    term3_treatment_mean[valid_mask] + term3_treatment_se[valid_mask],
                    color='green', alpha=0.1)
    
    # Add horizontal line at y=0
    ax.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.3)
    
    ax.set_xlabel('Age (years)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Term Value (1/day)', fontsize=12, fontweight='bold')
    if genotype is not None:
        title = f'Neuroinflammation Equation Terms ({genotype})'
    else:
        title = 'Neuroinflammation Equation Terms (All APOE Types)'
    if n_per_arm is not None:
        title += f' (n={n_per_arm} per arm)'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best', framealpha=0.9, ncol=2)
    ax.set_xlim(50, 90)
    sns.despine(ax=ax)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    if genotype is not None:
        filename = f'neuroinflammation_equation_terms_{genotype.lower()}.png'
    else:
        filename = 'neuroinflammation_equation_terms_mixed_population.png'
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved neuroinflammation equation terms plot to {output_dir}/{filename}")

def filter_data_by_genotype(data, genotype):
    """Filter trajectories and onset times by genotype."""
    genotype_mask = np.array(data['genotypes']) == genotype
    filtered_trajectories = [traj for traj in data['trajectories'] if traj['genotype'] == genotype]
    filtered_genotypes = [g for g in data['genotypes'] if g == genotype]
    filtered_onset_times = data['onset_times'][genotype_mask]
    
    result = {
        'trajectories': filtered_trajectories,
        'onset_times': filtered_onset_times,
        'genotypes': filtered_genotypes
    }
    
    # Include treatment_start_times if present
    if 'treatment_start_times' in data:
        filtered_treatment_start_times = data['treatment_start_times'][genotype_mask]
        result['treatment_start_times'] = filtered_treatment_start_times
    
    return result

def plot_genotype_specific_plots(control_data, treatment_data, output_dir, genotype, 
                                  follow_up_years=1.5, mci_threshold=5.2, n_per_arm=None):
    """Plot survival curves and neuroinflammation trajectories for a specific genotype."""
    # Filter data by genotype
    control_genotype_data = filter_data_by_genotype(control_data, genotype)
    treatment_genotype_data = filter_data_by_genotype(treatment_data, genotype)
    
    n_genotype_control = len(control_genotype_data['trajectories'])
    n_genotype_treatment = len(treatment_genotype_data['trajectories'])
    
    if n_genotype_control == 0 or n_genotype_treatment == 0:
        print(f"  Skipping {genotype}: insufficient data (control: {n_genotype_control}, treatment: {n_genotype_treatment})")
        return
    
    print(f"  Plotting {genotype} (control: {n_genotype_control}, treatment: {n_genotype_treatment})...")
    
    # Calculate statistics for this genotype
    control_trajs = control_genotype_data['trajectories']
    treatment_trajs = treatment_genotype_data['trajectories']
    treatment_start_times = treatment_genotype_data.get('treatment_start_times', [])
    
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
            control_genotype_data['onset_times'],
            treatment_genotype_data['onset_times']
        )
    
    # Plot survival curves
    plot_survival_curves(control_genotype_data, treatment_genotype_data, output_dir, 
                        stats_results, n_genotype_control, genotype)
    
    # Plot neuroinflammation trajectories
    plot_neuroinflammation_trajectories(control_genotype_data, treatment_genotype_data, output_dir,
                                       follow_up_years, mci_threshold, stats_results, 
                                       n_genotype_control, genotype)
    
    # Plot neuroinflammation difference
    plot_neuroinflammation_difference(control_genotype_data, treatment_genotype_data, output_dir,
                                     n_genotype_control, genotype)
    
    # Plot senescence difference
    plot_senescence_difference(control_genotype_data, treatment_genotype_data, output_dir,
                              n_genotype_control, genotype)

def plot_se_vs_sample_size(APOE33_params, sample_sizes=[10, 50, 100, 150, 200], 
                            mci_threshold=5.2, s_fold_change=0.3, interval_days=90.0,
                            treatment_duration_days=1.5*365.0, follow_up_years=1.5,
                            dt=1.0, base_seed=2025, sample_every_days=30.0, output_dir='mixed_population_trial_output'):
    """
    Run simulations for multiple sample sizes and plot neuroinflammation difference vs age.
    
    Parameters:
    -----------
    sample_sizes : list
        List of sample sizes to test
    """
    print("="*80)
    print("Neuroinflammation Difference vs Age for Multiple Sample Sizes")
    print("="*80)
    
    # Store difference trajectories and p-values for each sample size
    diff_trajectories = {}  # {sample_size: (t_common, diff_mean, diff_se)}
    p_values = {}  # {sample_size: p_value}
    followup_distributions = {}  # {sample_size: (control_n_at_followup, treatment_n_at_followup)}
    
    # Create color gradient (light to dark blue)
    from matplotlib.colors import LinearSegmentedColormap
    n_colors = len(sample_sizes)
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, n_colors))
    
    for idx, n_per_arm in enumerate(sample_sizes):
        print(f"\nRunning simulation with n={n_per_arm} per arm...")
        
        # Run simulation
        control_data, treatment_data = simulate_mixed_population_trial(
            APOE33_params, n_per_arm, mci_threshold, s_fold_change,
            interval_days, treatment_duration_days, follow_up_years,
            dt, base_seed + n_per_arm, sample_every_days  # Different seed for each sample size
        )
        
        # Control arm
        control_trajs = control_data['trajectories']
        control_n_arrays = [traj['n'] for traj in control_trajs]
        control_t_arrays = [traj['t'] for traj in control_trajs]
        
        # Treatment arm
        treatment_trajs = treatment_data['trajectories']
        treatment_n_arrays = [traj['n'] for traj in treatment_trajs]
        treatment_t_arrays = [traj['t'] for traj in treatment_trajs]
        treatment_start_times = treatment_data.get('treatment_start_times', [])
        
        # Calculate p-value at follow-up time
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
        
        # Calculate p-value using t-test
        p_value = np.nan
        if len(control_n_at_followup) > 0 and len(treatment_n_at_followup) > 0:
            control_n_array = np.array(control_n_at_followup)
            treatment_n_array = np.array(treatment_n_at_followup)
            # Remove any NaN or infinite values
            control_clean = control_n_array[np.isfinite(control_n_array)]
            treatment_clean = treatment_n_array[np.isfinite(treatment_n_array)]
            
            if len(control_clean) > 1 and len(treatment_clean) > 1:
                t_stat, p_value = stats.ttest_ind(treatment_clean, control_clean)
                print(f"  p-value (t-test): {p_value:.4f}")
            else:
                print(f"  p-value: insufficient data")
        else:
            print(f"  p-value: no valid data points")
        
        p_values[n_per_arm] = p_value
        followup_distributions[n_per_arm] = (control_n_at_followup, treatment_n_at_followup)
        
        # Find common time range (overlap of both arms), limited to 50-90 years
        min_t_all = max([max([t[0] for t in control_t_arrays]), max([t[0] for t in treatment_t_arrays])])
        max_t_all = min([min([t[-1] for t in control_t_arrays]), min([t[-1] for t in treatment_t_arrays])])
        # Limit to 50-90 years
        min_t_all = max(min_t_all, 50.0)
        max_t_all = min(max_t_all, 90.0)
        t_common = np.linspace(min_t_all, max_t_all, 500)
        
        # Interpolate control trajectories to common time
        n_interp_control = []
        for t, n in zip(control_t_arrays, control_n_arrays):
            f_n = interp1d(t, n, kind='linear', bounds_error=False, fill_value=np.nan)
            n_interp_control.append(f_n(t_common))
        
        n_interp_control = np.array(n_interp_control)
        n_mean_control = np.nanmean(n_interp_control, axis=0)
        n_std_control = np.nanstd(n_interp_control, axis=0)
        n_se_control = n_std_control / np.sqrt(len(control_trajs))
        
        # Interpolate treatment trajectories to common time
        n_interp_treatment = []
        for t, n in zip(treatment_t_arrays, treatment_n_arrays):
            f_n = interp1d(t, n, kind='linear', bounds_error=False, fill_value=np.nan)
            n_interp_treatment.append(f_n(t_common))
        
        n_interp_treatment = np.array(n_interp_treatment)
        n_mean_treatment = np.nanmean(n_interp_treatment, axis=0)
        n_std_treatment = np.nanstd(n_interp_treatment, axis=0)
        n_se_treatment = n_std_treatment / np.sqrt(len(treatment_trajs))
        
        # Calculate difference: control - treatment (positive means treatment reduces neuroinflammation)
        diff_n = n_mean_control - n_mean_treatment
        # SE of difference: sqrt(SE_control^2 + SE_treatment^2)
        diff_se_n = np.sqrt(n_se_control**2 + n_se_treatment**2)
        
        diff_trajectories[n_per_arm] = (t_common, diff_n, diff_se_n)
        print(f"  Calculated difference trajectory (n={n_per_arm})")
    
    # Plot all differences on the same plot with color gradient
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for idx, n_per_arm in enumerate(sample_sizes):
        if n_per_arm in diff_trajectories:
            t_common, diff_n, diff_se_n = diff_trajectories[n_per_arm]
            color = colors[idx]
            
            valid_mask = ~np.isnan(diff_n)
            ax.plot(t_common[valid_mask], diff_n[valid_mask], 
                   color=color, linewidth=2.5, 
                   label=f'n={n_per_arm}', alpha=0.8)
            ax.fill_between(t_common[valid_mask], 
                           diff_n[valid_mask] - diff_se_n[valid_mask],
                           diff_n[valid_mask] + diff_se_n[valid_mask],
                           color=color, alpha=0.15)
    
    ax.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.3)
    ax.set_xlabel('Age (years)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Δn (Control - Treatment)', fontsize=12, fontweight='bold')
    ax.set_title('Neuroinflammation Difference vs Age for Different Sample Sizes', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best', framealpha=0.9, title='Sample Size (n per arm)')
    ax.set_xlim(50, 90)
    sns.despine(ax=ax)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add p-value annotations at the top of the plot
    annotation_lines = []
    for n_per_arm in sample_sizes:
        if n_per_arm in p_values:
            p_val = p_values[n_per_arm]
            if np.isfinite(p_val):
                annotation_lines.append(f'n={n_per_arm}: p={p_val:.4f}')
            else:
                annotation_lines.append(f'n={n_per_arm}: p=NaN')
    
    if len(annotation_lines) > 0:
        annotation_text = '\n'.join(annotation_lines)
        ax.text(0.02, 0.98, annotation_text, 
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    filename = 'neuroinflammation_difference_vs_age_multiple_n.png'
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved neuroinflammation difference vs age plot to {output_dir}/{filename}")
    
    # Save difference trajectories to CSV
    print("\nSaving difference trajectories data to CSV...")
    diff_trajectories_data = []
    for n_per_arm in sample_sizes:
        if n_per_arm in diff_trajectories:
            t_common, diff_n, diff_se_n = diff_trajectories[n_per_arm]
            for age, delta_n, delta_se in zip(t_common, diff_n, diff_se_n):
                if np.isfinite(delta_n) and np.isfinite(delta_se):
                    diff_trajectories_data.append({
                        'sample_size': n_per_arm,
                        'age': age,
                        'delta_n': delta_n,
                        'delta_n_se': delta_se
                    })
    
    if len(diff_trajectories_data) > 0:
        df_diff = pd.DataFrame(diff_trajectories_data)
        csv_filename = 'neuroinflammation_difference_vs_age_data.csv'
        df_diff.to_csv(os.path.join(output_dir, csv_filename), index=False)
        print(f"  Saved {csv_filename}")
    
    # Save p-values to CSV
    pvalues_data = []
    for n_per_arm in sample_sizes:
        if n_per_arm in p_values:
            pvalues_data.append({
                'sample_size': n_per_arm,
                'p_value': p_values[n_per_arm]
            })
    
    if len(pvalues_data) > 0:
        df_pvalues = pd.DataFrame(pvalues_data)
        csv_filename = 'p_values_by_sample_size.csv'
        df_pvalues.to_csv(os.path.join(output_dir, csv_filename), index=False)
        print(f"  Saved {csv_filename}")
    
    # Save follow-up distributions to CSV
    distributions_data = []
    for n_per_arm in sample_sizes:
        if n_per_arm in followup_distributions:
            control_n, treatment_n = followup_distributions[n_per_arm]
            for val in control_n:
                if np.isfinite(val):
                    distributions_data.append({
                        'sample_size': n_per_arm,
                        'arm': 'control',
                        'neuroinflammation': val
                    })
            for val in treatment_n:
                if np.isfinite(val):
                    distributions_data.append({
                        'sample_size': n_per_arm,
                        'arm': 'treatment',
                        'neuroinflammation': val
                    })
    
    if len(distributions_data) > 0:
        df_dist = pd.DataFrame(distributions_data)
        csv_filename = 'neuroinflammation_distributions_at_followup.csv'
        df_dist.to_csv(os.path.join(output_dir, csv_filename), index=False)
        print(f"  Saved {csv_filename}")
    
    # Plot distributions of neuroinflammation at follow-up time
    n_subplots = len(sample_sizes)
    n_cols = 2
    n_rows = (n_subplots + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5*n_rows))
    if n_subplots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, n_per_arm in enumerate(sample_sizes):
        ax = axes[idx]
        
        if n_per_arm in followup_distributions:
            control_n_at_followup, treatment_n_at_followup = followup_distributions[n_per_arm]
            
            # Filter out NaN and infinite values
            control_clean = [n for n in control_n_at_followup if np.isfinite(n)]
            treatment_clean = [n for n in treatment_n_at_followup if np.isfinite(n)]
            
            if len(control_clean) > 0 and len(treatment_clean) > 0:
                # Plot histograms with density
                ax.hist(control_clean, bins=20, alpha=0.6, color='blue', 
                       label=f'Control (n={len(control_clean)})', density=True, edgecolor='black', linewidth=0.5)
                ax.hist(treatment_clean, bins=20, alpha=0.6, color='red', 
                       label=f'Treatment (n={len(treatment_clean)})', density=True, edgecolor='black', linewidth=0.5)
                
                # Add vertical lines for means
                mean_control = np.mean(control_clean)
                mean_treatment = np.mean(treatment_clean)
                ax.axvline(mean_control, color='blue', linestyle='--', linewidth=2, alpha=0.8, label=f'Control mean={mean_control:.3f}')
                ax.axvline(mean_treatment, color='red', linestyle='--', linewidth=2, alpha=0.8, label=f'Treatment mean={mean_treatment:.3f}')
                
                # Add p-value annotation (in scientific notation)
                if n_per_arm in p_values and np.isfinite(p_values[n_per_arm]):
                    p_val = p_values[n_per_arm]
                    # Format p-value in scientific notation
                    p_val_str = f'p={p_val:.2e}'
                    ax.text(0.95, 0.95, p_val_str, 
                           transform=ax.transAxes, fontsize=11, 
                           verticalalignment='top', horizontalalignment='right',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.set_xlabel('Neuroinflammation (n) at Follow-up', fontsize=11, fontweight='bold')
        ax.set_ylabel('Density', fontsize=11, fontweight='bold')
        ax.set_title(f'Sample Size: n={n_per_arm} per arm', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        sns.despine(ax=ax)
    
    # Hide unused subplots
    for idx in range(n_subplots, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    filename = 'neuroinflammation_distribution_at_followup_multiple_n.png'
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved neuroinflammation distribution plot to {output_dir}/{filename}")
    
    # Print summary
    print("\n" + "="*80)
    print("Summary: Neuroinflammation Difference Analysis")
    print("="*80)
    print("Difference trajectories calculated for all sample sizes.")
    print("Plot shows Δn (Control - Treatment) as a function of age.")
    print("="*80)

def plot_delta_n_vs_parameters(APOE33_params, arm_sizes=[10, 50, 100],
                               base_mci_threshold=5.2, base_s_fold_change=0.3,
                               base_interval_days=90.0, base_treatment_duration_days=1.5*365.0,
                               base_follow_up_years=1.5, dt=1.0, base_seed=2025,
                               sample_every_days=30.0, output_dir='mixed_population_trial_output'):
    """
    Plot delta n (neuroinflammation difference) vs different parameters.
    
    Creates 4 plots:
    1. Delta n vs frequency of treatment (interval_days)
    2. Delta n vs duration of trial (follow_up_years)
    3. Delta n vs MCI threshold
    4. Delta n vs dose (s_fold_change)
    
    Each plot shows curves for different arm sizes (10, 50, 100) with p-value annotations.
    """
    print("="*80)
    print("Delta n vs Parameters Analysis")
    print("="*80)
    
    # Define parameter ranges
    frequency_values = [30.0, 60.0, 90.0, 120.0, 180.0]  # days (1 month to 6 months)
    duration_values = [0.5, 1.0, 1.5, 2.0, 2.5]  # years
    mci_threshold_values = [4.5, 5.0, 5.2, 5.5, 6.0]
    dose_values = [0.1, 0.2, 0.3, 0.4, 0.5]  # s_fold_change (90% to 50% reduction)
    
    # Create color gradient for arm sizes
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(arm_sizes)))
    
    # 1. Plot Delta n vs Frequency of Treatment
    print("\n1. Calculating Delta n vs Treatment Frequency...")
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    
    frequency_data = []  # Store data for CSV
    
    for arm_idx, n_per_arm in enumerate(arm_sizes):
        delta_n_list = []
        p_value_list = []
        
        for interval_days in frequency_values:
            print(f"  n={n_per_arm}, interval={interval_days} days...")
            control_data, treatment_data = simulate_mixed_population_trial(
                APOE33_params, n_per_arm, base_mci_threshold, base_s_fold_change,
                interval_days, base_treatment_duration_days, base_follow_up_years,
                dt, base_seed + int(interval_days) + n_per_arm, sample_every_days
            )
            
            # Calculate delta n and p-value at follow-up with bootstrap
            delta_n, delta_n_se, delta_n_ci_lower, delta_n_ci_upper, p_val = calculate_delta_n_and_pvalue(
                control_data, treatment_data, base_mci_threshold, base_follow_up_years, n_bootstrap=100
            )
            delta_n_list.append(delta_n)
            p_value_list.append(p_val)
            
            # Store for CSV
            frequency_data.append({
                'arm_size': n_per_arm,
                'treatment_frequency_days': interval_days,
                'delta_n': delta_n,
                'delta_n_se': delta_n_se,
                'delta_n_ci_lower': delta_n_ci_lower,
                'delta_n_ci_upper': delta_n_ci_upper,
                'p_value': p_val
            })
        
        # Plot curve
        valid_mask = [i for i, (dn, pv) in enumerate(zip(delta_n_list, p_value_list)) 
                     if np.isfinite(dn) and np.isfinite(pv)]
        if len(valid_mask) > 0:
            valid_freq = [frequency_values[i] for i in valid_mask]
            valid_delta_n = [delta_n_list[i] for i in valid_mask]
            valid_p = [p_value_list[i] for i in valid_mask]
            
            # Get error bars from stored data
            valid_se = []
            for i in valid_mask:
                freq_val = frequency_values[i]
                matching_data = [d for d in frequency_data if d['arm_size'] == n_per_arm and 
                               d['treatment_frequency_days'] == freq_val]
                if len(matching_data) > 0 and np.isfinite(matching_data[0]['delta_n_se']):
                    valid_se.append(matching_data[0]['delta_n_se'])
                else:
                    valid_se.append(0.0)
            
            ax1.errorbar(valid_freq, valid_delta_n, yerr=valid_se, 
                        fmt='o-', color=colors[arm_idx], 
                        linewidth=2.5, markersize=8, capsize=5, capthick=2,
                        label=f'n={n_per_arm}', alpha=0.8)
            
            # Add p-value annotations
            for x, y, p in zip(valid_freq, valid_delta_n, valid_p):
                if np.isfinite(p):
                    p_str = f'p={p:.2e}'
                    ax1.annotate(p_str, (x, y), xytext=(5, 5), textcoords='offset points',
                               fontsize=8, alpha=0.7, bbox=dict(boxstyle='round,pad=0.3', 
                               facecolor='wheat', alpha=0.5))
    
    ax1.set_xlabel('Treatment Frequency (days)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Δn (Control - Treatment)', fontsize=12, fontweight='bold')
    ax1.set_title('Delta n vs Treatment Frequency', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11, loc='best', framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    sns.despine(ax=ax1)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'delta_n_vs_treatment_frequency.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved delta_n_vs_treatment_frequency.png")
    
    # Save frequency data to CSV
    if len(frequency_data) > 0:
        df_freq = pd.DataFrame(frequency_data)
        df_freq.to_csv(os.path.join(output_dir, 'delta_n_vs_treatment_frequency_data.csv'), index=False)
        print("  Saved delta_n_vs_treatment_frequency_data.csv")
    
    # 2. Plot Delta n vs Duration of Trial
    print("\n2. Calculating Delta n vs Trial Duration...")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    duration_data = []  # Store data for CSV
    
    for arm_idx, n_per_arm in enumerate(arm_sizes):
        delta_n_list = []
        p_value_list = []
        
        for follow_up_years in duration_values:
            print(f"  n={n_per_arm}, follow-up={follow_up_years} years...")
            control_data, treatment_data = simulate_mixed_population_trial(
                APOE33_params, n_per_arm, base_mci_threshold, base_s_fold_change,
                base_interval_days, base_treatment_duration_days, follow_up_years,
                dt, base_seed + int(follow_up_years*100) + n_per_arm, sample_every_days
            )
            
            delta_n, delta_n_se, delta_n_ci_lower, delta_n_ci_upper, p_val = calculate_delta_n_and_pvalue(
                control_data, treatment_data, base_mci_threshold, follow_up_years, n_bootstrap=100
            )
            delta_n_list.append(delta_n)
            p_value_list.append(p_val)
            
            # Store for CSV
            duration_data.append({
                'arm_size': n_per_arm,
                'trial_duration_years': follow_up_years,
                'delta_n': delta_n,
                'delta_n_se': delta_n_se,
                'delta_n_ci_lower': delta_n_ci_lower,
                'delta_n_ci_upper': delta_n_ci_upper,
                'p_value': p_val
            })
        
        valid_mask = [i for i, (dn, pv) in enumerate(zip(delta_n_list, p_value_list)) 
                     if np.isfinite(dn) and np.isfinite(pv)]
        if len(valid_mask) > 0:
            valid_dur = [duration_values[i] for i in valid_mask]
            valid_delta_n = [delta_n_list[i] for i in valid_mask]
            valid_p = [p_value_list[i] for i in valid_mask]
            
            # Get error bars from stored data
            valid_se = []
            for i in valid_mask:
                dur_val = duration_values[i]
                matching_data = [d for d in duration_data if d['arm_size'] == n_per_arm and 
                               d['trial_duration_years'] == dur_val]
                if len(matching_data) > 0 and np.isfinite(matching_data[0]['delta_n_se']):
                    valid_se.append(matching_data[0]['delta_n_se'])
                else:
                    valid_se.append(0.0)
            
            ax2.errorbar(valid_dur, valid_delta_n, yerr=valid_se, 
                        fmt='o-', color=colors[arm_idx], 
                        linewidth=2.5, markersize=8, capsize=5, capthick=2,
                        label=f'n={n_per_arm}', alpha=0.8)
            
            for x, y, p in zip(valid_dur, valid_delta_n, valid_p):
                if np.isfinite(p):
                    p_str = f'p={p:.2e}'
                    ax2.annotate(p_str, (x, y), xytext=(5, 5), textcoords='offset points',
                               fontsize=8, alpha=0.7, bbox=dict(boxstyle='round,pad=0.3', 
                               facecolor='wheat', alpha=0.5))
    
    ax2.set_xlabel('Trial Duration / Follow-up Time (years)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Δn (Control - Treatment)', fontsize=12, fontweight='bold')
    ax2.set_title('Delta n vs Trial Duration', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11, loc='best', framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    sns.despine(ax=ax2)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'delta_n_vs_trial_duration.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved delta_n_vs_trial_duration.png")
    
    # Save duration data to CSV
    if len(duration_data) > 0:
        df_dur = pd.DataFrame(duration_data)
        df_dur.to_csv(os.path.join(output_dir, 'delta_n_vs_trial_duration_data.csv'), index=False)
        print("  Saved delta_n_vs_trial_duration_data.csv")
    
    # 3. Plot Delta n vs MCI Threshold
    print("\n3. Calculating Delta n vs MCI Threshold...")
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    
    mci_threshold_data = []  # Store data for CSV
    
    for arm_idx, n_per_arm in enumerate(arm_sizes):
        delta_n_list = []
        p_value_list = []
        
        for mci_threshold in mci_threshold_values:
            print(f"  n={n_per_arm}, MCI threshold={mci_threshold}...")
            control_data, treatment_data = simulate_mixed_population_trial(
                APOE33_params, n_per_arm, mci_threshold, base_s_fold_change,
                base_interval_days, base_treatment_duration_days, base_follow_up_years,
                dt, base_seed + int(mci_threshold*100) + n_per_arm, sample_every_days
            )
            
            delta_n, delta_n_se, delta_n_ci_lower, delta_n_ci_upper, p_val = calculate_delta_n_and_pvalue(
                control_data, treatment_data, mci_threshold, base_follow_up_years, n_bootstrap=100
            )
            delta_n_list.append(delta_n)
            p_value_list.append(p_val)
            
            # Store for CSV
            mci_threshold_data.append({
                'arm_size': n_per_arm,
                'mci_threshold': mci_threshold,
                'delta_n': delta_n,
                'delta_n_se': delta_n_se,
                'delta_n_ci_lower': delta_n_ci_lower,
                'delta_n_ci_upper': delta_n_ci_upper,
                'p_value': p_val
            })
        
        valid_mask = [i for i, (dn, pv) in enumerate(zip(delta_n_list, p_value_list)) 
                     if np.isfinite(dn) and np.isfinite(pv)]
        if len(valid_mask) > 0:
            valid_mci = [mci_threshold_values[i] for i in valid_mask]
            valid_delta_n = [delta_n_list[i] for i in valid_mask]
            valid_p = [p_value_list[i] for i in valid_mask]
            
            # Get error bars from stored data
            valid_se = []
            for i in valid_mask:
                mci_val = mci_threshold_values[i]
                matching_data = [d for d in mci_threshold_data if d['arm_size'] == n_per_arm and 
                               d['mci_threshold'] == mci_val]
                if len(matching_data) > 0 and np.isfinite(matching_data[0]['delta_n_se']):
                    valid_se.append(matching_data[0]['delta_n_se'])
                else:
                    valid_se.append(0.0)
            
            ax3.errorbar(valid_mci, valid_delta_n, yerr=valid_se, 
                        fmt='o-', color=colors[arm_idx], 
                        linewidth=2.5, markersize=8, capsize=5, capthick=2,
                        label=f'n={n_per_arm}', alpha=0.8)
            
            for x, y, p in zip(valid_mci, valid_delta_n, valid_p):
                if np.isfinite(p):
                    p_str = f'p={p:.2e}'
                    ax3.annotate(p_str, (x, y), xytext=(5, 5), textcoords='offset points',
                               fontsize=8, alpha=0.7, bbox=dict(boxstyle='round,pad=0.3', 
                               facecolor='wheat', alpha=0.5))
    
    ax3.set_xlabel('MCI Threshold', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Δn (Control - Treatment)', fontsize=12, fontweight='bold')
    ax3.set_title('Delta n vs MCI Threshold', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=11, loc='best', framealpha=0.9)
    ax3.grid(True, alpha=0.3, linestyle='--')
    sns.despine(ax=ax3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'delta_n_vs_mci_threshold.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved delta_n_vs_mci_threshold.png")
    
    # Save MCI threshold data to CSV
    if len(mci_threshold_data) > 0:
        df_mci = pd.DataFrame(mci_threshold_data)
        df_mci.to_csv(os.path.join(output_dir, 'delta_n_vs_mci_threshold_data.csv'), index=False)
        print("  Saved delta_n_vs_mci_threshold_data.csv")
    
    # 4. Plot Delta n vs Dose
    print("\n4. Calculating Delta n vs Dose...")
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    
    dose_data = []  # Store data for CSV
    
    for arm_idx, n_per_arm in enumerate(arm_sizes):
        delta_n_list = []
        p_value_list = []
        
        for s_fold_change in dose_values:
            print(f"  n={n_per_arm}, dose (s_fold_change)={s_fold_change}...")
            control_data, treatment_data = simulate_mixed_population_trial(
                APOE33_params, n_per_arm, base_mci_threshold, s_fold_change,
                base_interval_days, base_treatment_duration_days, base_follow_up_years,
                dt, base_seed + int(s_fold_change*1000) + n_per_arm, sample_every_days
            )
            
            delta_n, delta_n_se, delta_n_ci_lower, delta_n_ci_upper, p_val = calculate_delta_n_and_pvalue(
                control_data, treatment_data, base_mci_threshold, base_follow_up_years, n_bootstrap=100
            )
            delta_n_list.append(delta_n)
            p_value_list.append(p_val)
            
            # Store for CSV
            dose_data.append({
                'arm_size': n_per_arm,
                'dose_s_fold_change': s_fold_change,
                'delta_n': delta_n,
                'delta_n_se': delta_n_se,
                'delta_n_ci_lower': delta_n_ci_lower,
                'delta_n_ci_upper': delta_n_ci_upper,
                'p_value': p_val
            })
        
        valid_mask = [i for i, (dn, pv) in enumerate(zip(delta_n_list, p_value_list)) 
                     if np.isfinite(dn) and np.isfinite(pv)]
        if len(valid_mask) > 0:
            valid_dose = [dose_values[i] for i in valid_mask]
            valid_delta_n = [delta_n_list[i] for i in valid_mask]
            valid_p = [p_value_list[i] for i in valid_mask]
            
            # Get error bars from stored data
            valid_se = []
            for i in valid_mask:
                dose_val = dose_values[i]
                matching_data = [d for d in dose_data if d['arm_size'] == n_per_arm and 
                               d['dose_s_fold_change'] == dose_val]
                if len(matching_data) > 0 and np.isfinite(matching_data[0]['delta_n_se']):
                    valid_se.append(matching_data[0]['delta_n_se'])
                else:
                    valid_se.append(0.0)
            
            ax4.errorbar(valid_dose, valid_delta_n, yerr=valid_se, 
                        fmt='o-', color=colors[arm_idx], 
                        linewidth=2.5, markersize=8, capsize=5, capthick=2,
                        label=f'n={n_per_arm}', alpha=0.8)
            
            for x, y, p in zip(valid_dose, valid_delta_n, valid_p):
                if np.isfinite(p):
                    p_str = f'p={p:.2e}'
                    ax4.annotate(p_str, (x, y), xytext=(5, 5), textcoords='offset points',
                               fontsize=8, alpha=0.7, bbox=dict(boxstyle='round,pad=0.3', 
                               facecolor='wheat', alpha=0.5))
    
    ax4.set_xlabel('Dose (s_fold_change)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Δn (Control - Treatment)', fontsize=12, fontweight='bold')
    ax4.set_title('Delta n vs Dose', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=11, loc='best', framealpha=0.9)
    ax4.grid(True, alpha=0.3, linestyle='--')
    sns.despine(ax=ax4)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'delta_n_vs_dose.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved delta_n_vs_dose.png")
    
    # Save dose data to CSV
    if len(dose_data) > 0:
        df_dose = pd.DataFrame(dose_data)
        df_dose.to_csv(os.path.join(output_dir, 'delta_n_vs_dose_data.csv'), index=False)
        print("  Saved delta_n_vs_dose_data.csv")
    
    print("\n" + "="*80)
    print("Delta n vs Parameters Analysis Complete!")
    print("="*80)

def calculate_delta_n_and_pvalue(control_data, treatment_data, mci_threshold, follow_up_years, n_bootstrap=100):
    """
    Helper function to calculate delta n and p-value at follow-up time with bootstrap confidence intervals.
    
    Parameters:
    -----------
    n_bootstrap : int
        Number of bootstrap iterations (default: 20)
    
    Returns:
    --------
    delta_n : float
        Mean difference in neuroinflammation (control - treatment)
    delta_n_se : float
        Standard error of delta_n from bootstrap
    delta_n_ci_lower : float
        Lower bound of 95% confidence interval
    delta_n_ci_upper : float
        Upper bound of 95% confidence interval
    p_value : float
        P-value from t-test
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
    
    # Calculate delta n and p-value
    delta_n = np.nan
    delta_n_se = np.nan
    delta_n_ci_lower = np.nan
    delta_n_ci_upper = np.nan
    p_value = np.nan
    
    if len(control_n_at_followup) > 0 and len(treatment_n_at_followup) > 0:
        control_n_array = np.array(control_n_at_followup)
        treatment_n_array = np.array(treatment_n_at_followup)
        
        control_clean = control_n_array[np.isfinite(control_n_array)]
        treatment_clean = treatment_n_array[np.isfinite(treatment_n_array)]
        
        if len(control_clean) > 0 and len(treatment_clean) > 0:
            mean_control = np.mean(control_clean)
            mean_treatment = np.mean(treatment_clean)
            delta_n = mean_control - mean_treatment
            
            # Bootstrap resampling for confidence intervals
            bootstrap_delta_n = []
            for _ in range(n_bootstrap):
                # Resample with replacement
                control_boot = np.random.choice(control_clean, size=len(control_clean), replace=True)
                treatment_boot = np.random.choice(treatment_clean, size=len(treatment_clean), replace=True)
                delta_n_boot = np.mean(control_boot) - np.mean(treatment_boot)
                bootstrap_delta_n.append(delta_n_boot)
            
            bootstrap_delta_n = np.array(bootstrap_delta_n)
            delta_n_se = np.std(bootstrap_delta_n, ddof=1)
            delta_n_ci_lower = np.percentile(bootstrap_delta_n, 2.5)
            delta_n_ci_upper = np.percentile(bootstrap_delta_n, 97.5)
            
            if len(control_clean) > 1 and len(treatment_clean) > 1:
                t_stat, p_value = stats.ttest_ind(treatment_clean, control_clean)
    
    return delta_n, delta_n_se, delta_n_ci_lower, delta_n_ci_upper, p_value

def calculate_effect_size_and_pvalue(control_data, treatment_data, mci_threshold, follow_up_years):
    """
    Calculate effect size (Cohen's d) and p-value at follow-up time.
    
    Returns:
    --------
    cohens_d : float
        Cohen's d effect size
    p_value : float
        P-value from t-test
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
    
    # Calculate effect size and p-value
    cohens_d = np.nan
    cohens_d_se = np.nan
    p_value = np.nan
    
    if len(control_n_at_followup) > 0 and len(treatment_n_at_followup) > 0:
        control_n_array = np.array(control_n_at_followup)
        treatment_n_array = np.array(treatment_n_at_followup)
        
        control_clean = control_n_array[np.isfinite(control_n_array)]
        treatment_clean = treatment_n_array[np.isfinite(treatment_n_array)]
        
        if len(control_clean) > 0 and len(treatment_clean) > 0:
            mean_control = np.mean(control_clean)
            mean_treatment = np.mean(treatment_clean)
            mean_diff = mean_control - mean_treatment  # control - treatment (positive for beneficial treatment)
            
            std_control = np.std(control_clean, ddof=1)
            std_treatment = np.std(treatment_clean, ddof=1)
            
            # Calculate pooled standard deviation for Cohen's d
            pooled_std = np.sqrt(((len(control_clean) - 1) * std_control**2 + 
                                 (len(treatment_clean) - 1) * std_treatment**2) / 
                                (len(control_clean) + len(treatment_clean) - 2))
            
            cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0.0
            
            # Calculate standard error of Cohen's d using the formula
            n1 = len(control_clean)
            n2 = len(treatment_clean)
            df = n1 + n2 - 2
            if df > 0 and pooled_std > 0:
                # Standard error of Cohen's d: SE(d) = sqrt((n1+n2)/(n1*n2) + d^2/(2*df))
                cohens_d_se = np.sqrt((n1 + n2) / (n1 * n2) + cohens_d**2 / (2 * df))
            else:
                cohens_d_se = np.nan
            
            if len(control_clean) > 1 and len(treatment_clean) > 1:
                t_stat, p_value = stats.ttest_ind(treatment_clean, control_clean)
    
    return cohens_d, cohens_d_se, p_value

def calculate_effect_size_and_pvalue_with_std(APOE33_params, n_per_arm, mci_threshold, 
                                             s_fold_change, interval_days, 
                                             treatment_duration_days, follow_up_years,
                                             dt, base_seed, sample_every_days):
    """
    Calculate effect size (Cohen's d) and p-value with standard error using the formula.
    
    Returns:
    --------
    cohens_d : float
        Cohen's d effect size
    cohens_d_se : float
        Standard error of Cohen's d (calculated using formula)
    p_value : float
        P-value from t-test
    """
    control_data, treatment_data = simulate_mixed_population_trial(
        APOE33_params, n_per_arm, mci_threshold, s_fold_change,
        interval_days, treatment_duration_days, follow_up_years,
        dt, base_seed, sample_every_days
    )
    
    cohens_d, cohens_d_se, p_val = calculate_effect_size_and_pvalue(
        control_data, treatment_data, mci_threshold, follow_up_years
    )
    
    return cohens_d, cohens_d_se, p_val

def plot_effect_size_and_pvalue_vs_arm_size(APOE33_params,
                                           base_mci_threshold=5.2, base_s_fold_change=0.3,
                                           base_interval_days=90.0, base_treatment_duration_days=1.5*365.0,
                                           base_follow_up_years=1.5, dt=1.0, base_seed=2025,
                                           sample_every_days=30.0, output_dir='mixed_population_trial_output'):
    """
    Plot effect size (Cohen's d) and p-value vs arm size for different parameter values.
    
    Creates 4 figures:
    1. Follow-up time: effect size and p-value vs arm size for different follow-up times
    2. Treatment frequency: effect size and p-value vs arm size for different frequencies
    3. Dose: effect size and p-value vs arm size for different doses
    4. MCI threshold: effect size and p-value vs arm size for different MCI thresholds
    
    Each figure has 2 rows: top row = effect size, bottom row = p-value
    Each row has subplots for different parameter values.
    """
    print("="*80)
    print("Effect Size and P-value vs Arm Size Analysis")
    print("="*80)
    
    arm_sizes = [10, 25, 50, 75, 100, 150]
    
    # Define parameter ranges
    duration_values = [0.5, 1.0, 1.5, 2.0, 2.5]  # years
    frequency_values = [30.0, 60.0, 90.0, 120.0, 180.0]  # days
    dose_values = [0.1, 0.2, 0.3, 0.4, 0.5]  # s_fold_change
    mci_threshold_values = [4.5, 5.0, 5.2, 5.5, 6.0]
    
    # Create color gradient for parameter values
    from matplotlib.colors import LinearSegmentedColormap
    
    # 1. Follow-up Time Figure
    print("\n1. Calculating Effect Size and P-value vs Arm Size for Follow-up Time...")
    n_subplots = len(duration_values)
    fig1, axes1 = plt.subplots(2, n_subplots + 1, figsize=(5*(n_subplots + 1), 10))
    if n_subplots == 1:
        axes1 = axes1.reshape(2, 2)
    
    colors_dur = plt.cm.viridis(np.linspace(0.2, 0.8, n_subplots))
    followup_data = []
    
    # Collect all effect size and p-value values to set common y-axis limits
    all_cohens_d_values = []
    all_p_value_values = []
    
    # Collect data for summary plots: organized by arm size
    summary_data_by_arm = {n: {'param_values': [], 'cohens_d': [], 'cohens_d_std': [], 'p_value': []} 
                           for n in arm_sizes}
    
    for param_idx, follow_up_years in enumerate(duration_values):
        ax_effect = axes1[0, param_idx]
        ax_pvalue = axes1[1, param_idx]
        
        cohens_d_list = []
        cohens_d_std_list = []
        p_value_list = []
        
        for n_per_arm in arm_sizes:
            print(f"  Follow-up={follow_up_years} years, n={n_per_arm}...")
            cohens_d_mean, cohens_d_std, p_val = calculate_effect_size_and_pvalue_with_std(
                APOE33_params, n_per_arm, base_mci_threshold, base_s_fold_change,
                base_interval_days, base_treatment_duration_days, follow_up_years,
                dt, base_seed + int(follow_up_years*100) + n_per_arm, sample_every_days
            )
            cohens_d_list.append(cohens_d_mean)
            cohens_d_std_list.append(cohens_d_std)
            p_value_list.append(p_val)
            
            if np.isfinite(cohens_d_mean):
                all_cohens_d_values.append(cohens_d_mean)
            if np.isfinite(p_val) and p_val > 0:
                all_p_value_values.append(p_val)
            
            # Store for summary plots
            if np.isfinite(cohens_d_mean) and np.isfinite(p_val):
                summary_data_by_arm[n_per_arm]['param_values'].append(follow_up_years)
                summary_data_by_arm[n_per_arm]['cohens_d'].append(cohens_d_mean)
                summary_data_by_arm[n_per_arm]['cohens_d_std'].append(cohens_d_std)
                summary_data_by_arm[n_per_arm]['p_value'].append(p_val)
            
            followup_data.append({
                'follow_up_years': follow_up_years,
                'arm_size': n_per_arm,
                'cohens_d': cohens_d_mean,
                'cohens_d_std': cohens_d_std,
                'p_value': p_val
            })
        
        # Plot effect size with error bars
        valid_mask = [i for i, (cd, pv) in enumerate(zip(cohens_d_list, p_value_list)) 
                     if np.isfinite(cd) and np.isfinite(pv)]
        if len(valid_mask) > 0:
            valid_arm = [arm_sizes[i] for i in valid_mask]
            valid_cohens_d = [cohens_d_list[i] for i in valid_mask]
            valid_cohens_d_std = [cohens_d_std_list[i] for i in valid_mask]
            
            ax_effect.errorbar(valid_arm, valid_cohens_d, yerr=valid_cohens_d_std,
                             fmt='o-', color=colors_dur[param_idx],
                             linewidth=2.5, markersize=8, capsize=5, capthick=2,
                             label=f'{follow_up_years} years', alpha=0.8)
            ax_effect.set_xlabel('Arm Size (n)', fontsize=11, fontweight='bold')
            ax_effect.set_ylabel("Cohen's d", fontsize=11, fontweight='bold')
            ax_effect.set_title(f'Follow-up: {follow_up_years} years', fontsize=12, fontweight='bold')
            ax_effect.grid(True, alpha=0.3, linestyle='--')
            sns.despine(ax=ax_effect)
            ax_effect.set_xticks(arm_sizes)
        
        # Plot p-value with horizontal line at 0.05
        if len(valid_mask) > 0:
            valid_p = [p_value_list[i] for i in valid_mask]
            ax_pvalue.plot(valid_arm, valid_p, 's--', color=colors_dur[param_idx],
                          linewidth=2.5, markersize=8, label=f'{follow_up_years} years')
            ax_pvalue.axhline(y=0.05, color='red', linestyle='--', linewidth=2, 
                            alpha=0.7, label='p=0.05')
            ax_pvalue.set_xlabel('Arm Size (n)', fontsize=11, fontweight='bold')
            ax_pvalue.set_ylabel('P-value', fontsize=11, fontweight='bold')
            ax_pvalue.set_yscale('log')
            ax_pvalue.grid(True, alpha=0.3, linestyle='--')
            sns.despine(ax=ax_pvalue)
            ax_pvalue.set_xticks(arm_sizes)
    
    # Set common y-axis limits for all effect size subplots
    if len(all_cohens_d_values) > 0:
        y_min = np.nanmin(all_cohens_d_values)
        y_max = np.nanmax(all_cohens_d_values)
        y_range = y_max - y_min
        y_padding = y_range * 0.1  # 10% padding
        for param_idx in range(n_subplots):
            axes1[0, param_idx].set_ylim(y_min - y_padding, y_max + y_padding)
    
    # Set common y-axis limits for all p-value subplots (log scale)
    if len(all_p_value_values) > 0:
        p_min = np.nanmin(all_p_value_values)
        p_max = np.nanmax(all_p_value_values)
        # For log scale, use multiplicative padding
        p_min_lim = p_min / 1.5  # Extend downward
        p_max_lim = p_max * 1.5  # Extend upward
        # Ensure we include 0.05 if it's not already in range
        if 0.05 < p_min_lim:
            p_min_lim = 0.05 / 2
        for param_idx in range(n_subplots):
            axes1[1, param_idx].set_ylim(p_min_lim, p_max_lim)
    
    # Create summary plots
    ax_effect_summary = axes1[0, n_subplots]
    ax_pvalue_summary = axes1[1, n_subplots]
    
    # Summary plot: Effect size vs follow-up time
    colors_arm = plt.cm.Blues(np.linspace(0.4, 0.9, len(arm_sizes)))
    for arm_idx, n_per_arm in enumerate(arm_sizes):
        if len(summary_data_by_arm[n_per_arm]['param_values']) > 0:
            # Sort by parameter value
            sorted_indices = np.argsort(summary_data_by_arm[n_per_arm]['param_values'])
            param_vals = [summary_data_by_arm[n_per_arm]['param_values'][i] for i in sorted_indices]
            cohens_d_vals = [summary_data_by_arm[n_per_arm]['cohens_d'][i] for i in sorted_indices]
            cohens_d_std_vals = [summary_data_by_arm[n_per_arm]['cohens_d_std'][i] for i in sorted_indices]
            
            ax_effect_summary.errorbar(param_vals, cohens_d_vals, yerr=cohens_d_std_vals,
                                     fmt='o-', color=colors_arm[arm_idx],
                                     linewidth=2.5, markersize=8, capsize=5, capthick=2,
                                     label=f'n={n_per_arm}', alpha=0.8)
    
    ax_effect_summary.set_xlabel('Follow-up Time (years)', fontsize=11, fontweight='bold')
    ax_effect_summary.set_ylabel("Cohen's d", fontsize=11, fontweight='bold')
    ax_effect_summary.set_title('Summary: Effect Size', fontsize=12, fontweight='bold')
    ax_effect_summary.grid(True, alpha=0.3, linestyle='--')
    sns.despine(ax=ax_effect_summary)
    ax_effect_summary.legend(fontsize=9, loc='best', framealpha=0.9)
    if len(all_cohens_d_values) > 0:
        ax_effect_summary.set_ylim(y_min - y_padding, y_max + y_padding)
    
    # Summary plot: P-value vs follow-up time
    for arm_idx, n_per_arm in enumerate(arm_sizes):
        if len(summary_data_by_arm[n_per_arm]['param_values']) > 0:
            sorted_indices = np.argsort(summary_data_by_arm[n_per_arm]['param_values'])
            param_vals = [summary_data_by_arm[n_per_arm]['param_values'][i] for i in sorted_indices]
            p_vals = [summary_data_by_arm[n_per_arm]['p_value'][i] for i in sorted_indices]
            
            ax_pvalue_summary.plot(param_vals, p_vals, 's--', color=colors_arm[arm_idx],
                                  linewidth=2.5, markersize=8, label=f'n={n_per_arm}')
    
    ax_pvalue_summary.axhline(y=0.05, color='red', linestyle='--', linewidth=2, 
                            alpha=0.7, label='p=0.05')
    ax_pvalue_summary.set_xlabel('Follow-up Time (years)', fontsize=11, fontweight='bold')
    ax_pvalue_summary.set_ylabel('P-value', fontsize=11, fontweight='bold')
    ax_pvalue_summary.set_title('Summary: P-value', fontsize=12, fontweight='bold')
    ax_pvalue_summary.set_yscale('log')
    ax_pvalue_summary.grid(True, alpha=0.3, linestyle='--')
    sns.despine(ax=ax_pvalue_summary)
    ax_pvalue_summary.legend(fontsize=9, loc='best', framealpha=0.9)
    if len(all_p_value_values) > 0:
        ax_pvalue_summary.set_ylim(p_min_lim, p_max_lim)
    
    fig1.suptitle('Effect Size and P-value vs Arm Size: Follow-up Time', 
                  fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'effect_size_pvalue_vs_arm_size_followup.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved effect_size_pvalue_vs_arm_size_followup.png")
    
    if len(followup_data) > 0:
        df_followup = pd.DataFrame(followup_data)
        df_followup.to_csv(os.path.join(output_dir, 'effect_size_pvalue_vs_arm_size_followup_data.csv'), 
                          index=False)
        print("  Saved effect_size_pvalue_vs_arm_size_followup_data.csv")
    
    # 2. Treatment Frequency Figure
    print("\n2. Calculating Effect Size and P-value vs Arm Size for Treatment Frequency...")
    n_subplots = len(frequency_values)
    fig2, axes2 = plt.subplots(2, n_subplots + 1, figsize=(5*(n_subplots + 1), 10))
    if n_subplots == 1:
        axes2 = axes2.reshape(2, 2)
    
    colors_freq = plt.cm.plasma(np.linspace(0.2, 0.8, n_subplots))
    frequency_data = []
    
    # Collect all effect size and p-value values to set common y-axis limits
    all_cohens_d_values = []
    all_p_value_values = []
    
    # Collect data for summary plots: organized by arm size
    summary_data_by_arm = {n: {'param_values': [], 'cohens_d': [], 'cohens_d_std': [], 'p_value': []} 
                           for n in arm_sizes}
    
    for param_idx, interval_days in enumerate(frequency_values):
        ax_effect = axes2[0, param_idx]
        ax_pvalue = axes2[1, param_idx]
        
        cohens_d_list = []
        cohens_d_std_list = []
        p_value_list = []
        
        for n_per_arm in arm_sizes:
            print(f"  Frequency={interval_days} days, n={n_per_arm}...")
            cohens_d_mean, cohens_d_std, p_val = calculate_effect_size_and_pvalue_with_std(
                APOE33_params, n_per_arm, base_mci_threshold, base_s_fold_change,
                interval_days, base_treatment_duration_days, base_follow_up_years,
                dt, base_seed + int(interval_days) + n_per_arm, sample_every_days
            )
            cohens_d_list.append(cohens_d_mean)
            cohens_d_std_list.append(cohens_d_std)
            p_value_list.append(p_val)
            
            if np.isfinite(cohens_d_mean):
                all_cohens_d_values.append(cohens_d_mean)
            if np.isfinite(p_val) and p_val > 0:
                all_p_value_values.append(p_val)
            
            # Store for summary plots
            if np.isfinite(cohens_d_mean) and np.isfinite(p_val):
                summary_data_by_arm[n_per_arm]['param_values'].append(interval_days)
                summary_data_by_arm[n_per_arm]['cohens_d'].append(cohens_d_mean)
                summary_data_by_arm[n_per_arm]['cohens_d_std'].append(cohens_d_std)
                summary_data_by_arm[n_per_arm]['p_value'].append(p_val)
            
            frequency_data.append({
                'treatment_frequency_days': interval_days,
                'arm_size': n_per_arm,
                'cohens_d': cohens_d_mean,
                'cohens_d_std': cohens_d_std,
                'p_value': p_val
            })
        
        # Plot effect size with error bars
        valid_mask = [i for i, (cd, pv) in enumerate(zip(cohens_d_list, p_value_list)) 
                     if np.isfinite(cd) and np.isfinite(pv)]
        if len(valid_mask) > 0:
            valid_arm = [arm_sizes[i] for i in valid_mask]
            valid_cohens_d = [cohens_d_list[i] for i in valid_mask]
            valid_cohens_d_std = [cohens_d_std_list[i] for i in valid_mask]
            
            ax_effect.errorbar(valid_arm, valid_cohens_d, yerr=valid_cohens_d_std,
                             fmt='o-', color=colors_freq[param_idx],
                             linewidth=2.5, markersize=8, capsize=5, capthick=2,
                             label=f'{interval_days} days', alpha=0.8)
            ax_effect.set_xlabel('Arm Size (n)', fontsize=11, fontweight='bold')
            ax_effect.set_ylabel("Cohen's d", fontsize=11, fontweight='bold')
            ax_effect.set_title(f'Frequency: {interval_days} days', fontsize=12, fontweight='bold')
            ax_effect.grid(True, alpha=0.3, linestyle='--')
            sns.despine(ax=ax_effect)
            ax_effect.set_xticks(arm_sizes)
        
        # Plot p-value with horizontal line at 0.05
        if len(valid_mask) > 0:
            valid_p = [p_value_list[i] for i in valid_mask]
            ax_pvalue.plot(valid_arm, valid_p, 's--', color=colors_freq[param_idx],
                          linewidth=2.5, markersize=8, label=f'{interval_days} days')
            ax_pvalue.axhline(y=0.05, color='red', linestyle='--', linewidth=2, 
                            alpha=0.7, label='p=0.05')
            ax_pvalue.set_xlabel('Arm Size (n)', fontsize=11, fontweight='bold')
            ax_pvalue.set_ylabel('P-value', fontsize=11, fontweight='bold')
            ax_pvalue.set_yscale('log')
            ax_pvalue.grid(True, alpha=0.3, linestyle='--')
            sns.despine(ax=ax_pvalue)
            ax_pvalue.set_xticks(arm_sizes)
    
    # Set common y-axis limits for all effect size subplots
    if len(all_cohens_d_values) > 0:
        y_min = np.nanmin(all_cohens_d_values)
        y_max = np.nanmax(all_cohens_d_values)
        y_range = y_max - y_min
        y_padding = y_range * 0.1  # 10% padding
        for param_idx in range(n_subplots):
            axes2[0, param_idx].set_ylim(y_min - y_padding, y_max + y_padding)
    
    # Set common y-axis limits for all p-value subplots (log scale)
    if len(all_p_value_values) > 0:
        p_min = np.nanmin(all_p_value_values)
        p_max = np.nanmax(all_p_value_values)
        # For log scale, use multiplicative padding
        p_min_lim = p_min / 1.5  # Extend downward
        p_max_lim = p_max * 1.5  # Extend upward
        # Ensure we include 0.05 if it's not already in range
        if 0.05 < p_min_lim:
            p_min_lim = 0.05 / 2
        for param_idx in range(n_subplots):
            axes2[1, param_idx].set_ylim(p_min_lim, p_max_lim)
    
    # Create summary plots
    ax_effect_summary = axes2[0, n_subplots]
    ax_pvalue_summary = axes2[1, n_subplots]
    
    # Summary plot: Effect size vs treatment frequency
    colors_arm = plt.cm.Blues(np.linspace(0.4, 0.9, len(arm_sizes)))
    for arm_idx, n_per_arm in enumerate(arm_sizes):
        if len(summary_data_by_arm[n_per_arm]['param_values']) > 0:
            sorted_indices = np.argsort(summary_data_by_arm[n_per_arm]['param_values'])
            param_vals = [summary_data_by_arm[n_per_arm]['param_values'][i] for i in sorted_indices]
            cohens_d_vals = [summary_data_by_arm[n_per_arm]['cohens_d'][i] for i in sorted_indices]
            cohens_d_std_vals = [summary_data_by_arm[n_per_arm]['cohens_d_std'][i] for i in sorted_indices]
            
            ax_effect_summary.errorbar(param_vals, cohens_d_vals, yerr=cohens_d_std_vals,
                                     fmt='o-', color=colors_arm[arm_idx],
                                     linewidth=2.5, markersize=8, capsize=5, capthick=2,
                                     label=f'n={n_per_arm}', alpha=0.8)
    
    ax_effect_summary.set_xlabel('Treatment Frequency (days)', fontsize=11, fontweight='bold')
    ax_effect_summary.set_ylabel("Cohen's d", fontsize=11, fontweight='bold')
    ax_effect_summary.set_title('Summary: Effect Size', fontsize=12, fontweight='bold')
    ax_effect_summary.grid(True, alpha=0.3, linestyle='--')
    sns.despine(ax=ax_effect_summary)
    ax_effect_summary.legend(fontsize=9, loc='best', framealpha=0.9)
    if len(all_cohens_d_values) > 0:
        ax_effect_summary.set_ylim(y_min - y_padding, y_max + y_padding)
    
    # Summary plot: P-value vs treatment frequency
    for arm_idx, n_per_arm in enumerate(arm_sizes):
        if len(summary_data_by_arm[n_per_arm]['param_values']) > 0:
            sorted_indices = np.argsort(summary_data_by_arm[n_per_arm]['param_values'])
            param_vals = [summary_data_by_arm[n_per_arm]['param_values'][i] for i in sorted_indices]
            p_vals = [summary_data_by_arm[n_per_arm]['p_value'][i] for i in sorted_indices]
            
            ax_pvalue_summary.plot(param_vals, p_vals, 's--', color=colors_arm[arm_idx],
                                  linewidth=2.5, markersize=8, label=f'n={n_per_arm}')
    
    ax_pvalue_summary.axhline(y=0.05, color='red', linestyle='--', linewidth=2, 
                            alpha=0.7, label='p=0.05')
    ax_pvalue_summary.set_xlabel('Treatment Frequency (days)', fontsize=11, fontweight='bold')
    ax_pvalue_summary.set_ylabel('P-value', fontsize=11, fontweight='bold')
    ax_pvalue_summary.set_title('Summary: P-value', fontsize=12, fontweight='bold')
    ax_pvalue_summary.set_yscale('log')
    ax_pvalue_summary.grid(True, alpha=0.3, linestyle='--')
    sns.despine(ax=ax_pvalue_summary)
    ax_pvalue_summary.legend(fontsize=9, loc='best', framealpha=0.9)
    if len(all_p_value_values) > 0:
        ax_pvalue_summary.set_ylim(p_min_lim, p_max_lim)
    
    fig2.suptitle('Effect Size and P-value vs Arm Size: Treatment Frequency', 
                  fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'effect_size_pvalue_vs_arm_size_frequency.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved effect_size_pvalue_vs_arm_size_frequency.png")
    
    if len(frequency_data) > 0:
        df_freq = pd.DataFrame(frequency_data)
        df_freq.to_csv(os.path.join(output_dir, 'effect_size_pvalue_vs_arm_size_frequency_data.csv'), 
                      index=False)
        print("  Saved effect_size_pvalue_vs_arm_size_frequency_data.csv")
    
    # 3. Dose Figure
    print("\n3. Calculating Effect Size and P-value vs Arm Size for Dose...")
    n_subplots = len(dose_values)
    fig3, axes3 = plt.subplots(2, n_subplots + 1, figsize=(5*(n_subplots + 1), 10))
    if n_subplots == 1:
        axes3 = axes3.reshape(2, 2)
    
    colors_dose = plt.cm.coolwarm(np.linspace(0.2, 0.8, n_subplots))
    dose_data = []
    
    # Collect all effect size and p-value values to set common y-axis limits
    all_cohens_d_values = []
    all_p_value_values = []
    
    # Collect data for summary plots: organized by arm size
    summary_data_by_arm = {n: {'param_values': [], 'cohens_d': [], 'cohens_d_std': [], 'p_value': []} 
                           for n in arm_sizes}
    
    for param_idx, s_fold_change in enumerate(dose_values):
        ax_effect = axes3[0, param_idx]
        ax_pvalue = axes3[1, param_idx]
        
        cohens_d_list = []
        cohens_d_std_list = []
        p_value_list = []
        
        for n_per_arm in arm_sizes:
            print(f"  Dose={s_fold_change}, n={n_per_arm}...")
            cohens_d_mean, cohens_d_std, p_val = calculate_effect_size_and_pvalue_with_std(
                APOE33_params, n_per_arm, base_mci_threshold, s_fold_change,
                base_interval_days, base_treatment_duration_days, base_follow_up_years,
                dt, base_seed + int(s_fold_change*1000) + n_per_arm, sample_every_days
            )
            cohens_d_list.append(cohens_d_mean)
            cohens_d_std_list.append(cohens_d_std)
            p_value_list.append(p_val)
            
            if np.isfinite(cohens_d_mean):
                all_cohens_d_values.append(cohens_d_mean)
            if np.isfinite(p_val) and p_val > 0:
                all_p_value_values.append(p_val)
            
            # Store for summary plots
            if np.isfinite(cohens_d_mean) and np.isfinite(p_val):
                summary_data_by_arm[n_per_arm]['param_values'].append(s_fold_change)
                summary_data_by_arm[n_per_arm]['cohens_d'].append(cohens_d_mean)
                summary_data_by_arm[n_per_arm]['cohens_d_std'].append(cohens_d_std)
                summary_data_by_arm[n_per_arm]['p_value'].append(p_val)
            
            dose_data.append({
                'dose_s_fold_change': s_fold_change,
                'arm_size': n_per_arm,
                'cohens_d': cohens_d_mean,
                'cohens_d_std': cohens_d_std,
                'p_value': p_val
            })
        
        # Plot effect size with error bars
        valid_mask = [i for i, (cd, pv) in enumerate(zip(cohens_d_list, p_value_list)) 
                     if np.isfinite(cd) and np.isfinite(pv)]
        if len(valid_mask) > 0:
            valid_arm = [arm_sizes[i] for i in valid_mask]
            valid_cohens_d = [cohens_d_list[i] for i in valid_mask]
            valid_cohens_d_std = [cohens_d_std_list[i] for i in valid_mask]
            
            ax_effect.errorbar(valid_arm, valid_cohens_d, yerr=valid_cohens_d_std,
                             fmt='o-', color=colors_dose[param_idx],
                             linewidth=2.5, markersize=8, capsize=5, capthick=2,
                             label=f'{s_fold_change}', alpha=0.8)
            ax_effect.set_xlabel('Arm Size (n)', fontsize=11, fontweight='bold')
            ax_effect.set_ylabel("Cohen's d", fontsize=11, fontweight='bold')
            ax_effect.set_title(f'Dose: {s_fold_change}', fontsize=12, fontweight='bold')
            ax_effect.grid(True, alpha=0.3, linestyle='--')
            sns.despine(ax=ax_effect)
            ax_effect.set_xticks(arm_sizes)
        
        # Plot p-value with horizontal line at 0.05
        if len(valid_mask) > 0:
            valid_p = [p_value_list[i] for i in valid_mask]
            ax_pvalue.plot(valid_arm, valid_p, 's--', color=colors_dose[param_idx],
                          linewidth=2.5, markersize=8, label=f'{s_fold_change}')
            ax_pvalue.axhline(y=0.05, color='red', linestyle='--', linewidth=2, 
                            alpha=0.7, label='p=0.05')
            ax_pvalue.set_xlabel('Arm Size (n)', fontsize=11, fontweight='bold')
            ax_pvalue.set_ylabel('P-value', fontsize=11, fontweight='bold')
            ax_pvalue.set_yscale('log')
            ax_pvalue.grid(True, alpha=0.3, linestyle='--')
            sns.despine(ax=ax_pvalue)
            ax_pvalue.set_xticks(arm_sizes)
    
    # Set common y-axis limits for all effect size subplots
    if len(all_cohens_d_values) > 0:
        y_min = np.nanmin(all_cohens_d_values)
        y_max = np.nanmax(all_cohens_d_values)
        y_range = y_max - y_min
        y_padding = y_range * 0.1  # 10% padding
        for param_idx in range(n_subplots):
            axes3[0, param_idx].set_ylim(y_min - y_padding, y_max + y_padding)
    
    # Set common y-axis limits for all p-value subplots (log scale)
    if len(all_p_value_values) > 0:
        p_min = np.nanmin(all_p_value_values)
        p_max = np.nanmax(all_p_value_values)
        # For log scale, use multiplicative padding
        p_min_lim = p_min / 1.5  # Extend downward
        p_max_lim = p_max * 1.5  # Extend upward
        # Ensure we include 0.05 if it's not already in range
        if 0.05 < p_min_lim:
            p_min_lim = 0.05 / 2
        for param_idx in range(n_subplots):
            axes3[1, param_idx].set_ylim(p_min_lim, p_max_lim)
    
    # Create summary plots
    ax_effect_summary = axes3[0, n_subplots]
    ax_pvalue_summary = axes3[1, n_subplots]
    
    # Summary plot: Effect size vs dose
    colors_arm = plt.cm.Blues(np.linspace(0.4, 0.9, len(arm_sizes)))
    for arm_idx, n_per_arm in enumerate(arm_sizes):
        if len(summary_data_by_arm[n_per_arm]['param_values']) > 0:
            sorted_indices = np.argsort(summary_data_by_arm[n_per_arm]['param_values'])
            param_vals = [summary_data_by_arm[n_per_arm]['param_values'][i] for i in sorted_indices]
            cohens_d_vals = [summary_data_by_arm[n_per_arm]['cohens_d'][i] for i in sorted_indices]
            cohens_d_std_vals = [summary_data_by_arm[n_per_arm]['cohens_d_std'][i] for i in sorted_indices]
            
            ax_effect_summary.errorbar(param_vals, cohens_d_vals, yerr=cohens_d_std_vals,
                                     fmt='o-', color=colors_arm[arm_idx],
                                     linewidth=2.5, markersize=8, capsize=5, capthick=2,
                                     label=f'n={n_per_arm}', alpha=0.8)
    
    ax_effect_summary.set_xlabel('Dose (s_fold_change)', fontsize=11, fontweight='bold')
    ax_effect_summary.set_ylabel("Cohen's d", fontsize=11, fontweight='bold')
    ax_effect_summary.set_title('Summary: Effect Size', fontsize=12, fontweight='bold')
    ax_effect_summary.grid(True, alpha=0.3, linestyle='--')
    sns.despine(ax=ax_effect_summary)
    ax_effect_summary.legend(fontsize=9, loc='best', framealpha=0.9)
    if len(all_cohens_d_values) > 0:
        ax_effect_summary.set_ylim(y_min - y_padding, y_max + y_padding)
    
    # Summary plot: P-value vs dose
    for arm_idx, n_per_arm in enumerate(arm_sizes):
        if len(summary_data_by_arm[n_per_arm]['param_values']) > 0:
            sorted_indices = np.argsort(summary_data_by_arm[n_per_arm]['param_values'])
            param_vals = [summary_data_by_arm[n_per_arm]['param_values'][i] for i in sorted_indices]
            p_vals = [summary_data_by_arm[n_per_arm]['p_value'][i] for i in sorted_indices]
            
            ax_pvalue_summary.plot(param_vals, p_vals, 's--', color=colors_arm[arm_idx],
                                  linewidth=2.5, markersize=8, label=f'n={n_per_arm}')
    
    ax_pvalue_summary.axhline(y=0.05, color='red', linestyle='--', linewidth=2, 
                            alpha=0.7, label='p=0.05')
    ax_pvalue_summary.set_xlabel('Dose (s_fold_change)', fontsize=11, fontweight='bold')
    ax_pvalue_summary.set_ylabel('P-value', fontsize=11, fontweight='bold')
    ax_pvalue_summary.set_title('Summary: P-value', fontsize=12, fontweight='bold')
    ax_pvalue_summary.set_yscale('log')
    ax_pvalue_summary.grid(True, alpha=0.3, linestyle='--')
    sns.despine(ax=ax_pvalue_summary)
    ax_pvalue_summary.legend(fontsize=9, loc='best', framealpha=0.9)
    if len(all_p_value_values) > 0:
        ax_pvalue_summary.set_ylim(p_min_lim, p_max_lim)
    
    fig3.suptitle('Effect Size and P-value vs Arm Size: Dose', 
                  fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'effect_size_pvalue_vs_arm_size_dose.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved effect_size_pvalue_vs_arm_size_dose.png")
    
    if len(dose_data) > 0:
        df_dose = pd.DataFrame(dose_data)
        df_dose.to_csv(os.path.join(output_dir, 'effect_size_pvalue_vs_arm_size_dose_data.csv'), 
                      index=False)
        print("  Saved effect_size_pvalue_vs_arm_size_dose_data.csv")
    
    # 4. MCI Threshold Figure
    print("\n4. Calculating Effect Size and P-value vs Arm Size for MCI Threshold...")
    n_subplots = len(mci_threshold_values)
    fig4, axes4 = plt.subplots(2, n_subplots + 1, figsize=(5*(n_subplots + 1), 10))
    if n_subplots == 1:
        axes4 = axes4.reshape(2, 2)
    
    colors_mci = plt.cm.inferno(np.linspace(0.2, 0.8, n_subplots))
    mci_data = []
    
    # Collect all effect size and p-value values to set common y-axis limits
    all_cohens_d_values = []
    all_p_value_values = []
    
    # Collect data for summary plots: organized by arm size
    summary_data_by_arm = {n: {'param_values': [], 'cohens_d': [], 'cohens_d_std': [], 'p_value': []} 
                           for n in arm_sizes}
    
    for param_idx, mci_threshold in enumerate(mci_threshold_values):
        ax_effect = axes4[0, param_idx]
        ax_pvalue = axes4[1, param_idx]
        
        cohens_d_list = []
        cohens_d_std_list = []
        p_value_list = []
        
        for n_per_arm in arm_sizes:
            print(f"  MCI threshold={mci_threshold}, n={n_per_arm}...")
            cohens_d_mean, cohens_d_std, p_val = calculate_effect_size_and_pvalue_with_std(
                APOE33_params, n_per_arm, mci_threshold, base_s_fold_change,
                base_interval_days, base_treatment_duration_days, base_follow_up_years,
                dt, base_seed + int(mci_threshold*100) + n_per_arm, sample_every_days
            )
            cohens_d_list.append(cohens_d_mean)
            cohens_d_std_list.append(cohens_d_std)
            p_value_list.append(p_val)
            
            if np.isfinite(cohens_d_mean):
                all_cohens_d_values.append(cohens_d_mean)
            if np.isfinite(p_val) and p_val > 0:
                all_p_value_values.append(p_val)
            
            # Store for summary plots
            if np.isfinite(cohens_d_mean) and np.isfinite(p_val):
                summary_data_by_arm[n_per_arm]['param_values'].append(mci_threshold)
                summary_data_by_arm[n_per_arm]['cohens_d'].append(cohens_d_mean)
                summary_data_by_arm[n_per_arm]['cohens_d_std'].append(cohens_d_std)
                summary_data_by_arm[n_per_arm]['p_value'].append(p_val)
            
            mci_data.append({
                'mci_threshold': mci_threshold,
                'arm_size': n_per_arm,
                'cohens_d': cohens_d_mean,
                'cohens_d_std': cohens_d_std,
                'p_value': p_val
            })
        
        # Plot effect size with error bars
        valid_mask = [i for i, (cd, pv) in enumerate(zip(cohens_d_list, p_value_list)) 
                     if np.isfinite(cd) and np.isfinite(pv)]
        if len(valid_mask) > 0:
            valid_arm = [arm_sizes[i] for i in valid_mask]
            valid_cohens_d = [cohens_d_list[i] for i in valid_mask]
            valid_cohens_d_std = [cohens_d_std_list[i] for i in valid_mask]
            
            ax_effect.errorbar(valid_arm, valid_cohens_d, yerr=valid_cohens_d_std,
                             fmt='o-', color=colors_mci[param_idx],
                             linewidth=2.5, markersize=8, capsize=5, capthick=2,
                             label=f'{mci_threshold}', alpha=0.8)
            ax_effect.set_xlabel('Arm Size (n)', fontsize=11, fontweight='bold')
            ax_effect.set_ylabel("Cohen's d", fontsize=11, fontweight='bold')
            ax_effect.set_title(f'MCI Threshold: {mci_threshold}', fontsize=12, fontweight='bold')
            ax_effect.grid(True, alpha=0.3, linestyle='--')
            sns.despine(ax=ax_effect)
            ax_effect.set_xticks(arm_sizes)
        
        # Plot p-value with horizontal line at 0.05
        if len(valid_mask) > 0:
            valid_p = [p_value_list[i] for i in valid_mask]
            ax_pvalue.plot(valid_arm, valid_p, 's--', color=colors_mci[param_idx],
                          linewidth=2.5, markersize=8, label=f'{mci_threshold}')
            ax_pvalue.axhline(y=0.05, color='red', linestyle='--', linewidth=2, 
                            alpha=0.7, label='p=0.05')
            ax_pvalue.set_xlabel('Arm Size (n)', fontsize=11, fontweight='bold')
            ax_pvalue.set_ylabel('P-value', fontsize=11, fontweight='bold')
            ax_pvalue.set_yscale('log')
            ax_pvalue.grid(True, alpha=0.3, linestyle='--')
            sns.despine(ax=ax_pvalue)
            ax_pvalue.set_xticks(arm_sizes)
    
    # Set common y-axis limits for all effect size subplots
    if len(all_cohens_d_values) > 0:
        y_min = np.nanmin(all_cohens_d_values)
        y_max = np.nanmax(all_cohens_d_values)
        y_range = y_max - y_min
        y_padding = y_range * 0.1  # 10% padding
        for param_idx in range(n_subplots):
            axes4[0, param_idx].set_ylim(y_min - y_padding, y_max + y_padding)
    
    # Set common y-axis limits for all p-value subplots (log scale)
    if len(all_p_value_values) > 0:
        p_min = np.nanmin(all_p_value_values)
        p_max = np.nanmax(all_p_value_values)
        # For log scale, use multiplicative padding
        p_min_lim = p_min / 1.5  # Extend downward
        p_max_lim = p_max * 1.5  # Extend upward
        # Ensure we include 0.05 if it's not already in range
        if 0.05 < p_min_lim:
            p_min_lim = 0.05 / 2
        for param_idx in range(n_subplots):
            axes4[1, param_idx].set_ylim(p_min_lim, p_max_lim)
    
    # Create summary plots
    ax_effect_summary = axes4[0, n_subplots]
    ax_pvalue_summary = axes4[1, n_subplots]
    
    # Summary plot: Effect size vs MCI threshold
    colors_arm = plt.cm.Blues(np.linspace(0.4, 0.9, len(arm_sizes)))
    for arm_idx, n_per_arm in enumerate(arm_sizes):
        if len(summary_data_by_arm[n_per_arm]['param_values']) > 0:
            sorted_indices = np.argsort(summary_data_by_arm[n_per_arm]['param_values'])
            param_vals = [summary_data_by_arm[n_per_arm]['param_values'][i] for i in sorted_indices]
            cohens_d_vals = [summary_data_by_arm[n_per_arm]['cohens_d'][i] for i in sorted_indices]
            cohens_d_std_vals = [summary_data_by_arm[n_per_arm]['cohens_d_std'][i] for i in sorted_indices]
            
            ax_effect_summary.errorbar(param_vals, cohens_d_vals, yerr=cohens_d_std_vals,
                                     fmt='o-', color=colors_arm[arm_idx],
                                     linewidth=2.5, markersize=8, capsize=5, capthick=2,
                                     label=f'n={n_per_arm}', alpha=0.8)
    
    ax_effect_summary.set_xlabel('MCI Threshold', fontsize=11, fontweight='bold')
    ax_effect_summary.set_ylabel("Cohen's d", fontsize=11, fontweight='bold')
    ax_effect_summary.set_title('Summary: Effect Size', fontsize=12, fontweight='bold')
    ax_effect_summary.grid(True, alpha=0.3, linestyle='--')
    sns.despine(ax=ax_effect_summary)
    ax_effect_summary.legend(fontsize=9, loc='best', framealpha=0.9)
    if len(all_cohens_d_values) > 0:
        ax_effect_summary.set_ylim(y_min - y_padding, y_max + y_padding)
    
    # Summary plot: P-value vs MCI threshold
    for arm_idx, n_per_arm in enumerate(arm_sizes):
        if len(summary_data_by_arm[n_per_arm]['param_values']) > 0:
            sorted_indices = np.argsort(summary_data_by_arm[n_per_arm]['param_values'])
            param_vals = [summary_data_by_arm[n_per_arm]['param_values'][i] for i in sorted_indices]
            p_vals = [summary_data_by_arm[n_per_arm]['p_value'][i] for i in sorted_indices]
            
            ax_pvalue_summary.plot(param_vals, p_vals, 's--', color=colors_arm[arm_idx],
                                  linewidth=2.5, markersize=8, label=f'n={n_per_arm}')
    
    ax_pvalue_summary.axhline(y=0.05, color='red', linestyle='--', linewidth=2, 
                            alpha=0.7, label='p=0.05')
    ax_pvalue_summary.set_xlabel('MCI Threshold', fontsize=11, fontweight='bold')
    ax_pvalue_summary.set_ylabel('P-value', fontsize=11, fontweight='bold')
    ax_pvalue_summary.set_title('Summary: P-value', fontsize=12, fontweight='bold')
    ax_pvalue_summary.set_yscale('log')
    ax_pvalue_summary.grid(True, alpha=0.3, linestyle='--')
    sns.despine(ax=ax_pvalue_summary)
    ax_pvalue_summary.legend(fontsize=9, loc='best', framealpha=0.9)
    if len(all_p_value_values) > 0:
        ax_pvalue_summary.set_ylim(p_min_lim, p_max_lim)
    
    fig4.suptitle('Effect Size and P-value vs Arm Size: MCI Threshold', 
                  fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'effect_size_pvalue_vs_arm_size_mci_threshold.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved effect_size_pvalue_vs_arm_size_mci_threshold.png")
    
    if len(mci_data) > 0:
        df_mci = pd.DataFrame(mci_data)
        df_mci.to_csv(os.path.join(output_dir, 'effect_size_pvalue_vs_arm_size_mci_threshold_data.csv'), 
                     index=False)
        print("  Saved effect_size_pvalue_vs_arm_size_mci_threshold_data.csv")
    
    print("\n" + "="*80)
    print("Effect Size and P-value vs Arm Size Analysis Complete!")
    print("="*80)


def main():
    """Main function to run the mixed population trial simulation."""
    # Base parameters (APOE33) - matching clinical_trial_simulation_apoe33_apoe44.py
    APOE33_params = {
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
    
    # Simulation parameters
    n_per_arm = 200
    mci_threshold = 5.2
    s_fold_change = 0.3  # 70% reduction in s
    interval_days = 90.0  # 3 months
    treatment_duration_days = 1.5 * 365.0  # 1.5 years
    follow_up_years = 1.5
    dt = 1.0
    base_seed = 2025
    sample_every_days = 30.0
    
    # Create output directory
    output_dir = 'mixed_population_trial_output'
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*80)
    print("Mixed Population Clinical Trial Simulation")
    print("="*80)
    print(f"Population distribution:")
    print(f"  - 20% APOE44")
    print(f"  - 30% APOE34")
    print(f"  - 10% APOE24")
    print(f"  - 30% APOE33")
    print(f"  - 10% APOE23")
    print(f"Each arm: {n_per_arm} trajectories")
    print(f"Treatment: s elimination every {interval_days/30:.1f} months for {treatment_duration_days/365:.1f} years")
    print("="*80)
    
    # Run simulation
    control_data, treatment_data = simulate_mixed_population_trial(
        APOE33_params, n_per_arm, mci_threshold, s_fold_change,
        interval_days, treatment_duration_days, follow_up_years,
        dt, base_seed, sample_every_days
    )
    
    # Print genotype distribution
    print("\nControl arm genotype distribution:")
    from collections import Counter
    control_genotype_counts = Counter(control_data['genotypes'])
    for genotype, count in sorted(control_genotype_counts.items()):
        print(f"  {genotype}: {count} ({100*count/n_per_arm:.1f}%)")
    
    print("\nTreatment arm genotype distribution:")
    treatment_genotype_counts = Counter(treatment_data['genotypes'])
    for genotype, count in sorted(treatment_genotype_counts.items()):
        print(f"  {genotype}: {count} ({100*count/n_per_arm:.1f}%)")
    
    # Calculate statistics on neuroinflammation values at follow-up time
    print("\nCalculating statistics...")
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
        print(f"  Control n at follow-up: {stats_results['mean_control']:.3f} ± {stats_results['std_control']:.3f}")
        print(f"  Treatment n at follow-up: {stats_results['mean_treatment']:.3f} ± {stats_results['std_treatment']:.3f}")
        print(f"  Mean difference: {stats_results['mean_difference']:.3f}")
        if np.isfinite(stats_results.get('p_value_ttest', np.nan)):
            print(f"  p-value (t-test): {stats_results['p_value_ttest']:.4f}")
        if np.isfinite(stats_results.get('cohens_d', np.nan)):
            print(f"  Cohen's d: {stats_results['cohens_d']:.3f}")
        if np.isfinite(stats_results.get('delay', np.nan)):
            print(f"  Delay: {stats_results['delay']:.2f} years")
    
    # Plot survival curves (mixed population)
    print("\nPlotting survival curves (mixed population)...")
    plot_survival_curves(control_data, treatment_data, output_dir, stats_results, n_per_arm)
    
    # Plot neuroinflammation trajectories (mixed population)
    print("Plotting neuroinflammation trajectories (mixed population)...")
    plot_neuroinflammation_trajectories(control_data, treatment_data, output_dir, follow_up_years, mci_threshold, stats_results, n_per_arm)
    
    # Plot neuroinflammation difference (mixed population)
    print("Plotting neuroinflammation difference (mixed population)...")
    plot_neuroinflammation_difference(control_data, treatment_data, output_dir, n_per_arm)
    
    # Plot senescence difference (mixed population)
    print("Plotting senescence difference (mixed population)...")
    plot_senescence_difference(control_data, treatment_data, output_dir, n_per_arm)
    
    # Plot neuroinflammation equation terms (mixed population)
    print("Plotting neuroinflammation equation terms (mixed population)...")
    plot_neuroinflammation_equation_terms(control_data, treatment_data, APOE33_params, output_dir, n_per_arm)
    
    # Plot genotype-specific figures
    print("\nPlotting genotype-specific figures...")
    genotypes = ['APOE44', 'APOE34', 'APOE24', 'APOE33', 'APOE23']
    for genotype in genotypes:
        plot_genotype_specific_plots(control_data, treatment_data, output_dir, genotype,
                                    follow_up_years, mci_threshold, n_per_arm)
    
    print("\n" + "="*80)
    print("Simulation complete!")
    print(f"Results saved to: {output_dir}/")
    print("="*80)
    
    # Run SE vs sample size analysis
    print("\n" + "="*80)
    print("Running SE vs Sample Size Analysis...")
    print("="*80)
    plot_se_vs_sample_size(
        APOE33_params, 
        sample_sizes=[10, 50, 100, 150, 200],
        mci_threshold=mci_threshold,
        s_fold_change=s_fold_change,
        interval_days=interval_days,
        treatment_duration_days=treatment_duration_days,
        follow_up_years=follow_up_years,
        dt=dt,
        base_seed=base_seed,
        sample_every_days=sample_every_days,
        output_dir=output_dir
    )
    
    # Plot delta n vs parameters
    print("\n" + "="*80)
    print("Running Delta n vs Parameters Analysis...")
    print("="*80)
    plot_delta_n_vs_parameters(
        APOE33_params,
        arm_sizes=[10, 50, 100],
        base_mci_threshold=mci_threshold,
        base_s_fold_change=s_fold_change,
        base_interval_days=interval_days,
        base_treatment_duration_days=treatment_duration_days,
        base_follow_up_years=follow_up_years,
        dt=dt,
        base_seed=base_seed,
        sample_every_days=sample_every_days,
        output_dir=output_dir
    )
    
    # Plot effect size and p-value vs arm size
    print("\n" + "="*80)
    print("Running Effect Size and P-value vs Arm Size Analysis...")
    print("="*80)
    plot_effect_size_and_pvalue_vs_arm_size(
        APOE33_params,
        base_mci_threshold=mci_threshold,
        base_s_fold_change=s_fold_change,
        base_interval_days=interval_days,
        base_treatment_duration_days=treatment_duration_days,
        base_follow_up_years=follow_up_years,
        dt=dt,
        base_seed=base_seed,
        sample_every_days=sample_every_days,
        output_dir=output_dir
    )

if __name__ == "__main__":
    main()
