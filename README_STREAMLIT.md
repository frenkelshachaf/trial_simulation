# Interactive Perturbation Platform - Streamlit App

## Overview

This Streamlit application provides an interactive web interface to explore the effects of various treatment parameters on neuroinflammation trajectories, senescence trajectories, survival curves, and statistical measures.

## Installation

1. Install required dependencies:
```bash
pip install -r requirements_streamlit.txt
```

Or install Streamlit separately:
```bash
pip install streamlit
```

## Running Locally

Navigate to the `simulations` directory and run:

```bash
streamlit run interactive_perturbation_platform.py
```

The app will open in your default web browser at `http://localhost:8501`

## Deployment

### Streamlit Cloud

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Select `interactive_perturbation_platform.py` as the main file
5. Deploy!

### Other Platforms

The app can also be deployed on:
- Heroku
- AWS EC2
- Google Cloud Run
- Any platform that supports Python web applications

## Usage

1. **Adjust Parameters**: Use the sidebar sliders to modify:
   - Treatment parameters (s reduction factor, frequency, duration)
   - Trial parameters (arm size, MCI threshold, follow-up time)
   - Plotting parameters (age range)

2. **Run Simulation**: Click the "Run Simulation" button

3. **View Results**: 
   - Statistics summary shows key metrics
   - Four interactive plots show trajectories and differences

## Parameters Explained

- **s_fold_change**: Factor multiplying senescence (s) during treatment (0.3 = 70% reduction)
- **Treatment frequency**: Interval between treatments in days
- **Treatment duration**: Total length of treatment period in years
- **Arm size**: Number of simulated trajectories per arm
- **MCI threshold**: Neuroinflammation level that triggers treatment
- **Follow-up time**: Time after treatment start to measure outcomes
- **Age range**: X-axis limits for trajectory plots

## Notes

- Simulations are cached for performance
- Larger arm sizes will take longer to compute
- Results update automatically when parameters change
