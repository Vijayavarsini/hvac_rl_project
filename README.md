# Smart HVAC Reinforcement Learning System

A Reinforcement Learning (RL) based solution for optimizing Heating, Ventilation, and Air Conditioning (HVAC) control using Proximal Policy Optimization (PPO). This system balances indoor thermal comfort and energy efficiency by learning from weather data and simulated user feedback.

## Features

- **Custom RL Environment**: Implements a `gymnasium` environment specifically for HVAC dynamics.
- **Personalized Comfort**: Allows users to set preferred temperature ranges at runtime.
- **Adaptive Feedback**: Simulates occupant feedback and automatically adjusts the target comfort range.
- **Real-world Data**: Integrates with [OpenWeatherMap API](https://openweathermap.org/api) for external temperature forecasts.
- **Comparative Analysis**: benchmarks RL performance against a baseline "full power" mode.
- **Visualization**: Generates detailed plots of temperature, agent actions, and energy consumption.

## Project Structure

```text
hvac_rl_project/
├── main.py              # Entry point for training and simulation
├── plot_utils.py        # Utility functions for data visualization
├── test_plot.py         # Test script for plotting functionality
├── envs/
│   └── hvac_env.py      # Custom Gymnasium environment (HVACEnv)
├── models/              # Directory for trained model checkpoints
└── utils/               # General utility functions
```

## Setup and Installation

### Prerequisites

- Python 3.8+
- [OpenWeatherMap API Key](https://openweathermap.org/api) (required for real-time weather integration)

### Installation

1. Clone the repository and navigate to the project directory.
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```
3. Install the required dependencies:
   ```bash
   pip install gymnasium stable-baselines3 matplotlib numpy requests torch
   ```

## Usage

1. **Configure API Key**: Add your OpenWeatherMap API key in `main.py` if you wish to use live forecasts.
2. **Run the Simulation**:
   ```bash
   python main.py
   ```
3. **Training & Evaluation**:
   - The script will prompt you for your comfort range (e.g., 20°C to 24°C).
   - An RL agent will train for 20,000 steps.
   - A 24-hour simulation will run, followed by a baseline comparison and performance visualization.

## Performance Metrics

The system evaluates performance based on:
- **Time in Comfort**: Percentage of time the indoor temperature remained within the user's preferred range.
- **Energy Efficiency**: Energy savings achieved compared to a non-intelligent baseline.

## License

MIT License
