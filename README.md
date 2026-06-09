# Smart HVAC Reinforcement Learning System

A sophisticated Reinforcement Learning (RL) project using **Proximal Policy Optimization (PPO)** to optimize HVAC (Heating, Ventilation, and Air Conditioning) control. This system balances indoor thermal comfort against energy efficiency by learning from real-word weather data and adaptive occupant feedback.

## 🚀 Key Features

- **Custom Gymnasium Environment**: A first-principles thermal simulation environment ([envs/hvac_env.py](envs/hvac_env.py)).
- **Hybrid Weather Data**: Integrates the **OpenWeatherMap API** for real-time forecasting, with a fallback to synthetic sine-wave modeling.
- **Adaptive Personalization**: Learns occupant preferences through a feedback loop, dynamically shifting comfort bounds based on "too hot" or "too cold" signals.
- **Energy-Aware Control**: Continuous action space control that minimizes power consumption while maintaining homeostasis.
- **Comparative Benchmarking**: Direct analytics comparing RL performance against a "Full Power" baseline.

## 🛠️ Technical Specifications

### Reinforcement Learning Model
- **Algorithm**: PPO (Stable-Baselines3)
- **State Space (3-Dimensions)**:
  - Inside Temperature (°C)
  - Outside Temperature (°C)
  - Time of Day (Normalized 0-1)
- **Action Space**: Continuous control `[-1.0, 1.0]` (Negative: Cooling, Positive: Heating).

### Thermal Physics Implementation
The simulation uses a first-order thermal decay model with active HVAC influence:
$$\Delta T = K_{leak}(T_{out} - T_{in}) + (A \times P_{max} \times \eta)$$
where $K_{leak}$ is the thermal leakage coefficient and $\eta$ is the system efficiency.

### Reward Function
The agent is optimized via a composite reward signal:
$$R = - (w_c \cdot \text{ComfortLoss} + w_e \cdot \text{EnergyLoss}) + \text{FeedbackReward}$$
- **ComfortLoss**: Linear penalty for deviations from the dynamic target range.
- **EnergyLoss**: Quadratic penalty on effort ($Action^2$).

## 📁 Project Structure

```text
hvac_rl_project/
├── main.py              # Training loop, CLI, and Baseline comparison
├── plot_utils.py        # Dual-axis visualization (Temp vs. HVAC Action)
├── test_plot.py         # Unit testing for visualization components
├── envs/
│   └── hvac_env.py      # Core HVACEnv class & Physics simulation
├── models/              # Trained PPO checkpoints (.zip)
└── utils/               # Shared helper functions
```

## ⚙️ Installation

### Prerequisites
- Python 3.8+
- [OpenWeatherMap API Key](https://openweathermap.org/api) (Optional but recommended)

### Setup
1. **Clone & Virtual Env**:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # Windows
   # or source venv/bin/activate on Linux/Mac
   ```
2. **Install Dependencies**:
   ```bash
   pip install gymnasium stable-baselines3 matplotlib numpy requests torch
   ```

## 📈 Usage

1. **Configure API**: Insert your `API_KEY` in `main.py` at line 55.
2. **Execute**:
   ```bash
   python main.py
   ```
3. **Flow**:
   - Provide your initial comfort range (e.g., `19`, `23`).
   - The agent trains for 20,000 steps (Configurable).
   - Simulation runs for 24 hours.
   - Comparison plots are displayed showing energy savings and temperature stability.

## 📊 Analytics & Metrics

The system tracks:
- **Energy Savings %**: Kilowatt-hour reduction vs. baseline.
- **Comfort Score**: Percentage of time within the target range.
- **Mean Reward**: Training convergence metric.

## 📜 License
MIT License
