# 🌡️ HVAC Energy Optimization with Deep RL + Computer Vision

> **Occupancy-aware thermal management using PPO-based reinforcement learning, YOLOv8 real-time detection, and a custom Gymnasium simulation environment.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29-black)](https://gymnasium.farama.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple)](https://ultralytics.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 Overview

Traditional HVAC systems operate on fixed schedules or simple temperature thresholds, wasting significant energy when spaces are partially or fully unoccupied. This project addresses that gap by building a **closed-loop, occupancy-aware HVAC control system** that:

- Uses **YOLOv8 + OpenCV** to detect the number of people in a room in real time
- Feeds live headcount as a **4th state variable** into a PPO-based RL agent (alongside temperature, humidity, and outdoor weather)
- Trains in a **custom Gymnasium environment** with adaptive reward shaping balancing energy cost and thermal comfort
- Achieves **20%+ reduction in simulated energy consumption** with stable policy convergence within 500K steps

This project sits at the intersection of **computer vision**, **deep reinforcement learning**, and **building energy systems**.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         HVAC Control Loop                           │
│                                                                     │
│  ┌──────────────┐     ┌──────────────────┐     ┌────────────────┐  │
│  │  Camera Feed │────▶│  YOLOv8 + OpenCV │────▶│  Headcount (n) │  │
│  └──────────────┘     └──────────────────┘     └───────┬────────┘  │
│                                                         │           │
│  ┌──────────────┐                                       ▼           │
│  │ OpenWeather  │──── outdoor_temp ──▶  ┌─────────────────────┐    │
│  │     API      │                       │    State Vector s    │    │
│  └──────────────┘                       │  [T_in, RH, T_out, n]│   │
│                                         └──────────┬────────────┘   │
│  ┌──────────────┐                                  │                │
│  │  Room Sensor │──── T_in, RH ──────────────────▶ │                │
│  └──────────────┘                                  ▼                │
│                                         ┌─────────────────────┐    │
│                                         │    PPO Agent (MLP)   │    │
│                                         │   Actor + Critic     │    │
│                                         └──────────┬────────────┘   │
│                                                    │                │
│                                                    ▼                │
│                                         ┌─────────────────────┐    │
│                                         │   HVAC Actuator      │    │
│                                         │  (setpoint control)  │    │
│                                         └─────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🔧 Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.10+ |
| Deep RL | PyTorch, Stable-Baselines3 (PPO) |
| CV / Detection | YOLOv8 (Ultralytics), OpenCV |
| Simulation | Custom Gymnasium Environment |
| Weather Data | OpenWeatherMap API |
| Experiment Tracking | TensorBoard |

---

## 📂 Project Structure

```
hvac-rl-optimization/
│
├── env/
│   ├── hvac_env.py              # Custom Gymnasium environment
│   ├── reward_shaping.py        # Adaptive reward functions
│   └── thermal_model.py         # Room thermal dynamics simulator
│
├── vision/
│   ├── occupancy_detector.py    # YOLOv8 + OpenCV headcount pipeline
│   ├── zone_mapper.py           # Spatial zone occupancy mapping
│   └── utils.py
│
├── agents/
│   ├── ppo_agent.py             # PPO training + evaluation
│   └── baselines.py             # Rule-based baseline for comparison
│
├── data/
│   ├── weather/                 # Cached OpenWeatherMap data
│   └── simulation_logs/         # Episode reward and energy logs
│
├── notebooks/
│   ├── 01_env_exploration.ipynb
│   ├── 02_training_analysis.ipynb
│   └── 03_results_visualization.ipynb
│
├── configs/
│   └── ppo_config.yaml          # Hyperparameters
│
├── tests/
│   ├── test_env.py
│   └── test_detector.py
│
├── requirements.txt
├── train.py                     # Main training script
├── evaluate.py                  # Policy evaluation + visualization
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended for YOLOv8 inference)
- OpenWeatherMap API key (free tier works fine)

### Installation

```bash
# Clone the repository
git clone https://github.com/vruthulashruthi/hvac-rl-optimization.git
cd hvac-rl-optimization

# Create a virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the root directory:

```env
OPENWEATHER_API_KEY=your_api_key_here
CITY=Coimbatore
CAMERA_INDEX=0          # 0 for default webcam, or RTSP URL for IP camera
```

### Running

**Train the PPO agent:**
```bash
python train.py --config configs/ppo_config.yaml --timesteps 500000
```

**Evaluate a trained policy:**
```bash
python evaluate.py --checkpoint checkpoints/ppo_best_model.zip --render
```

**Run occupancy detection only (for testing):**
```bash
python vision/occupancy_detector.py --source 0   # webcam
python vision/occupancy_detector.py --source path/to/video.mp4
```

---

## 🧠 Reinforcement Learning Design

### State Space

| Variable | Description | Range |
|---|---|---|
| `T_in` | Indoor temperature (°C) | [15, 35] |
| `RH` | Relative humidity (%) | [20, 80] |
| `T_out` | Outdoor temperature (°C) | [-5, 45] |
| `n` | Occupant headcount | [0, 20] |

### Action Space

Continuous action: HVAC setpoint adjustment `Δ ∈ [-2.0, +2.0]°C`

### Reward Function

```python
def compute_reward(state, action, occupancy):
    comfort_penalty = -abs(state['T_in'] - COMFORT_SETPOINT) * occupancy_weight(occupancy)
    energy_cost     = -ENERGY_COEFF * abs(action)
    idle_penalty    = -IDLE_COEFF * abs(action) if occupancy == 0 else 0
    return comfort_penalty + energy_cost + idle_penalty
```

The reward function **scales comfort penalty with occupancy** — when the room is empty, energy savings are prioritized over tight temperature control.

### PPO Hyperparameters

```yaml
policy: MlpPolicy
learning_rate: 3e-4
n_steps: 2048
batch_size: 64
n_epochs: 10
gamma: 0.99
gae_lambda: 0.95
clip_range: 0.2
ent_coef: 0.005
```

---

## 📊 Results

| Metric | Rule-Based Baseline | PPO Agent |
|---|---|---|
| Avg. Energy Consumption | 100% (baseline) | **~78%** (-22%) |
| Comfort Violations (hrs) | 4.2 / day | **1.8 / day** |
| Policy Convergence | — | **~500K steps** |
| Avg. Episode Reward | -312 | **-187** |

Training curve shows stable convergence with no catastrophic forgetting after ~350K steps.

---

## 📸 Occupancy Detection

The vision module runs YOLOv8n (nano, optimized for speed) on each frame, filters detections by class `person` with confidence > 0.5, and returns a headcount that is polled every 5 seconds into the RL state vector.

```python
from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

def get_headcount(frame):
    results = model(frame, classes=[0], conf=0.5, verbose=False)
    return len(results[0].boxes)
```

---

## 🧪 Testing

```bash
pytest tests/ -v
```

Tests cover:
- Gymnasium environment `step()` / `reset()` correctness
- Reward function edge cases (empty room, overcrowded)
- YOLOv8 detector on sample frames

---

## ⚠️ Limitations

- Simulation environment uses a simplified thermal model; real-world deployment would require calibration with actual building data.
- YOLOv8 headcount accuracy degrades under occlusion and low-light conditions.
- Weather data is polled every 10 minutes from OpenWeatherMap; real deployments should use on-site sensors.
- Current action space is single-zone; multi-zone control is a planned extension.

---

## 🗺️ Roadmap

- [ ] Multi-zone spatial control using zone-level occupancy maps
- [ ] MediaPipe pose estimation for activity-level-aware comfort modelling (PMV index)
- [ ] Deploy trained policy as a FastAPI microservice for edge hardware integration
- [ ] Real-world pilot with ESP32-based temperature/humidity sensors

---

## 🤝 Contributing

Contributions are welcome! Please open an issue first to discuss what you'd like to change. For major changes, fork the repo and submit a pull request.

```bash
git checkout -b feature/your-feature-name
git commit -m "feat: add your feature"
git push origin feature/your-feature-name
```

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 👩‍💻 Author

**Vruthula Shruthi GS** — [GitHub](https://github.com/vruthulashruthi) · [LinkedIn](https://linkedin.com/in/vruthula-shruthi-gs-647014286)

> *If you found this project useful, consider giving it a ⭐ — it helps others find it!*
