# test_plot.py
from envs.hvac_env import HVACEnv
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
import numpy as np
import os
from plot_utils import plot_results

# -------------------------------
# 1. Environment setup
# -------------------------------
API_KEY = "074d9d358936a05d8b72f14bd95cc1b2" 
LATITUDE = 13.0827
LONGITUDE = 80.2707
MAX_TIME = 24
PERSONAL_COMFORT = (21.0, 24.0)

env = HVACEnv(
    api_key=API_KEY,
    latitude=LATITUDE,
    longitude=LONGITUDE,
    fetch_interval=1,
    max_time=MAX_TIME,
    weather_data=None,
    comfort_range=PERSONAL_COMFORT,
)

# -------------------------------
# 2. Load trained model
# -------------------------------
model_path = "models/hvac_rl_model.zip"
if not os.path.exists(model_path):
    model_path_alt = "models/hvac_rl_model"
    if os.path.exists(model_path_alt):
        model_path = model_path_alt
    else:
        raise FileNotFoundError(f"Could not find trained model at {model_path} or {model_path_alt}")

model = PPO.load(model_path, env=env)

# -------------------------------
# 3. Test agent
# -------------------------------
obs, info = env.reset()
temps = []
actions = []
rewards = []
energy_used = 0.0
comfort_hours = 0

for t in range(env.max_time):
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)

    temps.append(float(obs[0]))
    act_val = float(action[0])
    actions.append(act_val)
    rewards.append(float(reward))

    # Energy proxy: absolute action times a scaling factor 
    energy_used += abs(act_val) * 1.0  

    # Count comfort hours using the current personalized bounds 
    if PERSONAL_COMFORT[0] <= obs[0] <= PERSONAL_COMFORT[1]:
        comfort_hours += 1

    if terminated:
        break

# -------------------------------
# 4. Calculate metrics
# -------------------------------
avg_reward = np.mean(rewards) if rewards else 0.0
comfort_percentage = (comfort_hours / env.max_time) * 100

print(f"Total energy proxy used: {energy_used:.3f} units")
print(f"Percentage of time in comfort range: {comfort_percentage:.1f}%")
print(f"Average reward: {avg_reward:.3f}")

# -------------------------------
# 5. Plot
# -------------------------------
plot_results(temps, actions)

plt.figure(figsize=(12,3))
plt.title("HVAC Actions Over Time (test)")
plt.bar(range(len(actions)), actions, alpha=0.4)
plt.xlabel("Hour")
plt.ylabel("Action (continuous: -1..1)")
plt.show()
