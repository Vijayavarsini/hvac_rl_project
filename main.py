# main.py
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from envs.hvac_env import HVACEnv
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
import numpy as np
from plot_utils import plot_results

# --------------------------
# Baseline HVAC always ON
# --------------------------
def run_baseline(env, P_MAX=3.5, STEP_HOURS=1.0):
    """
    Simulates HVAC running at full power all the time.
    Returns total energy used (kWh) and temperature trace.
    """
    obs, info = env.reset()
    temps, actions = [], []
    total_energy_kwh = 0.0

    for _ in range(env.max_time):
        action = np.array([1.0]) 
        obs, reward, terminated, truncated, info = env.step(action)

        temps.append(float(info["indoor_temp"]))
        actions.append(float(action[0]))
        total_energy_kwh += abs(action[0]) * P_MAX * STEP_HOURS

        if terminated:
            break

    return temps, actions, total_energy_kwh

# --------------------------
# Ask user for personalized comfort range
# --------------------------
print("Welcome to the Smart HVAC RL System!")
print("Let's personalize your comfort range (in °C).")

while True:
    try:
        lower = float(input("Enter your preferred LOWER temperature limit (e.g., 21): "))
        upper = float(input("Enter your preferred UPPER temperature limit (e.g., 24): "))
        if upper <= lower:
            print("⚠️  Upper limit must be greater than lower limit. Try again.")
            continue
        break
    except ValueError:
        print("❌ Invalid input. Please enter numeric values.")

PERSONAL_COMFORT = (lower, upper)
print(f"✅ Personalized comfort range set to {PERSONAL_COMFORT}°C")

# --------------------------
# Environment / Model Settings
# --------------------------
API_KEY = "074d9d358936a05d8b72f14bd95cc1b2"  
LATITUDE = 13.0827
LONGITUDE = 80.2707
MAX_TIME = 24
TOTAL_TIMESTEPS = 20000  
P_MAX = 3.5  
STEP_HOURS = 1.0

# --------------------------
# Create environment with personalized comfort
# --------------------------
env = HVACEnv(
    api_key=API_KEY,
    latitude=LATITUDE,
    longitude=LONGITUDE,
    fetch_interval=1,
    max_time=MAX_TIME,
    weather_data=None,
    comfort_range=PERSONAL_COMFORT,
    energy_coeff=0.3,
    comfort_weight=5.0,
    adaptive_feedback_threshold=3,
    adaptive_step=0.3,
)

# --------------------------
# Train RL agent
# --------------------------
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=TOTAL_TIMESTEPS)
model.save("models/hvac_rl_model")

# --------------------------
# Test the trained agent
# --------------------------
obs, info = env.reset()
temps, actions, rewards, comfort_ranges, feedbacks = [], [], [], [], []

for _ in range(env.max_time):
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)

    temps.append(float(obs[0]))
    actions.append(float(action[0]))
    rewards.append(float(reward))
    comfort_ranges.append(info.get("comfort_range", (None, None)))
    feedbacks.append(info.get("feedback", "na"))

    if terminated:
        break

# --------------------------
# Evaluation Metrics for RL
# --------------------------
temps = np.array(temps)
actions = np.array(actions)
rewards = np.array(rewards)

time_in_comfort = np.mean((temps >= PERSONAL_COMFORT[0]) & (temps <= PERSONAL_COMFORT[1])) * 100
avg_reward = np.mean(rewards)
total_energy_kwh = np.sum(np.abs(actions) * P_MAX * STEP_HOURS)

# --------------------------
# Baseline HVAC full power
# --------------------------
print("\nRunning HVAC at full power all the time for comparison...")
full_on_temps, full_on_actions, full_on_energy = run_baseline(env)

print(f"Full-time HVAC total energy consumed: {full_on_energy:.2f} kWh")

# --------------------------
# Energy comparison
# --------------------------
energy_saving = (full_on_energy - total_energy_kwh) / full_on_energy * 100

print("\n===== Evaluation Metrics =====")
print(f"RL-based HVAC Total Energy: {total_energy_kwh:.2f} kWh")
print(f"Full-time HVAC Energy: {full_on_energy:.2f} kWh")
print(f"Energy saving with RL: {energy_saving:.1f}%")
print(f"Average Reward: {avg_reward:.3f}")
print(f"Time in comfort range {PERSONAL_COMFORT}: {time_in_comfort:.2f}%")
print("Final comfort range (after adaptation):", comfort_ranges[-1])
print("Feedback trace (last 10):", feedbacks[-10:])
print("================================\n")

# --------------------------
# Visualization
# --------------------------
plot_results(temps.tolist(), actions.tolist())