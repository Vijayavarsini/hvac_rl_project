# plot_utils.py
import matplotlib.pyplot as plt
import numpy as np

def plot_results(temps, actions):
    """
    temps: list or 1D array of indoor temps
    actions: list or 1D array of continuous actions (-1..1)
    """
    temps = np.array(temps)
    actions = np.array(actions).flatten()

    plt.figure(figsize=(12,5))
    plt.plot(temps, label="Indoor Temp (°C)", marker='o')
    plt.plot([20]*len(temps), "--", label="Lower Comfort Limit (ref 20°C)")
    plt.plot([25]*len(temps), "--", label="Upper Comfort Limit (ref 25°C)")

    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax2.bar(range(len(actions)), actions, alpha=0.3, label="HVAC Action (neg=cool, pos=heat)")
    ax2.set_ylabel("Action (-1..1)")

    ax1.set_xlabel("Time Step (hour)")
    ax1.set_ylabel("Temperature (°C)")
    ax1.set_title("HVAC RL Simulation Results")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.show()
