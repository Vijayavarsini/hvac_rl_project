# envs/hvac_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import requests
import os

class HVACEnv(gym.Env):
    """
    Gymnasium environment for an HVAC system with personalized comfort modeling.

    Major additions:
    - personalized comfort range (comfort_lower, comfort_upper) passed at init
    - simulated user feedback each step: "too_cold", "too_hot", or "ok"
    - adaptive comfort shift: if the same feedback appears repeatedly, the comfort range shifts slightly
    - observation contains [indoor_temp, outdoor_temp, time_of_day_norm]
    - reward uses personalized comfort range + energy penalty + small feedback bonus
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        api_key="074d9d358936a05d8b72f14bd95cc1b2",
        latitude=13.0827,
        longitude=80.2707,
        fetch_interval=1,
        max_time=24,
        weather_data=None,
        comfort_range=(21.0, 24.0),
        energy_coeff=0.5,
        comfort_weight=2.0,
        adaptive_feedback_threshold=3,
        adaptive_step=0.2,
    ):
        super(HVACEnv, self).__init__()

        self.api_key = api_key
        self.latitude = latitude
        self.longitude = longitude
        self.fetch_interval = fetch_interval
        self.max_time = max_time
        self.current_step = 0

        # Weather: use provided or fetch from API
        if weather_data is None and api_key is not None:
            try:
                self.weather_data = self.fetch_weather_data()
            except Exception:
                print("Weather fetch failed; using synthetic weather.")
                self.weather_data = self._synthetic_weather()
        elif weather_data is not None:
            self.weather_data = weather_data
        else:
            self.weather_data = self._synthetic_weather()

        # Initial indoor temp
        self.indoor_temp = 22.0 

        # Personalized comfort: (lower, upper)
        self.comfort_lower, self.comfort_upper = comfort_range

        # Reward and adaptation parameters
        self.energy_coeff = energy_coeff
        self.comfort_weight = comfort_weight

        # Feedback-adaptation parameters
        # If the same feedback occurs `adaptive_feedback_threshold` times in a row,
        # shift the comfort range by adaptive_step toward current temp direction.
        self.adaptive_feedback_threshold = adaptive_feedback_threshold
        self.adaptive_step = adaptive_step
        self.feedback_history = [] 

        # Action space: continuous in [-1, 1] (negative -> cooling, positive -> heating)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Observation: indoor temp, outdoor temp, normalized time of day
        low = np.array([ -50.0, -50.0, 0.0 ], dtype=np.float32)
        high = np.array([ 60.0, 60.0, 1.0 ], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.power_coefficient = 3.0  # HVAC max effect scaling factor

    def fetch_weather_data(self):
        url = (
            f"https://api.openweathermap.org/data/2.5/forecast?"
            f"lat={self.latitude}&lon={self.longitude}&appid={self.api_key}&units=metric"
        )
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        # Extract next 24 hourly temperatures (if available)
        temps = [entry["main"]["temp"] for entry in data["list"][:24]]
        if len(temps) < 24:
            temps += [temps[-1]] * (24 - len(temps))
        return temps

    def _synthetic_weather(self):
        base = 25.0
        hours = np.arange(24)
        temps = base + 5 * np.sin((hours - 14) * (2 * np.pi / 24)) 
        return temps.tolist()

    def _simulate_feedback(self, temp):
        """
        Synthetic user feedback:
        - 'too_cold' if temp < comfort_lower - 0.3
        - 'too_hot' if temp > comfort_upper + 0.3
        - 'ok' otherwise
        This is a synthetic signal to emulate occupant feedback.
        """
        if temp < (self.comfort_lower - 0.3):
            return "too_cold"
        elif temp > (self.comfort_upper + 0.3):
            return "too_hot"
        else:
            return "ok"

    def _adaptive_adjust_comfort(self):
        """
        If the same feedback appears repeatedly (>= threshold), shift the comfort range
        slightly toward the current indoor_temp. For example, if repeated 'too_cold',
        increase both lower and upper bounds a bit so the agent learns occupant prefers warmer.
        """
        if len(self.feedback_history) < self.adaptive_feedback_threshold:
            return

        # Check the last N feedbacks
        recent = self.feedback_history[-self.adaptive_feedback_threshold :]
        if all(f == "too_cold" for f in recent):
            # shift range upward
            self.comfort_lower += self.adaptive_step
            self.comfort_upper += self.adaptive_step
            # reset history to avoid continuous shifts
            self.feedback_history = []
        elif all(f == "too_hot" for f in recent):
            # shift range downward
            self.comfort_lower -= self.adaptive_step
            self.comfort_upper -= self.adaptive_step
            self.feedback_history = []

        # Keep bounds reasonable
        self.comfort_lower = float(np.clip(self.comfort_lower, 15.0, 28.0))
        self.comfort_upper = float(np.clip(self.comfort_upper, self.comfort_lower + 0.5, 35.0))

    def step(self, action):
        """
        action: array-like with single continuous value in [-1,1].
        Negative -> cooling, Positive -> heating.
        Returns: obs, reward, terminated, truncated, info
        """
        action_val = float(np.clip(action[0], -1.0, 1.0))

        # Outdoor temperature for current step
        outdoor_temp = float(self.weather_data[self.current_step % len(self.weather_data)])

        # HVAC effect: scaled by power_coefficient and action
        hvac_effect = action_val * self.power_coefficient  # degrees per step roughly
        # Indoor temp update: simple linear thermal model
        # indoor moves toward outdoor (passive) and is adjusted by HVAC effect
        self.indoor_temp += 0.05 * (outdoor_temp - self.indoor_temp) + hvac_effect * 0.7

        # Compute feedback and possibly adapt comfort range
        feedback = self._simulate_feedback(self.indoor_temp)
        self.feedback_history.append(feedback)
        self._adaptive_adjust_comfort()

        # Comfort penalty: zero inside personalized range, otherwise proportional to distance
        if self.comfort_lower <= self.indoor_temp <= self.comfort_upper:
            comfort_penalty = 0.0
        elif self.indoor_temp < self.comfort_lower:
            comfort_penalty = self.comfort_weight * (self.comfort_lower - self.indoor_temp)
        else:
            comfort_penalty = self.comfort_weight * (self.indoor_temp - self.comfort_upper)

        # Energy penalty: proportional to absolute action magnitude
        energy_penalty = self.energy_coeff * abs(action_val)

        # Small feedback bonus: if occupant says 'ok' give tiny positive reward, else small negative
        feedback_bonus = 0.1 if feedback == "ok" else -0.05

        reward = - (comfort_penalty + energy_penalty) + feedback_bonus

        self.current_step += 1
        terminated = self.current_step >= self.max_time
        truncated = False

        obs = np.array(
            [
                float(self.indoor_temp),
                float(outdoor_temp),
                float((self.current_step % 24) / 23.0), 
            ],
            dtype=np.float32,
        )

        info = {
            "indoor_temp": float(self.indoor_temp),
            "outdoor_temp": float(outdoor_temp),
            "action": float(action_val),
            "comfort_range": (float(self.comfort_lower), float(self.comfort_upper)),
            "feedback": feedback,
        }

        return obs, float(reward), terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        # Reset indoor temp to a comfortable default
        self.indoor_temp = 22.0
        self.feedback_history = []
        # Return obs, info (Gymnasium-style)
        outdoor_temp = float(self.weather_data[0])
        obs = np.array([self.indoor_temp, outdoor_temp, 0.0], dtype=np.float32)
        info = {
            "indoor_temp": float(self.indoor_temp),
            "outdoor_temp": float(outdoor_temp),
            "comfort_range": (float(self.comfort_lower), float(self.comfort_upper)),
        }
        return obs, info

    def render(self, mode="human"):
        print(
            f"Step {self.current_step} | Indoor: {self.indoor_temp:.2f}°C | "
            f"Comfort: [{self.comfort_lower:.2f}, {self.comfort_upper:.2f}]"
        )

    def close(self):
        pass