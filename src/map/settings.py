from dataclasses import dataclass

@dataclass
class Settings:
    n_chasers: int = 3
    n_fake_runners: int = 5
    random_seed: int | None = None
    chaser_false_negative_probability: float = 0.9
    chaser_detection_radius: float = 3.0
    runner_false_negative_probability: float = 0.5
    runner_false_positive_probability: float = 0.5
    runner_detection_radius: float = 3.0


