from dataclasses import dataclass

@dataclass
class Settings:
    n_chasers: int = 3
    n_fake_runners: int = 4
    random_seed: int | None = None
    chaser_false_negative_probability: float = 0.25
    chaser_detection_radius: float = 4.0
    runner_false_negative_probability: float = 0.25
    runner_false_positive_probability: float = 0.25
    runner_detection_radius: float = 4.0


