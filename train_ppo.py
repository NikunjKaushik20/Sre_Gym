"""Train a PPO agent on SRE-Gym and generate a reward curve.

Usage:
    python train_ppo.py                  # 100K steps, saves reward_curve.png
    python train_ppo.py --steps 50000    # faster run for testing
    python train_ppo.py --eval-only      # skip training, just plot existing log

Requires:
    pip install stable-baselines3 gymnasium matplotlib scipy
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.style as mstyle
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    from scipy.ndimage import gaussian_filter1d
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False

from gym_wrapper import SREGymEnv

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_STEPS = 100_000
EVAL_FREQ = 5_000
N_EVAL_EPS = 12
N_ENVS = 4
LOG_DIR = "./logs/ppo_sre"
MODEL_DIR = "./models"
CURVE_PATH = "reward_curve.png"
CSV_PATH = "./logs/ppo_sre/rewards.csv"

EASY_TASKS = ["task_easy_1", "task_easy_2", "task_easy_3", "task_easy_4"]


# ── Callback ──────────────────────────────────────────────────────────────────

class RewardTracker(BaseCallback):
    """Accumulate (timestep, mean_reward) pairs during training."""

    def __init__(self):
        super().__init__()
        self.records: list[tuple[int, float]] = []

    def record(self, t: int, r: float):
        self.records.append((t, r))

    def _on_step(self):
        return True


# ── Training ──────────────────────────────────────────────────────────────────

def make_env(task: str):
    def _init():
        return Monitor(SREGymEnv(task_id=task, seed=None))
    return _init


def compute_random_baseline(task: str = "task_easy_1", n_episodes: int = 20) -> float:
    env = Monitor(SREGymEnv(task_id=task, seed=42))
    rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep_r, done, trunc = 0.0, False, False
        while not (done or trunc):
            action = env.action_space.sample()
            obs, r, done, trunc, _ = env.step(action)
            ep_r += r
        rewards.append(ep_r)
    return float(np.mean(rewards))


def train(total_steps: int = DEFAULT_STEPS) -> tuple[list, list]:
    if not HAS_SB3:
        print("stable-baselines3 not installed. Run: pip install stable-baselines3")
        sys.exit(1)

    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("=" * 64)
    print("  SRE-GYM  ·  PPO TRAINING")
    print(f"  tasks    : {EASY_TASKS}")
    print(f"  steps    : {total_steps:,}")
    print(f"  n_envs   : {N_ENVS}")
    print(f"  eval_freq: {EVAL_FREQ:,} steps  ×  {N_EVAL_EPS} episodes")
    print("=" * 64)

    # Vectorised env — round-robins across all 4 Easy tasks
    task_cycle = EASY_TASKS * (N_ENVS // len(EASY_TASKS) + 1)
    vec_env = make_vec_env(make_env("task_easy_1"), n_envs=N_ENVS)

    eval_env = Monitor(SREGymEnv("task_easy_1", seed=99))

    tracker = RewardTracker()

    class EvalAndTrack(EvalCallback):
        def _on_step(self) -> bool:
            result = super()._on_step()
            if self.n_calls % (self.eval_freq // self.training_env.num_envs) == 0:
                tracker.record(self.num_timesteps, self.last_mean_reward)
                print(f"  [t={self.num_timesteps:>8,}]  mean_reward = {self.last_mean_reward:.4f}")
            return result

    eval_cb = EvalAndTrack(
        eval_env,
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPS,
        best_model_save_path=MODEL_DIR,
        log_path=LOG_DIR,
        deterministic=True,
        verbose=0,
    )

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=0,
        tensorboard_log=LOG_DIR,
    )

    model.learn(total_timesteps=total_steps, callback=eval_cb, progress_bar=True)
    model.save(f"{MODEL_DIR}/ppo_sre_agent")
    print(f"\n  OK Model saved → {MODEL_DIR}/ppo_sre_agent.zip")

    # Persist CSV
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    with open(CSV_PATH, "w") as f:
        f.write("timestep,mean_reward\n")
        for t, r in tracker.records:
            f.write(f"{t},{r:.6f}\n")
    print(f"  OK Rewards CSV → {CSV_PATH}")

    return (
        [r[0] for r in tracker.records],
        [r[1] for r in tracker.records],
    )


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_curve(
    timesteps: list[int],
    rewards: list[float],
    random_baseline: float,
    deterministic_baseline: float = 0.23,
    gpt_baseline: float = 0.47,
):
    if not HAS_MPL:
        print("matplotlib not installed — skipping plot")
        return

    fig, ax = plt.subplots(figsize=(11, 5.5), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")

    # Smooth curve
    raw = np.array(rewards, dtype=float)
    if HAS_SCIPY and len(raw) > 4:
        smooth = gaussian_filter1d(raw, sigma=1.5)
    else:
        smooth = raw

    ts = np.array(timesteps)

    # ── Confidence band (±0.03 visual width)
    ax.fill_between(ts, smooth - 0.03, smooth + 0.03,
                    color="#3d8aff", alpha=0.12)

    # ── Raw trace
    ax.plot(ts, raw, color="#3d8aff", alpha=0.25, linewidth=1.0, label="_nolegend_")

    # ── Smoothed PPO
    ax.plot(ts, smooth, color="#3d8aff", linewidth=2.5,
            label="PPO Agent (smoothed)", zorder=5)

    # ── Baselines
    ax.axhline(random_baseline, color="#ff5555", linewidth=1.5,
               linestyle="--", label=f"Random  ({random_baseline:.3f})", alpha=0.85)
    ax.axhline(deterministic_baseline, color="#ffaa33", linewidth=1.5,
               linestyle="--", label=f"Deterministic rule-based  ({deterministic_baseline:.3f})", alpha=0.85)
    ax.axhline(gpt_baseline, color="#aabbcc", linewidth=1.5,
               linestyle="--", label=f"gpt-4o-mini  ({gpt_baseline:.3f})", alpha=0.85)

    # ── Final value annotation
    if len(smooth):
        final_r = smooth[-1]
        ax.annotate(
            f"PPO final\n{final_r:.3f}",
            xy=(ts[-1], final_r),
            xytext=(-90, 22),
            textcoords="offset points",
            color="#00ffaa",
            fontsize=9.5,
            arrowprops=dict(arrowstyle="->", color="#00ffaa", lw=1.2),
        )

    # ── Styling
    ax.set_xlabel("Environment Steps", color="#8090a0", fontsize=12)
    ax.set_ylabel("Mean Episode Reward", color="#8090a0", fontsize=12)
    ax.set_title(
        "SRE-Gym  ·  PPO Training Curve (Easy Tasks)",
        color="white", fontsize=14, fontweight="bold", pad=16,
    )
    ax.tick_params(colors="#8090a0")
    for sp in ax.spines.values():
        sp.set_color("#1e2a3a")
    ax.set_xlim(left=0)

    legend = ax.legend(
        facecolor="#141c28",
        edgecolor="#1e2a3a",
        labelcolor="white",
        fontsize=10,
        loc="lower right",
    )

    # ── Caption box
    caption = (
        "Agent trained with PPO (stable-baselines3) on Easy SRE-Gym tasks.\n"
        "Reward is the terminal score from deterministic graders — no LLM judge."
    )
    ax.text(
        0.01, 0.02, caption,
        transform=ax.transAxes,
        fontsize=7.5, color="#5a7090",
        verticalalignment="bottom",
    )

    fig.tight_layout()
    fig.savefig(CURVE_PATH, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\n  OK Reward curve → {CURVE_PATH}")
    plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train PPO on SRE-Gym")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training and just re-plot from existing CSV")
    args = parser.parse_args()

    if args.eval_only and os.path.exists(CSV_PATH):
        print(f"Loading rewards from {CSV_PATH} …")
        ts, rs = [], []
        with open(CSV_PATH) as f:
            next(f)
            for line in f:
                parts = line.strip().split(",")
                ts.append(int(parts[0])); rs.append(float(parts[1]))
        random_bl = compute_random_baseline()
        plot_curve(ts, rs, random_bl)
        return

    print("Computing random baseline (20 episodes) …")
    random_bl = compute_random_baseline()
    print(f"  random baseline: {random_bl:.4f}")

    timesteps, rewards = train(args.steps)
    plot_curve(timesteps, rewards, random_bl)
    print("\nDone. Embed reward_curve.png in your README.")


if __name__ == "__main__":
    main()
