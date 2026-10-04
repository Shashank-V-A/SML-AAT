"""
Permutation Test Simulation - SML AAT 2
=======================================
Scenario: Clinical drug trial comparing systolic blood-pressure reduction (mmHg)
  Group A - Placebo
  Group B - Active drug (new antihypertensive)

H0: mean(A) = mean(B)   (no difference in BP reduction)
H1: mean(A) != mean(B)  (two-sided alternative)

Statistic: difference in sample means  (mean_B - mean_A)
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------
SEED = 42
N_A = 30          # patients on placebo
N_B = 30          # patients on active drug
N_PERM = 10_000   # number of random permutations
ALPHA = 0.05      # significance level

# Simulated true BP-reduction distributions (mmHg); used only to generate data
MEAN_A, SD_A = 4.0, 5.0    # placebo: small average reduction
MEAN_B, SD_B = 9.5, 5.0    # drug: larger average reduction

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 2. Data generation
# ---------------------------------------------------------------------------
def generate_data(
    n_a: int = N_A,
    n_b: int = N_B,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate two independent groups of BP reduction values (mmHg)."""
    rng = np.random.default_rng(seed)
    group_a = rng.normal(MEAN_A, SD_A, size=n_a)
    group_b = rng.normal(MEAN_B, SD_B, size=n_b)
    # Clip to a plausible clinical range of reduction
    group_a = np.clip(group_a, -5, 25)
    group_b = np.clip(group_b, -5, 25)
    return group_a, group_b


# ---------------------------------------------------------------------------
# 3. Observed difference
# ---------------------------------------------------------------------------
def mean_difference(a: np.ndarray, b: np.ndarray) -> float:
    """Test statistic: mean(B) - mean(A)."""
    return float(np.mean(b) - np.mean(a))


# ---------------------------------------------------------------------------
# 4. Permutation test
# ---------------------------------------------------------------------------
def permutation_test(
    a: np.ndarray,
    b: np.ndarray,
    n_perm: int = N_PERM,
    seed: int = SEED,
) -> tuple[float, np.ndarray, float]:
    """
    Two-sided permutation test on the difference of means.

    Under H0 the group labels are exchangeable, so we pool all values,
    randomly reassign them to two groups of the original sizes, and
    recompute the mean difference many times.

    Returns
    -------
    observed : float
        Observed mean difference (B - A).
    null_diffs : ndarray of shape (n_perm,)
        Mean differences under random label shuffles.
    p_value : float
        Two-sided p-value = proportion of |null| >= |observed|.
    """
    observed = mean_difference(a, b)
    pooled = np.concatenate([a, b])
    n_a = len(a)

    rng = np.random.default_rng(seed + 1)
    null_diffs = np.empty(n_perm)

    for i in range(n_perm):
        shuffled = rng.permutation(pooled)
        perm_a = shuffled[:n_a]
        perm_b = shuffled[n_a:]
        null_diffs[i] = mean_difference(perm_a, perm_b)

    # Two-sided p-value (include observed for slightly conservative estimate)
    p_value = (np.sum(np.abs(null_diffs) >= np.abs(observed)) + 1) / (n_perm + 1)
    return observed, null_diffs, p_value


# ---------------------------------------------------------------------------
# 5. Plots
# ---------------------------------------------------------------------------
def plot_group_comparison(a: np.ndarray, b: np.ndarray, path: str) -> None:
    """Boxplots + strip of the two trial arms."""
    fig, ax = plt.subplots(figsize=(7, 5))
    data = [a, b]
    bp = ax.boxplot(
        data,
        tick_labels=["Placebo (A)", "Drug (B)"],
        patch_artist=True,
        widths=0.5,
    )
    colors = ["#4C78A8", "#F58518"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)

    rng = np.random.default_rng(0)
    for i, values in enumerate(data, start=1):
        x = rng.normal(i, 0.04, size=len(values))
        ax.scatter(x, values, alpha=0.55, s=28, color=colors[i - 1], zorder=3)

    ax.set_ylabel("Systolic BP reduction (mmHg)")
    ax.set_title("Observed BP Reduction by Treatment Arm")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle=":", alpha=0.7)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_permutation_distribution(
    null_diffs: np.ndarray,
    observed: float,
    p_value: float,
    path: str,
) -> None:
    """Histogram of the null (permutation) distribution with observed statistic."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(
        null_diffs,
        bins=40,
        color="#4C78A8",
        edgecolor="white",
        alpha=0.85,
        density=True,
        label="Permutation distribution under H0",
    )
    ax.axvline(
        observed,
        color="#E45756",
        linewidth=2.2,
        linestyle="--",
        label=f"Observed delta = {observed:.3f}",
    )
    ax.axvline(-observed, color="#E45756", linewidth=1.4, linestyle=":", alpha=0.8)
    ax.set_xlabel("Mean difference (B - A) in mmHg")
    ax.set_ylabel("Density")
    ax.set_title(
        f"Permutation Null Distribution  |  two-sided p ~ {p_value:.4f}"
    )
    ax.legend(loc="upper right", frameon=True)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 60)
    print(" PERMUTATION TEST SIMULATION - Drug Trial (BP Reduction)")
    print("=" * 60)

    # --- Data ---
    group_a, group_b = generate_data()
    df = pd.DataFrame(
        {
            "bp_reduction_mmHg": np.concatenate([group_a, group_b]),
            "group": ["A"] * len(group_a) + ["B"] * len(group_b),
            "arm": ["Placebo"] * len(group_a) + ["Drug"] * len(group_b),
        }
    )
    data_path = os.path.join(OUTPUT_DIR, "bp_reduction.csv")
    df.to_csv(data_path, index=False)

    print("\n--- Descriptive statistics ---")
    summary = df.groupby("arm")["bp_reduction_mmHg"].agg(
        ["count", "mean", "std", "min", "max"]
    )
    print(summary.round(3).to_string())
    print(f"\nSaved dataset -> {data_path}")

    # --- Observed difference ---
    observed = mean_difference(group_a, group_b)
    print(f"\nObserved mean difference (B - A): {observed:.4f} mmHg")

    # --- Permutation test ---
    print(f"\nRunning {N_PERM:,} permutations...")
    observed, null_diffs, p_value = permutation_test(group_a, group_b, N_PERM)
    decision = "Reject H0" if p_value < ALPHA else "Fail to reject H0"

    print("\n--- Permutation test results ---")
    print(f"  Number of permutations : {N_PERM:,}")
    print(f"  Observed delta (B - A) : {observed:.4f} mmHg")
    print(f"  Null mean (should ~ 0) : {null_diffs.mean():.4f}")
    print(f"  Null SD                : {null_diffs.std():.4f}")
    print(f"  Two-sided p-value      : {p_value:.4f}")
    print(f"  alpha                  : {ALPHA}")
    print(f"  Decision               : {decision}")

    results = {
        "n_placebo": len(group_a),
        "n_drug": len(group_b),
        "mean_placebo": float(np.mean(group_a)),
        "mean_drug": float(np.mean(group_b)),
        "observed_diff_drug_minus_placebo": observed,
        "n_permutations": N_PERM,
        "null_mean": float(null_diffs.mean()),
        "null_std": float(null_diffs.std()),
        "p_value_two_sided": p_value,
        "alpha": ALPHA,
        "decision": decision,
    }
    results_df = pd.DataFrame([results])
    results_path = os.path.join(OUTPUT_DIR, "results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved results  -> {results_path}")

    # --- Plots ---
    box_path = os.path.join(OUTPUT_DIR, "group_comparison.png")
    hist_path = os.path.join(OUTPUT_DIR, "permutation_distribution.png")
    plot_group_comparison(group_a, group_b, box_path)
    plot_permutation_distribution(null_diffs, observed, p_value, hist_path)
    print(f"Saved plot     -> {box_path}")
    print(f"Saved plot     -> {hist_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
