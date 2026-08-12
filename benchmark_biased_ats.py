"""
benchmark_biased_ats.py
========================

Active Teacher Selection (ATS) under biased teachers, in the Hidden
Utility Bandit (HUB) setting -- benchmark of an active, bias-aware
selection policy against Freedman et al. (2023)'s original ATS
assumption of unbiased teachers.

This module builds directly on the verified joint Bayesian filter in
``biased_teacher_hub.py`` and adds three things:

1. Utility identifiability fix
   ---------------------------
   ``biased_teacher_hub.BayesianBiasFilter`` correctly recovers the
   *shape* of U (item ranking, pairwise gaps) but leaves a free global
   additive shift, because pairwise comparisons only ever constrain
   differences ``U_i - U_j``. Here we remove that gauge freedom by
   anchoring one reference item's utility grid to the single point
   ``{0.0}`` (default: item 0), which pins the coordinate system
   without changing anything about the inference math elsewhere.

2. Active selection policies
   --------------------------
   ``select_query_biased_ats`` chooses the (teacher, item_i, item_j)
   query that maximizes the *expected reduction in posterior utility
   variance per unit query cost* -- i.e. Value-of-Information (VOI)
   divided by ``f_m``. ``select_query_standard_ats`` runs the identical
   VOI/cost rule, but against a filter that has its bias grid collapsed
   to the single point ``b_m = 0`` for every teacher, reproducing
   Freedman et al.'s original assumption that teachers are unbiased.

3. Benchmark
   ---------
   Three algorithms share one ``BiasedTeacherEnv`` and are compared
   over 150 rounds on utility-reconstruction error, bias-estimation
   error, and cumulative regret (arm-exploitation regret plus query
   costs paid), with results written to ``benchmark_results.png``.

Dependencies: NumPy, SciPy, and Matplotlib only.
"""

from __future__ import annotations

import itertools
from typing import Callable, Sequence

import matplotlib

matplotlib.use("Agg")  # headless-safe backend; we only ever save to disk
import matplotlib.pyplot as plt
import numpy as np

from biased_teacher_hub import BiasedTeacherEnv, TeacherSpec


def logsumexp(x: np.ndarray) -> float:
    """Numerically stable log-sum-exp over an entire array.

    Equivalent to ``scipy.special.logsumexp(x)`` (full reduction, no
    ``axis``), reimplemented in plain NumPy. The active-selection policies
    below call this on every hypothetical (teacher, item_i, item_j, choice)
    combination -- tens of thousands of times over a 150-round benchmark --
    and scipy's general-purpose implementation carries array-API dispatch
    overhead that dominates runtime at that call volume; this version is
    algorithmically identical but avoids that overhead.
    """
    x_max = np.max(x)
    if not np.isfinite(x_max):
        x_max = 0.0  # x is all -inf (zero total mass): fall back safely
    return float(x_max + np.log(np.sum(np.exp(x - x_max))))


# --------------------------------------------------------------------------- #
# 1. Identifiable Bayesian bias filter (anchored utility gauge)
# --------------------------------------------------------------------------- #


class AnchoredBayesianBiasFilter:
    """Joint discrete-grid Bayesian filter over (U, b) with a fixed utility gauge.

    This is a drop-in generalization of
    :class:`biased_teacher_hub.BayesianBiasFilter` with two extra knobs:

    - ``anchor_item``: if set (default item 0), that item's utility grid
      is collapsed to the single candidate value ``{0.0}``, removing
      the global shift symmetry that otherwise makes the absolute
      scale of U unidentifiable from pairwise data alone. All other
      items are still estimated relative to this fixed reference, so
      ``U_anchor = 0`` exactly and by construction, always.
    - ``assume_unbiased``: if True, every teacher's bias grid is
      collapsed to the single candidate value ``{0.0}`` as well, i.e.
      the filter is mathematically incapable of learning a bias term.
      This reproduces the standard (Freedman et al., 2023) ATS
      assumption that teachers answer without systematic bias, and is
      used as the misspecified baseline model in the benchmark below.

    Collapsing an axis to a single grid point is implemented uniformly
    (no special-casing): a size-1 axis simply carries all of its
    posterior mass on one candidate value, so marginalizing it out
    trivially returns that fixed value with zero variance.
    """

    def __init__(
        self,
        n_items: int,
        n_teachers: int,
        teacher_betas: Sequence[float],
        u_grid: np.ndarray | None = None,
        b_grid: np.ndarray | None = None,
        anchor_item: int | None = 0,
        assume_unbiased: bool = False,
    ) -> None:
        self.n_items = n_items
        self.n_teachers = n_teachers
        self.beta = np.asarray(teacher_betas, dtype=np.float64)
        if self.beta.shape[0] != n_teachers:
            raise ValueError("teacher_betas must have length n_teachers")

        base_u_grid = np.linspace(-10.0, 10.0, 15) if u_grid is None else np.asarray(u_grid, dtype=np.float64)
        base_b_grid = np.linspace(-5.0, 5.0, 11) if b_grid is None else np.asarray(b_grid, dtype=np.float64)

        self.anchor_item = anchor_item
        self.assume_unbiased = assume_unbiased
        self.n_axes = n_items + n_teachers

        # Per-axis candidate-value grids. Axes [0, n_items) are items,
        # axes [n_items, n_items + n_teachers) are teacher biases.
        self.axis_grids: list[np.ndarray] = []
        for k in range(n_items):
            if anchor_item is not None and k == anchor_item:
                self.axis_grids.append(np.array([0.0]))
            else:
                self.axis_grids.append(base_u_grid.copy())
        for _m in range(n_teachers):
            if assume_unbiased:
                self.axis_grids.append(np.array([0.0]))
            else:
                self.axis_grids.append(base_b_grid.copy())

        self.axis_sizes = [g.shape[0] for g in self.axis_grids]
        tensor_shape = tuple(self.axis_sizes)

        log_prior = np.zeros(tensor_shape, dtype=np.float64)  # uniform prior
        self.log_posterior = log_prior - logsumexp(log_prior)
        self.n_updates = 0

    # -- internal tensor plumbing ------------------------------------- #

    def _broadcast_shape(self, axis: int) -> tuple[int, ...]:
        shape = [1] * self.n_axes
        shape[axis] = self.axis_sizes[axis]
        return tuple(shape)

    def _logits_tensor(self, teacher_id: int, item_i: int, item_j: int) -> np.ndarray:
        """Clipped logits beta_m * ((U_i - U_j) + b_m), broadcastable to full shape."""
        beta_m = self.beta[teacher_id]
        bias_axis = self.n_items + teacher_id

        u_i_vals = self.axis_grids[item_i].reshape(self._broadcast_shape(item_i))
        u_j_vals = self.axis_grids[item_j].reshape(self._broadcast_shape(item_j))
        b_vals = self.axis_grids[bias_axis].reshape(self._broadcast_shape(bias_axis))

        logits = beta_m * ((u_i_vals - u_j_vals) + b_vals)
        return np.clip(logits, -30.0, 30.0)  # guard against exp overflow

    def _log_choice_likelihood(self, teacher_id: int, item_i: int, item_j: int, choice: int) -> np.ndarray:
        logits = self._logits_tensor(teacher_id, item_i, item_j)
        log_p_i_over_j = -np.log1p(np.exp(-logits))
        log_p_j_over_i = -np.log1p(np.exp(logits))
        return choice * log_p_i_over_j + (1 - choice) * log_p_j_over_i

    def posterior(self, log_tensor: np.ndarray | None = None) -> np.ndarray:
        """Normalized posterior probability tensor (defaults to current state)."""
        if log_tensor is None:
            log_tensor = self.log_posterior
        return np.exp(log_tensor - logsumexp(log_tensor))

    # -- inference API --------------------------------------------------- #

    def predict_choice_prob(self, teacher_id: int, item_i: int, item_j: int) -> float:
        """Posterior-predictive P(item_i > item_j) under current beliefs."""
        logits = self._logits_tensor(teacher_id, item_i, item_j)
        p_i_over_j = 1.0 / (1.0 + np.exp(-logits))
        return float(np.sum(self.posterior() * p_i_over_j))

    def simulate_update(self, teacher_id: int, item_i: int, item_j: int, choice: int) -> np.ndarray:
        """Pure (non-mutating) Bayes update: returns the resulting normalized log-posterior."""
        log_likelihood = self._log_choice_likelihood(teacher_id, item_i, item_j, choice)
        new_log_posterior = self.log_posterior + log_likelihood
        return new_log_posterior - logsumexp(new_log_posterior)

    def update(self, teacher_id: int, item_i: int, item_j: int, choice: int) -> None:
        """Commit a Bayesian update given an observed pairwise comparison."""
        self.log_posterior = self.simulate_update(teacher_id, item_i, item_j, choice)
        self.n_updates += 1

    def get_estimates(self, log_tensor: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Posterior marginal means for U (all items) and b (all teachers)."""
        posterior = self.posterior(log_tensor)

        u_est = np.zeros(self.n_items)
        for k in range(self.n_items):
            sum_axes = tuple(a for a in range(self.n_axes) if a != k)
            marginal = posterior.sum(axis=sum_axes)
            u_est[k] = np.dot(marginal, self.axis_grids[k])

        b_est = np.zeros(self.n_teachers)
        for m in range(self.n_teachers):
            axis = self.n_items + m
            sum_axes = tuple(a for a in range(self.n_axes) if a != axis)
            marginal = posterior.sum(axis=sum_axes)
            b_est[m] = np.dot(marginal, self.axis_grids[axis])

        return u_est, b_est

    def _variances_from_posterior(self, posterior: np.ndarray) -> tuple[float, float]:
        """Sum of item-utility marginal variances and of teacher-bias marginal
        variances, computed from an already-normalized posterior tensor.

        Factored out so the acquisition loop can pass in a posterior it has
        already normalized itself, instead of paying for a second full-tensor
        renormalization on every hypothetical outcome it evaluates.
        """
        var_u = 0.0
        for k in range(self.n_items):
            sum_axes = tuple(a for a in range(self.n_axes) if a != k)
            marginal = posterior.sum(axis=sum_axes)
            mean = np.dot(marginal, self.axis_grids[k])
            var_u += np.dot(marginal, (self.axis_grids[k] - mean) ** 2)

        var_b = 0.0
        for m in range(self.n_teachers):
            axis = self.n_items + m
            sum_axes = tuple(a for a in range(self.n_axes) if a != axis)
            marginal = posterior.sum(axis=sum_axes)
            mean = np.dot(marginal, self.axis_grids[axis])
            var_b += np.dot(marginal, (self.axis_grids[axis] - mean) ** 2)

        return float(var_u), float(var_b)

    def total_variances(self, log_tensor: np.ndarray | None = None) -> tuple[float, float]:
        """(Sum of item-utility variances, sum of teacher-bias variances) -- our joint VOI target."""
        return self._variances_from_posterior(self.posterior(log_tensor))

    def total_utility_variance(self, log_tensor: np.ndarray | None = None) -> float:
        """Sum of posterior marginal variances over all items only (legacy U-only accessor)."""
        return self.total_variances(log_tensor)[0]


# --------------------------------------------------------------------------- #
# 2. Active selection policies
# --------------------------------------------------------------------------- #


def _expected_variance_reduction_joint(
    filt: AnchoredBayesianBiasFilter,
    current_posterior: np.ndarray,
    teacher_id: int,
    item_i: int,
    item_j: int,
) -> tuple[float, float]:
    """Expected reduction in total posterior variance of U and of b from one query.

    Delta_Var_U(query) = Var_now[U] - E_{choice ~ predictive}[ Var_post[U | choice] ]
    Delta_Var_b(query) = Var_now[b] - E_{choice ~ predictive}[ Var_post[b | choice] ]

    Both are computed exactly (not sampled), weighting the two possible
    binary outcomes by their posterior-predictive probabilities. The two
    hypothetical posteriors (choice=1, choice=0) are each built from a
    single shared logits evaluation and reused for both the predictive
    probability and the variance calculation, rather than recomputing the
    logits/posterior from scratch per quantity -- this matters because this
    function is called ``len(teachers) * len(candidates)`` times per round.

    ``current_posterior`` (the caller's already-materialized ``filt.posterior()``)
    is passed in rather than recomputed here, since it is identical across
    every candidate evaluated within one call to the selection rule.
    """
    logits = filt._logits_tensor(teacher_id, item_i, item_j)
    log_p_i_over_j = -np.log1p(np.exp(-logits))
    log_p_j_over_i = -np.log1p(np.exp(logits))
    p_i_over_j = float(np.sum(current_posterior * np.exp(log_p_i_over_j)))
    p_j_over_i = 1.0 - p_i_over_j

    log_post_if_1 = filt.log_posterior + log_p_i_over_j
    log_post_if_1 = log_post_if_1 - logsumexp(log_post_if_1)
    log_post_if_0 = filt.log_posterior + log_p_j_over_i
    log_post_if_0 = log_post_if_0 - logsumexp(log_post_if_0)

    var_u_if_1, var_b_if_1 = filt._variances_from_posterior(np.exp(log_post_if_1))
    var_u_if_0, var_b_if_0 = filt._variances_from_posterior(np.exp(log_post_if_0))

    var_now_u, var_now_b = filt._variances_from_posterior(current_posterior)
    expected_var_u = p_i_over_j * var_u_if_1 + p_j_over_i * var_u_if_0
    expected_var_b = p_i_over_j * var_b_if_1 + p_j_over_i * var_b_if_0

    return var_now_u - expected_var_u, var_now_b - expected_var_b


def _select_by_voi_per_cost(
    filt: AnchoredBayesianBiasFilter,
    teachers: Sequence[TeacherSpec],
    candidates: Sequence[tuple[int, int]],
    lambda_b: float = 0.0,
) -> tuple[int, int, int]:
    """Shared active-selection rule: argmax over (teacher, i, j) of joint VOI / cost.

        VOI_Joint = (Delta_Var_U + lambda_b * Delta_Var_b) / cost_m

    ``lambda_b=0.0`` (the default) recovers the original myopic, U-only VOI
    rule used by Freedman-style ATS. ``lambda_b>0`` additionally rewards
    queries expected to shrink uncertainty about teacher bias, which is
    what lets the learner justify spending on the expensive honest teacher
    purely to pin down bias, even in rounds where doing so would not be the
    single best move for reducing Var[U] alone.
    """
    current_posterior = filt.posterior()  # identical for every candidate below; compute once

    best_query: tuple[int, int, int] | None = None
    best_roi = -np.inf

    for teacher_id, spec in enumerate(teachers):
        for item_i, item_j in candidates:
            delta_var_u, delta_var_b = _expected_variance_reduction_joint(
                filt, current_posterior, teacher_id, item_i, item_j
            )
            roi = (delta_var_u + lambda_b * delta_var_b) / spec.cost
            if roi > best_roi:
                best_roi = roi
                best_query = (teacher_id, item_i, item_j)

    assert best_query is not None
    return best_query


def select_query_biased_ats(
    filt: AnchoredBayesianBiasFilter,
    teachers: Sequence[TeacherSpec],
    candidates: Sequence[tuple[int, int]],
    lambda_b: float = 0.0,
) -> tuple[int, int, int]:
    """Myopic, U-only active selection for the bias-aware learner (the "Original" rule).

    Picks the (teacher_id, item_i, item_j) query maximizing expected
    posterior-variance reduction on U alone per unit query cost, using a
    filter that jointly models teacher bias. Because the filter can
    "explain away" a cheap teacher's systematic bias rather than being
    misled by it, this policy is free to exploit cheap-but-biased
    teachers once their bias is well estimated -- but because the
    acquisition target never rewards resolving bias for its own sake
    (``lambda_b=0`` by default), it tends to under-explore the expensive
    honest teacher, leaving that teacher's own bias poorly estimated. See
    :func:`select_query_joint_biased_ats` for the fix.
    """
    return _select_by_voi_per_cost(filt, teachers, candidates, lambda_b=lambda_b)


def select_query_joint_biased_ats(
    filt: AnchoredBayesianBiasFilter,
    teachers: Sequence[TeacherSpec],
    candidates: Sequence[tuple[int, int]],
    lambda_b: float = 1.0,
) -> tuple[int, int, int]:
    """Joint (U, b) active selection for the bias-aware learner (the "Proposed Fix").

    Identical machinery to :func:`select_query_biased_ats`, but the
    acquisition score also credits expected reduction in posterior bias
    variance, weighted by ``lambda_b``:

        VOI_Joint = (Delta_Var_U + lambda_b * Delta_Var_b) / cost_m

    With ``lambda_b=1.0``, a query that would mostly resolve lingering
    uncertainty about a teacher's bias (e.g. a cheap query to the honest
    teacher whose true bias is already well pinned near 0, versus one to
    a still-uncertain teacher) can now win out over a query that offers
    slightly higher Delta_Var_U alone -- directly counteracting the
    honest-teacher under-exploration pathology of the U-only rule.
    """
    return _select_by_voi_per_cost(filt, teachers, candidates, lambda_b=lambda_b)


def select_query_standard_ats(
    filt_unbiased_assumption: AnchoredBayesianBiasFilter,
    teachers: Sequence[TeacherSpec],
    candidates: Sequence[tuple[int, int]],
) -> tuple[int, int, int]:
    """Active selection for Freedman et al.'s original (unbiased-teacher) ATS.

    Identical VOI/cost acquisition rule, but ``filt_unbiased_assumption``
    must be constructed with ``assume_unbiased=True`` so that every
    teacher's bias is permanently pinned to 0. The resulting queries are
    chosen "as if" no teacher could be biased -- any systematic bias in
    the data is silently absorbed into the utility estimates instead.
    (``lambda_b`` is irrelevant here: Delta_Var_b is identically 0 when
    the bias grid is collapsed to a single point, so this always reduces
    to the plain U-only rule regardless.)
    """
    if not filt_unbiased_assumption.assume_unbiased:
        raise ValueError(
            "select_query_standard_ats expects a filter constructed with "
            "assume_unbiased=True (Freedman et al.'s original assumption)."
        )
    return _select_by_voi_per_cost(filt_unbiased_assumption, teachers, candidates, lambda_b=0.0)


def select_query_random(
    filt: AnchoredBayesianBiasFilter,
    teachers: Sequence[TeacherSpec],
    candidates: Sequence[tuple[int, int]],
) -> tuple[int, int, int]:
    """Uniformly random (teacher, item_i, item_j) query -- the passive baseline."""
    teacher_id = int(np.random.randint(len(teachers)))
    item_i, item_j = candidates[np.random.randint(len(candidates))]
    return teacher_id, item_i, item_j


QueryPolicy = Callable[[AnchoredBayesianBiasFilter, Sequence[TeacherSpec], Sequence[tuple[int, int]]], tuple[int, int, int]]


# --------------------------------------------------------------------------- #
# 3. Benchmark simulation
# --------------------------------------------------------------------------- #


class AlgorithmTrace:
    """Per-timestep metric history for one algorithm run."""

    def __init__(self, name: str, n_steps: int, n_teachers: int = 0) -> None:
        self.name = name
        self.utility_sse = np.zeros(n_steps)  # ||True_U - Est_U||^2 per timestep
        self.bias_mae = np.zeros(n_steps)  # mean_m |True_b_m - Est_b_m| per timestep
        self.b_est_history = np.zeros((n_steps, n_teachers))  # per-teacher b estimate per timestep
        self.regret_per_step = np.zeros(n_steps)
        self.cumulative_regret = np.zeros(n_steps)
        self.teacher_choice_counts = None  # filled in after the run
        self.final_u_est: np.ndarray | None = None
        self.final_b_est: np.ndarray | None = None


def run_algorithm(
    name: str,
    env: BiasedTeacherEnv,
    teacher_specs: Sequence[TeacherSpec],
    true_U: np.ndarray,
    true_b: np.ndarray,
    filt: AnchoredBayesianBiasFilter,
    query_policy: QueryPolicy,
    candidates: Sequence[tuple[int, int]],
    n_steps: int,
    seed: int,
) -> AlgorithmTrace:
    """Simulate one algorithm for ``n_steps`` rounds against a shared environment.

    Each round: (1) select a query via ``query_policy``, (2) observe the
    teacher's noisy biased response, (3) update ``filt``, (4) "exploit"
    the arm the current posterior mean judges best, and (5) accrue
    regret = (optimal arm utility - exploited arm utility) + query cost.
    """
    np.random.seed(seed)

    n_teachers = len(teacher_specs)
    trace = AlgorithmTrace(name, n_steps, n_teachers=n_teachers)
    optimal_utility = float(np.max(true_U))
    teacher_query_counts = np.zeros(n_teachers, dtype=int)

    cumulative = 0.0
    for t in range(n_steps):
        teacher_id, item_i, item_j = query_policy(filt, teacher_specs, candidates)
        teacher_query_counts[teacher_id] += 1

        choice = env.sample_response(teacher_id, item_i, item_j)
        filt.update(teacher_id, item_i, item_j, choice)

        u_est, b_est = filt.get_estimates()

        exploited_arm = int(np.argmax(u_est))
        exploited_utility = float(true_U[exploited_arm])
        cost_paid = teacher_specs[teacher_id].cost

        regret_t = (optimal_utility - exploited_utility) + cost_paid
        cumulative += regret_t

        trace.utility_sse[t] = float(np.sum((true_U - u_est) ** 2))
        trace.bias_mae[t] = float(np.mean(np.abs(true_b - b_est)))
        trace.b_est_history[t] = b_est
        trace.regret_per_step[t] = regret_t
        trace.cumulative_regret[t] = cumulative

    trace.final_u_est, trace.final_b_est = filt.get_estimates()
    trace.teacher_choice_counts = teacher_query_counts
    return trace


# --------------------------------------------------------------------------- #
# 4. Reporting and plotting
# --------------------------------------------------------------------------- #


def _print_summary_table(traces: Sequence[AlgorithmTrace], true_U: np.ndarray, true_b: np.ndarray) -> None:
    print("\n" + "=" * 88)
    print("BENCHMARK SUMMARY (final timestep)")
    print("=" * 88)
    header = f"{'Algorithm':<22}{'Final Regret':>14}{'Final SSE(U)':>15}{'Final MAE(b)':>15}{'Best Arm':>12}"
    print(header)
    print("-" * 88)

    true_best_arm = int(np.argmax(true_U))
    for trace in traces:
        est_best_arm = int(np.argmax(trace.final_u_est))
        arm_flag = f"{est_best_arm}{'  (correct)' if est_best_arm == true_best_arm else '  (WRONG)'}"
        print(
            f"{trace.name:<22}"
            f"{trace.cumulative_regret[-1]:>14.3f}"
            f"{trace.utility_sse[-1]:>15.4f}"
            f"{trace.bias_mae[-1]:>15.4f}"
            f"{arm_flag:>12}"
        )

    print("-" * 88)
    print(f"True optimal arm: item {true_best_arm} (U = {true_U[true_best_arm]:.2f})")
    for trace in traces:
        counts = trace.teacher_choice_counts
        counts_str = ", ".join(f"teacher {m}: {c}" for m, c in enumerate(counts))
        print(f"{trace.name:<22} query allocation -> {counts_str}")
    print("=" * 88)


def _plot_results(traces: Sequence[AlgorithmTrace], n_steps: int, out_path: str) -> None:
    timesteps = np.arange(1, n_steps + 1)
    colors = {"Biased-ATS": "#1f77b4", "Standard-ATS": "#d62728", "Random-Biased-ATS": "#2ca02c"}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for trace in traces:
        color = colors.get(trace.name)
        axes[0].plot(timesteps, trace.utility_sse, label=trace.name, color=color, linewidth=1.8)
        axes[1].plot(timesteps, trace.bias_mae, label=trace.name, color=color, linewidth=1.8)
        axes[2].plot(timesteps, trace.cumulative_regret, label=trace.name, color=color, linewidth=1.8)

    axes[0].set_title("Utility Reconstruction Error")
    axes[0].set_xlabel("Timestep")
    axes[0].set_ylabel(r"$\|\,\mathrm{True}_U - \mathrm{Est}_U\,\|^2$")

    axes[1].set_title("Bias Estimation Error")
    axes[1].set_xlabel("Timestep")
    axes[1].set_ylabel(r"mean$_m\,|\,\mathrm{True}_b - \mathrm{Est}_b\,|$")

    axes[2].set_title("Cumulative Regret")
    axes[2].set_xlabel("Timestep")
    axes[2].set_ylabel("Cumulative (arm regret + query cost)")

    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=9)

    fig.suptitle("Biased-ATS vs. Standard-ATS vs. Random-Biased-ATS on a Biased-Teacher HUB", fontsize=13)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 5. Acquisition-rule comparison: myopic U-only VOI vs. joint (U, b) VOI
#
# Follow-up experiment to the finding above: a pure Var[U] acquisition rule
# starves the expensive honest teacher of queries (6-9 out of 150), which is
# enough to get the item ranking right most of the time but never enough to
# pin down that teacher's own bias is 0. Adding lambda_b * Delta_Var_b to the
# acquisition score gives the policy a direct incentive to query whichever
# teacher still has uncertain bias, not just whichever query best shrinks
# Var[U] this instant.
# --------------------------------------------------------------------------- #


ACQUISITION_SEEDS: tuple[int, ...] = (1, 11, 42, 123, 777)


def run_acquisition_comparison(
    env: BiasedTeacherEnv,
    teacher_specs: Sequence[TeacherSpec],
    true_U: np.ndarray,
    true_b: np.ndarray,
    candidates: Sequence[tuple[int, int]],
    u_grid: np.ndarray,
    b_grid: np.ndarray,
    n_steps: int,
    seeds: Sequence[int] = ACQUISITION_SEEDS,
) -> dict[str, list[AlgorithmTrace]]:
    """Run both acquisition rules across every seed on identical filter setups.

    Both methods use the same bias-aware, anchored filter configuration
    (``assume_unbiased=False``); they differ only in which query-selection
    function chooses (teacher, item_i, item_j) each round.

    Returns
    -------
    dict mapping method name -> list of one :class:`AlgorithmTrace` per seed.
    """
    n_items = true_U.shape[0]
    n_teachers = len(teacher_specs)
    teacher_betas = [spec.beta for spec in teacher_specs]

    methods: dict[str, QueryPolicy] = {
        "Myopic U-Only VOI (Original)": select_query_biased_ats,
        "Joint (U,b) VOI (Proposed Fix)": select_query_joint_biased_ats,
    }

    results: dict[str, list[AlgorithmTrace]] = {name: [] for name in methods}

    for method_name, policy in methods.items():
        print(f"\nRunning acquisition mode: {method_name} ...")
        for seed in seeds:
            filt = AnchoredBayesianBiasFilter(
                n_items, n_teachers, teacher_betas, u_grid=u_grid, b_grid=b_grid,
                anchor_item=0, assume_unbiased=False,
            )
            trace = run_algorithm(
                f"{method_name} (seed={seed})", env, teacher_specs, true_U, true_b,
                filt, policy, candidates, n_steps, seed=seed,
            )
            results[method_name].append(trace)
            print(f"  seed={seed:4d}  teacher queries={trace.teacher_choice_counts.tolist()}  done.")

    return results


def _print_acquisition_comparison_tables(
    results: dict[str, list[AlgorithmTrace]],
    seeds: Sequence[int],
    true_U: np.ndarray,
) -> None:
    true_best_arm = int(np.argmax(true_U))

    for method_name, traces in results.items():
        print("\n" + "=" * 100)
        print(f"ACQUISITION MODE: {method_name}")
        print("=" * 100)
        header = (
            f"{'Seed':>6}  {'Teacher Queries [Honest, Biased]':>34}  "
            f"{'Teacher-0 Bias Est. (True=0)':>29}  {'Utility MSE':>12}  {'Best Arm Correct?':>18}"
        )
        print(header)
        print("-" * 100)

        mse_values = []
        bias0_values = []
        n_correct = 0
        for seed, trace in zip(seeds, traces):
            counts = trace.teacher_choice_counts.tolist()
            bias0 = float(trace.final_b_est[0])
            mse = float(trace.utility_sse[-1])
            correct = int(np.argmax(trace.final_u_est)) == true_best_arm
            n_correct += int(correct)
            mse_values.append(mse)
            bias0_values.append(bias0)

            print(
                f"{seed:>6}  {str(counts):>34}  {bias0:>29.3f}  {mse:>12.4f}  "
                f"{'Y' if correct else 'N':>18}"
            )

        print("-" * 100)
        print(
            f"{'Mean':>6}  {'':>34}  {np.mean(bias0_values):>29.3f}  "
            f"{np.mean(mse_values):>12.4f}  {f'{n_correct}/{len(seeds)} correct':>18}"
        )

    print("=" * 100)


def _plot_acquisition_comparison(
    results: dict[str, list[AlgorithmTrace]], n_steps: int, out_path: str
) -> None:
    """Save mean +/- std trajectories (over seeds) of teacher-0 bias and utility MSE."""
    timesteps = np.arange(1, n_steps + 1)
    colors = {
        "Myopic U-Only VOI (Original)": "#d62728",
        "Joint (U,b) VOI (Proposed Fix)": "#1f77b4",
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for method_name, traces in results.items():
        color = colors.get(method_name)
        bias0_stack = np.stack([tr.b_est_history[:, 0] for tr in traces])  # (n_seeds, n_steps)
        mse_stack = np.stack([tr.utility_sse for tr in traces])

        bias0_mean = bias0_stack.mean(axis=0)
        bias0_std = bias0_stack.std(axis=0)
        mse_mean = mse_stack.mean(axis=0)
        mse_std = mse_stack.std(axis=0)

        axes[0].plot(timesteps, bias0_mean, label=method_name, color=color, linewidth=2.0)
        axes[0].fill_between(timesteps, bias0_mean - bias0_std, bias0_mean + bias0_std, color=color, alpha=0.15)

        axes[1].plot(timesteps, mse_mean, label=method_name, color=color, linewidth=2.0)
        axes[1].fill_between(
            timesteps, np.maximum(mse_mean - mse_std, 0.0), mse_mean + mse_std, color=color, alpha=0.15
        )

    axes[0].axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.6, label="True bias (= 0)")
    axes[0].set_title("Teacher-0 Bias Estimate (mean $\\pm$ std over seeds)")
    axes[0].set_xlabel("Timestep")
    axes[0].set_ylabel(r"$\hat{b}_0$   (true $b_0 = 0$)")

    axes[1].set_title("Utility Reconstruction Error (mean $\\pm$ std over seeds)")
    axes[1].set_xlabel("Timestep")
    axes[1].set_ylabel(r"$\|\,\mathrm{True}_U - \mathrm{Est}_U\,\|^2$")

    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=9)

    n_seeds = len(next(iter(results.values())))
    fig.suptitle(f"Myopic U-Only VOI vs. Joint (U,b) VOI -- {n_seeds} seeds each", fontsize=13)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 6. Main
# --------------------------------------------------------------------------- #


def main() -> None:
    """Run the 3-algorithm benchmark, then the acquisition-rule comparison, end to end."""
    true_U = np.array([0.0, 6.0, 2.0, 8.0])  # Item 3 is best.
    n_items = true_U.shape[0]

    teacher_specs = [
        TeacherSpec(beta=2.0, true_bias=0.0, cost=5.0, name="Honest Expert"),
        TeacherSpec(beta=1.5, true_bias=3.5, cost=1.0, name="Cheap Biased Annotator (favors item 0)"),
    ]
    n_teachers = len(teacher_specs)
    true_b = np.array([spec.true_bias for spec in teacher_specs])

    env = BiasedTeacherEnv(true_U=true_U, teachers=teacher_specs)

    candidates = list(itertools.combinations(range(n_items), 2))
    n_steps = 150
    teacher_betas = [spec.beta for spec in teacher_specs]

    u_grid = np.linspace(-10.0, 10.0, 15)
    b_grid = np.linspace(-5.0, 5.0, 11)

    print("=" * 88)
    print("Biased-Teacher HUB: Active Selection Benchmark")
    print("=" * 88)
    print(f"True U : {true_U.tolist()}  (best item = {int(np.argmax(true_U))})")
    for m, spec in enumerate(teacher_specs):
        print(f"Teacher {m} ({spec.name}): beta={spec.beta}, true_bias={spec.true_bias:+.2f}, cost={spec.cost}")
    print(f"Rounds per algorithm: {n_steps}\n")

    # ------------------------------------------------------------------ #
    # Algorithm A: Biased-ATS (active selection + bias-aware filter)
    # ------------------------------------------------------------------ #
    print("Running Algorithm A: Biased-ATS ...")
    filt_biased = AnchoredBayesianBiasFilter(
        n_items, n_teachers, teacher_betas, u_grid=u_grid, b_grid=b_grid, anchor_item=0, assume_unbiased=False
    )
    trace_biased = run_algorithm(
        "Biased-ATS", env, teacher_specs, true_U, true_b, filt_biased,
        select_query_biased_ats, candidates, n_steps, seed=1,
    )

    # ------------------------------------------------------------------ #
    # Algorithm B: Standard-ATS (active selection + naive unbiased filter)
    # ------------------------------------------------------------------ #
    print("Running Algorithm B: Standard-ATS ...")
    filt_standard = AnchoredBayesianBiasFilter(
        n_items, n_teachers, teacher_betas, u_grid=u_grid, b_grid=b_grid, anchor_item=0, assume_unbiased=True
    )
    trace_standard = run_algorithm(
        "Standard-ATS", env, teacher_specs, true_U, true_b, filt_standard,
        select_query_standard_ats, candidates, n_steps, seed=2,
    )

    # ------------------------------------------------------------------ #
    # Algorithm C: Random-Biased-ATS (random selection + bias-aware filter)
    # ------------------------------------------------------------------ #
    print("Running Algorithm C: Random-Biased-ATS ...")
    filt_random = AnchoredBayesianBiasFilter(
        n_items, n_teachers, teacher_betas, u_grid=u_grid, b_grid=b_grid, anchor_item=0, assume_unbiased=False
    )
    trace_random = run_algorithm(
        "Random-Biased-ATS", env, teacher_specs, true_U, true_b, filt_random,
        select_query_random, candidates, n_steps, seed=3,
    )

    traces = [trace_biased, trace_standard, trace_random]

    _print_summary_table(traces, true_U, true_b)

    out_path = "benchmark_results.png"
    _plot_results(traces, n_steps, out_path)
    print(f"\nSaved plots to: {out_path}")

    # ------------------------------------------------------------------ #
    # Acquisition-rule comparison: myopic U-only VOI vs. joint (U,b) VOI
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 88)
    print("ACQUISITION RULE COMPARISON: Myopic U-Only VOI vs. Joint (U,b) VOI")
    print("=" * 88)
    print(f"Seeds: {list(ACQUISITION_SEEDS)}")

    comparison_results = run_acquisition_comparison(
        env, teacher_specs, true_U, true_b, candidates, u_grid, b_grid, n_steps, seeds=ACQUISITION_SEEDS
    )
    _print_acquisition_comparison_tables(comparison_results, ACQUISITION_SEEDS, true_U)

    comparison_out_path = "joint_voi_benchmark.png"
    _plot_acquisition_comparison(comparison_results, n_steps, comparison_out_path)
    print(f"\nSaved acquisition-rule comparison plot to: {comparison_out_path}")


if __name__ == "__main__":
    main()
