"""
biased_teacher_hub.py
======================

Extension of the "Active Teacher Selection" (ATS) framework
(Freedman et al., 2023) to the setting of *biased* teachers in a
Hidden Utility Bandit (HUB).

Background
----------
In the original HUB / ATS formulation, a learner queries one of several
teachers for pairwise preferences over items whose utilities U are
hidden. Each teacher m answers according to a Bradley-Terry-style noisy
oracle governed by a *known* rationality parameter beta_m:

    P(i > j | U, beta_m) = sigmoid(beta_m * (U_i - U_j))

This script extends that oracle with an *unknown* additive bias term
b_m that systematically favors (or disfavors) certain comparisons:

    P(i > j | U, beta_m, b_m) = sigmoid(beta_m * ((U_i - U_j) + b_m))

The learner knows each teacher's rationality beta_m and query cost
f_m, but must jointly infer:
    1. The true hidden utility vector U (shared across all teachers).
    2. Each teacher's unknown scalar bias b_m.

We solve this joint inference problem with a discrete-grid Bayesian
filter that maintains a joint posterior tensor over (U_0, ..., U_{N-1},
b_0, ..., b_{M-1}) and updates it in log-space after every observed
pairwise comparison.

Dependencies: NumPy and SciPy only.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.special import logsumexp


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #


@dataclass
class TeacherSpec:
    """Static properties of a single (possibly biased) teacher.

    Attributes
    ----------
    beta : float
        Rationality / inverse-noise parameter. Known to the learner.
    true_bias : float
        Ground-truth additive bias b_m. Hidden from the learner; the
        environment uses it to generate responses, and the Bayesian
        filter tries to recover it.
    cost : float
        Cost f_m of querying this teacher once. Not used by the simple
        round-robin demo below, but carried through so that this
        environment plugs directly into a downstream active teacher
        selection (ATS) policy that trades off information gain against
        query cost.
    name : str
        Human-readable label for logging.
    """

    beta: float
    true_bias: float
    cost: float
    name: str = ""


class BiasedTeacherEnv:
    """Simulates a Hidden Utility Bandit with biased pairwise teachers.

    The environment owns the ground-truth utility vector ``U`` and a
    roster of teachers, each with a known rationality ``beta_m`` and a
    hidden bias ``b_m``. Queries take the form "does item i beat item
    j?" and are answered stochastically according to

        P(i > j | U, beta_m, b_m) = 1 / (1 + exp(-beta_m * ((U_i - U_j) + b_m)))
    """

    def __init__(self, true_U: Sequence[float], teachers: Sequence[TeacherSpec]) -> None:
        """
        Parameters
        ----------
        true_U : Sequence[float]
            Ground-truth hidden utility for each of the N items.
        teachers : Sequence[TeacherSpec]
            One entry per teacher, in teacher-id order.
        """
        self.true_U = np.asarray(true_U, dtype=np.float64)
        self.teachers = list(teachers)
        self.n_items = self.true_U.shape[0]
        self.n_teachers = len(self.teachers)

    def preference_probability(self, teacher_id: int, item_i: int, item_j: int) -> float:
        """Return P(i > j) under the ground-truth biased preference model."""
        spec = self.teachers[teacher_id]
        logit = spec.beta * ((self.true_U[item_i] - self.true_U[item_j]) + spec.true_bias)
        logit = np.clip(logit, -30.0, 30.0)  # guard against exp overflow
        return 1.0 / (1.0 + np.exp(-logit))

    def sample_response(self, teacher_id: int, item_i: int, item_j: int) -> int:
        """Query ``teacher_id`` to compare ``item_i`` against ``item_j``.

        Returns
        -------
        int
            1 if the teacher reports "item_i > item_j", else 0.
        """
        p_i_over_j = self.preference_probability(teacher_id, item_i, item_j)
        return 1 if np.random.rand() < p_i_over_j else 0


# --------------------------------------------------------------------------- #
# Bayesian inference over utilities and biases
# --------------------------------------------------------------------------- #


class BayesianBiasFilter:
    """Discrete-grid joint Bayesian filter over hidden utilities and teacher biases.

    Maintains a single joint log-posterior tensor over the Cartesian
    product of per-item utility grids and per-teacher bias grids:

        log_posterior[u_0, u_1, ..., u_{N-1}, b_0, ..., b_{M-1}]

    Every pairwise observation updates the *entire* tensor via Bayes'
    rule in log-space (log-prior + log-likelihood, renormalized with a
    numerically stable log-sum-exp), which correctly propagates
    correlations between utility estimates and bias estimates (e.g. a
    teacher's inferred bias is discounted once the utility gap it
    "explains" is otherwise well supported by unbiased teachers).
    """

    def __init__(
        self,
        n_items: int,
        n_teachers: int,
        teacher_betas: Sequence[float],
        u_grid: np.ndarray | None = None,
        b_grid: np.ndarray | None = None,
    ) -> None:
        """
        Parameters
        ----------
        n_items : int
            Number N of items with hidden utility.
        n_teachers : int
            Number M of teachers.
        teacher_betas : Sequence[float]
            Known rationality parameter beta_m for each teacher.
        u_grid : np.ndarray, optional
            Candidate discrete values for each item's utility.
            Defaults to ``np.linspace(-10, 10, 21)``.
        b_grid : np.ndarray, optional
            Candidate discrete values for each teacher's bias.
            Defaults to ``np.linspace(-5, 5, 21)``.
        """
        self.n_items = n_items
        self.n_teachers = n_teachers
        self.beta = np.asarray(teacher_betas, dtype=np.float64)
        if self.beta.shape[0] != n_teachers:
            raise ValueError("teacher_betas must have length n_teachers")

        self.u_grid = np.linspace(-10.0, 10.0, 21) if u_grid is None else np.asarray(u_grid, dtype=np.float64)
        self.b_grid = np.linspace(-5.0, 5.0, 21) if b_grid is None else np.asarray(b_grid, dtype=np.float64)

        self.n_u = self.u_grid.shape[0]
        self.n_b = self.b_grid.shape[0]
        self.n_axes = n_items + n_teachers

        # Axis layout of the joint tensor:
        #   axes [0, n_items)               -> per-item utility grid index
        #   axes [n_items, n_items+n_teachers) -> per-teacher bias grid index
        tensor_shape = (self.n_u,) * n_items + (self.n_b,) * n_teachers

        # Uniform prior in log-space.
        log_prior = np.zeros(tensor_shape, dtype=np.float64)
        self.log_posterior = log_prior - logsumexp(log_prior)

        # Track number of updates for logging / diagnostics.
        self.n_updates = 0

    def _broadcast_shape(self, axis: int, size: int) -> tuple[int, ...]:
        """Return a tensor shape of all 1's except ``size`` at ``axis``."""
        shape = [1] * self.n_axes
        shape[axis] = size
        return tuple(shape)

    def update(self, teacher_id: int, item_i: int, item_j: int, choice: int) -> None:
        """Bayesian update given one observed pairwise comparison.

        Parameters
        ----------
        teacher_id : int
            Index of the teacher who answered.
        item_i, item_j : int
            The two items that were compared.
        choice : int
            1 if the teacher reported "item_i > item_j", 0 otherwise.
        """
        beta_m = self.beta[teacher_id]
        bias_axis = self.n_items + teacher_id

        u_i_vals = self.u_grid.reshape(self._broadcast_shape(item_i, self.n_u))
        u_j_vals = self.u_grid.reshape(self._broadcast_shape(item_j, self.n_u))
        b_vals = self.b_grid.reshape(self._broadcast_shape(bias_axis, self.n_b))

        # Broadcasts to a full (n_u,)*N + (n_b,)*M tensor of logits, one
        # entry per hypothesis (U_0, ..., U_{N-1}, b_0, ..., b_{M-1}).
        logits = beta_m * ((u_i_vals - u_j_vals) + b_vals)
        logits = np.clip(logits, -30.0, 30.0)  # prevent exp overflow below

        # Numerically stable log-sigmoid / log(1 - sigmoid) via softplus.
        log_p_i_over_j = -np.log1p(np.exp(-logits))
        log_p_j_over_i = -np.log1p(np.exp(logits))

        log_likelihood = choice * log_p_i_over_j + (1 - choice) * log_p_j_over_i

        self.log_posterior = self.log_posterior + log_likelihood
        self.log_posterior -= logsumexp(self.log_posterior)  # renormalize
        self.n_updates += 1

    def get_estimates(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute posterior marginal means for U and for teacher biases.

        Returns
        -------
        U_est : np.ndarray, shape (n_items,)
            Posterior expectation E[U_k] for each item k.
        b_est : np.ndarray, shape (n_teachers,)
            Posterior expectation E[b_m] for each teacher m.
        """
        posterior = np.exp(self.log_posterior - logsumexp(self.log_posterior))

        u_est = np.zeros(self.n_items)
        for k in range(self.n_items):
            sum_axes = tuple(a for a in range(self.n_axes) if a != k)
            marginal = posterior.sum(axis=sum_axes)
            u_est[k] = np.dot(marginal, self.u_grid)

        b_est = np.zeros(self.n_teachers)
        for m in range(self.n_teachers):
            axis = self.n_items + m
            sum_axes = tuple(a for a in range(self.n_axes) if a != axis)
            marginal = posterior.sum(axis=sum_axes)
            b_est[m] = np.dot(marginal, self.b_grid)

        return u_est, b_est

    def get_marginal_std(self) -> tuple[np.ndarray, np.ndarray]:
        """Posterior marginal standard deviations for U and teacher biases.

        Useful as an uncertainty signal for a downstream active teacher
        selection policy (query the teacher/pair expected to shrink
        posterior variance the most, discounted by query cost).
        """
        posterior = np.exp(self.log_posterior - logsumexp(self.log_posterior))

        u_std = np.zeros(self.n_items)
        for k in range(self.n_items):
            sum_axes = tuple(a for a in range(self.n_axes) if a != k)
            marginal = posterior.sum(axis=sum_axes)
            mean = np.dot(marginal, self.u_grid)
            var = np.dot(marginal, (self.u_grid - mean) ** 2)
            u_std[k] = np.sqrt(max(var, 0.0))

        b_std = np.zeros(self.n_teachers)
        for m in range(self.n_teachers):
            axis = self.n_items + m
            sum_axes = tuple(a for a in range(self.n_axes) if a != axis)
            marginal = posterior.sum(axis=sum_axes)
            mean = np.dot(marginal, self.b_grid)
            var = np.dot(marginal, (self.b_grid - mean) ** 2)
            b_std[m] = np.sqrt(max(var, 0.0))

        return u_std, b_std


# --------------------------------------------------------------------------- #
# Experimental demo
# --------------------------------------------------------------------------- #


def _format_vector(values: np.ndarray, width: int = 7, precision: int = 3) -> str:
    return "[" + ", ".join(f"{v:{width}.{precision}f}" for v in values) + "]"


def _print_progress(step: int, filt: BayesianBiasFilter, true_U: np.ndarray, true_b: np.ndarray) -> None:
    u_est, b_est = filt.get_estimates()
    u_err = np.abs(u_est - true_U)
    b_err = np.abs(b_est - true_b)

    print(f"\n--- After {step} queries ---")
    print(f"  Estimated U : {_format_vector(u_est)}   |  True U : {_format_vector(true_U)}")
    print(f"  |U error|   : {_format_vector(u_err)}   |  RMSE(U): {np.sqrt(np.mean(u_err ** 2)):.4f}")
    print(f"  Estimated b : {_format_vector(b_est)}   |  True b : {_format_vector(true_b)}")
    print(f"  |b error|   : {_format_vector(b_err)}   |  RMSE(b): {np.sqrt(np.mean(b_err ** 2)):.4f}")


def main() -> None:
    """Run the biased-teacher HUB inference demo end to end."""
    rng_seed = 42
    np.random.seed(rng_seed)

    # ------------------------------------------------------------------ #
    # 1. Ground truth setup
    # ------------------------------------------------------------------ #
    true_U = np.array([2.0, 8.0, 4.0])  # Item 1 is the best item.
    n_items = true_U.shape[0]

    teacher_specs = [
        TeacherSpec(beta=2.0, true_bias=0.0, cost=5.0, name="Honest Expert"),
        TeacherSpec(beta=1.5, true_bias=3.5, cost=1.0, name="Biased Annotator (favors item 0)"),
    ]
    n_teachers = len(teacher_specs)
    true_b = np.array([spec.true_bias for spec in teacher_specs])

    env = BiasedTeacherEnv(true_U=true_U, teachers=teacher_specs)
    bias_filter = BayesianBiasFilter(
        n_items=n_items,
        n_teachers=n_teachers,
        teacher_betas=[spec.beta for spec in teacher_specs],
        u_grid=np.linspace(-10.0, 10.0, 21),
        b_grid=np.linspace(-5.0, 5.0, 21),
    )

    print("=" * 72)
    print("Biased Teacher Hidden Utility Bandit -- Bayesian Inference Demo")
    print("=" * 72)
    print(f"True U               : {_format_vector(true_U)}")
    for m, spec in enumerate(teacher_specs):
        print(
            f"Teacher {m} ({spec.name}): beta={spec.beta:.2f}, "
            f"true_bias={spec.true_bias:+.2f}, cost={spec.cost:.2f}"
        )
    print(f"Random seed          : {rng_seed}")

    # ------------------------------------------------------------------ #
    # 2. Simulate queries, round-robin across teachers
    # ------------------------------------------------------------------ #
    n_queries = 120
    report_every = 20
    all_item_pairs = list(itertools.permutations(range(n_items), 2))

    for step in range(1, n_queries + 1):
        teacher_id = (step - 1) % n_teachers  # alternate teachers evenly
        item_i, item_j = all_item_pairs[np.random.randint(len(all_item_pairs))]

        choice = env.sample_response(teacher_id, item_i, item_j)
        bias_filter.update(teacher_id, item_i, item_j, choice)

        if step % report_every == 0:
            _print_progress(step, bias_filter, true_U, true_b)

    # ------------------------------------------------------------------ #
    # 3. Final summary
    # ------------------------------------------------------------------ #
    final_U, final_b = bias_filter.get_estimates()
    u_std, b_std = bias_filter.get_marginal_std()
    u_err = np.abs(final_U - true_U)
    b_err = np.abs(final_b - true_b)

    print("\n" + "=" * 72)
    print("FINAL CONVERGENCE SUMMARY")
    print("=" * 72)
    print(f"{'Item':<10}{'True U':>10}{'Est. U':>10}{'|Error|':>10}{'Post. Std':>12}")
    for k in range(n_items):
        print(f"{k:<10}{true_U[k]:>10.3f}{final_U[k]:>10.3f}{u_err[k]:>10.3f}{u_std[k]:>12.3f}")

    print(f"\n{'Teacher':<10}{'True b':>10}{'Est. b':>10}{'|Error|':>10}{'Post. Std':>12}")
    for m in range(n_teachers):
        print(f"{m:<10}{true_b[m]:>10.3f}{final_b[m]:>10.3f}{b_err[m]:>10.3f}{b_std[m]:>12.3f}")

    print(f"\nOverall RMSE(U) : {np.sqrt(np.mean(u_err ** 2)):.4f}")
    print(f"Overall RMSE(b) : {np.sqrt(np.mean(b_err ** 2)):.4f}")

    # ------------------------------------------------------------------ #
    # 3b. Identifiability note: pairwise comparisons only constrain
    # differences (U_i - U_j), so the *absolute* level of U is a free
    # additive constant (adding c to every U_k leaves every logit, and
    # hence every likelihood, unchanged). Teacher biases do NOT share
    # this ambiguity, since b_m enters as a per-teacher additive term
    # rather than a global one -- which is why b typically converges
    # much tighter than raw U above. To make this explicit, we report
    # the best-fit global shift that aligns the estimated scale with
    # the true scale, i.e. the error that remains once the genuinely
    # unidentifiable degree of freedom is factored out.
    # ------------------------------------------------------------------ #
    shift = np.mean(true_U - final_U)
    shifted_U = final_U + shift
    shifted_err = np.abs(shifted_U - true_U)

    print("\n" + "-" * 72)
    print("IDENTIFIABILITY NOTE: pairwise comparisons pin down only")
    print("(U_i - U_j), leaving a free global additive shift in U.")
    print(f"Best-fit alignment shift c = {shift:+.3f} (applied as U_est + c):")
    print(f"{'Item':<10}{'True U':>10}{'Shifted Est.':>14}{'|Error|':>10}")
    for k in range(n_items):
        print(f"{k:<10}{true_U[k]:>10.3f}{shifted_U[k]:>14.3f}{shifted_err[k]:>10.3f}")
    print(f"Shift-corrected RMSE(U): {np.sqrt(np.mean(shifted_err ** 2)):.4f}  "
          f"(vs. raw RMSE(U) = {np.sqrt(np.mean(u_err ** 2)):.4f})")
    print("-" * 72)

    ranking_est = np.argsort(-final_U)
    ranking_true = np.argsort(-true_U)
    rank_match = "MATCHES" if np.array_equal(ranking_est, ranking_true) else "DOES NOT MATCH"
    print(f"\nEstimated item ranking (best -> worst): {ranking_est.tolist()}")
    print(f"True item ranking      (best -> worst): {ranking_true.tolist()}")
    print(f"Ranking recovery: {rank_match} ground truth.")
    print("=" * 72)


if __name__ == "__main__":
    main()
