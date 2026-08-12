# Active Teacher Selection under Biased Teachers

**A Hidden Utility Bandit (HUB) research codebase, built on Freedman et al. (2023), "Active Teacher Selection for Reward Learning" (arXiv:[2310.15288](https://arxiv.org/abs/2310.15288)) — with an added extension to systematically biased annotators.**

This repository is organized in two layers:

1. **Foundation** — a replication of the Hidden Utility Bandit / Active Teacher Selection (HUB/ATS) framework introduced by Freedman, Svegliato, Wray, and Russell (UC Berkeley, Center for Human-Compatible AI), formulated as a POMDP and solved with a modified [POMCPOW](https://github.com/JuliaPOMDP/POMCPOW.jl) online planner (Julia).
2. **Extension** — a self-contained NumPy/SciPy re-implementation of the HUB teacher/filter machinery that relaxes ATS's core assumption of *unbiased* teachers, adding a jointly-inferred systematic bias term per teacher and an active-selection policy that accounts for it (Python).

---

## 1. Key Concepts & Paper Summary

### 1.1 Motivation

Reward learning from human feedback typically assumes a single, homogeneous source of preference labels. In practice, feedback is pooled from **heterogeneous annotators** — crowdworkers, domain experts, LLM judges — who differ along at least two axes:

- **Rationality** ($\beta_m$): how consistently teacher $m$'s stated preferences track the true utility difference between two items (a noise/temperature parameter).
- **Query cost** ($f_m$): what it costs the learner — in money, latency, or opportunity cost — to obtain one label from teacher $m$.

A cheap, noisy crowd-annotator and an expensive, careful domain expert are both valid teachers, but a learner that queries them interchangeably squanders its budget. Freedman et al. formalize this as a **Hidden Utility Bandit (HUB)**: the learner must simultaneously (a) *learn* a hidden utility function from a panel of imperfect teachers and (b) *act* on its current belief to accumulate reward, all under a shared, finite interaction budget.

### 1.2 The HUB Framework

A HUB instance is defined by:

| Symbol | Meaning |
|---|---|
| $U = [U_0, \dots, U_{N-1}]$ | Hidden ground-truth utility of each of $N$ items. Never directly observed. |
| $K$ arms | Bandit arms, each inducing a distribution over items; pulling one yields reward $\mathbb{E}[U]$ under that distribution. |
| $M$ teachers | Each with a known rationality $\beta_m$ (and, in the general formulation, a query cost $f_m$). |

At every timestep the learner chooses one of two action types:

- **Pull an arm** $k$ — collect (unobserved, stochastic) reward $\mathbb{E}_{d_k}[U]$, and receive an item observation from the arm's induced distribution.
- **Query a teacher** $m$ about a pair of items $(i, j)$ — receive a noisy pairwise preference label, but *no reward this step*.

This is precisely the classic **explore/exploit** trade-off, except "exploration" here is not free stochastic reward but a deliberate purchase of information from a chosen source. This repository's Julia implementation (`experiment_scripts/ATS_finite.jl`, `ATS_infinite.jl`) casts this exactly as a finite/infinite-horizon POMDP — belief state $=(U, \{d_k\}, \{\beta_m\})$, solved online with POMCPOW — and reward for a query action is `0`, so the *cost* of information-gathering is realized implicitly through the shared, finite action budget: every timestep spent querying is a timestep not spent collecting utility.

### 1.3 Preference Model: Rationality vs. Bias

Teachers do not report $U$ directly; they report **noisy pairwise comparisons**. Freedman et al.'s teacher model is a Bradley–Terry / Boltzmann-rational choice rule parameterized only by rationality:

$$
P(i \succ j \mid U, \beta_m) = \frac{1}{1 + \exp\!\big(-\beta_m (U_i - U_j)\big)}
$$

- $\beta_m \to 0$: the teacher answers uniformly at random (uninformative).
- $\beta_m \to \infty$: the teacher deterministically reports the higher-utility item (noiseless oracle).

Standard ATS assumes every teacher's comparisons are **centered on the truth** — noisy, but unbiased. This repository's extension (Section 2) relaxes exactly that assumption, introducing a per-teacher **systematic offset** $b_m$ that shifts a teacher's comparisons in a fixed direction regardless of how much data is collected:

$$
P(i \succ j \mid U, \beta_m, b_m) = \frac{1}{1 + \exp\!\big(-\beta_m\big((U_i - U_j) + b_m\big)\big)}
$$

Rationality and bias are qualitatively different failure modes: $\beta_m$ controls *variance* in a teacher's labels (more queries always average it out), while $b_m$ controls *systematic error* (more queries to a biased teacher alone never remove it — the learner must recognize and model the offset itself).

### 1.4 Active Teacher Selection (ATS)

Given $M$ teachers with different $(\beta_m, f_m)$, ATS is the policy question: *which teacher, about which pair of items, should the learner query next?* The original formulation (`experiment_scripts/ATS_finite.jl` / `ATS_infinite.jl`) answers this via online POMDP planning over the full belief state, implicitly trading off a query's expected information gain against the opportunity cost of not exploiting. This repository also ships the standard baselines it is benchmarked against:

| Script | Policy |
|---|---|
| `ATS_finite.jl` / `ATS_infinite.jl` | **Active Teacher Selection** — POMCPOW plans over which teacher/pair to query. |
| `passive_finite.jl` / `passive_infinite.jl` | **Passive selection** — teacher is chosen uniformly at random; only arm/query timing is planned. |
| `naive.jl` | Fixed exploration budget with one designated teacher, then pure exploitation. |
| `random.jl` | Uniformly random action (arm or query) at every step. |
| `arms.jl` | Random arm-pulling only — never queries a teacher at all. |

---

## 2. Extension: Biased Teachers in the HUB

Real annotators are not just noisy — they can be **systematically skewed**: an annotator who consistently over-rates outputs resembling their own writing style, a reward model with a known sycophancy bias, or a rubric that silently favors longer responses. None of this is captured by $\beta_m$ alone. This extension asks the natural follow-up question to Freedman et al.'s framework: *if a teacher's comparisons are shifted by an unknown constant $b_m$, can the learner still recover $U$ — and can it learn to query around the bias rather than being misled by it?*

### 2.1 Joint Bayesian Filter over $(U, b)$

The extension (`biased_teacher_hub.py`, generalized in `benchmark_biased_ats.py`) replaces point estimation with a **discrete-grid joint posterior** over both the utility vector and every teacher's bias simultaneously — a single tensor of shape $(\text{grid}_U)^N \times (\text{grid}_b)^M$, updated in log-space via Bayes' rule after each observed comparison. Tracking $U$ and every $b_m$ *jointly* (rather than fitting bias per teacher in isolation) is what lets the filter correctly discount a cheap teacher's skew using evidence gathered from other, less biased teachers.

One immediate identifiability subtlety falls out of the pairwise-only likelihood: since only differences $U_i - U_j$ ever enter the model, the *absolute* scale of $U$ is a free gauge parameter — infinitely many equally-likely global shifts of $U$ explain the same data. `benchmark_biased_ats.py` fixes this by anchoring one reference item's utility grid to the single value $\{0\}$, which pins the coordinate system without altering the inference elsewhere.

### 2.2 Acquisition Dynamics: Myopic Var[U] vs. Joint (U, b) VOI

Extending ATS to biased teachers raises a policy question, not just a modeling one. A natural active-selection rule is to query the (teacher, item pair) that maximizes expected posterior variance reduction on $U$ per unit cost:

$$
\text{Query}^\star = \arg\max_{m, i, j} \ \frac{\Delta\mathrm{Var}[U \mid m, i, j]}{f_m}
$$

Because a cheap, biased teacher can still shrink $\mathrm{Var}[U]$ efficiently once its bias is roughly known, this **myopic, $U$-only** rule turns out to systematically **under-explore the expensive anchor teacher** — in our seed sweep, it allocated only 6–9 of 150 queries to the honest, unbiased teacher. That is enough to get the item ranking right most of the time, but not enough independent, bias-free evidence to correctly pin down that the anchor teacher's *own* bias is exactly zero — its bias estimate drifted to $-1.4$ to $-3.6$ across every seed tested, despite the ground truth being $0$.

The fix is to make the acquisition objective reward resolving bias uncertainty directly, not just as a side effect of reducing $\mathrm{Var}[U]$:

$$
\text{VOI}_{\text{joint}}(m, i, j) = \frac{\Delta\mathrm{Var}[U \mid m,i,j] \;+\; \lambda_b \cdot \Delta\mathrm{Var}[b \mid m,i,j]}{f_m}
$$

With $\lambda_b = 1.0$, this joint objective increases honest-teacher exploration in most seeds and reduces mean utility reconstruction error by roughly a third — a **real, but partial**, correction (see Section 4 for the numbers; we did not observe full convergence of the anchor teacher's bias estimate to zero at this $\lambda_b$, which we report as an open direction rather than a solved problem).

---

## 3. Code Base Structure

```
ATS/
├── experiment_scripts/          # Original HUB/ATS POMDP formulation (Julia)
│   ├── ATS_finite.jl            #   Active Teacher Selection, finite horizon
│   ├── ATS_infinite.jl          #   Active Teacher Selection, infinite horizon
│   ├── passive_finite.jl        #   Passive-selection baseline, finite horizon
│   ├── passive_infinite.jl      #   Passive-selection baseline, infinite horizon
│   ├── naive.jl                 #   Fixed-teacher explore-then-exploit baseline
│   ├── random.jl                #   Uniformly random action baseline
│   └── arms.jl                  #   Random-arm-only baseline (no querying)
├── run_scripts/                 # Shell wrappers invoking the above with paper hyperparameters
├── POMCPOW_modified/             # Vendored, modified fork of JuliaPOMDP/POMCPOW.jl (online solver)
├── utils/                       # Julia plotting/import utilities
├── make_plots/                  # Jupyter notebooks reproducing the paper's figures
├── data/, sims/, logs/          # Raw experiment logs and simulation output from prior runs
│
├── biased_teacher_hub.py        # Extension, Part 1: environment + joint (U, b) Bayesian filter
├── benchmark_biased_ats.py      # Extension, Part 2: identifiability fix, active-selection
│                                 #   policies, and the full benchmark/plotting pipeline
├── benchmark_results.png        # Generated: Biased-ATS vs. Standard-ATS vs. Random-Biased-ATS
└── joint_voi_benchmark.png      # Generated: myopic vs. joint (U, b) VOI acquisition, 5-seed sweep
```

**`biased_teacher_hub.py`** defines the biased-teacher environment and filter in isolation:

- `BiasedTeacherEnv` — ground-truth $U$ and a roster of teachers, each with $(\beta_m, b_m, f_m)$; `sample_response(teacher_id, i, j)` draws a stochastic pairwise label from the biased preference model.
- `BayesianBiasFilter` — the discrete-grid joint posterior over $(U, b)$ described in §2.1, with `update()` (log-space Bayes update) and `get_estimates()` (posterior marginal means).
- `main()` runs a 120-query, 2-teacher demo (one honest expert, one cheap biased annotator) and reports convergence, including a worked illustration of the utility-shift identifiability issue that motivates the anchoring fix below.

**`benchmark_biased_ats.py`** builds the full research pipeline on top of it:

- `AnchoredBayesianBiasFilter` — the identifiable filter (anchored utility gauge; optional `assume_unbiased=True` mode that reproduces the original ATS assumption exactly, used as a baseline).
- `select_query_biased_ats` / `select_query_joint_biased_ats` / `select_query_standard_ats` / `select_query_random` — the four query-selection policies compared in this repository.
- `run_algorithm` / `run_acquisition_comparison` — simulation drivers producing per-timestep utility error, bias error, and cumulative regret traces.
- `main()` runs both benchmark experiments described in §4 end to end and writes both PNGs.

---

## 4. Quick Start

### 4.1 Python Extension (biased-teacher HUB)

**Requirements:** Python 3.10+, NumPy, SciPy, Matplotlib.

```bash
pip install numpy scipy matplotlib
```

Run the standalone filter/environment demo:

```bash
python3 biased_teacher_hub.py
```

Run the full active-selection benchmark (identifiability fix, 3-algorithm comparison, and the 5-seed myopic-vs-joint-VOI acquisition sweep):

```bash
python3 benchmark_biased_ats.py
```

This is a single, self-contained entry point — no configuration files or external services required. Expect a runtime on the order of several minutes: the acquisition-rule comparison alone runs 10 independent 150-round active-search simulations (2 policies × 5 seeds). It prints the summary tables described below to the terminal and writes `benchmark_results.png` and `joint_voi_benchmark.png` to the working directory.

### 4.2 Original HUB/ATS Replication (Julia)

**Requirements:** Julia with `POMDPs.jl`, `QuickPOMDPs.jl`, `BasicPOMCP.jl`, `D3Trees.jl`, `GridInterpolations.jl`, and the other packages listed at the top of each script in `experiment_scripts/`. `POMCPOW_modified/` is a vendored, modified fork of `POMCPOW.jl` and is included in-repo — do not `Pkg.add` a separate copy.

```bash
julia run_scripts/run_all.sh
```

or invoke an individual experiment directly, e.g.:

```bash
julia experiment_scripts/ATS_finite.jl false 3 3 21 21 10 50 1 20 0
```

(see the header comment of each script for its positional-argument signature). Output logs are written to `logs/`, raw simulation traces to `sims/`; reproduce the paper's figures via the notebooks in `make_plots/`.

---

## 5. Benchmark Replication & Experimental Results

Both experiments below share one biased-teacher environment: $N = 4$ items with true utility $U = [0, 6, 2, 8]$ (item 3 is optimal), an **honest expert** ($\beta = 2.0$, $b = 0$, $f = 5.0$) and a **cheap, biased annotator** ($\beta = 1.5$, $b = +3.5$ toward item 0, $f = 1.0$).

### 5.1 Algorithm Comparison (`benchmark_results.png`)

150 rounds each, comparing active bias-aware selection, the original (bias-blind) ATS assumption, and random querying with a bias-aware filter:

| Algorithm | Final Cumulative Regret | Final Utility SSE | Final Bias MAE | Best Arm Found |
|---|---:|---:|---:|:---:|
| **Biased-ATS** (active + bias-aware filter) | **368.0** | 13.52 | 2.11 | ✅ item 3 |
| Standard-ATS (active + naive unbiased filter) | 450.0 | 30.97 | 1.75 | ❌ item 1 |
| Random-Biased-ATS (random + bias-aware filter) | 694.0 | 11.07 | 2.69 | ❌ item 1 |

Modeling bias jointly with utility (Biased-ATS) achieves the lowest regret of the three and is the only one of the three to recover the correct optimal arm — the naive filter that assumes $b_m \equiv 0$ (Standard-ATS) is confidently misled by the cheap biased annotator, which it ends up querying exclusively (150/150 queries).

### 5.2 Acquisition-Rule Comparison (`joint_voi_benchmark.png`)

Myopic $U$-only VOI vs. joint $(U, b)$ VOI ($\lambda_b = 1.0$), 5 seeds each (1, 11, 42, 123, 777), mean over seeds:

| Acquisition Rule | Mean Teacher-0 Bias Estimate (true $= 0$) | Mean Utility MSE | Best Arm Correct |
|---|---:|---:|:---:|
| Myopic $U$-only VOI (original) | $-2.699$ | 6.16 | 4 / 5 |
| Joint $(U, b)$ VOI (proposed) | $-2.244$ | **4.02** | 4 / 5 |

The joint objective increases honest-teacher exploration in most seeds (e.g. 6→14 queries in one run) and lowers mean utility error by roughly a third, without changing the overall hit rate on the best arm. It does **not** fully close the anchor-teacher bias gap at $\lambda_b = 1.0$ — a real, partial improvement rather than a complete fix, and we leave tuning $\lambda_b$ further (or adding an explicit per-teacher exploration floor) as an open direction.

---

## 6. Citation

If you use the original HUB/ATS framework, please cite the foundational paper this repository replicates:

```bibtex
@article{freedman2023activeteacher,
  title   = {Active Teacher Selection for Reward Learning},
  author  = {Freedman, Rachel and Svegliato, Justin and Wray, Kyle and Russell, Stuart},
  journal = {arXiv preprint arXiv:2310.15288},
  year    = {2023},
  url     = {https://arxiv.org/abs/2310.15288}
}
```

The biased-teacher extension (`biased_teacher_hub.py`, `benchmark_biased_ats.py`) is an independent addition built on top of the above framework and is not part of the original publication; please attribute the HUB/ATS formulation itself to Freedman et al. (2023) and treat the bias-aware filter, identifiability fix, and joint-VOI acquisition rule as this repository's own contribution.
