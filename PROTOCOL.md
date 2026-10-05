# Rare Failure Eval: fresh-seed replication protocol

Local pre-run specification, October 4, 2026. Frozen before this collection,
not a public preregistration. This is a fresh-seed replication after inspecting
an earlier 128-world pilot. The generator and comparisons are unchanged.
Primary execution requires an explicit operator flag after review and tests. No API, network, private data,
real agent, paid inference, online safety or frontier-performance claim.

## Question and target

Adverse events affect the score:
`U = R - 800 F`, with ordinary reward `R` bounded in `[0,1]`, observed binary
failure `F`, and utility bounded in `[-800,1]`. A draw costs one simulated
evaluation irrespective of its outcome. Failures are acceptable in this offline
simulation, not asserted acceptable in deployment. The estimand for each agent
is its expectation under a fixed, known task distribution, not the distribution
of tasks an allocation policy happens to sample. Sampling and final selection
are separate decisions. This is a stratified two-stage allocation experiment,
not an implementation of a simple-regret-optimized bandit algorithm.

## Frozen generator and seeds

Four agents and eight tasks (32 cells). Base agent rewards are
`[0.74, 0.70, 0.62, 0.56]`. Base task weights are `[1,1,1,1,2,2,4,4]/16`.
Task offsets `[-.06,.04,-.03,.06,-.04,.02,.01,0]` are centered by their weighted
mean. Task reward half-widths are `[.02,.04,.08,.12,.16,.06,.10,.18]`.
`R[a,t] = base[a] + centered_offset[t] + half_width[t] Z`, where `Z` is
independent and equally likely to be minus or plus one. Thus reward noise is
heteroskedastic and bounded; true weighted ordinary rewards equal agent bases.

For each world seed, independently permute agent identities and task identities
(weights, offsets and widths move together), uniformly choose a risky task, and
choose independent random tie orders for agents and integer-allocation cells.
These choices are identical across policies, budgets and scenarios for that
seed. The risky agent in the concentrated scenario is the nominal best agent;
its physical index therefore varies. Truth is computed analytically, not from
a high-sample reference estimator. All true utilities have distinct values.

Scenarios, fixed before smoke or primary results:

| Scenario | Failure probability | True weighted utilities in base-agent order |
| --- | --- | --- |
| no_rare_loss | zero everywhere | .74, .70, .62, .56 |
| concentrated | .001 in nominal-best agent's randomly chosen task only | .74 minus .8 times risky task weight; .70, .62, .56 |
| diffuse | every task, agent-specific rates [.00020,.00010,.00004,.00002] | .58, .62, .588, .544 |

At the minimum task weight, the concentrated loss is .05 and reverses the .04
nominal-best gap. Other weights increase the reversal. The diffuse and
concentrated settings do not match total event mass, so their difference is not
a controlled causal effect of concentration. `R` and `F` are independent within
each cell. This is an explicit modeling choice, not an empirical agent property.

Primary seeds are integers `970000` through `971023`, inclusive (1,024 worlds).
Development seeds are `940000`, `940001`, `940002`; tests use other development
seeds outside the primary range. Budgets are exactly 6,400 and 25,600 draws per
world/scenario/policy. The full primary grid is 24,576 rows and 393,216,000 draws.
Common random numbers use reproducible independent per-cell, per-phase and
per-component NumPy PCG64 streams. Same seed/cell/phase/component prefixes are
replayed across policies, budgets and scenarios; scenario differences use common
failure uniforms. Pilot and estimation streams have distinct phase keys.

## Policies and exact budget allocation

Each policy has exactly one quarter of the total budget per agent. The three
matched two-stage policies pay for four pilot draws in each of 32 cells (128
draws). All remaining draws are stage-two estimation samples, with allocation frozen using
only public task weights, observed pilot outcomes, a public standard-deviation
floor .02, and independently generated tie order. Neither policy gets world
means, true risk, scenario labels, future samples, or true best-agent identity.

1. `uniform_all`: practical all-budget balanced baseline, no pilot. Exactly
   200 or 800 independent estimation draws per cell, all included in estimation.
2. `uniform`: matched balanced stage-two allocation across 32 cells.
3. `neyman`: stage-two cell scores `w[t] * max(pilot_SD(U[a,t]), .02)`, with
   unbiased sample variance (`ddof=1`) inside the square root. This targets the
   sum of variances of stratified agent-utility estimates, not specifically
   terminal best-agent identification and not the exact constrained optimum.
4. `event_following`: stage-two scores are one for pilot cells with at least one
   failure and zero elsewhere. Within each agent with no pilot failure, scores
   are uniform. This
   follows observed events only once at the pilot boundary, not continuously.

Within each agent, estimation proportions are `.25/8 + .75 * normalized
score`. Normalize separately for each agent, whose estimation budget is
`(budget - pilot_cost)/4`. Convert to exact integer counts by flooring expected
counts and assigning remaining draws to largest fractional remainders within
that agent; independent shared random
cell order breaks exact remainder ties. Thus each cell receives at least
`floor(.25 * (budget - 128) / 32)` independent estimation draws, at least 49 or
199 under the two primary budgets. The uniform policy has exactly 196 or 796
stage-two draws per cell. Cost equality includes pilot observations, even though
matched uniform does not need them for allocation. All-budget uniform has no
pilot cost or pilot-hit metric, and uses its entire budget for estimation. A four-draw pilot at p=.001 detects
an event with probability `1 - .999**4 = .003994003999`, about 0.4%; it usually
cannot discover the risky cell. No claim that the warmup adequately estimates
rare-event risk is permitted. Primary data do not tune the pilot, floor or loss.

## Estimation, ranking and risk intervals

Only independent stage-two draws enter cell means of utility and failure rate.
Agent means use the original fixed task weights, never sample frequencies.
Conditional on the entire pilot and its frozen allocation, these point estimates
are unbiased under the specified IID generator. Pilot outcomes affect allocation
and discovery diagnostics, never these point estimates. The selected maximum,
its rank, regret, and bootstrap summaries are not called unbiased. Pooled pilot
plus stage-two means and adaptively stopped means are not reported as unbiased.

Rank estimated weighted utility descending, resolving exact ties with the common
independent agent tie order. The primary terminal outcomes are simple regret
`max(true_utility) - true_utility[selected_agent]` and best-agent selection error.
Secondary: full-order normalized Kendall inversion fraction (six pairs), utility
RMSE, task-weighted risk RMSE, any event discovered across pilot and stage two,
fraction of genuinely risky cells discovered, pilot-hit indicator, cell-level
risk estimates and confidence endpoints, risk-family coverage, mean risky-cell
interval width, zero-event cell fraction, and mean upper bound among stage-two
zero-event cells. Undefined risky-cell metrics in no-risk worlds and undefined
zero-cell metrics are null, not zero. Raw counts retain all cells.

For 32 independent-stage-two Bernoulli cell counts, use two-sided exact
Clopper-Pearson intervals with `alpha_cell = .05/32` and alpha/2 in each tail.
Conditional on pilot allocation, each interval has at least its nominal coverage
and a union bound gives at least 95% simultaneous coverage of all 32 cells at
this one terminal look within one policy/world/scenario/budget. Independence
between cell intervals is not required by Bonferroni. At zero events the lower
endpoint is zero and upper is `1 - (.05/64)**(1/n)`, strictly positive. Zero
observed events do not establish impossibility. The intervals are NOT anytime
bounds, not confidence sequences, and have no simultaneous guarantee across
policies, budgets, worlds, scenarios or bootstrap comparisons. No interim look
or optional stopping is used. Both fixed budgets are reported separately.

## Reporting and frozen comparisons

For each scenario and budget, report every policy and all named scalar metrics.
Utility RMSE and risk RMSE are computed across agents within each world, then
averaged across worlds. Their aggregate labels mean **mean per-world RMSE**,
not a globally pooled root mean squared error.
For each, report all six paired contrasts: uniform minus uniform_all, neyman
minus uniform_all, event_following minus uniform_all, neyman minus uniform,
event_following minus uniform, and neyman minus event_following. All primary and
secondary scalar metrics get the same complete comparisons; null diagnostics
remain explicit. No subset, favorable seed, metric, scenario, budget or policy
is selected post hoc. Retain full per-seed rows and cell-level arrays.

Resample the 1,024 paired seed indices 2,000 times using PCG64 seed 950001, shared
across contrasts, budgets and scenarios. Report mean differences and descriptive
2.5/97.5 percentile bootstrap endpoints. These are exploratory, unadjusted
descriptive intervals, not p-values or significance tests; do not treat a zero
width interval as proof of zero population probability. Development results are
labeled smoke only, never included in primary averages or interpreted as a
policy winner. Primary and secondary hierarchy is descriptive, not confirmatory.

## Reproducibility and gates

Require at least 80% coverage,
complete-grid validation, source/protocol SHA-256 hashes and package versions in
a pre-run manifest, immutable new output directory, and a post-run unchanged
source/protocol-hash check. No partial, duplicate, cost-mismatched, altered-source
or missing-grid artifact may be summarized. Unit tests cover utility reversal,
cost/positivity, task weights, sample separation, zero-event bounds,
reproducibility, absence of oracle-policy inputs and incomplete-grid rejection.
Independent code and Python review plus protocol review precede primary
execution. A primary run requires explicit `--approved-primary` as an additional
operator gate; that flag is not itself evidence of scientific review.

The experiment is not an Atari replication or a test of the paper's full framework.
The paper's v2 section 5.2.1 describes Gaussian simulations based on Agent57
summaries, a different experimental setting. Sequential Halving and other best-arm algorithms are
untested next steps requiring a separate protocol and cost-matched comparison.
There is no claim of novelty, frontier significance, real-world safety, formal
failure impossibility, or a general advantage over uniform sampling.

## Attribution and independence

Independent synthetic extension inspired by the public paper [Active Evaluation
of General Agents](https://arxiv.org/abs/2601.07651v2), Lanctot et al. (2026).
This project does not implement their Elo or Soft Condorcet methods and is not
a replication of their paper. No employer data, private conversation content,
external collaborator contribution or organizational endorsement is claimed.

Fresh-seed replication checks Monte Carlo stability inside this same generator.
It does not resolve external validity or introduce new empirical agent evidence.
No outcomes from seeds 970000 through 971023 were inspected when this plan was written.
