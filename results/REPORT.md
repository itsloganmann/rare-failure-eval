# PRIMARY results: bounded synthetic risk utility

Exploratory synthetic results, not real agent evaluation.

All policies, scenarios and budgets are reported below. Every scalar metric
and all six paired contrasts are retained in `summary.json`.

RMSE summaries are mean per-world RMSE, not globally pooled RMSE.

| Scenario | Budget | Policy | Simple regret | Best-agent error | Kendall error |
| --- | ---: | --- | ---: | ---: | ---: |
| no_rare_loss | 6400 | uniform_all | 0.000000 | 0.000000 | 0.000000 |
| no_rare_loss | 6400 | uniform | 0.000000 | 0.000000 | 0.000000 |
| no_rare_loss | 6400 | neyman | 0.000000 | 0.000000 | 0.000000 |
| no_rare_loss | 6400 | event_following | 0.000000 | 0.000000 | 0.000000 |
| no_rare_loss | 25600 | uniform_all | 0.000000 | 0.000000 | 0.000000 |
| no_rare_loss | 25600 | uniform | 0.000000 | 0.000000 | 0.000000 |
| no_rare_loss | 25600 | neyman | 0.000000 | 0.000000 | 0.000000 |
| no_rare_loss | 25600 | event_following | 0.000000 | 0.000000 | 0.000000 |
| concentrated | 6400 | uniform_all | 0.050605 | 0.841797 | 0.246745 |
| concentrated | 6400 | uniform | 0.050781 | 0.844727 | 0.246908 |
| concentrated | 6400 | neyman | 0.044160 | 0.836914 | 0.218587 |
| concentrated | 6400 | event_following | 0.050713 | 0.842773 | 0.246582 |
| concentrated | 25600 | uniform_all | 0.026758 | 0.444336 | 0.158854 |
| concentrated | 25600 | uniform | 0.026768 | 0.445312 | 0.159017 |
| concentrated | 25600 | neyman | 0.019326 | 0.526367 | 0.175781 |
| concentrated | 25600 | event_following | 0.026777 | 0.446289 | 0.158854 |
| diffuse | 6400 | uniform_all | 0.030914 | 0.780273 | 0.351562 |
| diffuse | 6400 | uniform | 0.031172 | 0.786133 | 0.352376 |
| diffuse | 6400 | neyman | 0.030672 | 0.772461 | 0.353516 |
| diffuse | 6400 | event_following | 0.031133 | 0.785156 | 0.351888 |
| diffuse | 25600 | uniform_all | 0.020770 | 0.528320 | 0.331217 |
| diffuse | 25600 | uniform | 0.020816 | 0.529297 | 0.330566 |
| diffuse | 25600 | neyman | 0.022887 | 0.587891 | 0.364909 |
| diffuse | 25600 | event_following | 0.020816 | 0.529297 | 0.330566 |

## Limits

Four pilot draws detect p=.001 risk only about 0.4% of the time.
Failure intervals are one-look Clopper-Pearson with Bonferroni over
32 cells per run, not anytime bounds. Zero events do not establish safety.
Only independent estimation samples contribute to point estimates.
Task weights and per-agent budgets remain fixed. Selected maxima may be optimistic.
Bootstrap intervals are descriptive, unadjusted and not significance tests.
Sequential Halving, real-agent transfer and frontier claims were not tested.
