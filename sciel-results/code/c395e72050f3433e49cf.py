import numpy as np
import time as _time_module

_START_TIME = _time_module.perf_counter()
_DEADLINE = _START_TIME + 140.0  # safety margin before 150s kill


def dispatch(ready_operations, time):
    """
    Dispatch rule for job-shop scheduling (makespan minimization).

    Combined scoring:
      - Urgency: remaining_ops (jobs with more work left are more critical)
      - Speed: 1/min_duration (faster completion)
      - Contention avoidance: penalize operations forced onto congested machines.
        For each operation we compute a per-machine utilization proxy
        (machine_load / duration) and take the WORST (highest) across its
        options; lower worst-utilization => less blocked => higher score.
      - Slack: mild time-aware bonus for jobs whose remaining work is tight.
      - Secondary: lower remaining_work (job closer to completion).
    """
    if not ready_operations:
        return 0

    if _time_module.perf_counter() > _DEADLINE:
        return 0

    n_ops = len(ready_operations)

    # First pass: machine loads (sum of durations per machine across ready ops)
    machine_loads = {}
    for op in ready_operations:
        machine_options = op.get("machine_options", {})
        for m, dur in machine_options.items():
            machine_loads[m] = machine_loads.get(m, 0.0) + dur

    if not machine_loads:
        # No load info; fall back to pure urgency/speed
        machine_loads = {}

    # Normalization constant: average machine load, used to scale utilization
    if machine_loads:
        avg_load = sum(machine_loads.values()) / len(machine_loads)
        if avg_load <= 0:
            avg_load = 1.0
    else:
        avg_load = 1.0

    best_index = 0
    best_score = -float('inf')
    best_secondary = float('inf')

    epsilon = 1e-9
    beta = 10.0   # soft-min temperature (higher = sharper toward true min)
    alpha = 0.55  # contention weight (moderate, so speed/urgency still dominate)

    for i, op in enumerate(ready_operations):
        machine_options = op.get("machine_options", {})
        if not machine_options:
            soft_min_duration = 1.0
            worst_util = 0.0
        else:
            durations = list(machine_options.values())
            if not durations:
                soft_min_duration = 1.0
                worst_util = 0.0
            else:
                # Soft-min via log-sum-exp (numerically stable)
                min_d = min(durations)
                if min_d <= 0:
                    min_d = 1e-9
                neg_d_over_beta = [-d / beta for d in durations]
                max_neg = max(neg_d_over_beta)
                sum_exp = sum(np.exp(x - max_neg) for x in neg_d_over_beta)
                soft_min_duration = -(max_neg + np.log(sum_exp)) * beta
                if soft_min_duration <= 0:
                    soft_min_duration = 1e-9

                # Worst-case utilization across this op's machines.
                # util_m = load_m / duration_m  (higher = more congested relative
                # to how long this op needs it). Normalize by avg_load for scale.
                worst_util = 0.0
                for m, d in machine_options.items():
                    if d <= 0:
                        d = 1e-9
                    load = machine_loads.get(m, 0.0)
                    util = (load / d) / avg_load
                    if util > worst_util:
                        worst_util = util

        remaining_ops = op.get("remaining_ops", 1)
        if remaining_ops <= 0:
            remaining_ops = 1

        remaining_work = op.get("remaining_work", 0)
        if remaining_work is None:
            remaining_work = 0.0

        # Time-aware slack: how much time remains relative to remaining work.
        # Tighter slack => slightly more urgent.
        if time is not None:
            slack = remaining_work - time
        else:
            slack = remaining_work
        if slack <= 0:
            slack = 1e-9

        # Primary score: urgency * speed * contention-avoidance
        base_score = remaining_ops / soft_min_duration
        # Contention: reward low worst-utilization. Map worst_util (>=0) to a
        # multiplicative factor in [~0.5, ~1.8].
        contention_score = 1.0 + alpha * (1.0 - min(worst_util, 2.0))
        # Clamp to keep it bounded and positive.
        contention_score = max(0.4, min(2.0, contention_score))

        # Small slack bonus: jobs with tighter slack get a mild boost.
        slack_factor = 1.0 + 0.05 * (remaining_work / (slack + 1e-9))
        slack_factor = min(slack_factor, 1.3)

        score = base_score * contention_score * slack_factor

        # Secondary: prefer lower remaining_work (job closer to done)
        secondary = remaining_work

        if score > best_score + epsilon:
            best_score = score
            best_secondary = secondary
            best_index = i
        elif abs(score - best_score) <= epsilon:
            if secondary < best_secondary:
                best_secondary = secondary
                best_index = i

    return int(best_index)
