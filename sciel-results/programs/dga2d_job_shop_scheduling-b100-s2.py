import numpy as np
import time as _time_module

_START_TIME = _time_module.perf_counter()
_DEADLINE = _START_TIME + 140.0  # 10s safety margin before 150s kill


def dispatch(ready_operations, time):
    """
    Dispatch rule for job-shop scheduling (makespan minimization).
    
    Uses a combined scoring approach:
    - Urgency: remaining_ops (more remaining ops = more critical)
    - Speed: 1/min_duration (faster completion)
    - Contention avoidance: prefer operations whose best machine has low utilization
      (ratio of min duration to total available machine time)
    - Secondary: lower remaining_work (job closer to completion)
    
    The score balances urgency and speed while rewarding operations that can
    be scheduled on less congested machines.
    """
    if not ready_operations:
        return 0

    if _time_module.perf_counter() > _DEADLINE:
        return 0

    n_ops = len(ready_operations)
    
    # First pass: compute machine loads (sum of durations per machine)
    # to estimate contention
    machine_loads = {}
    for op in ready_operations:
        machine_options = op.get("machine_options", {})
        for m, dur in machine_options.items():
            if m not in machine_loads:
                machine_loads[m] = 0.0
            machine_loads[m] += dur

    # Compute average load per machine for normalization
    if machine_loads:
        avg_load = sum(machine_loads.values()) / len(machine_loads)
    else:
        avg_load = 1.0

    best_index = 0
    best_score = -float('inf')
    best_secondary = float('inf')

    epsilon = 1e-9
    beta = 10.0  # Temperature for soft-min (higher = sharper)
    gamma = 5.0  # Temperature for contention soft-max (lower = sharper)

    for i, op in enumerate(ready_operations):
        machine_options = op.get("machine_options", {})
        if not machine_options:
            soft_min_duration = 1.0
            contention_score = 1.0  # Neutral when no options
        else:
            durations = list(machine_options.values())
            if not durations:
                soft_min_duration = 1.0
                contention_score = 1.0
            else:
                # Soft-min via log-sum-exp: smooth approximation of min
                # soft_min = -log(sum(exp(-d/beta)))/beta
                min_d = min(durations)
                if min_d <= 0:
                    min_d = 1e-9
                
                # Compute soft-min using log-sum-exp for numerical stability
                neg_d_over_beta = [-d / beta for d in durations]
                max_neg = max(neg_d_over_beta)
                sum_exp = sum(np.exp(x - max_neg) for x in neg_d_over_beta)
                soft_min_duration = -(max_neg + np.log(sum_exp)) * beta
                if soft_min_duration <= 0:
                    soft_min_duration = 1e-9
                
                # Contention avoidance: score based on ratio of min duration
                # to total available machine time for this operation
                # Higher ratio = less constrained by machine contention
                total_machine_time = sum(machine_loads.get(m, 0.0) for m in machine_options)
                if total_machine_time > 0:
                    # Ratio: min_d / total_time — higher means this op's best
                    # machine is relatively less loaded compared to alternatives
                    contention_ratio = min_d / total_machine_time
                else:
                    contention_ratio = 1.0  # Neutral if no load info
                
                # Use soft-max style boost: operations with higher contention_ratio
                # get a larger multiplicative boost
                # Apply a smooth transformation to avoid extreme values
                # contention_score = 1 + alpha * (contention_ratio - 1)
                # Clip to reasonable range to prevent domination
                alpha = 0.4
                raw_boost = 1.0 + alpha * (contention_ratio - 1.0)
                # Ensure score stays positive and bounded
                contention_score = max(0.5, min(2.0, raw_boost))

        remaining_ops = op.get("remaining_ops", 1)
        if remaining_ops <= 0:
            remaining_ops = 1

        remaining_work = op.get("remaining_work", 0)
        if remaining_work is None:
            remaining_work = 0.0

        # Critical path slack: total remaining work minus current time
        # This estimates how much time is left for this job
        critical_path_slack = remaining_work - time if time is not None else remaining_work
        if critical_path_slack <= 0:
            critical_path_slack = 1e-9

        # Primary score: urgency × speed × contention_avoidance
        # Higher urgency, lower duration, and better machine availability → higher score
        base_score = remaining_ops / soft_min_duration
        
        # Multiply by contention score to reward less-congested machine choices
        score = base_score * contention_score

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
