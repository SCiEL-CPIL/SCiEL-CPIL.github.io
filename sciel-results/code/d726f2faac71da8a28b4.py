import numpy as np
import time as _time_module

# EVOLVE-BLOCK-START
_START_TIME = _time_module.perf_counter()
_DEADLINE = _START_TIME + 140.0  # 10s safety margin before 150s kill


def dispatch(ready_operations, time):
    """
    Dispatch rule: Maximize a composite score combining urgency (remaining_ops / soft_min_duration)
    and machine-contention avoidance (min_duration / sum_duration ratio).
    
    Uses a soft-min (log-sum-exp based) as a smooth surrogate for the hard min
    of machine durations. The contention term favors operations that can be
    scheduled on less loaded machines (small min relative to total), reducing
    idle time and congestion.
    
    Primary score: w1 * (remaining_ops / soft_min_duration) + w2 * (min_dur / sum_dur)
    Secondary: lower remaining_work preferred on near-ties.
    """
    if not ready_operations:
        return 0

    # Deadline guard: if we're running long, return quickly
    if _time_module.perf_counter() > _DEADLINE:
        return 0

    best_index = 0
    best_score = -float('inf')
    best_secondary = float('inf')

    n_ops = len(ready_operations)
    
    # Weights for composite score
    w_urgency = 1.0
    w_contention = 0.3  # Weight for machine-contention avoidance term

    for i, op in enumerate(ready_operations):
        # Calculate soft-min duration from machine options
        machine_options = op.get("machine_options", {})
        if not machine_options:
            # Fallback if no options (shouldn't happen)
            soft_min_duration = 1.0
            min_duration = 1.0
            sum_duration = 1.0
        else:
            durations = np.array(list(machine_options.values()), dtype=np.float64)
            if durations.size == 0:
                soft_min_duration = 1.0
                min_duration = 1.0
                sum_duration = 1.0
            else:
                # Clip to avoid log(0)
                durations = np.clip(durations, 1e-9, None)
                # Soft-min via log-sum-exp: -log(sum(exp(-x/tau))) * tau
                # As tau -> 0, this approaches min(x)
                tau = 0.3  # Lower temperature for sharper min approximation
                neg_durations = -durations / tau
                # Numerically stable log-sum-exp
                max_neg = np.max(neg_durations)
                log_sum_exp = max_neg + np.log(np.sum(np.exp(neg_durations - max_neg)))
                soft_min_duration = -tau * log_sum_exp
                if soft_min_duration <= 0:
                    soft_min_duration = 1e-9
                
                # Hard min and sum for contention ratio
                min_duration = np.min(durations)
                sum_duration = np.sum(durations)
                if sum_duration <= 0:
                    sum_duration = 1e-9

        remaining_ops = op.get("remaining_ops", 1)
        if remaining_ops <= 0:
            remaining_ops = 1

        remaining_work = op.get("remaining_work", 0)
        if remaining_work is None:
            remaining_work = 0.0

        # Calculate composite score
        # Urgency term: higher remaining_ops / soft_min_duration is better
        urgency_score = remaining_ops / soft_min_duration
        
        # Contention avoidance term: min/sum ratio favors less congested machines
        # Higher ratio means the min is closer to the sum, indicating more balanced/less congested options
        # Actually, we want to favor operations where the best machine is much better than others
        # So a LOW min/sum ratio means one machine is much faster -> less contention
        # But we want to AVOID contention, so we prefer operations that CAN go to less loaded machines
        # The ratio min/sum: if small, one machine is very fast (good, less contention on others)
        # If close to 1/n (n machines), all similar -> more potential contention
        # So we want to MAXIMIZE the "selectivity" or MINIMIZE the contention
        # Let's use: 1 - (min/sum) as a contention proxy? Or just min/sum inverted?
        # Actually, let's think: if min/sum is small, the fastest machine is much faster than average
        # This means we can schedule on a less-loaded machine -> good
        # So we want to MAXIMIZE (1 - min/sum) or equivalently minimize min/sum
        # But the directive says "ratio of the minimum required duration to the sum of durations"
        # and to use it for "machine-contention avoidance"
        # So let's use: contention_score = min_duration / sum_duration
        # And we want to MAXIMIZE this? Or minimize?
        # If min/sum is high (close to 1), all machines are similar -> high contention potential
        # If min/sum is low, one machine is much faster -> we can avoid contention
        # So we want to MINIMIZE min/sum, or MAXIMIZE (1 - min/sum)
        # Let's use: contention_avoidance = 1.0 - (min_duration / sum_duration)
        # This is higher when we have a clear fast option (less contention)
        contention_avoidance = 1.0 - (min_duration / sum_duration)
        # Clamp to [0, 1]
        contention_avoidance = np.clip(contention_avoidance, 0.0, 1.0)

        score = w_urgency * urgency_score + w_contention * contention_avoidance

        # Decision: use primary score, with secondary as tie-breaker
        # Small epsilon to treat near-equal scores as ties
        epsilon = 1e-6
        if score > best_score + epsilon:
            best_score = score
            best_secondary = remaining_work
            best_index = i
        elif abs(score - best_score) <= epsilon:
            # Tie-break: prefer lower remaining_work (closer to completion)
            if remaining_work < best_secondary:
                best_score = score
                best_secondary = remaining_work
                best_index = i

    return int(best_index)
# EVOLVE-BLOCK-END
