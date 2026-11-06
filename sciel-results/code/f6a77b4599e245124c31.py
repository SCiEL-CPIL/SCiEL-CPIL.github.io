"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

# Deadline guard: evaluator kills after 150s wall-clock.
# We track a global start time and enforce a safety margin.
_START_TIME = _time.time()
_DEADLINE_MARGIN = 5.0  # seconds

def _elapsed():
    return _time.time() - _START_TIME

def _deadline_exceeded():
    return _elapsed() > (150.0 - _DEADLINE_MARGIN)


def dispatch(ready_operations, time):
    if not ready_operations:
        return 0
    
    # Check deadline
    if _deadline_exceeded():
        # Fallback: pick the operation with the most remaining work
        best_idx = 0
        best_rw = -1
        for i, op in enumerate(ready_operations):
            rw = op.get("remaining_work", 0)
            if rw > best_rw:
                best_rw = rw
                best_idx = i
        return int(best_idx)
    
    n = len(ready_operations)
    
    # Pre-compute normalization constants across all ready operations
    max_remaining_work = max(op.get("remaining_work", 0) for op in ready_operations) if ready_operations else 1
    if max_remaining_work <= 0:
        max_remaining_work = 1.0
    
    best_score = float('-inf')
    best_index = 0
    
    for i, op in enumerate(ready_operations):
        remaining_work = op.get("remaining_work", 0)
        machine_options = op.get("machine_options", {})
        remaining_ops = op.get("remaining_ops", 0)
        
        if not machine_options:
            score = float('-inf')
        else:
            durations = list(machine_options.values())
            min_duration = min(durations) if durations else 1.0
            avg_duration = sum(durations) / len(durations) if durations else 1.0
            
            # Soft-min: smooth surrogate for minimum duration
            # soft_min = min_d - log(sum(exp(-(d - min_d))))
            if durations:
                min_d = min_duration
                # Clamp to avoid overflow in exp
                exp_sum = 0.0
                for d in durations:
                    diff = min_d - d
                    if diff > 50:  # exp(-50) is negligible
                        exp_sum += 1.0
                    else:
                        exp_sum += np.exp(-diff)
                log_sum = np.log(exp_sum) if exp_sum > 0 else 0.0
                soft_min_duration = min_d - log_sum
            else:
                soft_min_duration = 1.0

            if min_duration <= 0:
                min_duration = 1.0
            if avg_duration <= 0:
                avg_duration = 1.0
            if soft_min_duration <= 0:
                soft_min_duration = 1.0
            
            # Feature 1: Constraint tightness (normalized to [0,1])
            num_options = len(machine_options)
            f1 = 1.0 / num_options  # linear inverse, simpler and more stable
            
            # Feature 2: Normalized remaining work (urgency)
            f2 = remaining_work / max_remaining_work
            
            # Feature 3: Shortest processing time (SPT) - normalized
            # Use 1/soft_min but cap to avoid extreme values
            f3 = 1.0 / (soft_min_duration + 1.0)  # +1 for stability
            
            # Feature 4: Remaining ops in job (urgency of completing the job)
            if remaining_ops > 0:
                f4 = 1.0 / remaining_ops
            else:
                f4 = 1.0
            
            # Feature 5: WSPT proxy - normalized work * SPT
            f5 = f2 * f3
            
            # Feature 6: Bottleneck mitigation via spread
            if len(durations) > 1:
                d_arr = np.array(durations, dtype=float)
                spread = np.std(d_arr)
                f6 = 1.0 / (1.0 + spread / max(avg_duration, 1e-9))
            else:
                f6 = 1.0
            
            # Feature 7: Speed advantage of best machine
            if len(durations) > 1 and avg_duration > 0:
                f7 = min_duration / avg_duration
            else:
                f7 = 1.0
            
            # Weighted combination with tuned weights
            # Emphasis on constraint tightness, SPT, and work urgency
            w1 = 3.0  # constraint tightness
            w2 = 2.0  # normalized remaining work
            w3 = 4.0  # SPT (soft, normalized)
            w4 = 1.0  # remaining ops
            w5 = 2.0  # WSPT combo
            w6 = 2.0  # bottleneck mitigation
            w7 = 1.0  # speed advantage
            
            score = w1 * f1 + w2 * f2 + w3 * f3 + w4 * f4 + w5 * f5 + w6 * f6 + w7 * f7
            
            if score > best_score:
                best_score = score
                best_index = i
    
    return int(best_index)
# EVOLVE-BLOCK-END
