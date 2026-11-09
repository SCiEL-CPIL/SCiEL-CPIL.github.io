"""Open-shop scheduling dispatch rule using soft-min regret-based priority with machine assignment."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np
from collections import defaultdict

# Deadline guard: evaluator kills after 150s wall-clock; use a safety margin.
_DEADLINE_START = _time.monotonic()
_DEADLINE_LIMIT = 140.0  # seconds, leave 10s margin


def _elapsed():
    return _time.monotonic() - _DEADLINE_START


def _soft_min(values, temperature=1.0):
    """
    Compute a smooth approximation of the minimum using log-sum-exp.
    soft_min(x) = -log(sum(exp(-x_i / T))) * T
    As T -> 0, this approaches min(x).
    """
    if len(values) == 0:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    # For numerical stability, subtract the minimum
    v = np.array(values, dtype=float)
    v_min = np.min(v)
    # If all values are equal, just return that value
    if np.allclose(v, v_min):
        return float(v_min)
    # Soft min with temperature
    exp_vals = np.exp(-(v - v_min) / temperature)
    log_sum = np.log(np.sum(exp_vals))
    return float(v_min - temperature * log_sum)


def _soft_max(values, temperature=1.0):
    """
    Compute a smooth approximation of the maximum using log-sum-exp.
    """
    if len(values) == 0:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    v = np.array(values, dtype=float)
    v_max = np.max(v)
    if np.allclose(v, v_max):
        return float(v_max)
    exp_vals = np.exp((v - v_max) / temperature)
    log_sum = np.log(np.sum(exp_vals))
    return float(v_max + temperature * log_sum)


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next in an event-driven open-shop simulation.
    
    Strategy: Regret-based priority with soft-min machine assignment.
    For each ready operation, compute a regret score based on:
    1. Soft-min duration across machine options (smooth surrogate for best machine)
    2. Machine congestion (how loaded the best machine is)
    3. Remaining work in the job
    4. Number of remaining operations
    
    Returns: (index, machine) tuple to specify both the operation and its assigned machine.
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        op = ready_operations[0]
        mo = op["machine_options"]
        if mo:
            best_machine = min(mo.keys(), key=lambda m: mo[m])
            return (0, best_machine)
        return 0

    # Quick deadline check
    if _elapsed() > _DEADLINE_LIMIT - 5.0:
        # Fallback: pick operation with most remaining work
        idx = int(np.argmax([op["remaining_work"] for op in ready_operations]))
        op = ready_operations[idx]
        mo = op["machine_options"]
        if mo:
            best_machine = min(mo.keys(), key=lambda m: mo[m])
            return (idx, best_machine)
        return idx

    # Gather features
    min_durations = []
    soft_min_durations = []
    for op in ready_operations:
        mo = op["machine_options"]
        if mo:
            durations = list(mo.values())
            min_durations.append(min(durations))
            # Use soft-min with small temperature for smoothness
            soft_min_durations.append(_soft_min(durations, temperature=0.5))
        else:
            min_durations.append(0)
            soft_min_durations.append(0)
    
    min_durations = np.array(min_durations, dtype=float)
    soft_min_durations = np.array(soft_min_durations, dtype=float)
    remaining_work = np.array([op["remaining_work"] for op in ready_operations], dtype=float)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=int)
    
    # Estimate machine load: for each machine, sum of durations of ops that can use it
    machine_load = defaultdict(float)
    for op in ready_operations:
        for m, dur in op["machine_options"].items():
            machine_load[m] += dur
    
    # Compute per-operation regret score
    regrets = np.zeros(n, dtype=float)
    best_machines = [None] * n
    
    # Pre-compute normalization factors
    max_rw = np.max(remaining_work) if np.max(remaining_work) > 0 else 1.0
    max_ro = float(np.max(remaining_ops)) if np.max(remaining_ops) > 0 else 1.0
    max_soft_min = np.max(soft_min_durations) if np.max(soft_min_durations) > 0 else 1.0
    total_load = sum(machine_load.values()) + 1e-9
    
    for i in range(n):
        op = ready_operations[i]
        mo = op["machine_options"]
        
        if not mo:
            regrets[i] = remaining_work[i] / max_rw
            best_machines[i] = None
            continue
        
        # Find the least-loaded machine this op can run on (by load, not duration)
        best_machine = None
        best_load = float('inf')
        best_duration = float('inf')
        for m, dur in mo.items():
            load = machine_load.get(m, 0.0)
            if load < best_load:
                best_load = load
                best_machine = m
                best_duration = dur
        
        best_machines[i] = best_machine
        
        # Regret components:
        # 1. Normalized remaining work (more work = more urgent)
        rw_norm = remaining_work[i] / max_rw
        
        # 2. Congestion on the best machine (relative to total load)
        congestion = best_load / total_load
        
        # 3. Normalized remaining ops (more ops left = higher priority)
        ro_norm = remaining_ops[i] / max_ro
        
        # 4. Soft-min duration normalized (shorter ops get priority on congested machines)
        dur_norm = soft_min_durations[i] / max_soft_min
        
        # 5. Flexibility: number of machine options (more options = less constrained)
        flexibility = len(mo) / (float(max(len(op2["machine_options"]) for op2 in ready_operations)) + 1e-9)
        
        # Weighted regret with tuned coefficients
        # Prioritize: high remaining work, high congestion, many remaining ops, short duration, low flexibility
        regret = (
            0.35 * rw_norm +
            0.25 * congestion +
            0.20 * ro_norm +
            0.10 * (1.0 - dur_norm) +
            0.10 * (1.0 - flexibility)
        )
        
        regrets[i] = regret

    # Add a small tie-breaker based on remaining work to ensure determinism
    regrets += remaining_work * 1e-6

    best_idx = int(np.argmax(regrets))
    best_machine = best_machines[best_idx]
    
    if best_machine is not None:
        return (best_idx, best_machine)
    else:
        return best_idx
# EVOLVE-BLOCK-END
