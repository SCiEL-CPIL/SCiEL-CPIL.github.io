"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import math

# Structural change: Lookahead-based dispatch with machine contention awareness.
#
# Instead of a single-step greedy rule, we:
# 1. Compute a contention-aware priority for each ready operation
# 2. For top candidates, do a 2-step lookahead simulation to estimate
#    which choice leads to lower projected makespan
# 3. Select the machine that minimizes contention for the chosen operation
#
# Key insight: In flexible job-shop, machine selection matters as much as
# operation selection. Contention (multiple ops wanting the same machine)
# causes idle time and delays. We penalize operations whose best machines
# are heavily contested.


def _compute_contention_map(ready_operations):
    """Count how many ready operations can use each machine."""
    contention = {}
    for op in ready_operations:
        for machine in op["machine_options"]:
            contention[machine] = contention.get(machine, 0) + 1
    return contention


def _score_operation(op, contention, total_ops):
    """
    Score a single ready operation.
    Higher score = more urgent/better to dispatch now.
    
    Components:
    - remaining_work: dominant factor (critical path)
    - urgency: jobs with fewer remaining ops are closer to completion
    - machine contention penalty: if the best machine is heavily contested,
      it may be better to use an alternative or wait
    """
    rw = op["remaining_work"]
    ro = op["remaining_ops"]
    machine_options = op["machine_options"]
    
    if not machine_options:
        return rw * 1000.0
    
    min_dur = min(machine_options.values())
    best_machine = min(machine_options, key=machine_options.get)
    
    # Contention on best machine
    best_contention = contention.get(best_machine, 1)
    
    # Second-best machine duration (for flexibility assessment)
    sorted_machines = sorted(machine_options.items(), key=lambda x: x[1])
    if len(sorted_machines) >= 2:
        second_dur = sorted_machines[1][1]
        second_machine = sorted_machines[1][0]
        second_contention = contention.get(second_machine, 1)
    else:
        second_dur = min_dur
        second_contention = best_contention
    
    # Flexibility: how many machines are available and not too contested
    flexible = sum(1 for m in machine_options if contention.get(m, 1) <= 2)
    
    # Urgency factor: jobs with high remaining_work relative to remaining_ops
    # are on the critical path. Also, jobs near completion (low ro) should
    # be finished to free up... actually in job-shop, finishing a job early
    # doesn't free machines, but it reduces the tail of the makespan.
    
    # Score components:
    # 1. remaining_work (dominant)
    # 2. Urgency: rw / ro (average remaining processing time per op) - higher means
    #    each remaining op is "heavier", so this job is more critical
    avg_op_work = rw / max(ro, 1)
    
    # 3. Contention penalty: if best machine is heavily contested, reduce score
    #    unless there's a good alternative
    contention_penalty = 0.0
    if best_contention > 2:
        # Heavy contention on best machine
        # If second machine is also good, less penalty
        dur_ratio = second_dur / max(min_dur, 0.001)
        if dur_ratio < 1.5:
            # Good alternative exists, mild penalty
            contention_penalty = 0.1 * (best_contention - 2)
        else:
            # No good alternative, heavier penalty
            contention_penalty = 0.3 * (best_contention - 2)
    
    # 4. Flexibility bonus: more available machines = more scheduling freedom
    flexibility_bonus = 0.05 * flexible
    
    # Composite score: higher is better
    score = rw + 2.0 * avg_op_work - contention_penalty + flexibility_bonus
    
    return score


def _lookahead_estimate(ready_operations, chosen_idx, time):
    """
    Rough lookahead: after dispatching chosen_idx, simulate what the
    next decision point might look like. Returns an estimated makespan
    contribution (lower is better).
    
    We simulate: remove chosen op, see what's left, and estimate the
    "pressure" from remaining ready ops.
    """
    if chosen_idx < 0 or chosen_idx >= len(ready_operations):
        return 0.0
    
    chosen = ready_operations[chosen_idx]
    chosen_rw = chosen["remaining_work"]
    chosen_ro = chosen["remaining_ops"]
    
    # After dispatching, this job's remaining work decreases by the duration
    # on the chosen machine (we don't know exact, so use min duration)
    machine_options = chosen["machine_options"]
    if machine_options:
        min_dur = min(machine_options.values())
    else:
        min_dur = 0
    
    # Estimate: how much "work pressure" remains in the system
    total_remaining = 0.0
    for i, op in enumerate(ready_operations):
        if i == chosen_idx:
            # This op is being dispatched; its remaining work decreases
            new_rw = max(chosen_rw - min_dur, 0)
            total_remaining += new_rw
        else:
            total_remaining += op["remaining_work"]
    
    # Also account for the fact that other jobs have remaining ops not in ready set
    # We can't see those, so we just use what's available
    
    # Pressure metric: total remaining work / number of ready ops (after dispatch)
    n_after = len(ready_operations) - 1
    if n_after <= 0:
        return 0.0
    
    pressure = total_remaining / n_after
    
    # If the chosen op was the one with highest remaining work, pressure drops more
    # This is a rough proxy
    
    return pressure


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next, and optionally which machine.
    
    Returns:
        - int: index of the ready operation to start, OR
        - (int, machine): index and preferred machine
    
    Strategy (lookahead + contention-aware):
        1. Compute contention map across all ready operations
        2. Score each operation using contention-aware heuristic
        3. For top 3 candidates, do a lightweight lookahead
        4. Select the best candidate and its optimal machine
    """
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    
    # Deadline guard
    start_time = _time.time()
    deadline = start_time + 10.0  # 10s max for dispatch logic
    
    # Step 1: Compute contention map
    contention = _compute_contention_map(ready_operations)
    
    # Step 2: Score all operations
    scored = []
    for i, op in enumerate(ready_operations):
        s = _score_operation(op, contention, n)
        scored.append((s, i))
    
    # Sort by score descending
    scored.sort(key=lambda x: x[0], reverse=True)
    
    # Step 3: Lookahead for top candidates (up to 3)
    top_k = min(3, n)
    best_idx = scored[0][1]
    best_combined = float('-inf')
    
    for rank in range(top_k):
        s, idx = scored[rank]
        
        # Check deadline
        if _time.time() > deadline - 1.0:
            break
        
        # Lookahead estimate (lower is better)
        la = _lookahead_estimate(ready_operations, idx, time)
        
        # Combine: high score is good, low lookahead pressure is good
        # Normalize lookahead: subtract from score
        combined = s - 0.5 * la
        
        if combined > best_combined:
            best_combined = combined
            best_idx = idx
    
    # Step 4: Choose machine for the selected operation
    chosen_op = ready_operations[best_idx]
    machine_options = chosen_op["machine_options"]
    
    if not machine_options:
        return best_idx
    
    # Machine selection: minimize a combination of duration and contention
    # Prefer machines with lower duration, but also consider contention
    # A machine with slightly higher duration but much lower contention
    # may be better overall
    
    best_machine = None
    best_machine_score = float('inf')
    
    for machine, dur in machine_options.items():
        m_contention = contention.get(machine, 1)
        # Machine score: duration + contention penalty
        # Weight contention by how many other ops want this machine
        m_score = dur + 0.5 * (m_contention - 1)
        
        if m_score < best_machine_score:
            best_machine_score = m_score
            best_machine = machine
    
    return (best_idx, best_machine)


# EVOLVE-BLOCK-END
