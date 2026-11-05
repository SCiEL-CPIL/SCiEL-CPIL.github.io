"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

_DEADLINE = _time.time() + 140.0  # safety margin before 150s kill


def _deadline_exceeded():
    return _time.time() > _DEADLINE


def _soft_min(durs, temperature=0.3):
    """Smooth surrogate for min using log-sum-exp.
    As T->0 this approaches hard min; for small T it's a smooth, differentiable
    approximation that handles near-ties gracefully."""
    d = np.asarray(durs, dtype=np.float64)
    if len(d) == 1:
        return float(d[0])
    d_min = d.min()
    scaled = -(d - d_min) / temperature
    scaled = np.clip(scaled, -50, 50)
    log_sum = np.log(np.sum(np.exp(scaled)))
    return d_min - temperature * log_sum


def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    n = len(ready_operations)

    # Fast path for single op
    if n == 1:
        return 0

    # Deadline guard: if we're near the limit, fall back to a cheap rule
    if _deadline_exceeded():
        # Pick the op with the most remaining work
        best_idx = 0
        best_work = ready_operations[0]["remaining_work"]
        for i in range(1, n):
            if ready_operations[i]["remaining_work"] > best_work:
                best_work = ready_operations[i]["remaining_work"]
                best_idx = i
        return best_idx

    # Extract key metrics
    remaining_works = np.array([op["remaining_work"] for op in ready_operations], dtype=np.float64)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=np.float64)

    # Soft-min duration and avg duration across machine options
    best_durs = np.empty(n, dtype=np.float64)
    avg_durs = np.empty(n, dtype=np.float64)

    # Count how many ready ops compete for each machine (load-balancing signal)
    machine_competition = {}
    for op in ready_operations:
        for m in op["machine_options"]:
            machine_competition[m] = machine_competition.get(m, 0) + 1

    for i, op in enumerate(ready_operations):
        durs = np.array(list(op["machine_options"].values()), dtype=np.float64)
        best_durs[i] = _soft_min(durs, temperature=0.3)
        avg_durs[i] = np.mean(durs)

    # Normalization to [0, 1] (higher = more urgent / more efficient)
    def _normalize(arr):
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-12:
            return np.ones_like(arr) * 0.5
        return (arr - mn) / (mx - mn)

    # Urgency: more remaining work and more remaining ops => more urgent
    urgency_work = _normalize(remaining_works)
    urgency_ops = _normalize(remaining_ops)
    urgency = 0.7 * urgency_work + 0.3 * urgency_ops

    # Efficiency: shorter operations reduce idle time; invert so shorter = higher
    eff_best = _normalize(-best_durs)
    eff_avg = _normalize(-avg_durs)
    efficiency = 0.6 * eff_best + 0.4 * eff_avg

    # Weighted SPT signal: remaining_work / best_duration captures
    # "critical path per unit time" — jobs with lots of work that can be
    # done quickly on a good machine should be prioritized.
    # Guard against zero duration
    safe_best_durs = np.maximum(best_durs, 1e-9)
    weighted_spt_raw = remaining_works / safe_best_durs
    weighted_spt = _normalize(weighted_spt_raw)

    # Look-ahead bottleneck heuristic:
    # For each operation, estimate the earliest completion time if assigned
    # to the least-loaded eligible machine (fewest competing ready ops).
    # This captures immediate machine contention impact.
    lookahead_scores = np.empty(n, dtype=np.float64)
    for i, op in enumerate(ready_operations):
        # Find the machine option that minimizes effective cost:
        # duration * (1 + competition_penalty) where competition_penalty
        # scales with how many other ready ops want that machine
        best_cost = float('inf')
        for m, dur in op["machine_options"].items():
            comp = machine_competition.get(m, 1)
            # Effective completion time considering contention:
            # If k ops compete for a machine, each may wait ~dur/k on average
            # but we use a simpler penalty: dur * (1 + 0.5*(comp-1)/max_comp)
            comp_penalty = 0.5 * (comp - 1) / max(comp, 1)
            cost = dur * (1.0 + comp_penalty)
            if cost < best_cost:
                best_cost = cost
        
        lookahead_scores[i] = best_cost

    # Invert and normalize: lower lookahead score = higher priority
    load_balance = _normalize(-lookahead_scores)

    # Combined score: urgency + weighted SPT dominant, efficiency secondary, look-ahead tertiary
    scores = 0.8 * urgency + 0.3 * weighted_spt + 0.35 * efficiency + 0.25 * load_balance

    # Pick the operation with the highest combined score
    best_idx = int(np.argmax(scores))

    # Optionally choose the best machine considering both duration and contention
    best_op = ready_operations[best_idx]
    best_machine = min(
        best_op["machine_options"].items(),
        key=lambda item: item[1] * (1.0 + 0.5 * (machine_competition.get(item[0], 1) - 1) / max(machine_competition.get(item[0], 1), 1))
    )[0]

    return best_idx
# EVOLVE-BLOCK-END
