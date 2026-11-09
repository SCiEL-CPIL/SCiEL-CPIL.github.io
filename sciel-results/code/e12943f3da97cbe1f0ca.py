"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time

_DEADLINE_START = _time.time()
_MAX_WALL = 135.0  # Safety margin before the 150s kill


def _check_deadline(fallback):
    """If we're getting close to the wall-clock limit, return fallback immediately."""
    if _time.time() - _DEADLINE_START > _MAX_WALL:
        return fallback
    return None


def dispatch(ready_operations, time):
    # Check deadline first; if we're over, pick the first available (index 0)
    fallback = 0
    early = _check_deadline(fallback)
    if early is not None:
        return early

    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # Strategy: Composite heuristic combining remaining work with machine-option quality.
    # We use a smooth, composite priority that balances:
    #   1. Remaining work (LRPT) - higher is better
    #   2. Maximum processing time among options (LPT) - higher is better (bottleneck clearing)
    #   3. Minimum processing time among options (SPT) - lower is better (quick clears)
    #   4. Average processing time among options - lower is better (efficiency)
    #
    # The priority is a weighted sum designed to capture the multi-objective trade-off
    # without hard lexicographic ties, which can be brittle.

    best_idx = 0
    best_score = float('-inf')

    for i, op in enumerate(ready_operations):
        rw = op["remaining_work"]
        options = op["machine_options"]
        
        if not options:
            min_dur = 0.0
            max_dur = 0.0
            avg_dur = 0.0
        else:
            vals = list(options.values())
            min_dur = min(vals)
            max_dur = max(vals)
            avg_dur = sum(vals) / len(vals)
        
        # Composite score:
        # - Primary: remaining work (larger is better)
        # - Secondary: max duration (larger is better, clears bottlenecks)
        # - Tertiary: -min duration (smaller is better, quick tasks)
        # - Quaternary: -avg duration (smaller is better, efficiency)
        #
        # We normalize implicitly by the scale of these values.
        # The weights are chosen to prioritize remaining work most heavily,
        # then use duration statistics as tie-breakers with smooth blending.
        
        score = rw + 0.1 * max_dur - 0.05 * min_dur - 0.03 * avg_dur
        
        if score > best_score:
            best_score = score
            best_idx = i

    return best_idx
# EVOLVE-BLOCK-END
