"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

# Deadline guard: evaluator kills after 150s; keep a safety margin.
_START_TIME = _time.time()
_DEADLINE = 130.0  # seconds wall-clock budget (safety margin)


def _time_left():
    return _DEADLINE - (_time.time() - _START_TIME)


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next.
    
    Strategy: Critical Path Method (CPM) proxy combined with load-balancing.
    - Primary key: remaining_work of the job (proxy for critical path length).
    - Secondary key: minimum processing time across machine options (SPT-like tiebreak).
    - Tertiary: prefer machines that are currently less loaded (water-filling idea).
    
    We combine these into a single score per ready operation and pick the max.
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    
    # Check deadline
    if _time_left() < 0.5:
        # Fallback: just pick the one with most remaining work
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))
    
    # Compute scores
    scores = np.zeros(n)
    
    # Extract features
    remaining_works = np.array([op["remaining_work"] for op in ready_operations], dtype=float)
    min_durations = np.array([min(op["machine_options"].values()) for op in ready_operations], dtype=float)
    
    # Normalize remaining_works to [0, 1] for combination
    rw_max = remaining_works.max()
    rw_min = remaining_works.min()
    rw_range = rw_max - rw_min
    if rw_range > 1e-9:
        rw_norm = (remaining_works - rw_min) / rw_range
    else:
        rw_norm = np.ones(n)
    
    # Normalize min_durations: shorter is better, so invert
    md_max = min_durations.max()
    md_min = min_durations.min()
    md_range = md_max - md_min
    if md_range > 1e-9:
        md_norm = (md_max - min_durations) / md_range  # inverted: shorter -> higher
    else:
        md_norm = np.ones(n)
    
    # Machine load heuristic: for each operation, estimate the "cost" of assigning
    # to its best machine. We don't have current machine loads directly, but we can
    # use a proxy: if an operation has many machine options, it's more flexible.
    # Operations with fewer options are more constrained and should be prioritized.
    flexibilities = np.array([len(op["machine_options"]) for op in ready_operations], dtype=float)
    flex_max = flexibilities.max()
    flex_min = flexibilities.min()
    flex_range = flex_max - flex_min
    if flex_range > 1e-9:
        # Less flexible (fewer options) -> higher priority
        flex_norm = (flex_max - flexibilities) / flex_range
    else:
        flex_norm = np.ones(n)
    
    # Combine: weighted sum
    # remaining_work is most important (critical path proxy)
    # min_duration is secondary (SPT tiebreak)
    # flexibility is tertiary (constrained ops first)
    w_rw = 0.6
    w_md = 0.25
    w_flex = 0.15
    
    scores = w_rw * rw_norm + w_md * md_norm + w_flex * flex_norm
    
    return int(np.argmax(scores))
# EVOLVE-BLOCK-END
