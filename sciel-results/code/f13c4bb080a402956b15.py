import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Worst-Fit heuristic:
    # Place the item in the bin with the largest remaining capacity.
    # This strategy helps maintain "room" in bins for potentially larger future items,
    # reducing fragmentation compared to Best-Fit, which is particularly beneficial
    # for item size distributions like Weibull that have a heavy tail.
    return bins - item
