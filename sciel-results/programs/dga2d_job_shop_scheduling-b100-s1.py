"""
Job-Shop Scheduling Dispatch Rule
Uses a composite heuristic combining Most Work Remaining (MWR),
Shortest Processing Time (SPT), and Most Operations Remaining (MOR).
"""
import numpy as np

def _get_min_duration(op):
    mo = op.get("machine_options", {})
    if not mo:
        return 0.0
    return min(mo.values())

def _get_remaining_work(op):
    return op.get("remaining_work", 0.0)

def _get_remaining_ops(op):
    return op.get("remaining_ops", 0)

def dispatch(ready_operations, time):
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # Extract features
    remaining_works = np.array([_get_remaining_work(op) for op in ready_operations], dtype=float)
    min_durations = np.array([_get_min_duration(op) for op in ready_operations], dtype=float)
    remaining_ops = np.array([_get_remaining_ops(op) for op in ready_operations], dtype=float)

    # Normalize features to [0, 1]
    def normalize(arr):
        min_val = np.min(arr)
        max_val = np.max(arr)
        if max_val - min_val < 1e-9:
            return np.zeros_like(arr)
        return (arr - min_val) / (max_val - min_val)

    norm_work = normalize(remaining_works)
    norm_dur = normalize(min_durations)
    norm_ops = normalize(remaining_ops)

    # Composite Score
    # We want to maximize remaining work, minimize duration, maximize remaining ops
    # Weights tuned for JSS performance
    w_work = 1.0
    w_dur = 1.0
    w_ops = 0.5

    scores = w_work * norm_work + w_dur * (1.0 - norm_dur) + w_ops * norm_ops

    # Select operation with highest score
    best_idx = int(np.argmax(scores))
    return best_idx
