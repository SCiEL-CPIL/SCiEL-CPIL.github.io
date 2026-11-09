"""Improved program: greedy construction + simulated annealing / hill climbing
with efficient AP-sum tracking and a strict deadline guard."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

N = 256
DEADLINE_MARGIN = 5.0  # seconds before the 90s kill to stop


def _max_abs_ap_sum(signs):
    """Return (max_abs_sum, worst_ap) for all arithmetic progressions in [1, N].
    signs is a list or numpy array of length N with values -1 or +1 (0-indexed).
    AP is defined by start s (0-indexed), step d >= 1, and length L >= 1,
    with s + (L-1)*d < N.
    """
    signs = np.asarray(signs, dtype=np.int8)
    max_abs = 0
    worst = None
    # Precompute prefix sums per step? That's expensive for all steps.
    # Instead, for each step d, we can compute sums along residue classes mod d.
    # For a fixed d, positions split into d residue classes. In each class,
    # the positions are r, r+d, r+2d, ... which form a contiguous subsequence.
    # Any AP with step d is a contiguous subsequence within one residue class.
    # So for fixed d, we just need max subarray sum (Kadane-like) on each residue class,
    # but we need max absolute sum, so we track both max and min prefix sums.

    for d in range(1, N + 1):
        # For each residue r in 0..d-1, consider the sequence signs[r], signs[r+d], ...
        # We want the max absolute value of any contiguous subsequence sum.
        # That is max over all i<=j of |sum_{k=i}^j signs[r+kd]|.
        # This equals max(prefix_max - prefix_min) where prefix is cumulative sum.
        for r in range(d):
            if r >= N:
                break
            # Build the sequence of signs at positions r, r+d, r+2d, ...
            # Instead of building a list, we can iterate and track min/max prefix.
            prefix = 0
            min_prefix = 0
            max_prefix = 0
            cur_abs_max = 0
            for pos in range(r, N, d):
                prefix += signs[pos]
                if prefix > max_prefix:
                    max_prefix = prefix
                if prefix < min_prefix:
                    min_prefix = prefix
                # The max absolute subarray sum ending at current position is
                # max(prefix - min_prefix, max_prefix - prefix) but actually
                # the global max absolute is max(max_prefix - min_prefix) over all prefixes.
                # Actually, max subarray sum = max_prefix - min_prefix (when min_prefix occurs before max_prefix).
                # For max absolute, we need max(|sum|) = max(max_subarray, -min_subarray)
                # where max_subarray = max over i<=j of sum(i..j) and min_subarray = min over i<=j.
                # min_subarray = min_prefix - max_prefix (when max_prefix occurs before min_prefix).
                # So max absolute = max(max_prefix - min_prefix, max_prefix - min_prefix) = max_prefix - min_prefix?
                # Wait, no. Let me think again.
                # max_subarray_sum = max_{i<=j} (P[j] - P[i]) where P is prefix sum with P[-1]=0.
                # This equals max over j of (P[j] - min_{i<j} P[i]).
                # min_subarray_sum = min_{i<=j} (P[j] - P[i]) = min over j of (P[j] - max_{i<j} P[i]).
                # So max_abs = max(max_subarray_sum, -min_subarray_sum).
                # We can track this incrementally.
                pass
            # Redo with proper tracking
            prefix = 0
            min_prefix = 0
            max_prefix = 0
            max_sub = 0
            min_sub = 0
            for pos in range(r, N, d):
                prefix += signs[pos]
                # Update max_sub: max over j of (P[j] - min_{i<j} P[i])
                if prefix - min_prefix > max_sub:
                    max_sub = prefix - min_prefix
                # Update min_sub: min over j of (P[j] - max_{i<j} P[i])
                if prefix - max_prefix < min_sub:
                    min_sub = prefix - max_prefix
                if prefix > max_prefix:
                    max_prefix = prefix
                if prefix < min_prefix:
                    min_prefix = prefix
            cur_abs_max = max(max_sub, -min_sub)
            if cur_abs_max > max_abs:
                max_abs = cur_abs_max
                # We could record worst AP but it's expensive; skip for speed
                worst = None
    return max_abs, worst


def _max_abs_ap_sum_fast(signs):
    """Faster version using numpy for the inner loop."""
    signs = np.asarray(signs, dtype=np.int8)
    max_abs = 0
    for d in range(1, N + 1):
        for r in range(min(d, N)):
            # Extract the subsequence
            seq = signs[r::d]
            if len(seq) == 0:
                continue
            # Compute prefix sums
            prefix = np.cumsum(seq, dtype=np.int32)
            prefix_with_zero = np.concatenate(([0], prefix))
            # max_subarray = max(prefix) - min(prefix) but need ordering constraint.
            # Actually for max subarray sum, we need:
            # max over j of (prefix[j] - min(prefix[0..j]))
            running_min = np.minimum.accumulate(prefix_with_zero)
            max_sub = np.max(prefix_with_zero - running_min)
            running_max = np.maximum.accumulate(prefix_with_zero)
            min_sub = np.min(prefix_with_zero - running_max)
            cur_abs = max(max_sub, -min_sub)
            if cur_abs > max_abs:
                max_abs = cur_abs
    return max_abs


def construct():
    start_time = time.time()
    rng = random.Random(42)

    # Greedy initialization: build signs one by one to minimize local AP sums.
    # For each position i, try +1 and -1, pick the one that minimizes the
    # maximum absolute AP sum involving position i (or a proxy).
    # Simpler: just use random init, then optimize.
    
    signs = [rng.choice((-1, 1)) for _ in range(N)]
    current_score = _max_abs_ap_sum_fast(signs)
    best_signs = signs[:]
    best_score = current_score

    # Simulated annealing
    T = 5.0
    T_min = 0.1
    cooling = 0.9995
    steps_since_improve = 0
    max_steps_no_improve = 5000

    while time.time() - start_time < 90 - DEADLINE_MARGIN:
        # Flip a random position
        i = rng.randrange(N)
        old_val = signs[i]
        signs[i] = -old_val
        new_score = _max_abs_ap_sum_fast(signs)

        delta = new_score - current_score
        if delta <= 0 or rng.random() < np.exp(-delta / T):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_signs = signs[:]
                steps_since_improve = 0
            else:
                steps_since_improve += 1
        else:
            signs[i] = old_val
            steps_since_improve += 1

        T *= cooling
        if T < T_min:
            T = T_min
            # Restart with higher temperature occasionally
            if steps_since_improve > max_steps_no_improve:
                T = 3.0
                steps_since_improve = 0
                # Perturb best solution
                signs = best_signs[:]
                for _ in range(5):
                    j = rng.randrange(N)
                    signs[j] = -signs[j]
                current_score = _max_abs_ap_sum_fast(signs)

        # Check deadline
        if time.time() - start_time > 90 - DEADLINE_MARGIN:
            break

    return best_signs
# EVOLVE-BLOCK-END
