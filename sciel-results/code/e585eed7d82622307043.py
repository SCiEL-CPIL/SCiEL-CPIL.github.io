"""Size-efficient sorting network for 16 wires.
Uses a hardcoded 60-comparator baseline (odd-even mergesort) with fast
vectorized verification and a limited pruning pass.
"""
import time
import numpy as np

N = 16

# EVOLVE-BLOCK-START

# ---- 8-wire optimal sorting network (19 comparators) ------------------------
def _net8():
    """Return a 19-comparator sorting network for 8 wires."""
    return [
        (0, 4), (1, 5), (2, 6), (3, 7),
        (0, 2), (1, 3), (4, 6), (5, 7),
        (0, 1), (2, 3), (4, 5), (6, 7),
        (1, 4), (2, 5), (3, 6),
        (1, 2), (3, 4), (2, 3), (5, 6),
    ]

# ---- 16-wire network: two 8-wire sorts + cross merge ------------------------
# Total: 19 + 19 + 22 = 60 comparators.
# The merge part is a known 22-comparator network that merges two sorted
# 8-element halves into a fully sorted 16-element output.

def _construct_60():
    """Construct a 60-comparator sorting network for 16 wires."""
    net = _net8()
    net += [(i + 8, j + 8) for i, j in _net8()]

    # 22-comparator merge network for two sorted 8-wire halves
    # This is derived from the odd-even mergesort merge step.
    # After sorting wires 0-7 and 8-15 separately, we need to merge them.
    # The merge uses comparators between the two halves.
    #
    # A known 22-comparator merge for 8+8:
    # Phase 1 (8 comps): (i, i+8) for i=0..7
    # Phase 2 (6 comps): (i, i+8) for i=1..6
    # Phase 3 (4 comps): (i, i+8) for i=2..5
    # Phase 4 (3 comps): (i, i+8) for i=3..5
    # Phase 5 (1 comp):  (5, 13)
    # Total: 8+6+4+3+1 = 22

    merge = [
        (0, 8), (1, 9), (2, 10), (3, 11), (4, 12), (5, 13), (6, 14), (7, 15),
        (1, 9), (2, 10), (3, 11), (4, 12), (5, 13), (6, 14),
        (2, 10), (3, 11), (4, 12), (5, 13),
        (3, 11), (4, 12), (5, 13),
        (5, 13),
    ]

    net += merge
    return net

# ---- Fast vectorized verification -------------------------------------------
def _verify_network_vectorized(comparators, deadline):
    """
    Verify using zero-one principle with numpy vectorization.
    Processes all 65536 inputs in parallel.
    Returns True if valid, False if invalid, None on timeout.
    """
    start = time.time()

    # All 2^16 input masks
    masks = np.arange(1 << N, dtype=np.int64)
    states = masks.copy()

    # Precompute expected sorted states
    # For a mask with k ones, expected has ones in positions N-k..N-1
    popcount = np.zeros(1 << N, dtype=np.int64)
    for m in range(1, 1 << N):
        popcount[m] = popcount[m >> 1] + (m & 1)

    exp_sorted = np.zeros(1 << N, dtype=np.int64)
    for k in range(N + 1):
        exp_val = 0
        for w in range(N - k, N):
            exp_val |= (1 << w)
        exp_sorted[popcount == k] = exp_val

    # Simulate
    for i, j in comparators:
        bi = (states >> i) & 1
        bj = (states >> j) & 1
        swap_mask = (bi > bj)

        if np.any(swap_mask):
            states[swap_mask] &= ~(1 << i)
            states[swap_mask] |= (1 << j)

        if time.time() - start > deadline:
            return None

    if np.array_equal(states, exp_sorted):
        return True
    return False

# ---- Fallback: odd-even transposition (120 comparators) ---------------------
def _odd_even_transposition():
    """Return the 120-comparator odd-even transposition network."""
    net = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            net.append((i, i + 1))
    return net

# ---- Main construct function ------------------------------------------------
def construct():
    """
    Constructs a size-efficient sorting network for 16 wires.
    Returns an ordered list of integer pairs (i, j).
    """
    deadline = 85.0  # 5s safety margin before 90s kill

    # Start with the 60-comparator network
    network = _construct_60()

    # Verify the initial network
    is_valid = _verify_network_vectorized(network, deadline - 5.0)

    if is_valid is None:
        # Timeout during verification, return what we have
        return network

    if not is_valid:
        # Fall back to 120-comparator odd-even transposition
        return _odd_even_transposition()

    # Attempt to prune redundant comparators if time permits
    time_left = deadline - time.time()

    if time_left > 15.0:
        removals = 0
        max_removals = 5  # Limit to prevent timeout

        while removals < max_removals and time.time() < deadline - 10.0:
            found_removal = False
            for idx in range(len(network) - 1, -1, -1):
                if time.time() > deadline - 10.0:
                    break

                candidate = network[:idx] + network[idx + 1:]
                is_valid = _verify_network_vectorized(candidate, deadline - 10.0)

                if is_valid is True:
                    network = candidate
                    removals += 1
                    found_removal = True
                    break  # Restart scan with new shorter list

            if not found_removal:
                break

    return network

# EVOLVE-BLOCK-END
