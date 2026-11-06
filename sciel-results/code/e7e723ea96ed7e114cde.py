"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 17
NUM_BOOL_INPUTS = 1 << N  # 131072
QUICK_CHECK_SIZE = 1000  # Number of random inputs for quick checks

# Precompute all Boolean inputs as a numpy array
def _get_all_bool_inputs():
    """Generate all 2^17 Boolean inputs as a numpy array of shape (131072, 17)."""
    inputs = np.zeros((NUM_BOOL_INPUTS, N), dtype=np.uint8)
    for j in range(N):
        block_size = 1 << (j + 1)
        half_block = 1 << j
        pattern = np.array([0] * half_block + [1] * half_block, dtype=np.uint8)
        num_tiles = NUM_BOOL_INPUTS // block_size
        inputs[:, j] = np.tile(pattern, num_tiles)
    return inputs

_ALL_INPUTS = _get_all_bool_inputs()

# Precompute per-wire sorted-input masks for fast full verification.
# _FULL_SORTED_MASKS[k] is 1 on rows where wire k >= wire k+1 (a violation).
_FULL_SORTED_MASKS = [(_ALL_INPUTS[:, k] > _ALL_INPUTS[:, k + 1]).astype(np.uint8)
                      for k in range(N - 1)]

# Precompute a subset for quick checks
_quick_rng = np.random.RandomState(42)
_QUICK_INPUTS = _quick_rng.randint(0, 2, size=(QUICK_CHECK_SIZE, N)).astype(np.uint8)
_QUICK_SORTED_MASKS = [(_QUICK_INPUTS[:, k] > _QUICK_INPUTS[:, k + 1]).astype(np.uint8)
                       for k in range(N - 1)]

def _simulate_network(network, inputs):
    """Simulate the network on given inputs. Modifies inputs in place."""
    for i, j in network:
        min_val = np.minimum(inputs[:, i], inputs[:, j])
        max_val = np.maximum(inputs[:, i], inputs[:, j])
        inputs[:, i] = min_val
        inputs[:, j] = max_val

def _quick_check(network):
    """Quick check using a subset of inputs. Returns True if passes."""
    inputs = _QUICK_INPUTS.copy()
    _simulate_network(network, inputs)
    for k in range(N - 1):
        if np.any(inputs[:, k] > inputs[:, k + 1]):
            return False
    return True

def _is_valid(network):
    """Full check using all Boolean inputs (vectorized with precomputed masks)."""
    inputs = _ALL_INPUTS.copy()
    _simulate_network(network, inputs)
    for k in range(N - 1):
        if np.any(_FULL_SORTED_MASKS[k] & inputs[:, k]):
            return False
    return True

def _construct_odd_even_transposition():
    """Generate the odd-even transposition sorting network (136 comparators)."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 85  # Safety margin before the 90s evaluator kill

    # Start with odd-even transposition (136 comparators)
    network = _construct_odd_even_transposition()
    best_network = network[:]
    best_count = len(network)

    # Verify initial network (it is valid, but guard against surprises)
    if not _is_valid(network):
        return best_network

    # Phase 1: Greedy removal (multi-pass to escape local minima)
    while time.time() < deadline:
        removed_any = False
        for idx in range(len(network) - 1, -1, -1):
            if time.time() >= deadline:
                break
            candidate = network[:idx] + network[idx + 1:]
            if _quick_check(candidate):
                if _is_valid(candidate):
                    network = candidate
                    best_network = network[:]
                    best_count = len(network)
                    removed_any = True
                    break
        if not removed_any:
            break

    # Phase 2: Fix comparator pairs (replace a redundant comparator with a
    # useful one, e.g. a missing adjacent pair). Keeps the comparator count.
    while time.time() < deadline:
        improved = False
        for idx in range(len(network)):
            if time.time() >= deadline:
                break
            i, j = network[idx]
            # Candidate replacements: adjacent pairs (most useful) + the original
            replacements = [(i, i + 1), (max(0, i - 1), i)]
            if j - i == 1:
                # already adjacent; also try the symmetric swap direction
                replacements.append((j, i))
            for new_i, new_j in replacements:
                if not (0 <= new_i < new_j < N):
                    continue
                if (new_i, new_j) == (i, j):
                    continue
                candidate = network[:idx] + [(new_i, new_j)] + network[idx + 1:]
                if _quick_check(candidate):
                    if _is_valid(candidate):
                        network = candidate
                        best_network = network[:]
                        best_count = len(network)
                        improved = True
                        break
            if improved:
                break
        if not improved:
            break

    # Phase 3: Another removal pass (fixes may have exposed new redundancies)
    while time.time() < deadline:
        removed_any = False
        for idx in range(len(network) - 1, -1, -1):
            if time.time() >= deadline:
                break
            candidate = network[:idx] + network[idx + 1:]
            if _quick_check(candidate):
                if _is_valid(candidate):
                    network = candidate
                    best_network = network[:]
                    best_count = len(network)
                    removed_any = True
                    break
        if not removed_any:
            break

    return best_network
# EVOLVE-BLOCK-END
