"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 17
NUM_INPUTS = 1 << N  # 131072
DEADLINE = 85  # seconds, safety margin before 90s kill

# EVOLVE-BLOCK-START

def _make_all_inputs_packed():
    """Create array of all 2^N inputs as uint64 array (each element is a packed N-bit value)."""
    return np.arange(NUM_INPUTS, dtype=np.uint64)

def _apply_comparator_packed(inputs, i, j):
    """Apply comparator (i, j) to packed inputs. min to i, max to j."""
    mask_i = np.uint64(1) << i
    mask_j = np.uint64(1) << j
    bit_i = (inputs & mask_i) != 0
    bit_j = (inputs & mask_j) != 0
    
    # min goes to i: both must be 1
    # max goes to j: at least one must be 1
    new_i = bit_i & bit_j
    new_j = bit_i | bit_j
    
    # Clear both bits, set new values
    inputs = inputs & ~(mask_i | mask_j)
    inputs = np.where(new_i, inputs | mask_i, inputs)
    inputs = np.where(new_j, inputs | mask_j, inputs)
    return inputs

def _is_sorting_network_packed(comparators, inputs):
    """Check if network sorts all inputs using zero-one principle, with packed representation."""
    wires = inputs.copy()
    for i, j in comparators:
        wires = _apply_comparator_packed(wires, i, j)
    
    # Check if each row is sorted (non-decreasing)
    # A row is sorted if for all k < N-1, bit k <= bit k+1
    # i.e., no pattern where bit k=1 and bit k+1=0
    for k in range(N - 1):
        mask_k = np.uint64(1) << k
        mask_k1 = np.uint64(1) << (k + 1)
        # Check: bit k is 1 AND bit k+1 is 0
        bad = (wires & mask_k) != 0
        bad &= (wires & mask_k1) == 0
        if np.any(bad):
            return False
    return True

def construct():
    start_time = time.time()
    
    # Base: odd-even transposition
    comps = [(i, i + 1) for phase in range(N)
             for i in range(phase % 2, N - 1, 2)]
    
    input_matrix = _make_all_inputs_packed()
    
    # Verify base is correct
    if not _is_sorting_network_packed(comps, input_matrix):
        # Fallback: should not happen, but return the base if it's somehow broken
        return comps
    
    # Iteratively remove redundant comparators
    changed = True
    while changed and (time.time() - start_time) < DEADLINE - 5:
        changed = False
        for idx in range(len(comps)):
            if (time.time() - start_time) > DEADLINE - 5:
                break
            # Try removing comparator at idx
            new_comps = comps[:idx] + comps[idx+1:]
            if _is_sorting_network_packed(new_comps, input_matrix):
                comps = new_comps
                changed = True
                break  # Restart loop since indices changed
    
    return comps

# EVOLVE-BLOCK-END
