"""Size-efficient sorting networks on 18 wires.
Uses a hybrid approach: start with a known good network (odd-even transposition),
then apply local search / simulated annealing to reduce comparator count while
maintaining correctness over all 2^18 Boolean inputs.
"""

import time
import random
import numpy as np

N = 18
NUM_BOOLS = 1 << N  # 262144

# Deadline guard
DEADLINE = 85.0  # seconds (5s safety margin before 90s kill)


def _is_sorted_mask(state):
    """Check if the 18-bit state is sorted (all 0s before all 1s)."""
    # A sorted state of k ones: the k most significant bits are 1, rest 0
    # Actually, min goes to wire 0, max to wire j. So sorted means:
    # wire 0 has min, wire 17 has max. For Boolean: all 0s then all 1s.
    # State bit i corresponds to wire i. Sorted means bits 0..k-1 are 0 and k..17 are 1
    # for some k (number of 1s).
    # Equivalently: state is of the form 000...0111...1
    # Check: if state == 0 or state == (2^18 - 1), sorted.
    # Otherwise, find the position of the first 1; all bits below must be 0, all above must be 1.
    # More efficiently: state is sorted iff state & (state - 1) pattern... 
    # Actually for 18 bits: sorted states are 0, 2^17, 2^17+2^16, ..., 2^18-1
    # i.e., states where the 1s form a contiguous suffix.
    # state is sorted iff (state | (state - 1)) & state == 0 ... no.
    # Let's just use: for a sorted state with k ones, state = (1 << N) - (1 << (N-k)) = (2^k - 1) << (N-k)
    # Check: state must be one of these N+1 values.
    # Fast check: state is sorted iff state has no pattern '10' in its binary representation
    # (reading from LSB to MSB, we go 0,0,...,0,1,1,...,1). So no '1' followed by '0'.
    # In other words, once we see a 1, all higher bits must be 1.
    # Equivalent: state & (~((1 << (N - popcount)) - 1)) ... let me just do it simply.
    # For speed with numpy, I'll precompute a lookup or use vectorized ops.
    pass


def _precompute_sorted_mask():
    """Precompute a boolean array: True if the state index is a sorted configuration."""
    mask = np.zeros(NUM_BOOLS, dtype=np.bool_)
    for k in range(N + 1):
        # k ones in the top k positions (wires N-k through N-1)
        val = ((1 << k) - 1) << (N - k)
        mask[val] = True
    return mask


def _apply_comparator_to_mask(mask, i, j):
    """Apply comparator (i, j) to the mask. 
    For each state, after comparing wires i and j:
    - if bit i > bit j (i.e., bit i = 1, bit j = 0), swap them
    - otherwise leave as is
    This transforms the set of states. We need to track which original states map to which new states.
    
    Actually, for the purpose of checking if ALL inputs get sorted, we need to track:
    for each of the 2^18 input states, what is the output state after applying the network.
    Then check if all outputs are sorted.
    
    A more efficient approach: represent the set of "unsorted" states. Initially all 2^18 states.
    After a comparator (i,j), a state becomes sorted if:
    - it was already sorted and stays sorted, OR
    - it was unsorted but the comparator fixes it.
    
    Actually, the cleanest approach for Boolean verification:
    Track for each input state what the current output state is.
    But 262144 * 18 bits = manageable with numpy.
    
    Let me use a different approach: 
    Represent the "unsorted set" as a bitset. Initially all states are "potentially unsorted."
    A comparator (i, j) maps each state s to state s'. 
    For Boolean inputs, comparator (i,j): if bit_i=1 and bit_j=0, swap; else no change.
    
    For tracking: maintain an array `current_state` of length 2^18 where current_state[s] = the state 
    after applying the network so far to input s.
    Initially current_state[s] = s for all s.
    After comparator (i,j): for each s, if current_state[s] has bit_i=1 and bit_j=0, swap those bits.
    
    Then at the end, check if all current_state[s] are in the sorted set.
    """
    pass


def _verify_network(comparators):
    """Verify that the network sorts all 2^18 Boolean inputs.
    Returns True if valid, False otherwise.
    """
    current = np.arange(NUM_BOOLS, dtype=np.int32)
    for i, j in comparators:
        bit_i = (current >> i) & 1
        bit_j = (current >> j) & 1
        # Where bit_i=1 and bit_j=0, swap bits i and j
        mask_swap = (bit_i == 1) & (bit_j == 0)
        # Swap: clear both bits, set j to 1, set i to 0... wait.
        # If bit_i=1, bit_j=0: after swap, bit_i=0, bit_j=1
        # So: clear bit_i, set bit_j
        # current = current & ~(1<<i) | (1<<j) for those where swap happens
        # But only where mask_swap is True
        clear_i = ~(1 << i)
        set_j = (1 << j)
        current = np.where(mask_swap, (current & clear_i) | set_j, current)
    
    # Check if all are sorted
    sorted_mask = _SORTED_MASK
    return np.all(sorted_mask[current])


# Precompute sorted mask at module level
_SORTED_MASK = _precompute_sorted_mask()


def _odd_even_transposition_network():
    """Generate the odd-even transposition sorting network for N wires."""
    comps = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            comps.append((i, i + 1))
    return comps


def _batcher_network():
    """Generate a Batcher sorting network for N wires (power of 2 or pad)."""
    # Batcher's odd-even mergesort for N=18 (pad to 32? No, let's do a simple recursive version)
    # Actually, let's just use the odd-even transposition as baseline and try to improve.
    # For now, return odd-even transposition.
    return _odd_even_transposition_network()


def _random_comparator():
    """Generate a random comparator (i, j) with i < j."""
    i = random.randint(0, N - 2)
    j = random.randint(i + 1, N - 1)
    return (i, j)


def _local_search(initial_comparators, max_time=80.0):
    """Local search to reduce comparator count while maintaining validity."""
    start_time = time.time()
    best_comparators = list(initial_comparators)
    best_len = len(best_comparators)
    
    # Try to remove redundant comparators
    # Strategy: for each comparator, try removing it and check validity.
    # Also try swapping order of adjacent comparators.
    
    current = list(best_comparators)
    current_len = len(current)
    
    # Phase 1: Try removing each comparator
    for idx in range(len(current) - 1, -1, -1):
        if time.time() - start_time > max_time:
            break
        if len(current) <= 68:  # Can't go below lower bound
            break
        
        # Try removing comparator at idx
        trial = current[:idx] + current[idx+1:]
        if _verify_network(trial):
            current = trial
            current_len = len(current)
            if current_len < best_len:
                best_len = current_len
                best_comparators = list(current)
    
    # Phase 2: Try moving comparators (swap adjacent)
    improved = True
    while improved:
        if time.time() - start_time > max_time:
            break
        improved = False
        for idx in range(len(current) - 1):
            if time.time() - start_time > max_time:
                break
            # Swap current[idx] and current[idx+1]
            trial = current[:]
            trial[idx], trial[idx+1] = trial[idx+1], trial[idx]
            if _verify_network(trial):
                # Check if this helps (e.g., allows further removal)
                current = trial
                improved = True
    
    # Phase 3: Try replacing pairs with single comparators
    # For each pair of comparators, see if a single comparator can replace them
    for idx in range(len(current) - 1):
        if time.time() - start_time > max_time:
            break
        if len(current) <= 68:
            break
        # Try replacing current[idx] and current[idx+1] with a single random comparator
        for _ in range(5):
            new_comp = _random_comparator()
            trial = current[:idx] + [new_comp] + current[idx+2:]
            if _verify_network(trial):
                current = trial
                if len(current) < best_len:
                    best_len = len(current)
                    best_comparators = list(current)
                break
    
    return best_comparators


def construct():
    """Construct a sorting network for 18 wires.
    Returns an ordered list of integer pairs (i, j).
    """
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Start with odd-even transposition network
    comparators = _odd_even_transposition_network()
    
    # Verify baseline
    if not _verify_network(comparators):
        # Should not happen, but fallback
        return comparators
    
    # Try local search to reduce size
    if time.time() < deadline - 5:
        comparators = _local_search(comparators, max_time=min(30, deadline - time.time() - 5))
    
    # Final verification
    if not _verify_network(comparators):
        # Fallback to known good network
        comparators = _odd_even_transposition_network()
    
    return comparators
