"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 16

# EVOLVE-BLOCK-START
def _apply_comp(state, i, j):
    """Apply a comparator to a 16-bit state. state is an int where bit k is wire k."""
    bi = (state >> i) & 1
    bj = (state >> j) & 1
    if bi < bj:
        # swap: set i to 1, j to 0
        state |= (1 << i)
        state &= ~(1 << j)
    return state

def _batcher_sort(n):
    """Generate Batcher's odd-even merge sort network for n wires (n power of 2)."""
    if n == 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    half = n // 2
    # Recursively sort each half
    left = _batcher_sort(half)
    right = [(i + half, j + half) for (i, j) in _batcher_sort(half)]
    
    # Odd-even merge
    merge = _batcher_merge(n)
    
    return left + right + merge

def _batcher_merge(n):
    """Odd-even merge for n wires (n power of 2, n >= 2)."""
    if n == 2:
        return [(0, 1)]
    
    half = n // 2
    # Recursively merge each half
    left = _batcher_merge(half)
    right = [(i + half, j + half) for (i, j) in _batcher_merge(half)]
    
    # Cross comparisons
    cross = []
    for i in range(half):
        cross.append((i, i + half))
    
    # Interleaved recursive merges
    interleave = []
    for k in range(1, half):
        # This is the standard odd-even merge step
        # Actually, let me use the standard Batcher merge construction
        pass
    
    # Let me use a cleaner recursive definition
    # Standard odd-even merge for n elements:
    # 1. Split into two halves of n/2
    # 2. Recursively merge each half
    # 3. Compare (i, i+n/2) for i in 0..n/2-1
    # 4. Interleave and recursively merge
    
    # Actually, the standard Batcher odd-even merge is:
    # For n = 2^k:
    #   If n == 2: return [(0,1)]
    #   Else:
    #     left = merge(n/2)
    #     right = merge(n/2) shifted by n/2
    #     cross = [(i, i+n/2) for i in range(n/2)]
    #     Then interleave: for i in 1..n/2-1 step 2:
    #       left_sub = merge(i) ... this gets complex
    
    # Let me just use a known-correct implementation
    # Batcher's merge for n wires:
    if n == 2:
        return [(0, 1)]
    
    result = []
    # Recursive merges on halves
    for i, j in _batcher_merge(half):
        result.append((i, j))
    for i, j in _batcher_merge(half):
        result.append((i + half, j + half))
    
    # Cross comparisons
    for i in range(half):
        result.append((i, i + half))
    
    # Interleaved sub-merges
    for i in range(1, half, 2):
        for a, b in _batcher_merge(i * 2):
            # This maps to wires: a in [0, 2i-1] and 2i-1-a in [half, half+2i-1]
            # Actually this is getting complicated. Let me use a simpler approach.
            pass
    
    # For simplicity and correctness, let me use a different well-known construction
    return result  # This is incomplete, let me use a different approach

def _generate_bitonic_sort(n):
    """Generate Batcher's bitonic sort network for n wires."""
    if n == 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    half = n // 2
    # Sort first half ascending, second half descending
    # Then merge
    
    # For simplicity, let's use the odd-even transposition as baseline
    # and try to improve with local search
    pass

def _odd_even_transposition(n):
    """Odd-even transposition sorting network."""
    comps = []
    for phase in range(n):
        for i in range(phase % 2, n - 1, 2):
            comps.append((i, i + 1))
    return comps

def _verify(comps):
    """Verify that the comparator network sorts all 2^16 inputs."""
    sorted_state = (1 << 16) - 1  # All 1s
    for state in range(1 << 16):
        s = state
        for (i, j) in comps:
            s = _apply_comp(s, i, j)
        if s != sorted_state:
            return False, state
    return True, None

def _is_sorted_state(state):
    """Check if a 16-bit state is sorted (all 0s then all 1s)."""
    # Sorted means: there exists k such that bits 0..k-1 are 0 and bits k..15 are 1
    # i.e., state = 0, 1, 3, 7, ..., 65535 (2^k - 1 for k=0..16)
    k = state.bit_length()
    return state == (1 << k) - 1

def _apply_comps_to_state(state, comps):
    """Apply all comparators to a state."""
    for (i, j) in comps:
        state = _apply_comp(state, i, j)
    return state

def _find_unsorted_inputs(comps):
    """Find all input states that are not sorted by the network."""
    unsorted = []
    for state in range(1 << 16):
        result = _apply_comps_to_state(state, comps)
        if not _is_sorted_state(result):
            unsorted.append(state)
    return unsorted

def construct():
    """Construct a sorting network for 16 wires."""
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill
    
    # Start with odd-even transposition (120 comparators)
    comps = _odd_even_transposition(N)
    
    # Try to remove redundant comparators using greedy local search
    # For each comparator, try removing it and check if the network still sorts
    # Use a faster verification: only check previously-failing states
    
    # First, get all states that were sorted by the original network
    # Then when we remove a comparator, we only need to re-verify those states
    # that might now fail
    
    # Better approach: track which states are "critical" - states that were
    # only sorted because of specific comparators
    
    # Simple greedy: try removing each comparator, verify full network
    # But 65536 states * 120 comps * 120 comps per check = too slow in pure Python
    
    # Use a smarter approach: for each comparator (i,j), determine if it's necessary
    # A comparator is necessary if there exists an input where it changes the output
    # and without it, the final output is not sorted
    
    # Faster verification using the fact that we only need to check if output is sorted
    # Let's precompute the effect of each comparator on each state
    
    # Alternative: use the zero-one principle more efficiently
    # A network is a sorting network iff it sorts all 2^16 boolean inputs
    # We can represent the set of unsorted outputs as a bitset
    
    # Let's try a different approach: start from a known good network and
    # try to find a smaller one using simulated annealing or greedy removal
    
    # Greedy removal with efficient checking:
    # For each comparator, remove it and check if network still sorts
    # To speed up, we can track the set of "witness" states (states that fail)
    
    # Let's use a more efficient verification:
    # Instead of checking all 65536 states, we can use the fact that
    # if a network sorts all inputs, then for any input, the output must be
    # of the form 0...01...1
    
    # Let's try random restarts with different initial networks
    
    best_comps = list(comps)
    best_count = len(comps)
    
    # Try removing comparators one by one
    # For efficiency, we'll only verify a subset of states first, then full verify
    # at the end
    
    # Key insight: if we remove comparator (i,j), only states where wire i > wire j
    # before that comparator are affected. We can track this.
    
    # For now, let's do a simple greedy removal with full verification
    # but limit the number of attempts based on time
    
    removed = []
    i = 0
    while i < len(comps) and time.time() < deadline - 5.0:
        # Try removing comps[i]
        test_comps = comps[:i] + comps[i+1:]
        
        # Quick check: verify on a sample of states
        # If it passes the sample, do full verification
        sample_states = list(range(0, 1 << 16, 16))  # Every 16th state
        passed_sample = True
        for s in sample_states:
            result = _apply_comps_to_state(s, test_comps)
            if not _is_sorted_state(result):
                passed_sample = False
                break
        
        if passed_sample:
            # Full verification
            passed_full = True
            for s in range(1 << 16):
                result = _apply_comps_to_state(s, test_comps)
                if not _is_sorted_state(result):
                    passed_full = False
                    break
            
            if passed_full:
                # Remove this comparator
                comps = test_comps
                best_comps = list(comps)
                best_count = len(comps)
                # Reset i to 0 since removal changes indices
                i = 0
            else:
                i += 1
        else:
            i += 1
    
    return best_comps
# EVOLVE-BLOCK-END
