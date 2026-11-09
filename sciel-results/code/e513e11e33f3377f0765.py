"""Size-efficient sorting network on 17 wires using bitmask-based pruning."""
import numpy as np

N = 17
NUM_INPUTS = 1 << N  # 131,072

def _build_wire_bitmasks():
    """Build the initial bitmask for each wire."""
    masks = [0] * N
    # Optimization: Use list comprehension or bit manipulation tricks?
    # The loop is ~2.2M iterations, which is fast enough.
    for k in range(NUM_INPUTS):
        for i in range(N):
            if (k >> i) & 1:
                masks[i] |= (1 << k)
    return masks

_WIRE_MASKS = _build_wire_bitmasks()

def _is_sorted(state):
    """Check if the network state is sorted for all inputs."""
    for i in range(N - 1):
        if (state[i] | state[i+1]) != state[i+1]:
            return False
    return True

def _apply_comparator(state, i, j):
    """Apply compare-exchange on wires i and j (i < j)."""
    new_i = state[i] & state[j]
    new_j = state[i] | state[j]
    state[i] = new_i
    state[j] = new_j
    return state

def _validate_network(comps):
    """Validate if the given list of comparators sorts all inputs."""
    st = list(_WIRE_MASKS)
    for i, j in comps:
        _apply_comparator(st, i, j)
    return _is_sorted(st)

def construct():
    """Construct a sorting network for N=17 wires."""
    
    def _odd_even_transposition(n):
        comps = []
        for phase in range(n):
            if phase % 2 == 0:
                for i in range(0, n - 1, 2):
                    comps.append((i, i + 1))
            else:
                for i in range(1, n - 1, 2):
                    comps.append((i, i + 1))
        return comps
    
    network = _odd_even_transposition(N)
    
    # Verify initial network
    if not _validate_network(network):
        return network

    # Iterative forward pruning
    # We do multiple passes until no more comparators can be removed.
    # To avoid O(L^2) restarts, we scan forward and remove if possible,
    # keeping the index steady if a removal occurs.
    
    max_passes = 50
    for _ in range(max_passes):
        changed = False
        i = 0
        while i < len(network):
            # Try removing comparator at i
            # Create a copy without i
            # To avoid expensive list slicing in the inner loop, 
            # we can simulate with a "skip" flag, but slicing is O(L) anyway.
            # Given L ~ 100, slicing is fine.
            
            reduced = network[:i] + network[i+1:]
            if _validate_network(reduced):
                network = reduced
                changed = True
                # Do not increment i, check the new element at i
            else:
                i += 1
        
        if not changed:
            break
            
    return network
