"""Size-efficient sorting network on 17 wires using bitmask-based pruning."""
import numpy as np

N = 17
NUM_INPUTS = 1 << N  # 131,072

# Precompute the bitmask for each input wire.
# wire_bitmask[i] is an integer of NUM_INPUTS bits where bit k is 1 iff wire i is 1 in input k.
# We represent the state of each wire as a Python int (bitmask over all inputs).

def _build_wire_bitmasks():
    """Build the initial bitmask for each wire efficiently using block patterns."""
    masks = [0] * N
    for i in range(N):
        block_size = 1 << i
        ones_block = (1 << block_size) - 1
        m = 0
        # The wire is 1 for inputs k where (k // block_size) % 2 == 1
        # i.e., k in [block_size, 2*block_size), [3*block_size, 4*block_size), ...
        for start in range(block_size, NUM_INPUTS, 2 * block_size):
            m |= (ones_block << start)
        masks[i] = m
    return masks

# Precompute once at module load to save time in construct()
_WIRE_MASKS = _build_wire_bitmasks()


def _is_sorted(state):
    """Check if the network state is sorted for all inputs.
    state[i] is the bitmask of wire i.
    Sorted means: wire 0 <= wire 1 <= ... <= wire N-1 for every input.
    In bitmask terms: state[i] is a subset of state[i+1] for all i.
    """
    for i in range(N - 1):
        if (state[i] | state[i+1]) != state[i+1]:
            return False
    return True


def _apply_comparator(state, i, j):
    """Apply compare-exchange on wires i and j (i < j).
    min goes to i, max goes to j.
    min(a,b) = a & b, max(a,b) = a | b
    """
    new_i = state[i] & state[j]
    new_j = state[i] | state[j]
    state[i] = new_i
    state[j] = new_j
    return state


def construct():
    """Construct a sorting network for N=17 wires using bitmask-guided pruning.
    
    Strategy:
    1. Start with a reasonable seed network (odd-even transposition).
    2. Use bitmask simulation to verify correctness.
    3. Prune redundant comparators using a two-stage approach:
       a. Remove all no-op comparators in a single pass.
       b. If no no-ops, try greedy removal from the end of the network.
    
    Returns: list of (i, j) pairs.
    """
    # Step 1: Generate a seed network.
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
    
    # Start with odd-even transposition
    network = _odd_even_transposition(N)
    
    # Verify it's correct
    state = list(_WIRE_MASKS)
    for i, j in network:
        _apply_comparator(state, i, j)
    if not _is_sorted(state):
        return network
    
    # Helper to validate a network
    def _validate_network(comps):
        st = list(_WIRE_MASKS)
        for i, j in comps:
            _apply_comparator(st, i, j)
        return _is_sorted(st)
    
    # Helper to find and remove no-ops
    def _remove_no_ops(comps):
        st = list(_WIRE_MASKS)
        no_op_indices = []
        
        for idx, (i, j) in enumerate(comps):
            # Check if this comparator is a no-op given the current state
            if (st[i] | st[j]) == st[j]:
                no_op_indices.append(idx)
            else:
                # Apply comparator
                new_i = st[i] & st[j]
                new_j = st[i] | st[j]
                st[i] = new_i
                st[j] = new_j
        
        if not no_op_indices:
            return comps, False
        
        # Verify the network without no-ops is still valid (it should be, but just in case)
        no_op_set = set(no_op_indices)
        new_comps = [comp for idx, comp in enumerate(comps) if idx not in no_op_set]
        
        # Since no-ops don't change state, the final state is the same.
        # But we must ensure the network is still a sorting network.
        # It is, because the state after the last comparator is the same.
        return new_comps, True

    # Main pruning loop
    changed = True
    while changed:
        changed = False
        
        # Stage 1: Remove no-ops
        network, removed_no_ops = _remove_no_ops(network)
        if removed_no_ops:
            changed = True
            continue
        
        # Stage 2: Greedy removal from the end
        # Try removing comparators from the end backwards
        # This is often effective as end comparators are less critical
        for idx in range(len(network) - 1, -1, -1):
            i, j = network[idx]
            reduced = network[:idx] + network[idx+1:]
            if _validate_network(reduced):
                network = reduced
                changed = True
                break
    
    return network
