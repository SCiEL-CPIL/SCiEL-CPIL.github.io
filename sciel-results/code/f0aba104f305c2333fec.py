"""Size-efficient sorting network on 17 wires using bitmask-based pruning."""
import numpy as np
import time
import random

N = 17
NUM_INPUTS = 1 << N  # 131,072

# Precompute the bitmask for each input wire.
# wire_bitmask[i] is an integer of NUM_INPUTS bits where bit k is 1 iff wire i is 1 in input k.

def _build_wire_bitmasks():
    """Build the initial bitmask for each wire efficiently."""
    masks = [0] * N
    for i in range(N):
        step = 1 << i
        block_len = step * 2
        pattern = (1 << step) - 1
        num_blocks = (1 << N) // block_len
        m = 0
        # Unroll or use a loop, loop is fine for 2^17
        for b in range(num_blocks):
            m |= pattern << (b * block_len + step)
        masks[i] = m
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
    """Validate a network by simulating it."""
    st = list(_WIRE_MASKS)
    for i, j in comps:
        _apply_comparator(st, i, j)
    return _is_sorted(st)

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

def construct():
    start_time = time.time()
    # Start with odd-even transposition
    network = _odd_even_transposition(N)
    
    # Verify seed
    if not _validate_network(network):
        return network

    # Phase 1: Static Redundancy Removal
    # Iteratively remove comparators that are statically redundant
    # A comparator (i, j) is statically redundant if state[i] is a subset of state[j] before it is applied.
    
    def _remove_static_redundants(net):
        changed = True
        while changed:
            changed = False
            st = list(_WIRE_MASKS)
            to_remove = set()
            for idx, (i, j) in enumerate(net):
                # Check if i <= j already
                if (st[i] | st[j]) == st[j]: # st[i] is subset of st[j]
                    to_remove.add(idx)
                _apply_comparator(st, i, j)
            
            if to_remove:
                # Filter out removed indices
                new_net = [comp for idx, comp in enumerate(net) if idx not in to_remove]
                net = new_net
                changed = True
        return net

    network = _remove_static_redundants(network)

    # Phase 2: Local Search (Random Deletions)
    # Try to remove single or pairs of comparators to find jointly redundant sets.
    
    def _try_deletion(net, indices_to_remove):
        if not indices_to_remove:
            return net
        remove_set = set(indices_to_remove)
        new_net = [comp for idx, comp in enumerate(net) if idx not in remove_set]
        if _validate_network(new_net):
            return new_net
        return None

    # Simple Hill Climbing with Random Restarts if stuck?
    # Just iterate random deletions.
    
    # Shuffle network to break symmetry?
    # No, order matters for static redundancy, but for validity, any order that sorts is fine.
    # However, changing order might enable new static redundancies.
    # Let's just stick to deletion.
    
    max_iter = 50000
    for _ in range(max_iter):
        if time.time() - start_time > 55:
            break
        
        # Try single deletion
        if len(network) > 0:
            idx = random.randint(0, len(network) - 1)
            new_net = _try_deletion(network, [idx])
            if new_net is not None:
                network = new_net
                continue
        
        # Try pair deletion
        if len(network) > 1:
            idx1 = random.randint(0, len(network) - 1)
            idx2 = random.randint(0, len(network) - 1)
            if idx1 != idx2:
                new_net = _try_deletion(network, [idx1, idx2])
                if new_net is not None:
                    network = new_net
                    continue

    return network
