"""Size-efficient sorting networks on 17 wires using bitmask-based zero-one principle verification.

Strategy:
1. Generate a strong seed network (odd-even transposition, 136 comparators).
2. Aggressive greedy removal with dual-track bitmask pruning:
   - Pre-compute prefix states for O(1) no-op detection
   - Quick pre-filter: if a comparator is a no-op at its position, remove it instantly
   - For non-no-op candidates, simulate only the suffix (from index k+1 to end)
3. Additionally, we try to swap adjacent comparators (if they don't conflict) and
   check if the swapped version allows further removals.
"""
import time

N = 17
NUM_BOOL_INPUTS = 1 << N  # 131072
TIME_LIMIT = 85.0  # seconds, leaving margin for the 90s wall-clock limit


def _make_bitmasks():
    """Create the 17 bitmask integers representing each wire's state across all 131072 Boolean inputs."""
    masks = []
    for i in range(N):
        m = 0
        step = 1 << (i + 1)
        half = 1 << i
        pattern = ((1 << half) - 1) << half
        count = 1 << (N - i - 1)
        m = 0
        for _ in range(count):
            m = (m << step) | pattern
        masks.append(m)
    return masks


_MASKS = _make_bitmasks()


def _is_sorted(masks):
    """Check if the output is sorted: for all i, wire i <= wire i+1."""
    for i in range(N - 1):
        if (masks[i] & ~masks[i + 1]) != 0:
            return False
    return True


def _simulate_suffix(net, start_idx, masks):
    """Simulate comparators from start_idx to end, starting from given masks state."""
    masks = list(masks)
    for idx in range(start_idx, len(net)):
        i, j = net[idx]
        a = masks[i]
        b = masks[j]
        masks[i] = a & b
        masks[j] = a | b
    return masks


def _compute_prefix_states(net):
    """Pre-compute prefix states: prefix_states[k] = masks after applying comparators 0..k-1."""
    prefix_states = []
    masks = list(_MASKS)
    prefix_states.append(masks)
    for idx in range(len(net)):
        i, j = net[idx]
        a = masks[i]
        b = masks[j]
        masks[i] = a & b
        masks[j] = a | b
        prefix_states.append(masks[:])
    return prefix_states


def _verify_network(net):
    """Verify that the network sorts all inputs."""
    masks = list(_MASKS)
    for i, j in net:
        a = masks[i]
        b = masks[j]
        masks[i] = a & b
        masks[j] = a | b
    return _is_sorted(masks)


def _odd_even_transposition_network():
    """Generate the odd-even transposition network."""
    network = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            network.append((i, i + 1))
    return network


def _greedy_remove(net, time_limit):
    """Aggressively remove redundant comparators using dual-track bitmask pruning."""
    start = time.time()
    improved = True
    while improved:
        improved = False
        L = len(net)
        if L == 0:
            break
        
        # Pre-compute prefix states for this network
        prefix_states = _compute_prefix_states(net)
        
        # Try removing from the end first (often redundant comparators are at the end)
        for idx in range(L - 1, -1, -1):
            if time.time() > time_limit:
                return net
            
            i, j = net[idx]
            
            # Quick pre-filter: check if this comparator is a no-op at its position
            # If masks[i] <= masks[j] already, the comparator does nothing
            state_before = prefix_states[idx]
            if (state_before[i] & ~state_before[j]) == 0:
                # Comparator is a no-op, safe to remove without full verification
                net = net[:idx] + net[idx + 1:]
                improved = True
                break  # Restart the scan since indices have shifted
            
            # Not a no-op, need to verify by simulating the suffix without this comparator
            # Start from state_before (which is state after 0..idx-1), skip idx, apply idx+1..L-1
            trial_masks = _simulate_suffix(net, idx + 1, state_before)
            if _is_sorted(trial_masks):
                net = net[:idx] + net[idx + 1:]
                improved = True
                break  # Restart the scan since indices have shifted
    
    return net


def _try_swap(net, time_limit):
    """Try swapping adjacent non-conflicting comparators and see if it helps."""
    start = time.time()
    L = len(net)
    for i in range(L - 1):
        if time.time() > time_limit:
            return net
        
        a_i, a_j = net[i]
        b_i, b_j = net[i + 1]
        
        # Check if they can be swapped (no shared wires)
        if a_j != b_i and a_j != b_j and a_i != b_i and a_i != b_j:
            trial = list(net)
            trial[i], trial[i + 1] = net[i + 1], net[i]
            if _verify_network(trial):
                net = trial
                break
    
    return net


def construct():
    start_time = time.time()
    
    # Generate seed network
    network = _odd_even_transposition_network()
    
    # Verify seed
    if not _verify_network(network):
        raise ValueError("Seed network is not valid")
    
    best_network = list(network)
    
    # Phase 1: Aggressive greedy removal with dual-track pruning
    deadline = start_time + TIME_LIMIT - 5.0
    best_network = _greedy_remove(best_network, deadline)
    
    # Phase 2: Try swaps to enable further removals
    deadline = start_time + TIME_LIMIT - 3.0
    for _ in range(50):  # Limit iterations
        if time.time() > deadline:
            break
        old_len = len(best_network)
        best_network = _try_swap(best_network, deadline)
        if len(best_network) != old_len:
            continue
        # After swap, try more removals
        best_network = _greedy_remove(best_network, deadline)
    
    # Final verification
    if not _verify_network(best_network):
        # Fallback to seed
        best_network = _odd_even_transposition_network()
    
    return best_network
