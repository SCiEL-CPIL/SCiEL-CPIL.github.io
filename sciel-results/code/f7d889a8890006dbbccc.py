"""Size-efficient sorting networks on 17 wires."""
import time
import random

N = 17

# EVOLVE-BLOCK-START
import time
import random

N = 17

def _get_initial_masks(n_wires):
    if not hasattr(_get_initial_masks, '_cache'):
        _get_initial_masks._cache = {}
    if n_wires not in _get_initial_masks._cache:
        total_inputs = 1 << n_wires
        masks = []
        for w in range(n_wires):
            m = 0
            for k in range(total_inputs):
                if (k >> w) & 1:
                    m |= (1 << k)
            masks.append(m)
        _get_initial_masks._cache[n_wires] = masks
    return _get_initial_masks._cache[n_wires][:]

def _compute_final_masks(network, n_wires):
    masks = _get_initial_masks(n_wires)
    for i, j in network:
        mi = masks[i]
        mj = masks[j]
        masks[i] = mi & mj
        masks[j] = mi | mj
    return masks

def _is_sorted_final(masks, n_wires):
    for w in range(n_wires - 1):
        if masks[w] & ~masks[w + 1]:
            return False
    return True

def _verify_sorting_network(network, n_wires):
    masks = _get_initial_masks(n_wires)
    for i, j in network:
        if i >= j:
            return False
        mi = masks[i]
        mj = masks[j]
        masks[i] = mi & mj
        masks[j] = mi | mj
    return _is_sorted_final(masks, n_wires)

def _build_seed_network():
    network = []
    for phase in range(N):
        if phase % 2 == 0:
            for i in range(0, N - 1, 2):
                network.append((i, i + 1))
        else:
            for i in range(1, N - 1, 2):
                network.append((i, i + 1))
    return network

def _greedy_remove_fast(network, deadline):
    """Greedy removal with early no-op detection."""
    removed = 0
    changed = True
    while changed and time.time() < deadline:
        changed = False
        # Try removing from the end first (comparators near the output are more likely redundant)
        for i in range(len(network) - 1, -1, -1):
            if time.time() > deadline:
                return network, removed
            new_net = network[:i] + network[i+1:]
            if _verify_sorting_network(new_net, N):
                network = new_net
                removed += 1
                changed = True
                break
    return network, removed

def _perturb_and_clean(network, deadline):
    """Perturb one comparator, verify, then clean up."""
    new_net = network[:]
    if not new_net:
        return new_net
    
    idx = random.randint(0, len(new_net) - 1)
    i, j = new_net[idx]
    
    move_type = random.randint(0, 2)
    if move_type == 0:
        new_i = random.randint(0, j - 1)
        new_net[idx] = (new_i, j)
    elif move_type == 1:
        src = idx
        dst = random.randint(0, len(new_net) - 1)
        comp = new_net[src]
        del new_net[src]
        if dst > src:
            dst -= 1
        new_net.insert(dst, comp)
    else:
        new_j = random.randint(i + 1, N - 1)
        new_net[idx] = (i, new_j)
    
    if not _verify_sorting_network(new_net, N):
        return None
    
    cleaned, _ = _greedy_remove_fast(new_net, deadline)
    return cleaned

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    network = _build_seed_network()
    best_network = list(network)
    best_size = len(network)
    
    # Initial aggressive cleanup
    network, _ = _greedy_remove_fast(network, deadline)
    if len(network) < best_size:
        best_size = len(network)
        best_network = list(network)
    
    # Local search: perturb, verify, clean
    while time.time() < deadline - 2:
        candidate = _perturb_and_clean(best_network, deadline)
        if candidate is not None and len(candidate) < best_size:
            best_size = len(candidate)
            best_network = candidate
    
    return best_network

# EVOLVE-BLOCK-END
