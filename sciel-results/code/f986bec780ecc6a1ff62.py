import time
import random

N = 17
TOTAL_INPUTS = 1 << N  # 131072

# EVOLVE-BLOCK-START
def _build_initial_masks():
    """Build bitmask representation: mask[k] is a big int where bit m is set iff input m has 1 on wire k."""
    masks = [0] * N
    for m in range(TOTAL_INPUTS):
        for k in range(N):
            if (m >> k) & 1:
                masks[k] |= (1 << m)
    return masks

def _apply_comparator(masks, i, j):
    """Apply compare-exchange in-place on masks. Returns (new_i_mask, new_j_mask) for potential rollback."""
    mi = masks[i]
    mj = masks[j]
    new_i = mi & mj
    new_j = mi | mj
    masks[i] = new_i
    masks[j] = new_j
    return mi, mj

def _is_sorted(masks):
    """Check if the network produces sorted output for all inputs."""
    for k in range(N - 1):
        if (masks[k] & ~masks[k + 1]) != 0:
            return False
    return True

def _is_valid(net, initial_masks):
    """Check if network sorts all 2^N boolean inputs."""
    masks = initial_masks[:]
    for i, j in net:
        mi = masks[i]
        mj = masks[j]
        masks[i] = mi & mj
        masks[j] = mi | mj
    return _is_sorted(masks)

def _odd_even_transposition():
    """Generate the seed odd-even transposition network."""
    network = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            network.append((i, i + 1))
    return network

def _greedy_delete(net, initial_masks, deadline):
    """Aggressively remove redundant comparators."""
    net = list(net)
    changed = True
    while changed and time.time() < deadline:
        changed = False
        # Try from both ends
        for start, end, step in [(len(net) - 1, -1, -1), (0, len(net), 1)]:
            for idx in range(start, end, step):
                if time.time() >= deadline:
                    return net
                removed = net.pop(idx)
                if _is_valid(net, initial_masks):
                    changed = True
                    break
                else:
                    net.insert(idx, removed)
            if changed:
                break
    return net

def _greedy_insert(net, initial_masks, deadline, max_iterations=2000):
    """Try to find a smaller network by starting fresh and building up, or by local search."""
    # This is a simplified version; the main optimization is in simulated annealing
    return net

def _simulated_annealing(net, initial_masks, deadline):
    """Simulated annealing to find smaller sorting networks."""
    net = list(net)
    best_net = list(net)
    best_len = len(net)
    
    # Pre-compute initial masks
    current_masks = initial_masks[:]
    
    # Apply current network to get current state
    for i, j in net:
        mi = current_masks[i]
        mj = current_masks[j]
        current_masks[i] = mi & mj
        current_masks[j] = mi | mj
    
    # Temperature schedule
    start_temp = 50.0
    end_temp = 0.1
    total_steps = 50000
    
    for step in range(total_steps):
        if time.time() >= deadline:
            break
        
        # Temperature decay
        progress = step / total_steps
        temp = start_temp * (end_temp / start_temp) ** progress
        
        # Generate a random move
        move_type = random.random()
        
        if move_type < 0.5:
            # Delete a random comparator
            if len(net) <= best_len:
                # Try to delete
                idx = random.randint(0, len(net) - 1)
                removed = net.pop(idx)
                
                # Check validity
                masks = initial_masks[:]
                for i, j in net:
                    mi = masks[i]
                    mj = masks[j]
                    masks[i] = mi & mj
                    masks[j] = mi | mj
                valid = _is_sorted(masks)
                
                if valid:
                    best_net = list(net)
                    best_len = len(net)
                else:
                    net.insert(idx, removed)
            else:
                # Allow adding back if we're worse than best
                idx = random.randint(0, len(net) - 1)
                removed = net.pop(idx)
                masks = initial_masks[:]
                for i, j in net:
                    mi = masks[i]
                    mj = masks[j]
                    masks[i] = mi & mj
                    masks[j] = mi | mj
                valid = _is_sorted(masks)
                
                if valid:
                    best_net = list(net)
                    best_len = len(net)
                else:
                    # Accept with probability exp(-delta/temp) where delta is "cost"
                    # Here cost is just 1 for invalid
                    if random.random() < (1.0 / (1.0 + temp)):
                        net.insert(idx, removed)
                    else:
                        net.insert(idx, removed)
                        
        elif move_type < 0.8:
            # Swap two adjacent comparators
            if len(net) >= 2:
                idx = random.randint(0, len(net) - 2)
                net[idx], net[idx + 1] = net[idx + 1], net[idx]
                
                masks = initial_masks[:]
                for i, j in net:
                    mi = masks[i]
                    mj = masks[j]
                    masks[i] = mi & mj
                    masks[j] = mi | mj
                valid = _is_sorted(masks)
                
                if valid:
                    if len(net) < best_len:
                        best_net = list(net)
                        best_len = len(net)
                else:
                    # Revert
                    net[idx], net[idx + 1] = net[idx + 1], net[idx]
                    
        else:
            # Insert a random comparator
            new_comp = (random.randint(0, N - 2), random.randint(0, N - 2))
            if new_comp[0] > new_comp[1]:
                new_comp = (new_comp[1], new_comp[0])
            idx = random.randint(0, len(net))
            net.insert(idx, new_comp)
            
            masks = initial_masks[:]
            for i, j in net:
                mi = masks[i]
                mj = masks[j]
                masks[i] = mi & mj
                masks[j] = mi | mj
            valid = _is_sorted(masks)
            
            if valid and len(net) < best_len:
                best_net = list(net)
                best_len = len(net)
            else:
                # Revert
                net.pop(idx)
    
    return best_net

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    # 1. Generate initial network
    network = _odd_even_transposition()
    
    # 2. Pre-compute initial bitmasks
    initial_masks = _build_initial_masks()
    
    # 3. Greedy deletion
    network = _greedy_delete(network, initial_masks, deadline)
    
    # 4. Simulated annealing if time remains
    if time.time() < deadline - 5.0:
        network = _simulated_annealing(network, initial_masks, deadline)
    
    # 5. Final greedy pass
    if time.time() < deadline - 2.0:
        network = _greedy_delete(network, initial_masks, deadline)
    
    return network
# EVOLVE-BLOCK-END
