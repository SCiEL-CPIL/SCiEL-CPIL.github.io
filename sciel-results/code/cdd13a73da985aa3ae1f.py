import time
import random

N = 17
TOTAL_INPUTS = 1 << N  # 131072

def _build_initial_masks():
    """
    Builds bitmask representation for the 0-1 principle.
    mask[k] is an integer where bit m is set iff input m has a 1 on wire k.
    Input m is represented by its binary representation of m (0 to 2^N-1).
    """
    masks = [0] * N
    for m in range(TOTAL_INPUTS):
        for k in range(N):
            if (m >> k) & 1:
                masks[k] |= (1 << m)
    return masks

def _simulate_tail(masks_before, tail_comparators):
    """
    Simulates the effect of 'tail_comparators' on 'masks_before'.
    Returns the final masks.
    """
    masks = masks_before[:]
    for i, j in tail_comparators:
        mi = masks[i]
        mj = masks[j]
        masks[i] = mi & mj
        masks[j] = mi | mj
    return masks

def _is_sorted(masks):
    """
    Checks if the output masks represent a sorted network.
    """
    for k in range(N - 1):
        if (masks[k] & ~masks[k + 1]) != 0:
            return False
    return True

def _build_states(network, initial_masks):
    """
    Pre-computes the state (masks) after each comparator in the network.
    """
    states = [initial_masks]
    current_masks = initial_masks
    for comp in network:
        i, j = comp
        mi = current_masks[i]
        mj = current_masks[j]
        new_masks = current_masks[:]
        new_masks[i] = mi & mj
        new_masks[j] = mi | mj
        current_masks = new_masks
        states.append(current_masks)
    return states

def construct():
    start_time = time.time()
    # Safety margin: 85s deadline to avoid 90s kill.
    deadline = start_time + 85.0
    
    # 1. Generate initial network: Odd-Even Transposition
    network = []
    for phase in range(N):
        start = phase % 2
        for i in range(start, N - 1, 2):
            network.append((i, i + 1))
    
    # 2. Pre-compute initial masks
    initial_masks = _build_initial_masks()
    
    # 3. Pre-compute states
    states = _build_states(network, initial_masks)
    
    if not _is_sorted(states[-1]):
        pass

    # 4. Greedy Deletion
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for k in range(len(network) - 1, -1, -1):
            if time.time() >= deadline:
                break
            
            if k == len(network) - 1:
                if _is_sorted(states[k]):
                    network.pop(k)
                    states.pop()
                    changed = True
                    break
            else:
                tail = network[k+1:]
                final_masks = _simulate_tail(states[k], tail)
                if _is_sorted(final_masks):
                    network.pop(k)
                    states = _build_states(network, initial_masks)
                    changed = True
                    break
        
    best_network = list(network)
    best_len = len(network)
    
    # 5. Local Search
    for _ in range(1000):
        if time.time() >= deadline - 5:
            break
        
        current_net = list(best_network)
        
        move_type = random.randint(0, 2)
        
        if move_type == 0:
            pos = random.randint(0, len(current_net))
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            current_net.insert(pos, (i, j))
        elif move_type == 2:
            if len(current_net) >= 2:
                k1 = random.randint(0, len(current_net) - 1)
                k2 = random.randint(0, len(current_net) - 1)
                if k1 != k2:
                    current_net[k1], current_net[k2] = current_net[k2], current_net[k1]
        
        states = _build_states(current_net, initial_masks)
        
        if not _is_sorted(states[-1]):
            continue
            
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            for k in range(len(current_net) - 1, -1, -1):
                if time.time() >= deadline - 5:
                    break
                
                if k == len(current_net) - 1:
                    if _is_sorted(states[k]):
                        current_net.pop(k)
                        states.pop()
                        changed = True
                        break
                else:
                    tail = current_net[k+1:]
                    final_masks = _simulate_tail(states[k], tail)
                    if _is_sorted(final_masks):
                        current_net.pop(k)
                        states = _build_states(current_net, initial_masks)
                        changed = True
                        break
                        
        if len(current_net) < best_len:
            best_network = current_net
            best_len = len(current_net)
            
    final_masks = initial_masks
    for i, j in best_network:
        mi = final_masks[i]
        mj = final_masks[j]
        final_masks[i] = mi & mj
        final_masks[j] = mi | mj
        
    if _is_sorted(final_masks):
        return best_network
    else:
        return network
