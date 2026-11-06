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

def _fast_simulate(network, initial_masks):
    """
    Simulates the entire network and returns the final masks.
    Much faster than _build_states if we only need the final result.
    """
    masks = initial_masks[:]
    for i, j in network:
        mi = masks[i]
        mj = masks[j]
        masks[i] = mi & mj
        masks[j] = mi | mj
    return masks

def _is_valid_fast(network, initial_masks):
    """
    Checks if the network is valid using fast simulation.
    """
    final_masks = _fast_simulate(network, initial_masks)
    return _is_sorted(final_masks)

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

def _greedy_delete(network, initial_masks, deadline):
    """
    Performs greedy deletion on the network.
    Returns the minimized network.
    """
    # Rebuild states for the current network
    states = _build_states(network, initial_masks)
    
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for k in range(len(network) - 1, -1, -1):
            if time.time() >= deadline:
                return network
            
            if k == len(network) - 1:
                if _is_sorted(states[k]):
                    network.pop(k)
                    states.pop()
                    changed = True
                    break
            else:
                tail = network[k+1:]
                # Optimize: Only simulate tail if it's likely to be fast?
                # Tail length is len(network) - k - 1
                final_masks = _simulate_tail(states[k], tail)
                if _is_sorted(final_masks):
                    network.pop(k)
                    # Update states incrementally:
                    # States 0..k are valid.
                    # We need to re-simulate from k to end.
                    # Since we removed k, the new network is network[0:k] + network[k:]
                    # State k is now the state before what used to be k+1.
                    # We can just rebuild the tail of the states array.
                    new_states = states[:k+1]
                    current_masks = states[k]
                    for comp in network[k:]:
                        i, j = comp
                        mi = current_masks[i]
                        mj = current_masks[j]
                        new_masks = current_masks[:]
                        new_masks[i] = mi & mj
                        new_masks[j] = mi | mj
                        current_masks = new_masks
                        new_states.append(current_masks)
                    states = new_states
                    changed = True
                    break
    return network

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
    
    # 3. Initial Greedy Deletion
    network = _greedy_delete(network, initial_masks, deadline)
    
    best_network = list(network)
    best_len = len(network)
    
    # 4. Local Search
    # Use fast simulation for validity checks, and only run greedy deletion if valid.
    # This allows for many more iterations.
    
    # Random seed for reproducibility if needed, but random is fine
    random.seed(42)
    
    # We will do a mix of insertions, swaps, and deletions.
    # Insertion: Add a random comparator.
    # Swap: Swap two comparators.
    # Deletion: Remove a random comparator (only if valid? No, try and check).
    
    # To be efficient, we only run _greedy_delete if the candidate network is valid.
    # If it's invalid, we discard it immediately.
    
    steps = 0
    while time.time() < deadline - 5.0:
        steps += 1
        current_net = list(best_network)
        
        move_type = random.randint(0, 2)
        
        if move_type == 0:
            # Insert
            pos = random.randint(0, len(current_net))
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            current_net.insert(pos, (i, j))
        elif move_type == 1:
            # Swap
            if len(current_net) >= 2:
                k1 = random.randint(0, len(current_net) - 1)
                k2 = random.randint(0, len(current_net) - 1)
                if k1 != k2:
                    current_net[k1], current_net[k2] = current_net[k2], current_net[k1]
                else:
                    continue # Skip if no change
        else:
            # Delete
            if len(current_net) > 0:
                k = random.randint(0, len(current_net) - 1)
                current_net.pop(k)
        
        # Fast validity check
        if not _is_valid_fast(current_net, initial_masks):
            continue
            
        # If valid, try to minimize it with greedy deletion
        # Pass a local deadline to ensure we don't overshoot
        local_deadline = min(deadline - 2.0, time.time() + 2.0)
        minimized_net = _greedy_delete(current_net, initial_masks, local_deadline)
        
        if len(minimized_net) < best_len:
            best_network = minimized_net
            best_len = len(best_network)
            
    # Final verification
    if _is_valid_fast(best_network, initial_masks):
        return best_network
    else:
        # Fallback to initial network if something went wrong (should not happen)
        return network
