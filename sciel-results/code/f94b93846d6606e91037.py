import numpy as np
import time
import random

N = 17
DEADLINE = 85.0 # 5s safety margin

# Precompute shifts for bit unpacking
SHIFTS = np.arange(N - 1, -1, -1, dtype=np.uint32)
# Precompute indices
INDICES = np.arange(1 << N, dtype=np.uint32)

def _verify(net):
    """
    Verifies if the given network sorts all 2^N inputs.
    Returns True if valid, False otherwise.
    """
    # Unpack bits into a (2^N, N) array of uint8
    wires = (INDICES[:, None] >> SHIFTS) & 1
    
    # Apply comparators
    for i, j in net:
        # min to i, max to j
        min_v = np.minimum(wires[:, i], wires[:, j])
        max_v = np.maximum(wires[:, i], wires[:, j])
        wires[:, i] = min_v
        wires[:, j] = max_v
        
    # Check if sorted: w[k] <= w[k+1] for all k
    return not np.any(wires[:, :-1] > wires[:, 1:])

def _generate_seed():
    """
    Generates an Odd-Even Transposition network for N wires.
    """
    net = []
    for phase in range(N):
        start = phase % 2
        for i in range(start, N - 1, 2):
            net.append((i, i + 1))
    return net

def _local_search(net, deadline):
    """
    Perform local search by removing redundant comparators.
    Uses randomized restarts if stuck.
    """
    best_net = net[:]
    best_size = len(net)
    
    # Shuffle the network to break symmetry and find different local minima
    # We do this by shuffling the list of comparators.
    # Note: The order of comparators matters for the network's behavior,
    # but for the purpose of finding redundant ones, the set is what matters.
    # However, removing a comparator changes the network.
    
    while time.time() < deadline:
        # Try to remove comparators one by one
        # We iterate through the list. If we remove one, we restart from 0.
        # To add randomness, we can shuffle the list periodically if no progress is made.
        
        improved = False
        # Create a list of indices to try. We can shuffle this to change the order of attempts.
        # But if we remove an element, indices shift.
        # Strategy: Pick a random index to try removing first?
        # Or just iterate 0 to N-1.
        
        # Let's try a random order of indices for the first pass of this restart.
        indices = list(range(len(net)))
        random.shuffle(indices)
        
        for idx in indices:
            if time.time() > deadline:
                break
            
            # Check if idx is still valid (in case we removed something before it in this pass? 
            # No, we break and restart if we remove something).
            # Actually, if we remove net[idx], the list shrinks.
            # So we should just try to remove net[idx] from the current net.
            
            # But if we shuffled indices, and we removed an earlier index in this pass?
            # We restart the pass if we remove anything. So `improved` becomes True and we break.
            
            # Try removing net[idx]
            saved_comp = net[idx]
            net.pop(idx)
            
            if _verify(net):
                best_net = net[:]
                best_size = len(net)
                improved = True
                break # Restart the scan
            else:
                net.insert(idx, saved_comp)
        
        if not improved:
            # No more removals possible in this pass with this ordering.
            # If we haven't hit the deadline, we can try another random ordering.
            # This allows us to escape local minima where a specific comparator 
            # appears redundant only if another specific one is removed first.
            # But wait, if removing ANY single comparator fails, is it possible 
            # that removing TWO allows a better state? 
            # The search space is "remove one". If no single removal works, 
            # we are at a local minimum for the "remove one" move.
            # Randomizing the order doesn't help if NO single removal works.
            # Because if removing A fails and removing B fails, it doesn't matter 
            # if we try A then B or B then A.
            # So randomizing the order is useless if the move is "remove one".
            
            # What if we allow "remove two"?
            # That doubles the verification cost per step.
            # Maybe we should just accept the local minimum.
            
            # However, there is a subtlety: 
            # The set of redundant comparators depends on the network.
            # If we are at a local minimum, no single comparator is redundant.
            # We are stuck.
            
            # To improve, we might need to "swap" or "move" a comparator.
            # Or start from a different seed.
            
            # Given the time constraint, let's just break and return best.
            # Or, we can try to swap adjacent comparators if it leads to a removal?
            # That's too complex.
            
            break
            
    return best_net

def construct():
    """
    Constructs a size-efficient sorting network for N=17 wires.
    """
    start_time = time.time()
    
    # 1. Generate Seed
    net = _generate_seed()
    
    # 2. Verify Seed (should be valid)
    if not _verify(net):
        return net

    # 3. Local Search
    best_net = _local_search(net, DEADLINE)
    
    return best_net
