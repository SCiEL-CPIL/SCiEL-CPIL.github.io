"""Size-efficient sorting networks on 17 wires."""
import time
import random

N = 17

# Global cache for initial masks to avoid repeated attribute lookups
_INITIAL_MASKS_CACHE = {}

def _get_initial_masks(n_wires):
    """
    Get or generate the initial bitmask representations for all 2^n inputs.
    mask[w] is an integer where bit k is 1 if input k has a 1 on wire w.
    """
    if n_wires not in _INITIAL_MASKS_CACHE:
        total_inputs = 1 << n_wires
        masks = []
        # Optimization: Generate masks using bit patterns
        # For wire w, the pattern is 0...0 1...1 repeating with period 2^(w+1)
        # We can construct this efficiently.
        for w in range(n_wires):
            m = 0
            # The bit k is set if (k >> w) & 1 is 1
            # This is equivalent to: for blocks of size 2^(w+1), the second half is 1s.
            # We can build this by iterating, but for n=17, 131072 iterations is fine.
            # To be faster, we can use a recursive doubling method or just the loop.
            # The loop is ~2M ops total for all wires, which is < 0.1s in Python.
            for k in range(total_inputs):
                if (k >> w) & 1:
                    m |= (1 << k)
            masks.append(m)
        _INITIAL_MASKS_CACHE[n_wires] = masks
    return _INITIAL_MASKS_CACHE[n_wires]

def _verify_sorting_network(network, n_wires):
    """
    Verify that the network sorts all 2^n_wires boolean inputs using the zero-one principle.
    """
    if not network:
        return n_wires <= 1
        
    # Get initial masks
    init_masks = _get_initial_masks(n_wires)
    # Copy the masks to simulate
    masks = init_masks[:]
    
    # Simulate network
    m = masks
    for i, j in network:
        # Basic sanity check
        if i >= j:
            return False
        mi = m[i]
        mj = m[j]
        # Compare-exchange: min to i, max to j
        # min(a, b) = a & b
        # max(a, b) = a | b
        m[i] = mi & mj
        m[j] = mi | mj
    
    # Check sortedness: 0s then 1s.
    # This means for any wire w, if it has a 1, all wires w+1... must have 1s.
    # Equivalently, masks[w] is a subset of masks[w+1].
    # masks[w] & ~masks[w+1] must be 0.
    for w in range(n_wires - 1):
        if m[w] & ~m[w + 1]:
            return False
            
    return True

def _build_seed_network():
    """
    Build a seed sorting network using odd-even transposition.
    For N=17, this gives 136 comparators.
    """
    network = []
    for phase in range(N):
        if phase % 2 == 0:
            for i in range(0, N - 1, 2):
                network.append((i, i + 1))
        else:
            for i in range(1, N - 1, 2):
                network.append((i, i + 1))
    return network

def _greedy_remove(network, deadline):
    """
    Perform greedy removal of comparators.
    Returns the final network.
    """
    changed = True
    # Limit iterations to prevent infinite loops if verification is buggy (it shouldn't be)
    # But mostly limited by deadline.
    while changed:
        changed = False
        # Try removing from the end first, as later comparators are often redundant
        # if the network is already sorted by earlier ones.
        for i in range(len(network) - 1, -1, -1):
            if time.time() > deadline:
                return network
            
            # Create candidate network without comparator i
            # Optimization: Instead of slicing, we can pass the network and index to verifier?
            # Current verifier takes a list. Slicing is O(N). N is small (~100).
            new_net = network[:i] + network[i+1:]
            
            if _verify_sorting_network(new_net, N):
                network = new_net
                changed = True
                # Restart the scan from the end? Or continue?
                # Restarting from the end is safer to catch new redundancies created by removal.
                # However, it's O(N^2).
                # Let's break and restart the while loop to scan from the end again.
                break
                
    return network

def _local_search_step(network, deadline):
    """
    Perform a local search step: Perturb the network and try to clean it up.
    Returns the best network found.
    """
    best_net = list(network)
    best_size = len(best_net)
    
    # Number of attempts
    # We want to make many fast attempts.
    attempts = 0
    max_attempts = 500 # Increased attempts
    
    while attempts < max_attempts:
        if time.time() > deadline:
            break
        attempts += 1
        
        candidate = list(best_net)
        
        # Strategy 1: Swap two adjacent comparators (if they don't conflict)
        if len(candidate) > 1 and random.random() < 0.3:
            i = random.randint(0, len(candidate) - 2)
            # Check if swap is valid (no wire conflict)
            c1, c2 = candidate[i], candidate[i+1]
            if len(set(c1 + c2)) == 4:
                candidate[i], candidate[i+1] = candidate[i+1], candidate[i]
            else:
                continue
        # Strategy 2: Replace a comparator with a random valid one
        elif len(candidate) > 0 and random.random() < 0.7:
            idx = random.randint(0, len(candidate) - 1)
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            candidate[idx] = (i, j)
        # Strategy 3: Move a comparator to a random position (if valid)
        elif len(candidate) > 1 and random.random() < 0.9:
            idx = random.randint(0, len(candidate) - 1)
            new_idx = random.randint(0, len(candidate) - 1)
            if idx != new_idx:
                c = candidate.pop(idx)
                # Check if we can insert at new_idx
                # Simple check: just insert and verify later
                candidate.insert(new_idx, c)
        else:
            continue
        
        # Verify the perturbed network
        if _verify_sorting_network(candidate, N):
            # If valid, try to remove redundant comparators
            # Use a limited greedy removal to save time
            cleaned = _greedy_remove(candidate, deadline)
            
            if len(cleaned) < best_size:
                best_net = cleaned
                best_size = len(best_net)
                
    return best_net

def construct():
    """
    Construct a sorting network for 17 wires.
    Returns an ordered list of integer pairs (i, j).
    """
    start_time = time.time()
    # Set a strict deadline to ensure we return well before the 60s limit.
    # 55s leaves 5s margin.
    deadline = start_time + 55.0
    
    # Start with seed network
    network = _build_seed_network()
    best_network = list(network)
    best_size = len(network)
    
    # Phase 1: Initial Greedy Removal
    network = _greedy_remove(network, deadline)
    if len(network) < best_size:
        best_size = len(network)
        best_network = list(network)
    
    # Phase 2: Local Search Loop
    # Use remaining time for local search
    # We run local search steps until time runs out
    while time.time() < deadline - 2.0:
        # Perform a local search step
        # Pass the current best as the starting point
        net = _local_search_step(best_network, deadline)
        
        if len(net) < best_size:
            best_size = len(net)
            best_network = list(net)
            
            # If we improved, do a greedy cleanup on the new best immediately
            # to ensure we are at a local minimum of the removal graph
            if time.time() < deadline - 2.0:
                cleaned_net = _greedy_remove(best_network, deadline)
                if len(cleaned_net) < best_size:
                    best_size = len(cleaned_net)
                    best_network = list(cleaned_net)
        else:
            # If no improvement, we could try a different random seed or strategy
            # but for now, just continue. The random nature of _local_search_step
            # provides the diversification.
            pass

    return best_network
