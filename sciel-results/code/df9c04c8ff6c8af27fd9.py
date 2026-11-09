"""Search-based constructor for size-efficient sorting networks on 18 wires."""
import time
import random
import numpy as np

N = 18
DEADLINE = 58.0  # Safety margin before 60s kill

# Precompute all 2^N inputs as a NumPy array of shape (2^N, N)
# Each row is a bit vector of length N
# We represent the state as an array of uint8, where 1 is True and 0 is False.
# For speed, we can use uint8 for the states.

# Generate all masks
_MASKS = np.arange(1 << N, dtype=np.uint32)
# Expand to bits: shape (2^N, N)
# Bit i corresponds to wire i. 
# We want state[input_idx, wire_idx] = 1 if bit wire_idx is set in input_idx
_BITS = ((_MASKS[:, None] >> np.arange(N, dtype=np.uint32)) & 1).astype(np.uint8)

# Precompute expected sorted states for each mask
# If a mask has k ones, the sorted state has 0s in the first N-k positions and 1s in the last k positions.
_ONES_COUNT = np.sum(_BITS, axis=1)
_EXPECTED = np.zeros((1 << N, N), dtype=np.uint8)
# For each row, if ones_count is k, then indices N-k to N-1 should be 1.
# We can create a mask for this.
# Let's just compute it simply for now, it's a one-time cost.
for k in range(N + 1):
    mask_k = (_ONES_COUNT == k)
    if np.any(mask_k):
        # Set the last k columns to 1 for rows where count is k
        _EXPECTED[mask_k, N - k:] = 1

# EVOLVE-BLOCK-START

def _is_sorted_net_fast(net):
    """
    Check if network sorts all 2^N boolean inputs using NumPy vectorization.
    net: list of (i, j) tuples.
    Returns True if the network is a valid sorting network.
    """
    if not net:
        return N <= 1
    
    # Copy the input states
    states = _BITS.copy()
    
    # Apply each comparator
    for i, j in net:
        # We need to compare states[:, i] and states[:, j]
        # min goes to i, max goes to j
        # min(a, b) = a & b for booleans? No.
        # min(0,0)=0, min(0,1)=0, min(1,0)=0, min(1,1)=1 -> AND
        # max(0,0)=0, max(0,1)=1, max(1,0)=1, max(1,1)=1 -> OR
        
        # However, we must be careful about in-place updates.
        # We need the original values of i and j to compute the new values.
        # Since i and j are distinct wires, we can use temporary arrays or
        # compute both new values from the current state.
        
        col_i = states[:, i]
        col_j = states[:, j]
        
        new_i = np.minimum(col_i, col_j)
        new_j = np.maximum(col_i, col_j)
        
        states[:, i] = new_i
        states[:, j] = new_j
        
    # Check if all states match the expected sorted states
    return np.array_equal(states, _EXPECTED)

def _improve_by_removal(net, deadline):
    """
    Try to improve a network by removing redundant comparators.
    Uses a randomized approach to find removable comparators.
    """
    current = list(net)
    best = list(current)
    best_len = len(current)
    
    # First, do a full validation to make sure we start valid
    if not _is_sorted_net_fast(current):
        return current
    
    iteration = 0
    # We use a temperature-like mechanism or just random restarts
    # Let's try random removals. If stuck, try swaps.
    
    while time.time() < deadline:
        if not current:
            break
        
        # Try removing each comparator in random order
        rng = random.Random(iteration + 42)
        indices = list(range(len(current)))
        rng.shuffle(indices)
        
        improved = False
        for idx in indices:
            if time.time() > deadline:
                break
            
            candidate = current[:idx] + current[idx+1:]
            
            # Full check
            if _is_sorted_net_fast(candidate):
                current = candidate
                best = list(current)
                best_len = len(current)
                improved = True
                break
        
        if not improved:
            # Try some random perturbations to escape local optima
            # 1. Swap two comparators
            # 2. Move a comparator
            # 3. Replace a comparator with a random one (if size allows)
            
            perturbed = False
            for _ in range(20):
                if time.time() > deadline:
                    break
                
                new_net = list(current)
                op = rng.choice(['swap', 'move', 'replace'])
                
                if op == 'swap' and len(new_net) >= 2:
                    i, j = rng.sample(range(len(new_net)), 2)
                    new_net[i], new_net[j] = new_net[j], new_net[i]
                elif op == 'move' and len(new_net) >= 2:
                    i = rng.randint(0, len(new_net) - 1)
                    j = rng.randint(0, len(new_net) - 1)
                    if i != j:
                        item = new_net.pop(i)
                        new_net.insert(j, item)
                elif op == 'replace':
                    # Replace a random comparator with a random one
                    i = rng.randint(0, len(new_net) - 1)
                    w1 = rng.randint(0, N - 2)
                    w2 = rng.randint(w1 + 1, N - 1)
                    new_net[i] = (w1, w2)
                else:
                    continue
                
                if _is_sorted_net_fast(new_net):
                    current = new_net
                    if len(current) < best_len:
                        best = list(current)
                        best_len = len(current)
                    perturbed = True
                    break
            
            if not perturbed and len(current) < best_len:
                # We might have gone up in size or stayed same, but if we can't improve, 
                # and we are at a local minimum, we might want to restart or accept.
                # Since we are minimizing, if we can't remove and can't swap to smaller, 
                # we are stuck. 
                # However, swaps don't change size. 
                # If we can't remove, we are at a local minimum for this specific network structure.
                # We can try a more drastic restart: take a random subset of the current net?
                # Or just break to save time for other strategies.
                pass
        
        iteration += 1
    
    return best

def _odd_even_transposition():
    """Generate the seed odd-even transposition network."""
    net = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            net.append((i, i + 1))
    return net

def _bubble_sort_network():
    """Generate a bubble sort network (adjacent comparators, multiple passes)."""
    net = []
    for _ in range(N):
        for i in range(N - 1):
            net.append((i, i + 1))
    return net

def construct():
    start_time = time.time()
    
    # Start with the seed network
    seed_net = _odd_even_transposition()
    best_net = list(seed_net)
    best_len = len(seed_net)
    
    # Verify seed
    if not _is_sorted_net_fast(seed_net):
        # Shouldn't happen, but fallback
        raise ValueError("Seed network is invalid")
    
    # Strategy 1: Improve the seed network by removal
    # Allocate most of the time here as it's the most promising
    time_limit_1 = time.time() + 40.0
    improved = _improve_by_removal(seed_net, deadline=time_limit_1)
    if len(improved) < best_len:
        best_net = improved
        best_len = len(improved)
    
    # Strategy 2: Try bubble sort network and improve
    if time.time() < DEADLINE - 10:
        bubble_net = _bubble_sort_network()
        if _is_sorted_net_fast(bubble_net):
            if len(bubble_net) < best_len:
                best_net = list(bubble_net)
                best_len = len(bubble_net)
            
            if time.time() < DEADLINE - 5:
                improved_b = _improve_by_removal(bubble_net, deadline=time.time() + min(5, DEADLINE - time.time() - 2))
                if len(improved_b) < best_len:
                    best_net = improved_b
                    best_len = len(improved_b)
    
    # Strategy 3: Random search with guided improvement
    # Try to find smaller networks from scratch? 
    # Given the speed of verification, we can try random networks of size ~best_len - 5
    rng = random.Random(12345)
    
    if time.time() < DEADLINE - 5:
        for target_size in range(max(1, best_len - 10), best_len + 1):
            if time.time() > DEADLINE - 2:
                break
            
            for attempt in range(10):
                if time.time() > DEADLINE - 2:
                    break
                
                rand_net = []
                for _ in range(target_size):
                    i = rng.randint(0, N - 2)
                    j = rng.randint(i + 1, N - 1)
                    rand_net.append((i, j))
                
                # Quick check
                if _is_sorted_net_fast(rand_net):
                    if len(rand_net) < best_len:
                        best_net = rand_net
                        best_len = len(rand_net)
                        
                        # Try to improve this random find
                        if time.time() < DEADLINE - 2:
                            improved_r = _improve_by_removal(rand_net, deadline=time.time() + min(2, DEADLINE - time.time() - 1))
                            if len(improved_r) < best_len:
                                best_net = improved_r
                                best_len = len(improved_r)
                    break # Found a valid one of this size, try smaller size next
    
    # Final verification
    if not _is_sorted_net_fast(best_net):
        # Fallback to seed
        best_net = seed_net
    
    if len(best_net) > 4096:
        # Truncate (shouldn't happen)
        best_net = best_net[:4096]
    
    return best_net

# EVOLVE-BLOCK-END
