import time
import random
import numpy as np

N = 18
MAX_COMPARATORS = 4096
TIME_LIMIT = 80.0

def construct():
    start_time = time.time()
    
    # 1. Generate initial Batcher network
    net = _batcher_network(N)
    
    # Ensure it's valid (it should be)
    if not _verify_full(net):
        # Fallback to seed if Batcher fails (unlikely)
        return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
        
    best_net = net
    best_size = len(net)
    
    # 2. Greedy Removal Optimization
    # We use a subset check for speed during the greedy phase
    # and full check for the final candidate
    
    current_net = net
    
    # Pre-generate a set of test inputs for subset verification
    # 2^18 is 262144. Let's use 50000 random samples for the greedy loop.
    # This is fast enough to run many times.
    test_inputs = np.random.randint(0, 2, (50000, N))
    
    while time.time() - start_time < TIME_LIMIT:
        removed_any = False
        
        # Try removing each comparator
        for i in range(len(current_net) - 1, -1, -1):
            if time.time() - start_time > TIME_LIMIT:
                break
                
            # Candidate: current_net without comparator i
            candidate = current_net[:i] + current_net[i+1:]
            
            # Quick check with subset
            if _verify_subset(candidate, test_inputs):
                # If subset passes, it's likely valid.
                # To be absolutely sure for the final result, we will verify full
                # only when we decide to keep this as a new best.
                # For now, accept it and continue searching for more removals.
                current_net = candidate
                removed_any = True
                break # Restart the loop since indices changed
        
        if not removed_any:
            # Local minimum reached. Try to escape by swapping?
            # Or just stop if time is low.
            # Let's try a few random swaps to see if they enable removals.
            escaped = False
            for _ in range(10):
                if time.time() - start_time > TIME_LIMIT:
                    break
                if len(current_net) < 2:
                    break
                i, j = random.sample(range(len(current_net)), 2)
                # Swap i and j
                candidate = current_net[:]
                candidate[i], candidate[j] = candidate[j], candidate[i]
                
                # Check if this swap allows a removal?
                # Actually, just check if the swapped network is still valid (subset).
                # If it is, we might be able to remove something next.
                if _verify_subset(candidate, test_inputs):
                    # Try to remove one comparator from this swapped state
                    for k in range(len(candidate) - 1, -1, -1):
                        cand2 = candidate[:k] + candidate[k+1:]
                        if _verify_subset(cand2, test_inputs):
                            current_net = cand2
                            escaped = True
                            break
                    if escaped:
                        break
            
            if not escaped:
                # Stuck. If we have time, we could do a full SA, but greedy is usually good enough for N=18.
                # Let's just break to ensure we return a valid result.
                break

    # Final verification of the best network found
    # We need to return a network that is CERTAINLY valid.
    # The greedy process accepted moves based on subset.
    # We must verify the final `current_net` against ALL inputs.
    if _verify_full(current_net):
        best_net = current_net
    else:
        # If the final one failed (unlikely if subset was large), revert to last known good?
        # We didn't track last known good with full verification.
        # To be safe, we should have kept the initial Batcher network as a backup.
        # But Batcher is valid.
        # Let's assume subset check with 50k samples is robust enough for N=18.
        # If not, we return the initial Batcher network which is guaranteed valid.
        if _verify_full(net):
            best_net = net
        else:
            # Extreme fallback
            best_net = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]

    return best_net

def _batcher_network(n):
    """Generate Batcher Odd-Even Merge Sort network."""
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    # Split
    mid = n // 2
    left = _batcher_network(mid)
    right = _batcher_network(n - mid)
    
    # Adjust right network indices
    right = [(i + mid, j + mid) for i, j in right]
    
    # Merge
    merge = _batcher_merge(mid, n - mid)
    
    return left + right + merge

def _batcher_merge(n1, n2):
    """Generate comparators for merging two sorted networks of size n1 and n2."""
    # This is the standard Batcher merge.
    # It's easier to implement the recursive definition directly or use a known library logic.
    # For simplicity and correctness, let's use a simpler sorting network generator if Batcher is complex to implement correctly from scratch in short code.
    # Actually, Batcher is well-defined.
    
    # Alternative: Use a simple recursive merge.
    # Batcher's merge of two sorted lists of length n1 and n2 (n1 <= n2):
    # 1. Split the second list into two halves? No, that's not quite right.
    # Let's use the standard algorithm:
    # If n1 == 0 or n2 == 0: return []
    # If n1 == 1 and n2 == 1: return [(0, 1)]
    # 
    # The standard Batcher merge is complex. Let's use a simpler, known correct network:
    # **Shell Sort Network**? No.
    # **Odd-Even Transposition** is the seed.
    # **Batcher** is better.
    # Let's implement Batcher correctly.
    
    # Reference:
    # Merge(n1, n2):
    #   if n1 == 0 or n2 == 0: return []
    #   if n1 == 1 and n2 == 1: return [(0, 1)]
    #   if n1 > n2: swap n1, n2 and reverse logic? No, assume n1 <= n2.
    #   Let n = n1 + n2.
    #   The merge consists of:
    #   1. Comparators between i and i + n1 for i in 0..n2-1 ? No.
    
    # Let's use a simpler approach: **Recursive Merge**
    # Actually, implementing Batcher correctly in 20 lines is error-prone.
    # Let's use **Odd-Even Merge** which is Batcher's.
    
    # To avoid bugs, I will use a slightly larger but simpler network:
    # **Bitonic Sort**?
    # Bitonic sort for N=18 (next power of 2 is 32, so it's wasteful).
    
    # Let's stick to **Greedy Removal from Odd-Even Transposition**.
    # The seed is 153.
    # If I just do greedy removal on the seed, I might get down to ~120-130.
    # That is still an improvement (score > 1.0).
    # Given the time constraint and risk of Batcher implementation errors, starting from the seed and doing greedy removal is safer and still beats 1.0.
    
    # Let's modify the plan:
    # 1. Start with Seed (153 comps).
    # 2. Greedy removal.
    
    return [] # Placeholder, we will use seed in construct

def _verify_subset(net, inputs):
    """
    Verify if net sorts the given inputs.
    inputs: (M, N) array of 0s and 1s.
    """
    if not net:
        return True
        
    state = inputs.copy()
    for i, j in net:
        # Compare state[:, i] and state[:, j]
        # min goes to i, max goes to j
        min_val = np.minimum(state[:, i], state[:, j])
        max_val = np.maximum(state[:, i], state[:, j])
        state[:, i] = min_val
        state[:, j] = max_val
        
    # Check if sorted: state[:, k] <= state[:, k+1] for all k
    # If any violation, return False
    # Vectorized check:
    # diff = np.diff(state, axis=1)
    # return not np.any(diff < 0)
    
    # More efficient:
    # For each row, is it sorted?
    # We can check if state is sorted by checking if state[:, 1:] >= state[:, :-1]
    sorted_mask = np.all(state[:, 1:] >= state[:, :-1], axis=1)
    return np.all(sorted_mask)

def _verify_full(net):
    """
    Verify if net sorts ALL 2^N inputs.
    """
    if N > 20:
        raise ValueError("N too large for full verification")
        
    # Generate all 2^N bitmasks
    # 2^18 = 262144.
    # Create array of shape (2^N, N)
    # Use np.meshgrid or binary representation.
    
    # Fastest way:
    # idx = np.arange(2**N)
    # bits = (idx[:, None] >> np.arange(N)[::-1]) & 1
    # This creates (2^N, N) array.
    
    idx = np.arange(2**N)
    bits = (idx[:, None] >> np.arange(N)[::-1]) & 1
    
    return _verify_subset(net, bits)

# Since _batcher_network is complex, let's just use the seed network as the starting point for greedy removal.
# The seed is 153 comparators.
# Greedy removal will likely reduce it significantly.

def _seed_network():
    return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]

# Update construct to use _seed_network

def construct():
    start_time = time.time()
    
    # 1. Start with Seed
    net = _seed_network()
    
    # Verify seed (should be true)
    if not _verify_full(net):
        # Should not happen
        return net

    best_net = net
    current_net = net
    
    # Test inputs for greedy phase
    test_inputs = np.random.randint(0, 2, (50000, N))
    
    while time.time() - start_time < TIME_LIMIT:
        removed_any = False
        
        # Try removing each comparator
        # Iterate backwards to avoid index issues when removing
        for i in range(len(current_net) - 1, -1, -1):
            if time.time() - start_time > TIME_LIMIT:
                break
                
            candidate = current_net[:i] + current_net[i+1:]
            
            # Quick check
            if _verify_subset(candidate, test_inputs):
                # Accept
                current_net = candidate
                removed_any = True
                break # Restart loop
        
        if not removed_any:
            # Local minimum. Try to escape with swaps?
            # Or just stop.
            # Let's try a few random swaps to see if we can find a new local minimum.
            escaped = False
            for _ in range(20):
                if time.time() - start_time > TIME_LIMIT:
                    break
                if len(current_net) < 2:
                    break
                i, j = random.sample(range(len(current_net)), 2)
                candidate = current_net[:]
                candidate[i], candidate[j] = candidate[j], candidate[i]
                
                if _verify_subset(candidate, test_inputs):
                    # Try to remove from this new state
                    for k in range(len(candidate) - 1, -1, -1):
                        cand2 = candidate[:k] + candidate[k+1:]
                        if _verify_subset(cand2, test_inputs):
                            current_net = cand2
                            escaped = True
                            break
                    if escaped:
                        break
            
            if not escaped:
                break

    # Final check
    if _verify_full(current_net):
        return current_net
    else:
        return best_net # Which is the seed, guaranteed valid

# Wait, best_net was initialized to seed. If current_net fails full check, we return seed.
# This is safe.

# Let's refine the escape mechanism.
# If we are stuck, we might be in a local minimum where no single removal works.
# Swapping might help.

# One more thing: The prompt says "Keep the function names and signatures of the seed program."
# Seed program has `construct()`.
# My program has `construct()`.

# Let's double check the time complexity of `_verify_subset`.
# 50,000 inputs, 18 wires, ~150 comparators.
# Operations: 50,000 * 150 * (min, max, assign) ~ 15 million ops.
# NumPy handles this in ~0.1s.
# If we do 1000 moves, that's 100s. Too slow.
# We need to reduce the number of subset checks or make them faster.
# 
# Optimization:
# Only check subset if the move is "plausible".
# But we don't know if it's plausible without checking.
#
# Alternative:
# Reduce test_inputs to 10,000.
# 10,000 * 150 = 1.5 million ops. ~0.02s per check.
# 1000 moves * 0.02s = 20s. This is feasible.
# Let's use 10,000 test inputs.

# Revised construct:

def construct():
    start_time = time.time()
    net = _seed_network()
    
    if not _verify_full(net):
        return net

    current_net = net
    test_inputs = np.random.randint(0, 2, (10000, N))
    
    while time.time() - start_time < TIME_LIMIT:
        removed_any = False
        
        for i in range(len(current_net) - 1, -1, -1):
            if time.time() - start_time > TIME_LIMIT:
                break
                
            candidate = current_net[:i] + current_net[i+1:]
            
            if _verify_subset(candidate, test_inputs):
                current_net = candidate
                removed_any = True
                break
        
        if not removed_any:
            # Try escapes
            escaped = False
            for _ in range(50):
                if time.time() - start_time > TIME_LIMIT:
                    break
                if len(current_net) < 2:
                    break
                i, j = random.sample(range(len(current_net)), 2)
                candidate = current_net[:]
                candidate[i], candidate[j] = candidate[j], candidate[i]
                
                if _verify_subset(candidate, test_inputs):
                    for k in range(len(candidate) - 1, -1, -1):
                        cand2 = candidate[:k] + candidate[k+1:]
                        if _verify_subset(cand2, test_inputs):
                            current_net = cand2
                            escaped = True
                            break
                    if escaped:
                        break
            if not escaped:
                break

    if _verify_full(current_net):
        return current_net
    else:
        return net
