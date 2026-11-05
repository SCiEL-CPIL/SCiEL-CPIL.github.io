import time
import random
import numpy as np

def construct():
    """
    Finds a subset A of [0, 128] such that every integer in [128, 256] 
    is a sum of two elements of A (unordered, allowing equal summands).
    Minimizes the maximum number of representations for any sum in [128, 256].
    """
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin for 90s limit
    
    n_max = 128
    min_sum = 128
    max_sum = 256
    target_sums = np.arange(min_sum, max_sum + 1)
    
    # Helper to compute the cost (peak multiplicity) of a set A
    def get_cost(A):
        # A is a numpy array of unique integers
        # We need to count pairs (a, b) with a <= b such that a+b is in [128, 256]
        # Efficiently: for each a in A, count how many b in A satisfy:
        # 1. b >= a (to handle unordered with replacement correctly? 
        #    Actually, "unordered pairs permit equal summands" usually means 
        #    the pair {a,b} is the same as {b,a}. 
        #    If a < b, it's one pair. If a == b, it's one pair.
        #    So we count pairs {a,b} where a,b in A.
        #    Let's just compute all sums and count frequencies in the target range.
        
        # Vectorized approach for speed
        # Create all pair sums
        # A has size k. k^2 sums. 
        # If k is around 64, k^2 is 4096. Very fast.
        # If k is larger, say 100, k^2 is 10000. Still fast.
        
        sums = A[:, None] + A[None, :]
        # Filter sums in target range
        mask = (sums >= min_sum) & (sums <= max_sum)
        valid_sums = sums[mask]
        
        if len(valid_sums) == 0:
            return float('inf')
            
        # Count occurrences of each sum in target range
        # np.unique with return_counts
        unique_sums, counts = np.unique(valid_sums, return_counts=True)
        
        # Check coverage: all sums in [128, 256] must be present
        # Create a boolean array for presence
        presence = np.zeros(max_sum + 1, dtype=bool)
        presence[unique_sums] = True
        if not np.all(presence[target_sums]):
            return float('inf')
            
        return np.max(counts)

    # Helper to check validity (coverage)
    def is_valid(A):
        sums = A[:, None] + A[None, :]
        mask = (sums >= min_sum) & (sums <= max_sum)
        valid_sums = sums[mask]
        if len(valid_sums) == 0:
            return False
        unique_sums, _ = np.unique(valid_sums)
        presence = np.zeros(max_sum + 1, dtype=bool)
        presence[unique_sums] = True
        return np.all(presence[target_sums])

    # Initialize with a known valid set (the seed program's solution)
    # A = [64, 65, ..., 128]
    # This covers [128, 256].
    # 64+64=128, 128+128=256.
    # Let's see its cost.
    initial_A = np.array(list(range(64, 129)), dtype=int)
    current_cost = get_cost(initial_A)
    
    # Try a few random initializations to see if we can start better
    best_A = initial_A
    best_cost = current_cost
    
    # Random initialization strategy:
    # We need enough elements to cover the range.
    # A set of size k can cover at most k(k+1)/2 distinct sums (if all unique).
    # We need to cover 129 sums. k(k+1)/2 >= 129 => k approx 16.
    # However, random sets are unlikely to be valid.
    # Let's stick to the valid seed and improve it.
    
    # Simulated Annealing
    current_A = initial_A.copy()
    current_cost = best_cost
    
    T = 10.0
    cooling_rate = 0.999
    
    # Precompute some things for speed?
    # The cost function is the bottleneck.
    # We can optimize cost calculation.
    
    def fast_cost(A):
        # A is sorted np array
        # Use histogram for counting?
        # Sums range 0..256.
        # We only care about 128..256.
        # We can do:
        # For each a in A:
        #   We need to count b in A such that 128 <= a+b <= 256 and b >= a?
        #   No, the problem says "Pair representations are unordered and permit equal summands".
        #   This implies the set of pairs is { {a,b} | a,b in A }.
        #   If a != b, {a,b} is one pair.
        #   If a == b, {a,a} is one pair.
        #   So the number of representations for sum S is:
        #   count of pairs {a,b} with a+b=S.
        #   If we iterate a from A, and b from A, we count ordered pairs (a,b).
        #   For a != b, (a,b) and (b,a) are the same unordered pair.
        #   For a == b, (a,a) is one pair.
        #   So, if we compute all ordered sums, the count for S is:
        #   Let N_ordered(S) be the number of ordered pairs (a,b) in AxA such that a+b=S.
        #   Let N_diag(S) be the number of pairs (a,a) in AxA such that 2a=S. (0 or 1)
        #   Then N_unordered(S) = (N_ordered(S) - N_diag(S)) / 2 + N_diag(S)
        #   Wait, N_ordered(S) counts (a,b) and (b,a) separately for a!=b.
        #   So (N_ordered(S) - N_diag(S)) is the number of ordered pairs with a!=b.
        #   Dividing by 2 gives the number of unordered pairs with a!=b.
        #   Adding N_diag(S) gives the total unordered pairs.
        
        # This can be computed efficiently using convolution or just loops if A is small.
        # A size ~64. 64*64 = 4096 ops. Very fast in Python loop? Maybe.
        # Numpy is faster.
        
        # Let's use the unique method but optimized.
        sums = A[:, None] + A[None, :]
        # We only need counts for sums in [128, 256]
        # Create a mask for target range
        mask = (sums >= min_sum) & (sums <= max_sum)
        valid_sums = sums[mask]
        
        # Count occurrences
        # Since range is small (129 values), we can use bincount
        # Shift sums by -128 to fit in 0..128
        shifted = valid_sums - min_sum
        counts = np.bincount(shifted, minlength=max_sum - min_sum + 1)
        
        # Check coverage
        if np.any(counts == 0):
            return float('inf')
            
        return np.max(counts)

    # Update best with fast_cost check
    best_cost = fast_cost(best_A)
    
    # SA Loop
    while time.time() < deadline:
        # Generate neighbor
        # Randomly swap an element in A with a random element in [0, 128] not in A
        # Or remove one and add one.
        # Let's try swap: pick i, pick j. A[i] = A[j]. A[j] = old_A[i].
        # This keeps size constant.
        
        if len(current_A) < 2:
            break
            
        i = random.randint(0, len(current_A) - 1)
        j = random.randint(0, len(current_A) - 1)
        
        if i == j:
            continue
            
        old_i = current_A[i]
        old_j = current_A[j]
        
        # Swap
        current_A[i] = old_j
        current_A[j] = old_i
        
        # Sort to keep consistent? Not strictly necessary for cost, but good for uniqueness check if we did add/remove.
        # For swap, uniqueness is preserved if old_i != old_j.
        if old_i == old_j:
            current_A[i] = old_i
            current_A[j] = old_j
            continue
            
        new_cost = fast_cost(current_A)
        
        # Accept or reject
        if new_cost < current_cost:
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_A = current_A.copy()
        else:
            # Accept with probability exp(-(new_cost - current_cost) / T)
            delta = new_cost - current_cost
            if delta == 0:
                current_cost = new_cost
            else:
                prob = np.exp(-delta / T)
                if random.random() < prob:
                    current_cost = new_cost
                else:
                    # Revert
                    current_A[i] = old_i
                    current_A[j] = old_j
        
        # Cool down
        T *= cooling_rate
        if T < 0.01:
            T = 0.01

    # Final check
    if best_cost == float('inf'):
        # Fallback to seed
        return list(range(64, 129))
        
    return list(best_A)

# EVOLVE-BLOCK-END
