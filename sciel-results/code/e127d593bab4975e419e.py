"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random

def construct():
    """
    Construct a set of distinct integers in 1..256 such that:
    - Two subsets with different cardinalities have different sums.
    - Maximize the set size.
    
    Strategy:
    1. Use a bitwise DP to track achievable sums for each cardinality.
    2. Run greedy construction with multiple random orders and deterministic orders.
    3. Return the best (largest) valid set found.
    """
    
    max_val = 256
    max_size = 60
    
    def run_greedy(order):
        sums_by_k = [0] * (max_size + 1)
        sums_by_k[0] = 1  # bit 0 set
        
        result = []
        current_size = 0
        
        for x in order:
            # Check if we can add x
            # Compute new sums
            # Optimization: Only compute up to current_size + 1
            new_sums_by_k = [0] * (max_size + 1)
            new_sums_by_k[0] = 1
            
            # Calculate new bitsets
            # We only care about k up to current_size + 1
            # sums_by_k[k] for k > current_size is 0
            for k in range(1, current_size + 2):
                # new sums at k = old sums at k OR (old sums at k-1 shifted by x)
                # Note: sums_by_k[k] is already the set of sums for cardinality k in the current result
                # So new_sums_by_k[k] = sums_by_k[k] | (sums_by_k[k-1] << x)
                new_sums_by_k[k] = sums_by_k[k] | (sums_by_k[k - 1] << x)
            
            # Check disjointness
            valid = True
            # We only need to check k1, k2 <= current_size + 1
            limit = current_size + 1
            for k1 in range(limit + 1):
                if new_sums_by_k[k1] == 0:
                    continue
                for k2 in range(k1 + 1, limit + 1):
                    if new_sums_by_k[k2] == 0:
                        continue
                    if new_sums_by_k[k1] & new_sums_by_k[k2]:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                result.append(x)
                sums_by_k = new_sums_by_k
                current_size += 1
                if current_size >= max_size:
                    break
        
        return result

    # Try deterministic orders
    orders = [
        list(range(max_val, 0, -1)),
        list(range(1, max_val + 1)),
        list(range(max_val, 0, -1, 2)), # odds desc?
        list(range(max_val - 1, 0, -2)), # evens desc?
    ]
    
    best_result = []
    
    for order in orders:
        res = run_greedy(order)
        if len(res) > len(best_result):
            best_result = res
    
    # Try random orders
    # Number of iterations depends on time. 100-200 is safe.
    num_random_trials = 150
    for _ in range(num_random_trials):
        order = list(range(1, max_val + 1))
        random.shuffle(order)
        res = run_greedy(order)
        if len(res) > len(best_result):
            best_result = res
            
    # Ensure return type is list of ints
    return [int(x) for x in best_result]

# EVOLVE-BLOCK-END
