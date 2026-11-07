"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools

def construct():
    """
    Construct a set of distinct integers in 1..256 such that:
    - Two subsets with different cardinalities have different sums.
    - Maximize the set size.
    
    Strategy:
    1. Use a bitwise DP to track achievable sums for each cardinality.
    2. Greedily add numbers. We try multiple orders (decreasing, increasing) and pick the best result.
    3. The key constraint: for any k1 != k2, the set of sums achievable with cardinality k1
       must be disjoint from the set of sums achievable with cardinality k2.
    
    We use bitsets: sums_by_k[k] is a bitset where bit s is 1 if sum s is achievable with cardinality k.
    When adding a new number x, we check:
      For each existing k, the new sums at cardinality k+1 are (sums_by_k[k] << x).
      We need to ensure that for all k1 != k2, the new bitsets are disjoint.
    """
    
    max_val = 256
    max_size = 60
    
    def run_greedy(order):
        sums_by_k = [0] * (max_size + 1)
        sums_by_k[0] = 1  # bit 0 set
        
        result = []
        
        for x in order:
            # Check if we can add x
            new_sums_by_k = [0] * (max_size + 1)
            new_sums_by_k[0] = 1
            
            valid = True
            # Compute new sums and check validity in one pass or two.
            # First compute all new_sums
            for k in range(1, max_size + 1):
                new_sums_by_k[k] = sums_by_k[k] | (sums_by_k[k - 1] << x)
            
            # Check disjointness
            # Optimization: We only need to check if the *new* bits introduced cause collisions.
            # But full check is O(N^2) with N<=60, so it's fast enough.
            for k1 in range(max_size + 1):
                if new_sums_by_k[k1] == 0:
                    continue
                for k2 in range(k1 + 1, max_size + 1):
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
                if len(result) >= max_size:
                    break
        
        return result

    # Try decreasing order first, as it tends to separate sums better
    best_result = run_greedy(range(max_val, 0, -1))
    
    # Try increasing order
    result_inc = run_greedy(range(1, max_val + 1))
    if len(result_inc) > len(best_result):
        best_result = result_inc
        
    # Try a random-ish order? No, let's stick to deterministic for now.
    # Maybe try "large even" then "large odd"?
    # Let's try one more heuristic: sort by value descending, but maybe skip some?
    # For now, decreasing is the strongest heuristic.
    
    return best_result
# EVOLVE-BLOCK-END
