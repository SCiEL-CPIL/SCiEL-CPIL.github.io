import time
import random
import sys

def construct():
    """
    Finds a set of distinct integers in 1..256 maximizing the size,
    such that all non-empty subset sums are distinct and no two distinct
    subset sums divide one another.
    """
    start_time = time.time()
    time_limit = 58  # Keep 2s margin for evaluator overhead (90s kill, but be safe)
    
    best_set = [31, 32, 34]
    best_size = len(best_set)
    
    def compute_sums(S):
        """Compute all subset sums of S (including 0)."""
        sums = {0}
        for x in S:
            new_sums = set()
            for s in sums:
                new_sums.add(s + x)
            sums = sums | new_sums
        return sums
    
    def check_divisibility_antichain(non_empty_sums):
        """
        Check if non_empty_sums forms an antichain under divisibility.
        No two distinct values u, v should satisfy u|v or v|u.
        Optimized: sort and for each u, only check multiples.
        """
        sl = sorted(non_empty_sums)
        n = len(sl)
        # For each element, check if it divides any other
        for i in range(n):
            u = sl[i]
            # Check if u divides any sl[j] for j != i
            # Since sorted, sl[j] >= u for j > i
            # For j < i, sl[j] < u, so u cannot divide sl[j] (unless equal, but distinct)
            # So we only need to check j > i: does u divide sl[j]?
            for j in range(i + 1, n):
                v = sl[j]
                if v % u == 0:
                    return False
                # Early termination: if v > u * max_possible_ratio, break?
                # Actually, we can break if v is too large relative to u? No.
                # But we can break inner loop if v > some bound? Not straightforward.
                # Let's just continue.
        return True
    
    def is_valid_set(S):
        """
        Checks if set S is valid.
        1. All subset sums are distinct.
        2. No two distinct subset sums divide one another.
        """
        if not S:
            return True
        
        sums = compute_sums(S)
        non_empty_sums = {s for s in sums if s != 0}
        
        # Check distinctness: should have exactly 2^k - 1 non-empty sums
        if len(non_empty_sums) != (1 << len(S)) - 1:
            return False
            
        # Check divisibility antichain
        if not check_divisibility_antichain(non_empty_sums):
            return False
        
        return True
    
    def check_validity_incremental(S, x, sums_S):
        """
        Checks if S U {x} is valid, given S is valid.
        S: list of elements
        x: new element
        sums_S: precomputed subset sums of S (including 0)
        Returns (True, new_sums) if valid, (False, None) otherwise.
        """
        if x in S:
            return False, None
            
        # Generate new sums: x + s for s in sums_S
        new_sums = {x + s for s in sums_S}
        
        # Check distinctness: sums_S and new_sums must be disjoint
        # (since 0 is in sums_S but not in new_sums, and x > 0)
        if sums_S & new_sums:
            return False, None
        
        all_sums = sums_S | new_sums
        non_empty_all = {s for s in all_sums if s != 0}
        
        # Check divisibility antichain on non_empty_all
        # We know sums_S \ {0} is an antichain.
        # We need to check:
        # a) new_sums \ {0} is internally an antichain
        # b) cross-check between old and new
        # But simplest: just check the whole set.
        # For k up to ~12, |non_empty_all| = 2^(k+1) - 1 <= 4095
        # check_divisibility_antichain is O(N^2) worst case but with early termination.
        
        if not check_divisibility_antichain(non_empty_all):
            return False, None
        
        return True, all_sums
    
    # Search Strategy
    # Greedy with random restarts, preferring larger numbers
    # Larger numbers are less likely to divide smaller sums
    
    random.seed(42)
    
    # Pre-generate several candidate orderings
    # Strategy 1: descending (largest first)
    # Strategy 2: random
    # Strategy 3: random with bias toward larger
    
    num_restarts = 5000
    for attempt in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
            
        S = []
        sums_S = {0}
        
        # Choose candidate ordering
        candidates = list(range(1, 257))
        
        if attempt % 3 == 0:
            # Descending: prefer larger numbers
            candidates.sort(reverse=True)
        elif attempt % 3 == 1:
            # Random
            random.shuffle(candidates)
        else:
            # Random with bias: shuffle then sort by random key with bias toward larger
            # Actually, just shuffle but weight selection
            random.shuffle(candidates)
            # Reorder: put larger numbers earlier with some probability
            # Simpler: just use a different random seed pattern
            # Or: sort by random() but with bias
            # Let's just do random for now, and add a biased version
            pass
        
        for x in candidates:
            if time.time() - start_time > time_limit:
                break
            
            # Check if we can add x
            valid, new_sums = check_validity_incremental(S, x, sums_S)
            if valid:
                S.append(x)
                sums_S = new_sums
                if len(S) > best_size:
                    best_size = len(S)
                    best_set = S[:]
                
                # If we reach a large size, continue trying to add more
                # But if check becomes too slow, we'll time out naturally
            
            # If S is large, checks become expensive
            # For k >= 14, 2^k = 16384, cross-check is expensive
            # Let's cap the size we try to build greedily in one pass
            if len(S) >= 16:
                # Try a few more but be ready to stop
                if time.time() - start_time > time_limit:
                    break
    
    # Final verification of best_set
    if best_set and len(best_set) <= 14:
        if not is_valid_set(best_set):
            # Fallback to seed
            best_set = [31, 32, 34]
            best_size = 3
    
    return list(best_set)

if __name__ == "__main__":
    res = construct()
    print(res)
