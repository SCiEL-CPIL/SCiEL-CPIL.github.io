import time
import random
import numpy as np

def construct():
    start_time = time.time()
    timeout = 85.0  # Safety margin before 90s kill

    n = 512
    # We work with 0-indexed for numpy, but the problem is 1-indexed [1, n]
    # Let's stick to 0-indexed internally: 0..n-1 corresponding to 1..n
    
    # Precompute all 4-APs: (a, a+d, a+2d, a+3d)
    # Store as list of tuples of 4 indices (0-indexed)
    all_aps = []
    # d max is (n-1)/3
    for d in range(1, (n - 1) // 3 + 1):
        for a in range(0, n - 3 * d):
            all_aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    num_aps = len(all_aps)
    
    # Conflict graph: node i conflicts with node j if they are in the same AP
    # For MIS, we want max subset with no edge.
    # But here, the constraint is "no AP fully in S". This is NOT a standard graph MIS.
    # It's a hypergraph independent set problem (4-uniform hypergraph).
    # However, we can model it as: if we include 3 elements of an AP, we cannot include the 4th.
    # Or more simply, we can use a different search strategy.
    
    # Since the "MIS" directive is for 4-AP conflict graph, let's interpret it carefully.
    # A standard MIS on the graph where edges connect pairs in an AP doesn't work because
    # a 4-AP has 6 edges, and removing all edges in a 4-AP is not required, only the whole AP.
    # So MIS on the pair-conflict graph is too restrictive (it would forbid any 3-AP too).
    
    # The directive says: "Formulate the problem as a Maximum Independent Set (MIS) on the 4-AP conflict graph".
    # This is slightly ambiguous. If "4-AP conflict graph" means the hypergraph, MIS is not standard.
    # If it means the graph where vertices are APs and edges... no.
    # Let's stick to the structural change: Branch and Bound on the hypergraph.
    # Or, we can use the fact that the problem is equivalent to finding a max set S such that
    # for every AP (a,b,c,d), |S ∩ {a,b,c,d}| <= 3.
    
    # A powerful approach for small n=512 is to use a SAT-like or CP-SAT approach,
    # but we only have standard lib + numpy/scipy.
    # Let's try a sophisticated local search with "repair" and "kicks",
    # but improved by using bitsets for fast AP checking.
    
    # Representation:
    # S as a bitset (int or numpy array of uint64). 512 bits = 8 uint64s.
    # For each number x, we precompute the list of APs containing x.
    # For each AP, we store the 4 indices.
    
    # To check if adding x is valid:
    # For each AP containing x, check if the other 3 are in S.
    
    # To check if S is valid (for repair or verification):
    # For each AP, check if all 4 are in S.
    
    # Let's use Python integers for bitsets for speed.
    # 512 bits fit in a Python int easily.
    
    # Precompute APs as bitmasks?
    # Each AP is a 512-bit mask.
    # If (mask & S) == mask, then AP is in S.
    
    ap_masks = []
    for ap in all_aps:
        m = 0
        for idx in ap:
            m |= (1 << idx)
        ap_masks.append(m)
    
    # Also, for each x, list of AP indices containing x.
    aps_for_num = [[] for _ in range(n)]
    for i, ap in enumerate(all_aps):
        for idx in ap:
            aps_for_num[idx].append(i)

    def is_valid_addition(x, S_mask):
        # Check if adding x creates a full AP
        for ap_idx in aps_for_num[x]:
            # The AP contains x. Check if the other 3 are in S.
            # We can precompute for each x and each AP containing x, the mask of the OTHER 3.
            pass # We'll optimize this below

    # Optimization: Precompute for each x, for each AP containing x, the mask of the other 3 elements.
    # other_masks[x][ap_idx_in_list] = mask of other 3
    other_masks = [None] * n
    for x in range(n):
        other_masks[x] = []
        for ap_idx in aps_for_num[x]:
            ap = all_aps[ap_idx]
            m = 0
            for idx in ap:
                if idx != x:
                    m |= (1 << idx)
            other_masks[x].append(m)

    def can_add(x, S_mask):
        # Returns True if adding x does not create a 4-AP
        for m in other_masks[x]:
            if (S_mask & m) == m:
                return False
        return True

    def count_bits(mask):
        return bin(mask).count('1')

    # Initial construction: Base-4 digits {0,1}
    # This is a known 4-AP-free set.
    # Size is 2^5 = 32 for n=512?
    # Wait, 4^5 = 1024. So for n=512, we can take the first 512 numbers in base 4 with digits 0,1.
    # Actually, the set of numbers with base-4 digits only 0 or 1 is 4-AP free.
    # How many such numbers are <= 511?
    # 0..511 in base 4 is 00000 to 33333.
    # Numbers with digits 0,1: 2^5 = 32 numbers?
    # Let's check: 0, 1, 4, 5, 16, 17, ...
    # Max value with 5 digits (0,1) is 1+4+16+64+256 = 341.
    # So there are 32 such numbers in [0, 511].
    # This is a very small set. We can do much better.
    
    # Better initial set: Random or Greedy.
    # Let's use a greedy construction with randomization.
    
    def greedy_build(random_order=None):
        S_mask = 0
        if random_order is None:
            random_order = list(range(n))
            random.shuffle(random_order)
        for x in random_order:
            if can_add(x, S_mask):
                S_mask |= (1 << x)
        return S_mask

    # Local Search:
    # 1. Try to add any element that can be added (greedy fill).
    # 2. Try to swap: remove y, add x.
    # 3. Try to remove y, add x1, x2 (2-for-1).
    
    def greedy_fill(S_mask):
        # Try to add any valid element
        changed = True
        while changed:
            changed = False
            # Iterate through all x not in S
            # To be fast, we can just check all x
            for x in range(n):
                if not (S_mask & (1 << x)):
                    if can_add(x, S_mask):
                        S_mask |= (1 << x)
                        changed = True
        return S_mask

    def local_search(S_mask):
        S_mask = greedy_fill(S_mask)
        improved = True
        while improved:
            improved = False
            # Try 1-for-1 swaps
            # For each y in S, remove it, then try to add any x not in S
            # To speed up, we can precompute which x become valid when y is removed?
            # That's complex. Let's just do it naively but with a limit.
            
            # Optimization: Only try swaps if we have time.
            if time.time() > timeout - 5:
                break
                
            for y in range(n):
                if S_mask & (1 << y):
                    S_new = S_mask & ~(1 << y)
                    # Try to add any x
                    # We can try to add x in random order or sorted order
                    # Let's try to add x that were previously blocked by y
                    # How to find them?
                    # x is blocked by y if there is an AP containing x and y, and the other 2 are in S_new.
                    # This is hard to precompute quickly.
                    
                    # Just try greedy fill on S_new?
                    # That might be expensive.
                    # Let's try a limited greedy fill.
                    S_temp = S_new
                    # Try to add elements
                    # We'll just check a few candidates
                    # Actually, let's just do a full greedy fill if the swap looks promising?
                    # Or just check if we can add back y? (Pointless)
                    
                    # Let's try to add any x.
                    # To make this fast, we'll iterate x in random order.
                    # But we need to do this for many y.
                    
                    # Alternative:
                    # Remove y.
                    # Check all x not in S_new. If can_add(x, S_new), add it.
                    # If we added at least 1, and size increased (which it can't since we removed 1), 
                    # wait, if we add 1, size is same. If we add 2, size increases.
                    # So we need to add at least 2 to improve?
                    # Or if we remove y and add x, size is same. We want to break symmetry or escape local min.
                    # But we want to MAXIMIZE size.
                    # So 1-for-1 swaps don't increase size.
                    # We need 1-for-2 or 1-for-3 swaps.
                    
                    # Let's try 1-for-2:
                    # Remove y. Add x1, x2.
                    # This is O(n^2) per y, which is O(n^3) total. 512^3 is ~134M. Too slow in Python.
                    
                    # Let's stick to 1-for-1 for now to escape local minima, 
                    # and rely on random restarts for size increase.
                    # Or, implement a better greedy fill that prioritizes "loose" elements.
                    
                    pass
                    
        return S_mask

    # The above local search is weak. Let's try a different strategy.
    # Strategy:
    # 1. Generate multiple random greedy sets.
    # 2. For each, apply a "repair" or "improve" phase.
    # 3. The improve phase tries to find a 1-for-2 swap.
    
    def find_1_for_2_swap(S_mask):
        # Returns (y, x1, x2) if removing y allows adding x1 and x2, else None
        # y is in S, x1, x2 not in S.
        # This is expensive.
        
        # Heuristic:
        # For each y in S:
        #   S_temp = S_mask without y
        #   Try to add elements to S_temp.
        #   If we can add 2 or more, we have a win.
        
        # To make it faster, we can limit the number of x we try to add.
        
        for y in range(n):
            if not (S_mask & (1 << y)):
                continue
            S_temp = S_mask & ~(1 << y)
            added = []
            # Try to add elements
            # We'll iterate through x not in S_temp
            # To speed up, we can check only x that share an AP with y?
            # Not necessarily.
            
            # Let's just do a limited scan.
            # If this is too slow, we'll skip.
            
            # Actually, let's just use a simple random restart approach with better greedy.
            pass
        return None

    # Let's try a different structural change:
    # Use a SAT solver? No, not available.
    # Use a MILP? No, scipy.optimize.milp is available?
    # scipy.optimize.milp exists in newer scipy.
    # But solving MIS with 512 variables and ~10k constraints might be slow or not converge in 60s.
    
    # Let's try a smarter greedy:
    # Order elements by "degree" (number of APs they are in) and add in that order?
    # Or remove in that order?
    
    # Known result: The size of the largest 4-AP-free subset of [1, n] is roughly n / (log n)^c.
    # For n=512, we expect maybe 100-200 elements.
    # The base-4 set has 32.
    # A random set of size k has probability of containing a 4-AP...
    # There are ~n^2/12 APs.
    # Probability a specific AP is in S is (k/n)^4.
    # Expected number of APs is ~ n^2 * (k/n)^4 = k^4 / n^2.
    # We want this < 1. So k < n^(1/2) = ~22.
    # So random sets are small.
    # We need structured sets.
    
    # Let's try to use the "Moser" construction or similar.
    # Or, use a genetic algorithm.
    
    # Let's try a simple Genetic Algorithm:
    # Population of bitmasks.
    # Fitness: size - penalty for APs.
    # But we need valid sets.
    
    # Let's stick to the most robust method:
    # 1. Start with a good initial set (e.g., greedy with random order).
    # 2. Apply local search:
    #    - Try to add any valid element.
    #    - Try 1-for-1 swaps to escape local minima (even if size doesn't increase, it might allow future adds).
    #    - Try 1-for-2 swaps (expensive, but try a few).
    # 3. Random restarts.
    
    def improve(S_mask):
        S_mask = greedy_fill(S_mask)
        improved = True
        while improved:
            improved = False
            # 1-for-1 swaps
            # Randomize order of y
            ys = [y for y in range(n) if S_mask & (1 << y)]
            random.shuffle(ys)
            for y in ys:
                if time.time() > timeout - 2:
                    break
                S_temp = S_mask & ~(1 << y)
                # Try to add any x
                # We'll try x in random order
                xs = [x for x in range(n) if not (S_mask & (1 << x))]
                random.shuffle(xs)
                for x in xs:
                    if can_add(x, S_temp):
                        S_temp |= (1 << x)
                        # If we added x, we might be able to add more
                        # Let's do a quick greedy fill on S_temp
                        S_temp = greedy_fill(S_temp)
                        if len(bin(S_temp).count('1')) > len(bin(S_mask).count('1')):
                            S_mask = S_temp
                            improved = True
                            break
                        # Even if size didn't increase, we might have changed the set
                        # But we only accept if size increases?
                        # No, we should accept if it leads to a better local min.
                        # But for simplicity, let's only accept size increases.
                        # Or, accept with some probability (simulated annealing)?
                        break
        return S_mask

    best_mask = 0
    best_size = 0
    
    # Multiple random restarts
    num_restarts = 50
    for i in range(num_restarts):
        if time.time() > timeout - 5:
            break
        
        # Generate initial set
        S_mask = greedy_build()
        S_mask = improve(S_mask)
        
        size = bin(S_mask).count('1')
        if size > best_size:
            best_size = size
            best_mask = S_mask
    
    # Final greedy fill
    best_mask = greedy_fill(best_mask)
    
    # Convert back to 1-indexed list
    result = []
    for i in range(n):
        if best_mask & (1 << i):
            result.append(i + 1)
    
    return sorted(result)

# Quick test to ensure it runs
if __name__ == "__main__":
    s = construct()
    print(len(s))
