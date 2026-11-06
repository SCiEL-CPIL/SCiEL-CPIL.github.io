import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s limit, 5s buffer
    
    n = 256
    # Initial greedy construction
    selected = []
    selected_set = set()
    
    # Greedy: add x if it doesn't form AP with any pair in selected
    for x in range(1, n + 1):
        # Check if x forms AP with any a, b in selected
        valid = True
        for a in selected:
            # Check if 2*a - x is in selected (and != x)
            # Or 2*x - a is in selected (and != x)
            # Or a + x = 2*b for some b?
            # If we add x, we form AP if:
            # 1. There exists a, b in selected such that a+b=2x (x is midpoint)
            # 2. There exists a, b in selected such that a+x=2b (b is midpoint) -> 2b-a=x
            # 3. There exists a, b in selected such that x+b=2a (a is midpoint) -> 2a-b=x
            
            # Since we check against all a in selected:
            # If 2*a - x is in selected_set and 2*a - x != x:
            #   Then x, a, 2a-x form AP? No, 2a-x, x, a?
            #   Midpoint of (2a-x, a) is (3a-x)/2.
            #   Let's just check all 3 permutations or simply:
            #   Does {x, a, b} form an AP?
            #   Sort x, a, b. Check if diff is constant.
            
            # Efficient check:
            # For a given a, potential partners b are:
            # b = 2*a - x (if b in selected and b != x) -> AP: b, a, x? No.
            # If b, a, x are in AP, then 2a = b+x => b = 2a-x.
            # If a, b, x are in AP, then 2b = a+x => b = (a+x)/2.
            # If a, x, b are in AP, then 2x = a+b => b = 2x-a.
            
            # So for each a in selected:
            # 1. Check if b = 2*a - x is in selected_set and b != x.
            #    If so, {b, a, x} is AP.
            # 2. Check if (a+x) is even. If so, b = (a+x)//2.
            #    Check if b is in selected_set and b != x and b != a.
            #    If so, {a, b, x} is AP.
            # 3. Check if b = 2*x - a is in selected_set and b != x.
            #    If so, {a, x, b} is AP.
            
            b1 = 2 * a - x
            if b1 in selected_set and b1 != x:
                valid = False
                break
            
            if (a + x) % 2 == 0:
                b2 = (a + x) // 2
                if b2 in selected_set and b2 != x and b2 != a:
                    valid = False
                    break
            
            b3 = 2 * x - a
            if b3 in selected_set and b3 != x:
                valid = False
                break
        
        if valid:
            selected.append(x)
            selected_set.add(x)
        
        if time.time() > deadline:
            break

    # Local Search: Swap
    # We try to replace u in selected with v not in selected
    # such that 1/v > 1/u (i.e., v < u) and the new set is valid.
    
    # To speed up, we can iterate v from 1 to n, and for each v, try to find a u to remove.
    # Actually, iterating v in increasing order ensures we find the best v first.
    # For a fixed v, we want to remove the largest u possible?
    # No, we just want ANY u such that removal allows v.
    # And we want the gain 1/v - 1/u to be positive, which is guaranteed if v < u.
    # To maximize improvement, we should probably remove the largest u?
    # No, the constraint is structural.
    
    # Let's do a loop:
    # 1. List of candidates v not in selected_set, sorted by v (ascending).
    # 2. For each v, check if it can be added by removing some u.
    #    We can determine which u's MUST be removed.
    #    If v is blocked by pairs (a,b), we must remove at least one from each pair.
    #    If we restrict to single removal, v must be blocked by pairs that all share a common element u.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Sort candidates by value (smaller is better)
        candidates = [v for v in range(1, n + 1) if v not in selected_set]
        
        for v in candidates:
            if time.time() > deadline:
                break
            
            # Find all u in selected_set such that removing u allows adding v.
            # This means: In selected_set \ {u}, v does not form an AP with any pair.
            # Equivalently: Every AP formed by v and two elements of selected_set must contain u.
            
            # Let's collect the set of "conflict elements" for v.
            # An element a in selected_set is a "conflict" if there exists b in selected_set
            # such that {a, b, v} is an AP.
            # Actually, we need the set of elements involved in ANY AP with v.
            # If we remove u, v is valid if no AP containing v remains.
            # So, the set of elements that form APs with v must be a subset of {u}.
            # This implies that all such elements must be u itself?
            # Wait. If {a, b, v} is an AP, and we remove u, the AP is gone only if u is a, b, or v.
            # v is not in selected_set. So u must be a or b.
            # So, for v to be addable after removing u, ALL APs formed by v and pairs in selected_set
            # must include u.
            
            # Let's find all pairs (a,b) in selected_set that form AP with v.
            # For each such pair, u must be a or b.
            # So u must be in the intersection of {a,b} for all such pairs.
            
            blocking_pairs = []
            for a in selected_set:
                # Check if v forms AP with a and some b in selected_set
                # Case 1: b = 2*a - v. If b in selected_set and b != v:
                #   Pair (b, a) forms AP with v? No, {b, a, v} is AP if 2a = b+v.
                #   Here b = 2a-v => 2a = b+v. Yes.
                #   So (b, a) is a blocking pair.
                b1 = 2 * a - v
                if b1 in selected_set and b1 != v:
                    blocking_pairs.append((b1, a))
                
                # Case 2: b = (a+v)/2. If b in selected_set and b != v and b != a:
                #   {a, b, v} is AP if 2b = a+v.
                #   Here b = (a+v)/2 => 2b = a+v. Yes.
                #   So (a, b) is a blocking pair.
                if (a + v) % 2 == 0:
                    b2 = (a + v) // 2
                    if b2 in selected_set and b2 != v and b2 != a:
                        blocking_pairs.append((a, b2))
                
                # Case 3: b = 2*v - a. If b in selected_set and b != v:
                #   {a, v, b} is AP if 2v = a+b.
                #   Here b = 2v-a => 2v = a+b. Yes.
                #   So (a, b) is a blocking pair.
                b3 = 2 * v - a
                if b3 in selected_set and b3 != v:
                    blocking_pairs.append((a, b3))
            
            if not blocking_pairs:
                # v can be added directly!
                selected.append(v)
                selected_set.add(v)
                improved = True
                break # Restart search with new set
            
            # If there are blocking pairs, find common u
            # u must be in every pair.
            # Take the first pair, u must be one of its elements.
            # Check if that u is in all other pairs.
            
            first_pair = blocking_pairs[0]
            possible_us = set(first_pair)
            
            for u in possible_us:
                # Check if u is in all blocking pairs
                is_common = True
                for pair in blocking_pairs:
                    if u not in pair:
                        is_common = False
                        break
                
                if is_common:
                    # Removing u allows adding v
                    # Check if v < u (gain)
                    if v < u:
                        # Perform swap
                        selected_set.remove(u)
                        selected.remove(u) # O(N) but N is small
                        selected.append(v)
                        selected_set.add(v)
                        improved = True
                        break # Restart search with new set
            
            if improved:
                break

    return selected

# EVOLVE-BLOCK-END
