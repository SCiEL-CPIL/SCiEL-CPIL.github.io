import time
import random
from math import gcd

def construct():
    """
    Find a large subset of [1, 128] with no three distinct elements whose
    three pairwise gcds are equal.
    """
    n_max = 128
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    # Precompute GCD table
    gcd_table = [[0] * (n_max + 1) for _ in range(n_max + 1)]
    for i in range(1, n_max + 1):
        for j in range(i, n_max + 1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g

    # Precompute bad triples and conflict lists
    # bad_pairs_with_z[z] is a list of tuples (x, y) such that (x, y, z) is a bad triple
    # x < y < z is not required for storage, but we store unique pairs (min, max)
    # Actually, we just need to check if x in S and y in S.
    # We store (x, y) with x < y to avoid duplicates? 
    # The triple (a,b,c) is bad if gcd(a,b)==gcd(a,c)==gcd(b,c).
    # For a fixed z, we look at pairs (x,y) from [1, z-1] or just all other elements.
    
    bad_pairs_with_z = [[] for _ in range(n_max + 1)]
    
    # Iterate through all triples
    for a in range(1, n_max + 1):
        for b in range(a + 1, n_max + 1):
            g_ab = gcd_table[a][b]
            for c in range(b + 1, n_max + 1):
                if gcd_table[a][c] == g_ab and gcd_table[b][c] == g_ab:
                    # This is a bad triple (a, b, c)
                    # For element a: conflict pair is (b, c)
                    # For element b: conflict pair is (a, c)
                    # For element c: conflict pair is (a, b)
                    
                    # Normalize pairs to (min, max) for consistency, though 
                    # checking 'x in S and y in S' is symmetric.
                    p1 = (b, c)
                    p2 = (a, c)
                    p3 = (a, b)
                    
                    bad_pairs_with_z[a].append(p1)
                    bad_pairs_with_z[b].append(p2)
                    bad_pairs_with_z[c].append(p3)
    
    # Convert lists to sets of tuples for faster lookup? 
    # No, we iterate through them. Lists are fine.
    # However, if there are duplicates? 
    # Each triple is generated once. So (b,c) is added to a only once.
    # So no duplicates in the list for a specific z.
    
    all_elements = set(range(1, n_max + 1))
    
    def can_add(z, selected):
        """
        Check if z can be added to selected without creating a bad triple.
        selected is a set of elements.
        """
        # For each pair (x, y) that forms a bad triple with z,
        # check if both x and y are in selected.
        for x, y in bad_pairs_with_z[z]:
            if x in selected and y in selected:
                return False
        return True

    def greedy_build(order):
        """
        Build a valid set by adding elements in the given order.
        """
        selected = set()
        for x in order:
            if can_add(x, selected):
                selected.add(x)
        return selected

    def local_search(selected, timeout=10.0):
        """
        Improve the set using 1-swap moves (remove 1, add 1).
        """
        selected = set(selected)
        start_ls = time.time()
        improved = True
        
        while improved:
            if time.time() > start_ls + timeout:
                break
            improved = False
            
            excluded = [x for x in range(1, n_max + 1) if x not in selected]
            if not excluded:
                break
            
            # Try to add any excluded element (1-add)
            added_any = False
            for x in excluded:
                if can_add(x, selected):
                    selected.add(x)
                    improved = True
                    added_any = True
                    break
            
            if added_any:
                continue
            
            # Try 1-swap: remove u from selected, add v from excluded
            # We iterate through all u in selected and all v in excluded
            # To speed up, we can randomize or just do a full scan.
            # Full scan is O(|S| * |Excluded| * K).
            # With |S|~60, |Excluded|~60, K~50, it's ~180k ops.
            
            best_swap = None
            best_u = None
            best_v = None
            
            # Optimization: If we find a swap that increases size, we take it immediately?
            # A 1-swap keeps size same. 
            # Wait, 1-swap is remove 1, add 1. Size is constant.
            # So how do we improve?
            # The goal is to escape local minima where no single add is possible,
            # but a swap enables a subsequent add.
            # Or, if we are stuck at size K, we swap to a different configuration of size K
            # that allows an add to K+1.
            
            # So the move is:
            # 1. Try all 1-swaps.
            # 2. For each resulting set S', check if any element can be added to S'.
            # 3. If yes, accept the swap and the add.
            
            # This is expensive: O(|S| * |Excluded| * (Cost of checking add))
            # Cost of checking add for S' is O(K).
            # Total: O(|S| * |Excluded| * K * K)? No.
            # can_add(v, S - {u}) is O(K).
            # Then we need to check if we can add *any* w to S - {u} + {v}.
            # This is another O(K) check for each w in Excluded.
            # So O(|S| * |Excluded| * (K + |Excluded| * K)).
            # This might be too slow for a full scan every iteration.
            
            # Alternative: Just do random swaps?
            # Or, do a limited scan.
            
            # Let's try a simpler approach first: 
            # If no add is possible, try a random 1-swap and see if it helps.
            # If it doesn't help immediately, we might get stuck.
            
            # Better: 
            # Iterate u in selected.
            #   S_temp = selected - {u}
            #   Iterate v in excluded.
            #     if can_add(v, S_temp):
            #        S_new = S_temp | {v}
            #        # Check if S_new allows an add
            #        for w in excluded:
            #           if w != v and can_add(w, S_new):
            #               # Accept swap and add
            #               selected = S_new | {w}
            #               improved = True
            #               break
            #        if improved: break
            #   if improved: break
            
            # This is the "Remove and Refill" strategy from Inspiration 1.
            
            found_improvement = False
            for u in list(selected):
                if time.time() > start_ls + timeout:
                    return selected
                S_minus_u = selected - {u}
                
                for v in excluded:
                    if can_add(v, S_minus_u):
                        S_candidate = S_minus_u | {v}
                        
                        # Check if we can add anything to S_candidate
                        # We only need to find ONE add to improve size.
                        for w in excluded:
                            if w == v:
                                continue
                            if can_add(w, S_candidate):
                                # Accept
                                selected = S_candidate | {w}
                                improved = True
                                found_improvement = True
                                break
                        
                        if found_improvement:
                            break
                
                if found_improvement:
                    break
            
            if not found_improvement:
                # No improvement found in this pass
                break
        
        return selected

    # Strategy: Multiple randomized restarts
    random.seed(42)
    best_set = set()
    best_size = 0
    
    # Initial greedy with natural order
    if time.time() < deadline:
        order1 = list(range(1, n_max + 1))
        s1 = greedy_build(order1)
        s1 = local_search(s1, timeout=5.0)
        if len(s1) > best_size:
            best_size = len(s1)
            best_set = s1
    
    # Randomized restarts
    while time.time() < deadline - 10: # Leave 10s for final polish
        order = list(range(1, n_max + 1))
        random.shuffle(order)
        
        s = greedy_build(order)
        
        # Local search with time budget
        remaining_time = deadline - time.time() - 2.0
        if remaining_time > 1.0:
            s = local_search(s, timeout=remaining_time)
        
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
    
    # Final polish on the best set found
    if time.time() < deadline - 1 and best_set:
        remaining = deadline - time.time() - 1
        if remaining > 0:
            best_set = local_search(best_set, timeout=remaining)
    
    if not best_set:
        best_set = {1}
    
    return sorted(best_set)

# EVOLVE-BLOCK-END
