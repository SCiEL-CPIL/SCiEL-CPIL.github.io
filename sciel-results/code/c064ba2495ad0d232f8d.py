import time
import random
import math

def construct():
    start_time = time.time()
    DEADLINE = 110  # Safety margin for 150s limit

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def is_sum_free(subset):
        s = set(subset)
        for a in subset:
            for b in subset:
                if a + b in s:
                    return False
        return True

    def solve_instance(values):
        n = len(values)
        if n <= 2:
            return list(values)
        
        v_set = set(values)
        
        # 1. Build Conflict Graph
        # Edge (u, v) exists if u + v is in v_set OR u == v and 2*u is in v_set
        # Note: The problem says "a+b=c with a,b,c in it". 
        # If a=b, then 2a=c. So a and a "conflict" if 2a is in the set?
        # Wait, the definition is "no a+b=c". 
        # If I select {a, b}, and a+b is also selected, it's invalid.
        # The conflict graph for MIS should have an edge between x and y 
        # if they cannot coexist in a sum-free set.
        # When can x and y not coexist?
        # 1. If x + y is in the final set? No, the final set is the subset we are building.
        # The condition is: For any distinct a,b in S, a+b not in S. And 2a not in S.
        # So, if we pick x and y, we must ensure x+y is NOT picked, and 2x is NOT picked, and 2y is NOT picked.
        # This is a 3-uniform hypergraph independent set problem, not a graph.
        # However, we can approximate it or use a different strategy.
        
        # Alternative Strategy: 
        # The "Odds" set is sum-free. 
        # The "High" set (x > 5000) is sum-free.
        # The "Low" set (x < -5000) is sum-free.
        
        # Let's try a smarter Local Search that uses bitsets or faster conflict checking.
        # But the directive asks for a DIFFERENT strategy.
        
        # Strategy: Simulated Annealing / Random Restarts with Fast Greedy
        # The current program does "improve" which is a specific local search.
        # Let's try a "Greedy Set Packing" approach on the conflict hypergraph.
        
        # Actually, a very effective heuristic for sum-free sets is:
        # 1. Start with a large structural subset (e.g., Odds).
        # 2. Use "2-opt" style swaps: Remove a small set of elements to allow adding a larger set.
        
        # Let's implement a "Batch Local Search":
        # Instead of adding one element at a time, we identify "blocked" elements.
        # An element x is blocked if it conflicts with current S.
        # Conflict of x with S: 
        #   - 2x in S
        #   - Exists y in S such that x+y in S
        #   - Exists y in S such that x-y in S (which is same as x = y + (x-y))
        
        # Let's try a different structural initialization:
        # Partition by magnitude?
        
        # Let's try "Maximum Independent Set on Conflict Graph" approximation.
        # Even though it's a hypergraph, we can project it to a graph where edge (u,v) 
        # means "u and v cannot be in S together because their sum is in the *original* set?" 
        # No, the sum doesn't have to be in the original set, just the subset.
        
        # Let's stick to a robust Local Search but optimize the inner loop.
        # The current `improve` function is O(|C| * |S|) per iteration.
        # For n=240, |S|~80, |C|~160. 80*160 = 12800 ops per step. 
        # If we do 1000 steps, 12.8M ops. Python can do ~10M-50M simple ops/sec.
        # So it's borderline.
        
        # Different Strategy: **Greedy by "Freedom Score"**
        # Instead of adding free elements first, we sort all elements by how "constraining" they are.
        # An element x is "bad" if it creates many conflicts with other elements.
        # We pick elements that are "least bad" first.
        
        def greedy_by_degree(values):
            vals = list(values)
            v_set = set(vals)
            
            # Precompute "conflict count" for each element with respect to the whole set
            # This is static, but as we build S, the effective conflict changes.
            
            # Dynamic Greedy:
            S = set()
            remaining = set(vals)
            
            # Precompute sum relationships for speed?
            # sums = {}
            # for a in vals:
            #   for b in vals:
            #     if a+b in v_set:
            #        ...
            
            while remaining and time.time() < DEADLINE:
                # For each candidate in remaining, calculate how many conflicts it has with S
                # AND how many potential future conflicts it creates (static heuristic)
                
                best_candidate = None
                best_score = float('inf')
                
                # Optimization: Only check a subset of remaining if it's too slow?
                # n=240 is small enough to check all.
                
                for x in remaining:
                    # Check conflicts with current S
                    conf_count = 0
                    # 1. 2x in S
                    if 2*x in S:
                        conf_count += 1
                        # Early break if we just want to find 0-conflict?
                        # No, we want min conflict.
                        
                    # 2. x+y in S for y in S
                    # 3. x-y in S for y in S (equivalent to x = y + z where z in S)
                    # Basically, does x "break" any pair in S?
                    # If x + y in S, then x and y cannot coexist.
                    # If y - x in S, then y = x + (y-x), so x and y-x cannot coexist.
                    
                    # We can check:
                    # For each y in S:
                    #   if x+y in S: conflict
                    #   if y-x in S: conflict
                    #   if x-y in S: conflict (same as above with sign)
                    
                    # Actually, just checking `x+y in S` and `2*x in S` is sufficient?
                    # If x+y in S, then {x,y,x+y} is a bad triple.
                    # If we add x, we must ensure no y in S has x+y in S.
                    # And no y in S has y-x in S (which means y = x + (y-x)).
                    
                    # Let's just iterate S.
                    for y in S:
                        if x + y in S:
                            conf_count += 1
                            # We can break early if conf_count > best_score
                            if conf_count > best_score:
                                break
                        if y - x in S:
                            conf_count += 1
                            if conf_count > best_score:
                                break
                        if x - y in S:
                            conf_count += 1
                            if conf_count > best_score:
                                break
                    
                    if conf_count < best_score:
                        best_score = conf_count
                        best_candidate = x
                        if best_score == 0:
                            break # Found a free element, can't do better
                
                if best_candidate is not None:
                    S.add(best_candidate)
                    remaining.discard(best_candidate)
                    
            return list(S)

        def local_search_swap(s, c, v_set):
            """
            Advanced Local Search: Try to swap 1 out, 1 in.
            If that fails, try 2 out, 3 in?
            """
            s = set(s)
            c = set(c)
            
            improved = True
            while improved and time.time() < DEADLINE:
                improved = False
                
                # 1. Try to add any free element
                free_found = False
                for x in c:
                    if time.time() > DEADLINE: break
                    # Check if x is free
                    is_free = True
                    if 2*x in s:
                        is_free = False
                    else:
                        for y in s:
                            if x + y in s or y - x in s or x - y in s:
                                is_free = False
                                break
                    if is_free:
                        s.add(x)
                        c.discard(x)
                        improved = True
                        free_found = True
                        break
                
                if free_found:
                    continue
                
                # 2. Try swaps: Remove y from s, add x from c
                # We need x to be free *if* y is removed.
                # Conflicts of x with s\{y} must be 0.
                # This means x only conflicts with y in s.
                
                # Precompute conflicts of x with s?
                # For each x in c, find which elements in s conflict with it.
                # If count is 1, we can swap.
                
                best_swap = None
                min_conf = 2 # We only care about 1 conflict for net 0 size change? 
                             # Actually, swap 1 for 1 keeps size same. 
                             # We want to increase size. 
                             # So we should look for: Remove k from s, add k+1 to c.
                             # Start with k=1, add 2?
                             
                # Let's try "Remove 1, Add 1" to unlock further additions?
                # Or just "Remove 1, Add 2"?
                
                # Simpler: Just try to find x in c that has exactly 1 conflict in s.
                # Swap it. This might not increase size, but it might change the structure
                # to allow a free addition in the next iteration.
                
                for x in c:
                    if time.time() > DEADLINE: break
                    conflicts = []
                    for y in s:
                        if 2*x in s and y == x: # x==y case handled by 2x? No, y is in s, x in c.
                            pass
                        # Check conflict between x and y
                        # Conflict if:
                        # 1. x == y (not possible, disjoint)
                        # 2. 2*x == y (y is double of x) -> if we add x, and y is in s, is it bad?
                        #    Bad if x+x=y. So if 2x is in s, x conflicts with 2x? 
                        #    No, x and 2x can coexist ONLY IF x is not in s? 
                        #    Wait. If x is added, and 2x is in s, then x+x=2x is a violation.
                        #    So x conflicts with 2x.
                        # 3. x + y in s
                        # 4. y - x in s
                        # 5. x - y in s
                        
                        if y == 2*x:
                            conflicts.append(y)
                            continue
                        if x + y in s:
                            conflicts.append(y)
                            continue
                        if y - x in s:
                            conflicts.append(y)
                            continue
                        if x - y in s:
                            conflicts.append(y)
                            continue
                            
                    if len(conflicts) == 1:
                        y_to_remove = conflicts[0]
                        # Perform swap
                        s.remove(y_to_remove)
                        s.add(x)
                        c.discard(x)
                        c.add(y_to_remove)
                        improved = True
                        break
                        
            return list(s)

        # Run multiple strategies and pick best
        
        # Strategy A: Odds
        s_odds = set(x for x in values if x % 2 != 0)
        c_odds = set(x for x in values if x % 2 == 0)
        res_odds = local_search_swap(s_odds, c_odds, v_set)
        
        # Strategy B: High Pos
        s_high = set(x for x in values if x > 5000)
        c_high = set(x for x in values if x <= 5000)
        res_high = local_search_swap(s_high, c_high, v_set)
        
        # Strategy C: Greedy by Degree
        res_greedy = greedy_by_degree(values)
        # Refine greedy with local search
        c_greedy = v_set - set(res_greedy)
        res_greedy_refined = local_search_swap(set(res_greedy), c_greedy, v_set)
        
        # Strategy D: Random Restarts of Greedy
        best_rand = []
        for _ in range(5):
            if time.time() > DEADLINE: break
            perm = list(values)
            random.shuffle(perm)
            s_r = set()
            for x in perm:
                ok = True
                if 2*x in s_r:
                    ok = False
                else:
                    for y in s_r:
                        if x+y in s_r or y-x in s_r or x-y in s_r:
                            ok = False
                            break
                if ok:
                    s_r.add(x)
            if len(s_r) > len(best_rand):
                best_rand = list(s_r)
        
        c_rand = v_set - set(best_rand)
        res_rand_refined = local_search_swap(set(best_rand), c_rand, v_set)

        candidates = [res_odds, res_high, res_greedy_refined, res_rand_refined]
        
        best = []
        for c in candidates:
            if is_sum_free(c) and len(c) > len(best):
                best = c
                
        return best

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        if time.time() > DEADLINE:
            vals = instance(size, seed)
            results.append([x for x in vals if x % 2])
            continue
        
        vals = instance(size, seed)
        res = solve_instance(vals)
        
        if not is_sum_free(res):
            res = [x for x in vals if x % 2]
            
        results.append(res)
        
    return results
