import time
import random
from itertools import combinations

N = 192
MAX_SIZE = 48
K = 3

def construct():
    start = time.time()
    # Keep a safety margin. Evaluator kills at 90s. 
    # Let's aim to finish by 85s to be safe, but the current code used 80s.
    # I will use 85s as the hard stop for internal loops, returning best found.
    deadline = start + 85.0

    # Precompute conflict relationships?
    # Two numbers x and y "conflict" in a specific context?
    # The constraint is: for any x in S, x does not divide sum of any 3 distinct others in S.
    # This is a global property.
    # However, we can define a "pairwise conflict" or "degree" metric.
    
    # Let's precompute for every pair (a, b) and every potential x, 
    # does x divide a+b+c for some c? No, c is also in S.
    
    # A more direct approach for "degree":
    # When considering adding x to S, the "cost" is how many new conflicts it creates.
    # Conflict 1: x divides sum of 3 in S.
    # Conflict 2: Any a in S divides x + b + c for distinct b, c in S.
    
    # To speed up the greedy, we can precompute:
    # 1. For each x, list of triples (a,b,c) from 1..N such that x | (a+b+c).
    #    This is too many. N=192. Triples ~ 192^3 / 6 ~ 1.1M. 
    #    For each x, checking if any of these triples are in S is fast if we use a set of triples?
    #    Or better: for each x, precompute the set of sums S_x = {a+b+c | a,b,c distinct in 1..N, x | (a+b+c)}.
    #    Then check if any triple in S sums to a value in S_x? No, we need to check if the triple IS in S.
    
    # Alternative: 
    # Precompute for each x, a bitset or set of "forbidden pairs" (b, c) such that for ANY a in S, 
    # a | (x + b + c). This depends on S.
    
    # Let's stick to the on-the-fly check but optimize it.
    # The current program's is_safe_add is O(n^3) roughly. 
    # n=48. 48^3 = 110k operations per check.
    # 192 candidates * 48 additions * 110k ops = ~1M ops per greedy pass. Very fast.
    # The bottleneck in the current program might be the Python overhead of combinations and loops.
    
    # New Strategy: 
    # 1. Greedy with "Conflict Degree" ordering.
    #    Instead of just picking largest, or random, we pick the candidate x that minimizes the 
    #    number of "potential conflicts" it creates with the current set S.
    #    Actually, we just need to pick a valid one. 
    #    If we pick a valid one with low "impact", it might leave more room for others.
    #    Impact metric: How many other candidates y (not in S) would become INVALID if we add x?
    #    This is expensive to compute for all y.
    
    # Simpler Heuristic: 
    # Pick the x that is "least restrictive" for the future.
    # Small numbers are restrictive (divide many things). Large numbers are less restrictive.
    # So, prefer large numbers? The current program does this.
    # But maybe large numbers create specific conflicts?
    
    # Let's try a different angle: 
    # Randomized Restart with "Simulated Annealing" style swaps?
    # Or "Greedy + 2-Swap Local Search" is the standard improvement.
    # The current program has a weak local search.
    # Let's implement a robust 2-Swap local search.
    
    # 1. Generate initial solution using Greedy (Largest First).
    # 2. Local Search: 
    #    - Try to add any remaining element.
    #    - If cannot add, try swapping one element in S with one not in S.
    #    - If swap increases size (impossible if size is max, but if size < max, swap is just add+remove? No, swap keeps size same. We want to INCREASE size.)
    #    - If size < MAX_SIZE, we just need to find an add.
    #    - If no add is possible, we must remove an element to make room? 
    #      If we remove x, size becomes k-1. Then we try to add TWO elements?
    #      Or just add one element y. If y is safe with S\{x}, size becomes k. No gain.
    #      We need to add y AND z such that size becomes k+1?
    #      This is a "1-remove, 2-add" move.
    
    def is_safe_add(x, S, S_set):
        n = len(S)
        # Check 1: x divides sum of any 3 distinct in S
        if n >= 3:
            # Optimization: Precompute sums of pairs?
            # For fixed x, we need a+b+c % x == 0.
            # c % x == (-a-b) % x.
            # We can count residues of S mod x.
            # But we need distinct a,b,c.
            # Let's use a set of pair sums mod x.
            pair_sums = set()
            # This is O(n^2). 
            # Then for each c in S, check if (-c) % x is in pair_sums formed by a,b != c?
            # This is tricky with distinctness.
            # Given n<=48, O(n^3) is ~110k. 192*48*110k is ~1M. 
            # Let's just do the triple loop but break early.
            for a, b, c in combinations(S, 3):
                if (a + b + c) % x == 0:
                    return False
        
        # Check 2: a in S divides (x + b + c)
        if n >= 2:
            for a in S:
                # Need b, c in S \ {a}, distinct, such that (x+b+c) % a == 0
                # (b+c) % a == (-x) % a
                target = (-x) % a
                # Collect residues of S \ {a}
                # We need two distinct elements b, c from S\{a} summing to target mod a.
                # Use a set of residues? 
                # If we have multiple elements with same residue, we need to handle counts.
                # Since n is small, just iterate pairs.
                # To speed up, precompute list of (val, idx) for S\{a}?
                # Or just nested loops.
                found = False
                # Optimization: 
                # rem_b = b % a. Need c % a == (target - rem_b) % a.
                # Create a dictionary: residue -> list of values (or just count, but need distinctness)
                # Actually, we just need to know if there exist distinct b, c.
                # Let's map residue to a list of indices or values.
                res_map = {}
                for v in S:
                    if v == a: continue
                    r = v % a
                    if r not in res_map:
                        res_map[r] = []
                    res_map[r].append(v)
                
                # Now check pairs of residues
                # To avoid duplicate checks, iterate over keys?
                # Or just iterate b in S\{a} and check if needed c exists.
                for b in S:
                    if b == a: continue
                    rb = b % a
                    rc_needed = (target - rb) % a
                    if rc_needed in res_map:
                        # Check if there is a c in res_map[rc_needed] such that c != b
                        # If rc_needed == rb, we need at least 2 elements in res_map[rb] (one is b, need another)
                        # If rc_needed != rb, we just need 1 element in res_map[rc_needed] (and it's automatically != b if residues differ? No, values are distinct, but residues can be same. If residues differ, values are definitely different? No. 1%4=1, 5%4=1. Distinct values. If residues differ, values are distinct? Yes, because a,b are distinct integers. Wait. If r1 != r2, then v1 % a != v2 % a, so v1 != v2. So if residues differ, any c in res_map[rc_needed] is distinct from b.)
                        if rc_needed == rb:
                            # Need another element with same residue
                            if len(res_map[rc_needed]) >= 2:
                                found = True
                                break
                        else:
                            if len(res_map[rc_needed]) >= 1:
                                found = True
                                break
                if found:
                    return False
        return True

    def get_conflict_degree(x, S, S_set):
        # How many elements y NOT in S are blocked by adding x?
        # This is expensive. 
        # Alternative: Just return 0 for all, and rely on sorting by value?
        # Let's try: Prefer x that has fewest "near misses" or something?
        # For now, let's just use value as proxy.
        return 0

    def greedy_construct(pool, S, S_set, deadline):
        # Sort pool by value descending
        pool_sorted = sorted(pool, reverse=True)
        for x in pool_sorted:
            if len(S) >= MAX_SIZE:
                break
            if time.time() > deadline:
                break
            if x not in S_set:
                if is_safe_add(x, S, S_set):
                    S.append(x)
                    S_set.add(x)
        return S, S_set

    def local_search(S, S_set, deadline, max_iters=200):
        # 1. Try to add any element.
        # 2. If can't, try 1-remove, 2-add.
        
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1
            current_size = len(S)
            
            if current_size >= MAX_SIZE:
                break
                
            # Try adding
            remaining = [x for x in range(1, N+1) if x not in S_set]
            # Sort by value descending? Or random?
            # Let's try random order to escape local minima of "largest first"
            random.shuffle(remaining)
            
            added = False
            for x in remaining:
                if is_safe_add(x, S, S_set):
                    S.append(x)
                    S_set.add(x)
                    improved = True
                    added = True
                    break
            
            if not added:
                # Try 1-remove, 2-add
                # Remove x from S.
                # Then try to add two elements y, z.
                # To be efficient, for each x in S:
                #   S_temp = S \ {x}
                #   Try to add y to S_temp.
                #   If y added, try to add z to S_temp U {y}.
                #   If z added, we have size current_size + 1.
                
                # This is O(|S| * |Remaining| * |Remaining| * CostCheck).
                # 48 * 144 * 144 * 110k = ~10^10. Too slow.
                
                # Optimization: 
                # Only try a few x's. 
                # Or, precompute "safe pairs" for the remaining set relative to S\{x}?
                
                # Let's try a simpler swap: 1-remove, 1-add (size stays same). 
                # This doesn't increase size. 
                # But it might change the set to allow an ADD in the next iteration.
                
                # Let's try: Remove x, Add y (if safe). 
                # This is a "move" that keeps size same.
                # If we do this, we might unblock an add.
                
                # Strategy: 
                # Pick x to remove (e.g., smallest, or most "conflicting").
                # Pick y to add (from remaining).
                # If y is safe with S\{x}, do the swap.
                # Then loop back to try to add.
                
                # To avoid infinite loops, we can track visited states or just limit iterations.
                
                # Heuristic for x: Smallest element is most likely to be the blocker?
                # Or the one with highest "degree"?
                # Let's try removing the smallest element in S.
                
                if S:
                    x_remove = min(S)
                    S_temp = [v for v in S if v != x_remove]
                    S_temp_set = set(S_temp)
                    
                    # Try to add y to S_temp
                    # Prefer large y?
                    remaining_sorted = sorted(remaining, reverse=True)
                    for y in remaining_sorted:
                        if is_safe_add(y, S_temp, S_temp_set):
                            # Do the swap
                            S = S_temp + [y]
                            S_set = S_temp_set | {y}
                            improved = True
                            break
                            
        return S

    # Main Loop
    best_S = []
    best_set = set()
    
    # Run multiple random restarts
    num_restarts = 50
    for i in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        S = []
        S_set = set()
        
        # Randomize the initial pool order?
        # Or just use the standard greedy.
        # Let's mix: 50% largest first, 50% random.
        if i % 2 == 0:
            pool = list(range(N, 0, -1))
        else:
            pool = list(range(1, N+1))
            random.shuffle(pool)
            
        S, S_set = greedy_construct(pool, S, S_set, deadline)
        
        # Local Search
        S = local_search(S, S_set, deadline)
        
        if len(S) > len(best_S):
            best_S = S
            best_set = set(S)
            
            # Early exit if we hit max size
            if len(best_S) == MAX_SIZE:
                break
                
    return best_S

# Quick sanity check / self-test
if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
    print(res)
