"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from collections import Counter

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    N = 128
    # Precompute GCD table
    gcd_table = [[0]*(N+1) for _ in range(N+1)]
    for i in range(1, N+1):
        for j in range(i+1, N+1):
            gcd_table[i][j] = gcd(i, j)
            gcd_table[j][i] = gcd_table[i][j]
    
    def count_bad_triples_for(x, selected_set):
        """
        Count how many bad triples involve x.
        A bad triple is (a, b, x) where a, b are in selected_set 
        and gcd(a,b) == gcd(a,x) == gcd(b,x).
        """
        count = 0
        # Group other elements in selected_set by their gcd with x
        # If gcd(a, x) == g, then for (a, b, x) to be bad, 
        # we need gcd(a, b) == g and gcd(b, x) == g.
        # So we only need to check pairs (a, b) where both have gcd g with x.
        
        # Optimization: Instead of iterating all pairs, we can iterate 
        # and check. With N=128, max pairs is ~8000. This is fast enough.
        
        # Let's try a slightly smarter way:
        # For each a in selected_set:
        #   g_ax = gcd(a, x)
        #   For each b in selected_set, b > a:
        #     g_bx = gcd(b, x)
        #     if g_ax == g_bx:
        #        if gcd(a, b) == g_ax:
        #           count += 1
           
        # Actually, just checking all pairs is O(K^2) where K is set size.
        # K <= 128. 128^2 = 16384 operations. Very fast.
        
        lst = list(selected_set)
        k = len(lst)
        for i in range(k):
            a = lst[i]
            g_ax = gcd_table[a][x]
            for j in range(i+1, k):
                b = lst[j]
                g_bx = gcd_table[b][x]
                if g_ax == g_bx:
                    if gcd_table[a][b] == g_ax:
                        count += 1
        return count

    def greedy_removal(seed=0):
        """
        Start with all elements, iteratively remove the element 
        involved in the most bad triples.
        """
        selected = set(range(1, N+1))
        rng = random.Random(seed)
        
        while True:
            # Count bad triples for each element in selected
            counts = {}
            for x in selected:
                counts[x] = count_bad_triples_for(x, selected)
                
            # Find max count
            max_count = max(counts.values()) if counts else 0
            
            if max_count == 0:
                break
            
            # Find all elements with max count
            candidates = [x for x, c in counts.items() if c == max_count]
            
            # Pick one to remove. 
            # Strategy: Remove the one that is "most central" or just random/priority.
            # Removing any one with max count reduces the total number of bad triples 
            # by at least max_count. 
            # To be more effective, we might prefer removing numbers with high frequency 
            # in other contexts, but for now, let's try a few heuristics.
            
            # Heuristic 1: Remove smallest
            # Heuristic 2: Remove largest
            # Heuristic 3: Remove random
            
            # Let's use a mix. For seed 0, smallest. For others, random.
            if seed == 0:
                to_remove = min(candidates)
            elif seed == 1:
                to_remove = max(candidates)
            else:
                to_remove = rng.choice(candidates)
                
            selected.remove(to_remove)
            
            # Early termination if time is up
            if time.time() > deadline - 10:
                break
                
        return selected

    def local_search(selected, time_limit):
        """
        Try to improve the set by swapping elements.
        """
        best = set(selected)
        
        while time.time() < time_limit:
            improved = False
            
            # 1. Try to add any element not in set
            for x in range(1, N+1):
                if x in best:
                    continue
                # Check if x can be added
                # This is equivalent to count_bad_triples_for(x, best) == 0
                if count_bad_triples_for(x, best) == 0:
                    best.add(x)
                    improved = True
                    break
            
            if improved:
                continue
            
            # 2. Try swapping: remove one, see if we can add two
            # Or just remove one and see if we can add one (net 0 size change, but might enable future adds)
            # Let's try: remove x, add y if it allows adding z later? 
            # Simpler: Remove x. Try to add y. If size increases or stays same but allows more adds, keep?
            # Actually, just try to find a swap that increases size.
            
            # Try removing each x
            for x in list(best):
                temp_best = best - {x}
                
                # Try to add elements not in temp_best
                added = 0
                new_elements = []
                for y in range(1, N+1):
                    if y in temp_best or y == x:
                        continue
                    if count_bad_triples_for(y, temp_best) == 0:
                        # Check if adding y creates bad triples with previously added new_elements?
                        # No, we check against temp_best which doesn't have new_elements yet.
                        # We need to check if y is compatible with temp_best AND any new_elements added so far.
                        
                        # To keep it simple and fast, let's just try to add one element at a time 
                        # in a greedy fashion on the temp_best.
                        
                        pass # This nested logic is getting complex. Let's use a helper.
                
                # Let's use a helper to greedily fill temp_best
                temp_list = list(temp_best)
                # Check if we can add at least 2 elements to temp_best
                # We can simulate adding them one by one.
                
                # Try adding every possible y
                for y in range(1, N+1):
                    if y in temp_best or y == x:
                        continue
                    if count_bad_triples_for(y, temp_best) == 0:
                        # Try to add y
                        temp_best_with_y = temp_best | {y}
                        
                        # Now try to add another z
                        for z in range(1, N+1):
                            if z in temp_best_with_y or z == x:
                                continue
                            if count_bad_triples_for(z, temp_best_with_y) == 0:
                                # We found a swap: remove x, add y, add z. Net +1 size.
                                best = temp_best_with_y | {z}
                                improved = True
                                break
                        if improved:
                            break
                if improved:
                    break
            
            if not improved:
                break
                
        return best

    best_result = set()
    
    # Run multiple greedy removals with different seeds
    for seed in range(20):
        if time.time() > deadline - 15:
            break
        
        current = greedy_removal(seed)
        
        if len(current) > len(best_result):
            best_result = current
        
        # If we have time, do local search on the best one found so far
        if time.time() < deadline - 5:
            improved = local_search(best_result, deadline - 2)
            if len(improved) > len(best_result):
                best_result = improved

    # Final local search on the absolute best
    if time.time() < deadline - 1:
        final_result = local_search(best_result, deadline - 0.5)
        if len(final_result) > len(best_result):
            best_result = final_result

    if not best_result:
        best_result = {1}
        
    return list(best_result)

# EVOLVE-BLOCK-END
