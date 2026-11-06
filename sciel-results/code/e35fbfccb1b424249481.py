"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    N = 128
    
    # Precompute GCD table
    gcd_table = [[0]*(N+1) for _ in range(N+1)]
    for i in range(1, N+1):
        for j in range(i+1, N+1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    
    # Precompute conflict masks: conflict_mask[x] is a bitmask of elements y
    # such that there exist a, b in the current set... 
    # Actually, the conflict depends on the pair (a,b) already in the set.
    # We need a different representation for fast checking.
    
    # For a fixed set S, x can be added iff for all pairs (a,b) in S, 
    # NOT (gcd(a,x) == gcd(b,x) == gcd(a,b)).
    
    # Let's precompute for each x and each possible gcd value g, the set of y
    # such that gcd(x,y) == g.
    # gcd_sets[x][g] = bitmask of y where gcd(x,y) == g
    
    gcd_sets = {}
    for x in range(1, N+1):
        gcd_sets[x] = {}
        for y in range(1, N+1):
            if x == y:
                continue
            g = gcd_table[x][y]
            if g not in gcd_sets[x]:
                gcd_sets[x][g] = 0
            gcd_sets[x][g] |= (1 << y)
    
    def count_bad_triples(x, selected_mask):
        """
        Count bad triples involving x and two elements from selected_mask.
        A triple (a, b, x) is bad if gcd(a,x) == gcd(b,x) == gcd(a,b).
        Let g = gcd(a,x) = gcd(b,x). Then we need gcd(a,b) == g.
        
        For a fixed g, let A_g = {a in selected_mask : gcd(a,x) == g}.
        We need to count pairs (a,b) in A_g such that gcd(a,b) == g.
        
        Note: if gcd(a,x) = g and gcd(b,x) = g, then g divides a and b.
        Let a = g*a', b = g*b', x = g*x'.
        gcd(a,b) = g * gcd(a', b').
        We need g * gcd(a', b') == g => gcd(a', b') == 1.
        So we need to count pairs (a', b') from {a/g : a in A_g} that are coprime.
        
        Alternatively, just iterate.
        """
        # Optimization: group elements in selected_mask by their gcd with x.
        # For each g, get the list of elements a in selected_mask with gcd(a,x) == g.
        # Then count pairs (a,b) in that list with gcd(a,b) == g.
        
        count = 0
        # Iterate through elements in selected_mask
        # To do this fast, we can iterate through bits.
        # But N=128, so iterating 1..128 is fine.
        
        # Group by gcd with x
        groups = {}
        mask = selected_mask
        while mask:
            # Get lowest set bit
            lb = mask & (-mask)
            a = lb.bit_length()
            mask ^= lb
            g = gcd_table[a][x]
            if g not in groups:
                groups[g] = []
            groups[g].append(a)
        
        # For each group, count bad pairs
        for g, elems in groups.items():
            k = len(elems)
            if k < 2:
                continue
            for i in range(k):
                a = elems[i]
                for j in range(i+1, k):
                    b = elems[j]
                    if gcd_table[a][b] == g:
                        count += 1
        return count
    
    def can_add(x, selected_mask):
        """Check if x can be added to the set represented by selected_mask."""
        return count_bad_triples(x, selected_mask) == 0
    
    def greedy_addition(order, selected_mask=0):
        """Run greedy addition with given order."""
        for x in order:
            if (selected_mask >> x) & 1:
                continue
            if can_add(x, selected_mask):
                selected_mask |= (1 << x)
        return selected_mask
    
    def greedy_removal(seed=0):
        """
        Start with all elements, iteratively remove the element 
        involved in the most bad triples.
        """
        selected_mask = (1 << (N+1)) - 2  # bits 1..N set
        rng = random.Random(seed)
        
        while True:
            # Count bad triples for each element in selected
            counts = {}
            mask = selected_mask
            while mask:
                lb = mask & (-mask)
                x = lb.bit_length()
                mask ^= lb
                counts[x] = count_bad_triples(x, selected_mask)
            
            if not counts:
                break
            
            max_count = max(counts.values())
            if max_count == 0:
                break
            
            candidates = [x for x, c in counts.items() if c == max_count]
            
            if seed == 0:
                to_remove = min(candidates)
            elif seed == 1:
                to_remove = max(candidates)
            else:
                to_remove = rng.choice(candidates)
                
            selected_mask &= ~(1 << to_remove)
            
            if time.time() > deadline - 10:
                break
                
        return selected_mask
    
    def local_search(selected_mask, time_limit):
        """
        Try to improve the set via local search.
        1. Try to add any element.
        2. Try swaps: remove one, add two.
        """
        best = selected_mask
        
        while time.time() < time_limit:
            improved = False
            
            # 1. Try to add any element not in set
            for x in range(1, N+1):
                if (best >> x) & 1:
                    continue
                if can_add(x, best):
                    best |= (1 << x)
                    improved = True
                    break
            
            if improved:
                continue
            
            # 2. Try swapping: remove one, add two
            # Iterate over elements in best
            mask_iter = best
            while mask_iter:
                lb = mask_iter & (-mask_iter)
                x = lb.bit_length()
                mask_iter ^= lb
                
                temp_best = best & ~(1 << x)
                
                # Try to add two elements to temp_best
                added_count = 0
                new_mask = temp_best
                
                for y in range(1, N+1):
                    if (new_mask >> y) & 1 or y == x:
                        continue
                    if can_add(y, new_mask):
                        new_mask |= (1 << y)
                        added_count += 1
                        if added_count >= 2:
                            break
                
                if added_count >= 2:
                    best = new_mask
                    improved = True
                    break
            
            if not improved:
                break
                
        return best
    
    best_result_mask = 0
    
    # Strategy 1: Greedy removal with different seeds
    for seed in range(20):
        if time.time() > deadline - 20:
            break
        
        current_mask = greedy_removal(seed)
        current_size = bin(current_mask).count('1')
        best_size = bin(best_result_mask).count('1')
        
        if current_size > best_size:
            best_result_mask = current_mask
    
    # Strategy 2: Greedy addition with different orderings
    # Ascending
    result_mask = greedy_addition(range(1, N+1))
    if bin(result_mask).count('1') > bin(best_result_mask).count('1'):
        best_result_mask = result_mask
    
    # Descending
    result_mask = greedy_addition(range(N, 0, -1))
    if bin(result_mask).count('1') > bin(best_result_mask).count('1'):
        best_result_mask = result_mask
    
    # Random orderings
    rng = random.Random(42)
    for trial in range(10):
        if time.time() > deadline - 10:
            break
        order = list(range(1, N+1))
        rng.shuffle(order)
        result_mask = greedy_addition(order)
        if bin(result_mask).count('1') > bin(best_result_mask).count('1'):
            best_result_mask = result_mask
    
    # Local search on the best result
    best_result_mask = local_search(best_result_mask, deadline - 2)
    
    # Convert mask to list
    result = []
    mask = best_result_mask
    while mask:
        lb = mask & (-mask)
        x = lb.bit_length()
        mask ^= lb
        result.append(x)
    
    return result if result else [1]
# EVOLVE-BLOCK-END
