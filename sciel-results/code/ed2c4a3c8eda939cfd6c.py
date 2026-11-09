import time
import math
import sys

def construct():
    start_time = time.time()
    deadline = 55.0  # Safety margin for 60s limit (evaluator kills at 90s)
    
    N = 128
    full_mask = (1 << (N + 1)) - 2  # bits 1..N set
    
    # Precompute LCMs
    def lcm(a, b):
        return a * b // math.gcd(a, b)
    
    # Precompute bad triples: for each pair (i,j), store bitset of k such that {i,j,k} is bad
    conflict = [[0] * (N + 1) for _ in range(N + 1)]
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            lab = lcm(a, b)
            for c in range(b + 1, N + 1):
                if lab == lcm(a, c) == lcm(b, c):
                    conflict[a][b] |= (1 << c)
                    conflict[b][a] |= (1 << c)
                    conflict[a][c] |= (1 << b)
                    conflict[c][a] |= (1 << b)
                    conflict[b][c] |= (1 << a)
                    conflict[c][b] |= (1 << a)
    
    # Greedy lower bound: try multiple orderings
    def greedy(order):
        S = []
        S_mask = 0
        for x in order:
            if not (full_mask >> x) & 1:
                continue
            # Check if x is compatible with S
            compatible = True
            for s in S:
                if conflict[s][x] & S_mask:
                    compatible = False
                    break
            if compatible:
                S.append(x)
                S_mask |= (1 << x)
        return S
    
    # Try several greedy orderings
    best_set = []
    best_size = 0
    
    # Natural order
    S = greedy(list(range(1, N + 1)))
    if len(S) > best_size:
        best_size = len(S)
        best_set = S
    
    # Reverse order
    S = greedy(list(range(N, 0, -1)))
    if len(S) > best_size:
        best_size = len(S)
        best_set = S
    
    # Sort by number of prime factors (heuristic)
    def num_prime_factors(x):
        count = 0
        d = 2
        n = x
        while d * d <= n:
            while n % d == 0:
                count += 1
                n //= d
            d += 1
        if n > 1:
            count += 1
        return count
    
    order = sorted(range(1, N + 1), key=num_prime_factors)
    S = greedy(order)
    if len(S) > best_size:
        best_size = len(S)
        best_set = S
    
    # Branch and bound
    # State: selected (list), available (int mask of elements compatible with selected)
    # Invariant: available only contains elements that don't form bad triples with any pair in selected
    
    def upper_bound(available):
        """Upper bound on additional elements we can add.
        Use: the number of elements in available is an upper bound.
        Better: use independence number of the conflict graph on available.
        For now, use bit_count as a simple bound, but we can improve."""
        return available.bit_count()
    
    def solve(selected, available):
        nonlocal best_size, best_set, start_time
        
        if time.time() - start_time > deadline:
            return
        
        # Update best if we have a larger set
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = selected[:]
        
        if not available:
            return
        
        # Upper bound: can't exceed current best
        if len(selected) + available.bit_count() <= best_size:
            return
        
        # Branching: pick element with maximum degree in the conflict subgraph on available
        # This helps prune more aggressively
        # Degree of x in available: number of y in available such that conflict[x][y] & available != 0
        # Actually, we want to pick x that, if excluded, removes the most from available.
        # Or pick x with highest "impact".
        
        # Simple heuristic: pick the element with the most conflicts in available
        # For each x in available, count how many y in available have conflict[x][y] & available
        # This is expensive. Let's use a simpler heuristic: pick lowest x first, 
        # but try a few strategies.
        
        # Strategy: pick the element that appears in the most bad triples within available
        # Precompute degree: for each x, deg[x] = popcount(conflict[x][y] & available) for y in available
        # But we need to be careful: conflict[x][y] gives z such that {x,y,z} is bad.
        # The "degree" in the 3-uniform sense is complex.
        
        # Simpler: just pick the first available element (lowest bit)
        # But let's try picking one that has many conflicts to prune more
        
        # Find all available elements
        avail_list = []
        temp = available
        while temp:
            low = temp & (-temp)
            x = low.bit_length() - 1
            avail_list.append(x)
            temp ^= low
        
        if not avail_list:
            return
        
        # Pick element with highest conflict degree in available
        # For each x, count how many y in avail_list have (conflict[x][y] & available) != 0
        # This is O(k^2) per node, which is expensive.
        
        # Alternative: just pick the first one for speed
        x = avail_list[0]
        
        # Branch 1: Include x
        new_selected = selected + [x]
        new_avail = available & ~(1 << x)
        # Remove all elements that conflict with x given selected
        for s in selected:
            new_avail &= ~conflict[s][x]
        
        solve(new_selected, new_avail)
        
        if time.time() - start_time > deadline:
            return
        
        # Branch 2: Exclude x
        solve(selected, available & ~(1 << x))
    
    # Start search
    solve([], full_mask)
    
    return best_set
