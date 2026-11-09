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
        best_set = S[:]
    
    # Reverse order
    S = greedy(list(range(N, 0, -1)))
    if len(S) > best_size:
        best_size = len(S)
        best_set = S[:]
    
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
        best_set = S[:]
    
    # Sort by degree in conflict graph (elements with more conflicts first)
    def conflict_degree(x):
        deg = 0
        for y in range(1, N + 1):
            if y != x and (conflict[x][y] & full_mask):
                deg += 1
        return deg
    
    order2 = sorted(range(1, N + 1), key=conflict_degree, reverse=True)
    S = greedy(order2)
    if len(S) > best_size:
        best_size = len(S)
        best_set = S[:]
    
    # Local search polish: try to add elements or swap
    def local_improve(S):
        S_set = set(S)
        S_mask = 0
        for x in S:
            S_mask |= (1 << x)
        
        improved = True
        while improved:
            improved = False
            # Try adding each non-member
            for x in range(1, N + 1):
                if x in S_set:
                    continue
                # Check if x can be added
                can_add = True
                for s in S:
                    if conflict[s][x] & S_mask:
                        can_add = False
                        break
                if can_add:
                    S.append(x)
                    S_set.add(x)
                    S_mask |= (1 << x)
                    improved = True
            
            # Try swaps: remove one, add another
            for i, s in enumerate(S):
                S_without = S[:i] + S[i+1:]
                S_without_set = set(S_without)
                S_without_mask = 0
                for y in S_without:
                    S_without_mask |= (1 << y)
                
                for x in range(1, N + 1):
                    if x in S_without_set:
                        continue
                    can_add = True
                    for s2 in S_without:
                        if conflict[s2][x] & S_without_mask:
                            can_add = False
                            break
                    if can_add:
                        # Only swap if it helps (same size, but might enable more additions later)
                        # For now, just accept same-size swaps to explore
                        S = S_without + [x]
                        S_set = set(S)
                        S_mask = 0
                        for y in S:
                            S_mask |= (1 << y)
                        improved = True
                        break
                if improved:
                    break
        
        return S
    
    best_set = local_improve(best_set)
    best_size = len(best_set)
    
    # Branch and bound with better heuristics
    def solve(selected, available, depth=0):
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
        
        # Get list of available elements
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
        # For each x, count how many y in avail_list have conflicts with x within available
        best_x = avail_list[0]
        best_deg = -1
        for x in avail_list:
            deg = 0
            for y in avail_list:
                if x != y and (conflict[x][y] & available):
                    deg += 1
            if deg > best_deg:
                best_deg = deg
                best_x = x
        
        x = best_x
        
        # Branch 1: Include x
        new_selected = selected + [x]
        new_avail = available & ~(1 << x)
        # Remove all elements that conflict with x given selected
        for s in selected:
            new_avail &= ~conflict[s][x]
        
        solve(new_selected, new_avail, depth + 1)
        
        if time.time() - start_time > deadline:
            return
        
        # Branch 2: Exclude x
        solve(selected, available & ~(1 << x), depth + 1)
    
    # Start search
    solve([], full_mask)
    
    # Final local improvement
    if len(best_set) > 0:
        improved = local_improve(best_set)
        if len(improved) > len(best_set):
            best_set = improved
            best_size = len(best_set)
    
    return best_set
