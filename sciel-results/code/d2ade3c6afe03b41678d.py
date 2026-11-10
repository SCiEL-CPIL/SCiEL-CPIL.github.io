import random
import time
import math
from itertools import combinations

def _count_bad_6_subsets(adj, n):
    """Count the number of regular induced subgraphs of size 6."""
    count = 0
    for comb in combinations(range(n), 6):
        mask = 0
        for v in comb:
            mask |= (1 << v)
        first = -1
        regular = True
        for v in comb:
            d = (adj[v] & mask).bit_count()
            if first == -1:
                first = d
            elif d != first:
                regular = False
                break
        if regular:
            count += 1
    return count

def _find_any_bad_subset(adj, n, max_size=None):
    """Find any regular induced subgraph of size >= 6."""
    if max_size is None:
        max_size = n
    for size in range(6, max_size + 1):
        for comb in combinations(range(n), size):
            mask = 0
            for v in comb:
                mask |= (1 << v)
            first = -1
            regular = True
            for v in comb:
                d = (adj[v] & mask).bit_count()
                if first == -1:
                    first = d
                elif d != first:
                    regular = False
                    break
            if regular:
                return comb
    return None

def _flip_edge(adj, u, v):
    bit_v = 1 << v
    bit_u = 1 << u
    if adj[u] & bit_v:
        adj[u] &= ~bit_v
        adj[v] &= ~bit_u
    else:
        adj[u] |= bit_v
        adj[v] |= bit_u

def _to_edges(adj, n):
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i] & (1 << j):
                edges.append([i, j])
    return edges

def construct():
    start = time.time()
    deadline = start + 145
    
    rng = random.Random(42)
    best = None
    
    def elapsed():
        return time.time() - start
    
    def try_sa(n, time_budget):
        """Simulated Annealing to find a valid graph on n vertices."""
        target_time = time.time() + time_budget
        
        # Initialize
        adj = [0]*n
        for i in range(n):
            for j in range(i+1, n):
                if rng.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Precompute all 6-subsets masks for speed
        subsets_6 = list(combinations(range(n), 6))
        masks_6 = []
        for comb in subsets_6:
            mask = 0
            for v in comb:
                mask |= (1 << v)
            masks_6.append((comb, mask))
            
        def count_bad_6(adj):
            count = 0
            for comb, mask in masks_6:
                first = -1
                regular = True
                for v in comb:
                    d = (adj[v] & mask).bit_count()
                    if first == -1:
                        first = d
                    elif d != first:
                        regular = False
                        break
                if regular:
                    count += 1
            return count

        current_energy = count_bad_6(adj)
        
        if current_energy == 0:
             # Check larger sizes
             if _find_any_bad_subset(adj, n) is None:
                 return adj
             
        T = 1.0
        cooling = 0.999
        min_T = 0.01
        
        while time.time() < target_time and elapsed() < deadline - 5:
            if T < min_T:
                break
                
            # Pick random edge
            u = rng.randint(0, n-1)
            v = rng.randint(0, n-1)
            if u == v:
                continue
                
            # Flip
            _flip_edge(adj, u, v)
            new_energy = count_bad_6(adj)
            
            delta = new_energy - current_energy
            
            if delta < 0 or rng.random() < math.exp(-delta / T):
                current_energy = new_energy
                # If energy is 0, check if fully valid
                if current_energy == 0:
                    if _find_any_bad_subset(adj, n) is None:
                        return adj
            else:
                # Revert
                _flip_edge(adj, u, v)
                
            T *= cooling
            
        return None

    # Try n=20
    remaining = deadline - 10 - elapsed()
    if remaining > 0:
        budget = min(100, remaining)
        if budget > 10:
            adj = try_sa(20, budget)
            if adj is not None:
                # Double check
                if _find_any_bad_subset(adj, 20) is None:
                    edges = _to_edges(adj, 20)
                    best = (20, edges)
                    
    if best is None:
        # Try n=19
        remaining = deadline - 10 - elapsed()
        if remaining > 5:
            budget = min(30, remaining)
            if budget > 5:
                adj = try_sa(19, budget)
                if adj is not None:
                    if _find_any_bad_subset(adj, 19) is None:
                        edges = _to_edges(adj, 19)
                        best = (19, edges)
                        
    if best is None:
        # Try n=18
        remaining = deadline - 10 - elapsed()
        if remaining > 5:
            budget = min(20, remaining)
            if budget > 5:
                adj = try_sa(18, budget)
                if adj is not None:
                    if _find_any_bad_subset(adj, 18) is None:
                        edges = _to_edges(adj, 18)
                        best = (18, edges)
                        
    if best is None:
        for n_try in [17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5]:
            if elapsed() > deadline - 5:
                break
            remaining = deadline - 5 - elapsed()
            if remaining < 2:
                break
            adj = try_sa(n_try, min(5, remaining))
            if adj is not None:
                if _find_any_bad_subset(adj, n_try) is None:
                    edges = _to_edges(adj, n_try)
                    best = (n_try, edges)
                    break
                    
    if best is None:
        return {'n': 5, 'edges': []}
        
    return {'n': best[0], 'edges': best[1]}
