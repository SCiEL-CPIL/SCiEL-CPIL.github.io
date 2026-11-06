"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    # Precompute which numbers are "self-conflicting" (2a is a perfect square)
    # and build the conflict graph
    N = 256
    squares = set()
    for k in range(2, 24):  # k^2 up to 23^2=529 > 512
        sq = k * k
        if sq >= 2 and sq <= 2*N:
            squares.add(sq)
    
    # A number a is self-conflicting if 2a is a perfect square
    self_conflict = set()
    for a in range(1, N+1):
        if 2*a in squares:
            self_conflict.add(a)
    
    # Build adjacency: a conflicts with b if a+b is a perfect square
    # We'll use a bitset-like approach with lists for efficiency
    # conflicts[a] = set of b (b >= a) such that a+b is a perfect square
    conflicts = [set() for _ in range(N+1)]
    for a in range(1, N+1):
        for b in range(a, N+1):
            if a + b in squares:
                conflicts[a].add(b)
                if a != b:
                    conflicts[b].add(a)
    
    # Greedy: start with all non-self-conflicting numbers, then remove conflicts
    # Better: use a greedy coloring / independent set approach
    
    # Strategy: Greedy by degree (pick vertices with fewest conflicts first)
    # Then local search to improve
    
    def greedy_independent_set():
        # Start with all candidates (excluding self-conflicting)
        candidates = set(range(1, N+1)) - self_conflict
        # Remove self-conflicts from consideration
        result = set()
        remaining = set(candidates)
        
        # Greedy: repeatedly pick the vertex with minimum degree in remaining graph
        while remaining:
            if time.time() > deadline:
                break
            min_deg = float('inf')
            best_v = None
            for v in remaining:
                deg = len(conflicts[v] & remaining)
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
                    if deg == 0:
                        break
            if best_v is None:
                break
            result.add(best_v)
            remaining.discard(best_v)
            # Remove all neighbors of best_v from remaining
            to_remove = conflicts[best_v] & remaining
            remaining -= to_remove
        return result
    
    # Local search: try to add vertices back, or swap to improve
    def local_search(s):
        s = set(s)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try adding each excluded vertex
            excluded = set(range(1, N+1)) - s - self_conflict
            for v in sorted(excluded):
                if time.time() > deadline:
                    break
                # Check if v can be added
                if v in self_conflict:
                    continue
                if conflicts[v] & s:
                    continue
                # v can be added
                s.add(v)
                improved = True
            # Try swaps: remove one, add another
            for v in list(s):
                if time.time() > deadline:
                    break
                s.remove(v)
                # Try to add multiple vertices
                for w in sorted(excluded):
                    if time.time() > deadline:
                        break
                    if w in self_conflict:
                        continue
                    if w in s:
                        continue
                    if conflicts[w] & s:
                        continue
                    s.add(w)
                    if len(s) > 0:
                        improved = True
                s.add(v)  # restore
        return s
    
    # Start with greedy
    best_set = greedy_independent_set()
    best_set = local_search(best_set)
    best_size = len(best_set)
    
    # Try multiple random restarts
    import random
    random.seed(42)
    
    for restart in range(200):
        if time.time() > deadline - 5.0:
            break
        
        # Random greedy with random ordering
        candidates = list(set(range(1, N+1)) - self_conflict)
        random.shuffle(candidates)
        
        result = set()
        remaining = set(candidates)
        
        for v in candidates:
            if time.time() > deadline - 2.0:
                break
            if v not in remaining:
                continue
            if conflicts[v] & result:
                continue
            result.add(v)
            remaining.discard(v)
            remaining -= (conflicts[v] & remaining)
        
        # Local search
        result = local_search(result)
        
        if len(result) > best_size:
            best_size = len(result)
            best_set = result
    
    # Final local search on best
    best_set = local_search(best_set)
    
    return sorted(best_set)
# EVOLVE-BLOCK-END
