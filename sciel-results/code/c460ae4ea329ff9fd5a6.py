import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85 second safety margin
    
    N = 256
    # Precompute perfect squares up to 512
    max_sum = 2 * N
    squares = set()
    i = 1
    while i * i <= max_sum:
        squares.add(i * i)
        i += 1
    
    # Build conflict graph: conflict[a][b] = True if a+b is a perfect square
    # We only need a <= b
    # Use a set of conflicts for each node
    conflicts = [set() for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a, N + 1):
            if a + b in squares:
                conflicts[a].add(b)
                if a != b:
                    conflicts[b].add(a)
    
    # Greedy independent set with multiple random orderings
    best_set = set()
    best_size = 0
    
    # First try a deterministic greedy: pick vertices with lowest degree first
    def greedy_independent_set(order):
        selected = set()
        for v in order:
            # Check if v conflicts with any selected vertex
            if not (conflicts[v] & selected):
                selected.add(v)
        return selected
    
    # Try multiple random orderings
    random.seed(42)
    for trial in range(200):
        if time.time() > deadline:
            break
        order = list(range(1, N + 1))
        random.shuffle(order)
        s = greedy_independent_set(order)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
    
    # Local search: try to add vertices that don't conflict
    # Also try to swap: remove one vertex to add two others
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try adding each unselected vertex
        for v in range(1, N + 1):
            if v not in best_set:
                if not (conflicts[v] & best_set):
                    best_set.add(v)
                    best_size += 1
                    improved = True
        
        # Try swaps: remove one vertex, see if we can add more
        if time.time() > deadline:
            break
        for v in list(best_set):
            if time.time() > deadline:
                break
            # Temporarily remove v
            best_set.remove(v)
            added = []
            for w in range(1, N + 1):
                if w not in best_set and w != v:
                    if not (conflicts[w] & best_set):
                        best_set.add(w)
                        added.append(w)
            # If we added more than we removed, keep it
            if len(added) > 1:
                best_size += len(added) - 1
                improved = True
            else:
                # Revert
                for w in added:
                    best_set.remove(w)
                best_set.add(v)
    
    return sorted(best_set)
