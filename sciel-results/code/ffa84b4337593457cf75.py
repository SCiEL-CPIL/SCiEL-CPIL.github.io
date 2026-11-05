"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random
import sys

# EVOLVE-BLOCK-START
def _is_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def _build_adj(n=256):
    """Return list of sets: adj[i] = set of j != i such that i+j is a perfect square."""
    adj = [set() for _ in range(n + 1)]
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            if _is_square(a + b):
                adj[a].add(b)
                adj[b].add(a)
    return adj

def _greedy(adj, n, order_seed=0):
    """Greedy MIS: process vertices in given order, add if no conflict with selected."""
    rng = random.Random(order_seed)
    # Order: sort by degree ascending, ties broken randomly
    verts = list(range(1, n + 1))
    verts.sort(key=lambda v: (len(adj[v]), rng.random()))
    selected = set()
    for v in verts:
        if not selected & adj[v]:
            selected.add(v)
    return selected

def _try_add(adj, selected, v):
    """Check if v can be added to selected."""
    return not (selected & adj[v])

def _local_search(adj, selected, n, max_iters=200000, seed=0, time_limit=15.0):
    """Local search with add and swap moves."""
    rng = random.Random(seed)
    selected = set(selected)
    best = set(selected)
    best_size = len(best)
    start = time.time()
    
    # Precompute for each vertex its neighbors
    # Use a counter for how many selected neighbors each unselected vertex has
    neighbor_count = [0] * (n + 1)
    for v in selected:
        for u in adj[v]:
            if u not in selected:
                neighbor_count[u] += 1
    
    for _ in range(max_iters):
        if time.time() - start > time_limit:
            break
        
        # Try to add a vertex with 0 selected neighbors
        added = False
        # Find candidates with 0 neighbors in selected
        zero_candidates = [v for v in range(1, n + 1) if v not in selected and neighbor_count[v] == 0]
        
        if zero_candidates:
            v = rng.choice(zero_candidates)
            selected.add(v)
            for u in adj[v]:
                if u not in selected:
                    neighbor_count[u] += 1
            added = True
            if len(selected) > best_size:
                best = set(selected)
                best_size = len(selected)
        
        if not added:
            # Try swap: remove one vertex, see if we can add more
            improved = False
            for v in list(selected):
                # Remove v temporarily
                selected.remove(v)
                for u in adj[v]:
                    if u not in selected:
                        neighbor_count[u] -= 1
                        if neighbor_count[u] < 0:
                            neighbor_count[u] = 0
                
                # Try to add as many as possible
                added_count = 0
                # Get all vertices that can now be added
                addable = [u for u in range(1, n + 1) if u not in selected and neighbor_count[u] == 0]
                for u in addable:
                    selected.add(u)
                    added_count += 1
                    for w in adj[u]:
                        if w not in selected:
                            neighbor_count[w] += 1
                
                if added_count > 1:
                    improved = True
                    if len(selected) > best_size:
                        best = set(selected)
                        best_size = len(selected)
                else:
                    # Revert
                    for u in addable:
                        selected.discard(u)
                    # Re-add v
                    selected.add(v)
                    for u in adj[v]:
                        if u not in selected:
                            neighbor_count[u] += 1
                
                if improved:
                    break
            
            if not improved:
                break
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 75  # Safety margin
    
    n = 256
    adj = _build_adj(n)
    
    # Best solution so far
    best = set()
    best_size = 0
    
    # Multiple greedy starts
    for seed in range(200):
        if time.time() > deadline - 20:
            break
        candidate = _greedy(adj, n, order_seed=seed)
        if len(candidate) > best_size:
            best = candidate
            best_size = len(candidate)
    
    # Local search from best
    remaining = deadline - time.time()
    if remaining > 5:
        improved = _local_search(adj, best, n, max_iters=500000, seed=0, time_limit=min(remaining - 5, 30))
        if len(improved) > best_size:
            best = improved
            best_size = len(improved)
    
    # More local search
    remaining = deadline - time.time()
    for seed in range(1, 10):
        if remaining < 5:
            break
        improved = _local_search(adj, best, n, max_iters=200000, seed=seed, time_limit=min(remaining - 5, 15))
        if len(improved) > best_size:
            best = improved
            best_size = len(improved)
        remaining = deadline - time.time()
    
    return list(best)
# EVOLVE-BLOCK-END
