import time
import random
import math

def construct():
    start_time = time.time()
    DEADLINE = 55.0  # Safety margin before 60s
    
    N = 256
    
    # Precompute perfect squares up to 2*N = 512
    squares = set()
    k = 1
    while k * k <= 2 * N:
        squares.add(k * k)
        k += 1
    
    # Build adjacency/conflict sets
    # adj[i] contains j (0-indexed) such that (i+1) + (j+1) is a perfect square
    # Note: We treat the graph as undirected for the purpose of finding an Independent Set.
    # If i+j is a square, j+i is a square.
    adj = [set() for _ in range(N)]
    
    # Identify vertices with self-loops (2*v is a square)
    # These vertices cannot be in the solution set.
    forbidden = set()
    
    for i in range(N):
        v = i + 1
        # Check self-loop
        if 2 * v in squares:
            forbidden.add(i)
        
        # Find neighbors
        for sq in squares:
            u = sq - v
            if 1 <= u <= N:
                j = u - 1
                adj[i].add(j)
    
    # Candidates are vertices without self-loops
    candidates = [i for i in range(N) if i not in forbidden]
    
    # Precompute degrees for greedy ordering
    degrees = [len(adj[i]) for i in range(N)]
    
    best_size = 0
    best_set = frozenset()
    
    def greedy_solve(order=None):
        """
        Greedy construction. If order is None, use a randomized degree-based order.
        """
        if order is None:
            # Sort by degree with random tie-breaking
            # This tends to pick "less constrained" vertices first
            order = sorted(candidates, key=lambda v: (degrees[v], random.random()))
        
        current_set = set()
        
        for v in order:
            # Check if v conflicts with any u in current_set
            # v cannot conflict with itself (self-loops are removed)
            # We just need to ensure no neighbor is in current_set
            if not (adj[v] & current_set):
                current_set.add(v)
                
        return current_set

    def local_improve(s):
        """Try to add vertices that don't conflict."""
        s = set(s)
        improved = True
        while improved:
            improved = False
            # Try to add any vertex not in s that has no neighbors in s
            # Iterate over candidates to ensure we only add valid ones
            for v in candidates:
                if v in s:
                    continue
                if not (adj[v] & s):
                    s.add(v)
                    improved = True
        return s

    def local_search(s, max_iterations=100):
        """
        More sophisticated local search:
        1. Try to add vertices (like local_improve).
        2. If stuck, try to remove a vertex that allows adding multiple others?
           Or simply do random restarts with greedy + local_improve.
           Given N=256, greedy + local_improve is likely very effective.
           Let's add a simple swap heuristic:
           For each v in s, remove v, then run local_improve, if size increases, keep it.
        """
        s = set(s)
        # First, fill up
        s = local_improve(s)
        
        # Try swaps/removals
        improved = True
        iter_count = 0
        while improved and iter_count < max_iterations and (time.time() - start_time < DEADLINE - 1):
            improved = False
            iter_count += 1
            # Shuffle the order of vertices to try for removal to avoid bias
            vertices_to_try = list(s)
            random.shuffle(vertices_to_try)
            for v in vertices_to_try:
                # Try removing v
                s.remove(v)
                # Try to fill up again
                s_new = local_improve(s)
                if len(s_new) > len(s) + 1: # Must gain more than we lost
                    s = s_new
                    improved = True
                    break # Restart the loop with new s
                else:
                    # Restore v and try to fill (in case removal opened up space but not enough)
                    s.add(v)
                    s = local_improve(s)
        
        return s

    # Iterated Local Search / Random Restart Greedy
    while time.time() - start_time < DEADLINE:
        # Strategy 1: Random order
        if random.random() < 0.5:
            order = candidates[:]
            random.shuffle(order)
        else:
            # Strategy 2: Degree-based order with random tie-breaking
            order = sorted(candidates, key=lambda v: (degrees[v], random.random()))
            
        s = greedy_solve(order)
        s = local_improve(s)
        
        # Occasionally do a deeper local search if we have time
        if time.time() - start_time < DEADLINE - 5:
            s = local_search(s)
        
        if len(s) > best_size:
            best_size = len(s)
            best_set = frozenset(s)
            
    # Convert 0-based indices back to 1-based values
    return sorted([v + 1 for v in best_set])

# EVOLVE-BLOCK-END
