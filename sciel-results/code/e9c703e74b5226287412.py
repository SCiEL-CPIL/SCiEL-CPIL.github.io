import random
import time
import sys

sys.setrecursionlimit(10000)

def _time_up(deadline):
    return time.time() > deadline

def _has_triangle(adj, n):
    """Check if graph is triangle-free using bitmasks."""
    for i in range(n):
        for j in adj[i]:
            if j < i:
                continue
            if adj[i] & adj[j]:
                return True
    return False

def _exact_alpha(adj, n, deadline):
    """
    Returns the size of the maximum independent set.
    Uses branch and bound with bitmasks (n <= 64).
    """
    masks = [0] * n
    for i in range(n):
        m = 0
        for j in adj[i]:
            m |= (1 << j)
        masks[i] = m
        
    best = [0]
    
    # Pre-compute popcount for small integers is fast in Python 3.
    # We use bin(x).count('1') which is optimized in C.
    
    def dfs(candidates, current_size):
        if _time_up(deadline):
            return
        
        if not candidates:
            if current_size > best[0]:
                best[0] = current_size
            return
            
        # Upper bound: current_size + popcount(candidates)
        # If this bound <= best[0], prune
        if current_size + bin(candidates).count('1') <= best[0]:
            return
            
        # Choose a vertex to branch on
        # Heuristic: pick vertex with highest degree in the induced subgraph
        # to maximize pruning in the "exclude" branch (which removes fewer candidates? No, exclude removes v, include removes v+neighbors).
        # Actually, standard branch and bound for Independent Set:
        # Branch on v.
        # Include v: candidates -> candidates \ (N(v) U {v})
        # Exclude v: candidates -> candidates \ {v}
        # To prune effectively, we want to find a large set quickly (to raise best) and prune deep.
        # Picking high degree vertices for inclusion is good because it removes many candidates, leading to a smaller subtree for the "include" branch,
        # but the "exclude" branch is large.
        # A common heuristic is to pick a vertex with high degree in the remaining graph.
        
        best_v = -1
        max_deg = -1
        temp_cands = candidates
        while temp_cands:
            lsb = temp_cands & -temp_cands
            v = lsb.bit_length() - 1
            deg = bin(candidates & masks[v]).count('1')
            if deg > max_deg:
                max_deg = deg
                best_v = v
            temp_cands ^= lsb
            
        v = best_v
        if v == -1:
            return
            
        bit_v = 1 << v
        
        # Option 1: Include v
        new_cands_1 = candidates & ~masks[v] & ~bit_v
        dfs(new_cands_1, current_size + 1)
        
        # Option 2: Exclude v
        new_cands_2 = candidates & ~bit_v
        dfs(new_cands_2, current_size)
        
    all_mask = (1 << n) - 1
    dfs(all_mask, 0)
    
    return best[0]

def _seven_crown():
    """
    Returns the 7-crown graph (complement of C7) or a known small triangle-free graph.
    Actually, let's use a known triangle-free graph with small alpha.
    The 7-crown (or similar) isn't necessarily the best starting point.
    Let's use the C5 blowup or just start with a random valid small graph.
    A simple valid graph: C5 (n=5, alpha=2).
    Or the Clebsch graph (n=16, alpha=5, triangle-free).
    Or the Hoffman-Singleton graph (n=50, but n must be <= 41).
    
    Let's construct a graph based on the Clebsch graph or similar.
    The Clebsch graph is a 16-vertex triangle-free graph with alpha=5.
    We can use it as a seed for local search.
    
    Alternatively, we can just start with a random valid graph found by a quick search.
    """
    # Let's just return a C5 for now, or better, a random valid graph of size 10-20.
    # Actually, the local search will grow it.
    # Let's return a graph on 10 vertices that is triangle-free and has alpha < 10.
    # A complete bipartite graph K5,5 has no triangles and alpha=5.
    n = 10
    adj = [set() for _ in range(n)]
    for i in range(5):
        for j in range(5, 10):
            adj[i].add(j)
            adj[j].add(i)
    edges = []
    for i in range(n):
        for j in adj[i]:
            if i < j:
                edges.append([i, j])
    return n, edges

def construct():
    deadline = time.time() + 110.0 # Safety margin
    
    # Start with a known valid graph
    # K5,5 is valid (triangle-free, alpha=5)
    n, edges = _seven_crown()
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    best_n = n
    best_edges = edges.copy()
    
    # Local search: try to add vertices
    # We will attempt to grow the graph by adding vertices one by one.
    # For each new vertex, we try to connect it to existing vertices in a way that:
    # 1. No triangle is formed.
    # 2. The alpha doesn't increase (or stays < 10).
    
    # To add a vertex v, we need to choose its neighbors S in the existing graph.
    # S must be an independent set in the existing graph (otherwise triangle with v).
    # Also, adding v with neighbors S means:
    # - v is not in any independent set of size 10 if S is large enough?
    # - Actually, if v is not connected to an independent set I of size 9, then I U {v} is an independent set of size 10.
    # So, S must intersect every independent set of size 9 in the existing graph.
    # This is a hitting set problem.
    
    # A simpler heuristic:
    # Try to add a vertex connected to a random independent set S of the current graph.
    # Check if the new graph is valid.
    
    # We will do this for several steps.
    
    max_attempts = 5000
    attempts = 0
    
    while not _time_up(deadline) and attempts < max_attempts:
        attempts += 1
        
        # Try to add a vertex
        new_n = n + 1
        if new_n > 41:
            break
            
        # Choose a random independent set S in the current graph
        # To be efficient, we can pick a random vertex and expand to an independent set
        # Or just pick a random subset of vertices that is independent
        
        # Let's pick a random vertex v0 and find a maximal independent set containing it
        v0 = random.randint(0, n - 1)
        S = {v0}
        candidates = set(range(n)) - {v0}
        # Remove neighbors of v0
        candidates -= adj[v0]
        
        # Greedily add vertices to S
        while candidates:
            # Pick a random vertex from candidates
            v = random.choice(list(candidates))
            S.add(v)
            candidates -= adj[v]
            candidates -= {v}
            
        # S is an independent set
        # Create new vertex n with neighbors S
        new_adj = [set(x) for x in adj]
        new_adj.append(set(S))
        for v in S:
            new_adj[v].add(n)
            
        # Check if new graph is triangle-free
        # Since S is independent, no triangles involving n.
        # Existing triangles are none.
        # So it is triangle-free.
        
        # Check alpha
        # Quick heuristic rejection
        # Greedy independent set
        best_heur = 0
        for _ in range(5):
            vertices = list(range(new_n))
            random.shuffle(vertices)
            ind_set = set()
            for v in vertices:
                if not (ind_set & new_adj[v]):
                    ind_set.add(v)
            if len(ind_set) > best_heur:
                best_heur = len(ind_set)
        
        if best_heur >= 10:
            continue
            
        # Exact check
        alpha = _exact_alpha(new_adj, new_n, deadline)
        if alpha < 10:
            # Valid!
            n = new_n
            adj = new_adj
            edges = []
            for i in range(n):
                for j in adj[i]:
                    if i < j:
                        edges.append([i, j])
            if n > best_n:
                best_n = n
                best_edges = edges.copy()
                
            # Reset attempts if we made progress?
            # Or just continue.
            # If we reach 41, break.
            if n >= 41:
                break
        else:
            # Try to modify S? Or just restart with a different S?
            # For now, just try a new S.
            pass
            
    if best_n > 0:
        return {'n': best_n, 'edges': best_edges}
    else:
        # Fallback to C5 if nothing worked (shouldn't happen)
        return {'n': 5, 'edges': [[0,1],[1,2],[2,3],[3,4],[4,0]]}

if __name__ == "__main__":
    res = construct()
    print(res['n'])
