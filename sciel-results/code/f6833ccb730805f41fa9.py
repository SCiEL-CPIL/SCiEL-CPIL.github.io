import time
import random
import sys

def construct():
    """
    Construct a C5-free graph on n vertices with independence number < 9.
    Returns the best valid graph found, maximizing n.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s
    
    # Helper: check if adding edge (u,v) to graph G creates a C5
    def creates_c5(G, u, v, n):
        """Check if adding edge (u,v) creates a 5-cycle.
        A 5-cycle using edge (u,v) is a path of length 4 from u to v.
        """
        # Find paths of length 4 from u to v
        # Path: u - a - b - c - v
        # We need to find a, b, c such that (u,a), (a,b), (b,c), (c,v) are all edges in G
        
        # For efficiency, iterate over possible middle vertex b
        # b must be adjacent to some a (adjacent to u) and some c (adjacent to v)
        
        neighbors_u = set(G[u])
        neighbors_v = set(G[v])
        
        # a must be in neighbors_u, c must be in neighbors_v
        # b must be adjacent to a and to c
        
        # For each b in 0..n-1:
        #   if b == u or b == v: skip
        #   For each a in neighbors_u where (a,b) in G:
        #     For each c in neighbors_v where (b,c) in G:
        #       if a != c and a != v and c != u: found C5
        
        # More efficient: 
        # For each b:
        #   common_a = neighbors_u ∩ G[b]  (a's adjacent to both u and b)
        #   common_c = neighbors_v ∩ G[b]  (c's adjacent to both v and b)
        #   if common_a and common_c:
        #     for a in common_a:
        #       for c in common_c:
        #         if a != c: found
        #         # also need a != v, c != u, but a is neighbor of u so a != u; 
        #         # a could equal v? No, v is not adjacent to u yet (we're adding it)
        #         # Actually we need to be careful. Let me just check the full condition.
        
        for b in range(n):
            if b == u or b == v:
                continue
            # a adjacent to u and to b
            # c adjacent to v and to b
            found_a = []
            for a in neighbors_u:
                if a == b:
                    continue
                if (a, b) in G or (b, a) in G:
                    found_a.append(a)
            if not found_a:
                continue
            for c in neighbors_v:
                if c == b:
                    continue
                if (b, c) in G or (c, b) in G:
                    # Check: a != c, and the path u-a-b-c-v is a valid simple path
                    # a is neighbor of u, b neighbor of a, c neighbor of b, v neighbor of c
                    # Need: a != c, a != v, c != u, b != u, b != v (b != u,v already checked)
                    # a != u: a is in neighbors_u, so a != u (simple graph)
                    # c != v: c is in neighbors_v, so c != v
                    # a != v: a is adjacent to u, v is not adjacent to u (we're adding that edge)
                    #   so a could potentially be v only if (u,v) was already an edge, but it's not
                    # c != u: similar
                    for a in found_a:
                        if a != c:
                            return True
        return False
    
    # Helper: compute independence number
    def independence_number(G, n):
        """Compute the size of the maximum independent set."""
        # For n up to 35, we can use a simple backtracking or use the fact that
        # alpha(G) = n - chi(complement). For n=35, a simple algorithm works.
        # Use a simple greedy + backtracking for small n.
        
        # Actually, for our purposes, we just need to check if alpha >= 9.
        # Let's use a simple recursive approach with memoization on subsets... 
        # n=35 is too big for 2^n.
        
        # Use a branching algorithm:
        def max_independent_set(vertices):
            if not vertices:
                return 0
            v = next(iter(vertices))
            # Option 1: include v
            #   Then exclude all neighbors of v
            v_set = frozenset([v])
            neighbors = G.get(v, set())
            remaining_exclude = vertices - v_set - neighbors
            opt1 = 1 + max_independent_set(remaining_exclude)
            # Option 2: exclude v
            opt2 = max_independent_set(vertices - v_set)
            return max(opt1, opt2)
        
        return max_independent_set(frozenset(range(n)))
    
    # Helper: check if graph has no C5
    def has_c5(G, n):
        """Check if graph contains a 5-cycle as a subgraph."""
        # A 5-cycle is a cycle of exactly 5 vertices with 5 edges forming the cycle.
        # We need to check all 5-vertex subsets... too slow.
        # Better: for each edge (u,v), check if there's a path of length 4 from u to v.
        # That's what creates_c5 does, but we need to check existing edges.
        
        for u in range(n):
            for v in range(u+1, n):
                if (u, v) in G or (v, u) in G:
                    # Check if removing this edge, there's a path of length 4 from u to v
                    # Actually, simpler: just check if there exists a 5-cycle.
                    # For each pair (u,v) that is an edge, count paths of length 4 between them.
                    # If any such path exists where all intermediate vertices are distinct from u,v
                    # and from each other, we have a C5.
                    neighbors_u = set(G.get(u, set()))
                    neighbors_v = set(G.get(v, set()))
                    
                    # Path u-a-b-c-v
                    for b in range(n):
                        if b == u or b == v:
                            continue
                        found_a = []
                        for a in neighbors_u:
                            if a == b or a == v:
                                continue
                            if (a, b) in G or (b, a) in G:
                                found_a.append(a)
                        if not found_a:
                            continue
                        for c in neighbors_v:
                            if c == b or c == u:
                                continue
                            if (b, c) in G or (c, b) in G:
                                for a in found_a:
                                    if a != c:
                                        return True
        return False
    
    # Helper: check if graph is valid
    def is_valid(G, n):
        if has_c5(G, n):
            return False
        if independence_number(G, n) >= 9:
            return False
        return True
    
    # Try n from 35 down to 28
    best_result = None
    
    for n in range(35, 27, -1):
        if time.time() > deadline:
            break
        
        # Try multiple random constructions
        max_attempts = 50 if n >= 33 else 20
        found = False
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            random.seed(42 + attempt * 1000 + n)
            
            # Start with empty graph
            G = {i: set() for i in range(n)}
            
            # Add edges randomly, avoiding C5 creation
            edges_to_try = [(u, v) for u in range(n) for v in range(u+1, n)]
            random.shuffle(edges_to_try)
            
            for (u, v) in edges_to_try:
                if time.time() > deadline:
                    break
                if not creates_c5(G, u, v, n):
                    G[u].add(v)
                    G[v].add(u)
            
            # Check independence number
            alpha = independence_number(G, n)
            if alpha < 9:
                # Valid!
                edges = []
                for u in range(n):
                    for v in sorted(G[u]):
                        if u < v:
                            edges.append([u, v])
                if best_result is None or n > best_result['n']:
                    best_result = {'n': n, 'edges': edges}
                    found = True
                    break
            elif alpha < 9:
                pass  # already handled above
        
        if found:
            break
    
    if best_result is not None:
        return best_result
    
    # Fallback: 8 disjoint K4s for n=32
    edges = []
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])
    return {'n': 32, 'edges': edges}
