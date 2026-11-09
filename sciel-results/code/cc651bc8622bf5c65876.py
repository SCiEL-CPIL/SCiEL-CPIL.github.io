import random
import time
import sys

def construct():
    start = time.time()
    deadline = 145  # Safety margin before 150s kill

    def can_add_edge_fast(n, adj, u, v):
        """
        Check if adding (u,v) creates a C4.
        Creates C4 iff ∃ path of length 3 from u to v: u-a-b-v
        i.e., ∃ a∈N(u), b∈N(v), a≠b, a~b
        Since b∈N(a) implies b≠a, just check: ∃ a∈N(u) s.t. N(a)∩N(v)≠∅
        """
        for a in adj[u]:
            if adj[a] & adj[v]:
                return False
        return True

    def is_c4_free(n, adj):
        for u in range(n):
            for v in adj[u]:
                if v > u:
                    if len(adj[u] & adj[v]) > 1:
                        return False
        return True

    def min_degree(n, adj):
        return min(len(adj[i]) for i in range(n))

    def to_edges(adj, n):
        edges = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edges.append([u, v])
        return edges

    best = {'n': 39, 'edges': []}

    # Try descending n from 47 to 40
    for n in range(47, 39, -1):
        if time.time() > deadline:
            break
        
        found = False
        attempts = 0
        max_attempts = 1000  # More attempts with faster check

        while attempts < max_attempts and time.time() < deadline and not found:
            attempts += 1
            
            adj = [set() for _ in range(n)]
            
            # Generate random order of edges
            edges_to_try = [(u, v) for u in range(n) for v in range(u + 1, n)]
            random.shuffle(edges_to_try)
            
            # Greedy addition
            for u, v in edges_to_try:
                if time.time() > deadline:
                    break
                if can_add_edge_fast(n, adj, u, v):
                    adj[u].add(v)
                    adj[v].add(u)
            
            if time.time() > deadline:
                break

            # Check if min degree is sufficient
            target_deg = n - 39
            if min_degree(n, adj) >= target_deg:
                # Verify C4-free just in case
                if is_c4_free(n, adj):
                    best = {'n': n, 'edges': to_edges(adj, n)}
                    found = True
                    break
            
            # If min degree is too low, try local repair
            # This is a fallback if greedy didn't reach the degree
            # We can try to add edges to low-degree vertices specifically
            if min_degree(n, adj) < target_deg:
                improved = True
                repair_rounds = 0
                while improved and time.time() < deadline and repair_rounds < 100:
                    improved = False
                    repair_rounds += 1
                    # Sort vertices by degree, lowest first
                    vertices = sorted(range(n), key=lambda x: len(adj[x]))
                    for u in vertices:
                        if len(adj[u]) >= target_deg:
                            continue
                        # Try to add edges to u
                        candidates = [v for v in range(n) if v != u and v not in adj[u]]
                        random.shuffle(candidates)
                        for v in candidates:
                            if time.time() > deadline:
                                break
                            if can_add_edge_fast(n, adj, u, v):
                                adj[u].add(v)
                                adj[v].add(u)
                                improved = True
                                if len(adj[u]) >= target_deg:
                                    break
                        if time.time() > deadline:
                            break
                    
                    if min_degree(n, adj) >= target_deg and is_c4_free(n, adj):
                        best = {'n': n, 'edges': to_edges(adj, n)}
                        found = True
                        break

        if found:
            break

    return best

# EVOLVE-BLOCK-END
