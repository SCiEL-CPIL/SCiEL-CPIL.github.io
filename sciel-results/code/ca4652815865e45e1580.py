import numpy as np
import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    # Target n
    n = 35
    all_mask = (1 << n) - 1
    
    # Fallback: Paley(17)
    def paley_17():
        squares = {x * x % 17 for x in range(1, 17)}
        return {'n': 17, 'edges': [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]}
    
    # Best result found so far
    best_result = paley_17()
    
    # Helper to convert bitmask N to edge list
    def to_edge_list(N):
        edges = []
        for u in range(n):
            mask = N[u]
            # Only consider v > u
            v = u + 1
            while v < n:
                if (mask >> v) & 1:
                    edges.append([u, v])
                v += 1
        return edges

    # Helper to check if a set of vertices is an independent set
    # v_list is a list of 6 vertices
    def is_independent(v_list):
        for i in range(len(v_list)):
            for j in range(i+1, len(v_list)):
                u, v = v_list[i], v_list[j]
                if (N[u] >> v) & 1:
                    return False
        return True

    # Find a K4
    def find_k4():
        # Returns (u, v, w, x) or None
        for u in range(n):
            for v in range(u+1, n):
                if (N[u] >> v) & 1:
                    # Common neighbors
                    common = N[u] & N[v]
                    # Remove u and v from common
                    common &= ~(1 << u)
                    common &= ~(1 << v)
                    
                    # Find an edge in common
                    # Iterate through bits in common
                    c = common
                    while c:
                        # Get lowest set bit
                        w = (c & -c).bit_length() - 1
                        c &= c - 1
                        # Check if w is connected to any other in common
                        # We need x in common, x > w (to avoid duplicates? No, just any x)
                        # But we need w-x edge.
                        # Let's check neighbors of w in common
                        common_x = N[w] & common
                        # Remove w
                        common_x &= ~(1 << w)
                        if common_x:
                            # Get any x
                            x = (common_x & -common_x).bit_length() - 1
                            return (u, v, w, x)
        return None

    # Find an I6
    # Use backtracking with pruning
    def find_i6():
        # We want to find 6 vertices with no edges between them.
        # This is equivalent to finding a clique of size 6 in the complement graph.
        # Or simply, independent set in G.
        
        # Recursive search
        # candidates: set of vertices that can be added to the current independent set
        # current: list of vertices in the current independent set
        def backtrack(candidates, current):
            if len(current) == 6:
                return current[:]
            if not candidates:
                return None
            
            # Pruning: if we can't reach size 6
            if len(current) + candidates.bit_count() < 6:
                return None
            
            # Pick a vertex from candidates
            # To make it deterministic or random? Random helps escape.
            # Let's pick the first one, or a random one.
            # For speed, picking the first one (lowest index) is fast.
            # But we might miss solutions.
            # Let's try to pick a vertex that has the fewest neighbors in candidates?
            # That's a good heuristic for independent sets.
            
            # For simplicity and speed, let's just iterate.
            c = candidates
            while c:
                v = (c & -c).bit_length() - 1
                c &= c - 1
                # If we include v, the new candidates are those in candidates that are NOT neighbors of v
                # and have index > v? No, index doesn't matter for independence, just non-adjacency.
                # But to avoid permutations, we can enforce order?
                # Actually, for finding *any* I6, order doesn't matter.
                # But to avoid duplicate checks, we can restrict to vertices with index > last picked?
                # No, that's for cliques. For independent sets, we just need non-neighbors.
                # However, if we don't enforce order, we might check the same set in different orders.
                # To be efficient, let's enforce that we only pick vertices with index > current[-1] if current is not empty?
                # No, that's not correct for independent sets.
                # Wait, if we pick v, the next vertices must be non-neighbors of v.
                # They can be any index.
                
                new_candidates = candidates & ~N[v] & ~(1 << v)
                # Also, we should probably remove v from candidates for the next step?
                # Yes, new_candidates already excludes v.
                
                res = backtrack(new_candidates, current + [v])
                if res:
                    return res
                    
            return None
        
        return backtrack(all_mask, [])

    # Main Search Loop
    # We will try multiple restarts
    for restart in range(50):
        if time.time() > deadline:
            break
            
        # Initialize random graph
        # Density around 0.5
        N = [0] * n
        for u in range(n):
            for v in range(u+1, n):
                if random.random() < 0.5:
                    N[u] |= (1 << v)
                    N[v] |= (1 << u)
        
        steps = 0
        max_steps = 20000
        
        while steps < max_steps and time.time() < deadline:
            steps += 1
            
            # Check K4
            k4 = find_k4()
            if k4:
                u, v, w, x = k4
                # Remove a random edge from the K4
                edges_in_k4 = [(u,v), (u,w), (u,x), (v,w), (v,x), (w,x)]
                e_u, e_v = random.choice(edges_in_k4)
                N[e_u] &= ~(1 << e_v)
                N[e_v] &= ~(1 << e_u)
                continue
            
            # Check I6
            i6 = find_i6()
            if i6:
                # Add a random edge inside the I6
                # i6 is a list of 6 vertices
                u, v = random.sample(i6, 2)
                N[u] |= (1 << v)
                N[v] |= (1 << u)
                continue
                
            # If neither K4 nor I6 found, we have a valid graph!
            edges = to_edge_list(N)
            return {'n': n, 'edges': edges}
            
            # If we are stuck, do a random flip to escape
            if steps % 100 == 0:
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u != v:
                    if (N[u] >> v) & 1:
                        N[u] &= ~(1 << v)
                        N[v] &= ~(1 << u)
                    else:
                        N[u] |= (1 << v)
                        N[v] |= (1 << u)

    return best_result

# EVOLVE-BLOCK-END
