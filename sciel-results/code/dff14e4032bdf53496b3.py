"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

def construct():
    from random import Random
    
    def instance(splits, seed, palette=5):
        rng = Random(seed)
        # Cube faces: 0-1-2-3 (top), 4-5-6-7 (bottom), sides
        # Standard cube labeling:
        # 0 1
        # 3 2
        # 4 5
        # 7 6
        # Let's verify the seed's faces definition.
        # faces = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
        # This implies edges:
        # 0-1, 1-2, 2-3, 3-0 (Top)
        # 4-5, 5-6, 6-7, 7-4 (Bottom)
        # 0-4, 1-5, 2-6, 3-7 (Verticals)
        # This is a standard cube.
        
        faces = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
        edges = {(min(a, b), max(a, b)) for f in faces for a, b in zip(f, f[1:] + f[:1])}
        n = 8
        for _ in range(splits):
            face = faces.pop(rng.randrange(len(faces)))
            inner = list(range(n, n + 4))
            n += 4
            for i in range(4):
                a, b = face[i], face[(i + 1) % 4]
                edges.add((min(a, inner[i]), max(a, inner[i])))
                edges.add((min(inner[i], inner[(i + 1) % 4]), max(inner[i], inner[(i + 1) % 4])))
                faces.append([a, b, inner[(i + 1) % 4], inner[i]])
            faces.append(inner)
        lists = [sorted(rng.sample(range(palette), 3)) for _ in range(n)]
        return n, sorted(edges), lists

    def solve_instance(n, edges, lists):
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Precompute degrees for tie-breaking in MRV
        degrees = [len(adj[i]) for i in range(n)]
        
        colour = [-1] * n
        unassigned = list(range(n))
        
        # Increase recursion limit just in case, though n is small
        sys.setrecursionlimit(10000)
        
        def backtrack():
            if not unassigned:
                return True
            
            # MRV Heuristic: Find unassigned vertex with fewest available colours
            # Tie-break: Highest degree
            best_v = -1
            min_avail = 100
            best_degree = -1
            
            for v in unassigned:
                # Calculate available colours
                taken = set()
                for u in adj[v]:
                    if colour[u] != -1:
                        taken.add(colour[u])
                
                avail = [c for c in lists[v] if c not in taken]
                
                if len(avail) < min_avail:
                    min_avail = len(avail)
                    best_v = v
                    best_degree = degrees[v]
                    # Early exit if domain is 1? No, we still need to check if it's valid.
                    # But if min_avail is 1, it's the best possible. 
                    # We can break if min_avail == 1? No, another vertex might have 0 (failure) or 1 (same).
                    # Actually, if we find a vertex with 0 available, we should return False immediately.
                    if min_avail == 0:
                        return False
                    # If we find a vertex with 1, it's the minimum possible non-zero. 
                    # We can't break immediately because another vertex might also have 1, 
                    # but we want the one with higher degree. 
                    # However, if we find 1, no vertex can have < 1 (and not 0). 
                    # So we can skip the "len < min" check for subsequent vertices if min==1?
                    # No, we still need to check for 0s. 
                    # So we must scan all unassigned to ensure no 0s? 
                    # Yes, forward checking effectively does this, but here we compute on the fly.
                    # Optimization: If min_avail == 1, we still need to check others for 0.
                    # So we can't break.
            
            if best_v == -1:
                return False # Should not happen if unassigned is not empty

            # Check if best_v actually has 0 available (handled above, but double check)
            taken = set()
            for u in adj[best_v]:
                if colour[u] != -1:
                    taken.add(colour[u])
            avail = [c for c in lists[best_v] if c not in taken]
            
            if not avail:
                return False

            # Try each colour in avail
            # Value ordering: Least Constraining Value (LCV) is complex, try sorted order
            for c in avail:
                colour[best_v] = c
                unassigned.remove(best_v)
                
                if backtrack():
                    return True
                
                # Backtrack
                unassigned.append(best_v)
                colour[best_v] = -1
                
            return False

        if backtrack():
            return colour
        else:
            return None

    out = []
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin

    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > deadline:
            # Fallback: return empty or partial? 
            # The task says "Any broken rule makes the whole result invalid".
            # So we must return a valid list of 4 lists.
            # If we run out of time, we can't guarantee validity.
            # However, these instances are small. The solver is extremely fast.
            # We assume we finish in time.
            pass

        n, edges, lists = instance(splits, seed)
        solution = solve_instance(n, edges, lists)
        
        if solution is None:
            # Should not happen, but handle gracefully
            # Return a dummy valid list? No, it must be valid.
            # Since Alon-Tarsi guarantees solvability, this is a safeguard against bugs.
            solution = [0] * n
            
        # Ensure types are Python int
        out.append([int(c) for c in solution])

    return out
# EVOLVE-BLOCK-END
