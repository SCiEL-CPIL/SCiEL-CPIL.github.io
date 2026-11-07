"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

def construct():
    from random import Random

    def instance(splits, seed, palette=5):
        rng = Random(seed)
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
        
        # Initialize domains as sets for fast removal
        domains = [set(l) for l in lists]
        assigned = [-1] * n
        
        start_time = time.time()
        deadline = start_time + 110.0  # Safety margin before 120s limit
        
        # MRV + Forward Checking Backtracking
        # We use a list to track unassigned vertices for faster MRV selection if needed,
        # but for n~400, scanning is O(n) which is fine.
        
        def backtrack(idx):
            # Check time
            if time.time() > deadline:
                return False
            
            # Find MRV vertex
            min_size = float('inf')
            min_vertex = -1
            for v in range(n):
                if assigned[v] == -1:
                    size = len(domains[v])
                    if size < min_size:
                        min_size = size
                        min_vertex = v
                        if size == 0:
                            return False
                        if size == 1:
                            break
            
            if min_vertex == -1:
                return True  # All assigned
            
            v = min_vertex
            # Save state for backtracking
            # We need to save the domain of v and the domains of its neighbors
            # Actually, it's easier to just copy the domains or use a trail.
            # For simplicity and given small n, copying domains of affected nodes is okay,
            # but a trail is more efficient. Let's use a simple approach:
            # Save the old domain of v.
            # Save the old domains of neighbors.
            
            old_domain_v = domains[v]
            old_domains_neighbors = {u: domains[u] for u in adj[v] if assigned[u] == -1}
            
            # Try each color in domain
            for color in list(old_domain_v):  # Iterate over a copy
                # Assign color to v
                assigned[v] = color
                domains[v] = set()  # Mark as assigned/empty domain
                
                # Forward Checking: remove color from neighbors
                consistent = True
                for u in adj[v]:
                    if assigned[u] == -1:
                        if color in domains[u]:
                            domains[u].remove(color)
                            if len(domains[u]) == 0:
                                consistent = False
                                break
                
                if consistent:
                    if backtrack(idx + 1):
                        return True
                
                # Undo
                assigned[v] = -1
                domains[v] = old_domain_v
                for u, old_dom in old_domains_neighbors.items():
                    domains[u] = old_dom
            
            return False

        # Initial check
        if not backtrack(0):
            # If failed due to time or no solution, return best effort or empty
            # Since Alon-Tarsi guarantees solvability, failure is likely timeout or bug.
            # Return a list of -1s? No, must be int 0..4.
            # If we time out, we might have partial assignment.
            # But the contract says "wrong length or type makes the whole result invalid".
            # It also says "An instance whose colouring breaks a list or an edge scores 0".
            # So returning a valid-looking but wrong coloring is better than crashing.
            # However, if we return [0]*n, it might be valid or invalid.
            # Let's just return the current `assigned` state, filling unassigned with 0.
            # But if we timed out, `assigned` might be partial.
            # To be safe, if we don't have a full solution, we return a dummy valid type.
            # But wait, if we return a partial coloring where some are -1, that's invalid type/value.
            # So we must ensure we return a list of ints 0..4 of length n.
            
            # If backtrack failed, we don't have a solution.
            # We will return a list of zeros. This will likely score 0 for that instance,
            # but it keeps the program alive and type-correct.
            return [0] * n
        
        # Convert to Python ints
        return [int(c) for c in assigned]

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        colouring = solve_instance(n, edges, lists)
        out.append(colouring)
    
    # Final type check: ensure lists of ints
    result = []
    for c in out:
        result.append([int(x) for x in c])
    return result
# EVOLVE-BLOCK-END
