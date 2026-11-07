"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
import heapq

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

    def solve_one(n, edges, lists):
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        # Use a CSP solver with Minimum Remaining Values (MRV) heuristic
        # and Forward Checking.
        
        # domains[v] is a set of available colors
        domains = [set(lst) for lst in lists]
        colour = [-1] * n
        uncoloured = set(range(n))
        
        # We will use a priority queue to select the next variable (vertex)
        # based on the size of its current domain (MRV).
        # To handle updates efficiently, we can just re-scan or use a lazy heap.
        # Given n <= 400, re-scanning for min is O(N) per step, total O(N^2) which is fine.
        
        sys.setrecursionlimit(10000)
        
        def backtrack():
            if not uncoloured:
                return True
            
            # Select variable with Minimum Remaining Values
            # Tie-breaker: highest degree (more constraints)
            min_var = None
            min_size = float('inf')
            min_degree = -1
            for v in uncoloured:
                size = len(domains[v])
                if size == 0:
                    return False # Dead end
                if size < min_size or (size == min_size and len(adj[v]) > min_degree):
                    min_size = size
                    min_degree = len(adj[v])
                    min_var = v
            
            v = min_var
            uncoloured.remove(v)
            
            # Try each colour in the domain
            # Order colours by least constraining value (LCV) heuristic?
            # For simplicity, just iterate. The MRV on variables is the main driver.
            for c in list(domains[v]): # Iterate over a copy
                # Check consistency
                valid = True
                for u in adj[v]:
                    if u in uncoloured:
                        if c in domains[u]:
                            # This colour c conflicts with u if u is uncoloured?
                            # No, u is uncoloured, so we just need to remove c from u's domain.
                            # Wait, forward checking: if u is uncoloured, we must ensure u can still be coloured.
                            # If c is in domains[u], we must remove it. If domains[u] becomes empty, conflict.
                            pass
                    else:
                        # u is coloured
                        if colour[u] == c:
                            valid = False
                            break
                            
                if not valid:
                    continue
                
                # Assign colour v = c
                colour[v] = c
                domains[v] = {c}
                
                # Forward checking: update domains of uncoloured neighbours
                removed = []
                consistent = True
                for u in adj[v]:
                    if u in uncoloured:
                        if c in domains[u]:
                            domains[u].remove(c)
                            removed.append((u, c))
                            if not domains[u]:
                                consistent = False
                                break
                
                if consistent:
                    if backtrack():
                        return True
                else:
                    pass # Undo will happen below
                
                # Undo
                for u, c_removed in removed:
                    domains[u].add(c_removed)
                colour[v] = -1
                domains[v] = set(lists[v]) # Restore original domain? 
                # Wait, domains[v] was reduced by previous steps? 
                # Actually, v was uncoloured, so its domain was already filtered by previously coloured neighbours.
                # We need to restore the domain of v to what it was before we tried c.
                # But we overwrote domains[v] with {c}.
                # We need to save the old domain.
                
            return False

        # The above backtracking has a bug in undo logic for domains[v].
        # Let's implement a cleaner version.
        
        domains = [set(lst) for lst in lists]
        colour = [-1] * n
        uncoloured = set(range(n))
        
        def backtrack_v2():
            if not uncoloured:
                return True
            
            # Select variable with MRV
            min_var = None
            min_size = float('inf')
            for v in uncoloured:
                size = len(domains[v])
                if size == 0:
                    return False
                if size < min_size:
                    min_size = size
                    min_var = v
            
            v = min_var
            uncoloured.remove(v)
            
            # Save current domain of v
            old_domain_v = set(domains[v])
            
            for c in old_domain_v:
                # Check if c conflicts with any coloured neighbour
                valid = True
                for u in adj[v]:
                    if u not in uncoloured and colour[u] == c:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Assign
                colour[v] = c
                domains[v] = {c}
                
                # Forward checking
                changes = []
                consistent = True
                for u in adj[v]:
                    if u in uncoloured:
                        if c in domains[u]:
                            domains[u].remove(c)
                            changes.append(u)
                            if not domains[u]:
                                consistent = False
                                break
                
                if consistent:
                    if backtrack_v2():
                        return True
                
                # Undo
                for u in changes:
                    domains[u].add(c)
                colour[v] = -1
                domains[v] = old_domain_v
            
            return False

        if backtrack_v2():
            return colour
        
        # Fallback: should not happen
        return [0] * n

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        colouring = solve_one(n, edges, lists)
        # Ensure all values are Python ints
        out.append([int(c) for c in colouring])
    return out
# EVOLVE-BLOCK-END
