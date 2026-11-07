"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

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

    def solve_exact(n, edges, lists):
        """
        Exact backtracking with MRV, Degree Heuristic, Forward Checking, and LCV.
        Optimized for speed and correctness on small bipartite graphs.
        """
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Sort neighbors by degree (descending) to help tie-breaking in MRV
        degrees = [len(adj[i]) for i in range(n)]
        for i in range(n):
            adj[i].sort(key=lambda x: -degrees[x])

        # Domains are lists of available colors
        domains = [list(l) for l in lists]
        color = [-1] * n
        
        # Unassigned vertices
        unassigned = set(range(n))

        def backtrack():
            if not unassigned:
                return True
            
            # 1. MRV: Pick variable with smallest domain
            min_var = -1
            min_size = float('inf')
            min_deg = -1
            
            for v in unassigned:
                d_size = len(domains[v])
                if d_size == 0:
                    return False
                if d_size < min_size or (d_size == min_size and degrees[v] > min_deg):
                    min_size = d_size
                    min_deg = degrees[v]
                    min_var = v
                    if d_size == 1:
                        break
            
            # 2. LCV: Order colors by least constraining value
            # A color is "least constraining" if it appears in the domains of fewest unassigned neighbors
            colors = domains[min_var]
            
            # If only one color, no need to sort
            if len(colors) > 1:
                def lcv_score(c):
                    count = 0
                    for nb in adj[min_var]:
                        if nb in unassigned and c in domains[nb]:
                            count += 1
                    return count
                colors.sort(key=lcv_score)

            # 3. Try each color
            for c in colors:
                color[min_var] = c
                unassigned.discard(min_var)
                
                # Save state of neighbors' domains for undo
                saved_domains = []
                valid = True
                
                for nb in adj[min_var]:
                    if nb in unassigned:
                        if c in domains[nb]:
                            saved_domains.append((nb, domains[nb][:])) # Copy list
                            # Remove c from neighbor's domain
                            idx = domains[nb].index(c)
                            domains[nb].pop(idx)
                            if not domains[nb]:
                                valid = False
                                break
                        else:
                            # c was not in domain, so no change needed, but we need to track 
                            # that we didn't change it? No, we only undo changes.
                            pass
                            # Actually, we only save if we changed it. 
                            # If c not in domain, we didn't change it, so no undo needed.
                            
                if valid and backtrack():
                    return True
                
                # Undo
                for nb, old_domain in saved_domains:
                    domains[nb] = old_domain
                unassigned.add(min_var)
                color[min_var] = -1
            
            return False

        if backtrack():
            return color
        return None

    def solve_greedy(n, edges, lists):
        """
        Greedy fallback with DSATUR-like ordering.
        """
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        colour = [None] * n
        # Order vertices by degree descending
        order = sorted(range(n), key=lambda u: -len(adj[u]))
        
        for v in order:
            taken = {colour[u] for u in adj[v] if colour[u] is not None}
            free = [c for c in lists[v] if c not in taken]
            if free:
                colour[v] = free[0]
            else:
                # Conflict: pick any from list (will be invalid, but better than crash)
                colour[v] = lists[v][0]
        return colour

    deadline = time.time() + 110.0 # Leave 10s margin
    out = []
    
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        
        # Try exact solver
        result = solve_exact(n, edges, lists)
        
        # If exact solver fails (shouldn't happen) or times out (unlikely for these sizes), use greedy
        if result is None:
            result = solve_greedy(n, edges, lists)
            
        # Ensure types are Python int
        out.append([int(c) for c in result])
        
        # Check time
        if time.time() > deadline:
            # If we are running out of time, we just return what we have.
            # Since we process sequentially, if one takes too long, we might not finish others.
            # But these are small, so this is a safety net.
            break

    # If we broke early, we need to fill the rest with greedy to ensure 4 results
    while len(out) < 4:
        splits, seed = ((20, 701), (40, 702), (60, 703), (100, 704))[len(out)]
        n, edges, lists = instance(splits, seed)
        result = solve_greedy(n, edges, lists)
        out.append([int(c) for c in result])

    return out
# EVOLVE-BLOCK-END
