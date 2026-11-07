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
        """Exact backtracking with MRV and forward checking."""
        # Build adjacency
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Convert to lists for faster iteration
        adj_list = [sorted(s) for s in adj]
        
        # Initial domains
        domains = [set(l) for l in lists]
        
        # Color assignment: -1 means uncolored
        color = [-1] * n
        
        # Count remaining uncolored
        uncolored = set(range(n))
        
        # MRV: pick variable with smallest domain
        # Forward checking: when assigning, reduce domains of neighbors
        
        def backtrack():
            if not uncolored:
                return True
            
            # MRV: find variable with smallest domain
            min_var = None
            min_size = float('inf')
            for v in uncolored:
                d = len(domains[v])
                if d == 0:
                    return False
                if d < min_size:
                    min_size = d
                    min_var = v
                    if d == 1:
                        break
            
            # Try colors in the domain of min_var
            # Order: least constraining value (try colors that appear in fewer neighbors' domains)
            colors = list(domains[min_var])
            # Sort by least constraining
            def lcv(c):
                count = 0
                for nb in adj_list[min_var]:
                    if nb in uncolored and c in domains[nb]:
                        count += 1
                return count
            colors.sort(key=lcv)
            
            for c in colors:
                # Assign color c to min_var
                color[min_var] = c
                uncolored.discard(min_var)
                
                # Save old domains for undo
                saved = {}
                valid = True
                for nb in adj_list[min_var]:
                    if nb in uncolored:
                        if c in domains[nb]:
                            saved[nb] = domains[nb].copy()
                            domains[nb].discard(c)
                            if not domains[nb]:
                                valid = False
                                break
                
                if valid and backtrack():
                    return True
                
                # Undo
                for nb, old_domain in saved.items():
                    domains[nb] = old_domain
                uncolored.add(min_var)
                color[min_var] = -1
            
            return False
        
        if backtrack():
            return color
        return None

    def solve_greedy(n, edges, lists):
        """Greedy coloring as fallback."""
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        colour = [None] * n
        for v in sorted(range(n), key=lambda u: -len(adj[u])):
            taken = {colour[u] for u in adj[v] if colour[u] is not None}
            free = [c for c in lists[v] if c not in taken]
            colour[v] = free[0] if free else lists[v][0]
        return colour

    deadline = time.time() + 120.0
    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        
        # Try exact backtracking first
        result = solve_exact(n, edges, lists)
        
        if result is None or time.time() > deadline - 10.0:
            # Fallback to greedy
            result = solve_greedy(n, edges, lists)
        
        # Ensure all values are Python int
        out.append([int(c) for c in result])
    
    return out
# EVOLVE-BLOCK-END
