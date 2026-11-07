"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

def construct():
    from random import Random
    import sys

    DEADLINE = 140.0
    START = time.time()

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

    def solve_instance(n, edges, lists, timeout=30.0):
        # Build adjacency
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Precompute color bitmasks for each vertex's list
        # Colors are 0..4, so bitmask has bits 0-4
        color_mask = [0] * n
        for v in range(n):
            for c in lists[v]:
                color_mask[v] |= (1 << c)
        
        # Order vertices by degree descending (most constrained first)
        order = sorted(range(n), key=lambda u: -len(adj[u]))
        order_index = {v: i for i, v in enumerate(order)}
        
        # AC-3 constraint propagation with bitmasks
        def ac3(domains):
            """
            domains: list of int bitmasks, one per vertex
            Returns True if no domain becomes empty, False otherwise.
            Modifies domains in place.
            """
            # Queue of (i, j) pairs where i's domain might need revision due to j
            # We process arcs (i, j) meaning "check if i's domain is consistent with j's domain"
            q = deque()
            for v in range(n):
                for u in adj[v]:
                    q.append((v, u))
            
            in_queue = [[False] * n for _ in range(n)]
            for v in range(n):
                for u in adj[v]:
                    in_queue[v][u] = True
            
            while q:
                if time.time() - START > timeout:
                    return True  # Assume OK on timeout to avoid infinite loop
                
                i, j = q.popleft()
                in_queue[i][j] = False
                
                # Check if every color in domain[i] has a compatible color in domain[j]
                # Two colors are compatible if they're different
                # domain[i] & ~(1<<c) should intersect with domain[j] for each c in domain[i]
                # Equivalent: for each c in domain[i], (domain[j] & ~(1<<c)) != 0
                # Or: domain[i] must be subset of {c : domain[j] has some color != c}
                # Simpler: domain[i] & domain[j] == 0 means they can only use same color -> conflict
                # Actually: if domain[i] == domain[j] == (1<<c), then conflict.
                # General: remove c from domain[i] if domain[j] == (1<<c)
                
                di = domains[i]
                dj = domains[j]
                
                if di == 0:
                    return False
                
                # Find colors in di that have no support in dj
                # c has no support in dj if dj == (1 << c)
                # So: unsupported = di & dj where each bit c in (di & dj) means dj only has c
                # Actually: c is unsupported if (dj & ~(1<<c)) == 0, i.e., dj == (1<<c)
                # So unsupported colors = {c in di : dj == (1<<c)}
                # = di & (OR of (1<<c) for c where dj == (1<<c))
                
                # Efficient: if dj is a single bit (1<<c), then remove c from di
                # More generally, we need to check each color in di
                # But with only 3 colors per list, this is fast
                
                new_di = 0
                for c in range(5):
                    if di & (1 << c):
                        # Check if c has support in dj: is there some c' in dj with c' != c?
                        if dj & ~(1 << c):
                            new_di |= (1 << c)
                
                if new_di != di:
                    domains[i] = new_di
                    if new_di == 0:
                        return False
                    # Add all arcs (k, i) where k != j
                    for k in adj[i]:
                        if k != j and not in_queue[k][i]:
                            q.append((k, i))
                            in_queue[k][i] = True
            
            return True
        
        # Backtracking with AC-3
        sys.setrecursionlimit(10000)
        
        def backtrack(pos, domains, colour):
            if time.time() - START > timeout:
                return False
            
            if pos == n:
                return True
            
            v = order[pos]
            
            # Try colors in the current domain for v
            dv = domains[v]
            for c in range(5):
                if dv & (1 << c):
                    # Assign color c to v
                    colour[v] = c
                    new_domains = domains[:]
                    new_domains[v] = (1 << c)
                    
                    # Propagate constraints
                    if ac3(new_domains):
                        if backtrack(pos + 1, new_domains, colour):
                            return True
                    
                    colour[v] = None
            
            return False
        
        # Initial domains from lists
        initial_domains = color_mask[:]
        
        # Try AC-3 first on the full problem
        test_domains = initial_domains[:]
        if not ac3(test_domains):
            # No solution exists (shouldn't happen per problem statement)
            # Return a best-effort assignment
            colour = [0] * n
            for v in range(n):
                if lists[v]:
                    colour[v] = lists[v][0]
            return colour
        
        # Run backtracking
        colour = [None] * n
        if backtrack(0, test_domains, colour):
            # Ensure all are Python ints
            return [int(c) for c in colour if c is not None]
        
        # Fallback: greedy with restarts
        rng = Random(42)
        for attempt in range(200):
            if time.time() - START > timeout:
                break
            colour = [None] * n
            success = True
            for v in order:
                taken_mask = 0
                for u in adj[v]:
                    if colour[u] is not None:
                        taken_mask |= (1 << colour[u])
                free = [c for c in lists[v] if not (taken_mask & (1 << c))]
                if free:
                    colour[v] = rng.choice(free)
                else:
                    success = False
                    break
            
            if success:
                # Verify
                valid = True
                for v in range(n):
                    if colour[v] not in lists[v]:
                        valid = False
                        break
                    for u in adj[v]:
                        if u > v and colour[u] == colour[v]:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    return [int(c) for c in colour]
        
        # Last resort: return lists[v][0] for all
        return [int(lists[v][0]) for v in range(n)]

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        colouring = solve_instance(n, edges, lists)
        
        # Ensure types are correct: Python int, list of lists
        final_colouring = [int(c) for c in colouring]
        out.append(final_colouring)
    
    # Ensure exactly 4 outputs
    while len(out) < 4:
        out.append([])
    
    return out

# EVOLVE-BLOCK-END
