"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

def construct():
    start_time = time.time()
    DEADLINE = 110.0  # safety margin before 120s limit

    def instance(splits, seed, palette=5):
        from random import Random
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

    def build_adj(n, edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def is_valid_colouring(colour, adj, lists):
        for v in range(len(colour)):
            c = colour[v]
            if c not in lists[v]:
                return False
            for u in adj[v]:
                if colour[u] == c:
                    return False
        return True

    def greedy_color(n, adj, lists, order):
        colour = [None] * n
        for v in order:
            taken = set()
            for u in adj[v]:
                if colour[u] is not None:
                    taken.add(colour[u])
            free = [c for c in lists[v] if c not in taken]
            if free:
                colour[v] = free[0]
            else:
                # Fallback: pick any colour from list (may cause conflict, but better than None)
                colour[v] = lists[v][0]
        return colour

    def backtrack_color(n, adj, lists, order, max_nodes=500000):
        """
        Backtracking with forward checking. Returns a valid coloring or None.
        Uses the given order (list of vertices in the order to color).
        """
        colour = [None] * n
        # Precompute available colors for each vertex
        # Use a list of lists for the domains
        domains = [list(lists[v]) for v in range(n)]
        
        # For efficiency, we'll use a stack-based approach
        # But for simplicity, let's do recursive with memoization of failed states
        # Actually, for these small graphs, a well-ordered backtracking should be fast
        
        # Let's use a simpler recursive backtracking with forward checking
        # We'll use the order as the sequence in which we assign colors
        
        # Map vertex index in order to its position
        # Actually, let's just iterate through the order
        
        def backtrack(idx, domains_state):
            if time.time() - start_time > DEADLINE:
                return None
            
            if idx == len(order):
                return list(domains_state)
            
            v = order[idx]
            for c in domains_state[v]:
                # Check if c conflicts with already-colored neighbors
                conflict = False
                for u in adj[v]:
                    if domains_state[u] is not None and domains_state[u] == c:
                        conflict = True
                        break
                if conflict:
                    continue
                
                # Try assigning c to v
                new_domains = list(domains_state)
                new_domains[v] = c
                
                # Forward checking: for uncolored neighbors, remove c from their domains
                valid = True
                for u in adj[v]:
                    if new_domains[u] is None:  # u is uncolored
                        if c in new_domains[u]:
                            new_domains[u] = [x for x in new_domains[u] if x != c]
                            if not new_domains[u]:
                                valid = False
                                break
                
                if not valid:
                    continue
                
                result = backtrack(idx + 1, new_domains)
                if result is not None:
                    return result
            
            return None
        
        # Initialize domains_state: None means uncolored, list means domain
        # But we need to track which are colored. Let's use a different representation.
        # Actually, let's simplify: use a separate colour array and track domains separately.
        
        # Let me redo this with a cleaner approach
        colour2 = [None] * n
        # domains[v] is the current domain for vertex v (only relevant if colour2[v] is None)
        doms = [list(lists[v]) for v in range(n)]
        
        def bt2(idx):
            if time.time() - start_time > DEADLINE:
                return False
            
            if idx == len(order):
                return True
            
            v = order[idx]
            for c in doms[v]:
                # Check conflicts with colored neighbors
                ok = True
                for u in adj[v]:
                    if colour2[u] is not None and colour2[u] == c:
                        ok = False
                        break
                if not ok:
                    continue
                
                colour2[v] = c
                # Forward checking
                saved_doms = []
                ok2 = True
                for u in adj[v]:
                    if colour2[u] is None:
                        if c in doms[u]:
                            saved_doms.append((u, list(doms[u])))
                            doms[u] = [x for x in doms[u] if x != c]
                            if not doms[u]:
                                ok2 = False
                                break
                
                if ok2 and bt2(idx + 1):
                    return True
                
                # Undo
                colour2[v] = None
                for u, old_dom in saved_doms:
                    doms[u] = old_dom
            
            return False
        
        if bt2(0):
            return list(colour2)
        return None

    def solve_instance(n, edges, lists):
        adj = build_adj(n, edges)
        
        # Try multiple orderings
        # 1. Greedy by degree (descending)
        order1 = sorted(range(n), key=lambda v: -len(adj[v]))
        
        # 2. Greedy by degree (ascending)
        order2 = sorted(range(n), key=lambda v: len(adj[v]))
        
        # 3. Random order
        import random
        rng = random.Random(42)
        order3 = list(range(n))
        rng.shuffle(order3)
        
        # 4. Minimum remaining values heuristic (dynamic)
        # For now, try the static orders first with greedy, then backtrack
        
        # First, try greedy with each order
        for order in [order1, order2, order3]:
            colour = greedy_color(n, adj, lists, order)
            if is_valid_colouring(colour, adj, lists):
                return colour
        
        # If greedy fails, try backtracking
        for order in [order1, order2, order3]:
            if time.time() - start_time > DEADLINE:
                break
            colour = backtrack_color(n, adj, lists, order)
            if colour is not None and is_valid_colouring(colour, adj, lists):
                return colour
        
        # Last resort: return a greedy coloring (may be invalid)
        return greedy_color(n, adj, lists, order1)

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() - start_time > DEADLINE:
            # Return whatever we have so far, padded with None if needed
            # But the contract says wrong length makes it invalid, so we need all 4
            # Let's just do a quick greedy for the remaining
            while len(out) < 4:
                n, edges, lists = instance(splits, seed)
                adj = build_adj(n, edges)
                order = sorted(range(n), key=lambda v: -len(adj[v]))
                out.append(greedy_color(n, adj, lists, order))
            break
        
        n, edges, lists = instance(splits, seed)
        colour = solve_instance(n, edges, lists)
        # Ensure it's a list of Python ints
        colour = [int(c) for c in colour]
        out.append(colour)
    
    # Ensure exactly 4 entries
    while len(out) < 4:
        # This shouldn't happen, but just in case
        out.append([])
    
    return out[:4]
# EVOLVE-BLOCK-END
