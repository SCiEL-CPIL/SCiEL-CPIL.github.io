import random
import itertools

def construct():
    n = 20
    num_colors = 20
    
    best_edges = []
    best_count = 0
    
    # Run a few times to find a good solution
    for run_seed in range(10):
        random.seed(1000 + run_seed)
        
        edges = {}
        adj = {i: set() for i in range(n)}
        all_possible_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        
        for u, v in all_possible_edges:
            c = random.randint(0, num_colors - 1)
            edges[(u, v)] = c
            adj[u].add(v)
            adj[v].add(u)
            
        def get_color(u, v):
            return edges.get((min(u, v), max(u, v)))
            
        def is_rainbow_cycle(u, v, w, x):
            c1 = get_color(u, v)
            c2 = get_color(v, w)
            c3 = get_color(w, x)
            c4 = get_color(x, u)
            if c1 is None or c2 is None or c3 is None or c4 is None:
                return False
            return len({c1, c2, c3, c4}) == 4

        def count_local_violations(u, v):
            # Count 4-cycles containing edge (u,v) that are NOT rainbow
            count = 0
            for w in adj[v]:
                if w == u: continue
                for x in adj[u]:
                    if x == v or x == w: continue
                    if x in adj[w]:
                        if not is_rainbow_cycle(u, v, w, x):
                            count += 1
            return count

        # Main repair loop
        # We will iterate until no improvements or max steps
        max_steps = 3000
        for _ in range(max_steps):
            # Pick a random edge
            if not edges:
                break
            u, v = random.choice(list(edges.keys()))
            
            # Check if this edge is involved in any violations
            local_viol = count_local_violations(u, v)
            
            if local_viol == 0:
                continue # No need to change this edge
            
            # Try to find a color that reduces local violations
            old_c = get_color(u, v)
            best_new_c = None
            min_viol = local_viol
            
            for new_c in range(num_colors):
                if new_c == old_c: continue
                edges[(min(u,v), max(u,v))] = new_c
                new_viol = count_local_violations(u, v)
                if new_viol < min_viol:
                    min_viol = new_viol
                    best_new_c = new_c
                edges[(min(u,v), max(u,v))] = old_c
                
            if best_new_c is not None:
                edges[(min(u,v), max(u,v))] = best_new_c
            else:
                # If no color helps, remove the edge
                del edges[(min(u,v), max(u,v))]
                adj[u].remove(v)
                adj[v].remove(u)
                
        # Final validation
        is_valid = True
        for combo in itertools.combinations(range(n), 4):
            a, b, c, d = combo
            if b in adj[a] and c in adj[b] and d in adj[c] and a in adj[d]:
                if not is_rainbow_cycle(a, b, c, d):
                    is_valid = False
                    break
            if d in adj[a] and b in adj[d] and c in adj[b] and a in adj[c]:
                if not is_rainbow_cycle(a, b, d, c):
                    is_valid = False
                    break
            if c in adj[a] and b in adj[c] and d in adj[b] and a in adj[d]:
                if not is_rainbow_cycle(a, c, b, d):
                    is_valid = False
                    break
            if not is_valid: break
            
        if is_valid:
            if len(edges) > best_count:
                best_count = len(edges)
                best_edges = list([[u, v, c] for (u,v), c in edges.items()])
                
    return best_edges
