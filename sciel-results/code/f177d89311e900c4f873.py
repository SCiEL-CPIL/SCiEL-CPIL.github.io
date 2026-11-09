import time
import random

def construct():
    n = 15
    start = time.time()
    deadline = start + 55.0  # Safety margin

    # Initialize with a dense graph, e.g., K15
    edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            edges.add((i, j))

    def get_adj_deg(edges):
        adj = [set() for _ in range(n)]
        deg = [0] * n
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1
        return adj, deg

    def count_violations(adj, deg):
        # Group vertices by degree
        deg_map = {}
        for i in range(n):
            deg_map.setdefault(deg[i], []).append(i)
        
        violations = 0
        for d, verts in deg_map.items():
            if len(verts) < 2:
                continue
            for i in range(len(verts)):
                for j in range(i + 1, len(verts)):
                    s = verts[i]
                    t = verts[j]
                    # Check if there is a path s-u-v-t
                    # u in adj[s], v in adj[u] \ {s,t}, t in adj[v]
                    found = False
                    for u in adj[s]:
                        if u == t:
                            continue
                        for v in adj[u]:
                            if v == s or v == t:
                                continue
                            if t in adj[v]:
                                found = True
                                break
                        if found:
                            break
                    if found:
                        violations += 1
        return violations

    # Simulated Annealing
    current_edges = set(edges)
    adj, deg = get_adj_deg(current_edges)
    current_violations = count_violations(adj, deg)
    
    best_edges = set(current_edges)
    best_violations = current_violations
    best_edges_count = len(current_edges)

    temp = 10.0
    cooling = 0.99
    
    while time.time() < deadline:
        # Generate a move: add or remove a random edge
        all_possible = set()
        for i in range(n):
            for j in range(i + 1, n):
                all_possible.add((i, j))
        
        # Pick a random edge from all possible
        u, v = random.choice(list(all_possible))
        
        if (u, v) in current_edges:
            current_edges.remove((u, v))
        else:
            current_edges.add((u, v))
            
        new_adj, new_deg = get_adj_deg(current_edges)
        new_violations = count_violations(new_adj, new_deg)
        
        # Acceptance
        # We want to minimize violations, then maximize edges
        # Score = -violations * 1000 + edges
        
        curr_score = -current_violations * 1000 + len(current_edges)
        new_score = -new_violations * 1000 + len(current_edges)
        
        delta = new_score - curr_score
        
        if delta > 0 or random.random() < pow(2.718, delta / temp):
            current_edges = set(current_edges) # Copy
            current_violations = new_violations
            adj, deg = new_adj, new_deg
            
            # Update best
            if new_violations == 0:
                if len(current_edges) > best_edges_count:
                    best_edges = set(current_edges)
                    best_edges_count = len(current_edges)
                    best_violations = 0
            elif new_violations < best_violations:
                # Keep track of best even if not valid, in case we timeout
                # But we prefer valid.
                pass

        temp *= cooling
        if temp < 0.01:
            temp = 0.01

    # Return best valid graph found
    if best_violations == 0:
        out = [[u, v] for u, v in best_edges]
        return out
    else:
        # Fallback to star if no valid graph found (should not happen)
        out = [[0, v] for v in range(1, n)]
        return out
