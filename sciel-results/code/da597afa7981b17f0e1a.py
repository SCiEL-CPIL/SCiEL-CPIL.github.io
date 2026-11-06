"""Search-based construction for 16-vertex 65-edge graph minimizing max triangle degree sum."""
import time
import random
import itertools

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    n = 16
    target_edges = 65
    
    # Start from K_{8,8} + edge (0,1)
    # Part A: 0-7, Part B: 8-15
    def make_base():
        edges = set()
        for u in range(8):
            for v in range(8, 16):
                edges.add((min(u,v), max(u,v)))
        edges.add((0, 1))
        return edges
    
    def edge_set_to_list(es):
        return [[u, v] for (u, v) in sorted(es)]
    
    def compute_score(es):
        """Compute max triangle degree sum using bitmask adjacency."""
        adj = [0] * n
        deg = [0] * n
        for (u, v) in es:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            deg[u] += 1
            deg[v] += 1
        
        # Find all triangles via bitmask intersection
        best = 0
        # For each edge (u,v), check common neighbors w
        for (u, v) in es:
            common = adj[u] & adj[v]
            # Iterate over set bits in common
            c = common
            while c:
                w = (c & -c).bit_length() - 1
                c &= c - 1
                s = deg[u] + deg[v] + deg[w]
                if s > best:
                    best = s
        return best, deg, adj
    
    def is_triangle_free_addition(es, adj, u, v):
        """Check if adding edge (u,v) creates any triangle."""
        common = adj[u] & adj[v]
        return common == 0  # no common neighbor means no new triangle
    
    def would_create_worse_triangle(es, deg, adj, u, v, current_max):
        """Check if adding (u,v) creates a triangle with sum > current_max."""
        common = adj[u] & adj[v]
        c = common
        while c:
            w = (c & -c).bit_length() - 1
            c &= c - 1
            s = deg[u] + deg[v] + deg[w]
            if s > current_max:
                return True
        return False
    
    # Initial graph
    best_es = make_base()
    best_score, best_deg, best_adj = compute_score(best_es)
    
    # Local search: try edge swaps
    # A swap: remove edge (a,b), add edge (c,d) where (c,d) is not in graph
    # Must maintain exactly 65 edges and be a simple graph
    
    def try_local_search(es, score, deg, adj, max_iters=50000):
        """Degree-ordered edge replacement heuristic."""
        current_es = set(es)
        current_score = score
        current_deg = list(deg)
        current_adj = list(adj)
        
        all_vertices = list(range(n))
        non_edges = [(u, v) for u in range(n) for v in range(u+1, n) if (u, v) not in current_es]
        
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1
            
            # Find the worst triangle
            # Re-find worst triangle
            worst_sum = 0
            worst_tri = None
            for (u, v) in current_es:
                common = current_adj[u] & current_adj[v]
                c = common
                while c:
                    w = (c & -c).bit_length() - 1
                    c &= c - 1
                    s = current_deg[u] + current_deg[v] + current_deg[w]
                    if s > worst_sum:
                        worst_sum = s
                        worst_tri = (u, v, w)
            
            if worst_tri is None:
                break
            
            u, v, w = worst_tri
            # The three edges of this triangle
            tri_edges = [(min(u,v), max(u,v)), (min(u,w), max(u,w)), (min(v,w), max(v,w))]
            
            # Priority: remove edge incident to highest-degree vertex in the triangle
            # Sort triangle edges by max degree of endpoints (descending)
            tri_edges_sorted = sorted(tri_edges, key=lambda e: current_deg[e[0]] + current_deg[e[1]], reverse=True)
            
            found_swap = False
            for (a, b) in tri_edges_sorted:
                # Try removing (a,b) and adding some non-edge
                # After removal, degrees change
                new_deg_a = current_deg[a] - 1
                new_deg_b = current_deg[b] - 1
                
                # Candidate non-edges to add: prefer edges that don't create triangles
                # or create triangles with lower sum
                # Sort non-edges by preference: lower degree sum of endpoints first
                candidates = []
                for (c, d) in non_edges:
                    if (c, d) == (a, b) or (c, d) == (b, a):
                        continue
                    # Check if (c,d) is already in current_es minus (a,b)
                    if (c, d) in current_es:
                        continue
                    # Estimate: after swap, what's the new max?
                    # This is expensive; use a heuristic filter
                    # Prefer edges between low-degree vertices
                    candidates.append((current_deg[c] + current_deg[d], c, d))
                
                # Sort by degree sum ascending (prefer low-degree connections)
                candidates.sort()
                
                for (deg_sum, c, d) in candidates[:50]:  # limit candidates
                    if time.time() > deadline:
                        return current_es, current_score
                    # Try the swap
                    trial_es = current_es - {(a, b)} | {(c, d)}
                    trial_score, trial_deg, trial_adj = compute_score(trial_es)
                    if trial_score < current_score:
                        current_es = trial_es
                        current_score = trial_score
                        current_deg = trial_deg
                        current_adj = trial_adj
                        # Update non_edges
                        non_edges = [(u2, v2) for u2 in range(n) for v2 in range(u2+1, n) if (u2, v2) not in current_es]
                        improved = True
                        found_swap = True
                        break
                
                if found_swap:
                    break
            
            if not found_swap:
                # Try random perturbations to escape local optimum
                if iters % 100 == 0 and random.random() < 0.5:
                    # Pick a random edge to remove and random non-edge to add
                    if len(current_es) > 0 and len(non_edges) > 0:
                        rem_edge = random.choice(list(current_es))
                        add_edge = random.choice(non_edges)
                        if add_edge not in current_es:
                            trial_es = current_es - {rem_edge} | {add_edge}
                            trial_score, _, _ = compute_score(trial_es)
                            if trial_score <= current_score + 2:  # allow small increase
                                current_es = trial_es
                                current_score = trial_score
                                current_deg, current_adj = compute_score(current_es)[1:3]
                                non_edges = [(u2, v2) for u2 in range(n) for v2 in range(u2+1, n) if (u2, v2) not in current_es]
        
        return current_es, current_score
    
    # Run local search
    best_es, best_score = try_local_search(best_es, best_score, best_deg, best_adj)
    
    # Try multiple restarts with different random seeds
    rng = random.Random(42)
    for restart in range(20):
        if time.time() > deadline:
            break
        
        # Create a perturbed starting point
        start_es = set(best_es)
        # Make 2-3 random edge swaps
        for _ in range(rng.randint(2, 4)):
            if len(start_es) < 65:
                break
            rem = rng.choice(list(start_es))
            non_edges_now = [(u, v) for u in range(n) for v in range(u+1, n) if (u, v) not in start_es]
            if not non_edges_now:
                break
            add = rng.choice(non_edges_now)
            start_es = start_es - {rem} | {add}
        
        start_score, start_deg, start_adj = compute_score(start_es)
        
        # Run local search from this start
        es_result, score_result = try_local_search(start_es, start_score, start_deg, start_adj, max_iters=20000)
        
        if score_result < best_score:
            best_es = es_result
            best_score = score_result
    
    # Final check
    final_score, final_deg, final_adj = compute_score(best_es)
    if len(best_es) != target_edges:
        # Fallback to base
        best_es = make_base()
        final_score = 24
    
    return {'n': n, 'edges': edge_set_to_list(best_es)}

# Quick self-test
if __name__ == "__main__":
    result = construct()
    assert result['n'] == 16
    assert len(result['edges']) == 65
    print(f"Edges: {len(result['edges'])}")
    # Compute score
    n = 16
    deg = [0] * n
    for (u, v) in result['edges']:
        deg[u] += 1
        deg[v] += 1
    adj = [0] * n
    for (u, v) in result['edges']:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    best = 0
    for (u, v) in result['edges']:
        common = adj[u] & adj[v]
        c = common
        while c:
            w = (c & -c).bit_length() - 1
            c &= c - 1
            s = deg[u] + deg[v] + deg[w]
            if s > best:
                best = s
    print(f"Score: {best}")
