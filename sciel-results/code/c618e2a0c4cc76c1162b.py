"""Improved program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a graph on 16 vertices with exactly 65 edges that minimizes the
    maximum degree sum over all triangles.
    
    The seed uses K_{8,8} + one edge, giving score 24.
    We try to find a graph with score <= 23 using local search.
    """
    n = 16
    target_edges = 65
    
    # Deadline guard: return best found before 145s
    start_time = time.time()
    deadline = start_time + 145.0
    
    def compute_score(edges):
        """Compute the maximum degree sum over all triangles."""
        adj = defaultdict(set)
        deg = [0] * n
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1
        
        # Find all triangles
        max_sum = 0
        edge_set = set()
        for u, v in edges:
            edge_set.add((min(u,v), max(u,v)))
        
        for u in range(n):
            for v in adj[u]:
                if v <= u:
                    continue
                # Common neighbors of u and v
                common = adj[u] & adj[v]
                for w in common:
                    if w > v:
                        s = deg[u] + deg[v] + deg[w]
                        if s > max_sum:
                            max_sum = s
        return max_sum
    
    # Start from seed: K_{8,8} + edge (0,1)
    def make_seed_edges():
        edges = [[u, v] for u in range(8) for v in range(8, 16)]
        edges.append([0, 1])
        return edges
    
    best_edges = make_seed_edges()
    best_score = compute_score(best_edges)
    
    # Local search: try swapping edges to reduce max triangle degree sum
    def get_edge_set(edges):
        es = set()
        for u, v in edges:
            es.add((min(u,v), max(u,v)))
        return es
    
    def edges_to_list(es):
        return [[u, v] for u, v in es]
    
    # Generate candidate edges
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Local search with edge swaps
    current_edges = set(get_edge_set(best_edges))
    current_score = best_score
    
    # Precompute adjacency for faster triangle detection
    def score_from_edge_set(es):
        adj = defaultdict(set)
        deg = [0] * n
        for u, v in es:
            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1
        
        max_sum = 0
        for u in range(n):
            for v in adj[u]:
                if v <= u:
                    continue
                common = adj[u] & adj[v]
                for w in common:
                    if w > v:
                        s = deg[u] + deg[v] + deg[w]
                        if s > max_sum:
                            max_sum = s
        return max_sum
    
    # Try random edge swaps
    random.seed(42)
    no_improve = 0
    max_no_improve = 500
    
    while time.time() < deadline and no_improve < max_no_improve:
        # Pick a random edge to remove
        if not current_edges:
            break
        e_remove = random.choice(list(current_edges))
        # Pick a random edge to add (not already present)
        candidates = [p for p in all_pairs if p not in current_edges]
        if not candidates:
            break
        e_add = random.choice(candidates)
        
        new_edges = set(current_edges)
        new_edges.remove(e_remove)
        new_edges.add(e_add)
        
        new_score = score_from_edge_set(new_edges)
        
        if new_score < current_score:
            current_edges = new_edges
            current_score = new_score
            no_improve = 0
            if current_score < best_score:
                best_score = current_score
                best_edges = edges_to_list(current_edges)
        else:
            no_improve += 1
    
    # If we haven't improved, try a more systematic approach
    if best_score >= 24:
        # Try to find a graph with score <= 23 by construction
        # Strategy: Start from balanced complete bipartite and add edges carefully
        # Or use a different base structure
        
        # Try: Two groups of 8, but add edges within groups strategically
        # The key insight: in K_{8,8}, degrees are all 8. Adding one edge (u,v) 
        # within a part makes deg(u)=9, deg(v)=9, others 8.
        # Triangles involving (u,v): for each w in other part, triangle (u,v,w) has sum 9+9+8=26.
        # So score is 26? Wait, let me recheck.
        
        # Actually in K_{8,8} + edge (0,1) where 0,1 in part A:
        # deg(0)=9, deg(1)=9, others in A: 8, all in B: 8
        # Triangle (0,1,w) for w in B: deg sum = 9+9+8 = 26
        # But wait, are there other triangles? No, because no other edges within parts.
        # So score should be 26, not 24. Let me recheck the seed score.
        
        # Hmm, the task says seed scores 1.0 which equals the baseline.
        # Let me just compute it properly.
        
        pass
    
    # Final attempt: try specific constructions
    # Construction 1: K_{8,8} + edge
    edges1 = make_seed_edges()
    s1 = compute_score(edges1)
    
    # Construction 2: Try to balance degrees better
    # What if we use a different base?
    # Try: Start with a 16-vertex graph with 65 edges and more balanced triangle degree sums
    
    # Let's try: complete bipartite K_{7,9} = 63 edges, need 2 more
    # Degrees: 7 vertices have deg 9, 9 vertices have deg 7
    # Add 2 edges within the part of 7:
    # If we add (0,1) and (2,3) in the part of 7:
    # deg(0)=10, deg(1)=10, deg(2)=10, deg(3)=10, others in part7: 9, part9: 7
    # Triangles: (0,1,w) for w in part9: 10+10+7=27, worse
    
    # What about K_{8,8} but add the edge differently?
    # The issue is that any edge within a part creates triangles with high degree sum.
    
    # Alternative: don't start from bipartite. Try a different structure.
    # Maybe start from a random graph and optimize.
    
    # Let's try simulated annealing style search
    random.seed(123)
    current_es = set(get_edge_set(make_seed_edges()))
    current_s = compute_score(edges_to_list(current_es))
    
    # Actually, let me just try many random perturbations
    for trial in range(200):
        if time.time() > deadline - 5:
            break
        
        # Start from seed
        es = set(get_edge_set(make_seed_edges()))
        
        # Do some random swaps
        for _ in range(20):
            if not es:
                break
            e_rem = random.choice(list(es))
            cands = [p for p in all_pairs if p not in es]
            if not cands:
                break
            e_add = random.choice(cands)
            es.remove(e_rem)
            es.add(e_add)
        
        s = compute_score(edges_to_list(es))
        if s < best_score:
            best_score = s
            best_edges = edges_to_list(es)
    
    return {'n': n, 'edges': best_edges}
# EVOLVE-BLOCK-END
