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
    
    Uses bitmask-based scoring for speed, starts from K_{6,10} base with
    internal edges in the 10-vertex part, and runs local search with
    simulated annealing to find score <= 23.
    """
    n = 16
    target_edges = 65
    
    # Deadline guard: return best found before 148s (2s safety margin)
    start_time = time.time()
    deadline = start_time + 148.0
    
    # Fast scoring using bitmasks
    def score_from_adj(adj_masks, deg):
        """Compute max triangle degree sum using bitmask adjacency."""
        max_sum = 0
        for u in range(n):
            mu = adj_masks[u]
            for v in range(u + 1, n):
                if mu & (1 << v):
                    common = mu & adj_masks[v]
                    # Iterate over set bits in common
                    cm = common
                    while cm:
                        lsb = cm & (-cm)
                        w = lsb.bit_length() - 1
                        s = deg[u] + deg[v] + deg[w]
                        if s > max_sum:
                            max_sum = s
                        cm ^= lsb
        return max_sum
    
    def build_adj(deg, edge_set):
        """Build bitmask adjacency and degree array from edge set."""
        adj_masks = [0] * n
        deg = [0] * n
        for u, v in edge_set:
            adj_masks[u] |= (1 << v)
            adj_masks[v] |= (1 << u)
            deg[u] += 1
            deg[v] += 1
        return adj_masks, deg
    
    def compute_score(edges):
        """Compute the maximum degree sum over all triangles."""
        es = set()
        for u, v in edges:
            es.add((min(u, v), max(u, v)))
        adj_masks, deg = build_adj(None, es)
        return score_from_adj(adj_masks, deg)
    
    # Start from K_{6,10} base: vertices 0-5 in part A (size 6), 6-15 in part B (size 10)
    # K_{6,10} has 60 edges. Need 5 more edges within part B.
    # Degrees in K_{6,10}: part A vertices have deg 10, part B vertices have deg 6.
    # Adding edges within B increases degrees of B vertices.
    # Strategy: add 5 edges in B to balance degrees and minimize triangle sums.
    
    def make_seed_edges():
        """K_{6,10} + 5 internal edges in the 10-vertex part."""
        edges = []
        # Cross edges: all pairs between {0..5} and {6..15}
        for u in range(6):
            for v in range(6, 16):
                edges.append([u, v])
        # 5 internal edges in part B (vertices 6-15)
        # Use a matching-like structure to keep degrees balanced
        # Add edges: (6,7), (8,9), (10,11), (12,13), (14,15)
        internal_edges = [(6,7), (8,9), (10,11), (12,13), (14,15)]
        for u, v in internal_edges:
            edges.append([u, v])
        return edges
    
    best_edges = make_seed_edges()
    best_score = compute_score(best_edges)
    
    # Also try K_{8,8} + 1 edge as alternative seed
    def make_seed_edges_v2():
        """K_{8,8} + one internal edge."""
        edges = [[u, v] for u in range(8) for v in range(8, 16)]
        edges.append([0, 1])
        return edges
    
    s2 = compute_score(make_seed_edges_v2())
    if s2 < best_score:
        best_score = s2
        best_edges = make_seed_edges_v2()
    
    def edges_to_set(edges):
        es = set()
        for u, v in edges:
            es.add((min(u, v), max(u, v)))
        return es
    
    def edges_to_list(es):
        return [[u, v] for u, v in es]
    
    # Generate candidate edges
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Local search with edge swaps using bitmask scoring
    current_es = edges_to_set(best_edges)
    current_adj, current_deg = build_adj(None, current_es)
    current_score = best_score
    
    random.seed(42)
    no_improve = 0
    max_no_improve = 2000  # Allow more iterations with faster scoring
    
    while time.time() < deadline - 5 and no_improve < max_no_improve:
        # Pick a random edge to remove
        if not current_es:
            break
        e_remove = random.choice(list(current_es))
        # Pick a random edge to add (not already present)
        candidates = [p for p in all_pairs if p not in current_es]
        if not candidates:
            break
        e_add = random.choice(candidates)
        
        new_es = set(current_es)
        new_es.remove(e_remove)
        new_es.add(e_add)
        
        new_adj, new_deg = build_adj(None, new_es)
        new_score = score_from_adj(new_adj, new_deg)
        
        if new_score < current_score:
            current_es = new_es
            current_adj, current_deg = new_adj, new_deg
            current_score = new_score
            no_improve = 0
            if current_score < best_score:
                best_score = current_score
                best_edges = edges_to_list(current_es)
        else:
            no_improve += 1
    
    # Simulated annealing phase if still at or above 24
    if best_score >= 24 and time.time() < deadline - 5:
        random.seed(123)
        current_es = edges_to_set(best_edges)
        current_adj, current_deg = build_adj(None, current_es)
        current_score = best_score
        
        T = 5.0
        T_min = 0.1
        cooling = 0.999
        
        for iteration in range(50000):
            if time.time() > deadline - 5:
                break
            if T < T_min:
                break
            
            # Pick a random edge swap
            if not current_es:
                break
            e_remove = random.choice(list(current_es))
            candidates = [p for p in all_pairs if p not in current_es]
            if not candidates:
                break
            e_add = random.choice(candidates)
            
            new_es = set(current_es)
            new_es.remove(e_remove)
            new_es.add(e_add)
            
            new_adj, new_deg = build_adj(None, new_es)
            new_score = score_from_adj(new_adj, new_deg)
            
            delta = new_score - current_score
            
            if delta < 0 or random.random() < pow(2.718281828, -delta / T):
                current_es = new_es
                current_adj, current_deg = new_adj, new_deg
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_edges = edges_to_list(current_es)
            
            T *= cooling
    
    # Final polish: try all single-edge swaps from best
    if time.time() < deadline - 2:
        best_es = edges_to_set(best_edges)
        best_adj, best_deg = build_adj(None, best_es)
        best_s = best_score
        
        # Try removing each edge and adding each non-edge
        # This is O(E * (V^2 - E)) which is manageable for n=16
        for e_remove in list(best_es):
            if time.time() > deadline - 2:
                break
            for e_add in all_pairs:
                if e_add in best_es:
                    continue
                new_es = set(best_es)
                new_es.remove(e_remove)
                new_es.add(e_add)
                new_adj, new_deg = build_adj(None, new_es)
                new_s = score_from_adj(new_adj, new_deg)
                if new_s < best_s:
                    best_s = new_s
                    best_es = new_es
                    best_adj, best_deg = new_adj, new_deg
                    best_edges = edges_to_list(best_es)
                    best_score = best_s
    
    return {'n': n, 'edges': best_edges}
# EVOLVE-BLOCK-END
