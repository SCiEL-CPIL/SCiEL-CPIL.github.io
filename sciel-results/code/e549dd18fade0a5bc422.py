"""Improved program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
from collections import defaultdict
import math

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a graph on 16 vertices with exactly 65 edges that minimizes the
    maximum degree sum over all triangles.
    
    Uses incremental bitmask-based scoring for speed, starts from K_{6,10} base,
    and runs local search + simulated annealing + 2-swap polish.
    """
    n = 16
    target_edges = 65
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 148.0
    
    # Full score from scratch using bitmasks
    def score_from_adj(adj_masks, deg):
        """Compute max triangle degree sum using bitmask adjacency."""
        max_sum = 0
        for u in range(n):
            mu = adj_masks[u]
            for v in range(u + 1, n):
                if mu & (1 << v):
                    common = mu & adj_masks[v]
                    cm = common
                    while cm:
                        lsb = cm & (-cm)
                        w = lsb.bit_length() - 1
                        s = deg[u] + deg[v] + deg[w]
                        if s > max_sum:
                            max_sum = s
                        cm ^= lsb
        return max_sum
    
    def build_adj(edge_set):
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
        adj_masks, deg = build_adj(es)
        return score_from_adj(adj_masks, deg)
    
    # Incremental score update for a single edge swap (remove e1, add e2)
    def score_after_swap(adj_masks, deg, e1, e2, current_score):
        """
        Compute new max triangle degree sum after removing edge e1=(a,b) and adding e2=(c,d).
        Only triangles involving vertices {a,b,c,d} can change.
        Returns (new_adj_masks, new_deg, new_score).
        """
        a, b = e1
        c, d = e2
        affected = {a, b, c, d}
        
        # Apply changes to adj and deg
        adj_masks[a] ^= (1 << b)
        adj_masks[b] ^= (1 << a)
        deg[a] -= 1
        deg[b] -= 1
        adj_masks[c] |= (1 << d)
        adj_masks[d] |= (1 << c)
        deg[c] += 1
        deg[d] += 1
        
        # Collect all triangles involving at least one affected vertex
        # A triangle (u,v,w) is affected if any of u,v,w is in affected set
        # AND the triangle existed before or exists after the swap
        # We need to check: triangles that existed before but are gone, and new triangles
        
        # Strategy: recompute score over all triangles that involve at least one affected vertex
        # Triangles not involving any affected vertex are unchanged
        # But we also need to check if removing e1 destroyed a triangle that was the max
        # and if adding e2 created a new triangle that is the max
        
        # Collect all triangles that could be affected:
        # - Triangles containing a or b (since e1 removed, some may disappear)
        # - Triangles containing c or d (since e2 added, new ones may appear)
        # - Triangles containing any of a,b,c,d with the other two vertices
        
        tri_sums = []
        
        # For each affected vertex, find all its triangles in the NEW graph
        for av in affected:
            mau = adj_masks[av]
            # For each neighbor v of av
            mv = mau
            while mv:
                lsb = mv & (-mv)
                v = lsb.bit_length() - 1
                if v > av:  # avoid double counting
                    common = mau & adj_masks[v]
                    cm = common
                    while cm:
                        lsb2 = cm & (-cm)
                        w = lsb2.bit_length() - 1
                        if w > v:
                            s = deg[av] + deg[v] + deg[w]
                            tri_sums.append(s)
                        cm ^= lsb2
                mv ^= lsb
        
        # Also need to check: did removing e1 destroy the max triangle?
        # The old max might have involved a,b and a third vertex.
        # We need to find the max over ALL triangles in the new graph.
        # Since only triangles involving affected vertices changed, and we've collected
        # all triangles involving affected vertices in the new graph, the new max is:
        # max(old_max_not_involving_affected, max_of_new_triangles_involving_affected)
        
        # But we don't track which triangles don't involve affected vertices.
        # Simpler: just compute the full score. For n=16 it's fast enough.
        # Actually let me count: number of triangles in a 65-edge graph on 16 vertices
        # is at most C(16,3) = 560. That's very fast.
        
        # Let's just do full recompute - it's O(E * avg_degree) which is tiny
        new_max = 0
        for u in range(n):
            mu = adj_masks[u]
            for v in range(u + 1, n):
                if mu & (1 << v):
                    common = mu & adj_masks[v]
                    cm = common
                    while cm:
                        lsb = cm & (-cm)
                        w = lsb.bit_length() - 1
                        s = deg[u] + deg[v] + deg[w]
                        if s > new_max:
                            new_max = s
                        cm ^= lsb
        
        return adj_masks, deg, new_max
    
    def make_seed_edges():
        """K_{6,10} + 5 internal edges in the 10-vertex part."""
        edges = []
        for u in range(6):
            for v in range(6, 16):
                edges.append([u, v])
        internal_edges = [(6,7), (8,9), (10,11), (12,13), (14,15)]
        for u, v in internal_edges:
            edges.append([u, v])
        return edges
    
    def make_seed_edges_v2():
        """K_{8,8} + one internal edge."""
        edges = [[u, v] for u in range(8) for v in range(8, 16)]
        edges.append([0, 1])
        return edges
    
    def edges_to_set(edges):
        es = set()
        for u, v in edges:
            es.add((min(u, v), max(u, v)))
        return es
    
    def edges_to_list(es):
        return [[u, v] for u, v in es]
    
    # Try multiple seeds
    best_edges = None
    best_score = float('inf')
    
    for seed_fn in [make_seed_edges, make_seed_edges_v2]:
        seed_edges = seed_fn()
        s = compute_score(seed_edges)
        if s < best_score:
            best_score = s
            best_edges = seed_edges
    
    # Generate candidate edges
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    all_pairs_set = set(all_pairs)
    
    # Local search with edge swaps
    current_es = edges_to_set(best_edges)
    current_adj, current_deg = build_adj(current_es)
    current_score = best_score
    
    random.seed(42)
    no_improve = 0
    max_no_improve = 5000
    
    while time.time() < deadline - 10 and no_improve < max_no_improve:
        e_remove = random.choice(list(current_es))
        candidates = [p for p in all_pairs if p not in current_es]
        if not candidates:
            break
        e_add = random.choice(candidates)
        
        new_adj, new_deg, new_score = score_after_swap(
            current_adj[:], current_deg[:], e_remove, e_add, current_score
        )
        
        if new_score < current_score:
            current_es.discard(e_remove)
            current_es.add(e_add)
            current_adj = new_adj
            current_deg = new_deg
            current_score = new_score
            no_improve = 0
            if current_score < best_score:
                best_score = current_score
                best_edges = edges_to_list(current_es)
        else:
            no_improve += 1
    
    # Simulated annealing phase
    if best_score >= 23 and time.time() < deadline - 10:
        random.seed(123)
        current_es = edges_to_set(best_edges)
        current_adj, current_deg = build_adj(current_es)
        current_score = best_score
        
        T = 5.0
        T_min = 0.1
        cooling = 0.9995
        
        for iteration in range(200000):
            if time.time() > deadline - 10:
                break
            if T < T_min:
                break
            
            e_remove = random.choice(list(current_es))
            candidates = [p for p in all_pairs if p not in current_es]
            if not candidates:
                break
            e_add = random.choice(candidates)
            
            new_adj, new_deg, new_score = score_after_swap(
                current_adj[:], current_deg[:], e_remove, e_add, current_score
            )
            
            delta = new_score - current_score
            
            if delta < 0 or random.random() < math.exp(-delta / T):
                current_es.discard(e_remove)
                current_es.add(e_add)
                current_adj = new_adj
                current_deg = new_deg
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_edges = edges_to_list(current_es)
            
            T *= cooling
    
    # Final polish: try all single-edge swaps
    if time.time() < deadline - 3:
        best_es = edges_to_set(best_edges)
        best_s = best_score
        
        for e_remove in list(best_es):
            if time.time() > deadline - 3:
                break
            for e_add in all_pairs:
                if e_add in best_es:
                    continue
                new_adj, new_deg, new_s = score_after_swap(
                    build_adj(best_es)[0][:], build_adj(best_es)[1][:], e_remove, e_add, best_s
                )
                if new_s < best_s:
                    best_s = new_s
                    best_es.discard(e_remove)
                    best_es.add(e_add)
                    best_edges = edges_to_list(best_es)
                    best_score = best_s
    
    # 2-edge swap polish: try removing 2 edges and adding 2 edges
    if time.time() < deadline - 1:
        best_es = edges_to_set(best_edges)
        best_s = best_score
        best_adj, best_deg = build_adj(best_es)
        
        es_list = list(best_es)
        non_es = [p for p in all_pairs if p not in best_es]
        
        # Try pairs of edges to remove and pairs to add
        # Limit to a subset for time
        max_2swap_checks = 200000
        checks = 0
        
        for i in range(len(es_list)):
            if time.time() > deadline - 1 or checks >= max_2swap_checks:
                break
            for j in range(i + 1, len(es_list)):
                if time.time() > deadline - 1 or checks >= max_2swap_checks:
                    break
                e1 = es_list[i]
                e2 = es_list[j]
                for k in range(len(non_es)):
                    if time.time() > deadline - 1 or checks >= max_2swap_checks:
                        break
                    for l in range(k + 1, len(non_es)):
                        if time.time() > deadline - 1 or checks >= max_2swap_checks:
                            break
                        f1 = non_es[k]
                        f2 = non_es[l]
                        checks += 1
                        
                        # Apply 2-swap
                        adj = best_adj[:]
                        deg = best_deg[:]
                        for a, b in [e1, e2]:
                            adj[a] ^= (1 << b)
                            adj[b] ^= (1 << a)
                            deg[a] -= 1
                            deg[b] -= 1
                        for a, b in [f1, f2]:
                            adj[a] |= (1 << b)
                            adj[b] |= (1 << a)
                            deg[a] += 1
                            deg[b] += 1
                        
                        # Compute score
                        new_max = 0
                        for u in range(n):
                            mu = adj[u]
                            for v in range(u + 1, n):
                                if mu & (1 << v):
                                    common = mu & adj[v]
                                    cm = common
                                    while cm:
                                        lsb = cm & (-cm)
                                        w = lsb.bit_length() - 1
                                        s = deg[u] + deg[v] + deg[w]
                                        if s > new_max:
                                            new_max = s
                                        cm ^= lsb
                        
                        if new_max < best_s:
                            best_s = new_max
                            best_es.discard(e1)
                            best_es.discard(e2)
                            best_es.add(f1)
                            best_es.add(f2)
                            best_edges = edges_to_list(best_es)
                            best_score = best_s
                            best_adj, best_deg = build_adj(best_es)
                            es_list = list(best_es)
                            non_es = [p for p in all_pairs if p not in best_es]
                            break
    
    return {'n': n, 'edges': best_edges}
# EVOLVE-BLOCK-END
