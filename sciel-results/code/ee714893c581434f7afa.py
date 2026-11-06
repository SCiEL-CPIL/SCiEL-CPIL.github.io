"""Improved program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a graph on 16 vertices with exactly 65 edges that minimizes the
    maximum degree sum over all triangles.
    
    Uses bitmask-based scoring, multi-restart local search with simulated
    annealing, and a final exhaustive polish.
    """
    n = 16
    target_edges = 65
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Precompute all pairs
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    all_pairs_set = set(all_pairs)
    
    # Fast scoring using bitmasks
    def build_adj(edge_set):
        adj_masks = [0] * n
        deg = [0] * n
        for u, v in edge_set:
            adj_masks[u] |= (1 << v)
            adj_masks[v] |= (1 << u)
            deg[u] += 1
            deg[v] += 1
        return adj_masks, deg
    
    def score_from_adj(adj_masks, deg):
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
    
    def compute_score(edge_set):
        adj_masks, deg = build_adj(edge_set)
        return score_from_adj(adj_masks, deg)
    
    def apply_swap(adj_masks, deg, e_remove, e_add):
        """Apply a swap in-place, return copies of modified arrays."""
        new_adj = adj_masks[:]
        new_deg = deg[:]
        a, b = e_remove
        c, d = e_add
        new_adj[a] ^= (1 << b)
        new_adj[b] ^= (1 << a)
        new_deg[a] -= 1
        new_deg[b] -= 1
        new_adj[c] |= (1 << d)
        new_adj[d] |= (1 << c)
        new_deg[c] += 1
        new_deg[d] += 1
        return new_adj, new_deg
    
    def make_seed_k6_10():
        """K_{6,10} + 5 internal edges in the 10-vertex part."""
        edges = set()
        for u in range(6):
            for v in range(6, 16):
                edges.add((u, v))
        internal_edges = [(6, 7), (8, 9), (10, 11), (12, 13), (14, 15)]
        for u, v in internal_edges:
            edges.add((u, v))
        return edges
    
    def make_seed_k8_8():
        """K_{8,8} + one internal edge."""
        edges = set()
        for u in range(8):
            for v in range(8, 16):
                edges.add((u, v))
        edges.add((0, 1))
        return edges
    
    def make_seed_random(seed_val):
        """Random graph with 65 edges."""
        rng = random.Random(seed_val)
        edges = set()
        while len(edges) < target_edges:
            u, v = rng.choice(all_pairs)
            edges.add((u, v))
        return edges
    
    def local_search(edge_set, max_iter=50000, seed_val=42):
        """Greedy local search with single edge swaps."""
        rng = random.Random(seed_val)
        current_es = set(edge_set)
        current_adj, current_deg = build_adj(current_es)
        current_score = score_from_adj(current_adj, current_deg)
        best_es = set(current_es)
        best_score = current_score
        no_improve = 0
        
        for _ in range(max_iter):
            if time.time() > deadline - 5:
                break
            if no_improve > 3000:
                break
            
            e_remove = rng.choice(tuple(current_es))
            non_edges = [p for p in all_pairs if p not in current_es]
            if not non_edges:
                break
            e_add = rng.choice(non_edges)
            
            new_adj, new_deg = apply_swap(current_adj, current_deg, e_remove, e_add)
            new_score = score_from_adj(new_adj, new_deg)
            
            if new_score < current_score:
                current_es.discard(e_remove)
                current_es.add(e_add)
                current_adj = new_adj
                current_deg = new_deg
                current_score = new_score
                no_improve = 0
                if current_score < best_score:
                    best_score = current_score
                    best_es = set(current_es)
            else:
                no_improve += 1
        
        return best_es, best_score
    
    def simulated_annealing(edge_set, max_iter=100000, seed_val=123):
        """Simulated annealing with single edge swaps."""
        rng = random.Random(seed_val)
        current_es = set(edge_set)
        current_adj, current_deg = build_adj(current_es)
        current_score = score_from_adj(current_adj, current_deg)
        best_es = set(current_es)
        best_score = current_score
        
        T = 4.0
        T_min = 0.05
        cooling = 0.9997
        
        for _ in range(max_iter):
            if time.time() > deadline - 5:
                break
            if T < T_min:
                break
            
            e_remove = rng.choice(tuple(current_es))
            non_edges = [p for p in all_pairs if p not in current_es]
            if not non_edges:
                break
            e_add = rng.choice(non_edges)
            
            new_adj, new_deg = apply_swap(current_adj, current_deg, e_remove, e_add)
            new_score = score_from_adj(new_adj, new_deg)
            
            delta = new_score - current_score
            if delta < 0 or (T > 0 and rng.random() < math.exp(-delta / T)):
                current_es.discard(e_remove)
                current_es.add(e_add)
                current_adj = new_adj
                current_deg = new_deg
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_es = set(current_es)
            
            T *= cooling
        
        return best_es, best_score
    
    def final_polish(edge_set, max_rounds=10):
        """Exhaustive 1-swap polish until no improvement."""
        current_es = set(edge_set)
        current_score = compute_score(current_es)
        
        for _ in range(max_rounds):
            if time.time() > deadline - 3:
                break
            improved = False
            for e_remove in tuple(current_es):
                if time.time() > deadline - 3:
                    break
                for e_add in all_pairs:
                    if e_add in current_es:
                        continue
                    test_es = set(current_es)
                    test_es.discard(e_remove)
                    test_es.add(e_add)
                    test_score = compute_score(test_es)
                    if test_score < current_score:
                        current_es = test_es
                        current_score = test_score
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        
        return current_es, current_score
    
    # Multi-restart search
    best_es = None
    best_score = float('inf')
    
    # Phase 1: Local search from multiple seeds
    seeds = [
        make_seed_k6_10(),
        make_seed_k8_8(),
        make_seed_random(42),
        make_seed_random(123),
        make_seed_random(777),
    ]
    
    for i, seed_es in enumerate(seeds):
        if time.time() > deadline - 30:
            break
        es, score = local_search(seed_es, max_iter=30000, seed_val=42 + i)
        if score < best_score:
            best_score = score
            best_es = es
    
    # Phase 2: Simulated annealing from best found
    if time.time() < deadline - 20:
        es, score = simulated_annealing(best_es, max_iter=150000, seed_val=123)
        if score < best_score:
            best_score = score
            best_es = es
    
    # Phase 3: Another SA with different parameters
    if time.time() < deadline - 15:
        es, score = simulated_annealing(best_es, max_iter=100000, seed_val=999)
        if score < best_score:
            best_score = score
            best_es = es
    
    # Phase 4: Final polish
    if time.time() < deadline - 5:
        es, score = final_polish(best_es, max_rounds=5)
        if score < best_score:
            best_score = score
            best_es = es
    
    # Phase 5: One more local search pass from polished result
    if time.time() < deadline - 3:
        es, score = local_search(best_es, max_iter=10000, seed_val=555)
        if score < best_score:
            best_score = score
            best_es = es
    
    # Phase 6: Final polish again
    if time.time() < deadline - 2:
        es, score = final_polish(best_es, max_rounds=3)
        if score < best_score:
            best_score = score
            best_es = es
    
    # Convert to required format
    edges = [[u, v] for u, v in sorted(best_es)]
    
    return {'n': n, 'edges': edges}
# EVOLVE-BLOCK-END
