"""
Dense Q3-free graph on 16 vertices.

Approach:
1. Precompute all forbidden hyperedges: each Q3 corresponds to a 12-tuple of edge-indices
   (positions in a canonical edge list). A graph (edge set S) is Q3-free iff S contains no such tuple.
2. Use a greedy + local search hybrid:
   - Greedy: add edges in a randomized order, skipping any that create a Q3.
   - Local search: try edge additions (if valid), edge removals, and edge swaps.
3. Use bit-parallel Q3 detection for speed.
"""
import time
import random
import math
from itertools import combinations

# Precompute the edge positions for each Q3 embedding
# Q3 has 12 edges. We list all 12 edges as pairs (a,b) where a < b.
# We'll use a fixed labeling of Q3 vertices.
# Vertices: 0..7
# Edges:
Q3_EDGES = [
    (0, 1), (0, 2), (0, 4),
    (1, 3), (1, 5),
    (2, 3), (2, 6),
    (3, 7),
    (4, 5), (4, 6),
    (5, 7),
    (6, 7)
]

def construct():
    deadline = time.time() + 115
    n = 16
    full_mask = (1 << n) - 1
    
    # List of all possible edges (u, v) with u < v
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    m = len(all_edges)
    
    # Map edge to index
    edge_idx = {}
    for i, (u, v) in enumerate(all_edges):
        edge_idx[(u, v)] = i
    
    # Precompute all Q3 hyperedges: each is a frozenset of 12 edge indices
    # We need to check all 8-subsets of vertices and all Q3 embeddings on them.
    # C(16, 8) = 12870 subsets. For each, number of Q3 embeddings is 8! / 48 = 1344.
    # 12870 * 1344 ~ 17 million. This might be too slow to precompute fully in Python.
    # Let's only precompute a smaller set: all Q3s on a fixed 8-vertex set, then use isomorphism?
    # No, we need exact edge indices.
    
    # Alternative: precompute all 12-edge subsets that form a Q3 by iterating over all 8-subsets.
    # But 17M is too much. Let's sample or use a smarter method.
    
    # Actually, let's just precompute all Q3s on the 16 vertices by iterating over all 8-subsets
    # and for each, checking if the induced subgraph is isomorphic to Q3. But that's also heavy.
    
    # Let's use a different approach: 
    # Precompute for each 8-subset S, whether it contains a Q3. If yes, record the edge set.
    # But we need the exact 12 edges.
    
    # Let's just generate all Q3s by choosing 8 vertices and mapping Q3 to them.
    # We'll do this lazily or with a smaller precomputation.
    
    # Given time constraints, let's use a fast Q3 detection function instead of full precomputation.
    
    def creates_q3(u, v, adj):
        """Check if adding edge (u,v) creates a Q3. adj is list of bitmasks."""
        N_u = adj[u] & ~(1 << v)
        N_v = adj[v] & ~(1 << u)
        
        cand_u = []
        mask = N_u
        while mask:
            bit = mask & (-mask)
            idx = bit.bit_length() - 1
            cand_u.append(idx)
            mask -= bit
            
        cand_v = []
        mask = N_v
        while mask:
            bit = mask & (-mask)
            idx = bit.bit_length() - 1
            cand_v.append(idx)
            mask -= bit
        
        for x in cand_u:
            for y in cand_u:
                if x == y:
                    continue
                for z in cand_v:
                    for w in cand_v:
                        if z == w or z in (x, y) or w in (x, y):
                            continue
                        if not (adj[x] & (1 << z)):
                            continue
                        if not (adj[y] & (1 << w)):
                            continue
                        c6 = (adj[x] & adj[y]) & ~(1 << u) & ~(1 << v) & ~(1 << x) & ~(1 << y) & ~(1 << z) & ~(1 << w)
                        if not c6:
                            continue
                        c7 = (adj[z] & adj[w]) & ~(1 << u) & ~(1 << v) & ~(1 << x) & ~(1 << y) & ~(1 << z) & ~(1 << w)
                        if not c7:
                            continue
                        c6_bits = c6
                        while c6_bits:
                            bit6 = c6_bits & (-c6_bits)
                            v6 = bit6.bit_length() - 1
                            c6_bits -= bit6
                            c7_check = c7 & ~(1 << v6)
                            if c7_check and (adj[v6] & c7_check):
                                return True
        return False
    
    def count_edges(adj):
        count = 0
        for u in range(n):
            count += adj[u].bit_count()
        return count // 2
    
    def edges_from_adj(adj):
        result = []
        for u in range(n):
            mask = adj[u]
            while mask:
                bit = mask & (-mask)
                v = bit.bit_length() - 1
                if v > u:
                    result.append([u, v])
                mask -= bit
        return result
    
    # Local search with edge swaps
    def local_search(adj, steps=10000, deadline=None):
        best_adj = adj[:]
        best_count = count_edges(adj)
        current_adj = adj[:]
        current_count = best_count
        T = 1.0
        cooling = 0.9995
        
        for i in range(steps):
            if deadline and time.time() > deadline - 5:
                break
            
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
                
            is_present = bool(current_adj[u] & (1 << v))
            
            if is_present:
                # Remove edge
                trial_adj = current_adj[:]
                trial_adj[u] &= ~(1 << v)
                trial_adj[v] &= ~(1 << u)
                new_count = current_count - 1
                if new_count > current_count or random.random() < math.exp(-1.0 / T):
                    current_adj = trial_adj
                    current_count = new_count
            else:
                # Try add edge
                if creates_q3(u, v, current_adj):
                    continue
                trial_adj = current_adj[:]
                trial_adj[u] |= (1 << v)
                trial_adj[v] |= (1 << u)
                new_count = current_count + 1
                current_adj = trial_adj
                current_count = new_count
                if current_count > best_count:
                    best_count = current_count
                    best_adj = current_adj[:]
                    
            T *= cooling
        
        return best_adj, best_count
    
    # Edge swap: remove (u1,v1) and add (u2,v2)
    def edge_swap_search(adj, steps=5000, deadline=None):
        best_adj = adj[:]
        best_count = count_edges(adj)
        current_adj = adj[:]
        current_count = best_count
        
        for i in range(steps):
            if deadline and time.time() > deadline - 5:
                break
            
            # Pick an existing edge to remove
            u1 = random.randint(0, n - 1)
            v1 = random.randint(0, n - 1)
            if u1 == v1 or not (current_adj[u1] & (1 << v1)):
                continue
            if u1 > v1:
                u1, v1 = v1, u1
            
            # Pick a missing edge to add
            u2 = random.randint(0, n - 1)
            v2 = random.randint(0, n - 1)
            if u2 == v2 or (current_adj[u2] & (1 << v2)):
                continue
            if u2 > v2:
                u2, v2 = v2, u2
            
            trial_adj = current_adj[:]
            trial_adj[u1] &= ~(1 << v1)
            trial_adj[v1] &= ~(1 << u1)
            if creates_q3(u2, v2, trial_adj):
                continue
            trial_adj[u2] |= (1 << v2)
            trial_adj[v2] |= (1 << u2)
            
            new_count = current_count  # swap: same count
            # Accept if it doesn't reduce count (it doesn't) and random
            if random.random() < 0.3:
                current_adj = trial_adj
                current_count = new_count
                if current_count > best_count:
                    best_count = current_count
                    best_adj = current_adj[:]
        
        return best_adj, best_count
    
    random.seed(42)
    best_adj = [0] * n
    best_count = -1
    
    for seed in range(15):
        if time.time() > deadline - 15:
            break
        
        random.seed(seed)
        adj = [0] * n
        
        # Greedy init
        edge_order = all_edges[:]
        random.shuffle(edge_order)
        for u, v in edge_order:
            if time.time() > deadline - 5:
                break
            if not (adj[u] & (1 << v)) and not creates_q3(u, v, adj):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        # Local search
        adj, cnt = local_search(adj, steps=8000, deadline=deadline)
        if cnt > best_count:
            best_count = cnt
            best_adj = adj[:]
        
        # Edge swap search
        adj, cnt = edge_swap_search(adj, steps=3000, deadline=deadline)
        if cnt > best_count:
            best_count = cnt
            best_adj = adj[:]
        
        # Final polish: try adding more edges
        if time.time() < deadline - 2:
            remaining = [(u, v) for u, v in all_edges if not (best_adj[u] & (1 << v))]
            random.shuffle(remaining)
            for u, v in remaining:
                if time.time() > deadline - 1:
                    break
                if not (best_adj[u] & (1 << v)) and not creates_q3(u, v, best_adj):
                    best_adj[u] |= (1 << v)
                    best_adj[v] |= (1 << u)
                    best_count += 1
    
    return edges_from_adj(best_adj)
