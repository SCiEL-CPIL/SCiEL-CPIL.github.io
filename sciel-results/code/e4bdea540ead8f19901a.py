import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    n = 20
    vertices = list(range(n))
    
    # Precompute triples for validation
    triples = list(combinations(vertices, 3))
    
    # --- Algebraic Construction: Complement of 2-(20, 4, 1) design ---
    # The biplane on 20 points has parameters (v, k, lambda) = (20, 4, 1).
    # It has b = 20 blocks. Each block has 4 points.
    # The incidence graph is bipartite with 20+20 vertices.
    # But we need a graph on 20 vertices.
    # The complement of the block design's point graph?
    # Actually, a known K3,3-free graph on 20 vertices is the complement of the Clebsch graph? No, Clebsch is 16.
    # Let's use a known construction: The graph derived from the biplane.
    # Vertices are the 20 points.
    # Two points are adjacent if they are in the SAME block.
    # Since lambda=1, any pair of points is in exactly one block.
    # So if we connect points in the same block, every pair is connected. That's K20. Bad.
    
    # Alternative: Vertices are the 20 blocks.
    # Two blocks are adjacent if they intersect in exactly 1 point.
    # In a 2-(20,4,1) design, any two blocks intersect in 0 or 1 point.
    # Number of blocks = 20.
    # Let's check if this graph is K3,3-free.
    # This is the "block intersection graph".
    # Actually, a simpler base: Start with a random regular graph of degree 6-7 and repair.
    # But the directive suggests a specific structure.
    # Let's try a different algebraic base: The polarity graph of F_19 is not undirected.
    # What about the graph where u~v if (v-u) is a cube in F_19?
    # Cubes in F_19: Since 19-1=18, gcd(3,18)=3.
    # Cubes are {x^3 mod 19 | x in 1..18}.
    # x^3 mod 19:
    # 1->1, 2->8, 3->27=8, 4->64=7, 5->125=11, 6->216=7, 7->343=1, 8->512=17, 9->729=11
    # 10->1000=12, 11->1331=17, 12->1728=12, 13->2197=8, 14->2744=1, 15->3375=7, 16->4096=11, 17->4913=17, 18->5832=12
    # Unique cubes: {1, 7, 8, 11, 12, 17}. Size 6.
    # -1 = 18. Is 18 a cube? No.
    # So u~v if (v-u) in Cubes is not symmetric.
    # Symmetrize: u~v if (v-u) in Cubes OR (u-v) in Cubes.
    # Cubes = {1, 7, 8, 11, 12, 17}. -Cubes = {18, 12, 11, 8, 7, 2}.
    # Intersection: {7, 8, 11, 12}. Union: {1, 2, 7, 8, 11, 12, 17, 18}. Size 8.
    # So degree is 8. Total edges 20*8/2 = 80.
    # Is it K3,3-free?
    # Let's test this base.
    
    def is_k33_free(adj_mask):
        for t in triples:
            u, v, w = t
            common = adj_mask[u] & adj_mask[v] & adj_mask[w]
            if common.bit_count() > 2:
                return False
        return True

    def edge_count(adj_mask):
        return sum(m.bit_count() for m in adj_mask) // 2

    def mask_to_edges(adj_mask):
        edges = []
        for i in range(n):
            m = adj_mask[i]
            for j in range(i + 1, n):
                if m & (1 << j):
                    edges.append([i, j])
        return edges

    # Precompute cubes
    cubes = set()
    for x in range(1, 19):
        cubes.add((x * x * x) % 19)
    # Symmetrize: D = Cubes union -Cubes
    D = set()
    for d in cubes:
        D.add(d)
        D.add((-d) % 19)
    D.discard(0)
    
    # Build base graph
    base_adj = [0] * n
    for u in range(19):
        for v in range(19):
            if u != v and (v - u) % 19 in D:
                base_adj[u] |= (1 << v)
                base_adj[v] |= (1 << u)
    # Vertex 19: Connect to all? Or randomly?
    # Let's connect 19 to vertices 0..8 (first 9) to start.
    for v in range(9):
        base_adj[19] |= (1 << v)
        base_adj[v] |= (1 << 19)
        
    # Check validity of base
    if not is_k33_free(base_adj):
        # Repair base if invalid
        # Remove edges causing violations
        for t in triples:
            u, v, w = t
            common = base_adj[u] & base_adj[v] & base_adj[w]
            if common.bit_count() > 2:
                # Remove one common neighbor
                # Pick the first one
                bits = common & ~(1 << u) & ~(1 << v) & ~(1 << w)
                if bits:
                    lsb = bits & -bits
                    k = lsb.bit_length() - 1
                    base_adj[u] &= ~(1 << k)
                    base_adj[k] &= ~(1 << u)
                    base_adj[v] &= ~(1 << k)
                    base_adj[k] &= ~(1 << v)
                    base_adj[w] &= ~(1 << k)
                    base_adj[k] &= ~(1 << w)
    
    best_adj = list(base_adj)
    best_edges = edge_count(best_adj)
    
    # Simulated Annealing
    # State: adj_mask
    # Move: Add or remove a random edge
    # Constraint: Must remain K3,3-free
    
    def can_add(u, v, adj_mask):
        """
        Check if adding edge (u, v) keeps the graph K3,3-free.
        Only checks triples containing u or v.
        """
        bit_v = 1 << v
        bit_u = 1 << u
        new_u = adj_mask[u] | bit_v
        new_v = adj_mask[v] | bit_u
        
        # Check triples containing u
        for t in triples:
            a, b, c = t
            if u not in (a, b, c) and v not in (a, b, c):
                continue
                
            ma = adj_mask[a]
            mb = adj_mask[b]
            mc = adj_mask[c]
            
            if a == u: ma = new_u
            elif a == v: ma = new_v
            if b == u: mb = new_u
            elif b == v: mb = new_v
            if c == u: mc = new_u
            elif c == v: mc = new_v
            
            if (ma & mb & mc).bit_count() > 2:
                return False
        return True

    def can_remove(u, v, adj_mask):
        """
        Removing an edge always preserves K3,3-freeness.
        """
        return True

    # Precompute list of all pairs
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Simulated Annealing Parameters
    T = 1.0
    cooling = 0.999
    min_T = 0.01
    
    adj = list(base_adj)
    
    while time.time() < deadline and T > min_T:
        # Pick a random move
        u, v = random.choice(all_pairs)
        is_edge = bool(adj[u] & (1 << v))
        
        if is_edge:
            # Try to remove
            # Always valid
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            new_edges = edge_count(adj)
            delta = new_edges - best_edges
            
            # Accept if better, or with probability exp(delta/T) if worse
            if delta > 0:
                best_adj = list(adj)
                best_edges = new_edges
            else:
                if random.random() < np.exp(delta / T):
                    pass # Accept
                else:
                    # Revert
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        else:
            # Try to add
            if can_add(u, v, adj):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                new_edges = edge_count(adj)
                delta = new_edges - best_edges
                
                if delta > 0:
                    best_adj = list(adj)
                    best_edges = new_edges
                else:
                    if random.random() < np.exp(delta / T):
                        pass # Accept
                    else:
                        # Revert
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
            else:
                # Cannot add, try to remove a random edge to escape?
                # Or just skip.
                pass
        
        T *= cooling
        
        # Occasionally try a "greedy add" to fill gaps
        if random.random() < 0.1:
            for u, v in all_pairs:
                if not (adj[u] & (1 << v)):
                    if can_add(u, v, adj):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
            new_edges = edge_count(adj)
            if new_edges > best_edges:
                best_adj = list(adj)
                best_edges = new_edges

    # Final greedy add
    for u, v in all_pairs:
        if not (best_adj[u] & (1 << v)):
            if can_add(u, v, best_adj):
                best_adj[u] |= (1 << v)
                best_adj[v] |= (1 << u)
    
    edges = mask_to_edges(best_adj)
    return edges
