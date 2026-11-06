"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 145.0  # safety margin before 150s kill
    n = 18
    random.seed(42)
    
    # Bitmask-based adjacency for speed
    # adj_mask[v] is a bitmask of neighbors of v
    
    def adj_to_masks(adj_set):
        masks = [0] * n
        for edge in adj_set:
            u, v = tuple(edge)
            masks[u] |= (1 << v)
            masks[v] |= (1 << u)
        return masks
    
    def masks_to_adj(masks):
        adj = set()
        for u in range(n):
            for v in range(u+1, n):
                if masks[u] & (1 << v):
                    adj.add(frozenset([u, v]))
        return adj
    
    def is_k4_free_masks(masks):
        # Check no K4 using bitmasks
        for a in range(n):
            for b in range(a+1, n):
                if not (masks[a] & (1 << b)):
                    continue
                common_ab = masks[a] & masks[b]
                for c in range(b+1, n):
                    if not (common_ab & (1 << c)):
                        continue
                    common_abc = common_ab & masks[c]
                    # Check if any d > c is in common_abc
                    remaining = common_abc & ~((1 << (c+1)) - 1)
                    if remaining:
                        return False
        return True
    
    def max_triangle_free_induced_masks(masks):
        # Find largest triangle-free induced subgraph using bitmask branch and bound
        # A vertex v can be added to set S iff v's neighbors in S form an independent set
        # Precompute triangle conflicts: for each pair (u,v) that is an edge,
        # any w adjacent to both u and v creates a triangle
        
        best = [0]
        
        # Precompute: for each vertex v, the set of vertices that conflict with v
        # when both are in a set that also contains a common neighbor
        # Actually, let's use a different approach:
        # For a partial set S (bitmask), a vertex v can be added iff
        # (masks[v] & S) is an independent set, i.e., no edge exists within (masks[v] & S)
        
        def is_independent(mask):
            # Check if vertices in mask form an independent set
            m = mask
            while m:
                v = (m & -m).bit_length() - 1
                m &= m - 1
                if masks[v] & m:
                    return False
            return True
        
        def can_add(S, v):
            # v can be added to S iff neighbors of v in S are independent
            nbrs_in_S = masks[v] & S
            return is_independent(nbrs_in_S)
        
        def search(S, candidates_mask, depth):
            nonlocal deadline
            if time.time() > deadline - 5:
                return
            pop_count = bin(candidates_mask).count('1')
            if pop_count + bin(S).count('1') <= best[0]:
                return
            if candidates_mask == 0:
                sz = bin(S).count('1')
                if sz > best[0]:
                    best[0] = sz
                return
            
            # Pick the lowest set bit in candidates
            v_bit = candidates_mask & -candidates_mask
            v = v_bit.bit_length() - 1
            remaining = candidates_mask ^ v_bit
            
            # Try including v
            if can_add(S, v):
                new_S = S | v_bit
                # Filter candidates: only those that can be added to new_S
                new_cands = 0
                m = remaining
                while m:
                    w_bit = m & -m
                    w = w_bit.bit_length() - 1
                    if can_add(new_S, w):
                        new_cands |= w_bit
                    m ^= w_bit
                search(new_S, new_cands, depth + 1)
            
            # Try excluding v
            search(S, remaining, depth + 1)
        
        search(0, (1 << n) - 1, 0)
        return best[0]
    
    def random_greedy_k4_free():
        # Build a random K4-free graph greedily
        # Start empty, add edges randomly, reject if creates K4
        masks = [0] * n
        all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
        random.shuffle(all_edges)
        for u, v in all_edges:
            # Check if adding edge (u,v) creates a K4
            # A K4 containing (u,v) needs two common neighbors w,x of u and v that are also adjacent
            common = masks[u] & masks[v]
            # Check if common contains an edge
            conflict = False
            m = common
            while m:
                w_bit = m & -m
                w = w_bit.bit_length() - 1
                if masks[w] & (m ^ w_bit):
                    conflict = True
                    break
                m ^= w_bit
            if not conflict:
                masks[u] |= (1 << v)
                masks[v] |= (1 << u)
        return masks
    
    def random_greedy_with_density():
        # Try to build denser K4-free graphs
        # Use a priority: prefer edges that don't create K4
        masks = [0] * n
        all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
        random.shuffle(all_edges)
        for u, v in all_edges:
            common = masks[u] & masks[v]
            conflict = False
            m = common
            while m:
                w_bit = m & -m
                w = w_bit.bit_length() - 1
                if masks[w] & (m ^ w_bit):
                    conflict = True
                    break
                m ^= w_bit
            if not conflict:
                masks[u] |= (1 << v)
                masks[v] |= (1 << u)
        return masks
    
    # Generate multiple random starting points
    best_masks = None
    best_score = n + 1
    
    # First try the tripartite K6,6,6 as a baseline
    tri_masks = [0] * n
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    for i in range(3):
        for j in range(i + 1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    tri_masks[a] |= (1 << b)
                    tri_masks[b] |= (1 << a)
    
    score = max_triangle_free_induced_masks(tri_masks)
    if score < best_score:
        best_score = score
        best_masks = tri_masks[:]
    
    # Try random greedy constructions
    attempts = 0
    while time.time() < deadline - 30:
        attempts += 1
        masks = random_greedy_with_density()
        if not is_k4_free_masks(masks):
            continue
        score = max_triangle_free_induced_masks(masks)
        if score < best_score:
            best_score = score
            best_masks = masks[:]
            # Local search from this point
            while time.time() < deadline - 15:
                improved = False
                all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
                random.shuffle(all_edges)
                for u, v in all_edges:
                    if time.time() > deadline - 15:
                        break
                    # Try adding edge
                    if not (masks[u] & (1 << v)):
                        # Check K4
                        common = masks[u] & masks[v]
                        conflict = False
                        m = common
                        while m:
                            w_bit = m & -m
                            w = w_bit.bit_length() - 1
                            if masks[w] & (m ^ w_bit):
                                conflict = True
                                break
                            m ^= w_bit
                        if not conflict:
                            masks[u] |= (1 << v)
                            masks[v] |= (1 << u)
                            new_score = max_triangle_free_induced_masks(masks)
                            if new_score < best_score:
                                best_score = new_score
                                best_masks = masks[:]
                                improved = True
                            else:
                                masks[u] ^= (1 << v)
                                masks[v] ^= (1 << u)
                    # Try removing edge
                    else:
                        masks[u] ^= (1 << v)
                        masks[v] ^= (1 << u)
                        new_score = max_triangle_free_induced_masks(masks)
                        if new_score < best_score:
                            best_score = new_score
                            best_masks = masks[:]
                            improved = True
                        else:
                            masks[u] |= (1 << v)
                            masks[v] |= (1 << u)
                if not improved:
                    break
    
    # Convert best_masks to edge list
    result = []
    for u in range(n):
        for v in range(u+1, n):
            if best_masks[u] & (1 << v):
                result.append([u, v])
    return result
# EVOLVE-BLOCK-END
