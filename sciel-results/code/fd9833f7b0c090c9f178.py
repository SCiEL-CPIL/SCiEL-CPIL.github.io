"""Spectral optimization approach for K4-free graphs on 18 vertices."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def construct():
    deadline = time.time() + 115.0  # safety margin before 120s kill
    n = 18
    random.seed(42)
    np.random.seed(42)
    
    def is_k4_free(masks):
        # Check no K4 using bitmasks
        for a in range(n):
            for b in range(a+1, n):
                if not (masks[a] >> b) & 1:
                    continue
                common = masks[a] & masks[b]
                common &= ~(1 << a)
                common &= ~(1 << b)
                c_bits = common
                while c_bits:
                    c = c_bits & (-c_bits)
                    c_idx = c.bit_length() - 1
                    c_bits ^= c
                    if (masks[c_idx] & common & ~(c - 1) & ~c) != 0:
                        return False
        return True

    def get_triangles(masks):
        """Return list of triangle masks."""
        triangles = set()
        for a in range(n):
            for b in range(a+1, n):
                if (masks[a] >> b) & 1:
                    common = masks[a] & masks[b]
                    common &= ~(1 << a)
                    common &= ~(1 << b)
                    c_bits = common
                    while c_bits:
                        c = c_bits & (-c_bits)
                        c_idx = c.bit_length() - 1
                        c_bits ^= c
                        if (masks[c_idx] >> b) & 1: # Check edge c-b
                            tri = (1 << a) | (1 << b) | (1 << c_idx)
                            triangles.add(tri)
        return list(triangles)

    def min_transversal_size_fast(triangles, n):
        """
        Calculate the minimum number of vertices to hit all triangles.
        Uses subset enumeration with bitmask optimization.
        """
        if not triangles:
            return 0
        
        T = len(triangles)
        all_tri = (1 << T) - 1
        
        # Map triangle mask to index
        tri_indices = {t: i for i, t in enumerate(triangles)}
        
        # hit_mask[v] : bitset of triangles containing v
        hit_mask = [0] * n
        for t in triangles:
            bits = t
            while bits:
                b = bits & (-bits)
                v = b.bit_length() - 1
                bits ^= b
                idx = tri_indices[t]
                hit_mask[v] |= (1 << idx)
        
        # We want min k such that exists S of size k with OR(hit_mask[v] for v in S) == all_tri
        for k in range(1, n+1):
            for S in combinations(range(n), k):
                acc = 0
                for v in S:
                    acc |= hit_mask[v]
                    if acc == all_tri:
                        return k
        return n

    def max_triangle_free_induced(masks):
        triangles = get_triangles(masks)
        tau = min_transversal_size_fast(triangles, n)
        return n - tau

    def spectral_optimize():
        # Start with Turan graph T(18, 3)
        # Parts: 0-5, 6-11, 12-17
        masks = [0] * n
        for i in range(n):
            part_i = i // 6
            for j in range(n):
                if j > i:
                    part_j = j // 6
                    if part_i != part_j:
                        masks[i] |= (1 << j)
                        masks[j] |= (1 << i)
        
        current_score = max_triangle_free_induced(masks)
        best_score = current_score
        best_masks = masks[:]
        
        temp = 1.0
        cooling = 0.99
        iterations = 0
        
        while time.time() < deadline - 20 and temp > 0.01:
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            
            # Flip edge
            masks[u] ^= (1 << v)
            masks[v] ^= (1 << u)
            
            if is_k4_free(masks):
                new_score = max_triangle_free_induced(masks)
                delta = new_score - current_score
                if delta <= 0 or random.random() < np.exp(-delta / temp):
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_masks = masks[:]
                else:
                    # Revert
                    masks[u] ^= (1 << v)
                    masks[v] ^= (1 << u)
            else:
                # Revert
                masks[u] ^= (1 << v)
                masks[v] ^= (1 << u)
            
            temp *= cooling
            iterations += 1
            
        return best_masks, best_score

    best_masks, best_score = spectral_optimize()
    
    # Convert to edge list
    result = []
    for u in range(n):
        for v in range(u+1, n):
            if (best_masks[u] >> v) & 1:
                result.append([u, v])
    return result
# EVOLVE-BLOCK-END
