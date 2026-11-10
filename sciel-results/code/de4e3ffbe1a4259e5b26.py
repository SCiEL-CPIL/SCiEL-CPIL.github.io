"""Rewritten program: construct largest pairwise non-coprime subset of [1,255] containing 255."""
import time
import math
from math import gcd

# EVOLVE-BLOCK-START
def construct():
    """
    Find the maximum subset S ⊆ [1, 255] containing 255 such that gcd(a,b) > 1 
    for all distinct pairs a,b in S.
    
    Uses Tomita-style maximum clique with bitmask adjacency for speed.
    """
    
    start_time = time.time()
    deadline = start_time + 50.0  # 10s safety margin before 60s limit
    
    # 1. Generate candidates C = {x in [1,255] : gcd(x,255) > 1}
    C = [x for x in range(1, 256) if gcd(x, 255) > 1]
    
    # 2. V = C \ {255}
    V = [x for x in C if x != 255]
    n = len(V)
    
    # Build bitmask adjacency: adj_mask[i] has bit j set iff gcd(V[i],V[j])>1
    adj_mask = [0] * n
    for i in range(n):
        vi = V[i]
        mask = 0
        for j in range(n):
            if i != j and gcd(vi, V[j]) > 1:
                mask |= (1 << j)
        adj_mask[i] = mask
    
    # 3. Tomita maximum clique with bitmasks
    best_clique = []
    best_size = 0
    time_check_counter = 0
    
    def color_sort_bm(cand_mask):
        """
        Greedy coloring on bitmask candidate set.
        Returns list of (vertex_idx, color) sorted by color descending.
        """
        if cand_mask == 0:
            return []
        
        # Get vertex list from bitmask
        verts = []
        m = cand_mask
        while m:
            lb = m & (-m)
            idx = lb.bit_length() - 1
            verts.append(idx)
            m ^= lb
        
        # Sort by degree within candidates (descending) for better coloring
        verts.sort(key=lambda v: bin(adj_mask[v] & cand_mask).count('1'), reverse=True)
        
        color_of = {}
        for v in verts:
            # Find colors used by already-colored neighbors
            neighbor_colors = set()
            nm = adj_mask[v] & cand_mask
            while nm:
                lb = nm & (-nm)
                u = lb.bit_length() - 1
                if u in color_of:
                    neighbor_colors.add(color_of[u])
                nm ^= lb
            c = 1
            while c in neighbor_colors:
                c += 1
            color_of[v] = c
        
        # Sort by color descending
        sorted_verts = sorted(verts, key=lambda v: color_of[v], reverse=True)
        return [(v, color_of[v]) for v in sorted_verts]
    
    def expand(clique, cand_mask):
        nonlocal best_clique, best_size, time_check_counter
        
        if cand_mask == 0:
            if len(clique) > best_size:
                best_size = len(clique)
                best_clique = clique[:]
            return
        
        # Periodic time check
        time_check_counter += 1
        if time_check_counter % 50 == 0:
            if time.time() > deadline:
                return
        
        # Color-based upper bound
        colored = color_sort_bm(cand_mask)
        if not colored:
            return
        
        # Max color gives upper bound on extension
        max_color = colored[-1][1]
        if len(clique) + max_color <= best_size:
            return
        
        # Process vertices in descending color order
        for v, c in colored:
            if len(clique) + c <= best_size:
                break
            
            new_clique = clique + [v]
            new_cand = cand_mask & adj_mask[v]
            expand(new_clique, new_cand)
            cand_mask ^= (1 << v)  # remove v from candidates
    
    # Start search
    full_mask = (1 << n) - 1
    expand([], full_mask)
    
    # 4. Construct final set
    result_values = [V[i] for i in best_clique]
    result_values.append(255)
    
    # 5. Verification
    s = set(result_values)
    assert 255 in s
    assert len(s) == len(result_values)
    for a in s:
        assert 1 <= a <= 255
    for i in range(len(result_values)):
        for j in range(i + 1, len(result_values)):
            assert gcd(result_values[i], result_values[j]) > 1
            
    return result_values
# EVOLVE-BLOCK-END
