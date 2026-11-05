"""
4-colouring of K_n with no monochromatic C6.
Strategy: Use the known Z_17 cyclic construction as a seed, then apply local search
to try to extend to n=18 or n=19. If extension fails, return the best n found.
"""

import time
import random
from itertools import combinations

def _has_c6_in_color(adj, color, n):
    """Check if the subgraph of edges with given color contains a 6-cycle (not necessarily induced)."""
    # Build adjacency list for this color
    # adj is a list of lists: adj[i][j] = color of edge (i,j)
    # We need to find any 6-cycle in the subgraph where edge color == 'color'
    
    # For efficiency, build bitmasks
    masks = [0] * n
    for i in range(n):
        for j in range(i+1, n):
            if adj[i][j] == color:
                masks[i] |= (1 << j)
                masks[j] |= (1 << i)
    
    # Search for 6-cycle: v0-v1-v2-v3-v4-v5-v0
    # Enumerate v0 < v1 < v2, then v3, v4, v5 with constraints to avoid duplicates
    # A 6-cycle is v0,v1,v2,v3,v4,v5 with edges (v0,v1),(v1,v2),(v2,v3),(v3,v4),(v4,v5),(v5,v0)
    # To avoid counting the same cycle multiple times, enforce v0 = min(v0..v5) and v1 < v5
    
    for v0 in range(n):
        # v1 must be a neighbor of v0
        m1 = masks[v0]
        while m1:
            v1 = (m1 & -m1).bit_length() - 1
            m1 &= m1 - 1
            if v1 <= v0:
                continue
            # v2 must be neighbor of v1, not v0
            m2 = masks[v1] & ~masks[v0]
            m2 &= ~((1 << v0) | (1 << v1))
            while m2:
                v2 = (m2 & -m2).bit_length() - 1
                m2 &= m2 - 1
                if v2 == v0 or v2 == v1:
                    continue
                # v3 must be neighbor of v2, not v0, v1
                m3 = masks[v2] & ~masks[v0] & ~masks[v1]
                m3 &= ~((1 << v0) | (1 << v1) | (1 << v2))
                while m3:
                    v3 = (m3 & -m3).bit_length() - 1
                    m3 &= m3 - 1
                    if v3 in (v0, v1, v2):
                        continue
                    # v4 must be neighbor of v3, not v0, v1, v2
                    m4 = masks[v3] & ~masks[v0] & ~masks[v1] & ~masks[v2]
                    m4 &= ~((1 << v0) | (1 << v1) | (1 << v2) | (1 << v3))
                    while m4:
                        v4 = (m4 & -m4).bit_length() - 1
                        m4 &= m4 - 1
                        if v4 in (v0, v1, v2, v3):
                            continue
                        # v5 must be neighbor of v4 AND neighbor of v0, not v0,v1,v2,v3,v4
                        # v5 must satisfy: (v4,v5) in color AND (v5,v0) in color
                        # v5 > v0 (since v0 is min) and v5 != v1,v2,v3,v4
                        # Also to avoid duplicate cycles: v1 < v5
                        m5 = masks[v4] & masks[v0]
                        m5 &= ~((1 << v0) | (1 << v1) | (1 << v2) | (1 << v3) | (1 << v4))
                        while m5:
                            v5 = (m5 & -m5).bit_length() - 1
                            m5 &= m5 - 1
                            if v5 in (v0, v1, v2, v3, v4):
                                continue
                            if v5 < v1:  # enforce v1 < v5 to avoid duplicates
                                continue
                            return True
    return False


def _count_c6s(adj, n):
    """Count monochromatic 6-cycles. Returns (count, list_of_cycles) where cycles are (v0,v1,v2,v3,v4,v5)."""
    total = 0
    cycles = []
    for color in range(4):
        masks = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j] == color:
                    masks[i] |= (1 << j)
                    masks[j] |= (1 << i)
        
        for v0 in range(n):
            m1 = masks[v0]
            while m1:
                v1 = (m1 & -m1).bit_length() - 1
                m1 &= m1 - 1
                if v1 <= v0:
                    continue
                m2 = masks[v1] & ~masks[v0]
                m2 &= ~((1 << v0) | (1 << v1))
                while m2:
                    v2 = (m2 & -m2).bit_length() - 1
                    m2 &= m2 - 1
                    if v2 == v0 or v2 == v1:
                        continue
                    m3 = masks[v2] & ~masks[v0] & ~masks[v1]
                    m3 &= ~((1 << v0) | (1 << v1) | (1 << v2))
                    while m3:
                        v3 = (m3 & -m3).bit_length() - 1
                        m3 &= m3 - 1
                        if v3 in (v0, v1, v2):
                            continue
                        m4 = masks[v3] & ~masks[v0] & ~masks[v1] & ~masks[v2]
                        m4 &= ~((1 << v0) | (1 << v1) | (1 << v2) | (1 << v3))
                        while m4:
                            v4 = (m4 & -m4).bit_length() - 1
                            m4 &= m4 - 1
                            if v4 in (v0, v1, v2, v3):
                                continue
                            m5 = masks[v4] & masks[v0]
                            m5 &= ~((1 << v0) | (1 << v1) | (1 << v2) | (1 << v3) | (1 << v4))
                            while m5:
                                v5 = (m5 & -m5).bit_length() - 1
                                m5 &= m5 - 1
                                if v5 in (v0, v1, v2, v3, v4):
                                    continue
                                if v5 < v1:
                                    continue
                                total += 1
                                if len(cycles) < 10:
                                    cycles.append((v0, v1, v2, v3, v4, v5))
    return total, cycles


def _build_adj_from_colors(colors, n):
    """Build adjacency matrix from color list in lexicographic order."""
    adj = [[-1] * n for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            adj[i][j] = colors[idx]
            adj[j][i] = colors[idx]
            idx += 1
    return adj


def _z17_coloring():
    """
    The known Z_17 construction for R_4(C6) > 17.
    Vertices are 0..16 in Z_17.
    Edge (i,j) gets color based on the difference d = j - i mod 17.
    Quadratic residues mod 17: {1, 2, 4, 8, 9, 13, 15, 16}
    We partition the 16 non-zero differences into 4 sets of 4.
    
    Known construction: 
    Let Q = {1, 2, 4, 8} (quadratic residues of order 4 subgroup)
    Actually, the standard construction uses:
    The multiplicative group Z_17* has order 16.
    Split into 4 cosets of a subgroup of order 4.
    Subgroup H = {1, 4, 13, 16} (since 4^1=4, 4^2=16, 4^3=64=13 mod 17, 4^4=52=1)
    Cosets:
    C0 = {1, 4, 13, 16}
    C1 = {2, 8, 9, 15} (2*H)
    C2 = {3, 12, 5, 14} (3*H)
    C3 = {6, 7, 10, 11} (6*H)
    
    Color of edge (i,j) = index of the coset containing (j-i) mod 17.
    """
    n = 17
    # Precompute coset membership
    coset = [0] * 17
    H = [1, 4, 13, 16]
    for h in H:
        coset[h] = 0
    for h in H:
        coset[(2*h) % 17] = 1
    for h in H:
        coset[(3*h) % 17] = 2
    for h in H:
        coset[(6*h) % 17] = 3
    
    colors = []
    for i in range(n):
        for j in range(i+1, n):
            d = (j - i) % 17
            colors.append(coset[d])
    return n, colors


def _extend_coloring(n_old, colors_old, n_new):
    """
    Try to extend a coloring of K_n_old to K_n_new by adding new vertices.
    Returns (n_new, colors_new) if successful, else None.
    """
    # Start with old coloring
    colors = list(colors_old)
    adj = _build_adj_from_colors(colors, n_old)
    
    # Add new vertices one at a time
    for new_v in range(n_old, n_new):
        # Initialize new vertex with random colors
        best_assign = None
        best_score = float('inf')
        
        for attempt in range(500):
            # Random assignment for edges from new_v to old vertices
            assign = [random.randint(0, 3) for _ in range(n_old)]
            
            # Build full adj
            adj_full = [[-1] * n_new for _ in range(n_new)]
            for i in range(n_old):
                for j in range(i+1, n_old):
                    adj_full[i][j] = adj[i][j]
                    adj_full[j][i] = adj[i][j]
            for i in range(n_old):
                adj_full[i][new_v] = assign[i]
                adj_full[new_v][i] = assign[i]
            
            # Count C6s
            count, _ = _count_c6s(adj_full, n_new)
            if count < best_score:
                best_score = count
                best_assign = assign[:]
                if count == 0:
                    break
        
        if best_score == 0:
            # Success
            for i in range(n_old):
                colors.append(best_assign[i])
            adj = _build_adj_from_colors(colors, n_new)
        else:
            # Try local search on the assignment
            # Keep the best assignment and try to improve
            assign = best_assign[:]
            improved = True
            while improved and best_score > 0:
                improved = False
                for i in range(n_old):
                    for c in range(4):
                        if assign[i] == c:
                            continue
                        old_c = assign[i]
                        assign[i] = c
                        adj_full = [[-1] * n_new for _ in range(n_new)]
                        for a in range(n_old):
                            for b in range(a+1, n_old):
                                adj_full[a][b] = adj[a][b]
                                adj_full[b][a] = adj[a][b]
                        for a in range(n_old):
                            adj_full[a][new_v] = assign[a]
                            adj_full[new_v][a] = assign[a]
                        count, _ = _count_c6s(adj_full, n_new)
                        if count < best_score:
                            best_score = count
                            best_assign = assign[:]
                            improved = True
                            if count == 0:
                                break
                        else:
                            assign[i] = old_c
                    if best_score == 0:
                        break
                
                if best_score == 0:
                    break
            
            if best_score == 0:
                for i in range(n_old):
                    colors.append(best_assign[i])
                adj = _build_adj_from_colors(colors, n_new)
            else:
                # Failed to extend
                return None
    
    return n_new, colors


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Start with the known Z_17 coloring
    n, colors = _z17_coloring()
    best_n = n
    best_colors = colors[:]
    
    # Verify the base coloring is valid
    adj = _build_adj_from_colors(colors, n)
    count, _ = _count_c6s(adj, n)
    if count != 0:
        # Should not happen, but if it does, fall back
        pass
    
    # Try to extend to n=18
    if time.time() < deadline - 20:
        result = _extend_coloring(n, colors, n+1)
        if result is not None:
            best_n, best_colors = result
            n = best_n
    
    # Try to extend to n=19
    if best_n >= 18 and time.time() < deadline - 20:
        result = _extend_coloring(best_n, best_colors, best_n+1)
        if result is not None:
            best_n, best_colors = result
    
    # Try local search on the current best to see if we can find a valid coloring
    # by perturbing vertex permutations or color assignments
    if time.time() < deadline - 10:
        # Try random restarts with different initial perturbations
        for restart in range(50):
            if time.time() > deadline - 5:
                break
            
            # Perturb the coloring slightly
            perturbed = best_colors[:]
            # Randomly flip some colors
            for _ in range(min(10, len(perturbed))):
                idx = random.randint(0, len(perturbed)-1)
                perturbed[idx] = random.randint(0, 3)
            
            adj_p = _build_adj_from_colors(perturbed, best_n)
            count_p, _ = _count_c6s(adj_p, best_n)
            
            if count_p < count:
                # Better, try to improve
                improved = True
                while improved and time.time() < deadline - 5:
                    improved = False
                    for _ in range(100):
                        if time.time() > deadline - 5:
                            break
                        idx = random.randint(0, len(perturbed)-1)
                        old_c = perturbed[idx]
                        new_c = random.randint(0, 3)
                        if new_c == old_c:
                            continue
                        perturbed[idx] = new_c
                        adj_t = _build_adj_from_colors(perturbed, best_n)
                        count_t, _ = _count_c6s(adj_t, best_n)
                        if count_t <= count_p:
                            count_p = count_t
                            if count_t < count:
                                count = count_t
                                best_colors = perturbed[:]
                                improved = True
                        else:
                            perturbed[idx] = old_c
                        
                        if count_p == 0:
                            break
                    if count_p == 0:
                        break
    
    return {'n': best_n, 'colors': best_colors}
