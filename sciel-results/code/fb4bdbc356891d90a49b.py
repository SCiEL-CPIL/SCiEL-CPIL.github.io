"""R_4(C6) - 4-colouring of K_n without monochromatic C6.
Combines Z_17 cyclic quadratic-residue construction, bit-parallel C6 counting,
and targeted local search with C6-breaking moves.
"""
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    # ---------- Z_17 construction (known best) ----------
    def z17_construction():
        """Build the known 17-vertex 4-colouring of K_17 with no mono C6.
        Uses the Z_17 cyclic group with a quadratic-residue-based edge weight.
        Returns (n, colors) where colors is in lexicographic order.
        """
        n = 17
        # In Z_17, the quadratic residues are {1,2,4,8,9,13,15,16} (8 of them)
        # and non-residues are {3,5,6,7,10,11,12,14} (8 of them), plus 0.
        # We need a 4-colouring. Use a known construction:
        # Colour edge (i,j) by f(i-j mod 17) where f maps differences to colours.
        # The standard construction uses a (17,8,4) difference set or similar.
        
        # Let's use the Paley-type construction adapted for 4 colours.
        # Actually, for R_4(C6) > 17, we need a specific known construction.
        # The standard approach: partition Z_17 into 4 sets based on the
        # multiplicative structure or use a known (17,4) design.
        
        # A known valid construction for n=17:
        # Use the fact that 17 = 4*4 + 1. The vertices are Z_17.
        # Colour c(i,j) = ((i-j) mod 17) mapped through a specific function.
        
        # Let me use a concrete known valid colouring.
        # The colour of edge (i,j) in Z_17 is determined by (i-j) mod 17.
        # We need to assign each non-zero difference d in {1..16} to a colour in {0,1,2,3}
        # such that no colour class contains a C6.
        
        # For a C6 in the colour class of difference set S, we need
        # a sequence a, a+s1, a+s1+s2, ..., a+s1+...+s5 where each si in S
        # and the sum s1+...+s5 = 0 mod 17 (to close the cycle back to a).
        
        # Let me just hardcode a known valid assignment.
        # The standard (17,8,4) difference set D = {1,2,4,8,9,13,15,16} gives a (17,8,4) BIBD.
        # For 4-colouring, we can use:
        # Colour 0: differences in D
        # But D has 8 elements, and we need 4 colours covering 16 differences.
        
        # Actually, let's use a different approach. The known result R_4(C6) > 17
        # uses a specific construction. Let me use a search-based approach
        # to find a valid colouring of Z_17, starting from a heuristic.
        
        # Heuristic: assign colour based on the "quadratic character" and another property.
        # QR in Z_17: {1,2,4,8,9,13,15,16}
        # NQR: {3,5,6,7,10,11,12,14}
        
        # Split QR into two sets and NQR into two sets:
        # QR1 = {1,2,4,8}, QR2 = {9,13,15,16}
        # NQR1 = {3,5,6,7}, NQR2 = {10,11,12,14}
        
        # Colour 0: QR1 ∪ QR2 = QR (all quadratic residues)
        # Colour 1: NQR1
        # Colour 2: NQR2
        # Colour 3: ... wait, this doesn't partition into 4 equal-ish sets.
        
        # Let me try: 
        # Colour 0: {1, 2, 4, 8}
        # Colour 1: {9, 13, 15, 16}
        # Colour 2: {3, 5, 6, 7}
        # Colour 3: {10, 11, 12, 14}
        
        # Check if each set is C6-free in Z_17.
        # A C6 exists in a difference set S if there are s1,...,s5 in S with sum = 0 mod 17.
        # Since |S|=4, we need 5 elements from S (with repetition allowed? No, the cycle
        # uses 5 edges, each with a difference in S, and the vertices must be distinct).
        
        # Actually, for a C6 in the graph with edge set {(i,j): i-j in S},
        # we need vertices v0,v1,...,v5 such that vi+1 - vi in S for i=0..4 and v0 - v5 in S.
        # So we need s0,s1,s2,s3,s4,s5 in S with s0+...+s5 = 0 mod 17 and
        # all partial sums distinct mod 17.
        
        # Let me just test this construction directly.
        
        def color_of_diff(d):
            d = d % 17
            if d == 0:
                return -1  # shouldn't happen for i != j
            if d in (1, 2, 4, 8):
                return 0
            elif d in (9, 13, 15, 16):
                return 1
            elif d in (3, 5, 6, 7):
                return 2
            else:  # 10, 11, 12, 14
                return 3
        
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                d = (i - j) % 17
                colors.append(color_of_diff(d))
        
        # Verify
        if _verify_coloring(colors, n):
            return n, colors
        
        # If the above doesn't work, try other partitions
        # Try: Colour 0: QR, Colour 1-3: split NQR
        def color_of_diff2(d):
            d = d % 17
            if d == 0:
                return -1
            if d in (1, 2, 4, 8, 9, 13, 15, 16):
                return 0
            elif d in (3, 5, 6, 7):
                return 1
            elif d in (10, 11, 12, 14):
                return 2
            else:
                return 3
        
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                d = (i - j) % 17
                colors.append(color_of_diff2(d))
        if _verify_coloring(colors, n):
            return n, colors
        
        # Fallback: random search over Z_17 colourings
        for attempt in range(200):
            if time.time() > deadline - 10:
                break
            # Random assignment of 16 differences to 4 colours
            diffs = list(range(1, 17))
            assignment = {}
            for d in diffs:
                assignment[d] = random.randint(0, 3)
            
            colors = []
            for i in range(n):
                for j in range(i + 1, n):
                    d = (i - j) % 17
                    colors.append(assignment[d])
            
            if _verify_coloring(colors, n):
                return n, colors
        
        # Last resort: return n=9 construction
        n9 = 9
        colors9 = [min(j - i, 9 - j + i) - 1 for i in range(n9) for j in range(i + 1, n9)]
        return n9, colors9
    
    # ---------- Bit-parallel C6 detection ----------
    def _verify_coloring(colors, n):
        """Check if coloring has no monochromatic C6 using bitmasks."""
        E = n * (n - 1) // 2
        if len(colors) != E:
            return False
        
        # Build adjacency bitmasks for each colour
        adj = [[0] * n for _ in range(4)]
        k = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[k]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                k += 1
        
        for c in range(4):
            if _has_c6(adj[c], n):
                return False
        return True
    
    def _has_c6(adj, n):
        """Check if graph with adjacency bitmasks has a C6."""
        full_mask = (1 << n) - 1
        for v0 in range(n):
            nbrs0 = adj[v0]
            for v1 in range(v0 + 1, n):
                if not (nbrs0 >> v1) & 1:
                    continue
                for v2 in range(v1 + 1, n):
                    if v2 == v0 or not (adj[v1] >> v2) & 1:
                        continue
                    for v3 in range(v2 + 1, n):
                        if v3 in (v0, v1, v2) or not (adj[v2] >> v3) & 1:
                            continue
                        for v4 in range(v3 + 1, n):
                            if v4 in (v0, v1, v2, v3) or not (adj[v3] >> v4) & 1:
                                continue
                            # Check if there's v5 adjacent to both v4 and v0, not in {v0..v4}
                            cands = adj[v4] & adj[v0]
                            mask = full_mask
                            for v in (v0, v1, v2, v3, v4):
                                mask &= ~(1 << v)
                            if cands & mask:
                                return True
        return False
    
    def _find_c6(adj, n):
        """Find a specific C6. Returns tuple of 6 vertices or None."""
        full_mask = (1 << n) - 1
        for v0 in range(n):
            nbrs0 = adj[v0]
            for v1 in range(v0 + 1, n):
                if not (nbrs0 >> v1) & 1:
                    continue
                for v2 in range(v1 + 1, n):
                    if v2 == v0 or not (adj[v1] >> v2) & 1:
                        continue
                    for v3 in range(v2 + 1, n):
                        if v3 in (v0, v1, v2) or not (adj[v2] >> v3) & 1:
                            continue
                        for v4 in range(v3 + 1, n):
                            if v4 in (v0, v1, v2, v3) or not (adj[v3] >> v4) & 1:
                                continue
                            cands = adj[v4] & adj[v0]
                            mask = full_mask
                            for v in (v0, v1, v2, v3, v4):
                                mask &= ~(1 << v)
                            cands &= mask
                            if cands:
                                # Find lowest set bit
                                v5 = (cands & -cands).bit_length() - 1
                                return (v0, v1, v2, v3, v4, v5)
        return None
    
    def _count_c6s(adj, n, limit=100):
        """Count C6s up to limit."""
        full_mask = (1 << n) - 1
        count = 0
        for v0 in range(n):
            nbrs0 = adj[v0]
            for v1 in range(v0 + 1, n):
                if not (nbrs0 >> v1) & 1:
                    continue
                for v2 in range(v1 + 1, n):
                    if v2 == v0 or not (adj[v1] >> v2) & 1:
                        continue
                    for v3 in range(v2 + 1, n):
                        if v3 in (v0, v1, v2) or not (adj[v2] >> v3) & 1:
                            continue
                        for v4 in range(v3 + 1, n):
                            if v4 in (v0, v1, v2, v3) or not (adj[v3] >> v4) & 1:
                                continue
                            cands = adj[v4] & adj[v0]
                            mask = full_mask
                            for v in (v0, v1, v2, v3, v4):
                                mask &= ~(1 << v)
                            cands &= mask
                            if cands:
                                count += cands.bit_count()
                                if count >= limit:
                                    return count
        return count
    
    # ---------- Local search to repair a near-valid colouring ----------
    def _local_search(colors, n, max_iters=10000, time_limit=None):
        """Local search to eliminate C6s by recoloring edges in cycles."""
        if time_limit is None:
            time_limit = deadline
        
        E = n * (n - 1) // 2
        colors = colors[:]
        
        def build_adj(colors):
            adj = [[0] * n for _ in range(4)]
            k = 0
            for i in range(n):
                for j in range(i + 1, n):
                    c = colors[k]
                    adj[c][i] |= (1 << j)
                    adj[c][j] |= (1 << i)
                    k += 1
            return adj
        
        def edge_idx(i, j):
            """Get index in lexicographic order."""
            if i > j:
                i, j = j, i
            idx = 0
            for a in range(i):
                idx += (n - 1 - a)
            idx += (j - i - 1)
            return idx
        
        adj = build_adj(colors)
        
        for it in range(max_iters):
            if time.time() > time_limit:
                break
            
            # Find a C6 in any colour class
            found_cycle = None
            found_color = -1
            for c in range(4):
                cyc = _find_c6(adj[c], n)
                if cyc is not None:
                    found_cycle = cyc
                    found_color = c
                    break
            
            if found_cycle is None:
                return colors  # Valid!
            
            # Try to break the cycle by recoloring one of its edges
            v0, v1, v2, v3, v4, v5 = found_cycle
            cycle_edges = [(v0, v1), (v1, v2), (v2, v3), (v3, v4), (v4, v5), (v5, v0)]
            
            # Shuffle edges for diversity
            random.shuffle(cycle_edges)
            
            improved = False
            for (i, j) in cycle_edges:
                if i > j:
                    i, j = j, i
                idx = edge_idx(i, j)
                old_c = colors[idx]
                
                for new_c in range(4):
                    if new_c == old_c:
                        continue
                    
                    # Tentatively recolor
                    colors[idx] = new_c
                    adj_new = build_adj(colors)
                    
                    # Check if new colour class has a C6
                    if _has_c6(adj_new[new_c], n):
                        colors[idx] = old_c
                        continue
                    
                    # Also verify old colour class is still C6-free (it should be, since we removed an edge)
                    # But let's be safe
                    if _has_c6(adj_new[old_c], n):
                        colors[idx] = old_c
                        continue
                    
                    # Check all other colour classes
                    valid = True
                    for c in range(4):
                        if c == old_c or c == new_c:
                            continue
                        if _has_c6(adj_new[c], n):
                            valid = False
                            break
                    
                    if valid:
                        adj = adj_new
                        improved = True
                        break
                    else:
                        colors[idx] = old_c
                
                if improved:
                    break
            
            if not improved:
                # Stuck: try a random perturbation
                for _ in range(3):
                    idx = random.randint(0, E - 1)
                    old_c = colors[idx]
                    new_c = random.randint(0, 3)
                    if new_c == old_c:
                        continue
                    colors[idx] = new_c
                    adj = build_adj(colors)
                    if not _has_c6(adj[new_c], n) and not _has_c6(adj[old_c], n):
                        break
        
        return colors
    
    # ---------- Main ----------
    best_n = 9
    best_colors = [min(j - i, 9 - j + i) - 1 for i in range(9) for j in range(i + 1, 9)]
    
    # First: try the Z_17 construction
    n17, colors17 = z17_construction()
    if n17 >= 17 and _verify_coloring(colors17, n17):
        best_n = n17
        best_colors = colors17
    
    # Try to find n=18 via local search starting from perturbed Z_17
    if time.time() < deadline - 15:
        for n in [18, 17]:
            if n <= best_n:
                continue
            if time.time() > deadline - 10:
                break
            
            E = n * (n - 1) // 2
            found = False
            
            for attempt in range(30):
                if time.time() > deadline - 8:
                    break
                
                # Initialize: for n=18, take Z_17 coloring and add vertex 17 with random edges
                if n == 18 and best_n == 17:
                    colors = best_colors[:]
                    # Add edges from vertex 17 to vertices 0..16
                    for i in range(17):
                        colors.append(random.randint(0, 3))
                else:
                    colors = [random.randint(0, 3) for _ in range(E)]
                
                # Local search
                result = _local_search(colors, n, max_iters=5000, time_limit=deadline - 8)
                if result is not None and _verify_coloring(result, n):
                    best_n = n
                    best_colors = result
                    found = True
                    break
            
            if found:
                break
    
    # Fallback
    if best_n < 9:
        best_n = 9
        best_colors = [min(j - i, 9 - j + i) - 1 for i in range(9) for j in range(i + 1, 9)]
    
    return {'n': best_n, 'colors': best_colors}
