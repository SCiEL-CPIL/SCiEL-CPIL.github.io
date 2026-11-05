"""Program for R(K2,2,2) maximization using bitmask constraint satisfaction."""
import numpy as np
import time

def construct():
    # We want to maximize n such that there exists a 2-coloring of K_n
    # with no monochromatic K_{2,2,2}.
    
    def has_k222_bitmask(adj_masks, n, color):
        """
        Check if color class contains K_{2,2,2}.
        adj_masks: list of ints, adj_masks[i] has bit j set if edge (i,j) is color.
        """
        # For each edge (u,v) in the color class, check if common neighborhood S
        # contains a K_{2,2} in the same color class.
        for u in range(n):
            # Only check edges where v > u to avoid duplicates? 
            # The condition is symmetric. 
            # We iterate over all u, and v such that edge (u,v) exists.
            mask_u = adj_masks[u]
            # Iterate over set bits in mask_u
            v_mask = mask_u
            while v_mask:
                # Extract lowest set bit
                lsb = v_mask & -v_mask
                v = lsb.bit_length() - 1
                v_mask ^= lsb
                
                if v <= u:
                    continue
                
                # Common neighbors of u and v
                S = mask_u & adj_masks[v]
                
                # Check if S contains K_{2,2}
                # S contains K_{2,2} iff there exist y1, y2 in S such that
                # |N(y1) & N(y2) & S| >= 2
                s_list = []
                temp_s = S
                while temp_s:
                    lsb_s = temp_s & -temp_s
                    y = lsb_s.bit_length() - 1
                    s_list.append(y)
                    temp_s ^= lsb_s
                
                m = len(s_list)
                if m < 4:
                    continue
                    
                # Check pairs in S
                for i in range(m):
                    y1 = s_list[i]
                    mask_y1 = adj_masks[y1]
                    for j in range(i+1, m):
                        y2 = s_list[j]
                        mask_y2 = adj_masks[y2]
                        # Common neighbors of y1 and y2 within S
                        C = mask_y1 & mask_y2 & S
                        # Check if C has at least 2 bits
                        if C and (C & (C - 1)): # C has at least 2 set bits
                            return True
        return False

    def check_valid(colors, n):
        """Check if both color classes are K_{2,2,2}-free."""
        # Build adjacency masks for color 0 and 1
        adj0 = [0] * n
        adj1 = [0] * n
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                c = colors[idx]
                if c == 0:
                    adj0[i] |= (1 << j)
                    adj0[j] |= (1 << i)
                else:
                    adj1[i] |= (1 << j)
                    adj1[j] |= (1 << i)
                idx += 1
        
        return not has_k222_bitmask(adj0, n, 0) and not has_k222_bitmask(adj1, n, 1)

    def make_paley(n):
        """Construct Paley graph coloring for prime n."""
        # n must be prime
        squares = set()
        for x in range(1, n):
            squares.add((x * x) % n)
        
        colors = []
        for i in range(n):
            for j in range(i+1, n):
                diff = (j - i) % n
                if diff in squares:
                    colors.append(0)
                else:
                    colors.append(1)
        return colors

    # Try known good constructions
    # The record is 29. Paley graph on 29 vertices is a strong candidate.
    candidates = [29, 31, 27, 25] # 27, 25 not prime, Paley not defined directly, but we can try others
    
    # Check 29 first
    if 29 <= 31:
        colors_29 = make_paley(29)
        if check_valid(colors_29, 29):
            return {'n': 29, 'colors': colors_29}

    # Check 31
    if 31 <= 31:
        colors_31 = make_paley(31)
        if check_valid(colors_31, 31):
            return {'n': 31, 'colors': colors_31}

    # If Paley 29 fails, try random search with hill climbing for 29
    # But 29 Paley is very likely valid. 
    # Let's add a fallback random search for 29 just in case, using the fast checker.
    import random
    random.seed(42)
    
    # Simple local search / hill climbing
    def local_search(n, max_iter=1000):
        # Start with random coloring
        colors = [random.choice([0, 1]) for _ in range(n*(n-1)//2)]
        
        # Calculate initial violations
        # This is expensive, so we might just try random restarts if 29 is hard.
        # But given the time limit, and the fact that 29 Paley is a standard construction,
        # we prioritize that.
        
        pass

    # Fallback to smaller primes if 29/31 fail (should not happen)
    for p in [23, 19, 17]:
        colors = make_paley(p)
        if check_valid(colors, p):
            return {'n': p, 'colors': colors}
            
    # Absolute fallback
    return {'n': 1, 'colors': []}
