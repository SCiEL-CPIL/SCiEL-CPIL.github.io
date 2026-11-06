"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import numpy as np

def construct():
    deadline = time.time() + 140  # Safety margin before 150s limit
    n = 16
    # Precompute all K4s (quads) and their edge indices for fast validation
    # Edge (i, j) with i < j has index idx = i*(n-1) - i*(i+1)//2 + (j-i-1)
    # Or simpler: just use a 2D array for colors.
    
    quads = list(itertools.combinations(range(n), 4))
    
    def count_colors(coloring):
        return max(coloring) + 1
    
    def build_ec(coloring):
        """Build edge color lookup as a 2D array."""
        ec = np.zeros((n, n), dtype=np.int32)
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                ec[i, j] = coloring[idx]
                ec[j, i] = coloring[idx]
                idx += 1
        return ec
    
    def is_valid_fast(coloring):
        """Fast validation using a 2D array."""
        ec = build_ec(coloring)
        # For each quad, check if the 6 edges have at least 5 distinct colors
        for a, b, c, d in quads:
            # Get the 6 colors
            c1 = ec[a, b]
            c2 = ec[a, c]
            c3 = ec[a, d]
            c4 = ec[b, c]
            c5 = ec[b, d]
            c6 = ec[c, d]
            
            # Check distinctness efficiently
            # If any 3 are same, fail? No, just need < 5 distinct.
            # A set of 6 items has < 5 distinct elements iff there is a duplicate.
            # Actually, < 5 distinct means at most 4 distinct.
            # This happens if there are at least 2 duplicates in the 6 edges.
            # i.e., the number of distinct values is 4 or 3 or 2 or 1.
            
            # Quick check:
            if c1 == c2:
                # We have a duplicate. Need to ensure we don't end up with < 5 distinct.
                # If c1==c2, we have 5 other edges. If any of those 5 is also c1, we have 2 duplicates of c1.
                # If c3==c1 or c4==c1 etc, we might still have 5 distinct if the other 4 are all different and not c1.
                # Example: 1,1,2,3,4,5 -> distinct {1,2,3,4,5} size 5. OK.
                # Example: 1,1,2,2,3,4 -> distinct {1,2,3,4} size 4. FAIL.
                # So we need to count distinct values properly.
                pass
            
            # Just use a set for simplicity, it's fast enough for 1820 quads * 120 candidates
            colors = {c1, c2, c3, c4, c5, c6}
            if len(colors) < 5:
                return False
        return True

    def make_coloring_formula(formula_fn):
        colors = []
        for i in range(n):
            for j in range(i+1, n):
                colors.append(formula_fn(i, j))
        return colors

    candidates = []
    
    # 1. Try linear forms (a*i + b*j) % m
    # We want to minimize m (number of colors).
    # Try m from 13 to 17.
    for m in range(13, 18):
        for a in range(1, 8):
            for b in range(1, 8):
                if a == b:
                    continue
                # Ensure a and b are not both 0 mod m (they are >= 1)
                cand = make_coloring_formula(lambda i, j, a=a, b=b, m=m: (a * i + b * j) % m)
                candidates.append(cand)
                
    # 2. Try (i + j) % m specifically, as it's a strong candidate
    for m in range(13, 18):
        cand = make_coloring_formula(lambda i, j, m=m: (i + j) % m)
        candidates.append(cand)
        
    # 3. Try (i - j) based? No, must be symmetric.
    # (i+j) is symmetric. (i*j) is symmetric.
    # Try (i*j) % m
    for m in range(13, 18):
        cand = make_coloring_formula(lambda i, j, m=m: (i * j) % m)
        candidates.append(cand)
        
    # 4. Try (i*i + j*j) % m
    for m in range(13, 18):
        cand = make_coloring_formula(lambda i, j, m=m: (i*i + j*j) % m)
        candidates.append(cand)

    # Sort candidates by number of colors used (ascending) to find minimum faster
    candidates.sort(key=lambda c: count_colors(c))
    
    best = None
    best_num_colors = float('inf')
    
    for cand in candidates:
        if time.time() > deadline:
            break
        nc = count_colors(cand)
        if nc >= best_num_colors:
            break  # No need to check candidates with more colors
        if is_valid_fast(cand):
            best_num_colors = nc
            best = cand
    
    if best is not None:
        return best
    
    # Fallback: return a known valid coloring with 17 colors
    # (i + j) % 17
    return make_coloring_formula(lambda i, j: (i + j) % 17)
    
# EVOLVE-BLOCK-END
