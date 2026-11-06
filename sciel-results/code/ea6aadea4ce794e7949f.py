import time
import numpy as np
import sys

N = 16
TOTAL_POINTS = N * N
ALL_MASK = (1 << TOTAL_POINTS) - 1

# Global variables for precomputation
IsoMask = None
dist2 = None

def _precompute():
    global IsoMask, dist2
    if IsoMask is not None:
        return
        
    xs = np.arange(N)
    ys = np.arange(N)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    pts = np.stack([X.flatten(), Y.flatten()], axis=1)
    
    # Compute squared distances
    # dist2[i, j]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist2 = np.sum(diff**2, axis=2)
    
    IsoMask = [[0] * TOTAL_POINTS for _ in range(TOTAL_POINTS)]
    
    for i in range(TOTAL_POINTS):
        for j in range(TOTAL_POINTS):
            if i == j:
                continue
            d_ij = dist2[i, j]
            # Condition 1: dist(i, k) == dist(i, j)
            c1 = (dist2[i, :] == d_ij)
            # Condition 2: dist(j, k) == dist(i, j)
            c2 = (dist2[j, :] == d_ij)
            # Condition 3: dist(i, k) == dist(j, k)
            c3 = (dist2[i, :] == dist2[j, :])
            
            valid_k = c1 | c2 | c3
            valid_k[i] = False
            valid_k[j] = False
            
            mask = 0
            # Efficiently convert boolean array to bitmask
            # Using np.nonzero
            idxs = np.nonzero(valid_k)[0]
            for k in idxs:
                mask |= (1 << k)
            IsoMask[i][j] = mask
            # Symmetry: IsoMask[j][i] is the same set of k?
            # The set of k such that {i,j,k} is isosceles is symmetric in i,j.
            # So IsoMask[j][i] should be the same mask.
            # We can just assign it to save time.
            IsoMask[j][i] = mask

def construct():
    start_time = time.time()
    deadline = start_time + 290
    
    _precompute()
    
    best_size = TOTAL_POINTS + 1
    best_set_mask = 0
    
    # Helper to check time
    def time_up():
        return time.time() > deadline

    def search(S_mask, Covered_mask, Available_mask, depth):
        nonlocal best_size, best_set_mask
        
        if time_up():
            return

        # Check if valid maximal independent set
        if (S_mask | Covered_mask) == ALL_MASK:
            if depth < best_size:
                best_size = depth
                best_set_mask = S_mask
                # We could return True to prune, but we want to find smaller ones.
                # However, if we found a solution, we only need to search for depth < best_size.
                # The pruning below handles this.
            return

        # Pruning: If current depth is already >= best, no point continuing
        if depth >= best_size:
            return

        # If no points available to add, but not all covered, dead end
        if Available_mask == 0:
            return

        # Lower Bound Pruning
        # Uncovered points
        Uncovered = ALL_MASK & ~(S_mask | Covered_mask)
        # If Uncovered is not empty, we need more points.
        # Each new point covers at most MAX_COVER points.
        # We can use a static upper bound on coverage per point to estimate min additional points.
        # But a dynamic bound is better.
        # For now, simple bound: if depth + 1 > best_size, prune (already handled).
        
        # Generate candidates
        candidates = []
        avail = Available_mask
        while avail:
            p_bit = avail & -avail
            p = p_bit.bit_length() - 1
            avail ^= p_bit
            
            # Calculate NewCover for this p
            new_cover = 0
            s_iter = S_mask
            while s_iter:
                a_bit = s_iter & -s_iter
                a = a_bit.bit_length() - 1
                s_iter ^= a_bit
                new_cover |= IsoMask[p][a]
                
            # Points actually newly covered (not already covered)
            newly_covered = new_cover & ~Covered_mask
            # We care about covering Uncovered points
            gain = newly_covered & Uncovered
            gain_count = bin(gain).count('1')
            
            candidates.append((gain_count, p))
            
        if not candidates:
            return
            
        # Sort by gain (descending)
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        # To handle symmetry or diversify, we could randomize ties,
        # but greedy is usually strong for Set Cover-like problems.
        
        for gain_count, p in candidates:
            p_bit = 1 << p
            S_new = S_mask | p_bit
            
            # Recalculate NewCover (we did it above, but we need it for the recursive call state)
            # To avoid recomputation, we could store it, but it's cheap enough for small S.
            new_cover = 0
            s_iter = S_mask
            while s_iter:
                a_bit = s_iter & -s_iter
                a = a_bit.bit_length() - 1
                s_iter ^= a_bit
                new_cover |= IsoMask[p][a]
                
            Covered_new = Covered_mask | new_cover
            Available_new = ALL_MASK & ~(S_new | Covered_new)
            
            search(S_new, Covered_new, Available_new, depth + 1)
            
            # If we found a very good solution, we might break early?
            # No, we need to check if a smaller one exists.
            # But if we found size K, and we are exploring depth K, 
            # we know we can't beat K with depth >= K.
            # The pruning `depth >= best_size` handles this.

    # Start search
    # Initial state: S=0, Covered=0, Available=ALL
    search(0, 0, ALL_MASK, 0)
    
    # Fallback if no solution found (should not happen, as any maximal set is a solution)
    if best_set_mask == 0:
        # Construct a greedy maximal set as fallback
        S = 0
        Covered = 0
        Available = ALL_MASK
        while Available:
            # Pick first available
            p_bit = Available & -Available
            p = p_bit.bit_length() - 1
            S |= p_bit
            # Update Covered
            new_cover = 0
            s_iter = S
            # Note: S includes p now. Pairs are (p, a) for a in S \ {p}
            s_iter = S & ~p_bit
            while s_iter:
                a_bit = s_iter & -s_iter
                a = a_bit.bit_length() - 1
                s_iter ^= a_bit
                new_cover |= IsoMask[p][a]
            Covered |= new_cover
            Available = ALL_MASK & ~(S | Covered)
        best_set_mask = S
        best_size = bin(S).count('1')

    # Convert mask to list of tuples
    result = []
    for i in range(TOTAL_POINTS):
        if best_set_mask & (1 << i):
            x = i // N
            y = i % N
            result.append((x, y))
            
    return result
