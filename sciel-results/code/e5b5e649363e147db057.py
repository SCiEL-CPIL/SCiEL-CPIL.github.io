import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 10s margin for safety
    
    n = 128
    # Precompute all non-constant four-term progressions
    # (a, a+d, a+2d, a+3d) with 0-based indices
    aps = []
    ap_indices = [[] for _ in range(n)]
    max_d = (n - 1) // 3
    for d in range(1, max_d + 1):
        for a in range(n - 3 * d):
            idx = len(aps)
            b = a + d
            c = a + 2 * d
            dd = a + 3 * d
            aps.append((a, b, c, dd))
            ap_indices[a].append(idx)
            ap_indices[b].append(idx)
            ap_indices[c].append(idx)
            ap_indices[dd].append(idx)
    
    # Convert to NumPy arrays for vectorized operations
    aps_arr = np.array(aps, dtype=np.int32)
    n_aps = len(aps_arr)
    
    # For each position, precompute which APs it appears in and at which index (0,1,2,3)
    pos_ap_info = [[] for _ in range(n)]  # List of (ap_idx, slot) tuples
    for idx, (a, b, c, d) in enumerate(aps):
        pos_ap_info[a].append((idx, 0))
        pos_ap_info[b].append((idx, 1))
        pos_ap_info[c].append((idx, 2))
        pos_ap_info[d].append((idx, 3))
    
    # Convert to arrays
    pos_ap_indices = [np.array([x[0] for x in info], dtype=np.int32) for info in pos_ap_info]
    pos_ap_slots = [np.array([x[1] for x in info], dtype=np.int32) for info in pos_ap_info]
    
    def count_bad(coloring):
        """Count number of bad 4-term progressions using vectorized NumPy."""
        c = np.array(coloring, dtype=np.int32)
        # Get colors for each position in each AP
        a_colors = c[aps_arr[:, 0]]
        b_colors = c[aps_arr[:, 1]]
        c_colors = c[aps_arr[:, 2]]
        d_colors = c[aps_arr[:, 3]]
        
        # An AP is bad if it uses < 3 distinct colors
        # This happens when all 4 are the same, or 3 are one color and 1 is another, or 2 and 2
        # More precisely: bad if the set of colors has size 1 or 2
        
        # Count distinct colors per AP
        # Two colors are equal if diff is 0
        # For each AP, count how many pairs are equal
        # Actually, let's think differently:
        # An AP is bad iff NOT (at least 3 distinct colors)
        # At least 3 distinct means: not all same, not exactly 2 distinct
        
        # Let's check: is it the case that at most 2 distinct colors?
        # This is equivalent to: there exist colors x, y such that all 4 positions are in {x, y}
        
        # Simpler: count how many distinct colors
        # c0, c1, c2, c3
        # distinct = 4 - (number of equal pairs) / something... no that's not right
        
        # Let's just check directly:
        # All same: a==b==c==d
        # Exactly 2 distinct: various patterns
        
        # Most efficient: check if the set has size <= 2
        # size 1: all equal
        # size 2: not all equal, but only 2 values
        
        all_same = (a_colors == b_colors) & (b_colors == c_colors) & (c_colors == d_colors)
        
        # For size 2: at least one pair differs, but no three distinct values
        # This is equivalent to: the four values can be covered by 2 colors
        # i.e., there's no triple of distinct values
        
        # Check if any three are distinct:
        # (a,b,c) all distinct OR (a,b,d) all distinct OR (a,c,d) all distinct OR (b,c,d) all distinct
        # If none of these, then at most 2 distinct
        
        abc_distinct = (a_colors != b_colors) & (a_colors != c_colors) & (b_colors != c_colors)
        abd_distinct = (a_colors != b_colors) & (a_colors != d_colors) & (b_colors != d_colors)
        acd_distinct = (a_colors != c_colors) & (a_colors != d_colors) & (c_colors != d_colors)
        bcd_distinct = (b_colors != c_colors) & (b_colors != d_colors) & (c_colors != d_colors)
        
        has_three_distinct = abc_distinct | abd_distinct | acd_distinct | bcd_distinct
        
        # Bad if NOT has_three_distinct
        bad = ~has_three_distinct
        return np.sum(bad)
    
    def get_bad_mask(coloring):
        """Get boolean mask of bad progressions."""
        c = np.array(coloring, dtype=np.int32)
        a_colors = c[aps_arr[:, 0]]
        b_colors = c[aps_arr[:, 1]]
        c_colors = c[aps_arr[:, 2]]
        d_colors = c[aps_arr[:, 3]]
        
        abc_distinct = (a_colors != b_colors) & (a_colors != c_colors) & (b_colors != c_colors)
        abd_distinct = (a_colors != b_colors) & (a_colors != d_colors) & (b_colors != d_colors)
        acd_distinct = (a_colors != c_colors) & (a_colors != d_colors) & (c_colors != d_colors)
        bcd_distinct = (b_colors != c_colors) & (b_colors != d_colors) & (c_colors != d_colors)
        
        has_three_distinct = abc_distinct | abd_distinct | acd_distinct | bcd_distinct
        return ~has_three_distinct
    
    # Try to find a valid coloring with K colors
    for K in range(4, 10):
        if time.time() > deadline:
            break
        
        found = None
        best_bad_count = float('inf')
        best_coloring = None
        
        # Multiple restarts for local search
        for restart in range(100):
            if time.time() > deadline:
                break
            
            # Initialize random coloring
            c = [random.randint(0, K - 1) for _ in range(n)]
            
            # Count initial bad progressions
            bad_count = count_bad(c)
            if bad_count == 0:
                found = c
                break
            
            # Local search: minimize number of bad progressions
            current_bad = bad_count
            no_improve = 0
            max_no_improve = 500
            
            for step in range(30000):
                if time.time() > deadline:
                    break
                
                if current_bad == 0:
                    found = c
                    break
                
                # Pick a random position to change
                pos = random.randint(0, n - 1)
                old_color = c[pos]
                
                # Try all possible new colors and pick the one that minimizes bad count
                best_new_color = old_color
                best_new_bad = current_bad
                
                for new_color in range(K):
                    if new_color == old_color:
                        continue
                    
                    c[pos] = new_color
                    new_bad = count_bad(c)
                    c[pos] = old_color
                    
                    if new_bad < best_new_bad:
                        best_new_bad = new_bad
                        best_new_color = new_color
                
                if best_new_color == old_color:
                    # No improvement found, try a random move
                    no_improve += 1
                    if no_improve > max_no_improve:
                        break
                    
                    # Random move to escape local minimum
                    new_color = random.randint(0, K - 1)
                    if new_color == old_color:
                        continue
                    c[pos] = new_color
                    current_bad = count_bad(c)
                    continue
                
                # Apply the best move
                c[pos] = best_new_color
                current_bad = best_new_bad
                no_improve = 0
                
                # Update best
                if current_bad < best_bad_count:
                    best_bad_count = current_bad
                    best_coloring = c[:]
            
            if found:
                break
        
        if found:
            return found
        
        # If no valid coloring found with K colors, keep best for fallback
        if best_coloring is not None and best_bad_count == 0:
            return best_coloring

    # Fallback: Seed program logic (greedy)
    # This guarantees a valid solution, though potentially with more colors
    colors = []
    for x in range(1, 129):
        for color in range(128):
            valid = True
            # Check 4-APs ending at x
            for d in range(1, (x - 1) // 3 + 1):
                if x - 1 - 3 * d < 0:
                    continue
                c0 = colors[x - 1 - 3 * d]
                c1 = colors[x - 1 - 2 * d]
                c2 = colors[x - 1 - d]
                c3 = color
                if len({c0, c1, c2, c3}) < 3:
                    valid = False
                    break
            
            if not valid:
                continue
                
            # Check the "no 3 identical" heuristic from seed
            for d in range(1, (x - 1) // 2 + 1):
                if x - 1 - 2 * d < 0:
                    continue
                if colors[x - 1 - d] == color and colors[x - 1 - 2 * d] == color:
                    valid = False
                    break
            
            if valid:
                colors.append(color)
                break
        else:
            colors.append(0)
            
    return colors
