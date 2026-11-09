"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    n = 128
    
    # Precompute all 4-term APs as index arrays
    ap_indices = []
    for d in range(1, n):
        max_a = n - 3 * d
        for a in range(0, max_a):
            ap_indices.append((a, a + d, a + 2 * d, a + 3 * d))
    
    ap_indices = np.array(ap_indices, dtype=np.int32)
    
    def count_violations(colors_arr):
        """Count 4-term APs that use < 3 distinct colors using numpy."""
        c0 = colors_arr[ap_indices[:, 0]]
        c1 = colors_arr[ap_indices[:, 1]]
        c2 = colors_arr[ap_indices[:, 2]]
        c3 = colors_arr[ap_indices[:, 3]]
        
        # A 4-term AP is valid if it uses >= 3 distinct colors
        # Invalid if: all same, or exactly 2 colors
        # All same: c0==c1==c2==c3
        all_same = (c0 == c1) & (c1 == c2) & (c2 == c3)
        
        # Exactly 2 colors: not all same, but only 2 distinct values
        # This means the multiset has exactly 2 distinct values
        # Patterns: 3-1, 2-2
        # 3-1: three are one color, one is another
        # 2-2: two are one color, two are another
        
        # Check if exactly 2 distinct colors:
        # Sort the 4 values and check
        # Or: the number of distinct values is exactly 2
        
        # Efficient check: 
        # distinct_count == 2 means:
        # (c0==c1 and c2==c3 and c0!=c2) or
        # (c0==c2 and c1==c3 and c0!=c1) or
        # (c0==c3 and c1==c2 and c0!=c1) or
        # (c0==c1 and c0!=c2 and c2==c3) -- already covered
        # Let me think differently
        
        # Number of distinct colors in {c0,c1,c2,c3}:
        # = 4 - (number of pairs that are equal, adjusted)
        # Actually, let's just compute:
        # distinct = len(set) for each, but that's slow
        
        # Better: 
        # min_val = min(c0,c1,c2,c3), max_val = max(c0,c1,c2,c3)
        # If min == max: 1 color (all same)
        # Else if max - min == 1 and all values in {min, min+1}: 2 colors
        # But colors are arbitrary integers, not consecutive
        
        # Let me use a different approach:
        # A set has exactly 2 elements iff:
        # There exist two values a, b such that all c_i are in {a, b}
        
        # Efficient numpy approach:
        # Create a 2D array of the 4 values per AP
        vals = np.column_stack([c0, c1, c2, c3])
        
        # For each row, count distinct values
        # This is tricky in numpy. Let me use a different method.
        
        # Alternative: check if all 4 are in a set of size <= 2
        # For each AP, the number of distinct colors is:
        # 1 if all equal
        # 2 if exactly two distinct values
        # 3 if exactly three distinct values  
        # 4 if all four distinct
        
        # We want to count those with < 3, i.e., 1 or 2.
        
        # 1 color: all_same (computed above)
        
        # 2 colors: not all_same, but only 2 distinct values
        # This happens when:
        # The 4 values can be partitioned into 2 groups of equal values
        # Patterns: (a,a,a,b), (a,a,b,b), (a,b,a,a), etc.
        
        # A clean way: 
        # distinct_2 = (number of distinct values == 2)
        # We can check: sort the 4 values, then check if there are exactly 2 distinct
        
        # Let me just do it element-wise for speed in Python if needed,
        # but let's try a numpy trick:
        
        # For each row, compute:
        # is_2_color = (not all_same) and (there exist i,j with vals[i]!=vals[j]) and 
        #              (all values are in {vals[i], vals[j]})
        
        # Simpler: 
        # min_val = vals.min(axis=1)
        # max_val = vals.max(axis=1)
        # If min_val == max_val: 1 color
        # Else: check if all values are either min_val or max_val
        # If yes: 2 colors
        # If no: 3 or 4 colors
        
        min_val = vals.min(axis=1)
        max_val = vals.max(axis=1)
        
        # 1 color
        one_color = (min_val == max_val)
        
        # 2 colors: min_val < max_val and all values are min_val or max_val
        in_range = (vals == min_val[:, None]) | (vals == max_val[:, None])
        all_in_range = in_range.all(axis=1)
        two_colors = (~one_color) & all_in_range
        
        return int(np.sum(one_color | two_colors))
    
    # Try to find valid coloring with minimum colors
    best_result = None
    best_num_colors = n + 1
    
    for max_colors in range(3, 15):
        if time.time() > deadline - 15:
            break
        
        found_valid = False
        num_restarts = 30
        
        for restart in range(num_restarts):
            if time.time() > deadline - 10:
                break
            
            colors = np.random.randint(0, max_colors, size=n, dtype=np.int32)
            
            # Simulated annealing
            current_v = count_violations(colors)
            best_colors = colors.copy()
            best_v = current_v
            
            T = 2.0
            cooling = 0.9995
            max_iters = 3000
            
            for it in range(max_iters):
                if time.time() > deadline - 10:
                    break
                if current_v == 0:
                    break
                
                pos = random.randint(0, n - 1)
                old_color = colors[pos]
                new_color = random.randint(0, max_colors - 1)
                if new_color == old_color:
                    continue
                
                colors[pos] = new_color
                new_v = count_violations(colors)
                delta = new_v - current_v
                
                if delta < 0 or random.random() < np.exp(-delta / max(T, 0.01)):
                    current_v = new_v
                    if current_v < best_v:
                        best_v = current_v
                        best_colors = colors.copy()
                else:
                    colors[pos] = old_color
                
                T *= cooling
            
            if best_v == 0:
                found_valid = True
                if max_colors < best_num_colors:
                    best_num_colors = max_colors
                    best_result = best_colors.tolist()
                break
        
        if found_valid:
            break
    
    if best_result is not None:
        return best_result
    
    # Fallback: use a construction that's guaranteed valid
    # Use many colors: color each position uniquely in a pattern that avoids violations
    # Actually, let's try a simple valid construction with more colors
    # Color by x mod k for large k
    for k in range(5, 20):
        if time.time() > deadline - 3:
            break
        colors = np.array([i % k for i in range(n)], dtype=np.int32)
        v = count_violations(colors)
        if v == 0:
            return colors.tolist()
    
    # Last resort: all distinct
    return list(range(n))
# EVOLVE-BLOCK-END
