import numpy as np
import time
import random

def construct():
    N = 128
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin
    
    # Precompute all 4-term APs
    aps = []
    for d in range(1, (N - 1) // 3 + 1):
        for a in range(0, N - 3 * d):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    num_aps = len(aps)
    
    # Map position to AP indices
    pos_to_aps = [[] for _ in range(N)]
    for idx, ap in enumerate(aps):
        for pos in ap:
            pos_to_aps[pos].append(idx)
    
    # Fast check: does this 4-tuple use < 3 distinct colors?
    def is_bad(c0, c1, c2, c3):
        # 1 color: all equal
        if c0 == c1 == c2 == c3:
            return True
        # 2 colors: exactly two distinct values
        # Check all pairings for exactly 2 colors
        # If c0==c1: need c2,c3 to be from {c0, one other}
        # More efficiently: check if the set has size < 3
        # But let's avoid set creation for speed
        # 2 colors means: there exist a,b such that all of c0,c1,c2,c3 are in {a,b}
        # and a != b
        # Equivalently: NOT (all 4 are same) AND (at most 2 distinct)
        
        # Let's just count distinct with a few comparisons
        # Distinct count:
        # Start with 1
        # Add c1 if c1 != c0
        # Add c2 if c2 != c0 and c2 != (c1 if c1!=c0 else c0)
        # This is getting complex. Let's use a different approach.
        
        # Check if we have at least 3 distinct colors
        # We need: NOT (c0,c1,c2,c3 use only 1 or 2 colors)
        
        # 1 color: c0==c1==c2==c3
        # 2 colors: 
        #   Case 1: c0==c1. Then we need c2 and c3 to be in {c0, x} where x is the other color.
        #           So c2==c0 or c2==c3, and c3==c0 or c3==c2, and not all same.
        #           Actually: if c0==c1, the distinct colors are from {c0, c2, c3}.
        #           For exactly 2: {c2, c3} must be a subset of {c0, y} for some y, 
        #           and not all in {c0}. So either c2==c3!=c0, or one of c2,c3 equals c0 and the other is different.
        #           Wait: if c0==c1, distinct colors are among {c0, c2, c3}.
        #           For exactly 2 distinct: 
        #             - c2==c3 and c2!=c0 (colors: c0, c2)
        #             - c2==c0 and c3!=c0 (colors: c0, c3)
        #             - c3==c0 and c2!=c0 (colors: c0, c2)
        #             - c2!=c0, c3!=c0, c2==c3 (colors: c0, c2)
        #           Actually all these reduce to: c2 and c3 are either both c0, both some other x, or one c0 one x.
        #           In any case, the set {c2, c3} has size at most 2, and if c2==c3==c0 then 1 color total.
        #           
        #   This is getting complicated. Let me just use a simpler check.
        
        # Alternative: 
        # We have < 3 distinct colors iff:
        # (c0==c1 and c2==c3) or (c0==c2 and c1==c3) or (c0==c3 and c1==c2)
        # OR all four are the same (which is covered by the above if we're careful)
        # Actually:
        # 1 color: all same. Covered by any of the three pairings.
        # 2 colors: the 4 elements can be partitioned into two pairs of equal values.
        #   So either (c0==c1 and c2==c3) or (c0==c2 and c1==c3) or (c0==c3 and c1==c2)
        #   This covers all 2-color cases? Let's verify:
        #   If colors are {a,b,a,b}: c0==c2, c1==c3 ✓
        #   If colors are {a,a,b,b}: c0==c1, c2==c3 ✓
        #   If colors are {a,b,b,a}: c0==c3, c1==c2 ✓
        #   If colors are {a,a,a,b}: c0==c1, but c2!=c3. c0==c2, but c1!=c3. c0==c3? No. c1==c2? Yes.
        #     Wait: c0==c1==c2!=c3. Check: c0==c1 (yes) and c2==c3 (no). c0==c2 (yes) and c1==c3 (no). c0==c3 (no) and c1==c2 (yes).
        #     So none of the three conditions hold! But this is a 2-color case.
        #   
        #   So my condition is wrong. Let me think again.
        #   
        #   2 distinct colors means the multiset has exactly 2 distinct values.
        #   This happens iff NOT (all 4 same) AND (the number of distinct values is 2).
        #   
        #   Let me just count distinct values properly.
        
        # Count distinct among c0, c1, c2, c3
        # d = 1
        # if c1 != c0: d += 1
        # if c2 != c0 and (c2 != c1 or c1 == c0): d += 1
        # if c3 != c0 and (c3 != c1 or c1 == c0) and (c3 != c2 or c2 == c0): d += 1
        
        # Actually, let's just use a simple approach that's fast enough:
        # Check if there are at least 3 distinct colors
        # We can check: is there a pair that are different, and another pair that are different, 
        # and the "different" values are not all the same?
        
        # Simplest fast check:
        # If c0 != c1:
        #   If c2 != c0 and c2 != c1:
        #     return False  # at least 3 colors (c0, c1, c2)
        #   If c3 != c0 and c3 != c1:
        #     return False  # at least 3 colors (c0, c1, c3)
        #   # c2 in {c0,c1}, c3 in {c0,c1}
        #   if c2 != c3:
        #     # c2 and c3 are different, both in {c0,c1}, so they are c0 and c1
        #     # Colors: c0, c1, c2, c3 where c2,c3 are c0,c1 in some order
        #     # So only 2 colors
        #     return True
        #   else:
        #     # c2 == c3, both in {c0,c1}
        #     if c2 == c0:
        #       # c0, c1, c0, c0 -> 2 colors
        #       return True
        #     else:
        #       # c0, c1, c1, c1 -> 2 colors
        #       return True
        # else:
        #   # c0 == c1
        #   if c2 != c0:
        #     if c3 != c0 and c3 != c2:
        #       return False  # c0, c2, c3 are 3 distinct
        #     # c3 in {c0, c2}
        #     return True  # at most 2 colors
        #   else:
        #     # c0 == c1 == c2
        #     if c3 != c0:
        #       return True  # 2 colors
        #     else:
        #       return True  # 1 color
        
        # Let me just implement this cleanly:
        if c0 != c1:
            if c2 != c0 and c2 != c1:
                return False
            if c3 != c0 and c3 != c1:
                return False
            return True
        else:
            # c0 == c1
            if c2 != c0:
                if c3 != c0 and c3 != c2:
                    return False
                return True
            else:
                # c0 == c1 == c2
                return True
    
    def count_bad(colors):
        count = 0
        for ap in aps:
            if is_bad(colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]):
                count += 1
        return count
    
    # Strategy: Start with a valid coloring using many colors, then reduce colors
    # Phase 1: Find a valid coloring with k colors (start with k=10)
    # Phase 2: Try to merge colors to reduce count
    
    def try_reduce_colors(colors, k, time_limit):
        """Try to reduce the number of colors by merging color classes."""
        current_k = k
        current_colors = colors.copy()
        
        while current_k > 3:
            if time.time() > time_limit:
                break
            
            # Try merging each pair of colors
            best_merge = None
            best_cost = count_bad(current_colors)
            
            for i in range(current_k):
                for j in range(i + 1, current_k):
                    # Try merging color j into color i
                    test_colors = current_colors.copy()
                    test_colors[test_colors == j] = i
                    
                    cost = count_bad(test_colors)
                    if cost == 0:
                        return test_colors, current_k - 1
                    elif cost < best_cost:
                        best_cost = cost
                        best_merge = (i, j, test_colors.copy())
            
            if best_merge is not None and best_cost < count_bad(current_colors):
                current_colors = best_merge[2]
                # Remove color j from consideration
                # We need to remap colors
                mask = current_colors >= best_merge[1]
                current_colors[mask] -= 1
                current_k -= 1
            else:
                break
        
        return current_colors, current_k
    
    # Main search
    best_result = None
    best_num_colors = N  # worst case
    
    for k in [5, 6, 7, 8, 10, 12, 15, 20]:
        if time.time() > deadline:
            break
        
        for restart in range(30):
            if time.time() > deadline:
                break
            
            colors = np.random.randint(0, k, N)
            current_cost = count_bad(colors)
            
            if current_cost == 0:
                # Try to reduce colors
                reduced_colors, reduced_k = try_reduce_colors(colors, k, deadline)
                if reduced_k < best_num_colors:
                    best_num_colors = reduced_k
                    best_result = reduced_colors
                if reduced_k <= 4:
                    return reduced_colors.tolist()
                continue
            
            # Simulated annealing
            T = 5.0
            T_min = 0.01
            cooling = 0.999
            steps = 3000
            
            for step in range(steps):
                if time.time() > deadline:
                    break
                
                pos = random.randint(0, N - 1)
                old_c = colors[pos]
                new_c = random.randint(0, k - 1)
                if new_c == old_c:
                    continue
                
                delta = 0
                for idx in pos_to_aps[pos]:
                    ap = aps[idx]
                    c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
                    old_bad = 1 if is_bad(c0, c1, c2, c3) else 0
                    
                    if ap[0] == pos:
                        c0 = new_c
                    elif ap[1] == pos:
                        c1 = new_c
                    elif ap[2] == pos:
                        c2 = new_c
                    elif ap[3] == pos:
                        c3 = new_c
                    
                    new_bad = 1 if is_bad(c0, c1, c2, c3) else 0
                    
                    if old_bad == 0 and new_bad == 1:
                        delta += 1
                    elif old_bad == 1 and new_bad == 0:
                        delta -= 1
                
                if delta <= 0:
                    colors[pos] = new_c
                    current_cost += delta
                else:
                    if random.random() < np.exp(-delta / T):
                        colors[pos] = new_c
                        current_cost += delta
                
                if current_cost == 0:
                    reduced_colors, reduced_k = try_reduce_colors(colors, k, deadline)
                    if reduced_k < best_num_colors:
                        best_num_colors = reduced_k
                        best_result = reduced_colors
                    if reduced_k <= 4:
                        return reduced_colors.tolist()
                    break
                
                T *= cooling
                if T < T_min:
                    break
    
    # Return best result found
    if best_result is not None:
        return best_result.tolist()
    
    # Fallback: random coloring with many colors
    colors = np.random.randint(0, 50, N)
    return colors.tolist()
