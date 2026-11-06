import time
import random
import numpy as np
from collections import defaultdict, Counter

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    N = 128
    
    # Pre-compute all 4-term APs and index them by position for fast updates
    aps = []
    ap_indices = [[] for _ in range(N)]
    for d in range(1, N // 3 + 1):
        for i in range(N - 3 * d):
            ap = (i, i + d, i + 2 * d, i + 3 * d)
            ap_idx = len(aps)
            aps.append(ap)
            for idx in ap:
                ap_indices[idx].append(ap_idx)
                
    num_aps = len(aps)
    
    # Convert to numpy arrays for vectorized operations
    aps_np = np.array(aps, dtype=np.int32)  # shape (num_aps, 4)
    
    def count_violations(colors):
        colors = np.array(colors, dtype=np.int32)
        # Get colors at all positions of all APs
        ap_colors = colors[aps_np]  # shape (num_aps, 4)
        # Count distinct colors in each AP
        c0, c1, c2, c3 = ap_colors[:, 0], ap_colors[:, 1], ap_colors[:, 2], ap_colors[:, 3]
        # Violation if: all same, or exactly 2 distinct
        all_same = (c0 == c1) & (c1 == c2) & (c2 == c3)
        pair_01_23 = (c0 == c1) & (c2 == c3) & (c0 != c2)
        pair_01_32 = (c0 == c1) & (c3 == c2) & (c0 != c3)
        pair_02_13 = (c0 == c2) & (c1 == c3) & (c0 != c1)
        pair_03_12 = (c0 == c3) & (c1 == c2) & (c0 != c1)
        one_diff_0 = (c1 == c2) & (c2 == c3) & (c0 != c1)
        one_diff_1 = (c0 == c2) & (c2 == c3) & (c1 != c0)
        one_diff_2 = (c0 == c1) & (c1 == c3) & (c2 != c0)
        one_diff_3 = (c0 == c1) & (c1 == c2) & (c3 != c0)
        violation = all_same | pair_01_23 | pair_01_32 | pair_02_13 | pair_03_12 | \
                    one_diff_0 | one_diff_1 | one_diff_2 | one_diff_3
        return int(np.sum(violation))
    
    def run_seed():
        colors=[]
        for x in range(1,129):
            for color in range(128):
                if (all(len({color}|{colors[x-1-j*d] for j in (1,2,3)})>=3 for d in range(1,(x-1)//3+1))
                    and all(any(colors[x-1-j*d]!=color for j in (1,2)) for d in range(1,(x-1)//2+1))):
                    colors.append(color);break
        return colors

    # Get initial seed solution
    seed_colors = run_seed()
    best_colors = seed_colors[:]
    best_k = len(set(seed_colors))
    
    # Better local search with proper state management
    def local_search_improved(colors, target_k, time_limit, num_restarts=10):
        """
        Improved local search with proper violation tracking and simulated annealing.
        """
        colors = list(colors)
        k = len(set(colors))
        
        # Remap colors to 0..target_k-1 if needed
        if k > target_k:
            color_map = {}
            next_color = 0
            for c in sorted(set(colors)):
                if next_color < target_k:
                    color_map[c] = next_color
                    next_color += 1
                else:
                    color_map[c] = random.randint(0, target_k - 1)
            colors = [color_map[c] for c in colors]
        
        # Initialize violation state
        def compute_violations(col):
            vf = np.zeros(num_aps, dtype=bool)
            for ai in range(num_aps):
                ap = aps_np[ai]
                c0, c1, c2, c3 = col[ap[0]], col[ap[1]], col[ap[2]], col[ap[3]]
                vf[ai] = len({c0, c1, c2, c3}) < 3
            return vf, int(np.sum(vf))
        
        violation_flags, num_violations = compute_violations(colors)
        
        if num_violations == 0:
            return colors, 0
        
        best_violations = num_violations
        best_colors = colors[:]
        temperature = 1.0
        cooling_rate = 0.999
        
        for restart in range(num_restarts):
            if time.time() > time_limit:
                break
            
            if restart > 0:
                # Random restart with some structure
                for i in range(N):
                    colors[i] = random.randint(0, target_k - 1)
                violation_flags, num_violations = compute_violations(colors)
                best_violations = num_violations
                best_colors = colors[:]
                temperature = 1.0
            
            # Local search loop
            max_steps = 3000
            no_improve_steps = 0
            for step in range(max_steps):
                if time.time() > time_limit:
                    break
                
                if num_violations == 0:
                    return colors, 0
                
                if num_violations < best_violations:
                    best_violations = num_violations
                    best_colors = colors[:]
                    no_improve_steps = 0
                else:
                    no_improve_steps += 1
                
                # If stuck for too long, cool down and maybe restart
                if no_improve_steps > 500:
                    temperature *= 0.5
                    if temperature < 0.01:
                        break
                
                # Pick a random violated AP
                violated_idx = np.where(violation_flags)[0]
                if len(violated_idx) == 0:
                    break
                ai = int(random.choice(violated_idx))
                ap = aps_np[ai]
                
                # Pick a random position in the AP to change
                pos_in_ap = random.randint(0, 3)
                idx = int(ap[pos_in_ap])
                old_c = colors[idx]
                
                # Try all possible new colors
                best_new_c = old_c
                best_delta = 1  # Start with a bad delta
                best_affected = None
                
                affected_aps = ap_indices[idx]
                
                for new_c in range(target_k):
                    if new_c == old_c:
                        continue
                    
                    # Calculate delta for this change
                    delta = 0
                    new_flags = {}
                    for aai in affected_aps:
                        ap2 = aps_np[aai]
                        c0, c1, c2, c3 = (new_c if int(ap2[j]) == idx else colors[int(ap2[j])] for j in range(4))
                        is_violated = len({c0, c1, c2, c3}) < 3
                        was_violated = violation_flags[aai]
                        
                        if was_violated and not is_violated:
                            delta -= 1
                        elif not was_violated and is_violated:
                            delta += 1
                        new_flags[aai] = is_violated
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_new_c = new_c
                        best_affected = new_flags
                
                if best_new_c == old_c:
                    continue
                
                # Accept with simulated annealing
                if best_delta < 0:
                    accept = True
                elif best_delta == 0:
                    accept = random.random() < 0.5
                else:
                    accept = random.random() < np.exp(-best_delta / max(temperature, 0.01))
                
                if accept:
                    colors[idx] = best_new_c
                    for aai, flag in best_affected.items():
                        violation_flags[aai] = flag
                    num_violations += best_delta
                    temperature *= cooling_rate
                else:
                    # Don't change, just cool down slightly
                    temperature *= 0.999
        
        if best_violations < num_violations:
            return best_colors, best_violations
        return colors, num_violations
    
    # Improved color merging: try pairwise merges more systematically
    def try_merge_colors(colors, target_k, time_limit):
        """
        Try to merge colors more systematically.
        Returns (new_colors, new_k) or None.
        """
        colors = list(colors)
        k = len(set(colors))
        
        if k <= target_k:
            return colors, k
        
        color_freq = Counter(colors)
        sorted_colors = sorted(color_freq.keys(), key=lambda c: color_freq[c])
        
        # Try merging least frequent colors into others
        for i, elim_color in enumerate(sorted_colors):
            if time.time() > time_limit:
                break
            if i >= 3:  # Only try top 3 least frequent
                break
            
            positions = [i for i, c in enumerate(colors) if c == elim_color]
            
            for new_color in sorted_colors:
                if new_color == elim_color:
                    continue
                
                # Test remapping
                test_colors = colors[:]
                for pos in positions:
                    test_colors[pos] = new_color
                
                if count_violations(test_colors) == 0:
                    return test_colors, len(set(test_colors))
        
        return None, None
    
    # Main search loop
    # First, try to find a valid coloring with fewer colors
    for target_k in range(3, best_k + 1):
        if time.time() > deadline - 15:
            break
        
        time_limit = min(deadline - 10, time.time() + 6.0)
        
        # Try multiple restarts
        found = False
        for restart in range(5):
            if time.time() > time_limit:
                break
            
            # Start from seed or random
            if restart == 0:
                init_colors = seed_colors[:]
            else:
                init_colors = [random.randint(0, target_k - 1) for _ in range(N)]
            
            result, violations = local_search_improved(init_colors, target_k, time_limit, num_restarts=3)
            
            if violations == 0:
                # Valid coloring with target_k colors
                best_colors = result
                best_k = target_k
                found = True
                break
        
        if found:
            break
    
    # Try to further reduce colors via merging
    if time.time() < deadline - 5:
        for _ in range(20):
            if time.time() > deadline - 3:
                break
            
            merged, new_k = try_merge_colors(best_colors, best_k - 1, deadline - 3)
            if merged is not None and new_k < best_k:
                best_colors = merged
                best_k = new_k
            else:
                break
    
    # Final polish: local search on best solution to see if we can reduce further
    if time.time() < deadline - 2:
        for target_k in range(3, best_k):
            if time.time() > deadline - 1:
                break
            result, violations = local_search_improved(best_colors, target_k, deadline - 1, num_restarts=2)
            if violations == 0:
                best_colors = result
                best_k = target_k
                break
    
    return best_colors
# EVOLVE-BLOCK-END
