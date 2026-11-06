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
    
    # Randomized local search with incremental violation tracking
    def local_search_random(colors, target_k, time_limit, num_restarts=20):
        """
        Randomized local search to find a valid coloring with exactly target_k colors.
        Returns (colors, num_violations) or None if timeout.
        """
        colors = list(colors)
        k = len(set(colors))
        
        # Try to reduce to target_k colors first via remapping
        if k > target_k:
            # Map all colors to 0..target_k-1
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
        violation_flags = np.zeros(num_aps, dtype=bool)
        for ai in range(num_aps):
            ap = aps_np[ai]
            c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
            if len({c0, c1, c2, c3}) < 3:
                violation_flags[ai] = True
        num_violations = int(np.sum(violation_flags))
        
        if num_violations == 0:
            return colors, 0
        
        best_violations = num_violations
        best_colors = colors[:]
        
        for restart in range(num_restarts):
            if time.time() > time_limit:
                break
            
            if restart > 0:
                # Random restart
                for i in range(N):
                    colors[i] = random.randint(0, target_k - 1)
                for ai in range(num_aps):
                    ap = aps_np[ai]
                    c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
                    violation_flags[ai] = len({c0, c1, c2, c3}) < 3
                num_violations = int(np.sum(violation_flags))
                best_violations = num_violations
                best_colors = colors[:]
            
            # Local search loop
            max_steps = 5000
            for step in range(max_steps):
                if time.time() > time_limit:
                    break
                
                if num_violations == 0:
                    return colors, 0
                
                if num_violations < best_violations:
                    best_violations = num_violations
                    best_colors = colors[:]
                
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
                new_c = random.randint(0, target_k - 1)
                
                if new_c == old_c:
                    continue
                
                # Save old state for affected APs
                affected_aps = ap_indices[idx]
                old_flags = []
                for aai in affected_aps:
                    old_flags.append((aai, bool(violation_flags[aai])))
                
                # Apply change
                colors[idx] = new_c
                
                # Update violations for affected APs
                delta = 0
                for aai in affected_aps:
                    ap2 = aps_np[aai]
                    c0, c1, c2, c3 = colors[ap2[0]], colors[ap2[1]], colors[ap2[2]], colors[ap2[3]]
                    is_violated = len({c0, c1, c2, c3}) < 3
                    was_violated = violation_flags[aai]
                    
                    if was_violated and not is_violated:
                        delta -= 1
                    elif not was_violated and is_violated:
                        delta += 1
                    violation_flags[aai] = is_violated
                
                num_violations += delta
                
                # Accept if better, or with some probability if equal (simulated annealing-like)
                if delta < 0 or (delta == 0 and random.random() < 0.3):
                    pass  # Accept
                else:
                    # Revert
                    colors[idx] = old_c
                    for aai, old_flag in old_flags:
                        violation_flags[aai] = old_flag
                    num_violations -= delta
        
        if best_violations < num_violations:
            return best_colors, best_violations
        return colors, num_violations
    
    # Color merging: try to eliminate the least frequent color
    def try_merge_color(colors, target_k, time_limit):
        """
        Try to merge the least frequent color into another color.
        Returns (new_colors, new_k) or None if not possible.
        """
        colors = list(colors)
        k = len(set(colors))
        
        if k <= target_k:
            return colors, k
        
        color_freq = Counter(colors)
        sorted_colors = sorted(color_freq.keys(), key=lambda c: color_freq[c])
        
        # Try to eliminate the least frequent color
        for elim_color in sorted_colors[:3]:  # Try top 3 least frequent
            if time.time() > time_limit:
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
                    # Accept
                    colors = test_colors
                    new_k = len(set(colors))
                    return colors, new_k
        
        return None, None
    
    # Main search loop
    # First, try to find a valid coloring with fewer colors using randomized local search
    for target_k in range(3, best_k):
        if time.time() > deadline - 15:
            break
        
        time_limit = min(deadline - 10, time.time() + 8.0)
        
        # Try multiple restarts
        found = False
        for restart in range(3):
            if time.time() > time_limit:
                break
            
            # Start from seed or random
            if restart == 0:
                init_colors = seed_colors[:]
            else:
                init_colors = [random.randint(0, target_k - 1) for _ in range(N)]
            
            result, violations = local_search_random(init_colors, target_k, time_limit, num_restarts=5)
            
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
        for _ in range(10):
            if time.time() > deadline - 3:
                break
            
            merged, new_k = try_merge_color(best_colors, best_k - 1, deadline - 3)
            if merged is not None and new_k < best_k:
                best_colors = merged
                best_k = new_k
            else:
                break
    
    # Final local search on best solution
    if time.time() < deadline - 2:
        result, violations = local_search_random(best_colors, best_k, deadline - 1, num_restarts=3)
        if violations == 0 and len(set(result)) < best_k:
            best_colors = result
            best_k = len(set(best_colors))
    
    return best_colors
# EVOLVE-BLOCK-END
