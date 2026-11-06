"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    n = 128
    deadline = time.time() + 50.0  # Safety margin before 60s limit
    
    # Precompute all nonconstant 4-term APs (0-indexed positions)
    # AP: (i, i+d, i+2d, i+3d) where d >= 1, all indices < n
    aps = []
    for d in range(1, (n - 3) // 3 + 1):
        for i in range(0, n - 3 * d):
            aps.append((i, i + d, i + 2 * d, i + 3 * d))
            
    num_aps = len(aps)
    
    # Map each position to the APs it belongs to
    pos_to_aps = [[] for _ in range(n)]
    for idx, ap in enumerate(aps):
        for pos in ap:
            pos_to_aps[pos].append(idx)
            
    def count_violations(colors):
        """Count number of APs with fewer than 3 distinct colors."""
        v = 0
        for ap in aps:
            c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
            # Check if distinct count < 3
            # This means either 1 or 2 distinct colors.
            if c0 == c1 == c2 == c3:
                v += 1
            elif c0 == c1 == c2 or c0 == c1 == c3 or c0 == c2 == c3 or c1 == c2 == c3:
                v += 1
            elif c0 == c1 and c2 == c3:
                v += 1
            elif c0 == c2 and c1 == c3:
                v += 1
            elif c0 == c3 and c1 == c2:
                v += 1
        return v

    def local_search(k_colors, time_limit):
        """
        Simulated Annealing to find a valid coloring using at most k_colors.
        Returns (colors, violations, distinct_colors_used)
        """
        # Initialize randomly
        colors = [random.randint(0, k_colors - 1) for _ in range(n)]
        
        # Calculate initial violations
        current_violations = count_violations(colors)
        best_violations = current_violations
        best_colors = colors[:]
        
        temp = 1.0
        cooling = 0.999
        min_temp = 0.001
        
        # Precompute valid color sets for each AP to speed up checks? 
        # Or just check the 4 positions.
        
        # For efficiency, we can track violations per AP? 
        # No, n=128 is small. Recalculating affected APs is fine.
        
        start_time = time.time()
        
        while time.time() < time_limit and temp > min_temp:
            # Pick a random position
            pos = random.randint(0, n - 1)
            old_color = colors[pos]
            
            # Calculate current violations for APs containing 'pos'
            affected_aps = pos_to_aps[pos]
            
            curr_invalid_count = 0
            for ap_idx in affected_aps:
                ap = aps[ap_idx]
                c = {colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]}
                if len(c) < 3:
                    curr_invalid_count += 1
            
            # Try each color
            best_new_color = old_color
            best_delta = 0
            best_new_invalid = curr_invalid_count
            
            for new_color in range(k_colors):
                if new_color == old_color:
                    continue
                
                colors[pos] = new_color
                
                new_invalid_count = 0
                for ap_idx in affected_aps:
                    ap = aps[ap_idx]
                    c = {colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]}
                    if len(c) < 3:
                        new_invalid_count += 1
                
                delta = new_invalid_count - curr_invalid_count
                
                if delta < best_delta:
                    best_delta = delta
                    best_new_color = new_color
                    best_new_invalid = new_invalid_count
                elif delta == best_delta and random.random() < 0.1:
                    best_new_color = new_color
                    best_new_invalid = new_invalid_count
            
            # Restore old color to evaluate acceptance properly if we don't accept immediately
            # Actually, we can just evaluate the acceptance based on best_delta
            # But we need to update the global state if we accept.
            
            # Acceptance criteria
            accept = False
            if best_delta < 0:
                accept = True
            else:
                # Simulated annealing acceptance for equal or worse
                if best_delta == 0:
                    accept = random.random() < 0.5
                else:
                    # Energy increase
                    energy = best_delta
                    if temp > 0:
                        prob = math.exp(-energy / temp)
                        accept = random.random() < prob
            
            if accept:
                colors[pos] = best_new_color
                current_violations += best_delta
                
                if current_violations < best_violations:
                    best_violations = current_violations
                    best_colors = colors[:]
                    
                    if best_violations == 0:
                        break
            else:
                # Keep old color
                pass
                
            temp *= cooling
            
        # Final check
        final_violations = count_violations(best_colors)
        distinct_colors = len(set(best_colors))
        return best_colors, final_violations, distinct_colors

    def try_reduce_colors(colors, k_target, time_limit):
        """
        Try to reduce the number of colors in a valid coloring to k_target.
        """
        colors = colors[:]
        used_colors = sorted(set(colors))
        
        # If we already use <= k_target, return
        if len(used_colors) <= k_target:
            return colors, len(used_colors)
        
        # We want to map some colors to others to reduce the set.
        # Greedily try to merge colors.
        
        current_colors = colors[:]
        current_used = set(current_colors)
        
        # Try to eliminate one color at a time
        # Sort colors by frequency? Or just iterate.
        while len(current_used) > k_target and time.time() < time_limit:
            eliminated = False
            for c_to_remove in list(current_used):
                # Try to map c_to_remove to another color in current_used
                for c_target in list(current_used):
                    if c_to_remove == c_target:
                        continue
                    
                    # Temporarily map
                    new_colors = [c_target if c == c_to_remove else c for c in current_colors]
                    
                    # Check validity
                    v = count_violations(new_colors)
                    if v == 0:
                        current_colors = new_colors
                        current_used = set(current_colors)
                        eliminated = True
                        break
                
                if eliminated:
                    break
            
            if not eliminated:
                # Try random local search with restricted palette?
                # For now, break if we can't merge greedily
                break
                
        return current_colors, len(set(current_colors))

    # Strategy:
    # 1. Try to find a valid solution with 3 colors.
    # 2. If fails, try 4 colors.
    # 3. If fails, try 5 colors.
    # 4. Once found, try to reduce colors.
    
    best_result = None
    best_color_count = 128
    
    # Use a fixed seed for reproducibility in local search? 
    # The task doesn't forbid it, but random is fine.
    
    # Try k=3
    time_for_k3 = 20.0
    result, viol, dist = local_search(3, time.time() + time_for_k3)
    if viol == 0:
        best_result = result
        best_color_count = dist
        
    # If k=3 failed or took too long, try k=4
    if best_result is None or best_color_count > 4:
        time_for_k4 = 15.0
        result, viol, dist = local_search(4, time.time() + time_for_k4)
        if viol == 0:
            if best_result is None or dist < best_color_count:
                best_result = result
                best_color_count = dist
    
    # If k=4 failed, try k=5
    if best_result is None or best_color_count > 5:
        time_for_k5 = 10.0
        result, viol, dist = local_search(5, time.time() + time_for_k5)
        if viol == 0:
            if best_result is None or dist < best_color_count:
                best_result = result
                best_color_count = dist

    # If we have a valid result, try to reduce colors
    if best_result is not None and time.time() < deadline - 2.0:
        # Try to reduce to 3 colors if we have more
        if best_color_count > 3:
            reduced, reduced_count = try_reduce_colors(best_result, 3, deadline - 2.0)
            if reduced_count < best_color_count:
                best_result = reduced
                best_color_count = reduced_count
        
        # Try to reduce to 2 colors? (Unlikely to be valid for 4-APs)
        # But let's check if 2 is possible.
        if best_color_count > 2 and time.time() < deadline - 1.0:
            # 2 colors is very hard/impossible for 4-APs?
            # AP (0,1,2,3) with 2 colors: 0,1,0,1 (2 colors) -> Invalid.
            # 0,0,1,1 (2 colors) -> Invalid.
            # So 2 colors is likely impossible.
            pass

    # Fallback: If no valid solution found, return identity (valid but uses 128 colors)
    if best_result is None:
        # Identity mapping: i -> i. 
        # Any 4-AP has distinct indices, so distinct colors.
        # This is always valid.
        return list(range(n))
        
    return best_result
# EVOLVE-BLOCK-END
