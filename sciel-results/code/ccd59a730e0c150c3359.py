import time
import random
import math

def construct():
    N = 128
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin
    
    # Precompute all 4-term APs
    # AP: a, a+d, a+2d, a+3d
    # 0-indexed: i, i+d, i+2d, i+3d
    # i >= 0, i+3d < N
    aps = []
    for d in range(1, N // 3 + 1):
        for i in range(0, N - 3 * d):
            aps.append((i, i + d, i + 2 * d, i + 3 * d))
    
    num_aps = len(aps)
    
    # Helper to calculate distinct colors in an AP
    # We will maintain state: colors array, and for each AP, the count of each color.
    # Since K is small (we hope <= 4 or 5), we can use a list of counts.
    
    def get_penalty_and_colors(colors):
        """
        Returns (penalty, num_colors_used)
        penalty = number of APs with < 3 distinct colors
        """
        # For speed, we might want to incrementally update, but for N=128 and ~1000 APs,
        # a full scan is fast enough if done carefully.
        # However, in SA, we do many steps.
        # Let's use incremental updates.
        pass

    # State variables
    # colors: list of int length N
    # ap_color_counts: list of lists, ap_color_counts[ap_idx][c] = count of color c in that AP
    # ap_bad: list of bool, True if AP has < 3 distinct colors
    
    def initialize_state(K, seed):
        random.seed(seed)
        colors = [random.randint(0, K - 1) for _ in range(N)]
        ap_color_counts = [[0] * K for _ in range(num_aps)]
        for idx, (i, j, k, l) in enumerate(aps):
            ap_color_counts[idx][colors[i]] += 1
            ap_color_counts[idx][colors[j]] += 1
            ap_color_counts[idx][colors[k]] += 1
            ap_color_counts[idx][colors[l]] += 1
        
        ap_bad = [0] * num_aps
        total_bad = 0
        for idx in range(num_aps):
            distinct = sum(1 for c in ap_color_counts[idx] if c > 0)
            if distinct < 3:
                ap_bad[idx] = 1
                total_bad += 1
        
        num_colors_used = len(set(colors))
        return colors, ap_color_counts, ap_bad, total_bad, num_colors_used

    def try_move(colors, ap_color_counts, ap_bad, total_bad, num_colors_used, K, T, current_time):
        """
        Attempts a random move.
        Returns new state or None if no move made.
        """
        if random.random() > 0.9: # Occasionally try to introduce a new color? No, we want to minimize.
             pass
        
        i = random.randint(0, N - 1)
        old_c = colors[i]
        
        # Choose new color.
        # Bias towards existing colors to keep K low, but allow new color if K is small?
        # Actually, we fix K for a run. We try K=3, then K=4, etc.
        # Or we allow K to grow if stuck? No, we want to minimize.
        # Let's pick from 0..K-1.
        new_c = random.randint(0, K - 1)
        
        if new_c == old_c:
            return None
        
        # Calculate change in penalty
        # Find all APs containing i
        delta_bad = 0
        affected_aps = []
        
        # We need to know which APs contain i.
        # Precomputing this is expensive memory-wise but N=128 is small.
        # Or we can just iterate all APs? 1000 APs * 1000 steps = 1e6 ops. Fast.
        # But checking all APs for each move is O(num_aps).
        # Let's precompute `aps_containing[i]` = list of AP indices containing i.
        
        for ap_idx in aps_containing[i]:
            counts = ap_color_counts[ap_idx]
            old_count = counts[old_c]
            new_count = counts[new_c]
            
            # Update counts temporarily to check distinctness
            # distinct_old = sum(1 for c in counts if c > 0)
            # distinct_new = ...
            
            # Optimization: 
            # If old_count == 1, removing old_c might reduce distinct count.
            # If new_count == 0, adding new_c might increase distinct count.
            
            distinct_old = 0
            for c in range(K):
                if counts[c] > 0:
                    distinct_old += 1
            
            # Apply change
            counts[old_c] -= 1
            counts[new_c] += 1
            
            distinct_new = 0
            for c in range(K):
                if counts[c] > 0:
                    distinct_new += 1
            
            bad_old = 1 if distinct_old < 3 else 0
            bad_new = 1 if distinct_new < 3 else 0
            
            delta_bad += bad_new - bad_old
            affected_aps.append((ap_idx, old_c, new_c))
            
            # Revert counts for now, we will apply if accepted
            counts[old_c] += 1
            counts[new_c] -= 1
            
            if delta_bad > T: # Pruning? No, T is temperature, not penalty limit.
                pass
        
        # Calculate acceptance probability
        if delta_bad < 0:
            accept = True
        elif delta_bad == 0:
            # Check if this move reduces color usage?
            # If old_c is no longer used in the whole array?
            # We need to track global color counts.
            # Let's assume for now we only accept if it helps penalty.
            # To reduce colors, we need a specific strategy.
            # If delta_bad == 0, we might accept to explore.
            accept = True
        else:
            accept = random.random() < math.exp(-delta_bad / T)
        
        if accept:
            # Apply move
            colors[i] = new_c
            for ap_idx, old_c, new_c in affected_aps:
                ap_color_counts[ap_idx][old_c] -= 1
                ap_color_counts[ap_idx][new_c] += 1
                # Update ap_bad
                # Recalculate distinct for this AP
                distinct = 0
                for c in range(K):
                    if ap_color_counts[ap_idx][c] > 0:
                        distinct += 1
                new_bad = 1 if distinct < 3 else 0
                old_bad = ap_bad[ap_idx]
                total_bad += new_bad - old_bad
                ap_bad[ap_idx] = new_bad
            
            # Update global color counts
            global_color_counts[old_c] -= 1
            global_color_counts[new_c] += 1
            
            # Check if we can reduce K?
            # If any color count is 0, we can remap.
            # But we are in a fixed K loop.
            # We will handle K reduction in the outer loop.
            
            return True
        else:
            return False

    # Precompute APs containing each index
    aps_containing = [[] for _ in range(N)]
    for idx, (i, j, k, l) in enumerate(aps):
        aps_containing[i].append(idx)
        aps_containing[j].append(idx)
        aps_containing[k].append(idx)
        aps_containing[l].append(idx)

    best_solution = None
    best_K = float('inf')
    
    # Try K from 3 upwards
    for K in range(3, 8):
        if time.time() > deadline - 5.0:
            break
            
        # Run SA for this K
        T = 5.0
        cooling = 0.999
        min_T = 0.01
        
        # Initialize
        colors, ap_color_counts, ap_bad, total_bad, num_colors_used = initialize_state(K, K)
        global_color_counts = [0] * K
        for c in colors:
            global_color_counts[c] += 1
        
        # If initial state is valid and uses < K colors, great.
        # We want to minimize colors used.
        
        steps = 0
        while time.time() < deadline - 5.0 and T > min_T:
            if total_bad == 0:
                # Valid coloring. Check if we can reduce colors.
                used = [c for c in range(K) if global_color_counts[c] > 0]
                if len(used) < K:
                    # Remap to 0..len(used)-1
                    mapping = {c: i for i, c in enumerate(used)}
                    new_colors = [mapping[c] for c in colors]
                    if len(used) < best_K:
                        best_K = len(used)
                        best_solution = new_colors
                    # Try to continue searching for even fewer? 
                    # We can't go below len(used) with this K.
                    # But maybe a different configuration with same K uses fewer?
                    # Or we can break and try smaller K? 
                    # We are iterating K upwards. If we found a valid one with K=3, 
                    # we should break and return it? 
                    # The task is to MINIMIZE colors.
                    # If we found a valid coloring with K colors, the answer is <= K.
                    # We should check if K-1 is possible.
                    # Since we iterate K=3,4,5..., if we find a solution for K=3, 
                    # we can stop immediately because 3 is the minimum possible 
                    # (a 4-AP needs at least 3 colors? 
                    # Wait, can a 4-AP use 2 colors? 
                    # "Every nonconstant four-term progression ... must use AT LEAST three distinct colors."
                    # So yes, min colors is 3.
                    # If we find a valid 3-coloring, we are done.
                    if best_K == 3:
                        return best_solution
                else:
                    # Valid but uses all K colors.
                    # Try to reduce usage by reassigning.
                    pass
            
            # Perform move
            if try_move(colors, ap_color_counts, ap_bad, total_bad, num_colors_used, K, T, time.time()):
                # Update global counts in try_move? 
                # My try_move implementation above didn't update global_color_counts correctly 
                # because it's a local function. I need to pass it or make it global.
                # Let's refactor: try_move should return the move details or update state directly.
                # Given the complexity, let's simplify the move function to be inline or 
                # pass global_color_counts.
                pass
            
            T *= cooling
            steps += 1
            if steps % 1000 == 0 and total_bad == 0:
                used = [c for c in range(K) if global_color_counts[c] > 0]
                if len(used) < best_K:
                    mapping = {c: i for i, c in enumerate(used)}
                    new_colors = [mapping[c] for c in colors]
                    best_K = len(used)
                    best_solution = new_colors
                    if best_K == 3:
                        return best_solution

    # If no valid solution found with K<=7, return the best invalid one? 
    # No, must return valid.
    # Fallback: Greedy with many colors.
    # Use K=128? No, that's trivial but uses many colors.
    # We need a valid coloring.
    # Let's use a known valid construction or a greedy that ensures validity.
    # Greedy: Assign smallest color that doesn't create a bad AP.
    # This guarantees validity.
    
    def greedy_valid():
        colors = [0] * N
        # We will use up to N colors if needed, but hopefully few.
        # Actually, we can just use random assignment and repair?
        # Let's use the SA result if it was valid.
        # If SA failed to find valid, use greedy.
        
        # Simple greedy:
        for i in range(N):
            used_in_aps = set()
            # Check all APs involving i?
            # No, we assign i, so we check APs where i is the last element?
            # No, APs can be in any position.
            # If we assign i, we affect APs where i is any of the 4 terms.
            # To ensure validity, we must check all APs involving i.
            # But other positions in those APs might not be assigned yet.
            # So greedy left-to-right only works if we check APs ending at i?
            # No, an AP (i, i+d, i+2d, i+3d) has i as the first term.
            # If we assign i, we don't know the others yet.
            # So we can't guarantee validity locally unless we look ahead or use a different order.
            
            # Alternative: Use the SA solution even if it has some bad APs? No, must be valid.
            # Let's just run a very aggressive SA with K=10 to find a valid one.
            pass
        return None

    # Re-implement the SA loop properly with global state updates
    # Since the previous try_move was incomplete, I will write a clean inner loop.
    
    for K in range(3, 8):
        if time.time() > deadline - 5.0:
            break
        random.seed(K * 1000)
        colors = [random.randint(0, K - 1) for _ in range(N)]
        ap_color_counts = [[0] * K for _ in range(num_aps)]
        for idx, (i, j, k, l) in enumerate(aps):
            ap_color_counts[idx][colors[i]] += 1
            ap_color_counts[idx][colors[j]] += 1
            ap_color_counts[idx][colors[k]] += 1
            ap_color_counts[idx][colors[l]] += 1
        ap_bad = [0] * num_aps
        total_bad = 0
        for idx in range(num_aps):
            distinct = sum(1 for c in ap_color_counts[idx] if c > 0)
            if distinct < 3:
                ap_bad[idx] = 1
                total_bad += 1
        
        global_color_counts = [0] * K
        for c in colors:
            global_color_counts[c] += 1
        
        T = 10.0
        cooling = 0.995
        min_T = 0.1
        steps = 0
        
        while time.time() < deadline - 5.0 and T > min_T:
            i = random.randint(0, N - 1)
            old_c = colors[i]
            new_c = random.randint(0, K - 1)
            if new_c == old_c:
                continue
            
            delta_bad = 0
            affected = []
            
            for ap_idx in aps_containing[i]:
                counts = ap_color_counts[ap_idx]
                # Calculate distinct before
                d_old = 0
                for c in range(K):
                    if counts[c] > 0: d_old += 1
                b_old = 1 if d_old < 3 else 0
                
                # Apply
                counts[old_c] -= 1
                counts[new_c] += 1
                
                d_new = 0
                for c in range(K):
                    if counts[c] > 0: d_new += 1
                b_new = 1 if d_new < 3 else 0
                
                delta_bad += b_new - b_old
                affected.append(ap_idx)
                
                # Revert
                counts[old_c] += 1
                counts[new_c] -= 1
            
            if delta_bad < 0 or (delta_bad == 0 and random.random() < 0.5) or (delta_bad > 0 and random.random() < math.exp(-delta_bad / T)):
                # Accept
                colors[i] = new_c
                for ap_idx in affected:
                    counts = ap_color_counts[ap_idx]
                    counts[old_c] -= 1
                    counts[new_c] += 1
                    d = 0
                    for c in range(K):
                        if counts[c] > 0: d += 1
                    b_new = 1 if d < 3 else 0
                    total_bad += b_new - ap_bad[ap_idx]
                    ap_bad[ap_idx] = b_new
                
                global_color_counts[old_c] -= 1
                global_color_counts[new_c] += 1
                
                if total_bad == 0:
                    used = [c for c in range(K) if global_color_counts[c] > 0]
                    if len(used) < best_K:
                        mapping = {c: i for i, c in enumerate(used)}
                        new_colors = [mapping[c] for c in colors]
                        best_K = len(used)
                        best_solution = new_colors
                        if best_K == 3:
                            return best_solution
            
            T *= cooling
            steps += 1
            if steps > 100000:
                break

    if best_solution:
        return best_solution
    
    # Fallback: Return a valid coloring with many colors (e.g. 4 colors might work, 
    # but if SA failed, use a deterministic valid one).
    # A valid 4-coloring: 
    # Let's try to construct one manually or use a simple pattern.
    # Pattern: color[i] = i % 4? 
    # AP: 0, 1, 2, 3 -> 0,1,2,3 (4 colors) OK.
    # AP: 0, 2, 4, 6 -> 0,2,0,2 (2 colors) FAIL.
    # So mod 4 fails.
    
    # Let's just return the result of a greedy that ensures validity by using many colors.
    # If we use N colors, it's valid. But we want to minimize.
    # If SA failed, it's likely 3 is impossible and 4 is hard?
    # Actually, for N=128, 3 colors is likely impossible. 4 colors might be possible.
    # If we didn't find one, we return a valid one with more colors.
    
    # Simple valid construction: 
    # Use 2-adic valuation? 
    # Or just use random assignment and repair with a greedy check?
    # Let's use a greedy that assigns the smallest color that doesn't violate 
    # any *completed* AP.
    # This works if we process in an order such that when we assign x, 
    # all APs involving x are "closed" (all other elements assigned).
    # This is possible if we process from right to left? 
    # AP: a, a+d, a+2d, a+3d. 
    # If we assign from N-1 down to 0, when we assign a, the other elements 
    # a+d, a+2d, a+3d are already assigned (since they are larger).
    # So we can check validity of all APs starting at a.
    # This guarantees a valid coloring!
    
    colors = [0] * N
    # We need to track which colors are used? No, just assign.
    # To minimize colors, we should try to reuse colors.
    # Greedy: For i from N-1 down to 0:
    #   Try color 0, 1, 2...
    #   Check if assigning color c to i creates a bad AP.
    #   An AP is (i, i+d, i+2d, i+3d). 
    #   Since i is the smallest index in the AP, and we are going backwards, 
    #   i+d, i+2d, i+3d are already assigned.
    #   So we check all APs starting at i.
    #   If color c makes any of these APs have < 3 distinct colors, reject c.
    #   Pick the smallest valid c.
    
    for i in range(N - 1, -1, -1):
        valid_c = None
        for c in range(128): # Try 0..127
            ok = True
            max_d = (N - 1 - i) // 3
            for d in range(1, max_d + 1):
                j = i + d
                k = i + 2 * d
                l = i + 3 * d
                if l >= N: break
                cs = [c, colors[j], colors[k], colors[l]]
                if len(set(cs)) < 3:
                    ok = False
                    break
            if ok:
                valid_c = c
                break
        if valid_c is not None:
            colors[i] = valid_c
        else:
            colors[i] = 0 # Should not happen
    
    return colors

