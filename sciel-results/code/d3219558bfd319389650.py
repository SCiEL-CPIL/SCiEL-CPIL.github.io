import time
import random
import numpy as np

N = 128

def construct():
        # Precompute APs
        aps = []
        pos_to_aps = [[] for _ in range(N)]
        
        for d in range(1, N // 3 + 1):
            for a in range(1, N - 3 * d + 1):
                p = [a-1, a+d-1, a+2*d-1, a+3*d-1] # 0-indexed
                aps.append(p)
                for idx in p:
                    pos_to_aps[idx].append(len(aps) - 1)
        
        def count_bad(colors):
            bad_count = 0
            for p in aps:
                cs = {colors[p[0]], colors[p[1]], colors[p[2]], colors[p[3]]}
                if len(cs) < 3:
                    bad_count += 1
            return bad_count

        def get_bad_aps(colors):
            bads = []
            for i, p in enumerate(aps):
                cs = {colors[p[0]], colors[p[1]], colors[p[2]], colors[p[3]]}
                if len(cs) < 3:
                    bads.append(i)
            return bads

        start_time = time.time()
        deadline = start_time + 50 # 50s limit
        
        # Try K=3 first
        for K in [3, 4, 5]:
            if time.time() > deadline - 5:
                break
                
            # Initialize
            colors = np.random.randint(0, K, N)
            
            # Min-Conflicts
            best_colors = colors.copy()
            best_bad = count_bad(best_colors)
            
            steps = 0
            while time.time() < deadline - 5:
                # Pick a random bad AP
                bads = get_bad_aps(colors)
                if not bads:
                    break
                
                bad_ap_idx = random.choice(bads)
                p = aps[bad_ap_idx]
                
                # Pick a random position in this AP to change
                pos_idx = random.choice(p)
                
                # Try all colors for this position
                best_c = colors[pos_idx]
                min_bad_delta = float('inf')
                
                # We need to evaluate change in bad count for APs involving pos_idx
                # This is expensive to do exactly every time.
                # Approximation: just try a random new color and check global bad count?
                # Too slow.
                
                # Better: Check the specific APs involving pos_idx.
                # For each AP involving pos_idx, check if it is bad before and after.
                
                # Optimization: Pre-compute current state of APs?
                # Let's just try a random new color and check if it reduces local conflicts.
                
                new_c = random.randint(0, K - 1)
                if new_c == colors[pos_idx]:
                    continue
                    
                old_c = colors[pos_idx]
                colors[pos_idx] = new_c
                
                # Check if this change improved things
                # We can just recount bad APs involving pos_idx
                # But we need to know if TOTAL bad count decreased.
                
                # Simple heuristic: Accept if the specific bad AP we picked is fixed,
                # or if random.
                
                # Let's use a simpler hill climbing:
                # Try all K colors for pos_idx, pick the one that minimizes
                # the number of bad APs among those involving pos_idx.
                
                best_local_bad = float('inf')
                best_c_found = old_c
                
                for c in range(K):
                    if c == old_c:
                        continue
                    colors[pos_idx] = c
                    local_bad = 0
                    for ap_idx in pos_to_aps[pos_idx]:
                        pp = aps[ap_idx]
                        cs = {colors[pp[0]], colors[pp[1]], colors[pp[2]], colors[pp[3]]}
                        if len(cs) < 3:
                            local_bad += 1
                    if local_bad < best_local_bad:
                        best_local_bad = local_bad
                        best_c_found = c
                        if local_bad == 0:
                            break
                            
                colors[pos_idx] = best_c_found
                
                # Update global best
                # Recounting global bad is O(APs).
                # If N=128, APs~4000. 4000 * 4 checks is fast.
                # We can do this every 100 steps or if we suspect improvement.
                # Let's do it every 10 steps to be safe.
                steps += 1
                if steps % 10 == 0:
                    curr_bad = count_bad(colors)
                    if curr_bad < best_bad:
                        best_bad = curr_bad
                        best_colors = colors.copy()
                        if best_bad == 0:
                            break
                            # Can break outer loops too
                            
            if best_bad == 0:
                # Map colors to 0..K-1 (they are already)
                return list(best_colors)
                
        # Fallback: Return a valid coloring with many colors
        # Using distinct colors for each position guarantees validity (4 distinct colors per AP)
        # But we want to minimize.
        # A simple valid pattern:
        # Color i with i % 4? No, failed.
        # Color i with i? 128 colors.
        # Let's try to find a valid one quickly with 8 colors using the same search.
        
        # If we reached here, we failed to find 3,4,5.
        # Use 128 colors, all distinct.
        return list(range(N))
