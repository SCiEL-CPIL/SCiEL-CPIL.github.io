import time
import numpy as np
import random

def construct():
    n = 256
    deadline = time.time() + 115.0  # Safety margin
    
    # 1. Precompute all 4-term APs
    # AP is (a, a+d, a+2d, a+3d) with 1 <= a, a+3d <= n, d >= 1
    # Using 0-based indexing for internal arrays: 0..255 corresponds to 1..256
    aps_list = []
    for d in range(1, n // 3 + 1):
        max_a = n - 3 * d
        for a in range(max_a):
            # 0-based indices
            aps_list.append((a, a + d, a + 2 * d, a + 3 * d))
    
    aps = np.array(aps_list, dtype=np.int32)
    n_aps = len(aps)
    
    # 2. For each position, find which APs contain it
    pos_to_aps = [[] for _ in range(n)]
    for i in range(n_aps):
        for k in range(4):
            pos = aps[i, k]
            pos_to_aps[pos].append(i)
    
    # Convert to arrays for faster access? Lists are probably fine for small counts.
    # Average APs per position is 4 * n_aps / n.
    # n_aps approx n^2/6 ~ 10000. 4*10000/256 ~ 150 APs per position.
    
    def count_mono(color):
        """
        Vectorized count of monochromatic 4-APs.
        color: array of 0/1 of length n
        """
        c0 = color[aps[:, 0]]
        c1 = color[aps[:, 1]]
        c2 = color[aps[:, 2]]
        c3 = color[aps[:, 3]]
        mono = (c0 == c1) & (c1 == c2) & (c2 == c3)
        return int(np.sum(mono))
    
    def delta_flip(color, pos, current_count):
        """
        Calculate the change in mono count if we flip position 'pos'.
        Returns the new count.
        """
        delta = 0
        new_color_pos = 1 - color[pos]
        
        for ap_idx in pos_to_aps[pos]:
            # Get the 4 positions of this AP
            p0, p1, p2, p3 = aps[ap_idx]
            
            # Get current colors
            c0 = color[p0]
            c1 = color[p1]
            c2 = color[p2]
            c3 = color[p3]
            
            # Check if mono before
            was_mono = (c0 == c1) and (c1 == c2) and (c2 == c3)
            
            # Check if mono after flip
            # We only need to check the new color of 'pos' against the others
            # The others remain the same.
            # So, is (new_color_pos, c_other1, c_other2, c_other3) mono?
            # This is true if all others are equal to new_color_pos.
            
            # Find the indices of the other 3 positions
            # Since we know which position is 'pos', we can just check the other 3 values
            
            # Optimization: Instead of constructing new array, just check conditions
            # The AP is mono after flip iff:
            # 1. The 3 other positions are all equal to each other.
            # 2. That common value is equal to new_color_pos.
            
            # Let's extract the 3 other colors
            # We can do this by masking or simple logic
            # Since n is small, simple logic is fine
            
            # Identify which index is pos
            # But we can just collect the other 3
            other_colors = []
            if p0 == pos:
                other_colors = [c1, c2, c3]
            elif p1 == pos:
                other_colors = [c0, c2, c3]
            elif p2 == pos:
                other_colors = [c0, c1, c3]
            else: # p3 == pos
                other_colors = [c0, c1, c2]
            
            # Check if other_colors are all equal
            if other_colors[0] == other_colors[1] == other_colors[2]:
                if other_colors[0] == new_color_pos:
                    is_mono_after = True
                else:
                    is_mono_after = False
            else:
                is_mono_after = False
            
            if was_mono and not is_mono_after:
                delta -= 1
            elif not was_mono and is_mono_after:
                delta += 1
            # else: no change (0 or 1 -> 0 or 1)
                
        return current_count + delta

    def local_search(color, max_iters=2000):
        """
        Hill climbing local search.
        """
        color = color.copy()
        current = count_mono(color)
        
        # Pre-compute "was_mono" status for all APs? 
        # No, delta_flip calculates it on the fly. 
        # To speed up, we can maintain a boolean array `is_mono` for all APs.
        # Let's try that for a faster local search.
        
        # Re-initialize is_mono array
        c0 = color[aps[:, 0]]
        c1 = color[aps[:, 1]]
        c2 = color[aps[:, 2]]
        c3 = color[aps[:, 3]]
        is_mono = (c0 == c1) & (c1 == c2) & (c2 == c3)
        current = int(np.sum(is_mono))
        
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            best_delta = 0
            best_pos = -1
            
            # Evaluate all positions
            # We can vectorize the delta calculation? 
            # Delta for pos i depends on APs containing i.
            # It's hard to vectorize across positions easily without sparse matrix tricks.
            # Given n=256, looping 256 times is fast.
            
            for pos in range(n):
                # Calculate delta for flipping pos
                delta = 0
                new_color_pos = 1 - color[pos]
                
                for ap_idx in pos_to_aps[pos]:
                    # We need to know if this AP was mono, and if it will be.
                    # was_mono is is_mono[ap_idx]
                    # is_mono_after:
                    #   Check if the other 3 are equal to new_color_pos.
                    
                    p0, p1, p2, p3 = aps[ap_idx]
                    
                    # Check if other 3 are equal to new_color_pos
                    # This is equivalent to:
                    # (p0==pos ? new_color_pos : color[p0]) == ... etc.
                    # But we already know color[p0..3] except for pos.
                    
                    # Let's just check the condition directly using the current is_mono array?
                    # No, is_mono[ap_idx] is "was it mono".
                    # We need "will it be mono".
                    
                    # Will it be mono if:
                    # 1. The 3 other positions have color new_color_pos.
                    
                    # Get the other 3 colors
                    # We can pre-store the indices of the other 3 for each (pos, ap) pair?
                    # That's too much memory.
                    
                    # Simple check:
                    # Count how many of the 4 positions have color new_color_pos.
                    # If the pos itself is flipped to new_color_pos, and the other 3 are already new_color_pos, then it becomes mono.
                    # If the pos was mono before, it means all 4 were color[pos].
                    # After flip, pos is 1-color[pos]. The others are color[pos].
                    # So it becomes mono only if 1-color[pos] == color[pos], which is impossible.
                    # Wait.
                    # If it was mono (all c), and we flip pos (c -> 1-c), the AP is no longer mono.
                    # So if was_mono, delta is -1.
                    
                    # If it was NOT mono:
                    # It becomes mono if the other 3 are equal to new_color_pos.
                    
                    if is_mono[ap_idx]:
                        delta -= 1
                    else:
                        # Check if other 3 are new_color_pos
                        # We need to check colors of the other 3 positions.
                        # We can do this quickly.
                        
                        # Let's extract the other 3 indices
                        # To avoid if/else branching in the inner loop, we can just check all 4
                        # and substitute the flipped one.
                        
                        # c_new = np.array([color[p0], color[p1], color[p2], color[p3]])
                        # c_new[ap_idx == pos] = new_color_pos  # This is vectorized per AP, but we are looping APs
                        # if np.all(c_new == new_color_pos): delta += 1
                        
                        # Faster:
                        # Check if color[p0]==new_color_pos AND color[p1]==new_color_pos ...
                        # But one of them IS pos, so its color is new_color_pos by definition in the "after" state.
                        # So we just need to check the OTHER 3.
                        
                        # Let's find the other 3 indices
                        # We can precompute for each pos, and for each ap_idx in pos_to_aps[pos], the other 3 indices.
                        # Let's do that precomputation now to speed up the loop.
                        
                        pass # Placeholder, we will precompute outside
                
            # The inner loop is too slow if we do complex checks.
            # Let's precompute the "other 3 indices" for each (pos, ap) pair.
            # Actually, we can just precompute for each pos, a list of tuples (ap_idx, other_idx1, other_idx2, other_idx3).
            
            break # Break out of this inefficient local search implementation for now, let's rewrite properly.

    # --- Rewrite Local Search with Precomputation ---
    
    # Precompute for each position, the list of (ap_idx, other_idx_1, other_idx_2, other_idx_3)
    # This allows O(1) check for "will it be mono"
    pos_ap_info = [[] for _ in range(n)]
    for i in range(n_aps):
        p0, p1, p2, p3 = aps[i]
        # For p0, others are p1, p2, p3
        pos_ap_info[p0].append((i, p1, p2, p3))
        # For p1, others are p0, p2, p3
        pos_ap_info[p1].append((i, p0, p2, p3))
        # For p2, others are p0, p1, p3
        pos_ap_info[p2].append((i, p0, p1, p3))
        # For p3, others are p0, p1, p2
        pos_ap_info[p3].append((i, p0, p1, p2))
    
    def local_search_fast(color, max_iters=5000):
        color = color.copy()
        
        # Initial is_mono calculation
        c0 = color[aps[:, 0]]
        c1 = color[aps[:, 1]]
        c2 = color[aps[:, 2]]
        c3 = color[aps[:, 3]]
        is_mono = (c0 == c1) & (c1 == c2) & (c2 == c3)
        current = int(np.sum(is_mono))
        
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            best_delta = 0
            best_pos = -1
            
            for pos in range(n):
                delta = 0
                new_color_pos = 1 - color[pos]
                
                for ap_idx, o1, o2, o3 in pos_ap_info[pos]:
                    if is_mono[ap_idx]:
                        # Was mono, now not mono (since pos flips)
                        delta -= 1
                    else:
                        # Was not mono.
                        # Will be mono if other 3 are equal to new_color_pos
                        if color[o1] == new_color_pos and color[o2] == new_color_pos and color[o3] == new_color_pos:
                            delta += 1
                
                if delta < best_delta:
                    best_delta = delta
                    best_pos = pos
            
            if best_pos != -1:
                # Apply flip
                color[best_pos] = 1 - color[best_pos]
                
                # Update is_mono for affected APs
                new_color_pos = color[best_pos]
                for ap_idx, o1, o2, o3 in pos_ap_info[best_pos]:
                    # Check if mono now
                    if color[o1] == new_color_pos and color[o2] == new_color_pos and color[o3] == new_color_pos:
                        is_mono[ap_idx] = True
                    else:
                        is_mono[ap_idx] = False
                    # Note: The APs containing best_pos are exactly those in pos_ap_info[best_pos]
                    
                current += best_delta
                improved = True
        
        return color, current

    # --- Main Search Loop ---
    
    best_color = None
    best_score = float('inf')
    
    # Initial Candidates
    # 1. Thue-Morse
    tm = np.array([bin(i).count('1') % 2 for i in range(n)], dtype=np.int8)
    candidates = [tm.copy()]
    
    # 2. Random starts
    for _ in range(5):
        candidates.append(np.random.randint(0, 2, n).astype(np.int8))
    
    # 3. Maybe a "block" pattern?
    # Not strictly necessary, random + TM should be enough.
    
    for start in candidates:
        if time.time() > deadline - 10.0:
            break
            
        c, s = local_search_fast(start)
        
        if s < best_score:
            best_score = s
            best_color = c.copy()
            
        # If we found a very good solution, we might try to refine it further with more iterations?
        # local_search_fast already runs until no improvement.
        
    # Final check
    if best_color is None:
        # Fallback
        best_color = np.zeros(n, dtype=np.int8)
        best_score = count_mono(best_color)
        
    return [int(c) for c in best_color]

# Run construct to test? No, just define it.
# The evaluator will call construct().

if __name__ == "__main__":
    # Simple test
    res = construct()
    print(len(res))
    print(res[:10])
