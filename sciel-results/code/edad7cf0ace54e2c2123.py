import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    limit = 145.0  # Safety margin before 150s kill
    
    n = 256
    
    # Precompute all 4-term arithmetic progressions
    aps = []
    for d in range(1, (n - 3) // 3 + 1):
        max_a = n - 3 * d
        for a in range(1, max_a + 1):
            i1 = a - 1
            i2 = a + d - 1
            i3 = a + 2 * d - 1
            i4 = a + 3 * d - 1
            aps.append((i1, i2, i3, i4))
    
    total_aps = len(aps)
    aps_arr = np.array(aps, dtype=np.int32)  # shape (total_aps, 4)
    
    # Build position-to-AP mapping as a list of arrays
    pos_to_aps = [[] for _ in range(n)]
    for idx in range(total_aps):
        for pos in aps_arr[idx]:
            pos_to_aps[pos].append(idx)
    pos_to_aps_arr = [np.array(lst, dtype=np.int32) for lst in pos_to_aps]
    
    def count_mono(coloring):
        c = coloring[aps_arr]  # shape (total_aps, 4)
        return int(np.all(c == c[:, 0:1], axis=1).sum())
    
    def compute_deltas(coloring):
        """Compute delta for flipping each position. Vectorized."""
        c = coloring[aps_arr]  # shape (total_aps, 4)
        # For each AP, count how many positions have the same color as position 0
        # Actually, we need: for each AP, is it currently mono? And what would happen if we flip each of its 4 positions?
        
        # More efficient: for each AP, the number of "agreements" with the majority
        # An AP is mono if all 4 are same.
        # Flipping position p in an AP: 
        #   If currently mono and p was the flipped one: becomes non-mono (delta -1)
        #   If currently non-mono and after flip becomes mono: delta +1
        
        # Let's compute for each AP: how many of the 4 positions have color 0 and how many have color 1
        # If count_0 == 4 or count_1 == 4: mono
        # If we flip a position with color 0: count_0 -= 1, count_1 += 1
        #   New is mono if count_0_new == 4 (impossible since we decreased) or count_1_new == 4
        #   count_1_new == 4 means count_1 was 3 and count_0 was 1, and we flipped the single 0
        # Similarly for flipping a 1.
        
        # Let me think differently. For each AP:
        #   n0 = number of 0s, n1 = number of 1s (n0 + n1 = 4)
        #   Currently mono if n0==4 or n1==4
        #   If we flip a 0: new n0 = n0-1, new n1 = n1+1. Mono if new_n1==4, i.e., n1==3 and n0==1
        #   If we flip a 1: new n0 = n0+1, new n1 = n1-1. Mono if new_n0==4, i.e., n0==3 and n1==1
        
        # So for each AP:
        #   Let pos0_color = color of position 0 in the AP
        #   n0 = sum(1 for x in AP if x==0)
        #   If n0==4: mono. Flipping any position (which is 0) makes it non-mono. Delta = -1 for each of the 4 positions.
        #   If n1==4: mono. Flipping any position (which is 1) makes it non-mono. Delta = -1 for each of the 4 positions.
        #   If n0==3, n1==1: Not mono. Flipping the single 1 makes it all 0s (mono). Delta = +1 for that position.
        #   If n0==1, n1==3: Not mono. Flipping the single 0 makes it all 1s (mono). Delta = +1 for that position.
        #   If n0==2, n1==2: Not mono. Flipping any position keeps it non-mono. Delta = 0.
        
        # So the delta for a position p is:
        #   Sum over APs containing p of:
        #     -1 if AP is mono (all 4 same) and p has that color
        #     +1 if AP has 3 of one color and 1 of the other, and p is the odd one out
        #     0 otherwise
        
        # Vectorized approach:
        # For each AP, compute n0 (number of zeros)
        c = coloring[aps_arr]  # (total_aps, 4)
        n0 = (c == 0).sum(axis=1)  # (total_aps,)
        n1 = 4 - n0
        
        # mono_mask: APs that are currently mono
        mono_mask = (n0 == 4) | (n1 == 4)
        
        # For mono APs, flipping any of the 4 positions gives delta -1
        # For non-mono APs with n0==3 (n1==1), flipping the position that is 1 gives delta +1
        # For non-mono APs with n1==3 (n0==1), flipping the position that is 0 gives delta +1
        
        # Let's build the delta array
        deltas = np.zeros(n, dtype=np.int32)
        
        # Process mono APs: for each mono AP, subtract 1 from all 4 positions
        if mono_mask.any():
            mono_aps = aps_arr[mono_mask]  # (k, 4)
            # Use bincount to accumulate
            mono_flat = mono_aps.flatten()
            deltas -= np.bincount(mono_flat, minlength=n).astype(np.int32)
        
        # Process 3-1 APs (n0==3, n1==1): the odd one out is the position with color 1
        mask_31 = (n0 == 3) & (n1 == 1)
        if mask_31.any():
            ap_indices_31 = np.where(mask_31)[0]
            for idx in ap_indices_31:
                ap = aps_arr[idx]
                colors = c[idx]
                # Find the position with color 1
                odd_pos = ap[colors == 1][0]
                deltas[odd_pos] += 1
        
        # Process 1-3 APs (n0==1, n1==3): the odd one out is the position with color 0
        mask_13 = (n0 == 1) & (n1 == 3)
        if mask_13.any():
            ap_indices_13 = np.where(mask_13)[0]
            for idx in ap_indices_13:
                ap = aps_arr[idx]
                colors = c[idx]
                # Find the position with color 0
                odd_pos = ap[colors == 0][0]
                deltas[odd_pos] += 1
        
        return deltas
    
    def improve(coloring, max_sweeps=200, time_limit=None):
        """Greedy local search: repeatedly flip the position with the most negative delta."""
        current_score = count_mono(coloring)
        
        for sweep in range(max_sweeps):
            if time_limit is not None and time.time() > time_limit:
                break
            
            deltas = compute_deltas(coloring)
            min_delta = deltas.min()
            
            if min_delta >= 0:
                break  # No improvement possible
            
            # Flip all positions with delta == min_delta (most negative)
            flip_mask = (deltas == min_delta)
            flip_positions = np.where(flip_mask)[0]
            
            coloring[flip_positions] = 1 - coloring[flip_positions]
            current_score += min_delta * len(flip_positions)
        
        return coloring, current_score
    
    best_coloring = None
    best_score = float('inf')
    
    # Random restarts with local search
    rng = np.random.default_rng(42)
    
    while time.time() < limit - 5.0:
        # Initialize random coloring
        coloring = rng.integers(0, 2, n)
        
        # Quick improvement
        coloring, score = improve(coloring, max_sweeps=50, time_limit=limit - 3.0)
        
        if score < best_score:
            best_score = score
            best_coloring = coloring.copy()
        
        # Check if we have time for more
        if time.time() > limit - 10.0:
            break
    
    # Final polishing: do more sweeps on the best coloring
    if best_coloring is not None and time.time() < limit - 2.0:
        # Try a few more rounds of improvement on the best
        for _ in range(5):
            if time.time() > limit - 2.0:
                break
            coloring = best_coloring.copy()
            coloring, score = improve(coloring, max_sweeps=100, time_limit=limit - 1.0)
            if score < best_score:
                best_score = score
                best_coloring = coloring.copy()
    
    if best_coloring is None:
        return [int(x) for x in np.random.randint(0, 2, n)]
    return [int(x) for x in best_coloring]
# EVOLVE-BLOCK-END
