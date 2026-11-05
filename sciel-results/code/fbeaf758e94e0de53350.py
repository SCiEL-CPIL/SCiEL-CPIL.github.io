import time
import numpy as np

def construct():
    n = 256
    deadline = time.time() + 115  # safety margin for 120s limit
    
    # Precompute all 4-term APs: (a, a+d, a+2d, a+3d) with indices 0..255
    ap_list = []
    for d in range(1, n // 3 + 1):
        max_a = n - 1 - 3 * d
        for a in range(max_a + 1):
            ap_list.append((a, a + d, a + 2 * d, a + 3 * d))
    
    num_aps = len(ap_list)
    ap_array = np.array(ap_list, dtype=np.int32)  # shape (num_aps, 4)
    
    # Helper: count mono APs given a coloring (0/1 array)
    def count_mono(coloring):
        # coloring is np array of 0/1, length n
        # For each AP, check if all 4 are same
        c = coloring[ap_array]  # shape (num_aps, 4)
        # mono if all equal: c[:,0]==c[:,1] and c[:,1]==c[:,2] and c[:,2]==c[:,3]
        mono = (c[:, 0] == c[:, 1]) & (c[:, 1] == c[:, 2]) & (c[:, 2] == c[:, 3])
        return int(np.sum(mono))
    
    # Thue-Morse initialization
    def thue_morse(n):
        # Thue-Morse: t(i) = parity of popcount(i)
        tm = np.zeros(n, dtype=np.int32)
        for i in range(n):
            # Count bits
            x = i
            cnt = 0
            while x:
                cnt += x & 1
                x >>= 1
            tm[i] = cnt % 2
        return tm
    
    # Vectorized delta computation: for flipping bit i, compute change in mono count
    def compute_all_deltas(coloring):
        """
        For each position i (0..n-1), compute the delta in mono count if we flip coloring[i].
        Uses the fact that flipping i affects all APs containing i.
        """
        # For each AP, determine its current mono status
        c = coloring[ap_array]  # (num_aps, 4)
        is_mono = (c[:, 0] == c[:, 1]) & (c[:, 1] == c[:, 2]) & (c[:, 2] == c[:, 3])
        
        deltas = np.zeros(n, dtype=np.int32)
        
        for p in range(4):
            # For APs where position p is the one being flipped
            # The other 3 positions are the other indices
            other = [q for q in range(4) if q != p]
            other_vals = c[:, other]  # (num_aps, 3)
            # The 3 other values are all equal?
            others_equal = (other_vals[:, 0] == other_vals[:, 1]) & (other_vals[:, 1] == other_vals[:, 2])
            # The flipped value's new value is 1 - c[:, p]
            # For the AP to become mono after flip: others_equal AND others[0] == 1 - c[:, p]
            would_become_mono = others_equal & (other_vals[:, 0] == 1 - c[:, p])
            
            # For each AP j, the position being flipped is ap_array[j, p]
            pos = ap_array[:, p]  # (num_aps,)
            
            # For each AP j:
            #   if it was mono and flipping makes it non-mono: delta -= 1
            #   if it was non-mono and flipping makes it mono: delta += 1
            #   Note: if it was mono, flipping one element always makes it non-mono (since the flipped one differs)
            #   So: delta for AP j = (would_become_mono ? +1 : 0) - (is_mono ? 1 : 0)
            
            delta_j = np.where(would_become_mono, 1, 0) - np.where(is_mono, 1, 0)
            
            # Accumulate into deltas at position pos
            np.add.at(deltas, pos, delta_j)
        
        return deltas

    # Batched Local Search: flip all positions with negative delta simultaneously
    def batch_local_search(coloring, max_sweeps=10000):
        current = coloring.copy()
        current_count = count_mono(current)
        
        while time.time() < deadline:
            deltas = compute_all_deltas(current)
            # Find positions with negative delta (improvement)
            improving_positions = np.where(deltas < 0)[0]
            
            if len(improving_positions) == 0:
                break
            
            # Flip all improving positions
            current[improving_positions] = 1 - current[improving_positions]
            
            # Update count
            # Note: Since we flipped multiple bits, the simple sum of deltas is not accurate 
            # because flipping one bit affects the delta of others. 
            # However, we can just recount. Recounting is O(APs) which is fast.
            new_count = count_mono(current)
            
            # If count didn't improve (shouldn't happen with greedy batch unless oscillation), break
            if new_count >= current_count:
                # Revert? Or just stop? 
                # Greedy batch usually improves. If it doesn't, we might be in a local trap or oscillation.
                # Let's just break to be safe, or revert. Reverting is safer.
                current[improving_positions] = 1 - current[improving_positions]
                break
                
            current_count = new_count
        
        return current, current_count

    # Initialize with Thue-Morse
    coloring = thue_morse(n)
    best_coloring = coloring.copy()
    best_count = count_mono(coloring)
    
    # Multiple restarts with batch local search
    rng = np.random.RandomState(42)
    num_restarts = 1000 # Can afford more restarts because each is fast
    
    for restart in range(num_restarts):
        if time.time() > deadline - 2:
            break
        
        if restart == 0:
            start = coloring.copy()
        elif restart % 5 == 1:
            # Random start
            start = rng.randint(0, 2, n).astype(np.int32)
        else:
            # Perturb the best solution
            start = best_coloring.copy()
            # Flip a random subset
            num_flips = rng.randint(5, 30)
            flip_positions = rng.choice(n, size=num_flips, replace=False)
            start[flip_positions] = 1 - start[flip_positions]
        
        # Batch Local Search
        ls_result, ls_count = batch_local_search(start, max_sweeps=500)
        
        if ls_count < best_count:
            best_count = ls_count
            best_coloring = ls_result.copy()
            
            # If we found a new best, do a few more perturbations from it immediately
            # This is handled by the loop logic, but we can force a few quick checks
            # Actually, the loop continues, so it's fine.

    # Final polish: try to improve best_coloring further with batch search
    if time.time() < deadline - 1:
        final_result, final_count = batch_local_search(best_coloring, max_sweeps=1000)
        if final_count < best_count:
            best_count = final_count
            best_coloring = final_result.copy()
    
    # Return as list of 0/1 for integers 1..256
    return [int(x) for x in best_coloring]
