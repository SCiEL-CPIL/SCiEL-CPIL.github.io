import time
import random
import math

def construct():
    """
    Finds a 5-coloring of K_n without monochromatic C4s, maximizing n.
    Uses Simulated Annealing with efficient violation counting.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # We want to maximize n.
    # Try from high n down to a safe lower bound.
    # n=26 is the known max. n=21+ is a good target to beat 20.
    for target_n in range(26, 20, -1):
        if time.time() > deadline:
            break
        
        # Allocate time for this target_n
        # We have 5 targets (26, 25, 24, 23, 22, 21).
        # Give more time to higher n? Or equal?
        # Let's give equal time to each, or slightly more to 26.
        remaining = deadline - time.time()
        steps_left = target_n - 20
        time_per_n = remaining / steps_left
        
        colors = _solve_n(target_n, time_per_n)
        if colors is not None:
            # Flatten colors to list for return
            # The interface expects: one colour for each pair (i,j), i<j, in lexicographic order
            flat_colors = []
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    flat_colors.append(colors[i][j])
            
            return {'n': target_n, 'colors': flat_colors}
    
    # Fallback: Return the seed solution if nothing better found
    n = 11
    colors_list = [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]
    return {'n': n, 'colors': colors_list}

def _solve_n(n, time_limit):
    """
    Attempts to find a valid 5-coloring of K_n.
    Returns a 2D list of colors if successful, else None.
    """
    if n <= 1:
        return [[0] * n for _ in range(n)]
    
    # Initialize random coloring
    colors = [[0] * n for _ in range(n)]
    adj = [[0] * n for _ in range(5)]  # adj[c][i] is bitmask of neighbors of i in color c
    common = [[[0] * n for _ in range(n)] for _ in range(5)] # common[c][i][j]
    
    # Fill colors and adj
    for i in range(n):
        for j in range(i + 1, n):
            c = random.randint(0, 4)
            colors[i][j] = c
            colors[j][i] = c
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
            
    # Calculate initial common counts and total violations
    total_violations = 0
    for c in range(5):
        for i in range(n):
            # For each neighbor j of i in color c
            mask = adj[c][i]
            while mask:
                lsb = mask & -mask
                j = lsb.bit_length() - 1
                mask ^= lsb
                if j > i: # Only count each pair once? 
                    # common[c][i][j] is symmetric. 
                    # We need to compute common[c][i][j] for all i,j.
                    # common[c][i][j] = popcount(adj[c][i] & adj[c][j])
                    pass
        
    # Efficient initial calculation:
    # common[c][i][j] = popcount(adj[c][i] & adj[c][j])
    for c in range(5):
        for i in range(n):
            for j in range(i + 1, n):
                common_c = adj[c][i] & adj[c][j]
                cnt = common_c.bit_count()
                common[c][i][j] = cnt
                common[c][j][i] = cnt
                if cnt >= 2:
                    total_violations += cnt * (cnt - 1) // 2
                    
    best_colors = [row[:] for row in colors]
    best_violations = total_violations
    
    if total_violations == 0:
        return best_colors
        
    # Simulated Annealing
    T = 5.0
    cooling = 0.999
    min_T = 0.01
    
    # Precompute list of edges for random selection
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    while time.time() < time_limit and T > min_T:
        # Pick random edge
        i, j = random.choice(edges)
        old_c = colors[i][j]
        
        # Try a new color different from old_c
        new_c = random.randint(0, 4)
        if new_c == old_c:
            continue
            
        # Calculate delta V
        delta_v = 0
        
        # Remove from old_c
        # Neighbors of i in old_c (excluding j)
        mask_i = adj[old_c][i] & ~(1 << j)
        while mask_i:
            lsb = mask_i & -mask_i
            k = lsb.bit_length() - 1
            mask_i ^= lsb
            # common[old_c][j][k] decreases by 1
            cnt = common[old_c][j][k]
            if cnt > 0:
                delta_v -= (cnt - 1)
                
        mask_j = adj[old_c][j] & ~(1 << i)
        while mask_j:
            lsb = mask_j & -mask_j
            k = lsb.bit_length() - 1
            mask_j ^= lsb
            # common[old_c][i][k] decreases by 1
            cnt = common[old_c][i][k]
            if cnt > 0:
                delta_v -= (cnt - 1)
                
        # Add to new_c
        mask_i = adj[new_c][i] & ~(1 << j)
        while mask_i:
            lsb = mask_i & -mask_i
            k = lsb.bit_length() - 1
            mask_i ^= lsb
            # common[new_c][j][k] increases by 1
            cnt = common[new_c][j][k]
            delta_v += cnt
            
        mask_j = adj[new_c][j] & ~(1 << i)
        while mask_j:
            lsb = mask_j & -mask_j
            k = lsb.bit_length() - 1
            mask_j ^= lsb
            # common[new_c][i][k] increases by 1
            cnt = common[new_c][i][k]
            delta_v += cnt

        # Acceptance criterion
        if delta_v < 0 or random.random() < math.exp(-delta_v / T):
            # Apply move
            total_violations += delta_v
            colors[i][j] = new_c
            colors[j][i] = new_c
            
            # Update adj
            adj[old_c][i] ^= (1 << j)
            adj[old_c][j] ^= (1 << i)
            adj[new_c][i] ^= (1 << j)
            adj[new_c][j] ^= (1 << i)
            
            # Update common
            # We already calculated the deltas based on old values.
            # Now we must update the common array values to reflect the change.
            
            # For old_c:
            mask_i = adj[old_c][i] | (1 << j) # Wait, we already updated adj. 
            # We need to iterate over the neighbors that were affected.
            # The neighbors were those in the old adj BEFORE the move.
            # Let's re-iterate using the current adj (after move) is wrong because we removed j.
            # We should have stored the neighbors or iterate carefully.
            
            # Actually, the logic above for delta_v used the state BEFORE the move.
            # Now we need to update `common` to the state AFTER the move.
            
            # For old_c, j is no longer a neighbor of i, and i is no longer a neighbor of j.
            # The common counts for pairs (j,k) and (i,k) changed.
            
            # Let's re-do the update of common arrays.
            # This is tricky to do in-place without storing the previous neighbor list.
            # However, we can just re-calculate the affected common values? No, that's O(n).
            
            # Let's store the affected ks? No, just iterate again.
            # But adj[old_c][i] has changed.
            
            # Let's use a helper to update common.
            # It's safer to just update the specific common entries.
            
            # For old_c:
            # We removed edge (i,j).
            # For each k in N_old(i) \ {j}: common[old_c][j][k] -= 1
            # For each k in N_old(j) \ {i}: common[old_c][i][k] -= 1
            
            # We need N_old(i). We have adj[old_c][i] which is NOW N_new(i).
            # N_old(i) = N_new(i) | {j}
            
            mask_i_old = adj[old_c][i] | (1 << j)
            while mask_i_old:
                lsb = mask_i_old & -mask_i_old
                k = lsb.bit_length() - 1
                mask_i_old ^= lsb
                if k != j:
                    common[old_c][j][k] -= 1
                    common[old_c][k][j] -= 1
                    
            mask_j_old = adj[old_c][j] | (1 << i)
            while mask_j_old:
                lsb = mask_j_old & -mask_j_old
                k = lsb.bit_length() - 1
                mask_j_old ^= lsb
                if k != i:
                    common[old_c][i][k] -= 1
                    common[old_c][k][i] -= 1
                    
            # For new_c:
            # We added edge (i,j).
            # N_new(i) = N_old(i) | {j}
            # For each k in N_new(i) \ {j}: common[new_c][j][k] += 1
            # For each k in N_new(j) \ {i}: common[new_c][i][k] += 1
            
            mask_i_new = adj[new_c][i]
            while mask_i_new:
                lsb = mask_i_new & -mask_i_new
                k = lsb.bit_length() - 1
                mask_i_new ^= lsb
                if k != j:
                    common[new_c][j][k] += 1
                    common[new_c][k][j] += 1
                    
            mask_j_new = adj[new_c][j]
            while mask_j_new:
                lsb = mask_j_new & -mask_j_new
                k = lsb.bit_length() - 1
                mask_j_new ^= lsb
                if k != i:
                    common[new_c][i][k] += 1
                    common[new_c][k][i] += 1
            
            if total_violations < best_violations:
                best_violations = total_violations
                best_colors = [row[:] for row in colors]
                
            if best_violations == 0:
                return best_colors
                
        T *= cooling
        
    return None
