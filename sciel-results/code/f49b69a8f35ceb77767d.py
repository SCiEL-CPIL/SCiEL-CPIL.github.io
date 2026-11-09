import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 115.0  # safety margin before 120s kill
    n = 256
    
    # Precompute all 4-term APs as index tuples (0-indexed)
    # a, a+d, a+2d, a+3d with 1<=a, d>=1, a+3d<=256
    aps = []
    for d in range(1, n // 3 + 1):
        max_a = n - 3 * d
        for a in range(1, max_a + 1):
            i0 = a - 1
            i1 = a + d - 1
            i2 = a + 2 * d - 1
            i3 = a + 3 * d - 1
            aps.append((i0, i1, i2, i3))
            
    aps_arr = np.array(aps, dtype=np.int32)
    M = len(aps_arr)
    
    # Build per-position list of AP indices for delta computation
    # pos_aps[i] contains indices of APs where position i is involved
    # We need to know which position (0,1,2,3) it is in to compute the flip effect
    # Actually, for a flip at position i, the change in count for an AP (i0,i1,i2,i3) is:
    # Let c be the current color at i. New color is 1-c.
    # Let others be the colors of the other 3 positions.
    # Old mono count for this AP: 1 if all 4 same, else 0.
    # New mono count for this AP: 1 if (1-c) == other1 == other2 == other3, else 0.
    
    # To make this fast, we can precompute for each AP index the other 3 positions.
    # But even better: for each position i, store a list of (ap_idx, other_indices)
    # Or simply: for each position i, we can just iterate through the APs containing i.
    
    # Let's use a different vectorized approach for the whole objective if needed, 
    # but for SA, we need fast delta.
    
    # Structure:
    # pos_to_aps[i] = list of (ap_idx, pos_in_ap)
    # But we can be smarter. 
    # For a flip at i:
    # Delta = sum over APs containing i of (new_mono - old_mono)
    
    # Let's precompute for each position i, the list of APs it belongs to.
    pos_aps = [[] for _ in range(n)]
    for idx, ap in enumerate(aps):
        for pos_idx, val in enumerate(ap):
            pos_aps[val].append((idx, pos_idx))
            
    # Initial coloring: Random or Thue-Morse? 
    # Thue-Morse is a good start. Let's use it.
    # Thue-Morse: t(i) = popcount(i) % 2
    def thue_morse(i):
        return bin(i).count('1') % 2
        
    colors = np.array([thue_morse(i) for i in range(n)], dtype=np.int8)
    
    # Function to calculate total mono count
    def count_mono(c):
        # c is array of 0/1
        mono = 0
        for ap in aps_arr:
            if c[ap[0]] == c[ap[1]] == c[ap[2]] == c[ap[3]]:
                mono += 1
        return mono
        
    # Faster vectorized count for initial eval
    def count_mono_vec(c):
        # c[aps_arr] gives shape (M, 4)
        # Check if all 4 are equal
        # c[aps_arr[:,0]] == c[aps_arr[:,1]] == ...
        return np.sum((c[aps_arr[:,0]] == c[aps_arr[:,1]]) & 
                      (c[aps_arr[:,1]] == c[aps_arr[:,2]]) & 
                      (c[aps_arr[:,2]] == c[aps_arr[:,3]]))
                      
    current_score = count_mono_vec(colors)
    best_score = current_score
    best_colors = colors.copy()
    
    # Simulated Annealing Parameters
    T = 5.0
    T_min = 0.01
    cooling_rate = 0.9995
    
    # Precompute "other" indices for fast delta?
    # For a flip at position i:
    # For each AP idx in pos_aps[i]:
    #   Let j, k, l be the other 3 positions.
    #   old_val = c[i]
    #   new_val = 1 - old_val
    #   others = c[j], c[k], c[l]
    #   old_mono = 1 if old_val == others[0] == others[1] == others[2] else 0
    #   new_mono = 1 if new_val == others[0] == others[1] == others[2] else 0
    #   delta += new_mono - old_mono
    
    # To speed up delta, we can precompute for each position i, 
    # the list of other-indices tuples for each AP it's in.
    # pos_aps_detail[i] = list of (j, k, l) where (i, j, k, l) is an AP (in some order)
    # Actually, we just need the indices of the other 3.
    
    pos_others = [[] for _ in range(n)]
    for idx, ap in enumerate(aps_arr):
        for pos_idx in range(4):
            i = ap[pos_idx]
            others = [ap[p] for p in range(4) if p != pos_idx]
            pos_others[i].append(others)
            
    # SA Loop
    while time.time() < deadline:
        if T < T_min:
            break
            
        # Pick random position
        i = np.random.randint(n)
        
        # Calculate delta
        delta = 0
        old_val = colors[i]
        new_val = 1 - old_val
        
        for others in pos_others[i]:
            j, k, l = others
            c_j, c_k, c_l = colors[j], colors[k], colors[l]
            
            # Old mono
            if old_val == c_j and c_j == c_k and c_k == c_l:
                old_mono = 1
            else:
                old_mono = 0
                
            # New mono
            if new_val == c_j and c_j == c_k and c_k == c_l:
                new_mono = 1
            else:
                new_mono = 0
                
            delta += (new_mono - old_mono)
            
        # Acceptance
        if delta < 0:
            # Improvement
            colors[i] = new_val
            current_score += delta
            if current_score < best_score:
                best_score = current_score
                best_colors = colors.copy()
        else:
            # Worse or equal
            # P(accept) = exp(-delta / T)
            if delta == 0:
                # Accept neutral moves to explore
                if np.random.random() < 0.5:
                    colors[i] = new_val
                    current_score += delta
            else:
                r = np.random.random()
                if r < np.exp(-delta / T):
                    colors[i] = new_val
                    current_score += delta
                    
        T *= cooling_rate
        
    return [int(c) for c in best_colors]
