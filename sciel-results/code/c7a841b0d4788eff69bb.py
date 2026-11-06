import time
import random
import numpy as np

def construct():
    n = 256
    deadline = time.time() + 118.0
    
    # Precompute all 4-term arithmetic progressions (a, a+d, a+2d, a+3d)
    # Indices are 0-based in numpy, so positions are 0..255
    aps = []
    for d in range(1, n // 3 + 1):
        max_a = n - 3 * d
        for a in range(0, max_a):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    num_aps = len(aps)
    if num_aps == 0:
        return [0] * n
        
    # Convert APs to a numpy array for vectorized operations
    # Shape: (num_aps, 4)
    ap_array = np.array(aps, dtype=np.int32)
    
    # Build mapping: position -> list of AP indices containing it
    ap_indices_at_pos = [[] for _ in range(n)]
    for idx, ap in enumerate(aps):
        for pos in ap:
            ap_indices_at_pos[pos].append(idx)
    
    # Precompute "other 3" indices for each (ap_idx, position_in_ap) pair
    # This allows faster delta calculation.
    # For each AP index and each of its 4 positions, store the indices of the other 3 positions.
    # We can store this as a 2D array: others_for_flip[ap_idx, pos_in_ap] = array of 3 indices
    # But pos_in_ap is 0,1,2,3.
    # Actually, we can just precompute for each position i in the AP, the mask of the others.
    # Let's create a structure: for each i in 0..n-1, list of (ap_idx, other_indices_array)
    # This is heavy to precompute fully, so let's just compute others on the fly but efficiently.
    
    # Better: Precompute for each AP and each of its 4 slots, the 3 other indices.
    # others[ap_idx, slot] = np.array([i1, i2, i3])
    others = np.zeros((num_aps, 4, 3), dtype=np.int32)
    for idx, ap in enumerate(aps):
        a, b, c, d = ap
        others[idx, 0] = [b, c, d]
        others[idx, 1] = [a, c, d]
        others[idx, 2] = [a, b, d]
        others[idx, 3] = [a, b, c]
        
    # Map position to (ap_idx, slot) pairs for fast lookup
    # pos_to_ap_slot[i] = list of (ap_idx, slot)
    pos_to_ap_slot = [[] for _ in range(n)]
    for idx, ap in enumerate(aps):
        for slot, pos in enumerate(ap):
            pos_to_ap_slot[pos].append((idx, slot))

    def compute_score_and_states(color):
        """
        Computes the number of monochromatic APs and returns the state array.
        """
        # Vectorized check:
        # For each AP, check if all 4 colors are equal.
        # c[ap_array[:, 0]] == c[ap_array[:, 1]] == c[ap_array[:, 2]] == c[ap_array[:, 3]]
        c0 = color[ap_array[:, 0]]
        c1 = color[ap_array[:, 1]]
        c2 = color[ap_array[:, 2]]
        c3 = color[ap_array[:, 3]]
        
        # Mono if all equal
        # (c0==c1) & (c1==c2) & (c2==c3)
        states = ((c0 == c1) & (c1 == c2) & (c2 == c3)).astype(np.int8)
        score = np.sum(states)
        return score, states

    def calculate_delta(color, states, i):
        """
        Calculate the change in score if we flip bit at index i.
        Returns delta (negative is improvement).
        """
        delta = 0
        # Iterate over APs containing i
        for ap_idx, slot in pos_to_ap_slot[i]:
            if states[ap_idx] == 1:
                # Currently mono. Flipping i breaks it.
                delta -= 1
            else:
                # Currently not mono.
                # Check if other 3 are equal.
                others_idx = others[ap_idx, slot]
                c_other = color[others_idx]
                if c_other[0] == c_other[1] == c_other[2]:
                    # Others are same.
                    # If current color[i] != c_other[0], flipping makes it equal -> becomes mono.
                    # If current color[i] == c_other[0], it was already mono (handled above).
                    # So if not mono, color[i] must be != c_other[0].
                    # Flipping it makes it == c_other[0]. So it becomes mono.
                    delta += 1
        return delta

    def update_states_after_flip(color, states, i):
        """
        Update the states array after flipping bit i.
        """
        for ap_idx, slot in pos_to_ap_slot[i]:
            # Recalculate state for this AP
            a, b, c, d = ap_array[ap_idx]
            if color[a] == color[b] == color[c] == color[d]:
                states[ap_idx] = 1
            else:
                states[ap_idx] = 0

    best_score = float('inf')
    best_color = None
    
    # Initializations
    # 1. Thue-Morse
    init_thue_morse = np.array([bin(i).count('1') % 2 for i in range(n)], dtype=np.int8)
    # 2. Random
    init_random = np.random.randint(0, 2, n).astype(np.int8)
    # 3. Structured (e.g., alternating blocks)
    init_structured = np.zeros(n, dtype=np.int8)
    init_structured[0::4] = 0
    init_structured[1::4] = 1
    init_structured[2::4] = 1
    init_structured[3::4] = 0
    # Another structured: Rademacher-like or just random block
    init_blocks = np.repeat(np.random.randint(0, 2, 64), 4).astype(np.int8)
    
    initializations = [init_thue_morse, init_random, init_structured, init_blocks]
    
    for init_idx, init_color in enumerate(initializations):
        if time.time() > deadline - 10.0:
            break
            
        color = init_color.copy()
        score, states = compute_score_and_states(color)
        
        if score < best_score:
            best_score = score
            best_color = color.copy()
            
        # Simulated Annealing
        # T starts high, decays
        T_start = 10.0
        T_end = 0.01
        # Number of iterations per restart
        # Estimate max iterations possible in remaining time
        # Let's just run until time limit or convergence
        T = T_start
        alpha = 0.999 # Cooling rate
        
        while time.time() < deadline - 5.0:
            # Pick random index
            i = random.randint(0, n - 1)
            
            delta = calculate_delta(color, states, i)
            
            # Acceptance probability
            if delta < 0:
                accept = True
            else:
                # Avoid log(0)
                if T < 1e-9:
                    accept = False
                else:
                    accept = random.random() < np.exp(-delta / T)
            
            if accept:
                # Flip
                color[i] = 1 - color[i]
                score += delta
                update_states_after_flip(color, states, i)
                
                if score < best_score:
                    best_score = score
                    best_color = color.copy()
            
            # Cool down
            T *= alpha
            if T < T_end:
                break
                
    return best_color.tolist()
