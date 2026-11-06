import time
import random
import numpy as np

def construct():
    n = 256
    # Deadline: 150s limit, aim for 145s to be safe. 
    # However, the prompt says 120s for the function. 
    # "Each evaluation must finish within 120 s."
    # So deadline is 118s.
    deadline = time.time() + 118.0
    
    # Initialize with Thue-Morse sequence
    # Thue-Morse is a classic 2-coloring that minimizes short APs
    color = np.array([bin(i).count('1') % 2 for i in range(n)], dtype=np.int8)
    
    # Precompute all 4-term arithmetic progressions (a, a+d, a+2d, a+3d)
    # d >= 1, a+3d <= n
    # Indices are 0-based in numpy, so positions are 0..255
    # APs are (i, i+d, i+2d, i+3d)
    aps = []
    for d in range(1, n // 3 + 1):
        max_a = n - 3 * d
        for a in range(0, max_a):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    num_aps = len(aps)
    if num_aps == 0:
        return color.tolist()
        
    # Convert APs to a numpy array for vectorized operations
    # Shape: (num_aps, 4)
    ap_array = np.array(aps, dtype=np.int32)
    
    # --- Incremental Delta Update Setup ---
    # We need to know, for each position i (0..n-1), which APs contain i.
    # Let's build a list of lists: ap_indices_at_pos[i] = list of indices in ap_array where i appears.
    ap_indices_at_pos = [[] for _ in range(n)]
    for idx, ap in enumerate(aps):
        for pos in ap:
            ap_indices_at_pos[pos].append(idx)
    
    # Convert to numpy arrays for faster access if needed, but lists of lists are fine for small n
    # Actually, let's precompute the "state" of each AP.
    # state[ap_idx] = 0 if not mono, 1 if mono.
    # We also need to know the "potential" change.
    
    # Helper to compute initial score and states
    def compute_initial_states(c):
        states = np.zeros(num_aps, dtype=np.int8)
        for idx in range(num_aps):
            a, b, c_, d = ap_array[idx]
            if c[a] == c[b] == c[c_] == c[d]:
                states[idx] = 1
        return states
    
    current_states = compute_initial_states(color)
    current_score = np.sum(current_states)
    
    # Helper to compute delta for flipping bit at index i
    # If we flip bit i, it affects all APs in ap_indices_at_pos[i].
    # For an AP (a,b,c,d):
    #   If it was Mono (1):
    #     - If i is one of the positions, and we flip it, the colors will no longer be all equal (since i changes, others stay same). 
    #       So it becomes Non-Mono (0). Delta contribution: -1.
    #   If it was Non-Mono (0):
    #     - It might become Mono.
    #     - Let the colors at the 4 positions be $c_a, c_b, c_c, c_d$.
    #     - Suppose we flip position $p$ (which is one of a,b,c,d).
    #     - The new color at $p$ is $1 - c_p$.
    #     - For the AP to become Mono, the new color at $p$ must equal the colors of the other 3 positions.
    #     - This requires:
    #       1. The other 3 positions already have the same color.
    #       2. The new color at $p$ ($1 - c_p$) equals that common color.
    #     - Note: If the other 3 are same color $C$, and $c_p$ is also $C$, then the AP was already Mono (case 1).
    #     - So for Non-Mono APs, if the other 3 are same color $C$, then $c_p$ must be $1-C$ (since it's not Mono).
    #     - Then $1 - c_p = 1 - (1-C) = C$. So it becomes Mono.
    #     - If the other 3 are NOT same color, flipping one bit cannot make all 4 same.
    
    # So, logic for flipping bit at index i:
    # delta = 0
    # for ap_idx in ap_indices_at_pos[i]:
    #   if states[ap_idx] == 1:
    #     delta -= 1
    #   else:
    #     # Check if other 3 are same
    #     a, b, c_, d = ap_array[ap_idx]
    #     # Identify the other 3 indices
    #     others = [x for x in [a,b,c_,d] if x != i]
    #     if color[others[0]] == color[others[1]] == color[others[2]]:
    #        # They are same. Does flipping i make it match?
    #        # Since AP is not mono, color[i] != color[others[0]].
    #        # New color[i] = 1 - color[i] = color[others[0]].
    #        # So yes, it becomes mono.
    #        delta += 1
    #   # Update states? No, we only compute delta for decision. 
    #   # If accepted, we must update states[ap_idx] and color[i].
    
    # To make this faster, we can precompute the "other 3" masks or just rely on the fact that n=256 is small.
    # The number of APs containing a specific index i is small (max ~256 * 3 / 4? No, much less).
    # Average APs per index: 4 * num_aps / n. num_aps ~ 10k. 4*10k/256 ~ 150.
    # So ~150 checks per flip. Very fast.
    
    best_color = color.copy()
    best_score = current_score
    
    # Iterated Local Search
    # Phase 1: Local Optimization (Hill Climbing)
    # Phase 2: Perturbation
    
    max_time_per_phase = 5.0 # seconds
    total_deadline = deadline
    
    while time.time() < total_deadline - 1.0:
        
        # --- Local Search Phase ---
        # We will iterate through bits in random order and flip if it improves.
        # Repeat until no improvement found in a full pass.
        improved = True
        while improved and time.time() < total_deadline - 1.0:
            improved = False
            # Randomize order to avoid bias
            indices = list(range(n))
            random.shuffle(indices)
            
            for i in indices:
                # Calculate delta
                delta = 0
                # We need to know the current state of the APs containing i
                # We maintain `current_states` array.
                for ap_idx in ap_indices_at_pos[i]:
                    if current_states[ap_idx] == 1:
                        delta -= 1
                    else:
                        # Check if other 3 are same
                        a, b, c_, d = ap_array[ap_idx]
                        # Optimize: just check the 3 others
                        # We can't easily vectorize this inner loop without overhead, 
                        # but 150 iterations is fine in Python if simple.
                        
                        # Get colors of others
                        # This is the bottleneck in pure python if not careful.
                        # Let's try to speed it up.
                        
                        # Alternative: Precompute for each AP and each position in it, 
                        # whether the other 3 are equal. 
                        # But that depends on current colors.
                        
                        # Let's just do it directly.
                        # Find the 3 indices != i
                        # Since i is in (a,b,c,d), we can check:
                        if a == i:
                            others = [b, c_, d]
                        elif b == i:
                            others = [a, c_, d]
                        elif c_ == i:
                            others = [a, b, d]
                        else:
                            others = [a, b, c_]
                        
                        if color[others[0]] == color[others[1]] == color[others[2]]:
                            delta += 1
                
                if delta < 0:
                    # Accept flip
                    color[i] = 1 - color[i]
                    current_score += delta
                    improved = True
                    
                    # Update states for all APs containing i
                    for ap_idx in ap_indices_at_pos[i]:
                        # Recalculate state for this AP
                        a, b, c_, d = ap_array[ap_idx]
                        if color[a] == color[b] == color[c_] == color[d]:
                            current_states[ap_idx] = 1
                        else:
                            current_states[ap_idx] = 0
                        
                    if current_score < best_score:
                        best_score = current_score
                        best_color = color.copy()
                        
                # If delta >= 0, do not flip (strict hill climbing)
                
        # --- Perturbation Phase ---
        # If we reached a local minimum, perturb to escape.
        # Flip k random bits (k=5 to 10)
        k = random.randint(5, 10)
        pert_indices = random.sample(range(n), k)
        for i in pert_indices:
            color[i] = 1 - color[i]
            # Update states
            for ap_idx in ap_indices_at_pos[i]:
                a, b, c_, d = ap_array[ap_idx]
                if color[a] == color[b] == color[c_] == color[d]:
                    current_states[ap_idx] = 1
                else:
                    current_states[ap_idx] = 0
            
        # Recalculate score to ensure consistency (or sum states)
        current_score = np.sum(current_states)
        
        # If perturbation improved things, great. If not, local search will try to fix it.
        # But we might be in a much worse state. 
        # To be safe, if the perturbation makes score much worse, we could revert? 
        # Standard ILS accepts perturbation regardless, hoping local search finds a better basin.
        # However, if we get stuck in a very bad basin, we waste time.
        # Let's add a check: if current_score > best_score + 50, maybe revert to best?
        # No, that breaks the "iterated" nature if we always revert. 
        # Let's stick to standard ILS: Accept perturbation, then Local Search.
        
        # Update best if perturbation itself was a miracle (unlikely)
        if current_score < best_score:
            best_score = current_score
            best_color = color.copy()

    return best_color.tolist()

# EVOLVE-BLOCK-END
