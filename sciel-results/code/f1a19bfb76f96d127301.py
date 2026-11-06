import time
import random
import numpy as np

def construct():
    n = 256
    deadline = time.time() + 118.0
    
    # Precompute APs
    aps = []
    for d in range(1, n // 3 + 1):
        max_a = n - 3 * d
        for a in range(0, max_a):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    num_aps = len(aps)
    ap_array = np.array(aps, dtype=np.int32)
    
    # Build inverse mapping: for each index i, which APs does it participate in?
    # And what is its position (0, 1, 2, 3) in those APs?
    # This allows O(1) delta calculation for a single flip.
    # Actually, for a single flip at index i, we only need to check APs containing i.
    # Let's store: ap_indices_per_pos[i] = list of (ap_idx, pos_in_ap)
    
    ap_indices_per_pos = [[] for _ in range(n)]
    for ap_idx in range(num_aps):
        a0, a1, a2, a3 = ap_array[ap_idx]
        ap_indices_per_pos[a0].append((ap_idx, 0))
        ap_indices_per_pos[a1].append((ap_idx, 1))
        ap_indices_per_pos[a2].append((ap_idx, 2))
        ap_indices_per_pos[a3].append((ap_idx, 3))
        
    # Helper to check if an AP is monochromatic given the current color array
    # We will maintain a boolean array 'is_mono' of shape (num_aps,)
    
    def count_mono(color_vec):
        c0 = color_vec[ap_array[:, 0]]
        c1 = color_vec[ap_array[:, 1]]
        c2 = color_vec[ap_array[:, 2]]
        c3 = color_vec[ap_array[:, 3]]
        mask = (c0 == c1) & (c1 == c2) & (c2 == c3)
        return np.sum(mask)
        
    def compute_is_mono(color_vec):
        c0 = color_vec[ap_array[:, 0]]
        c1 = color_vec[ap_array[:, 1]]
        c2 = color_vec[ap_array[:, 2]]
        c3 = color_vec[ap_array[:, 3]]
        return (c0 == c1) & (c1 == c2) & (c2 == c3)
        
    def local_search(color_vec, is_mono, current_score, deadline):
        """
        Greedy local search with exact delta evaluation.
        Returns updated color_vec, is_mono, score.
        """
        improved = True
        while improved and time.time() < deadline - 0.5:
            improved = False
            # Shuffle order of indices to avoid bias
            indices = np.random.permutation(n)
            
            for idx in indices:
                # Calculate delta if we flip color_vec[idx]
                # Delta = (new_mono_count - old_mono_count)
                # For each AP containing idx:
                # If it was mono, it becomes non-mono (contribution -1)
                # If it was non-mono, it might become mono (contribution +1)
                
                delta = 0
                for ap_idx, pos in ap_indices_per_pos[idx]:
                    # Check current status
                    if is_mono[ap_idx]:
                        # It is currently mono. Flipping idx breaks it.
                        delta -= 1
                    else:
                        # It is currently non-mono. 
                        # Does flipping idx make it mono?
                        # This happens if the other 3 elements are equal to the NEW color of idx.
                        # New color of idx = 1 - color_vec[idx]
                        # Let other indices be the ones in ap_array[ap_idx] excluding pos.
                        other_indices = [j for j in range(4) if j != pos]
                        other_idx_0 = ap_array[ap_idx, other_indices[0]]
                        other_idx_1 = ap_array[ap_idx, other_indices[1]]
                        other_idx_2 = ap_array[ap_idx, other_indices[2]]
                        
                        c0 = color_vec[other_idx_0]
                        c1 = color_vec[other_idx_1]
                        c2 = color_vec[other_idx_2]
                        
                        # If c0 == c1 == c2, and c0 == 1 - color_vec[idx], then it becomes mono
                        if c0 == c1 and c1 == c2:
                            if c0 != color_vec[idx]:
                                delta += 1
                
                if delta < 0:
                    # Accept the flip
                    color_vec[idx] = 1 - color_vec[idx]
                    current_score += delta
                    
                    # Update is_mono for affected APs
                    for ap_idx, pos in ap_indices_per_pos[idx]:
                        # Recalculate mono status for this AP
                        # This is O(4) per AP, so O(1) overall per flip
                        a0, a1, a2, a3 = ap_array[ap_idx]
                        c0 = color_vec[a0]
                        c1 = color_vec[a1]
                        c2 = color_vec[a2]
                        c3 = color_vec[a3]
                        is_mono[ap_idx] = (c0 == c1) and (c1 == c2) and (c2 == c3)
                    
                    improved = True
                    # Optional: break early if score is 0
                    if current_score == 0:
                        return color_vec, is_mono, current_score
                        
        return color_vec, is_mono, current_score

    best_score = float('inf')
    best_color = None
    
    # Multi-start strategy
    # We will run multiple rounds. Each round starts with a different initialization.
    # Initialization types:
    # 1. Thue-Morse
    # 2. Random
    # 3. Random with low density
    # 4. Alternating blocks
    
    while time.time() < deadline - 1.0:
        # Choose initialization
        init_type = random.randint(0, 3)
        
        if init_type == 0:
            # Thue-Morse
            color = np.array([bin(i).count('1') % 2 for i in range(n)], dtype=np.int8)
        elif init_type == 1:
            # Random
            color = np.random.randint(0, 2, n, dtype=np.int8)
        elif init_type == 2:
            # Random with bias towards 0 (or just random, but let's try structured random)
            # Actually, random is fine. Let's try "blocky" random
            color = np.zeros(n, dtype=np.int8)
            # Randomly flip some blocks
            for _ in range(10):
                start = random.randint(0, n-1)
                length = random.randint(1, 10)
                color[start:start+length] = 1
        else:
            # Alternating blocks of 2 (seed strategy)
            color = np.array([(i // 2) % 2 for i in range(n)], dtype=np.int8)
            
        is_mono = compute_is_mono(color)
        current_score = np.sum(is_mono)
        
        # Local Search
        color, is_mono, current_score = local_search(color, is_mono, current_score, deadline)
        
        if current_score < best_score:
            best_score = current_score
            best_color = color.copy()
            
        # If we found 0, we can stop immediately
        if best_score == 0:
            break
            
        # Perturbation for next iteration if we don't break
        # We just loop back to a new random init. 
        # To be more efficient, we could perturb the best solution, but 
        # multi-start from scratch is often robust for this size.
        # Let's add a light perturbation to the current best to explore nearby basins?
        # No, the loop restarts with a random init. 
        # However, we might get stuck in the same local minima if inits are similar.
        # The random init ensures diversity.
        
        # To make it "Iterated", we can perturb the best solution found so far
        # and run local search again.
        if best_color is not None and time.time() < deadline - 2.0:
            # Perturb best_color
            perturbed = best_color.copy()
            # Flip 5-10 random bits
            num_flips = random.randint(5, 10)
            flip_indices = np.random.choice(n, num_flips, replace=False)
            perturbed[flip_indices] = 1 - perturbed[flip_indices]
            
            is_mono_p = compute_is_mono(perturbed)
            score_p = np.sum(is_mono_p)
            
            perturbed, is_mono_p, score_p = local_search(perturbed, is_mono_p, score_p, deadline)
            
            if score_p < best_score:
                best_score = score_p
                best_color = perturbed.copy()
                
                if best_score == 0:
                    break

    if best_color is None:
        # Fallback
        best_color = np.array([(i // 2) % 2 for i in range(n)], dtype=np.int8)
        
    return best_color.tolist()

# EVOLVE-BLOCK-END
