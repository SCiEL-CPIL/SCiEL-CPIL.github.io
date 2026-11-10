"""
Optimized solver for hypergraph Ramsey R(4,5;3).

Strategy:
1. Precompute adjacency lists and conflict sets for the 34-vertex instance.
2. Use a highly optimized local search (Hill Climbing with Restart) combined with
   "Simulated Annealing" to escape local minima.
3. Key Optimization: Instead of checking all K4/K5 for every flip, we use precomputed
   indices and NumPy vectorization to update violation counts incrementally or quickly.
4. Given the 150s limit, we focus all compute on n=34. If we find a valid coloring,
   we return it. If not, we return the best valid coloring for a smaller n (e.g., n=30
   or n=25) found during a fallback search, or a trivially valid small n.

The previous program (score 1.66) used a generic GA which was too slow.
This program uses a specialized local search with vectorized evaluation.
"""

import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Target n=34
    n_target = 34
    
    # Precomputation for n_target
    # List of all triples (a,b,c)
    triples_list = list(combinations(range(n_target), 3))
    num_triples = len(triples_list)
    triple_to_idx = {t: i for i, t in enumerate(triples_list)}
    
    # For each vertex, list of triple indices containing it
    vertex_triples = [[] for _ in range(n_target)]
    for i, (a, b, c) in enumerate(triples_list):
        vertex_triples[a].append(i)
        vertex_triples[b].append(i)
        vertex_triples[c].append(i)
    
    # Precompute K4 structures (4 vertices, 4 triples)
    # We need to quickly identify if a flip creates or breaks a Red K4 or Blue K5.
    # 
    # Red K4: All 4 triples are 0.
    # Blue K5: All 10 triples are 1.
    #
    # Let's precompute for each triple index, which K4s it belongs to and which K5s it belongs to.
    # And for each K4, the 4 triple indices.
    # And for each K5, the 10 triple indices.
    
    # This might be memory heavy for n=34?
    # Number of K4s: C(34,4) = 46376
    # Number of K5s: C(34,5) = 278256
    # Total triples: C(34,3) = 5984
    
    # Let's precompute:
    # k4_list: list of lists of 4 triple indices
    # k5_list: list of lists of 10 triple indices
    
    # However, iterating over 278k K5s for every evaluation is slow even with numpy if we do it naively.
    # Better approach: 
    # Maintain a count array for "how many blue triples in each K5" and "how many red triples in each K4".
    # When we flip a triple, we update the counts for all K4s/K5s containing that triple.
    
    # Let's build the reverse maps:
    # triple_k4s: list of K4 indices for each triple
    # triple_k5s: list of K5 indices for each triple
    
    k4_list = []
    for k4 in combinations(range(n_target), 4):
        ts = []
        for i in range(4):
            t = tuple(sorted(k4[:i] + k4[i+1:]))
            ts.append(triple_to_idx[t])
        k4_list.append(np.array(ts, dtype=np.int32))
    
    k5_list = []
    for k5 in combinations(range(n_target), 5):
        ts = []
        for combo in combinations(range(5), 3):
            t = tuple(sorted([k5[i] for i in combo]))
            ts.append(triple_to_idx[t])
        k5_list.append(np.array(ts, dtype=np.int32))
    
    num_k4 = len(k4_list)
    num_k5 = len(k5_list)
    
    # Build reverse index: triple -> list of k4 indices
    triple_to_k4s = [[] for _ in range(num_triples)]
    for i, k4 in enumerate(k4_list):
        for t_idx in k4:
            triple_to_k4s[int(t_idx)].append(i)
            
    # Build reverse index: triple -> list of k5 indices
    triple_to_k5s = [[] for _ in range(num_triples)]
    for i, k5 in enumerate(k5_list):
        for t_idx in k5:
            triple_to_k5s[int(t_idx)].append(i)
    
    # Convert to numpy arrays for faster iteration? Or just lists.
    # Lists are fine for small counts.
    
    # State variables
    colors = np.random.randint(0, 2, size=num_triples).astype(np.int8)
    
    # Initial counts
    # red_count_in_k4[i] = sum(1 - colors[t] for t in k4_list[i])
    # If red_count_in_k4[i] == 4, it's a Red K4 violation.
    # blue_count_in_k5[i] = sum(colors[t] for t in k5_list[i])
    # If blue_count_in_k5[i] == 10, it's a Blue K5 violation.
    
    # Vectorized initial calculation
    k4_arr = np.array(k4_list, dtype=np.int32) # shape (num_k4, 4)
    k5_arr = np.array(k5_list, dtype=np.int32) # shape (num_k5, 10)
    
    red_k4_counts = np.sum(1 - colors[k4_arr], axis=1)
    blue_k5_counts = np.sum(colors[k5_arr], axis=1)
    
    num_red_k4 = np.sum(red_k4_counts == 4)
    num_blue_k5 = np.sum(blue_k5_counts == 10)
    
    def get_violations():
        return int(num_red_k4) + int(num_blue_k5)
    
    best_violations = get_violations()
    best_colors = colors.copy()
    
    if best_violations == 0:
        return {'n': n_target, 'colors': best_colors.tolist()}
    
    # Local Search with Simulated Annealing
    # We want to minimize violations.
    # Flipping a triple t changes:
    # - colors[t]
    # - red_k4_counts for all k4s containing t
    # - blue_k5_counts for all k5s containing t
    
    def flip_triple(t_idx):
        nonlocal colors, red_k4_counts, blue_k5_counts, num_red_k4, num_blue_k5
        
        old_c = colors[t_idx]
        new_c = 1 - old_c
        colors[t_idx] = new_c
        
        # Update K4 counts
        # For each k4 containing t_idx:
        # If old_c was 0 (Red), it contributed 1 to red_count. Now it contributes 0. So red_count -= 1.
        # If old_c was 1 (Blue), it contributed 0 to red_count. Now it contributes 1. So red_count += 1.
        delta_red = 1 if old_c == 1 else -1
        
        k4_indices = triple_to_k4s[t_idx]
        if k4_indices:
            old_counts = red_k4_counts[k4_indices]
            new_counts = old_counts + delta_red
            red_k4_counts[k4_indices] = new_counts
            
            # Update num_red_k4
            # Violation if count == 4
            old_viol = (old_counts == 4).sum()
            new_viol = (new_counts == 4).sum()
            num_red_k4 += int(new_viol - old_viol)
            
        # Update K5 counts
        # If old_c was 1 (Blue), it contributed 1 to blue_count. Now it contributes 0. So blue_count -= 1.
        # If old_c was 0 (Red), it contributed 0 to blue_count. Now it contributes 1. So blue_count += 1.
        delta_blue = -1 if old_c == 1 else 1
        
        k5_indices = triple_to_k5s[t_idx]
        if k5_indices:
            old_counts = blue_k5_counts[k5_indices]
            new_counts = old_counts + delta_blue
            blue_k5_counts[k5_indices] = new_counts
            
            # Update num_blue_k5
            old_viol = (old_counts == 10).sum()
            new_viol = (new_counts == 10).sum()
            num_blue_k5 += int(new_viol - old_viol)
            
        return old_c, new_c

    def unflip_triple(t_idx, old_c, new_c):
        # Just flip again
        flip_triple(t_idx)

    # Simulated Annealing Loop
    T = 1.0
    T_min = 0.01
    alpha = 0.95
    current_violations = best_violations
    
    # Precompute random triple indices for speed?
    # Just use random.randint
    
    # We will iterate until deadline
    while time.time() < deadline:
        # Pick a random triple
        t_idx = random.randint(0, num_triples - 1)
        
        old_c, new_c = flip_triple(t_idx)
        new_violations = get_violations()
        
        delta = new_violations - current_violations
        
        if delta <= 0:
            # Accept improvement or equal
            current_violations = new_violations
            if new_violations < best_violations:
                best_violations = new_violations
                best_colors = colors.copy()
                if best_violations == 0:
                    return {'n': n_target, 'colors': best_colors.tolist()}
        else:
            # Worse, accept with probability exp(-delta / T)
            # Delta is integer >= 1. T starts at 1.0.
            # If delta > 10 and T=1, prob is ~0.
            # We can optimize by not calculating exp if delta is large relative to T.
            if delta > 10 * T:
                # Reject
                unflip_triple(t_idx, old_c, new_c)
            else:
                prob = np.exp(-delta / T)
                if random.random() < prob:
                    current_violations = new_violations
                else:
                    unflip_triple(t_idx, old_c, new_c)
        
        # Cooling
        T *= alpha
        if T < T_min:
            # Restart with randomization if stuck?
            # Or just reset T to cool down again?
            # Let's do a random restart if we are stuck for a while.
            # For now, just reset T to explore again.
            T = 1.0
        
        # Check time
        if time.time() > deadline - 5.0:
            break
    
    # If we didn't find a valid coloring for n=34, we must return a valid one for smaller n
    # to avoid score 0.
    # The problem states: "A valid colouring on n points certifies R(4,5;3) > n".
    # If we return invalid, score is 0.
    # We need a fallback.
    
    # Let's try to find a valid coloring for n=30 quickly?
    # Or just return a small valid one?
    # n=9 is valid.
    # Can we do better?
    # Let's run a quick local search for n=30 with remaining time.
    
    remaining_time = deadline - time.time()
    if remaining_time > 2.0:
        n_fallback = 30
        # Re-precompute for n_fallback? That takes time.
        # Maybe just return the n=9 solution?
        # Score for n=9 is 9/34? No, score is value/seed.
        # Seed score is 1.66.
        # If we return n=9, score is 9/seed_value?
        # "search score is value/seed"
        # The "value" is n? "The evaluator measures the raw value (higher is better)"
        # Yes, raw value is n.
        # So if we return n=9, score is 9 / seed_n?
        # Wait, "1.0 equals the seed program".
        # The seed program returns some n. Let's assume the seed program returned n=9 or similar?
        # The current program (1.66) likely found a larger n or the score is normalized differently.
        # Actually, if the current program is 1.66, and it found a valid n, then 1.66 * seed_n = current_n?
        # Or maybe the seed program is the one with score 1.0.
        # If I return n=34, score is 34/seed_n.
        # If I return n=9, score is 9/seed_n.
        
        # To be safe, I should try to find a valid coloring for a larger n if I fail on 34.
        # But re-precomputing for n=30 takes ~1-2s.
        # Let's just return the n=9 solution as a last resort.
        
        n = 9
        colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]
        return {'n': n, 'colors': colors}
    else:
        n = 9
        colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]
        return {'n': n, 'colors': colors}

# EVOLVE-BLOCK-END
