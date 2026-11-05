import numpy as np
import time
import random

def construct():
    N = 256
    deadline = time.time() + 145  # Safety margin for 150s limit
    
    # 1. Precompute AP structure
    # We need to know for each position i, which APs it belongs to and what the other indices are.
    # To make vectorization easy, let's create a sparse-like structure or just lists.
    # Actually, for the "Full Sweep Greedy", we need to calculate delta for all i at once.
    # Delta_i = sum_{ap in APs(i)} (1 if others are mono AND c_i == c_others else -1 if others mono AND c_i != c_others else 0)
    # Wait, if others are mono with color c, and c_i is c, flipping removes 1 mono AP (delta -1).
    # If others are mono with color c, and c_i is 1-c, flipping adds 1 mono AP (delta +1).
    # If others are NOT mono, flipping i doesn't change the mono status of that AP (delta 0).
    
    # Let's precompute for each position i, a list of "neighbor triples" (j, k, l).
    # There are ~10k APs total. Each AP has 4 positions. So total neighbor entries = 4 * 10k = 40k.
    # This is small enough to keep in Python lists or convert to numpy arrays for speed.
    
    aps = []
    for d in range(1, N // 3 + 1):
        max_a = N - 3 * d
        for a in range(max_a):
            idxs = (a, a+d, a+2*d, a+3*d)
            aps.append(idxs)
            
    # Build neighbor lists: neighbors[i] = list of tuples (j, k, l)
    neighbors = [[] for _ in range(N)]
    for ap in aps:
        for i in range(4):
            cur = ap[i]
            others = [ap[j] for j in range(4) if j != i]
            neighbors[cur].append(tuple(others))
    
    # To speed up the "Full Sweep Greedy", we can precompute these as numpy arrays?
    # Or just use the lists. With 40k entries total, a Python loop might be okay if we are careful,
    # but numpy is better. Let's stick to the vectorized idea for the sweep.
    
    # For vectorized sweep:
    # We need to count, for each i, how many triples in neighbors[i] are mono and what color.
    # This is hard to vectorize directly with ragged arrays.
    # Alternative: Iterate over APs. For each AP (a,b,c,d), if all 4 are same color, it contributes to cost.
    # To find best flip:
    # If we flip i, cost changes by:
    # -1 for each AP containing i where the other 3 are same color as i (breaking mono)
    # +1 for each AP containing i where the other 3 are same color as 1-i (creating mono)
    
    # Let's implement a fast delta calculator using numpy on the colors array.
    # colors is a numpy array of int8.
    
    def get_deltas(colors):
        """
        Returns an array of deltas for flipping each position.
        colors: np.array of shape (N,) with 0/1
        """
        deltas = np.zeros(N, dtype=np.int32)
        
        # Iterate over all APs. This is ~10k iterations.
        # In each iteration, we check if the AP is "almost mono" or "mono".
        # Actually, we can just check the 4 colors.
        # This loop is the bottleneck. 10k * (small ops) is fast in Python? 
        # 10k is small. 10k * 10 ops = 100k ops. Very fast.
        
        for ap in aps:
            a, b, c, d = ap
            ca, cb, cc, cd = colors[a], colors[b], colors[c], colors[d]
            
            # Count how many are 0 and 1
            # If 4-0 or 4-1: Mono.
            # If 3-1 or 1-3: Almost mono.
            
            # Let's just check explicitly.
            # Case 1: All same (Mono)
            if ca == cb == cc == cd:
                # Flipping any of them breaks the AP. Delta -1 for all 4.
                deltas[a] -= 1
                deltas[b] -= 1
                deltas[c] -= 1
                deltas[d] -= 1
            else:
                # Case 2: 3 same, 1 different.
                # If we flip the "different" one, it becomes Mono. Delta +1 for that one.
                # If we flip a "same" one, it becomes 2-2 (not mono). Delta 0.
                # Wait, if we flip a "same" one in a 3-1 config:
                # Original: 0,0,0,1. Flip a 0 -> 1,0,0,1 (2-2). Not mono. Delta 0.
                # So only the "odd one out" has delta +1.
                
                # Find the odd one out
                # Sum of colors?
                # If sum is 3 (three 1s, one 0): The 0 is the odd one.
                # If sum is 1 (one 1, three 0s): The 1 is the odd one.
                
                s = ca + cb + cc + cd
                if s == 3:
                    # Three 1s, one 0. The 0 becomes 1 -> Mono 1s. Delta +1.
                    if ca == 0: deltas[a] += 1
                    if cb == 0: deltas[b] += 1
                    if cc == 0: deltas[c] += 1
                    if cd == 0: deltas[d] += 1
                elif s == 1:
                    # One 1, three 0s. The 1 becomes 0 -> Mono 0s. Delta +1.
                    if ca == 1: deltas[a] += 1
                    if cb == 1: deltas[b] += 1
                    if cc == 1: deltas[c] += 1
                    if cd == 1: deltas[d] += 1
                # Else (2-2 or other), delta 0 for all.
                
        return deltas

    def count_mono(colors):
        cnt = 0
        for ap in aps:
            a, b, c, d = ap
            if colors[a] == colors[b] == colors[c] == colors[d]:
                cnt += 1
        return cnt

    def greedy_polish(colors):
        """
        Repeatedly find the flip with the most negative delta and apply it.
        Continue until no improvement.
        """
        current_cost = count_mono(colors)
        improved = True
        while improved:
            improved = False
            deltas = get_deltas(colors)
            min_delta = np.min(deltas)
            if min_delta < 0:
                # Pick the first index with min_delta
                idx = np.argmin(deltas)
                colors[idx] = 1 - colors[idx]
                current_cost += min_delta
                improved = True
        return colors, current_cost

    def simulated_annealing_multi_flip(colors, T_start, cooling, min_T, time_limit):
        """
        SA with multi-flip moves (flip 4-8 random bits).
        """
        current_cost = count_mono(colors)
        best_cost = current_cost
        best_colors = colors.copy()
        
        T = T_start
        end_time = time.time() + time_limit
        
        while time.time() < end_time:
            # Choose number of flips
            k = random.randint(4, 8)
            # Choose random indices
            indices = random.sample(range(N), k)
            
            # Calculate delta for this multi-flip
            # We need to evaluate the cost change.
            # Since k is small, we can just apply the flip, count, and revert if bad?
            # No, counting takes ~10k ops. 
            # Doing 10k ops per step is too slow for SA (we want thousands of steps).
            
            # Alternative: Use the delta approximation?
            # Delta for multi-flip is not simply sum of single deltas due to interactions.
            # However, if the flipped indices are far apart (not in same AP), deltas add up.
            # With N=256 and k=8, probability of interaction is low but not zero.
            
            # Let's try a different approach: 
            # Use single flips for SA, but use a much faster delta calculation?
            # The get_deltas function above is ~10k iterations. 
            # If we do 10k steps/sec, that's 100M ops/sec. Python might do 10M-50M simple ops/sec.
            # So maybe 1000-5000 steps/sec with get_deltas.
            # In 100s, that's 100k-500k steps.
            # This is comparable to the original program's speed, but the original used precomputed neighbor lists which might be faster?
            
            # Let's check the original program's speed.
            # Original: `delta_for_flip` iterates `aps_at[i]`. 
            # Average APs per node = 4 * 10k / 256 = ~156.
            # So original delta calc is ~156 ops.
            # My `get_deltas` is ~10k ops (iterates all APs).
            # So my get_deltas is ~60x SLOWER per single flip evaluation than the original.
            # BUT, I can evaluate ALL 256 flips in one pass of `get_deltas`.
            # Original: 256 flips * 156 ops = ~40k ops for a full sweep.
            # Mine: 10k ops for a full sweep (all 256 deltas).
            # So `get_deltas` is ~4x FASTER for a full sweep!
            
            # This means I can do full sweeps much faster.
            # Strategy:
            # 1. Do a full sweep greedy (find best flip, apply).
            # 2. This converges to a local optimum.
            # 3. To escape, do a "kick": flip K random bits.
            # 4. Repeat.
            
            # This is essentially "Iterated Local Search" (ILS).
            # ILS is often better than SA for these problems.
            
            # Let's switch to ILS.
            break # Exit SA loop, use ILS below
            
        return best_colors, best_cost

    # Revised Strategy: Iterated Local Search (ILS)
    # 1. Start with random or pattern.
    # 2. Polish to local optimum (using fast full-sweep greedy).
    # 3. Perturb (flip K random bits).
    # 4. Polish again.
    # 5. If better, keep; else, revert or keep with some probability.
    # 6. Repeat until time limit.

    def iterated_local_search(initial_colors, time_limit):
        current_colors = initial_colors.copy()
        current_cost = count_mono(current_colors)
        best_colors = current_colors.copy()
        best_cost = current_cost
        
        end_time = time.time() + time_limit
        
        while time.time() < end_time:
            # 1. Polish
            polished_colors, polished_cost = greedy_polish(current_colors)
            
            if polished_cost < best_cost:
                best_cost = polished_cost
                best_colors = polished_colors.copy()
                current_colors = polished_colors
            else:
                current_colors = polished_colors # Keep the polished state, even if not better than global best, to continue from there? 
                # Usually in ILS, you keep the current state.
            
            # 2. Perturb
            # Flip K random bits. K can vary.
            K = random.randint(4, 12)
            indices = random.sample(range(N), K)
            for idx in indices:
                current_colors[idx] = 1 - current_colors[idx]
                
            # Optional: Check if we are stuck at a very high cost?
            # If current_cost > best_cost + threshold, maybe restart from best?
            # Let's just keep going.
            
        return best_colors, best_cost

    # Main Loop
    global_best_colors = None
    global_best_cost = float('inf')
    
    # Initial candidates
    # 1. Random
    # 2. Alternating
    # 3. Block
    # 4. All 0 / All 1
    
    candidates = [
        np.random.randint(0, 2, N, dtype=np.int8),
        np.array([i % 2 for i in range(N)], dtype=np.int8),
        np.array([(i // 8) % 2 for i in range(N)], dtype=np.int8),
        np.zeros(N, dtype=np.int8),
        np.ones(N, dtype=np.int8),
    ]
    
    # Add some more structured randoms
    for _ in range(5):
        candidates.append(np.random.randint(0, 2, N, dtype=np.int8))
        
    # Time allocation
    # We have ~140s.
    # Let's try each candidate for a fixed amount of time, say 20s each?
    # Or just run one long ILS from a random start, and if it gets stuck, restart?
    
    # Let's do multiple short ILS runs from different starts.
    time_per_run = 15.0
    num_runs = int(140 / time_per_run) # ~9 runs
    
    for i in range(num_runs):
        if time.time() > deadline - 5: # Leave 5s for final polish
            break
            
        # Pick a start
        if i < len(candidates):
            start = candidates[i].copy()
        else:
            # Perturb global best
            if global_best_colors is not None:
                start = global_best_colors.copy()
                K = random.randint(8, 16)
                indices = random.sample(range(N), K)
                for idx in indices:
                    start[idx] = 1 - start[idx]
            else:
                start = np.random.randint(0, 2, N, dtype=np.int8)
        
        best_colors, best_cost = iterated_local_search(start, time_per_run)
        
        if best_cost < global_best_cost:
            global_best_cost = best_cost
            global_best_colors = best_colors.copy()
            
    # Final Polish on the best result found
    if global_best_colors is not None:
        final_colors, final_cost = greedy_polish(global_best_colors)
        if final_cost < global_best_cost:
            global_best_cost = final_cost
            global_best_colors = final_colors
        
        # Convert to list of ints for return
        return list(global_best_colors)
    
    # Fallback
    return [0] * N
