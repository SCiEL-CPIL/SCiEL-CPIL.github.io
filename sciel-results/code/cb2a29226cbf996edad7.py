"""
Combined program for Football Pool Problem K_3(7,1).
Uses bitmask-based greedy set cover for initialization and aggressive local search 
(removal, swap, and pair-removal moves) for optimization.
"""
import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet {0..q-1} with covering radius `radius`.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    # Handle trivial cases or specific known structures
    if q == 3 and n == 7 and radius == 1:
        N = 3 ** 7  # 2187
        
        # Precompute ball masks
        # We map each word (tuple of 7 ints) to an index 0..2186
        # ball_mask[i] is a bitmask where bit j is 1 if word j is within distance 1 of word i
        
        # Generate all words
        all_words = list(itertools.product(range(q), repeat=n))
        word_to_idx = {w: i for i, w in enumerate(all_words)}
        
        # Precompute neighbor masks using a more efficient approach
        neighbor_masks = [0] * N
        # Precompute position-value masks for faster neighbor lookup
        # For each word index, compute its mask by OR-ing 1<<idx for all neighbors
        for i, w in enumerate(all_words):
            mask = 1 << i
            w_list = list(w)
            for pos in range(n):
                orig_val = w_list[pos]
                for val in range(q):
                    if val == orig_val:
                        continue
                    w_list[pos] = val
                    idx = word_to_idx[tuple(w_list)]
                    mask |= (1 << idx)
                w_list[pos] = orig_val
            neighbor_masks[i] = mask
        
        # Precompute neighbor lists for faster iteration
        neighbor_lists = []
        for i in range(N):
            mask = neighbor_masks[i]
            nbs = []
            m = mask
            while m:
                lb = m & -m
                idx_nb = lb.bit_length() - 1
                nbs.append(idx_nb)
                m ^= lb
            neighbor_lists.append(nbs)
        
        full_mask = (1 << N) - 1
        
        # Helper: Convert list of indices to list of words
        def indices_to_words(indices):
            return [list(all_words[idx]) for idx in indices]

        # Greedy Set Cover using bitmasks
        # Returns a list of indices
        def greedy_set_cover(seed=None):
            if seed is not None:
                random.seed(seed)
            
            covered = 0
            code_indices = []
            
            # Use a list of candidate indices, shuffled for randomization
            candidates = list(range(N))
            random.shuffle(candidates)
            
            while covered != full_mask:
                best_gain = -1
                best_idx = -1
                
                # Iterate through shuffled candidates
                for idx in candidates:
                    gain = (neighbor_masks[idx] & ~covered).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_idx = idx
                        
                if best_gain == 0:
                    break
            
                code_indices.append(best_idx)
                covered |= neighbor_masks[best_idx]
                
            return code_indices

        # Local Search Optimizer
        # Takes a list of indices, returns optimized list of indices
        def optimize(initial_indices):
            current_indices = list(initial_indices)
            random.shuffle(current_indices)
            
            # For cover_count, we use an array of size N.
            cover_count = [0] * N
            in_code = [False] * N
            
            for idx in current_indices:
                in_code[idx] = True
                for nb in neighbor_lists[idx]:
                    cover_count[nb] += 1

            improved = True
            # Loop until no improvement or time limit
            while improved and time.time() < deadline - 5:
                improved = False
                random.shuffle(current_indices)
                
                # Removal Pass
                i = 0
                while i < len(current_indices):
                    if time.time() > deadline - 5:
                        break
                    
                    idx = current_indices[i]
                    nbs = neighbor_lists[idx]
                    
                    # Check if idx is redundant
                    # Redundant if all neighbors have cover_count > 1
                    redundant = True
                    for nb in nbs:
                        if cover_count[nb] <= 1:
                            redundant = False
                            break
                    
                    if redundant:
                        # Remove idx
                        for nb in nbs:
                            cover_count[nb] -= 1
                        
                        in_code[idx] = False
                        current_indices.pop(i)
                        improved = True
                        # Do not increment i, check next element
                    else:
                        i += 1

                # Pair Removal Pass: try removing two codewords if their union is covered by others
                if len(current_indices) > 2 and time.time() < deadline - 5:
                    pair_attempts = 20
                    for _ in range(pair_attempts):
                        if time.time() > deadline - 5:
                            break
                        if len(current_indices) < 2:
                            break
                        
                        # Pick two random distinct indices from current set
                        i1 = random.randint(0, len(current_indices) - 1)
                        i2 = random.randint(0, len(current_indices) - 1)
                        if i1 == i2:
                            continue
                        
                        idx_a = current_indices[i1]
                        idx_b = current_indices[i2]
                        
                        nbs_a = neighbor_lists[idx_a]
                        nbs_b = neighbor_lists[idx_b]
                        
                        # Check if all neighbors of A and B are covered by others
                        valid_pair_removal = True
                        for nb in nbs_a:
                            if cover_count[nb] <= 1:
                                valid_pair_removal = False
                                break
                        if valid_pair_removal:
                            for nb in nbs_b:
                                if cover_count[nb] <= 1:
                                    valid_pair_removal = False
                                    break
                        
                        if valid_pair_removal:
                            # Remove both
                            for nb in nbs_a:
                                cover_count[nb] -= 1
                            for nb in nbs_b:
                                cover_count[nb] -= 1
                            
                            in_code[idx_a] = False
                            in_code[idx_b] = False
                            
                            # Remove from list (careful with indices)
                            if i1 > i2:
                                i1, i2 = i2, i1
                            current_indices.pop(i1)
                            current_indices.pop(i2)
                            improved = True
                            break

                # Swap Pass
                # Try to swap A in S with B not in S if it allows further reduction
                if len(current_indices) > 185:
                    # Number of swap attempts
                    swap_attempts = 80
                    for _ in range(swap_attempts):
                        if time.time() > deadline - 5:
                            break
                        if not current_indices:
                            break
                        
                        # Pick random A from current set
                        idx_a = random.choice(current_indices)
                        
                        # Pick random B not in current set
                        idx_b = random.randint(0, N - 1)
                        while in_code[idx_b]:
                            idx_b = random.randint(0, N - 1)
                        
                        nbs_a = neighbor_lists[idx_a]
                        nbs_b = neighbor_lists[idx_b]
                        mask_b = neighbor_masks[idx_b]
                        
                        # Check validity of swap A->B
                        # Words covered only by A (cover_count == 1 and in nbs_a)
                        # must be covered by B.
                        valid_swap = True
                        for nb in nbs_a:
                            if cover_count[nb] == 1:
                                # Check if B covers nb
                                if not (mask_b & (1 << nb)):
                                    valid_swap = False
                                    break
                        
                        if valid_swap:
                            # Perform Swap
                            # Update cover counts
                            for nb in nbs_a:
                                cover_count[nb] -= 1
                            for nb in nbs_b:
                                cover_count[nb] += 1
                            
                            # Update structures
                            in_code[idx_a] = False
                            in_code[idx_b] = True
                            current_indices[current_indices.index(idx_a)] = idx_b
                            
                            improved = True
                            break 

            return current_indices

        # Main Loop: Multi-start
        best_size = float('inf')
        best_indices = []
        
        # Initial best: Tetracode (243 words) as a safe fallback
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        initial_tetracode = [tuple(h + list(tail)) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        best_indices = [word_to_idx[c] for c in initial_tetracode]
        best_size = len(best_indices)
        
        restart_count = 0
        while time.time() < deadline - 10:
            # 1. Generate initial solution using Greedy
            # Use a different seed for each restart to vary the greedy path
            greedy_indices = greedy_set_cover(seed=restart_count)
            
            # 2. Optimize
            optimized_indices = optimize(greedy_indices)
            
            # 3. Update best
            if len(optimized_indices) < best_size:
                best_size = len(optimized_indices)
                best_indices = optimized_indices
                
            restart_count += 1
            
        # Return the best found
        return indices_to_words(best_indices)

    # Generic fallback for other parameters
    # Simple greedy set cover
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    N = len(all_words)
    
    # Precompute neighbors
    neighbors = []
    for w in all_words:
        nbs = {word_to_idx[w]}
        w_list = list(w)
        for pos in range(n):
            orig_val = w_list[pos]
            for val in range(q):
                if val == orig_val:
                    continue
                w_list[pos] = val
                nbs.add(word_to_idx[tuple(w_list)])
            w_list[pos] = orig_val
        neighbors.append(nbs)
    
    uncovered = set(range(N))
    code_set = []
    
    while uncovered and time.time() < deadline:
        best_idx = -1
        best_count = 0
        for idx in range(N):
            if idx in code_set:
                continue
            cnt = sum(1 for nb in neighbors[idx] if nb in uncovered)
            if cnt > best_count:
                best_count = cnt
                best_idx = idx
        if best_idx == -1:
            break
        code_set.append(best_idx)
        for nb in neighbors[best_idx]:
            uncovered.discard(nb)
    
    return [list(all_words[idx]) for idx in code_set]
