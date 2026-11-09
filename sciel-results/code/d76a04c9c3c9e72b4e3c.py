"""Combined program for Football Pool Problem K_3(7,1).
Uses bitmask-based greedy set cover for initialization and aggressive local search 
(removal and swap moves) for optimization.
"""
import time
import random
import itertools
import sys

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet {0..q-1} with covering radius `radius`.
    Uses bitmask-based greedy set cover and local search to minimize size.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Handle the specific task instance
    if q == 3 and n == 7 and radius == 1:
        N = 3 ** 7  # 2187
        
        # Generate all words and map to indices
        all_words = list(itertools.product(range(q), repeat=n))
        word_to_idx = {w: i for i, w in enumerate(all_words)}
        
        # Precompute neighbor lists as Python lists (faster than sets for iteration)
        neighbors = [[] for _ in range(N)]
        for i, w in enumerate(all_words):
            neighbors[i].append(i)
            for pos in range(n):
                for val in range(q):
                    if val == w[pos]:
                        continue
                    new_w = list(w)
                    new_w[pos] = val
                    idx = word_to_idx[tuple(new_w)]
                    neighbors[i].append(idx)
        
        # Precompute neighbor masks as Python integers (bitsets)
        neighbor_masks = [0] * N
        for i in range(N):
            mask = 0
            for nb in neighbors[i]:
                mask |= (1 << nb)
            neighbor_masks[i] = mask
        
        full_mask = (1 << N) - 1
        
        def indices_to_words(indices):
            return [list(all_words[idx]) for idx in indices]
        
        def greedy_set_cover(seed=None):
            """Randomized greedy set cover using bitmasks."""
            if seed is not None:
                random.seed(seed)
            
            covered = 0
            code_indices = []
            candidates = list(range(N))
            random.shuffle(candidates)
            
            while covered != full_mask:
                best_gain = -1
                best_idx = -1
                
                for idx in candidates:
                    gain = (neighbor_masks[idx] & ~covered).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_idx = idx
                        if gain == (full_mask ^ covered).bit_count():
                            break
                
                if best_gain == 0:
                    break
                
                code_indices.append(best_idx)
                covered |= neighbor_masks[best_idx]
            
            return code_indices
        
        def optimize(initial_indices, time_limit=None):
            """Aggressive local search: removal passes + swap moves."""
            if time_limit is None:
                time_limit = deadline - 5
            
            current_indices = list(initial_indices)
            random.shuffle(current_indices)
            
            cover_count = [0] * N
            in_code = [False] * N
            
            for idx in current_indices:
                in_code[idx] = True
                for nb in neighbors[idx]:
                    cover_count[nb] += 1
            
            improved = True
            while improved and time.time() < time_limit:
                improved = False
                random.shuffle(current_indices)
                
                # Removal Pass
                i = 0
                while i < len(current_indices):
                    if time.time() > time_limit:
                        break
                    
                    idx = current_indices[i]
                    redundant = True
                    for nb in neighbors[idx]:
                        if cover_count[nb] <= 1:
                            redundant = False
                            break
                    
                    if redundant:
                        for nb in neighbors[idx]:
                            cover_count[nb] -= 1
                        in_code[idx] = False
                        current_indices.pop(i)
                        improved = True
                    else:
                        i += 1
                
                # Swap Pass - try to swap A in S with B not in S
                if len(current_indices) > 180:
                    swap_attempts = 100
                    for _ in range(swap_attempts):
                        if time.time() > time_limit:
                            break
                        if not current_indices:
                            break
                        
                        idx_a = random.choice(current_indices)
                        
                        # Pick random B not in current set
                        idx_b = random.randint(0, N - 1)
                        while in_code[idx_b]:
                            idx_b = random.randint(0, N - 1)
                        
                        mask_a = neighbor_masks[idx_a]
                        mask_b = neighbor_masks[idx_b]
                        
                        # Check validity: words covered only by A must be covered by B
                        valid_swap = True
                        for nb in neighbors[idx_a]:
                            if cover_count[nb] == 1:
                                # nb is covered only by A, must be in B's neighbors
                                if not (mask_b >> nb) & 1:
                                    valid_swap = False
                                    break
                        
                        if valid_swap:
                            # Perform swap: remove A, add B
                            for nb in neighbors[idx_a]:
                                cover_count[nb] -= 1
                            in_code[idx_a] = False
                            
                            for nb in neighbors[idx_b]:
                                cover_count[nb] += 1
                            in_code[idx_b] = True
                            
                            # Replace A with B in current_indices
                            pos_a = current_indices.index(idx_a)
                            current_indices[pos_a] = idx_b
                            improved = True
                            
                            # After swap, try to remove more
                            i = 0
                            while i < len(current_indices):
                                if time.time() > time_limit:
                                    break
                                idx = current_indices[i]
                                redundant = True
                                for nb in neighbors[idx]:
                                    if cover_count[nb] <= 1:
                                        redundant = False
                                        break
                                if redundant:
                                    for nb in neighbors[idx]:
                                        cover_count[nb] -= 1
                                    in_code[idx] = False
                                    current_indices.pop(i)
                                    improved = True
                                else:
                                    i += 1
            
            return current_indices
        
        # Try multiple strategies with different seeds
        best_set = None
        best_size = float('inf')
        
        # Strategy 1: Start with tetracode-based construction (243 words)
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        initial_codewords = [tuple(h + list(tail)) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        initial_indices = [word_to_idx[c] for c in initial_codewords]
        
        optimized = optimize(initial_indices, deadline - 30)
        if len(optimized) < best_size:
            best_size = len(optimized)
            best_set = set(optimized)
        
        # Strategy 2: Multiple random greedy set covers
        for seed in range(5):
            if time.time() > deadline - 20:
                break
            
            greedy_result = greedy_set_cover(seed=seed)
            optimized = optimize(greedy_result, deadline - 10)
            if len(optimized) < best_size:
                best_size = len(optimized)
                best_set = set(optimized)
        
        # Strategy 3: Random restarts from small random sets
        for _ in range(3):
            if time.time() > deadline - 10:
                break
            
            # Start with a random subset of size ~200
            random_start_size = 200
            random_indices = random.sample(range(N), random_start_size)
            
            # First, ensure coverage by adding words greedily
            covered = 0
            for idx in random_indices:
                covered |= neighbor_masks[idx]
            
            current = list(random_indices)
            while covered != full_mask:
                best_gain = -1
                best_idx = -1
                for idx in range(N):
                    if idx in current:
                        continue
                    gain = (neighbor_masks[idx] & ~covered).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_idx = idx
                if best_gain == 0:
                    break
                current.append(best_idx)
                covered |= neighbor_masks[best_idx]
            
            optimized = optimize(current, deadline - 5)
            if len(optimized) < best_size:
                best_size = len(optimized)
                best_set = set(optimized)
        
        # Return the best set found
        result = indices_to_words(list(best_set))
        return result
    
    # Generic fallback for other parameters
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    neighbors = []
    for w in all_words:
        nbs = [word_to_idx[w]]
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                nw = list(w)
                nw[pos] = val
                nbs.append(word_to_idx[tuple(nw)])
        neighbors.append(nbs)
    
    uncovered = set(range(len(all_words)))
    code_set = []
    while uncovered and time.time() < deadline:
        best_idx = -1
        best_count = 0
        for idx in range(len(all_words)):
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
