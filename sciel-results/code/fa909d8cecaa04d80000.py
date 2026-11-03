import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    total_space = q ** n

    # Precompute neighbors for each word
    neighbors = np.zeros((total_space, 1 + n * (q - 1)), dtype=np.int32)
    
    # Weights for fast index calculation
    weights = [q ** (n - 1 - i) for i in range(n)]

    def get_neighbors(idx):
        w = []
        tmp = idx
        for _ in range(n):
            w.append(tmp % q)
            tmp //= q
        w.reverse()
        
        nbs = [idx]
        
        if radius >= 1:
            for pos in range(n):
                orig_val = w[pos]
                for val in range(q):
                    if val == orig_val:
                        continue
                    n_idx = idx + (val - orig_val) * weights[pos]
                    nbs.append(n_idx)
        
        return nbs

    for i in range(total_space):
        nbs = get_neighbors(i)
        neighbors[i] = nbs

    # Precompute neighbor mask
    neighbor_mask = np.zeros((total_space, total_space), dtype=bool)
    for i in range(total_space):
        neighbor_mask[i, neighbors[i]] = True

    # Precompute a fast coverage gain lookup using bitsets
    # For each word, precompute which words it covers
    # Use a 2D array where covered_by[i, j] = 1 if i covers j
    covered_by = neighbor_mask  # covered_by[i, j] = True if word i covers word j

    def greedy_cover(randomize=False, seed=None):
        if seed is not None:
            random.seed(seed)
        
        covered = np.zeros(total_space, dtype=bool)
        code = []
        
        while not np.all(covered):
            # Gain for each candidate = number of newly covered words
            gains = np.sum(~covered[neighbors], axis=1)
            
            if randomize:
                max_gain = np.max(gains)
                candidates = np.where(gains == max_gain)[0]
                if len(candidates) > 0:
                    best_idx = candidates[random.randint(0, len(candidates) - 1)]
                else:
                    break
            else:
                best_idx = np.argmax(gains)
                if gains[best_idx] == 0:
                    break
            
            code.append(best_idx)
            covered[neighbors[best_idx]] = True
            
        return code

    def reduce_code(code, aggressive=False):
        """Remove redundant codewords from code. Returns reduced list."""
        if not code:
            return code
        
        current_code = list(code)
        
        if len(current_code) == 0:
            return current_code
            
        flat_neighbors = neighbors[current_code].flatten()
        cover_count = np.bincount(flat_neighbors, minlength=total_space)
        
        improved = True
        max_passes = 10 if aggressive else 3
        pass_count = 0
        
        while improved and pass_count < max_passes:
            improved = False
            random.shuffle(current_code)
            n_cw = len(current_code)
            
            for i in range(n_cw):
                if time.time() > deadline:
                    return current_code
                
                idx = current_code[i]
                min_count = np.min(cover_count[neighbors[idx]])
                
                if min_count > 1:
                    cover_count[neighbors[idx]] -= 1
                    current_code.pop(i)
                    improved = True
                    break
            
            pass_count += 1
        
        return current_code

    def swap_local_search(code, max_iterations=500, allow_neutral=False):
        """
        Perform swap-based local search to improve the covering code.
        Try removing a codeword and replacing it with a better one, then reduce.
        If allow_neutral, also accept swaps that maintain the same size (neutral moves).
        """
        if not code:
            return code
        
        current_code = list(code)
        best_code = current_code[:]
        best_len = len(current_code)
        
        if best_len == 0:
            return best_code
            
        # Build cover_count for current_code
        flat_neighbors = neighbors[best_code].flatten()
        cover_count = np.bincount(flat_neighbors, minlength=total_space)
        
        for iteration in range(max_iterations):
            if time.time() > deadline:
                break
            
            # Pick a random codeword to swap out
            i = random.randint(0, len(current_code) - 1)
            idx_out = current_code[i]
            
            # Tentatively remove idx_out
            cover_count[neighbors[idx_out]] -= 1
            current_code.pop(i)
            
            # Find uncovered words
            uncovered_mask = (cover_count == 0)
            uncovered_count = np.sum(uncovered_mask)
            
            if uncovered_count == 0:
                # It was redundant - try to find a replacement that allows further reduction
                if allow_neutral:
                    # Find a random replacement to explore
                    gains = np.sum(~(cover_count == 1), axis=0) if False else np.zeros(total_space)
                    # Actually, let's just pick a random word not in current_code
                    candidates = [j for j in range(total_space) if j not in current_code]
                    if candidates:
                        best_j = random.choice(candidates)
                        current_code.append(best_j)
                        cover_count[neighbors[best_j]] += 1
                        
                        # Try to reduce
                        new_code = list(current_code)
                        flat_new = neighbors[new_code].flatten()
                        new_cover_count = np.bincount(flat_new, minlength=total_space)
                        
                        improved = True
                        while improved:
                            improved = False
                            random.shuffle(new_code)
                            for k in range(len(new_code)):
                                idx = new_code[k]
                                if np.min(new_cover_count[neighbors[idx]]) > 1:
                                    new_cover_count[neighbors[idx]] -= 1
                                    new_code.pop(k)
                                    improved = True
                                    break
                        
                        if len(new_code) < best_len:
                            best_len = len(new_code)
                            best_code = new_code
                            current_code = new_code
                            flat_best = neighbors[best_code].flatten()
                            cover_count = np.bincount(flat_best, minlength=total_space)
                        else:
                            # Revert
                            current_code.insert(i, idx_out)
                            cover_count[neighbors[idx_out]] += 1
                            if best_j in current_code:
                                current_code.remove(best_j)
                            cover_count[neighbors[best_j]] -= 1
                else:
                    # Restore
                    current_code.insert(i, idx_out)
                    cover_count[neighbors[idx_out]] += 1
                continue
            
            # Find best replacement: maximize coverage of uncovered words
            gains = np.sum(uncovered_mask[neighbors], axis=1)
            
            # Exclude words already in current_code
            for cw in current_code:
                gains[cw] = -1
            
            best_j = np.argmax(gains)
            
            if gains[best_j] <= 0:
                # No good replacement found, revert
                current_code.insert(i, idx_out)
                cover_count[neighbors[idx_out]] += 1
                continue
            
            # Add best_j
            current_code.append(best_j)
            cover_count[neighbors[best_j]] += 1
            
            # Now try to reduce the new set
            new_code = list(current_code)
            flat_new = neighbors[new_code].flatten()
            new_cover_count = np.bincount(flat_new, minlength=total_space)
            
            # Try to remove redundant words
            improved = True
            while improved:
                improved = False
                random.shuffle(new_code)
                for k in range(len(new_code)):
                    idx = new_code[k]
                    if np.min(new_cover_count[neighbors[idx]]) > 1:
                        new_cover_count[neighbors[idx]] -= 1
                        new_code.pop(k)
                        improved = True
                        break
            
            if len(new_code) < best_len:
                best_len = len(new_code)
                best_code = new_code
                current_code = new_code
                flat_best = neighbors[best_code].flatten()
                cover_count = np.bincount(flat_best, minlength=total_space)
            else:
                # Revert
                current_code.insert(i, idx_out)
                cover_count[neighbors[idx_out]] += 1
                if best_j in current_code:
                    current_code.remove(best_j)
                cover_count[neighbors[best_j]] -= 1
        
        return best_code

    def simulated_annealing(code, max_iters=2000, smooth_energy=False):
        """
        Simulated annealing with optional smooth energy surrogate.
        """
        current_code = list(code)
        if not current_code:
            return current_code
            
        current_cover_count = np.bincount(neighbors[current_code].flatten(), minlength=total_space)
        current_uncovered = np.sum(current_cover_count == 0)
        current_size = len(current_code)
        
        best_code = current_code[:]
        best_uncovered = current_uncovered
        best_size = current_size
        
        # Temperature schedule
        T_start = 10.0
        alpha = 0.999
        
        T = T_start
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
                
            move_type = random.random()
            
            if move_type < 0.33:
                # Add a random codeword
                candidate = random.randint(0, total_space - 1)
                if candidate not in current_code:
                    gain = np.sum(current_cover_count[neighbors[candidate]] == 0)
                    delta_E = 1 - 1000 * gain
                    
                    if delta_E < 0 or random.random() < np.exp(-delta_E / T):
                        current_code.append(candidate)
                        current_cover_count[neighbors[candidate]] += 1
                        current_uncovered -= gain
                        current_size += 1
                        
                        if current_uncovered < best_uncovered or (current_uncovered == best_uncovered and current_size < best_size):
                            best_uncovered = current_uncovered
                            best_size = current_size
                            best_code = current_code[:]
                            
            elif move_type < 0.66:
                # Remove a random codeword
                if current_size == 0:
                    T *= alpha
                    continue
                    
                idx_to_remove = random.randint(0, current_size - 1)
                candidate = current_code[idx_to_remove]
                
                loss = np.sum(current_cover_count[neighbors[candidate]] == 1)
                delta_E = 1000 * loss - 1
                
                if delta_E < 0 or random.random() < np.exp(-delta_E / T):
                    current_code.pop(idx_to_remove)
                    current_cover_count[neighbors[candidate]] -= 1
                    current_uncovered += loss
                    current_size -= 1
                    
                    if current_uncovered < best_uncovered or (current_uncovered == best_uncovered and current_size < best_size):
                        best_uncovered = current_uncovered
                        best_size = current_size
                        best_code = current_code[:]
                        
            else:
                # Swap: Remove one, Add one
                if current_size == 0:
                    T *= alpha
                    continue
                    
                idx_to_remove = random.randint(0, current_size - 1)
                candidate_out = current_code[idx_to_remove]
                candidate_in = random.randint(0, total_space - 1)
                
                if candidate_in == candidate_out:
                    T *= alpha
                    continue
                
                cover_temp = current_cover_count.copy()
                cover_temp[neighbors[candidate_out]] -= 1
                uncovered_after_removal = np.sum(cover_temp == 0)
                gain_in = np.sum(cover_temp[neighbors[candidate_in]] == 0)
                uncovered_new = uncovered_after_removal - gain_in
                
                delta_E = 1000 * (uncovered_new - current_uncovered)
                
                if delta_E < 0 or random.random() < np.exp(-delta_E / T):
                    current_code[idx_to_remove] = candidate_in
                    current_cover_count[neighbors[candidate_out]] -= 1
                    current_cover_count[neighbors[candidate_in]] += 1
                    current_uncovered = uncovered_new
                    
                    if current_uncovered < best_uncovered or (current_uncovered == best_uncovered and current_size < best_size):
                        best_uncovered = current_uncovered
                        best_size = current_size
                        best_code = current_code[:]
                        
            T *= alpha
            
        return best_code

    # Strategy:
    # 1. Run deterministic greedy + reduce.
    # 2. Run randomized greedy + reduce + swap local search multiple times.
    # 3. Use Simulated Annealing on the best solutions found so far.
    # 4. Keep the best.

    best_code = greedy_cover(randomize=False)
    best_code = reduce_code(best_code)
    best_len = len(best_code)
    
    # Try swap local search on the initial solution
    improved_code = swap_local_search(best_code, max_iterations=200)
    if len(improved_code) < best_len:
        best_len = len(improved_code)
        best_code = improved_code

    # Try Simulated Annealing on the initial solution
    sa_code = simulated_annealing(best_code, max_iters=500)
    sa_code = reduce_code(sa_code)
    if len(sa_code) < best_len:
        best_len = len(sa_code)
        best_code = sa_code

    # Randomized restarts with neutral swaps
    num_restarts = 0
    while time.time() < deadline - 5.0:
        seed = random.randint(0, 2**31 - 1)
        cand_code = greedy_cover(randomize=True, seed=seed)
        cand_code = reduce_code(cand_code)
        
        # Alternate between swap local search (with neutral moves) and simulated annealing
        if num_restarts % 2 == 0:
            cand_code = swap_local_search(cand_code, max_iterations=100, allow_neutral=True)
        else:
            cand_code = simulated_annealing(cand_code, max_iters=300)
            cand_code = reduce_code(cand_code)
            
        num_restarts += 1
        
        if len(cand_code) < best_len:
            best_len = len(cand_code)
            best_code = cand_code

    # Final polish: Run aggressive reduce and swap on the absolute best
    best_code = reduce_code(best_code, aggressive=True)
    best_code = swap_local_search(best_code, max_iterations=150, allow_neutral=True)
    best_code = reduce_code(best_code, aggressive=True)
    
    # One more SA pass on the final best if time permits
    if time.time() < deadline - 2.0:
        sa_final = simulated_annealing(best_code, max_iters=200)
        sa_final = reduce_code(sa_final, aggressive=True)
        if len(sa_final) < len(best_code):
            best_code = sa_final

    # Convert indices back to words
    def idx_to_word(idx):
        w = []
        for i in range(n):
            w.append(idx % q)
            idx //= q
        w.reverse()
        return w

    return [idx_to_word(i) for i in best_code]
