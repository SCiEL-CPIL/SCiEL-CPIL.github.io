import time
import random
import numpy as np

def construct(q, n, radius):
    total_words = q ** n
    if q == 3 and n == 7 and radius == 1:
        # Precompute neighbor lists
        cover_count = 1 + n * (q - 1)  # 15
        
        def idx_to_base_q(idx):
            digits = [0] * n
            for i in range(n - 1, -1, -1):
                digits[i] = idx % q
                idx //= q
            return digits
        
        def base_q_to_idx(digits):
            idx = 0
            for d in digits:
                idx = idx * q + d
            return idx
        
        # Build neighbor lookup
        neighbors_arr = np.zeros((total_words, cover_count), dtype=np.int32)
        for i in range(total_words):
            digits = idx_to_base_q(i)
            idxs = [i]
            for pos in range(n):
                original_val = digits[pos]
                for new_val in range(q):
                    if new_val == original_val:
                        continue
                    new_digits = digits.copy()
                    new_digits[pos] = new_val
                    j = base_q_to_idx(new_digits)
                    idxs.append(j)
            for j, idx in enumerate(idxs):
                neighbors_arr[i, j] = idx
        
        # Build coverage matrix: cov[i, j] = 1 if word j is within radius of word i
        cov = np.zeros((total_words, total_words), dtype=np.bool_)
        for i in range(total_words):
            cov[i, neighbors_arr[i, :]] = True
        
        start_time = time.time()
        deadline = start_time + 115
        
        def greedy_construct(random_seed):
            if random_seed is not None:
                random.seed(random_seed)
            
            uncovered = np.ones(total_words, dtype=np.bool_)
            code = []
            
            while uncovered.any():
                gains = np.sum(uncovered[cov], axis=1)
                max_gain = np.max(gains)
                if max_gain == 0:
                    break
                best_candidates = np.where(gains == max_gain)[0]
                chosen = random.choice(best_candidates.tolist())
                code.append(chosen)
                uncovered[cov[chosen, :]] = False
            
            return code
        
        def reduce_code(code):
            if not code:
                return code
            counts = np.zeros(total_words, dtype=np.int16)
            for c in code:
                counts[neighbors_arr[c, :]] += 1
            
            changed = True
            while changed:
                changed = False
                new_code = []
                for c in code:
                    counts[neighbors_arr[c, :]] -= 1
                    if np.any(counts == 0):
                        counts[neighbors_arr[c, :]] += 1
                        new_code.append(c)
                    else:
                        changed = True
                code = new_code
            return code
        
        def simulated_annealing(initial_code, deadline, random_seed=0):
            random.seed(random_seed)
            code = list(initial_code)
            if not code:
                return code
            
            # Precompute coverage counts
            cov_counts = np.zeros(total_words, dtype=np.int16)
            for c in code:
                cov_counts[cov[c, :]] += 1
            
            current_energy = total_words - np.sum(cov_counts > 0)
            best_code = list(code)
            best_energy = current_energy
            
            # SA parameters
            T = 50.0
            T_min = 0.1
            cooling = 0.995
            steps_per_temp = 50
            
            step = 0
            while time.time() < deadline:
                if len(code) < 2:
                    break
                
                # Choose move type: remove, add, or swap
                r = random.random()
                
                if r < 0.3 and len(code) > 1:
                    # Remove a codeword
                    idx = random.randrange(len(code))
                    c_remove = code[idx]
                    code.pop(idx)
                    cov_counts[cov[c_remove, :]] -= 1
                    new_energy = total_words - np.sum(cov_counts > 0)
                    delta = new_energy - current_energy
                    if delta <= 0 or random.random() < np.exp(-delta / T):
                        current_energy = new_energy
                        if current_energy < best_energy:
                            best_energy = current_energy
                            best_code = list(code)
                    else:
                        code.insert(idx, c_remove)
                        cov_counts[cov[c_remove, :]] += 1
                        current_energy += 0  # no change
                    
                elif r < 0.5:
                    # Add a codeword
                    # Pick a random uncovered word, then a random neighbor
                    uncovered = np.where(cov_counts == 0)[0]
                    if len(uncovered) > 0:
                        target = random.choice(uncovered.tolist())
                        candidates = np.where(cov[:, target])[0]
                        c_add = random.choice(candidates.tolist())
                        if c_add not in code:
                            code.append(c_add)
                            cov_counts[cov[c_add, :]] += 1
                            new_energy = total_words - np.sum(cov_counts > 0)
                            delta = new_energy - current_energy
                            if delta <= 0 or random.random() < np.exp(-delta / T):
                                current_energy = new_energy
                                if current_energy < best_energy:
                                    best_energy = current_energy
                                    best_code = list(code)
                            else:
                                code.remove(c_add)
                                cov_counts[cov[c_add, :]] -= 1
                    else:
                        # All covered, try swapping
                        pass
                else:
                    # Swap: remove one, add another
                    idx = random.randrange(len(code))
                    c_remove = code[idx]
                    
                    # Pick a random candidate to add
                    c_add = random.randrange(total_words)
                    if c_add == c_remove:
                        continue
                    
                    # Apply swap
                    code[idx] = c_add
                    cov_counts[cov[c_remove, :]] -= 1
                    cov_counts[cov[c_add, :]] += 1
                    new_energy = total_words - np.sum(cov_counts > 0)
                    delta = new_energy - current_energy
                    if delta <= 0 or random.random() < np.exp(-delta / T):
                        current_energy = new_energy
                        if current_energy < best_energy:
                            best_energy = current_energy
                            best_code = list(code)
                    else:
                        code[idx] = c_remove
                        cov_counts[cov[c_add, :]] -= 1
                        cov_counts[cov[c_remove, :]] += 1
                        current_energy += 0  # no change
                
                step += 1
                if step % steps_per_temp == 0:
                    T = max(T_min, T * cooling)
                
                # Periodically try to reduce
                if step % 200 == 0 and time.time() < deadline - 5:
                    reduced = reduce_code(best_code)
                    if len(reduced) < len(best_code):
                        best_code = reduced
                        best_energy = total_words - np.sum(
                            np.any(np.array([cov[c, :] for c in reduced]), axis=0)
                        )
                        # Update current state to best
                        code = list(best_code)
                        cov_counts = np.zeros(total_words, dtype=np.int16)
                        for c in code:
                            cov_counts[cov[c, :]] += 1
                        current_energy = best_energy
        
        return best_code, cov, neighbors_arr, idx_to_base_q, total_words, deadline
        
        # Main search loop
        best_code = None
        best_size = float('inf')
        
        # First, get a good initial solution via greedy
        for seed in range(5):
            if time.time() >= deadline - 60:
                break
            code = greedy_construct(seed)
            code = reduce_code(code)
            if len(code) < best_size:
                best_size = len(code)
                best_code = code
        
        # Run SA from multiple starting points
        sa_iter = 0
        while time.time() < deadline - 10 and best_code is not None:
            # Use current best as starting point with some perturbation
            if sa_iter > 0 and random.random() < 0.5:
                # Perturb the best solution
                perturbed = list(best_code)
                k = random.randint(1, min(3, len(perturbed)))
                for _ in range(k):
                    if perturbed:
                        idx = random.randrange(len(perturbed))
                        perturbed.pop(idx)
                # Repair
                cov_counts = np.zeros(total_words, dtype=np.int16)
                for c in perturbed:
                    cov_counts[neighbors_arr[c, :]] += 1
                while np.any(cov_counts == 0):
                    uncovered = np.where(cov_counts == 0)[0]
                    gains = np.sum((cov_counts == 0)[cov], axis=1)
                    max_gain = np.max(gains)
                    if max_gain == 0:
                        break
                    best_candidates = np.where(gains == max_gain)[0]
                    chosen = random.choice(best_candidates.tolist())
                    perturbed.append(chosen)
                    cov_counts[neighbors_arr[chosen, :]] += 1
                start_code = perturbed
            else:
                start_code = best_code
            
            sa_result = simulated_annealing(start_code, deadline - 5, random_seed=sa_iter * 137 + 1)
            sa_code = sa_result
            
            if len(sa_code) < best_size:
                best_size = len(sa_code)
                best_code = sa_code
            
            sa_iter += 1
        
        # Final reduction
        if best_code is not None and time.time() < deadline - 3:
            best_code = reduce_code(best_code)
        
        result = []
        for idx in best_code:
            digits = idx_to_base_q(idx)
            result.append(digits)
        
        return result
    
    # Fallback for other parameters
    total_words = q ** n
    cover_count = 1 + n * (q - 1)
    
    def idx_to_tuple(idx):
        t = []
        for _ in range(n):
            t.append(idx % q)
            idx //= q
        return tuple(reversed(t))
    
    def tuple_to_idx(t):
        idx = 0
        for i in range(n):
            idx = idx * q + t[i]
        return idx
    
    neighbors_arr = np.zeros((total_words, cover_count), dtype=np.int32)
    for i in range(total_words):
        w = idx_to_tuple(i)
        idxs = [i]
        for pos in range(n):
            original_val = w[pos]
            for new_val in range(q):
                if new_val == original_val:
                    continue
                new_w = list(w)
                new_w[pos] = new_val
                j = tuple_to_idx(tuple(new_w))
                idxs.append(j)
        for j, idx in enumerate(idxs):
            neighbors_arr[i, j] = idx
    
    start_time = time.time()
    deadline = start_time + 115
    
    best_code = None
    best_size = float('inf')
    
    def greedy_construct(random_seed):
        if random_seed is not None:
            random.seed(random_seed)
        
        uncovered = np.ones(total_words, dtype=np.bool_)
        code = []
        
        while uncovered.any():
            gains = np.sum(uncovered[neighbors_arr], axis=1)
            valid = gains > 0
            if not valid.any():
                break
            max_gain = np.max(gains[valid])
            best_candidates = np.where((gains == max_gain) & valid)[0]
            chosen = random.choice(best_candidates.tolist())
            code.append(chosen)
            uncovered[neighbors_arr[chosen, :]] = False
        
        return code
    
    def reduce_code(code):
        if not code:
            return code
        counts = np.zeros(total_words, dtype=np.int16)
        for c in code:
            counts[neighbors_arr[c, :]] += 1
        
        changed = True
        while changed:
            changed = False
            new_code = []
            for c in code:
                counts[neighbors_arr[c, :]] -= 1
                if np.any(counts == 0):
                    counts[neighbors_arr[c, :]] += 1
                    new_code.append(c)
                else:
                    changed = True
            code = new_code
        return code
    
    def local_search(code, deadline):
        code = list(code)
        if not code:
            return code
        
        counts = np.zeros(total_words, dtype=np.int16)
        for c in code:
            counts[neighbors_arr[c, :]] += 1
        
        improved = True
        no_improve_count = 0
        max_no_improve = 3
        while improved and time.time() < deadline:
            improved = False
            indices = list(range(len(code)))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() >= deadline:
                    break
                
                c_remove = code[i]
                neighbors_c = neighbors_arr[c_remove, :]
                counts[neighbors_c] -= 1
                
                if np.any(counts == 0):
                    holes = np.where(counts == 0)[0]
                    candidates = set(neighbors_arr[holes[0]].tolist())
                    for h in holes[1:]:
                        if not candidates:
                            break
                        candidates &= set(neighbors_arr[h].tolist())
                    
                    candidates.discard(c_remove)
                    
                    if not candidates:
                        counts[neighbors_c] += 1
                        continue
                    
                    cand_list = list(candidates)
                    if len(cand_list) > 50:
                        cand_list = random.sample(cand_list, 50)
                    
                    cand_indices = np.array(cand_list, dtype=np.int32)
                    cand_neighbors = neighbors_arr[cand_indices, :]
                    scores = np.sum(counts[cand_neighbors], axis=1)
                    
                    best_idx = np.argmax(scores)
                    best_cand = cand_list[best_idx]
                    
                    code[i] = best_cand
                    counts[neighbors_arr[best_cand, :]] += 1
                    improved = True
                    no_improve_count = 0
                    break
                else:
                    code.pop(i)
                    improved = True
                    no_improve_count = 0
                    break
            
            if not improved:
                no_improve_count += 1
                if no_improve_count >= max_no_improve:
                    break
        return code
    
    seed = 0
    while time.time() < deadline:
        code = greedy_construct(seed)
        code = reduce_code(code)
        
        if time.time() < deadline - 5:
            code = local_search(code, deadline - 2)
            code = reduce_code(code)
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code
        
        seed += 1
    
    result = []
    for idx in best_code:
        w = idx_to_tuple(idx)
        result.append(list(w))
    
    return result
