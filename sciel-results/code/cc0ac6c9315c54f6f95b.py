import numpy as np
import time
import random

def construct(q, n, radius):
    if q == 3 and n == 8 and radius == 1:
        deadline = time.time() + 115.0
        N = q ** n  # 6561
        powers = [q ** i for i in range(n)]
        
        # Precompute neighbors
        neigh_size = 1 + n * (q - 1)
        neighbors = np.zeros((N, neigh_size), dtype=np.int32)
        
        for w in range(N):
            neighbors[w, 0] = w
            idx = 1
            for i in range(n):
                d = (w // powers[i]) % q
                for delta in range(1, q):
                    new_d = (d + delta) % q
                    diff = new_d - d
                    new_w = w + diff * powers[i]
                    neighbors[w, idx] = new_w
                    idx += 1
        
        # Precompute neighbor matrix for fast coverage and candidate finding
        # neighbor_matrix[i, j] is 1 if j is a neighbor of i (including i itself)
        neighbor_matrix = np.zeros((N, N), dtype=np.uint8)
        for i in range(N):
            neighbor_matrix[i, neighbors[i, :]] = 1
            
        def greedy_init(rng, deadline):
            uncovered = np.ones(N, dtype=bool)
            code = set()
            
            while np.any(uncovered) and time.time() < deadline - 5:
                # Calculate gain for each word: number of uncovered neighbors
                # gains[i] = sum(neighbor_matrix[i, :][uncovered])
                # Efficient way:
                # Create a vector of 1s where uncovered, 0 otherwise
                # gain = neighbor_matrix @ uncovered_vec
                uncovered_vec = uncovered.astype(np.uint8)
                gains = neighbor_matrix @ uncovered_vec
                
                max_gain = np.max(gains)
                if max_gain == 0:
                    break
                
                candidates = np.where(gains == max_gain)[0]
                w = rng.choice(candidates)
                code.add(w)
                uncovered[neighbors[w, :]] = False
                
            return code

        def local_search(code_set, rng, deadline):
            code_set = set(code_set)
            no_improve_count = 0
            max_no_improve = 50
            
            while time.time() < deadline - 2:
                if no_improve_count >= max_no_improve:
                    break
                
                # Compute coverage
                if not code_set:
                    break
                code_list = list(code_set)
                cov = neighbor_matrix[code_list, :].sum(axis=0)
                
                improved = False
                
                # 1. Deletion Pass - vectorized
                # A codeword c is redundant if all its neighbors have cov > 1
                # cov_neighbors[i] = cov[neighbors[i, :]]
                cov_neighbors = cov[neighbors]  # shape (N, neigh_size)
                redundant_mask = np.all(cov_neighbors > 1, axis=1)  # shape (N,)
                redundant = [c for c in code_list if redundant_mask[c]]
                
                if redundant:
                    for c in redundant:
                        code_set.remove(c)
                    no_improve_count = 0
                    continue # Recompute coverage
                
                # 2. Swap Pass (1-for-1) - optimized
                c_list = list(code_set)
                rng.shuffle(c_list)
                
                for c in c_list:
                    if time.time() > deadline - 2:
                        break
                    
                    c_neighbors = neighbors[c, :]
                    essential_words = c_neighbors[cov[c_neighbors] == 1]
                    
                    if len(essential_words) == 0:
                        continue
                    
                    if len(essential_words) > 3:
                        continue
                    
                    # Find candidates d that cover all essential_words
                    cand_mask = neighbor_matrix[essential_words[0], :].copy()
                    for ew in essential_words[1:]:
                        cand_mask &= neighbor_matrix[ew, :]
                        if not np.any(cand_mask):
                            break
                    
                    if not np.any(cand_mask):
                        continue
                    
                    candidates = np.where(cand_mask)[0]
                    valid_candidates = [d for d in candidates if d not in code_set]
                    if not valid_candidates:
                        continue
                    
                    d = rng.choice(valid_candidates)
                    code_set.remove(c)
                    code_set.add(d)
                    improved = True
                    no_improve_count = 0
                    break
                
                if improved:
                    continue
                
                # 3. Double Swap Pass (2-for-1)
                if time.time() < deadline - 5:
                    double_improved = False
                    c_list = list(code_set)
                    rng.shuffle(c_list)
                    num_pairs_to_try = min(30, len(c_list) // 2)
                    
                    for _ in range(num_pairs_to_try):
                        if time.time() > deadline - 5:
                            break
                        if len(c_list) < 2:
                            break
                        c1, c2 = rng.sample(c_list, 2)
                        
                        n1 = neighbors[c1, :]
                        n2 = neighbors[c2, :]
                        
                        e1 = n1[cov[n1] == 1]
                        e2 = n2[cov[n2] == 1]
                        
                        if len(e1) == 0 or len(e2) == 0:
                            continue
                        
                        all_ess = np.unique(np.concatenate([e1, e2]))
                        
                        if len(all_ess) > 4:
                            continue
                            
                        cand_mask = neighbor_matrix[all_ess[0], :].copy()
                        for ew in all_ess[1:]:
                            cand_mask &= neighbor_matrix[ew, :]
                            if not np.any(cand_mask):
                                break
                        
                        if not np.any(cand_mask):
                            continue
                        
                        candidates = np.where(cand_mask)[0]
                        valid_candidates = [d for d in candidates if d not in code_set]
                        
                        if valid_candidates:
                            d = rng.choice(valid_candidates)
                            code_set.remove(c1)
                            code_set.remove(c2)
                            code_set.add(d)
                            double_improved = True
                            no_improve_count = 0
                            break
                            
                    if double_improved:
                        continue
                
                # 4. 2-for-2 Swap: replace two codewords with two different ones
                if time.time() < deadline - 3:
                    swap22_improved = False
                    c_list = list(code_set)
                    rng.shuffle(c_list)
                    num_pairs_to_try = min(20, len(c_list) // 2)
                    
                    for _ in range(num_pairs_to_try):
                        if time.time() > deadline - 3:
                            break
                        if len(c_list) < 2:
                            break
                        c1, c2 = rng.sample(c_list, 2)
                        
                        # Essential words for each
                        n1 = neighbors[c1, :]
                        n2 = neighbors[c2, :]
                        e1 = n1[cov[n1] == 1]
                        e2 = n2[cov[n2] == 1]
                        
                        if len(e1) == 0 or len(e2) == 0:
                            continue
                        
                        # Find d1 that covers e1 (and is not c2)
                        if len(e1) > 3:
                            continue
                        cand1 = neighbor_matrix[e1[0], :].copy()
                        for ew in e1[1:]:
                            cand1 &= neighbor_matrix[ew, :]
                        if not np.any(cand1):
                            continue
                        cand1[c2] = 0  # exclude c2
                        c1_candidates = np.where(cand1)[0]
                        valid_c1 = [d for d in c1_candidates if d not in code_set or d == c1]
                        if not valid_c1:
                            continue
                        
                        # Find d2 that covers e2 (and is not c1 or d1)
                        if len(e2) > 3:
                            continue
                        cand2 = neighbor_matrix[e2[0], :].copy()
                        for ew in e2[1:]:
                            cand2 &= neighbor_matrix[ew, :]
                        if not np.any(cand2):
                            continue
                        cand2[c1] = 0
                        c2_candidates = np.where(cand2)[0]
                        
                        # Try to find d2 that's different from d1 choice
                        d1 = rng.choice(valid_c1)
                        cand2[d1] = 0
                        c2_candidates = np.where(cand2)[0]
                        valid_c2 = [d for d in c2_candidates if d not in code_set or d == c2]
                        if not valid_c2:
                            continue
                        
                        d2 = rng.choice(valid_c2)
                        
                        code_set.remove(c1)
                        code_set.remove(c2)
                        code_set.add(d1)
                        code_set.add(d2)
                        swap22_improved = True
                        no_improve_count = 0
                        break
                    
                    if swap22_improved:
                        continue
                
                no_improve_count += 1

            return code_set

        best_code = None
        best_size = float('inf')
        
        # Run multiple restarts
        for i in range(80):
            if time.time() > deadline - 8:
                break
                
            rng = np.random.default_rng(42 + i)
            
            code = greedy_init(rng, deadline)
            code = local_search(code, rng, deadline)
            
            if len(code) < best_size:
                best_size = len(code)
                best_code = code
                
            if best_size <= 486:
                # If we hit the known best, we can try to squeeze a bit more
                # or break if time is tight
                if best_size <= 480:
                    break

        if best_code is None:
            # Fallback
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            best_code = set()
            import itertools
            for h in tetracode:
                for tail in itertools.product(range(3), repeat=n - 4):
                    word = h + list(tail)
                    val = 0
                    for i, d in enumerate(word):
                        val += d * (q ** i)
                    best_code.add(val)

        final_code = []
        for w in best_code:
            word = []
            temp = w
            for _ in range(n):
                word.append(temp % q)
                temp //= q
            final_code.append(word)
        
        return final_code
    
    else:
        N = q ** n
        deadline = time.time() + 110.0
        powers = [q ** i for i in range(n)]
        neigh_size = 1 + n * (q - 1)
        neighbors = np.zeros((N, neigh_size), dtype=np.int32)
        
        for w in range(N):
            neighbors[w, 0] = w
            idx = 1
            for i in range(n):
                d = (w // powers[i]) % q
                for delta in range(1, q):
                    new_d = (d + delta) % q
                    diff = new_d - d
                    new_w = w + diff * powers[i]
                    neighbors[w, idx] = new_w
                    idx += 1
        
        rng = np.random.default_rng()
        uncovered = np.ones(N, dtype=bool)
        code = set()
        while np.any(uncovered) and time.time() < deadline:
            gains = (neighbors * uncovered[:, None]).sum(axis=1) # This is wrong, neighbors is 2D, uncovered is 1D
            # Correct:
            # gains[i] = sum(neighbors[i, :] * uncovered)
            # Use matrix multiplication if we had a matrix.
            # For generic case, just loop or use simple numpy
            gains = np.zeros(N)
            for i in range(N):
                gains[i] = np.sum(uncovered[neighbors[i, :]])
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            candidates = np.where(gains == max_gain)[0]
            w = rng.choice(candidates)
            code.add(w)
            uncovered[neighbors[w, :]] = False
        
        final_code = []
        for w in code:
            word = []
            temp = w
            for _ in range(n):
                word.append(temp % q)
                temp //= q
            final_code.append(word)
        return final_code
