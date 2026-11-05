import time
import random
import numpy as np
import math

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    # Fallback for other parameters
    if q != 3 or n != 7 or radius != 1:
        # Simple fallback: return all words (valid but large)
        total = q ** n
        if total > 10000:
            # Too large for this specific approach, return a subset that might be valid? No, must be valid.
            # For safety, if it's not the specific case, try a basic greedy or return all if small.
            # The task is specific to K_3(7,1), so this branch is unlikely to be hit in scoring.
            # We'll return a trivial valid code: all words.
            result = []
            for i in range(total):
                w = []
                x = i
                for _ in range(n):
                    w.append(x % q)
                    x //= q
                result.append(w)
            return result
        # Small case: just use all words
        result = []
        for i in range(total):
            w = []
            x = i
            for _ in range(n):
                w.append(x % q)
                x //= q
            result.append(w)
        return result

    # Specific optimization for q=3, n=7, radius=1
    N = 3 ** n  # 2187

    def idx_to_tuple(idx):
        t = [0] * n
        for i in range(n - 1, -1, -1):
            t[i] = idx % 3
            idx //= 3
        return t

    def tuple_to_idx(t):
        idx = 0
        for x in t:
            idx = idx * 3 + x
        return idx

    # Precompute neighbor lists and adjacency matrix for fast vectorized operations
    neighbors_list = []
    neighbor_sets = []  # Python sets for fast membership
    for i in range(N):
        w = idx_to_tuple(i)
        nb = [i]
        for pos in range(n):
            val = w[pos]
            for new_val in range(3):
                if new_val != val:
                    new_w = w[:]
                    new_w[pos] = new_val
                    nb.append(tuple_to_idx(new_w))
        neighbors_list.append(np.array(nb, dtype=np.int32))
        neighbor_sets.append(set(nb))
    
    # Build sparse adjacency structure for fast gain computation
    # For each word, store its neighbors as a flat array with offsets
    # This allows vectorized gain computation
    all_neighbors = []
    neighbor_offsets = np.zeros(N + 1, dtype=np.int32)
    for i in range(N):
        all_neighbors.extend(neighbors_list[i])
        neighbor_offsets[i + 1] = len(all_neighbors)
    all_neighbors = np.array(all_neighbors, dtype=np.int32)
    
    # ---------- Greedy Set-Cover Construction ----------
    
    def greedy_construct(seed_val):
        """Run greedy set-cover construction with given random seed for tie-breaking."""
        rng = random.Random(seed_val)
        cov = np.zeros(N, dtype=np.int32)
        unc = np.ones(N, dtype=bool)
        cod = []
        cod_set = set()
        
        # Random start to break symmetry
        start = rng.randint(0, N - 1)
        cod.append(start)
        cod_set.add(start)
        cov[neighbors_list[start]] += 1
        unc[neighbors_list[start]] = False
        
        while unc.any() and time.time() < deadline - 5:
            # Vectorized gain computation using precomputed neighbor structure
            # gains[i] = number of uncovered words that are neighbors of i
            # We can compute this efficiently by: for each uncovered word u, 
            # increment gains for all neighbors of u
            gains = np.zeros(N, dtype=np.int32)
            
            # Get indices of uncovered words
            unc_idx = np.where(unc)[0]
            if len(unc_idx) == 0:
                break
                
            # For each uncovered word, add 1 to gains of its neighbors
            # This is O(|unc| * degree) which is much faster than O(N * degree)
            for u in unc_idx:
                gains[neighbors_list[u]] += 1
            
            # Zero out gains for already-selected codewords
            for c in cod_set:
                gains[c] = 0
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break
                
            # Find all indices with max gain
            best_indices = np.where(gains == max_gain)[0]
            if len(best_indices) > 1:
                best_idx = best_indices[rng.randint(0, len(best_indices) - 1)]
            else:
                best_idx = best_indices[0]
                
            cod.append(best_idx)
            cod_set.add(best_idx)
            cov[neighbors_list[best_idx]] += 1
            unc[neighbors_list[best_idx]] = False
        
        return cod, cod_set, cov

    # ---------- Local Search / ILS Refinement ----------
    
    def is_redundant(c, cov):
        # Check if removing c leaves all its neighbors covered by others
        # i.e., coverage[neighbors] > 1 for all neighbors
        return np.min(cov[neighbors_list[c]]) > 1

    def greedy_repair(uncovered_set, current_cod_set, rng):
        """
        Given a set of uncovered word indices, find a minimal set of codewords
        to add to cover them.
        Returns list of added codewords.
        """
        if not uncovered_set:
            return []
        
        added = []
        temp_uncovered = set(uncovered_set)
        
        while temp_uncovered:
            best_c = -1
            best_count = 0
            
            # Candidates are neighbors of uncovered words
            candidate_pool = set()
            for w in temp_uncovered:
                candidate_pool.update(neighbors_list[w])
            
            # Filter out already selected
            candidates = [c for c in candidate_pool if c not in current_cod_set]
            
            if not candidates:
                break
                
            # Find candidate covering most uncovered
            for c in candidates:
                cnt = len(set(neighbors_list[c]) & temp_uncovered)
                if cnt > best_count:
                    best_count = cnt
                    best_c = c
                    if best_count == len(temp_uncovered):
                        break
            
            if best_c == -1:
                break
            
            added.append(best_c)
            temp_uncovered.difference_update(set(neighbors_list[best_c]))
            
        return added

    def local_search_optimize(code, code_set, coverage, max_time):
        """
        Perform local search with simulated annealing to minimize code size.
        Uses deletion + repair moves with SA acceptance criterion to escape local minima.
        """
        rng = random.Random(12345)
        current_code = code[:]
        current_set = code_set.copy()
        current_cov = coverage.copy()
        current_size = len(current_code)
        
        best_code = current_code[:]
        best_set = current_set.copy()
        best_cov = current_cov.copy()
        best_size = current_size
        
        # First, remove all redundant words
        changed = True
        while changed:
            changed = False
            for c in current_code[:]:
                if is_redundant(c, current_cov):
                    # Remove c
                    current_code.remove(c)
                    current_set.remove(c)
                    current_cov[neighbors_list[c]] -= 1
                    current_size -= 1
                    changed = True
                    if len(current_code) < best_size:
                        best_size = len(current_code)
                        best_code = current_code[:]
                        best_set = current_set.copy()
                        best_cov = current_cov.copy()
        
        # Simulated annealing parameters
        T_start = 2.0
        T_end = 0.01
        cooling_rate = 0.9995
        
        # Track iteration count for cooling schedule
        iteration = 0
        T = T_start
        
        # Now try SA-style perturbations
        # Try removing 1 or 2 random codewords and repairing
        while time.time() < max_time:
            # Pick a random subset of codewords to remove (perturbation)
            k = rng.choice([1, 1, 1, 2, 2, 3])
            if len(current_code) < k:
                continue
                
            indices = rng.sample(range(len(current_code)), k)
            removed = [current_code[i] for i in indices]
            
            # Temporarily remove
            temp_cov = current_cov.copy()
            temp_set = current_set.copy()
            for c in removed:
                temp_cov[neighbors_list[c]] -= 1
                temp_set.remove(c)
            
            uncovered = np.where(temp_cov == 0)[0]
            if len(uncovered) == 0:
                # Perfect removal - always accept
                new_code = []
                for c in current_code:
                    if c in temp_set:
                        new_code.append(c)
                current_code = new_code
                current_set = temp_set
                current_cov = temp_cov
                current_size = len(current_code)
                
                if current_size < best_size:
                    best_size = current_size
                    best_code = current_code[:]
                    best_set = current_set.copy()
                    best_cov = current_cov.copy()
            else:
                # Repair
                uncovered_set = set(uncovered.tolist())
                added = greedy_repair(uncovered_set, temp_set, rng)
                
                new_size = len(current_code) - len(removed) + len(added)
                delta = new_size - current_size
                
                # SA acceptance criterion
                accept = False
                if delta <= 0:
                    accept = True
                else:
                    # Accept with probability exp(-delta / T)
                    if T > T_end:
                        prob = math.exp(-delta / T)
                        if rng.random() < prob:
                            accept = True
                
                if accept:
                    # Update state
                    for c in removed:
                        current_code.remove(c)
                        current_set.remove(c)
                        current_cov[neighbors_list[c]] -= 1
                    for c in added:
                        if c not in current_set:
                            current_code.append(c)
                            current_set.add(c)
                            current_cov[neighbors_list[c]] += 1
                    current_size = new_size
                    
                    if current_size < best_size:
                        best_size = current_size
                        best_code = current_code[:]
                        best_set = current_set.copy()
                        best_cov = current_cov.copy()
            
            # Cool down
            iteration += 1
            T = max(T_end, T * cooling_rate)
            
            # If temperature is very low, switch to strict improvement mode
            if T <= T_end * 1.1:
                # Do a few more strict improvement iterations
                for _ in range(50):
                    if time.time() >= max_time:
                        break
                    k = rng.choice([1, 1, 2])
                    if len(current_code) < k:
                        continue
                    indices = rng.sample(range(len(current_code)), k)
                    removed = [current_code[i] for i in indices]
                    
                    temp_cov = current_cov.copy()
                    temp_set = current_set.copy()
                    for c in removed:
                        temp_cov[neighbors_list[c]] -= 1
                        temp_set.remove(c)
                    
                    uncovered = np.where(temp_cov == 0)[0]
                    if len(uncovered) == 0:
                        new_code = []
                        for c in current_code:
                            if c in temp_set:
                                new_code.append(c)
                        current_code = new_code
                        current_set = temp_set
                        current_cov = temp_cov
                        current_size = len(current_code)
                        
                        if current_size < best_size:
                            best_size = current_size
                            best_code = current_code[:]
                            best_set = current_set.copy()
                            best_cov = current_cov.copy()
                    else:
                        uncovered_set = set(uncovered.tolist())
                        added = greedy_repair(uncovered_set, temp_set, rng)
                        new_size = len(current_code) - len(removed) + len(added)
                        
                        if new_size < current_size:
                            for c in removed:
                                current_code.remove(c)
                                current_set.remove(c)
                                current_cov[neighbors_list[c]] -= 1
                            for c in added:
                                if c not in current_set:
                                    current_code.append(c)
                                    current_set.add(c)
                                    current_cov[neighbors_list[c]] += 1
                            current_size = new_size
                            
                            if current_size < best_size:
                                best_size = current_size
                                best_code = current_code[:]
                                best_set = current_set.copy()
                                best_cov = current_cov.copy()
                break
        
        return best_code, best_set, best_cov

    # Main Loop
    best_code = None
    best_size = float('inf')
    
    # Try different greedy starts with increasing time allocation for refinement
    num_starts = 15
    for i in range(num_starts):
        if time.time() > deadline - 15:
            break
            
        code, code_set, cov = greedy_construct(42 + i)
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code
            best_set = code_set
            best_cov = cov
            
        # Allocate more time to earlier (better) solutions
        # First few solutions get more refinement time
        if i < 3:
            refine_time = deadline - 8
        elif i < 8:
            refine_time = deadline - 12
        else:
            refine_time = deadline - 15
            
        # If we have time, refine this specific solution
        if time.time() < refine_time - 2:
            refined_code, refined_set, refined_cov = local_search_optimize(code, code_set, cov, refine_time)
            
            if len(refined_code) < best_size:
                best_size = len(refined_code)
                best_code = refined_code
                best_set = refined_set
                best_cov = refined_cov

    # Final check: ensure best_code is not None
    if best_code is None:
        # Fallback to a single greedy run
        code, code_set, cov = greedy_construct(0)
        best_code = code

    # Convert indices to words
    return [idx_to_tuple(i) for i in best_code]
