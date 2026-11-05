import numpy as np
import itertools
import time
import random
from collections import deque

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    words = list(itertools.product(range(q), repeat=n))
    N = len(words)
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    # Precompute neighbors for radius 1 (including self)
    neighbors = []
    neighbor_sets = []
    for i, w in enumerate(words):
        neigh = [i]
        w_list = list(w)
        for pos in range(n):
            orig_val = w_list[pos]
            for val in range(q):
                if val == orig_val:
                    continue
                w_list[pos] = val
                idx = word_to_idx[tuple(w_list)]
                neigh.append(idx)
                w_list[pos] = orig_val
        neighbors.append(neigh)
        neighbor_sets.append(set(neigh))
    
    # Convert to numpy arrays for fast operations
    neighbor_arrays = [np.array(neigh, dtype=np.int32) for neigh in neighbors]
    # Build adjacency matrix (sparse-friendly: use the arrays directly)
    # Also precompute reverse neighbors (who covers each point)
    reverse_neighbors = [set() for _ in range(N)]
    for i in range(N):
        for j in neighbors[i]:
            reverse_neighbors[j].add(i)
    reverse_neighbor_arrays = [np.array(list(s), dtype=np.int32) for s in reverse_neighbors]
    
    # Phase 1: Greedy Set Cover with vectorized gain computation
    def greedy_cover(random_seed=0):
        rng = random.Random(random_seed)
        uncovered = np.ones(N, dtype=bool)
        code = []
        code_set = set()
        
        while np.any(uncovered):
            if time.time() > deadline - 30:
                break
            
            # Vectorized gain computation
            # For each candidate w, gain = number of uncovered points in neighbors[w]
            # = sum over uncovered points p of [w in reverse_neighbors[p]]
            # Equivalently: for each uncovered p, add 1 to all w in reverse_neighbors[p]
            gains = np.zeros(N, dtype=np.int32)
            uncovered_indices = np.where(uncovered)[0]
            
            for idx in uncovered_indices:
                gains[reverse_neighbor_arrays[idx]] += 1
            
            max_gain = gains.max()
            if max_gain == 0:
                break
            
            top_candidates = np.where(gains == max_gain)[0]
            best_idx = int(top_candidates[rng.randint(0, len(top_candidates)-1)])
            
            code.append(best_idx)
            code_set.add(best_idx)
            uncovered[neighbor_arrays[best_idx]] = False
        
        return code, code_set
    
    # Phase 2: Local search with multiple strategies
    def local_search(code, code_set, time_budget=60):
        counts = np.zeros(N, dtype=np.int32)
        for idx in code:
            counts[neighbor_arrays[idx]] += 1
        
        best_code = code[:]
        best_code_set = code_set.copy()
        best_size = len(code)
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            
            # --- Deletion Pass ---
            to_remove = []
            for c in code:
                if counts[neighbor_arrays[c]].min() > 1:
                    to_remove.append(c)
                
            if to_remove:
                improved = True
                for c in to_remove:
                    code.remove(c)
                    code_set.remove(c)
                    counts[neighbor_arrays[c]] -= 1
            
            if len(code) < best_size:
                best_code = code[:]
                best_code_set = code_set.copy()
                best_size = len(code)
            
            # --- Swap Pass: Try to replace one codeword with another ---
            random.shuffle(code)
            
            for c in code:
                if time.time() > deadline - 5:
                    break
                    
                # Get critical points: points covered only by c
                neigh_c = neighbor_arrays[c]
                crit_points = neigh_c[counts[neigh_c] == 1]
                
                if len(crit_points) == 0:
                    continue
                
                random.shuffle(crit_points.tolist())
                
                # Use the first critical point to generate candidates
                p = int(crit_points[0])
                cands = [int(x) for x in reverse_neighbor_arrays[p] if x not in code_set]
                
                best_w = -1
                best_score = -1
                
                for w in cands:
                    # Check if w covers all critical points
                    ws = neighbor_sets[w]
                    valid = all(k in ws for k in crit_points)
                    if not valid:
                        continue
                    
                    # Score: maximize coverage of points with count==2
                    score = int((counts[neighbor_arrays[w]] == 2).sum())
                    
                    if score > best_score:
                        best_score = score
                        best_w = w
                        
                if best_w != -1:
                    code.remove(c)
                    code_set.remove(c)
                    counts[neighbor_arrays[c]] -= 1
                    
                    code.append(best_w)
                    code_set.add(best_w)
                    counts[neighbor_arrays[best_w]] += 1
                    
                    improved = True
                    break
            
            # --- 2-for-1 Exchange: remove two codewords, add one ---
            if not improved and len(code) > 2:
                samples = min(80, len(code) * (len(code) - 1) // 2)
                for _ in range(samples):
                    if time.time() > deadline - 5:
                        break
                    
                    i, j = random.sample(range(len(code)), 2)
                    c1, c2 = code[i], code[j]
                    
                    # Points that would become uncovered if we remove both
                    temp_counts = counts.copy()
                    temp_counts[neighbor_arrays[c1]] -= 1
                    temp_counts[neighbor_arrays[c2]] -= 1
                    
                    uncovered_after = np.where(temp_counts == 0)[0]
                    
                    if len(uncovered_after) == 0:
                        # Both are redundant
                        code.remove(c1)
                        code.remove(c2)
                        code_set.remove(c1)
                        code_set.remove(c2)
                        counts[neighbor_arrays[c1]] -= 1
                        counts[neighbor_arrays[c2]] -= 1
                        improved = True
                        break
                    
                    # Try to find one replacement covering all uncovered_after
                    # Candidate must be in intersection of reverse_neighbors of all uncovered points
                    # Start with candidates from first uncovered point
                    first_point = int(uncovered_after[0])
                    cands = [int(x) for x in reverse_neighbor_arrays[first_point] if x not in code_set]
                    
                    best_single = -1
                    best_single_score = -1
                    
                    for w in cands:
                        ws = neighbor_sets[w]
                        if all(k in ws for k in uncovered_after):
                            score = int((counts[neighbor_arrays[w]] >= 2).sum())
                            if score > best_single_score:
                                best_single_score = score
                                best_single = w
                    
                    if best_single != -1:
                        code.remove(c1)
                        code.remove(c2)
                        code_set.remove(c1)
                        code_set.remove(c2)
                        counts[neighbor_arrays[c1]] -= 1
                        counts[neighbor_arrays[c2]] -= 1
                        
                        code.append(best_single)
                        code_set.add(best_single)
                        counts[neighbor_arrays[best_single]] += 1
                        
                        improved = True
                        break
        
        # Final cleanup
        counts = np.zeros(N, dtype=np.int32)
        for idx in best_code:
            counts[neighbor_arrays[idx]] += 1
        
        changed = True
        while changed:
            changed = False
            for c in list(best_code):
                if counts[neighbor_arrays[c]].min() > 1:
                    best_code.remove(c)
                    best_code_set.remove(c)
                    counts[neighbor_arrays[c]] -= 1
                    changed = True
        
        return best_code
    
    # Phase 3: Multi-start with different random seeds + ILS perturbation
    best_code = None
    best_size = float('inf')
    
    num_starts = 8
    for seed in range(num_starts):
        if time.time() > deadline - 20:
            break
        
        code, code_set = greedy_cover(random_seed=seed)
        
        if len(code) < best_size:
            best_code = code[:]
            best_size = len(code)
        
        # Local search
        refined = local_search(code[:], code_set.copy())
        
        if len(refined) < best_size:
            best_code = refined[:]
            best_size = len(refined)
    
    # Phase 4: Iterated Local Search with perturbation on the best solution
    if best_code is not None and time.time() < deadline - 5:
        current_code = best_code[:]
        current_set = set(current_code)
        current_size = len(current_code)
        
        ilt = 0
        while time.time() < deadline - 3:
            ilt += 1
            # Perturbation: remove k random codewords, re-greedy to fill gaps
            k = max(1, current_size // 20)
            perturbed = current_code[:]
            perturbed_set = set(perturbed)
            to_remove = random.sample(perturbed, min(k, len(perturbed)))
            for c in to_remove:
                perturbed.remove(c)
                perturbed_set.remove(c)
            
            # Quick greedy fill
            counts = np.zeros(N, dtype=np.int32)
            for idx in perturbed:
                counts[neighbor_arrays[idx]] += 1
            uncovered = np.where(counts == 0)[0]
            
            while len(uncovered) > 0:
                if time.time() > deadline - 3:
                    break
                # Pick a random uncovered point, find best candidate for it
                p = int(uncovered[random.randint(0, len(uncovered)-1)])
                cands = [int(x) for x in reverse_neighbor_arrays[p] if x not in perturbed_set]
                if not cands:
                    break
                # Score by coverage of currently uncovered
                best_w = -1
                best_sc = -1
                for w in cands:
                    sc = int((counts[neighbor_arrays[w]] == 0).sum())
                    if sc > best_sc:
                        best_sc = sc
                        best_w = w
                if best_w == -1:
                    break
                perturbed.append(best_w)
                perturbed_set.add(best_w)
                counts[neighbor_arrays[best_w]] += 1
                uncovered = np.where(counts == 0)[0]
            
            # Local search on perturbed solution
            refined = local_search(perturbed[:], perturbed_set.copy())
            
            if len(refined) < current_size:
                current_code = refined[:]
                current_set = set(refined)
                current_size = len(refined)
                if current_size < best_size:
                    best_code = current_code[:]
                    best_size = current_size
    
    if best_code is None:
        best_code = [0]
    
    return [list(words[i]) for i in best_code]
