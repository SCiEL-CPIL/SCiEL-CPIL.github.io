import itertools
import numpy as np
import random
import time

def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
        # Fallback for non-target parameters
        all_words = list(itertools.product(range(q), repeat=n))
        word_to_idx = {w: i for i, w in enumerate(all_words)}
        neighbors = []
        for i, w in enumerate(all_words):
            neigh_set = set()
            neigh_set.add(i)
            for pos in range(n):
                for val in range(q):
                    if val == w[pos]:
                        continue
                    new_w_list = list(w)
                    new_w_list[pos] = val
                    new_w = tuple(new_w_list)
                    neigh_set.add(word_to_idx[new_w])
            neighbors.append(list(neigh_set))
        
        N = q ** n
        uncovered = np.ones(N, dtype=bool)
        C = []
        start_time = time.time()
        time_limit = 100.0
        
        while np.any(uncovered):
            best_idx = -1
            best_count = -1
            for i in range(N):
                cnt = np.sum(uncovered[neighbors[i]])
                if cnt > best_count:
                    best_count = cnt
                    best_idx = i
            if best_count == 0: break
            C.append(best_idx)
            uncovered[neighbors[best_idx]] = False
            if time.time() - start_time > time_limit: break
            
        # Reduction
        coverage_counts = np.zeros(N, dtype=int)
        for c in C:
            coverage_counts[neighbors[c]] += 1
        changed = True
        while changed:
            changed = False
            new_C = []
            for c in C:
                if np.min(coverage_counts[neighbors[c]]) >= 2:
                    coverage_counts[neighbors[c]] -= 1
                    changed = True
                else:
                    new_C.append(c)
            C = new_C
        return [list(all_words[i]) for i in C]

    # Specific optimization for q=3, n=8, radius=1
    N = 6561
    all_words = list(itertools.product(range(3), repeat=8))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbors as numpy arrays for fast operations
    neighbors = []
    for i, w in enumerate(all_words):
        neigh_set = set()
        neigh_set.add(i)
        for pos in range(8):
            for val in range(3):
                if val == w[pos]:
                    continue
                new_w_list = list(w)
                new_w_list[pos] = val
                new_w = tuple(new_w_list)
                neigh_set.add(word_to_idx[new_w])
        neighbors.append(np.array(list(neigh_set), dtype=int))
    
    # Precompute neighbor sets as Python sets for fast intersection
    neighbor_sets = [set(neigh.tolist()) for neigh in neighbors]
    
    def get_counts(code):
        counts = np.zeros(N, dtype=int)
        for c in code:
            counts[neighbors[c]] += 1
        return counts
    
    def reduce_code(code, counts):
        """
        Reduces code by removing redundant words.
        Returns new_code and new_counts.
        """
        if not code:
            return code, counts
        current_code = list(code)
        current_counts = counts.copy()
        
        while True:
            removed_any = False
            next_code = []
            for c in current_code:
                if np.min(current_counts[neighbors[c]]) >= 2:
                    # Remove c
                    current_counts[neighbors[c]] -= 1
                    removed_any = True
                else:
                    next_code.append(c)
            current_code = next_code
            if not removed_any:
                break
        return current_code, current_counts

    # 1. Tensor-product initialization: split 8D into two 4D subspaces
    start_time = time.time()
    time_limit = 115.0
    
    # Decompose: first 4 coords (A) and last 4 coords (B)
    # Each 4D subspace has 3^4 = 81 words
    # Ball in 8D: a word w=(a,b) is covered by c=(c_a,c_b) if d(a,c_a)+d(b,c_b)<=1
    # i.e., (d(a,c_a)<=1 and d(b,c_b)==0) or (d(a,c_a)==0 and d(b,c_b)<=1)
    # or (d(a,c_a)==0 and d(b,c_b)==0)
    
    # Precompute 4D neighbors
    words_4d = list(itertools.product(range(3), repeat=4))
    idx_4d = {w: i for i, w in enumerate(words_4d)}
    N4 = 81
    neighbors_4d = []
    for i, w in enumerate(words_4d):
        ns = set([i])
        for pos in range(4):
            for val in range(3):
                if val == w[pos]:
                    continue
                nw = list(w)
                nw[pos] = val
                ns.add(idx_4d[tuple(nw)])
        neighbors_4d.append(np.array(list(ns), dtype=int))
    
    # Map 8D index to (a_idx, b_idx)
    # 8D index = a_idx * 81 + b_idx where a_idx is first 4 coords, b_idx is last 4
    # Actually: index = w[0]*3^7 + w[1]*3^6 + ... + w[7]
    # a_idx = w[0]*3^3 + w[1]*3^2 + w[2]*3 + w[3]
    # b_idx = w[4]*3^3 + w[5]*3^2 + w[6]*3 + w[7]
    # 8D index = a_idx * 81 + b_idx
    
    # Generate structured initial code using slice decomposition
    # For each pair (a_code, b_code) in product of 4D codes, add (a_code, b_code)
    # But we need a smarter approach: use a small 4D covering code for each subspace
    
    # 4D ternary covering code with radius 1: need to cover 81 words, ball size in 4D = 1 + 4*2 = 9
    # Lower bound: ceil(81/9) = 9, so ~9-12 words per subspace
    # Use greedy to find small 4D code
    def greedy_4d():
        uncovered = np.ones(N4, dtype=bool)
        code = []
        while np.any(uncovered):
            best = -1
            best_cnt = -1
            for i in range(N4):
                cnt = np.sum(uncovered[neighbors_4d[i]])
                if cnt > best_cnt:
                    best_cnt = cnt
                    best = i
            if best_cnt <= 0:
                break
            code.append(best)
            uncovered[neighbors_4d[best]] = False
        return code
    
    code_4d = greedy_4d()
    
    # Build 8D code as product: for each a in code_4d, for each b in code_4d, add (a,b)
    # This gives |code_4d|^2 codewords
    # But this may be too large. Let's use a smarter approach:
    # For each a in code_4d, cover all b with a single "universal" b, plus targeted b's
    
    # Actually, let's use the product but then reduce heavily
    C = []
    for a in code_4d:
        for b in code_4d:
            C.append(a * N4 + b)
    
    # Reduce the product code
    def get_counts(code):
        counts = np.zeros(N, dtype=int)
        for c in code:
            counts[neighbors[c]] += 1
        return counts
    
    def reduce_code(code, counts):
        if not code:
            return code, counts
        current_code = list(code)
        current_counts = counts.copy()
        while True:
            removed_any = False
            next_code = []
            for c in current_code:
                if np.min(current_counts[neighbors[c]]) >= 2:
                    current_counts[neighbors[c]] -= 1
                    removed_any = True
                else:
                    next_code.append(c)
            current_code = next_code
            if not removed_any:
                break
        return current_code, current_counts
    
    C, counts = reduce_code(C, get_counts(C))
    
    # 2. Slice-aware local search with smooth surrogate scoring
    best_size = len(C)
    best_C = C.copy()
    best_counts = counts.copy()
    code_set = set(C)
    
    # Precompute slice membership: for each 8D word, its (a_idx, b_idx)
    # a_idx = first 4 coords mapped to 0..80, b_idx = last 4 coords mapped to 0..80
    # 8D index i: word = all_words[i]
    # a_idx = word[0]*27 + word[1]*9 + word[2]*3 + word[3]
    # b_idx = word[4]*27 + word[5]*9 + word[6]*3 + word[7]
    
    # Local Search Loop with slice-awareness
    no_improve_count = 0
    while time.time() - start_time < time_limit:
        # 1. Reduction Pass
        old_size = len(C)
        C, counts = reduce_code(C, counts)
        code_set = set(C)
        
        if len(C) < best_size:
            best_size = len(C)
            best_C = C.copy()
            best_counts = counts.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        if time.time() - start_time > time_limit - 5:
            break
        
        if no_improve_count > 50:
            break
            
        # 2. Slice-aware Swap Pass
        if len(C) == old_size:
            improved = False
            indices = list(range(len(C)))
            random.shuffle(indices)
            
            for idx in indices:
                if time.time() - start_time > time_limit - 5:
                    break
                c = C[idx]
                u_indices = neighbors[c]
                unique_mask = counts[u_indices] == 1
                unique_points = u_indices[unique_mask]
                
                if len(unique_points) == 0:
                    C.remove(c)
                    code_set.discard(c)
                    counts[neighbors[c]] -= 1
                    improved = True
                    break
                
                if len(unique_points) > 4:
                    continue
                
                # Find candidates covering all unique points
                if len(unique_points) == 1:
                    candidates_set = set(neighbor_sets[unique_points[0]])
                else:
                    candidates_set = set(neighbor_sets[unique_points[0]])
                    for u in unique_points[1:]:
                        candidates_set &= neighbor_sets[u]
                        if not candidates_set:
                            break
                
                if not candidates_set:
                    continue
                
                candidates_set.difference_update(code_set)
                if not candidates_set:
                    continue
                    
                # Slice-aware scoring: prefer candidates in same or adjacent slices
                c_a = c // N4
                c_b = c % N4
                
                best_c_prime = None
                best_score = -1e9
                cand_list = list(candidates_set)
                if len(cand_list) > 50:
                    cand_list = random.sample(cand_list, 50)
                
                for c_prime in cand_list:
                    new_ball = neighbors[c_prime]
                    # Smooth surrogate: weighted sum of coverage risks
                    # Lower is better for risk
                    risky_in_new = np.sum(counts[new_ball] == 1)
                    safe_in_new = np.sum(counts[new_ball] >= 2)
                    
                    # Slice bonus: prefer same a-slice or b-slice
                    p_a = c_prime // N4
                    p_b = c_prime % N4
                    slice_bonus = 0
                    if p_a == c_a:
                        slice_bonus += 5
                    if p_b == c_b:
                        slice_bonus += 5
                    
                    # Score: maximize safety, minimize risk, add slice bonus
                    score = 2.0 * safe_in_new - 1.5 * risky_in_new + slice_bonus
                    
                    if score > best_score:
                        best_score = score
                        best_c_prime = c_prime
                
                if best_c_prime is None:
                    continue
                    
                c_prime = best_c_prime
                
                old_C = C.copy()
                old_counts = counts.copy()
                old_code_set = code_set.copy()
                
                C.remove(c)
                code_set.discard(c)
                C.append(c_prime)
                code_set.add(c_prime)
                
                counts[neighbors[c]] -= 1
                counts[neighbors[c_prime]] += 1
                
                if np.any(counts < 1):
                    C = old_C
                    counts = old_counts
                    code_set = old_code_set
                    continue
                
                new_C, new_counts = reduce_code(C, counts)
                new_code_set = set(new_C)
                
                if len(new_C) < len(old_C):
                    C = new_C
                    counts = new_counts
                    code_set = new_code_set
                    if len(C) < best_size:
                        best_size = len(C)
                        best_C = C.copy()
                        best_counts = counts.copy()
                    improved = True
                    break
                    
                else:
                    # Accept neutral moves with some probability for exploration
                    if random.random() < 0.3:
                        C = new_C
                        counts = new_counts
                        code_set = new_code_set
                        improved = True
                        break
            
            if not improved:
                # Random kicks with slice awareness
                for _ in range(20):
                    if time.time() - start_time > time_limit - 5: break
                    if not C: break
                    c_idx = random.randint(0, len(C)-1)
                    c = C[c_idx]
                    cand = neighbors[c]
                    c_prime = cand[random.randint(0, len(cand)-1)]
                    if c_prime in code_set: continue
                    
                    old_C = C.copy()
                    old_counts = counts.copy()
                    old_code_set = code_set.copy()
                    
                    C.remove(c)
                    code_set.discard(c)
                    C.append(c_prime)
                    code_set.add(c_prime)
                    counts[neighbors[c]] -= 1
                    counts[neighbors[c_prime]] += 1
                    
                    if np.any(counts < 1):
                        C = old_C
                        counts = old_counts
                        code_set = old_code_set
                        continue
                        
                    new_C, new_counts = reduce_code(C, counts)
                    new_code_set = set(new_C)
                    if len(new_C) < len(old_C):
                        C = new_C
                        counts = new_counts
                        code_set = new_code_set
                        if len(new_C) < best_size:
                            best_size = len(new_C)
                            best_C = C.copy()
                            best_counts = counts.copy()
                        break
                    else:
                        if random.random() < 0.2:
                            C = new_C
                            counts = new_counts
                            code_set = new_code_set

    # Final Reduction with multiple passes
    for _ in range(3):
        best_C, _ = reduce_code(best_C, get_counts(best_C))
    
    return [list(all_words[i]) for i in best_C]
