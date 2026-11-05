import numpy as np
import time
import random
import sys

def construct(q, n, radius):
    # Handle non-target parameters with a generic greedy fallback
    if q != 3 or n != 8 or radius != 1:
        N = q ** n
        if N > 1000000:
            return [[0]*n]
        
        words = []
        for i in range(N):
            w = []
            x = i
            for _ in range(n):
                w.append(x % q)
                x //= q
            w.reverse()
            words.append(tuple(w))
        
        word_to_idx = {w: i for i, w in enumerate(words)}
        
        neighbors = []
        for i, w in enumerate(words):
            neigh = set([i])
            for pos in range(n):
                for val in range(q):
                    if val == w[pos]: continue
                    new_w = list(w)
                    new_w[pos] = val
                    new_w = tuple(new_w)
                    neigh.add(word_to_idx[new_w])
            neighbors.append(list(neigh))
        
        uncovered = np.ones(N, dtype=bool)
        C = []
        start = time.time()
        limit = 100.0
        while np.any(uncovered):
            best_idx = -1
            best_cnt = -1
            for i in range(N):
                cnt = np.sum(uncovered[neighbors[i]])
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_idx = i
            if best_cnt <= 0: break
            C.append(best_idx)
            uncovered[neighbors[best_idx]] = False
            if time.time() - start > limit: break
            
        counts = np.zeros(N, dtype=int)
        for c in C:
            counts[neighbors[c]] += 1
        changed = True
        while changed:
            changed = False
            new_C = []
            for c in C:
                if np.min(counts[neighbors[c]]) >= 2:
                    counts[neighbors[c]] -= 1
                    changed = True
                else:
                    new_C.append(c)
            C = new_C
        return [list(words[i]) for i in C]

    # Spectral relaxation + coordinate descent for q=3, n=8, radius=1
    N = 6561
    
    # Generate all words
    words = []
    for i in range(N):
        w = []
        x = i
        for _ in range(8):
            w.append(x % 3)
            x //= 3
        w.reverse()
        words.append(tuple(w))
    
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    # Precompute neighbors: each word has itself + 16 neighbors = 17 total
    neighbors = np.zeros((N, 17), dtype=int)
    for i, w in enumerate(words):
        neighbors[i, 0] = i
        cnt = 1
        for pos in range(8):
            for delta in [1, 2]:
                new_val = (w[pos] + delta) % 3
                new_w = list(w)
                new_w[pos] = new_val
                new_w = tuple(new_w)
                neighbors[i, cnt] = word_to_idx[new_w]
                cnt += 1
    
    start_time = time.time()
    time_limit = 115.0
    
    # Phase 1: Greedy initialization to get a starting solution
    uncovered = np.ones(N, dtype=bool)
    C_init = []
    while np.any(uncovered):
        best_idx = -1
        best_cnt = -1
        for i in range(N):
            cnt = np.sum(uncovered[neighbors[i]])
            if cnt > best_cnt:
                best_cnt = cnt
                best_idx = i
        if best_cnt <= 0: break
        C_init.append(best_idx)
        uncovered[neighbors[best_idx]] = False
        if time.time() - start_time > time_limit * 0.3:
            break
    
    # Phase 2: Spectral relaxation via coordinate descent
    # We want to minimize sum(x_i) subject to A @ x >= 1, x in [0,1]
    # where A[i,j] = 1 if word i is in neighbors of word j
    # Use projected gradient descent with soft constraints
    
    # Build the incidence matrix A (N x N) implicitly
    # A[i,j] = 1 if i is in neighbors[j]
    # For coordinate descent, we need column sums and row coverage
    
    # Start with x = 1 for initial codewords, 0 otherwise
    x = np.zeros(N, dtype=np.float64)
    for c in C_init:
        x[c] = 1.0
    
    # Coverage: cov[i] = sum_{j: i in neighbors[j]} x[j]
    # cov = A.T @ x where A[i,j] = 1 if i in neighbors[j]
    # Which means A.T[j,i] = A[i,j], so cov = A.T @ x
    
    # Actually, let's define: for each codeword j, it covers neighbors[j]
    # So coverage of point i = sum over j where i in neighbors[j] of x[j]
    # That's: cov[i] = sum_j A[i,j] * x[j] where A[i,j] = 1 if i in neighbors[j]
    # cov = A @ x
    
    # Precompute A as sparse-like structure
    # For each point i, which codewords cover it?
    # cov[i] = sum_{j: i in neighbors[j]} x[j]
    
    # Let's precompute: for each point i, the list of codewords that cover it
    coverers = [[] for _ in range(N)]
    for j in range(N):
        for i in neighbors[j]:
            coverers[i].append(j)
    
    # Compute initial coverage
    cov = np.zeros(N, dtype=np.float64)
    for i in range(N):
        s = 0.0
        for j in coverers[i]:
            s += x[j]
        cov[i] = s
    
    # Coordinate descent: minimize sum(x) subject to cov >= 1, x in [0,1]
    # For each coordinate j, the optimal x[j] given others is:
    # We need cov[i] >= 1 for all i in neighbors[j]
    # cov[i] = x[j] + sum_{k != j, i in neighbors[k]} x[k]
    # So x[j] >= max(0, 1 - sum_{k != j, i in neighbors[k]} x[k]) for all i in neighbors[j]
    # And x[j] <= 1
    # To minimize, set x[j] = max(0, max_{i in neighbors[j]} (1 - (cov[i] - x[j])))
    # But cov[i] includes x[j], so: cov[i] - x[j] = sum_{k != j} A[i,k] x[k]
    # So x[j]_opt = max(0, max_{i in neighbors[j]} (1 - (cov[i] - x[j])))
    # = max(0, 1 - min_{i in neighbors[j]} (cov[i] - x[j]))
    
    # For efficiency, precompute for each j: min_{i in neighbors[j]} cov[i]
    # But cov changes as we update x, so we need to update incrementally
    
    # Simpler approach: iterate coordinates, for each j compute the required x[j]
    # and update cov incrementally
    
    # Use a priority queue or just cycle through coordinates
    
    best_x = x.copy()
    best_size = int(np.sum(x > 0.5))
    
    # Number of iterations based on time
    max_iters = 2000000
    iter_count = 0
    
    # Random order for coordinate descent
    order = list(range(N))
    
    while iter_count < max_iters:
        if time.time() - start_time > time_limit * 0.85:
            break
        
        # Randomly pick a coordinate to update
        j = random.randint(0, N - 1)
        old_xj = x[j]
        
        if old_xj == 0.0:
            # Try setting to minimum needed
            # Need: for all i in neighbors[j], cov[i] + new_xj >= 1
            # new_xj >= 1 - cov[i] for all i in neighbors[j]
            # new_xj = max(0, max_{i in neighbors[j]} (1 - cov[i]))
            # But only if this helps (i.e., there's an uncovered point)
            min_cov = np.min(cov[neighbors[j]])
            if min_cov < 1.0:
                new_xj = 1.0 - min_cov
                new_xj = min(new_xj, 1.0)
                if new_xj > 0.0:
                    x[j] = new_xj
                    for i in neighbors[j]:
                        cov[i] += new_xj
                    iter_count += 1
                    continue
            iter_count += 1
            continue
        
        # Try reducing x[j]
        # Can reduce to: max(0, max_{i in neighbors[j]} (1 - (cov[i] - old_xj)))
        # = max(0, 1 - min_{i in neighbors[j]} (cov[i] - old_xj))
        min_cov_excl = np.min(cov[neighbors[j]] - old_xj)
        new_xj = 1.0 - min_cov_excl
        new_xj = max(0.0, new_xj)
        new_xj = min(new_xj, old_xj)
        
        if new_xj < old_xj:
            delta = old_xj - new_xj
            x[j] = new_xj
            for i in neighbors[j]:
                cov[i] -= delta
            iter_count += 1
            continue
        
        iter_count += 1
    
    # Phase 3: Extract discrete solution from continuous relaxation
    # Threshold x at 0.5, then repair coverage
    C_cont = [i for i in range(N) if x[i] > 0.5]
    
    # Check coverage and repair
    cov_check = np.zeros(N, dtype=np.float64)
    for c in C_cont:
        cov_check[neighbors[c]] += 1.0
    
    # Add missing coverage
    uncovered_mask = cov_check < 1.0
    if np.any(uncovered_mask):
        # Greedy add to cover remaining
        while np.any(uncovered_mask):
            if time.time() - start_time > time_limit * 0.95:
                break
            best_idx = -1
            best_cnt = -1
            for i in range(N):
                cnt = np.sum(uncovered_mask[neighbors[i]])
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_idx = i
            if best_cnt <= 0: break
            C_cont.append(best_idx)
            uncovered_mask[neighbors[best_idx]] = False
    
    C_final = list(set(C_cont))
    
    # Phase 4: Local search refinement
    def calc_counts(code):
        counts = np.zeros(N, dtype=int)
        for c in code:
            counts[neighbors[c]] += 1
        return counts
    
    def reduce_code(code, counts):
        current_code = list(code)
        current_counts = counts.copy()
        while True:
            removed = False
            next_code = []
            for c in current_code:
                if np.min(current_counts[neighbors[c]]) >= 2:
                    current_counts[neighbors[c]] -= 1
                    removed = True
                else:
                    next_code.append(c)
            current_code = next_code
            if not removed:
                break
        return current_code, current_counts
    
    C_final, counts = reduce_code(C_final, calc_counts(C_final))
    best_size = len(C_final)
    best_C = C_final.copy()
    best_counts = counts.copy()
    
    # Local search with swaps
    while time.time() - start_time < time_limit:
        old_size = len(C_final)
        C_final, counts = reduce_code(C_final, counts)
        
        if len(C_final) < best_size:
            best_size = len(C_final)
            best_C = C_final.copy()
            best_counts = counts.copy()
        
        if time.time() - start_time > time_limit * 0.98:
            break
        
        if len(C_final) == old_size:
            improved = False
            indices = list(range(len(C_final)))
            random.shuffle(indices)
            
            for idx in indices:
                if time.time() - start_time > time_limit * 0.98:
                    break
                c = C_final[idx]
                u_indices = neighbors[c]
                unique_mask = counts[u_indices] == 1
                unique_points = u_indices[unique_mask]
                
                if len(unique_points) == 0:
                    C_final.remove(c)
                    counts[neighbors[c]] -= 1
                    improved = True
                    break
                
                if len(unique_points) > 3:
                    continue
                
                if len(unique_points) == 1:
                    candidates_set = set(neighbors[unique_points[0]].tolist())
                else:
                    candidates_set = set(neighbors[unique_points[0]].tolist())
                    for u in unique_points[1:]:
                        candidates_set &= set(neighbors[u].tolist())
                        if not candidates_set:
                            break
                
                if not candidates_set:
                    continue
                
                candidates_set.difference_update(C_final)
                
                if not candidates_set:
                    continue
                
                c_prime = random.choice(list(candidates_set))
                
                old_C = C_final.copy()
                old_counts = counts.copy()
                
                C_final.remove(c)
                C_final.append(c_prime)
                counts[neighbors[c]] -= 1
                counts[neighbors[c_prime]] += 1
                
                if np.any(counts < 1):
                    C_final = old_C
                    counts = old_counts
                    continue
                
                new_C, new_counts = reduce_code(C_final, counts)
                
                if len(new_C) < len(old_C):
                    C_final = new_C
                    counts = new_counts
                    if len(C_final) < best_size:
                        best_size = len(C_final)
                        best_C = C_final.copy()
                        best_counts = counts.copy()
                    improved = True
                    break
                else:
                    C_final = new_C
                    counts = new_counts
                    improved = True
                    break
            
            if not improved:
                for _ in range(5):
                    if time.time() - start_time > time_limit * 0.98:
                        break
                    if not C_final:
                        break
                    c_idx = random.randint(0, len(C_final)-1)
                    c = C_final[c_idx]
                    cand = neighbors[c]
                    c_prime = cand[random.randint(0, len(cand)-1)]
                    if c_prime in C_final:
                        continue
                    
                    old_C = C_final.copy()
                    old_counts = counts.copy()
                    
                    C_final.remove(c)
                    C_final.append(c_prime)
                    counts[neighbors[c]] -= 1
                    counts[neighbors[c_prime]] += 1
                    
                    if np.any(counts < 1):
                        C_final = old_C
                        counts = old_counts
                        continue
                    
                    new_C, new_counts = reduce_code(C_final, counts)
                    if len(new_C) < len(old_C):
                        C_final = new_C
                        counts = new_counts
                        if len(new_C) < best_size:
                            best_size = len(C_final)
                            best_C = C_final.copy()
                            best_counts = counts.copy()
                        break
                    else:
                        C_final = new_C
                        counts = new_counts
    
    # Final reduction
    best_C, _ = reduce_code(best_C, calc_counts(best_C))
    
    return [list(words[i]) for i in best_C]
