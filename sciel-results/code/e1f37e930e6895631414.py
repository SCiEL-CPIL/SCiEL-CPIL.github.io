import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 112.0
    total_words = q ** n

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

    # Precompute neighbors as a numpy array of shape (total_words, cover_count)
    cover_count = 1 + n * (q - 1)
    neighbors_arr = np.zeros((total_words, cover_count), dtype=np.int32)
    
    # We also want the full boolean cover matrix for fast gain calculations
    # cover_matrix[i, j] = True if j is within distance radius of i
    cover_matrix = np.zeros((total_words, total_words), dtype=np.uint8)
    
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
            cover_matrix[i, idx] = 1

    # Convert cover_matrix to a view for faster dot products if possible, or just use as is
    # uint8 is good for small counts. Max count is cover_count which is < 255 for this problem.

    def greedy_construct(random_seed=None):
        if random_seed is not None:
            random.seed(random_seed)
        
        uncovered = np.ones(total_words, dtype=np.uint8)
        code = []
        
        while np.any(uncovered):
            # gains[i] = number of uncovered words covered by codeword i
            # This is the row sum of cover_matrix[i] where uncovered[j] is 1
            # i.e., dot product of row i and uncovered vector
            gains = cover_matrix.dot(uncovered)
            
            valid = gains > 0
            if not valid.any():
                break
            
            max_gain = np.max(gains[valid])
            best_candidates = np.where((gains == max_gain) & valid)[0]
            
            chosen = random.choice(best_candidates.tolist())
            code.append(chosen)
            uncovered[cover_matrix[chosen, :]] = 0
        
        return code

    def reduce_code(code):
        counts = np.zeros(total_words, dtype=np.int16)
        for c in code:
            counts[cover_matrix[c, :]] += 1
        
        changed = True
        while changed:
            changed = False
            new_code = []
            for c in code:
                counts[cover_matrix[c, :]] -= 1
                if np.any(counts == 0):
                    counts[cover_matrix[c, :]] += 1
                    new_code.append(c)
                else:
                    changed = True
            code = new_code
        return code

    def local_search(code, deadline, allow_same_size=False):
        code = list(code)
        counts = np.zeros(total_words, dtype=np.int16)
        for c in code:
            counts[cover_matrix[c, :]] += 1
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            indices = list(range(len(code)))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() >= deadline:
                    break
                
                c_remove = code[i]
                counts[cover_matrix[c_remove, :]] -= 1
                
                if np.any(counts == 0):
                    # Cannot remove, try to swap
                    counts[cover_matrix[c_remove, :]] += 1
                    
                    # Find critical words (count == 1) that are covered by c_remove
                    covered_by_c = cover_matrix[c_remove, :] == 1
                    critical_words_mask = (counts == 1) & covered_by_c
                    critical_words = np.where(critical_words_mask)[0]
                    
                    if len(critical_words) == 0:
                        continue
                    
                    # Find candidates that cover all critical words
                    # A candidate w is valid if for all cw in critical_words, cover_matrix[w, cw] == 1
                    # This is equivalent to: sum(cover_matrix[:, critical_words], axis=1) == len(critical_words)
                    
                    # Optimization: Instead of checking all candidates against all critical words,
                    # we can intersect the sets of codewords covering each critical word.
                    # But with numpy, the sum check is vectorized and likely fast enough for small critical sets.
                    
                    # Let's try the intersection approach with sets for small critical sets, 
                    # or the sum approach for larger ones.
                    # Given the sparsity, sum approach on the slice might be good.
                    
                    # candidates[w] = sum(cover_matrix[w, critical_words])
                    # We want candidates == len(critical_words)
                    
                    # This is a matrix-vector product again, but on a slice of columns.
                    # cover_matrix[:, critical_words] is (N, K)
                    # We want row sums.
                    # np.sum(cover_matrix[:, critical_words], axis=1)
                    
                    # This might be expensive if K is large. But K is usually small.
                    
                    # Alternative: Iterate through critical words and intersect candidate sets.
                    # Let's stick to the numpy sum for simplicity and vectorization.
                    
                    candidate_sums = np.sum(cover_matrix[:, critical_words], axis=1)
                    valid_candidates = np.where(candidate_sums == len(critical_words))[0]
                    
                    # Exclude c_remove itself (though it covers all its critical words, so it would be in the list)
                    valid_candidates = valid_candidates[valid_candidates != c_remove]
                    
                    if len(valid_candidates) == 0:
                        continue
                    
                    # Choose randomly from valid candidates
                    best_w = random.choice(valid_candidates.tolist())
                    
                    # Perform swap
                    counts[cover_matrix[c_remove, :]] -= 1
                    counts[cover_matrix[best_w, :]] += 1
                    code[i] = best_w
                    improved = True
                    break
                else:
                    # Can remove
                    code.pop(i)
                    improved = True
                    break
        
        return code

    def repair_greedy(code, deadline):
        code = list(code)
        counts = np.zeros(total_words, dtype=np.int16)
        for c in code:
            counts[cover_matrix[c, :]] += 1
        
        uncovered = (counts == 0).astype(np.uint8)
        
        while np.any(uncovered) and time.time() < deadline:
            gains = cover_matrix.dot(uncovered)
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            
            best_candidates = np.where(gains == max_gain)[0]
            chosen = random.choice(best_candidates.tolist())
            code.append(chosen)
            counts[cover_matrix[chosen, :]] += 1
            uncovered[cover_matrix[chosen, :]] = 0
            
        return code

    def perturb_repair(code, k, deadline):
        code = list(code)
        if len(code) <= k:
            return code
        
        remove_indices = random.sample(range(len(code)), k)
        for idx in sorted(remove_indices, reverse=True):
            code.pop(idx)
        
        code = repair_greedy(code, deadline)
        return code

    # Main search loop
    best_code = None
    best_size = float('inf')
    
    # Initial construction
    code = greedy_construct(0)
    code = reduce_code(code)
    code = local_search(code, deadline - 10, allow_same_size=True)
    code = reduce_code(code)
    
    best_code = code
    best_size = len(code)
    
    # Iterated Local Search
    seed = 1
    while time.time() < deadline - 5:
        k = random.randint(1, 3)
        perturbed_code = perturb_repair(best_code, k, deadline - 5)
        
        perturbed_code = local_search(perturbed_code, deadline - 5, allow_same_size=True)
        perturbed_code = reduce_code(perturbed_code)
        
        if len(perturbed_code) < best_size:
            best_size = len(perturbed_code)
            best_code = perturbed_code
        
        seed += 1
        
        # Occasional fresh restart
        if seed % 10 == 0 and time.time() < deadline - 20:
            fresh_code = greedy_construct(seed)
            fresh_code = reduce_code(fresh_code)
            fresh_code = local_search(fresh_code, deadline - 10, allow_same_size=True)
            fresh_code = reduce_code(fresh_code)
            if len(fresh_code) < best_size:
                best_size = len(fresh_code)
                best_code = fresh_code

    result = []
    for idx in best_code:
        w = idx_to_tuple(idx)
        result.append(list(w))
    
    return result
