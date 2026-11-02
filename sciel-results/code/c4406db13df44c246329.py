import time
import numpy as np
import random
from itertools import product

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    # Handle the specific case q=3, n=8, radius=1
    if q != 3 or n != 8 or radius != 1:
        # Fallback for other cases
        if q ** n > 100000:
            return []
        # Simple greedy for general case
        total = q ** n
        codewords = []
        uncovered = set(range(total))
        
        def idx_to_digits(idx):
            digits = [0] * n
            for i in range(n - 1, -1, -1):
                digits[i] = idx % q
                idx //= q
            return digits
        
        def digits_to_idx(digits):
            idx = 0
            for d in digits:
                idx = idx * q + d
            return idx
        
        # Precompute neighbors
        neighbors = {}
        for i in range(total):
            digits = idx_to_digits(i)
            nbs = [i]
            for pos in range(n):
                orig = digits[pos]
                for val in range(q):
                    if val == orig:
                        continue
                    new_digits = digits[:]
                    new_digits[pos] = val
                    nbs.append(digits_to_idx(new_digits))
            neighbors[i] = nbs
        
        while uncovered and time.time() < deadline:
            # Find codeword covering most uncovered
            best_cw = -1
            best_count = -1
            for cw in uncovered:
                count = sum(1 for nb in neighbors[cw] if nb in uncovered)
                if count > best_count:
                    best_count = count
                    best_cw = cw
            
            if best_cw == -1:
                break
            
            codewords.append(best_cw)
            for nb in neighbors[best_cw]:
                uncovered.discard(nb)
        
        result = [idx_to_digits(cw) for cw in codewords]
        seen = set()
        unique_result = []
        for cw in result:
            t = tuple(cw)
            if t not in seen:
                seen.add(t)
                unique_result.append(cw)
        return unique_result
    
    # Specific case: q=3, n=8, radius=1
    total_words = 3 ** 8  # 6561
    num_coords = 8
    alphabet = 3
    
    def idx_to_digits(idx):
        digits = [0] * num_coords
        for i in range(num_coords - 1, -1, -1):
            digits[i] = idx % alphabet
            idx //= alphabet
        return digits
    
    def digits_to_idx(digits):
        idx = 0
        for d in digits:
            idx = idx * alphabet + d
        return idx
    
    # Precompute neighbors
    neighbors = []
    for i in range(total_words):
        digits = idx_to_digits(i)
        nbs = [i]
        for pos in range(num_coords):
            orig_val = digits[pos]
            for val in range(alphabet):
                if val == orig_val:
                    continue
                new_digits = digits[:]
                new_digits[pos] = val
                nbs.append(digits_to_idx(new_digits))
        neighbors.append(nbs)
    
    # Build Coverage Matrix A
    # A[w, c] = 1 if w is in N(c)
    A = np.zeros((total_words, total_words), dtype=np.int8)
    for c in range(total_words):
        for w in neighbors[c]:
            A[w, c] = 1
    
    # Initialize code
    # Start with a greedy set cover to get a good baseline
    uncovered = np.ones(total_words, dtype=np.bool_)
    code_set = set()
    code_vec = np.zeros(total_words, dtype=np.int8)
    
    # Greedy Initialization
    # Precompute coverage counts for greedy
    # counts[w] = number of uncovered words covered by w
    # Actually, standard greedy: pick c that covers max uncovered.
    
    # We can use the matrix A for this too?
    # Uncovered is a vector u.
    # Count for c is sum(u * A[:, c]).
    
    u = np.ones(total_words, dtype=np.int8)
    
    # Initial greedy step
    while np.any(u) and time.time() < deadline - 10:
        # Find c that maximizes sum(u * A[:, c])
        # scores = A.T @ u
        scores = A.T @ u
        best_c = int(np.argmax(scores))
        
        if scores[best_c] == 0:
            break
            
        code_set.add(best_c)
        code_vec[best_c] = 1
        
        # Update u
        u[neighbors[best_c]] = 0
        
        # Check time
        if time.time() > deadline - 10:
            break

    # If still uncovered, add random codewords
    if np.any(u):
        uncovered_idx = np.where(u)[0]
        # Add random neighbors of uncovered
        for w in uncovered_idx:
            if time.time() > deadline - 5:
                break
            # Pick a random neighbor
            d = random.choice(neighbors[w])
            if d not in code_set:
                code_set.add(d)
                code_vec[d] = 1
                # Update u
                u[neighbors[d]] = 0

    # Local Search
    def get_cov():
        return A.dot(code_vec)

    def prune():
        nonlocal code_set, code_vec
        cov = get_cov()
        to_remove = []
        for c in list(code_set):
            # Check if removing c leaves any w with cov 0
            # Only need to check w in N(c)
            # If cov[w] > 1 for all w in N(c), then c is redundant
            # Because for w not in N(c), cov[w] is unchanged and >= 1 (since covered)
            # Wait, if w not in N(c), cov[w] could be 1?
            # If cov[w] == 1 and w not in N(c), removing c doesn't change cov[w].
            # So w remains covered.
            # We only risk losing coverage for w in N(c).
            # So yes, checking N(c) is sufficient.
            if np.all(cov[neighbors[c]] > 1):
                to_remove.append(c)
        
        for c in to_remove:
            code_set.remove(c)
            code_vec[c] = 0
        return len(to_remove) > 0

    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        cov = get_cov()
        
        # 1-swap
        c_list = list(code_set)
        random.shuffle(c_list)
        
        for c in c_list:
            if time.time() > deadline - 2:
                break
            
            # Critical words: w in N(c) with cov[w] == 1
            critical = [w for w in neighbors[c] if cov[w] == 1]
            
            if not critical:
                # Redundant, remove
                code_set.remove(c)
                code_vec[c] = 0
                improved = True
                cov = get_cov()
                continue
            
            # Find d that covers all critical
            candidates = set(neighbors[critical[0]])
            for w in critical[1:]:
                candidates &= set(neighbors[w])
            
            valid_d = None
            for d in candidates:
                if d in code_set:
                    continue
                if np.all(A[critical, d] == 1):
                    valid_d = d
                    break
            
            if valid_d is not None:
                code_set.remove(c)
                code_vec[c] = 0
                code_set.add(valid_d)
                code_vec[valid_d] = 1
                improved = True
                cov = get_cov()
        
        # 2-swaps
        if time.time() < deadline - 5:
            c_list = list(code_set)
            random.shuffle(c_list)
            n_cw = len(c_list)
            
            # Try a limited number of random pairs
            for _ in range(50):
                if time.time() > deadline - 5:
                    break
                if n_cw < 2:
                    break
                i = random.randint(0, n_cw - 1)
                j = random.randint(0, n_cw - 1)
                if i == j:
                    continue
                c1 = c_list[i]
                c2 = c_list[j]
                
                if c1 not in code_set or c2 not in code_set:
                    continue
                
                # Find critical words for (c1, c2)
                # w is critical if cov[w] == A[w,c1] + A[w,c2]
                # and w is in N(c1) U N(c2)
                
                # We can just check the union
                # To avoid duplicates, use a set
                critical_set = set()
                
                for w in neighbors[c1]:
                    if cov[w] == A[w, c1] + A[w, c2]:
                        critical_set.add(w)
                
                for w in neighbors[c2]:
                    if w not in critical_set:
                        if cov[w] == A[w, c1] + A[w, c2]:
                            critical_set.add(w)
                
                if not critical_set:
                    # Both redundant
                    code_set.remove(c1)
                    code_vec[c1] = 0
                    code_set.remove(c2)
                    code_vec[c2] = 0
                    improved = True
                    cov = get_cov()
                    break
                
                # Try to find d covering all critical_set
                # Only try if small
                if len(critical_set) <= 4:
                    critical_list = list(critical_set)
                    candidates = set(neighbors[critical_list[0]])
                    for w in critical_list[1:]:
                        candidates &= set(neighbors[w])
                    
                    for d in candidates:
                        if d in code_set:
                            continue
                        if np.all(A[critical_list, d] == 1):
                             code_set.remove(c1)
                             code_vec[c1] = 0
                             code_set.remove(c2)
                             code_vec[c2] = 0
                             code_set.add(d)
                             code_vec[d] = 1
                             improved = True
                             cov = get_cov()
                             break
                    if improved:
                        break

        # Prune
        prune()

    # Final conversion
    result = [idx_to_digits(c) for c in code_set]
    seen = set()
    unique_result = []
    for cw in result:
        t = tuple(cw)
        if t not in seen:
            seen.add(t)
            unique_result.append(cw)
    
    # Verify coverage
    if time.time() < deadline - 1:
        cov = get_cov()
        if not np.all(cov > 0):
             # Fallback
             uncovered_idx = np.where(cov == 0)[0]
             for w in uncovered_idx:
                 d = random.choice(neighbors[w])
                 if d not in code_set:
                     code_set.add(d)
                     code_vec[d] = 1
             
             result = [idx_to_digits(c) for c in code_set]
             seen = set()
             unique_result = []
             for cw in result:
                 t = tuple(cw)
                 if t not in seen:
                     seen.add(t)
                     unique_result.append(cw)

    return unique_result
