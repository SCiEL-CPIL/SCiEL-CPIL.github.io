import time
import math
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s limit
    
    best_score = -1
    best_set = [0]
    
    def compute_score(U):
        if not U or 0 not in U:
            return -1
        max_u = max(U)
        if max_u == 0:
            return 0
        
        n = len(U)
        if n < 5:
            # Small sets: use direct computation
            sumset = set()
            diffset = set()
            for a in U:
                for b in U:
                    sumset.add(a + b)
                    diffset.add(a - b)
            n_sum = len(sumset)
            n_diff = len(diffset)
        else:
            # FFT-based computation for larger sets
            max_u = max(U)
            # For sumset: need range [0, 2*max_u]
            # For diffset: need range [-max_u, max_u]
            
            # Sumset via FFT: convolve indicator with itself
            M_sum = 2 * max_u + 1
            # Round up to power of 2 for FFT efficiency
            M_sum_fft = 1
            while M_sum_fft < M_sum:
                M_sum_fft *= 2
            
            ind = np.zeros(M_sum_fft)
            for u in U:
                ind[u] = 1.0
            
            fft_ind = np.fft.fft(ind)
            conv = np.fft.ifft(fft_ind * fft_ind)
            conv = np.abs(conv)
            # Sumset size: count positions where conv > 0.5 (threshold for integer counts)
            n_sum = int(np.sum(conv[:M_sum] > 0.5))
            
            # Diffset: U - U = {a - b : a, b in U}
            # This is equivalent to the support of the "cross-correlation" or we can compute
            # by noting that diffset = {d : d = a - b}
            # We can compute this as: for each d in [-max_u, max_u], check if exists a,b in U with a-b=d
            # Alternative: diffset size = number of distinct differences
            # Use FFT: the diffset is related to the autocorrelation
            # Actually, |U-U| can be computed by: for each possible difference d, check if it occurs
            # Better: use the fact that |U-U| = number of d such that exists a in U with a+d in U
            # Let's use a direct but vectorized approach for medium sets
            U_arr = np.array(U)
            # Compute all differences
            diffs = U_arr[:, None] - U_arr[None, :]
            n_diff = len(np.unique(diffs))
        
        if n_sum == 0:
            return -1
        
        base = 2 * max_u + 1
        if base <= 1:
            return 0
        
        ratio = n_diff / n_sum
        if ratio <= 1:
            return 1.0
        score = 1 + math.log(ratio) / math.log(base)
        return score

    def try_set(cand):
        nonlocal best_score, best_set
        cand = sorted(set(cand))
        if not cand or 0 not in cand:
            return
        if max(cand) == 0:
            return
        s = compute_score(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)

    # Phase 1: Exhaustive search for small M
    for M in range(1, 14):
        if time.time() > deadline - 10:
            break
        for mask in range(1, 1 << M):
            if time.time() > deadline - 10:
                break
            cand = [0]
            for i in range(M):
                if mask & (1 << i):
                    cand.append(i + 1)
            try_set(cand)
            
    # Phase 2: Heuristic Search
    current = list(best_set)
    
    no_improve = 0
    max_no_improve = 1000
    
    while time.time() < deadline - 5 and no_improve < max_no_improve:
        improved = False
        max_cur = max(current)
        
        # 1. Try adding an element
        candidates_to_add = set()
        for i in range(1, max_cur + 30):
            if i not in current:
                candidates_to_add.add(i)
        for _ in range(30):
            candidates_to_add.add(random.randint(max_cur + 1, max_cur + 150))
            
        for new_el in candidates_to_add:
            if new_el in current:
                continue
            cand = sorted(current + [new_el])
            s = compute_score(cand)
            if s > best_score:
                best_score = s
                best_set = list(cand)
                current = list(cand)
                improved = True
                no_improve = 0
                break
        
        # 2. Try replacing an element
        if len(current) >= 3:
            for i in range(len(current)):
                if current[i] == 0:
                    continue
                old_el = current[i]
                for delta in range(-20, 21):
                    new_el = old_el + delta
                    if new_el < 1 or new_el in current:
                        continue
                    if new_el == old_el:
                        continue
                    cand = list(current)
                    cand[i] = new_el
                    cand = sorted(set(cand))
                    if 0 not in cand:
                        continue
                    s = compute_score(cand)
                    if s > best_score:
                        best_score = s
                        best_set = list(cand)
                        current = list(cand)
                        improved = True
                        no_improve = 0
                        break
                if improved:
                    break
        
        # 3. Try removing an element
        if len(current) > 3:
            for i in range(len(current)):
                if current[i] == 0:
                    continue
                cand = [x for j, x in enumerate(current) if j != i]
                s = compute_score(cand)
                if s > best_score:
                    best_score = s
                    best_set = list(cand)
                    current = list(cand)
                    improved = True
                    no_improve = 0
                    break
        
        if not improved:
            no_improve += 1
            # Random perturbation
            if random.random() < 0.3:
                if len(current) > 2:
                    i = random.randint(1, len(current)-1)
                    j = random.randint(1, len(current)-1)
                    if i != j:
                        cand = list(current)
                        cand[i], cand[j] = cand[j], cand[i]
                        cand = sorted(set(cand))
                        if 0 in cand:
                            try_set(cand)
                            current = list(best_set)
                else:
                    new_el = random.randint(1, max_cur + 50)
                    if new_el not in current:
                        cand = sorted(current + [new_el])
                        try_set(cand)
                        current = list(best_set)

    return list(best_set)
