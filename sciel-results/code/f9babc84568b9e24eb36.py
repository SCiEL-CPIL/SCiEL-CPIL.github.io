import time
import math
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    best_score = -1
    best_set = [0]
    
    def compute_score(U):
        if not U or 0 not in U:
            return -1
        max_u = max(U)
        if max_u == 0:
            return 0
        
        U_arr = np.array(U, dtype=np.int64)
        n = len(U_arr)
        
        # Use FFT-based method for larger sets, direct for small
        if n > 20:
            n_sum, n_diff = _fft_count_sums_diffs(U_arr, max_u)
        else:
            # Direct computation for small sets
            sums = U_arr[:, None] + U_arr[None, :]
            diffs = U_arr[:, None] - U_arr[None, :]
            n_sum = len(np.unique(sums))
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
    
    def _fft_count_sums_diffs(U_arr, max_u):
        """Count unique sums and differences using FFT convolution."""
        size = 2 * max_u + 2
        fft_size = 1
        while fft_size < size:
            fft_size *= 2
        
        indicator = np.zeros(fft_size, dtype=np.float64)
        indicator[U_arr] = 1.0
        
        fft_ind = np.fft.fft(indicator)
        autocorr = np.fft.ifft(fft_ind * np.conj(fft_ind)).real
        n_sum = np.sum(autocorr[:2*max_u+1] > 0.5)
        
        indicator_rev = np.zeros(fft_size, dtype=np.float64)
        indicator_rev[max_u - U_arr] = 1.0
        fft_rev = np.fft.fft(indicator_rev)
        cross_corr = np.fft.ifft(fft_ind * np.conj(fft_rev)).real
        n_diff = np.sum(cross_corr[max_u:max_u*2+1] > 0.5)
        
        return int(n_sum), int(n_diff)
    
    def _spectral_cyclic_score(U, N):
        """
        Compute sumset/difference set sizes in Z_N using DFT.
        Returns (n_sum, n_diff, score) where score is computed using N as base.
        """
        U_mod = [x % N for x in U]
        U_mod = sorted(set(U_mod))
        
        if len(U_mod) < 2 or 0 not in U_mod:
            return None
        
        # Create indicator vector in Z_N
        indicator = np.zeros(N, dtype=np.complex128)
        for u in U_mod:
            indicator[u] = 1.0
        
        # DFT of indicator
        F = np.fft.fft(indicator)
        
        # Sumset: |U+U| in Z_N = number of positions where autocorrelation > 0
        # Autocorrelation via IDFT of |F|^2
        autocorr = np.fft.ifft(np.abs(F)**2).real
        
        # Difference set: |U-U| in Z_N
        # For differences, we need cross-correlation with reversed
        # In Z_N, U-U = {u1 - u2 mod N : u1,u2 in U}
        # This equals the support of the circular cross-correlation
        
        # Actually, in Z_N, |U-U| can be computed as:
        # The number of d in Z_N such that there exist u1,u2 in U with u1-u2 ≡ d (mod N)
        # This is the support size of the circular convolution of U with -U
        
        # Let's compute it properly:
        # U+U support: positions s where sum_{u in U} indicator[s-u] > 0
        # U-U support: positions d where sum_{u in U} indicator[d+u] > 0
        
        # Using FFT:
        # Sumset indicator = circular convolution of indicator with itself
        sumset_circ = np.fft.ifft(F * F).real
        n_sum = int(np.sum(sumset_circ > 0.5))
        
        # For difference set: convolve indicator with indicator of -U
        neg_indicator = np.zeros(N, dtype=np.complex128)
        for u in U_mod:
            neg_indicator[(-u) % N] = 1.0
        F_neg = np.fft.fft(neg_indicator)
        diffset_circ = np.fft.ifft(F * F_neg).real
        n_diff = int(np.sum(diffset_circ > 0.5))
        
        if n_sum == 0 or n_diff == 0:
            return None
        
        ratio = n_diff / n_sum
        if ratio <= 1:
            return (n_sum, n_diff, 1.0)
        
        # Score using N as the effective "base" for the cyclic model
        # The actual score uses 2*max(U)+1, but we can use this for comparison
        score = 1 + math.log(ratio) / math.log(N)
        
        return (n_sum, n_diff, score)
    
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
    
    # Phase 1: Try structured sets
    for base in [2, 3, 4, 5, 7, 10]:
        g = [0, 1]
        while g[-1] * base < 5000:
            g.append(g[-1] * base)
        try_set(g)
    
    # Golomb ruler-like sets
    def generate_golomb_like(n, max_val=10000):
        U = [0, 1]
        for _ in range(2, n):
            best_pos = -1
            best_score_local = -1
            for pos in range(2, max_val + 1):
                if pos in U:
                    continue
                new_diffs = set()
                for u in U:
                    new_diffs.add(pos - u)
                    new_diffs.add(u - pos)
                all_diffs = set()
                for i in range(len(U)):
                    for j in range(len(U)):
                        all_diffs.add(U[i] - U[j])
                new_unique = len(new_diffs - all_diffs)
                if new_unique > best_score_local:
                    best_score_local = new_unique
                    best_pos = pos
                if best_score_local >= len(U):
                    break
            if best_pos > 0:
                U.append(best_pos)
            else:
                break
        return U
    
    for n in [5, 6, 7, 8, 10, 12, 15]:
        g = generate_golomb_like(n, 5000)
        try_set(g)
    
    # Phase 2: Random sampling of medium-sized sets (replaces slow exhaustive search)
    for _ in range(200):
        if time.time() > deadline - 30:
            break
        n = random.randint(3, 25)
        max_val = random.randint(50, 500)
        cand = [0] + random.sample(range(1, max_val + 1), n - 1)
        try_set(cand)
    
    # Phase 3: Spectral cyclic search - find good structures in Z_N
    for N in [31, 41, 43, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109]:
        if time.time() > deadline - 40:
            break
        
        # Try random subsets of Z_N
        for _ in range(30):
            n = random.randint(3, min(15, N // 2))
            U_rand = [0] + random.sample(range(1, N), n - 1)
            result = _spectral_cyclic_score(U_rand, N)
            if result:
                n_sum, n_diff, spec_score = result
                # Convert back to integer set and compute actual score
                U_int = sorted(U_rand)
                try_set(U_int)
    
    # Phase 4: Local search with FFT acceleration for larger sets
    current = list(best_set)
    no_improve = 0
    max_no_improve = 500
    
    while time.time() < deadline - 10 and no_improve < max_no_improve:
        improved = False
        max_cur = max(current)
        n_cur = len(current)
        
        # 1. Try adding an element
        candidates_to_add = set()
        for i in range(1, min(max_cur + 30, 2000)):
            if i not in current:
                candidates_to_add.add(i)
        for i in range(1, n_cur - 1):
            gap_start = current[i] + 1
            gap_end = current[i + 1] - 1
            for j in range(gap_start, min(gap_end + 1, gap_start + 10)):
                if j not in current:
                    candidates_to_add.add(j)
        for _ in range(30):
            candidates_to_add.add(random.randint(max_cur + 1, max_cur + 200))
        
        for new_el in candidates_to_add:
            if new_el in current or new_el < 1:
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
                    if new_el < 1 or new_el in current or new_el == old_el:
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
        
        # 4. Try growing the set significantly when n is small
        if len(current) < 30 and not improved:
            for _ in range(5):
                new_els = random.sample(range(1, max_cur + 100), min(5, max_cur + 99))
                cand = sorted(set(current + new_els))
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
            if random.random() < 0.4:
                if len(current) > 2:
                    i = random.randint(1, len(current) - 1)
                    j = random.randint(1, len(current) - 1)
                    if i != j:
                        cand = list(current)
                        cand[i], cand[j] = cand[j], cand[i]
                        cand = sorted(set(cand))
                        if 0 in cand:
                            try_set(cand)
                            current = list(best_set)
                new_el = random.randint(1, max_cur + 100)
                if new_el not in current:
                    cand = sorted(current + [new_el])
                    try_set(cand)
                    current = list(best_set)
    
    return list(best_set)
