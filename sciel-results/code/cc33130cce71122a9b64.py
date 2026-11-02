import time
import random
import math
import numpy as np

def _score_set(U):
    U = sorted(set(U))
    if 0 not in U:
        return -1.0
    mx = max(U)
    if mx == 0:
        return 1.0
    n = len(U)
    sset = set()
    for i in range(n):
        ui = U[i]
        for j in range(i, n):
            sset.add(ui + U[j])
    dset = set()
    for i in range(n):
        ui = U[i]
        for j in range(n):
            dset.add(ui - U[j])
    ratio = len(dset) / len(sset)
    if ratio <= 0:
        return -1.0
    return 1.0 + math.log(ratio) / math.log(2 * mx + 1)

def _score_set_fft(U):
    """Fast scoring using FFT for larger sets."""
    U = sorted(set(U))
    if 0 not in U:
        return -1.0
    mx = max(U)
    if mx == 0:
        return 1.0
    n = len(U)
    
    # Use FFT to compute sumset and diffset sizes
    # Sumset: U + U
    # Create indicator vector
    size = 2 * mx + 1
    vec = np.zeros(size, dtype=np.float64)
    for u in U:
        vec[u] = 1.0
    
    # FFT convolution for sumset
    fft_len = 1
    while fft_len < 2 * size:
        fft_len *= 2
    fft_vec = np.fft.fft(vec, fft_len)
    conv = np.fft.ifft(fft_vec * fft_vec).real
    # Sumset: elements i where conv[i] > 0.5 (at least one pair sums to i)
    # For i from 0 to 2*mx
    sum_count = np.sum(conv[:2*mx+1] > 0.5)
    
    # Diffset: U - U, range from -mx to mx
    # Use cross-correlation: shift U by mx, then correlate
    vec_shifted = np.zeros(2 * mx + 1, dtype=np.float64)
    for u in U:
        vec_shifted[u + mx] = 1.0
    
    fft_len2 = 1
    while fft_len2 < 4 * mx + 2:
        fft_len2 *= 2
    fft_vec2 = np.fft.fft(vec_shifted, fft_len2)
    # Cross-correlation: ifft(fft * conj(fft))
    corr = np.fft.ifft(fft_vec2 * np.conj(fft_vec2)).real
    # The diffset is centered at mx in the shifted space
    # Differences range from -mx to mx, which maps to indices 0 to 2*mx in shifted
    diff_count = np.sum(corr[:2*mx+1] > 0.5)
    
    if sum_count == 0:
        return -1.0
    ratio = diff_count / sum_count
    if ratio <= 0:
        return -1.0
    return 1.0 + math.log(ratio) / math.log(2 * mx + 1)

def construct():
    start = time.time()
    deadline = start + 110.0
    
    best_set = [0, 1, 3]
    best_score = _score_set(best_set)
    
    rng = random.Random(42)
    
    def local_search(U, max_iter=2000):
        U = list(U)
        cur_score = _score_set(U)
        improved = True
        it = 0
        while improved and it < max_iter and time.time() < deadline:
            improved = False
            it += 1
            n = len(U)
            mx = max(U)
            
            # Move 1: Change one element with wider search
            if not improved:
                for i in range(1, n):
                    orig = U[i]
                    # Wider search range
                    search_lo = max(1, orig - 20)
                    search_hi = min(mx + 30, orig + 40)
                    for new_val in range(search_lo, search_hi + 1):
                        if new_val == orig:
                            continue
                        trial = U[:]
                        trial[i] = new_val
                        if len(set(trial)) != len(trial):
                            continue
                        s = _score_set(trial)
                        if s > cur_score + 1e-12:
                            cur_score = s
                            U = trial
                            improved = True
                            break
                    if improved:
                        break
            
            # Move 2: Add element
            if not improved and n < 12:
                # Try adding numbers in a broader range
                for new_val in range(1, mx + 15):
                    if new_val in U:
                        continue
                    trial = U + [new_val]
                    s = _score_set(trial)
                    if s > cur_score + 1e-12:
                        cur_score = s
                        U = trial
                        improved = True
                        break
            
            # Move 3: Remove element
            if not improved and n > 3:
                for i in range(1, n):
                    trial = U[:i] + U[i+1:]
                    s = _score_set(trial)
                    if s > cur_score + 1e-12:
                        cur_score = s
                        U = trial
                        improved = True
                        break
                        
            # Move 4: Swap two elements
            if not improved and n >= 4:
                for i in range(1, n):
                    for j in range(i+1, n):
                        U[i], U[j] = U[j], U[i]
                        if len(set(U)) != len(U):
                            U[i], U[j] = U[j], U[i]
                            continue
                        s = _score_set(U)
                        if s > cur_score + 1e-12:
                            cur_score = s
                            improved = True
                            break
                        else:
                            U[i], U[j] = U[j], U[i]
                    if improved:
                        break
            
            # Move 5: Replace two elements simultaneously
            if not improved and n >= 4 and n <= 10:
                for i in range(1, n):
                    for j in range(i+1, n):
                        if improved:
                            break
                        orig_i, orig_j = U[i], U[j]
                        for new_i in range(max(1, orig_i-15), orig_i+16):
                            if new_i == orig_i:
                                continue
                            for new_j in range(max(1, orig_j-15), orig_j+16):
                                if new_j == orig_j:
                                    continue
                                trial = U[:]
                                trial[i] = new_i
                                trial[j] = new_j
                                if len(set(trial)) != len(trial):
                                    continue
                                s = _score_set(trial)
                                if s > cur_score + 1e-12:
                                    cur_score = s
                                    U = trial
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break
            
        return U, cur_score

    # Initial candidates - more diverse
    candidates = [
        [0, 1, 3],
        [0, 1, 3, 4],
        [0, 1, 3, 7],
        [0, 1, 3, 7, 12],
        [0, 1, 4, 6],
        [0, 1, 4, 10, 12],
        [0, 1, 5, 7, 12],
        [0, 2, 3, 7],
        [0, 1, 3, 8, 12],
        [0, 1, 4, 6, 10],
        [0, 2, 5, 7, 12],
    ]
    
    # Generate Mian-Chowla style (greedy for best score)
    def greedy_construct(n, max_val=30):
        U = [0, 1]
        while len(U) < n:
            best_c = None
            best_s = -1
            for c in range(max(U) + 1, min(max(U) + 25, max_val + 1)):
                trial = U + [c]
                s = _score_set(trial)
                if s > best_s:
                    best_s = s
                    best_c = c
            if best_c is None:
                break
            U.append(best_c)
        return U

    for n in range(3, 11):
        candidates.append(greedy_construct(n))

    # Random starts with varied sizes
    for _ in range(30):
        n = rng.randint(3, 10)
        max_v = rng.randint(15, 40)
        vals = rng.sample(range(1, max_v), min(n-1, max_v-1))
        candidates.append([0] + sorted(vals))
    
    # Staircase/superincreasing inspired candidates
    candidates.append([0, 1, 3, 7, 15, 31])
    candidates.append([0, 1, 3, 7, 15])
    candidates.append([0, 1, 3, 8, 16])
    candidates.append([0, 1, 4, 9, 16])
    candidates.append([0, 1, 3, 6, 10, 15])
    candidates.append([0, 1, 4, 8, 12, 16])

    for c in candidates:
        s = _score_set(c)
        if s > best_score:
            best_score = s
            best_set = list(c)

    # Run local search on best and some others
    search_count = 0
    while time.time() < deadline - 5:
        search_count += 1
        # Pick a random candidate or best with increasing diversity
        if rng.random() < 0.4:
            start_set = list(best_set)
        else:
            start_set = list(rng.choice(candidates))
        
        # Occasionally perturb the best set for diversity
        if rng.random() < 0.2 and len(best_set) > 3:
            start_set = list(best_set)
            idx = rng.randint(1, len(start_set) - 1)
            start_set[idx] = max(1, start_set[idx] + rng.randint(-5, 5))
            start_set = sorted(set(start_set))
        
        refined, score = local_search(start_set)
        if score > best_score:
            best_score = score
            best_set = list(refined)
            # Add the improved set to candidates for future exploration
            candidates.append(list(best_set))
    
    return best_set
