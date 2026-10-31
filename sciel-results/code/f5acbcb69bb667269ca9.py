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
        
        # FFT-based computation for both sumset and diffset
        max_u = max(U)
        
        # Sumset via FFT: convolve indicator with itself
        M_sum = 2 * max_u + 1
        M_sum_fft = 1
        while M_sum_fft < M_sum:
            M_sum_fft *= 2
        
        ind = np.zeros(M_sum_fft)
        for u in U:
            ind[u] = 1.0
        
        fft_ind = np.fft.fft(ind)
        conv = np.fft.ifft(fft_ind * fft_ind)
        conv = np.abs(conv)
        n_sum = int(np.sum(conv[:M_sum] > 0.5))
        
        # Diffset via autocorrelation
        M_diff = 2 * max_u + 1
        M_diff_fft = 1
        while M_diff_fft < M_diff:
            M_diff_fft *= 2
        
        ind_shifted = np.zeros(M_diff_fft)
        for u in U:
            ind_shifted[u + max_u] = 1.0
        
        fft_shifted = np.fft.fft(ind_shifted)
        auto = np.fft.ifft(fft_shifted * np.conj(fft_shifted))
        auto = np.abs(auto)
        
        n_diff = 0
        for d in range(-max_u, max_u + 1):
            if d >= 0:
                idx = d
            else:
                idx = M_diff_fft + d
            if auto[idx] > 0.5:
                n_diff += 1
        
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
    
    def compute_marginal_gain(current, new_el):
        """Compute marginal gain in score when adding new_el to current set."""
        cand = sorted(current + [new_el])
        if len(cand) != len(set(cand)):
            return -1
        s = compute_score(cand)
        cur_s = compute_score(current)
        return s - cur_s
    
    # Phase 1: Start with a known good structure and greedily expand
    # Use a base set that has good difference/sum ratio properties
    current = [0]
    
    # Add elements strategically: small gaps to maximize differences
    # Start with a pattern that spreads elements to maximize U-U size
    initial_elements = [0, 1, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191, 16383, 32767]
    for el in initial_elements:
        if time.time() > deadline - 30:
            break
        cand = sorted(current + [el])
        try_set(cand)
        current = list(best_set)
    
    # Phase 2: Greedy expansion with marginal gain calculation
    for step in range(300):
        if time.time() > deadline - 30:
            break
        
        max_cur = max(current)
        
        # Generate candidate elements to add
        candidates = set()
        # Nearby elements with various spacings
        for i in range(1, max_cur + 50):
            if i not in current:
                candidates.add(i)
        # Geometric spacing candidates
        for base in [2, 3, 5, 7, 11]:
            power = 1
            while power < max_cur + 200:
                if power not in current:
                    candidates.add(power)
                power *= base
        # Random candidates
        for _ in range(30):
            candidates.add(random.randint(max_cur + 1, max_cur + 300))
        
        # For each candidate, compute marginal gain
        best_cand = None
        best_gain = 0
        cur_score = compute_score(current)
        
        # Sort candidates by heuristic priority (smaller elements first for efficiency)
        sorted_cands = sorted(candidates)
        
        for new_el in sorted_cands:
            if time.time() > deadline - 30:
                break
            cand = sorted(current + [new_el])
            s = compute_score(cand)
            gain = s - cur_score
            if gain > best_gain:
                best_gain = gain
                best_cand = new_el
        
        if best_cand is not None:
            current = sorted(current + [best_cand])
            try_set(current)
        else:
            break
    
    # Phase 3: Local search refinement
    current = list(best_set)
    
    no_improve = 0
    max_no_improve = 200
    
    while time.time() < deadline - 5 and no_improve < max_no_improve:
        improved = False
        max_cur = max(current)
        
        # 1. Try adding an element
        candidates_to_add = set()
        for i in range(1, max_cur + 40):
            if i not in current:
                candidates_to_add.add(i)
        for _ in range(40):
            candidates_to_add.add(random.randint(max_cur + 1, max_cur + 200))
            
        for new_el in sorted(candidates_to_add):
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
                for delta in range(-30, 31):
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
            if random.random() < 0.5:
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
                    new_el = random.randint(1, max_cur + 100)
                    if new_el not in current:
                        cand = sorted(current + [new_el])
                        try_set(cand)
                        current = list(best_set)

    return list(best_set)
