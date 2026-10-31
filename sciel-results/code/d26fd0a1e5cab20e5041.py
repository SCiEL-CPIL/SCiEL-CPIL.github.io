"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random
from itertools import combinations
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill
    
    best_score = 0
    best_set = [0, 1, 3]  # seed fallback
    
    def compute_score(U):
        U = sorted(set(U))
        if not U or 0 not in U:
            return -1
        max_u = max(U)
        if max_u == 0:
            return 0
        n = len(U)
        
        # For small n, use set-based approach (faster for n < 15)
        if n < 15:
            sumset = set()
            diffset = set()
            for i in range(n):
                a = U[i]
                for j in range(n):
                    b = U[j]
                    sumset.add(a + b)
                    diffset.add(a - b)
            n_sum = len(sumset)
            n_diff = len(diffset)
        else:
            # FFT-based computation for larger n
            # Sumset: convolve indicator of U with itself
            # Difference set: convolve indicator of U with reversed U
            max_sum = 2 * max_u
            max_diff = max_u
            
            # For sumset
            len_sum = 2 * max_u + 1
            fft_size = 1
            while fft_size < 2 * len_sum:
                fft_size *= 2
            
            f_sum = np.zeros(fft_size, dtype=np.float64)
            for u in U:
                f_sum[u] = 1.0
            
            F_sum = np.fft.fft(f_sum)
            conv_sum = np.fft.ifft(F_sum * F_sum).real
            n_sum = int(np.sum(conv_sum > 0.5))
            
            # For difference set: U - U = {a - b : a, b in U}
            # This ranges from -max_u to max_u, so shift by max_u
            # diff[i] = 1 if there exist a,b in U with a-b = i
            # Equivalent: convolve U with reversed U
            # Let V = {max_u - u : u in U}, then (U+V)[i] counts pairs with u+v=i
            # u + (max_u - v) = max_u + (u-v), so u-v = i iff u + (max_u-v) = max_u + i
            len_diff = 2 * max_u + 1
            fft_size_d = 1
            while fft_size_d < 2 * len_diff:
                fft_size_d *= 2
            
            f_diff = np.zeros(fft_size_d, dtype=np.float64)
            for u in U:
                f_diff[u] = 1.0
            
            # Reverse: f_rev[i] = f_diff[max_u - i] for i in [0, max_u]
            f_rev = np.zeros(fft_size_d, dtype=np.float64)
            for u in U:
                f_rev[max_u - u] = 1.0
            
            F_diff = np.fft.fft(f_diff)
            F_rev = np.fft.fft(f_rev)
            conv_diff = np.fft.ifft(F_diff * F_rev).real
            n_diff = int(np.sum(conv_diff > 0.5))
        
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
    
    def is_golomb_ruler(U):
        """Check if all pairwise differences are distinct (up to sign)."""
        U = sorted(U)
        n = len(U)
        diffs = set()
        for i in range(n):
            for j in range(i + 1, n):
                d = U[j] - U[i]
                if d in diffs:
                    return False
                diffs.add(d)
        return True
    
    # Pre-compute seed
    best_score = compute_score([0, 1, 3])
    
    # Phase 1: Try known good structured sets (Golomb rulers, powers, etc.)
    # Known optimal Golomb rulers
    golomb_rulers = [
        [0, 1, 3],           # size 3, max 3
        [0, 1, 4, 6],        # size 4, max 6
        [0, 1, 4, 10, 12],   # size 5, max 12
        [0, 1, 4, 10, 12, 17], # size 6, max 17
        [0, 1, 4, 10, 18, 23, 25], # size 7, max 25
        [0, 1, 4, 10, 18, 23, 25, 34], # size 8, max 34
        [0, 1, 4, 10, 18, 23, 25, 34, 42], # size 9, max 42
        [0, 1, 4, 10, 18, 23, 25, 34, 42, 52], # size 10, max 52
    ]
    
    for gr in golomb_rulers:
        s = compute_score(gr)
        if s > best_score:
            best_score = s
            best_set = list(gr)
    
    # Also try scaled/shifted variants of Golomb rulers
    for gr in golomb_rulers[:6]:
        for scale in [2, 3, 4, 5]:
            scaled = [x * scale for x in gr]
            s = compute_score(scaled)
            if s > best_score:
                best_score = s
                best_set = list(scaled)
    
    # Phase 2: Systematic search over small sets
    # For each size and max, try structured candidates
    for max_u in range(1, 50):
        if time.time() > deadline - 5:
            break
        
        for size in range(3, min(12, max_u + 1)):
            if time.time() > deadline - 5:
                break
            
            candidates = []
            
            # Golomb ruler candidates (try to construct)
            # Simple heuristic: greedy construction
            if size >= 3:
                # Try different starting points and growth patterns
                for start2 in [1, 2, 3, 4]:
                    if start2 >= max_u:
                        continue
                    gr_candidate = [0, start2]
                    used_diffs = {start2}
                    for _ in range(size - 2):
                        # Try to add next element that maintains Golomb property
                        best_next = -1
                        for candidate in range(gr_candidate[-1] + 1, max_u + 1):
                            valid = True
                            for existing in gr_candidate:
                                d = candidate - existing
                                if d in used_diffs:
                                    valid = False
                                    break
                            if valid:
                                best_next = candidate
                                break
                        if best_next == -1:
                            break
                        gr_candidate.append(best_next)
                        for existing in gr_candidate[:-1]:
                            used_diffs.add(gr_candidate[-1] - existing)
                    if len(gr_candidate) == size:
                        candidates.append(gr_candidate)
            
            # Powers of 2
            if max_u >= 2 ** (size - 1):
                geo = [2 ** i for i in range(size)]
                if geo[-1] <= max_u:
                    candidates.append(geo)
            
            # Arithmetic progression
            if (max_u) % (size - 1) == 0:
                step = max_u // (size - 1)
                if step >= 1:
                    ap = [i * step for i in range(size)]
                    candidates.append(ap)
            
            # Fibonacci-like
            fib = [0, 1]
            while len(fib) < size:
                fib.append(fib[-1] + fib[-2])
            if fib[-1] <= max_u:
                candidates.append(fib[:size])
            
            # Evaluate unique candidates
            seen = set()
            for cand in candidates:
                cand = sorted(set(cand))
                if len(cand) != size or 0 not in cand:
                    continue
                key = tuple(cand)
                if key in seen:
                    continue
                seen.add(key)
                s = compute_score(cand)
                if s > best_score:
                    best_score = s
                    best_set = list(cand)
    
    # Phase 3: Local search around best set found
    if time.time() < deadline - 10:
        current = list(best_set)
        current_score = best_score
        no_improve_count = 0
        max_no_improve = 1000
        
        while no_improve_count < max_no_improve and time.time() < deadline - 5:
            improved = False
            
            # Try adding an element
            if len(current) < 40:
                max_cur = max(current)
                # Try elements near existing ones and some spread
                candidates_to_add = set()
                for el in current:
                    for delta in range(-3, 4):
                        new_el = el + delta
                        if new_el > 0 and new_el not in current:
                            candidates_to_add.add(new_el)
                # Also try some larger elements - geometric growth
                for new_el in range(max_cur + 1, max_cur + 20):
                    candidates_to_add.add(new_el)
                # Try geometric-like extensions
                if max_cur > 0:
                    for mult in [2, 3]:
                        new_el = max_cur * mult
                        if new_el not in current and new_el < 10000:
                            candidates_to_add.add(new_el)
                
                for new_el in sorted(candidates_to_add):
                    cand = sorted(current + [new_el])
                    s = compute_score(cand)
                    if s > current_score + 1e-15:
                        current = list(cand)
                        current_score = s
                        if s > best_score:
                            best_score = s
                            best_set = list(cand)
                        improved = True
                        break
            
            # Try replacing an element
            if len(current) >= 3:
                max_cur = max(current)
                for i in range(len(current)):
                    if current[i] == 0:
                        continue
                    old_el = current[i]
                    # Try nearby replacements with wider range
                    for new_el in range(max(1, old_el - 5), min(max_cur + 10, old_el + 6)):
                        if new_el == old_el or new_el in current:
                            continue
                        cand = list(current)
                        cand[i] = new_el
                        cand = sorted(set(cand))
                        if 0 not in cand or len(cand) != len(current):
                            continue
                        s = compute_score(cand)
                        if s > current_score + 1e-15:
                            current = list(cand)
                            current_score = s
                            if s > best_score:
                                best_score = s
                                best_set = list(cand)
                            improved = True
                            break
                    if improved:
                        break
            
            # Try removing an element (if set is large)
            if len(current) > 5 and not improved:
                for i in range(len(current) - 1, 0, -1):  # don't remove 0
                    cand = current[:i] + current[i+1:]
                    s = compute_score(cand)
                    if s > current_score + 1e-15:
                        current = list(cand)
                        current_score = s
                        if s > best_score:
                            best_score = s
                            best_set = list(cand)
                        improved = True
                        break
            
            if not improved:
                no_improve_count += 1
            else:
                no_improve_count = 0
    
    # Phase 4: Random restarts with various structured initializations
    if time.time() < deadline - 5:
        random.seed(12345)
        for _ in range(50):
            if time.time() > deadline - 5:
                break
            
            # Try different structured initializations
            init_type = random.randint(0, 3)
            
            if init_type == 0:
                # Golomb-ruler-like
                size = random.randint(4, 12)
                max_u = random.randint(10, 100)
                gr = [0, 1]
                used_diffs = {1}
                for _ in range(size - 2):
                    best_next = -1
                    for candidate in range(2, max_u + 1):
                        valid = True
                        for existing in gr:
                            d = candidate - existing
                            if d in used_diffs:
                                valid = False
                                break
                        if valid:
                            best_next = candidate
                            break
                    if best_next == -1:
                        break
                    gr.append(best_next)
                    for existing in gr[:-1]:
                        used_diffs.add(gr[-1] - existing)
                cand = gr
            elif init_type == 1:
                # Geometric-like
                size = random.randint(4, 15)
                ratio = random.uniform(1.5, 3.0)
                cand = [0]
                val = 1
                for _ in range(size - 1):
                    cand.append(int(val))
                    val *= ratio
            elif init_type == 2:
                # Sparse random
                size = random.randint(5, 25)
                max_u = random.randint(20, 200)
                cand = [0] + sorted(random.sample(range(1, max_u + 1), min(size - 1, max_u)))
            else:
                # Powers of small base
                size = random.randint(4, 12)
                base = random.choice([2, 3, 4, 5])
                cand = [base ** i for i in range(size)]
            
            if len(cand) >= 3 and 0 in cand:
                cand = sorted(set(cand))
                s = compute_score(cand)
                if s > best_score:
                    best_score = s
                    best_set = list(cand)
                elif s > best_score - 0.01:
                    # Do a quick local search from this candidate
                    current = list(cand)
                    current_score = s
                    for _ in range(20):
                        if time.time() > deadline - 5:
                            break
                        max_cur = max(current)
                        candidates_to_add = set()
                        for el in current:
                            for delta in range(-2, 3):
                                new_el = el + delta
                                if new_el > 0 and new_el not in current:
                                    candidates_to_add.add(new_el)
                        for new_el in range(max_cur + 1, max_cur + 5):
                            candidates_to_add.add(new_el)
                        
                        improved_inner = False
                        for new_el in sorted(candidates_to_add):
                            cand2 = sorted(current + [new_el])
                            s2 = compute_score(cand2)
                            if s2 > current_score + 1e-15:
                                current = list(cand2)
                                current_score = s2
                                if s2 > best_score:
                                    best_score = s2
                                    best_set = list(cand2)
                                improved_inner = True
                                break
                        if not improved_inner:
                            break
    
    return list(best_set)
# EVOLVE-BLOCK-END
