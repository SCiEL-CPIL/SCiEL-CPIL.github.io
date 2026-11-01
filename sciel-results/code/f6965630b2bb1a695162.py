import math
import time
import random
import numpy as np

def compute_score(u):
    """Compute the exact score for a given set U."""
    if not u:
        return 0.0
    u = sorted(set(u))
    if len(u) <= 1:
        return 1.0
    max_u = max(u)
    m = 2 * max_u + 1
    if m <= 1:
        return 1.0
    
    n = len(u)
    # For small sets, direct computation
    if n <= 25:
        sumset = set()
        diffset = set()
        for a in u:
            for b in u:
                sumset.add(a + b)
                diffset.add(a - b)
    else:
        # FFT-based computation for larger sets
        size = 2 * max_u + 1
        fft_size = 1
        while fft_size < size:
            fft_size *= 2
        
        f = np.zeros(fft_size, dtype=np.float64)
        for val in u:
            f[val] = 1.0
        
        # Sumset: f * f (convolution)
        fft_f = np.fft.rfft(f)
        conv = np.fft.irfft(fft_f * fft_f, n=fft_size)
        # Count nonzero entries in [0, 2*max_u]
        sumset_size = int(np.sum(conv[:2*max_u+1] > 0.5))
        
        # Diffset: f * reversed_f
        f_rev = np.zeros(fft_size, dtype=np.float64)
        for val in u:
            f_rev[fft_size - val] = 1.0
        
        fft_frev = np.fft.rfft(f_rev)
        conv_diff = np.fft.irfft(fft_f * fft_frev, n=fft_size)
        # Diffset values range from -max_u to max_u, stored at offsets
        # The convolution gives counts for differences
        # Difference d appears at index (d + max_u) in the appropriate range
        # Actually, let's just do it directly for diffset since it's symmetric
        diffset = set()
        for a in u:
            for b in u:
                diffset.add(a - b)
        diffset_size = len(diffset)
    
    if n <= 25:
        r = len(diffset) / len(sumset)
    else:
        r = diffset_size / sumset_size
    
    if r <= 0:
        return 1.0
    return 1 + math.log(r) / math.log(m)

def is_golomb(u):
    """Check if all pairwise differences are distinct."""
    u = sorted(u)
    diffs = set()
    n = len(u)
    for i in range(n):
        for j in range(i + 1, n):
            d = u[j] - u[i]
            if d in diffs:
                return False
            diffs.add(d)
    return True

def count_distinct_diffs(u):
    """Count number of distinct pairwise differences."""
    u = sorted(u)
    diffs = set()
    n = len(u)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(u[j] - u[i])
    return len(diffs)

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    # Known good Golomb rulers
    golomb_rulers = [
        [0, 1, 3],
        [0, 1, 4, 6],
        [0, 1, 4, 9, 11],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 12, 17, 25],
        [0, 1, 4, 10, 12, 17, 25, 34],
        [0, 1, 4, 10, 12, 17, 25, 34, 44],
        [0, 1, 4, 10, 12, 17, 25, 34, 44, 55],
    ]
    
    best_set = None
    best_score = 0.0
    
    # Evaluate initial candidates
    for u in golomb_rulers:
        if time.time() > deadline:
            break
        u = sorted(set(u))
        if 0 not in u:
            u = [0] + u
        score = compute_score(u)
        if score > best_score:
            best_score = score
            best_set = u
    
    if best_set is None:
        return [0, 1, 3]
    
    # Phase 1: Systematic Golomb ruler extension
    # Try to extend the best Golomb ruler by adding marks
    def try_extend_golomb(base_u, max_size, time_limit=30):
        """Greedily extend a Golomb ruler by adding one mark at a time."""
        start_time = time.time()
        current = list(base_u)
        current_score = compute_score(current)
        
        for target_size in range(len(current) + 1, max_size + 1):
            if time.time() - start_time > time_limit:
                break
            
            best_candidate = None
            best_candidate_score = current_score
            
            current_max = max(current)
            # Search range for new mark
            search_hi = current_max + target_size * target_size * 2
            
            for new_mark in range(current_max + 1, search_hi + 1):
                if time.time() - start_time > time_limit:
                    break
                
                candidate = sorted(current + [new_mark])
                if not is_golomb(candidate):
                    continue
                
                s = compute_score(candidate)
                if s > best_candidate_score:
                    best_candidate_score = s
                    best_candidate = candidate
            
            if best_candidate is not None:
                current = best_candidate
                current_score = best_candidate_score
            else:
                break
        
        return current, current_score
    
    # Try extending from several base rulers
    extension_results = []
    for base in golomb_rulers[-4:]:  # Last 4 known rulers
        if time.time() > deadline - 40:
            break
        ext_u, ext_score = try_extend_golomb(base, len(base) + 4, time_limit=15)
        if ext_u and len(ext_u) > len(base):
            extension_results.append((ext_score, ext_u))
    
    # Also try greedy construction from scratch
    def greedy_golomb(n, time_limit=20):
        start_time = time.time()
        u = [0]
        for i in range(1, n):
            if time.time() - start_time > time_limit:
                break
            pos = 0
            found = False
            while not found:
                pos += 1
                candidate = sorted(u + [pos])
                if is_golomb(candidate):
                    u = candidate
                    found = True
                if pos > n * n * 30:
                    break
        return u
    
    greedy_u = greedy_golomb(12, time_limit=20)
    if greedy_u and len(greedy_u) >= 6:
        g_score = compute_score(greedy_u)
        extension_results.append((g_score, greedy_u))
    
    # Update best from extensions
    for s, u in extension_results:
        if s > best_score:
            best_score = s
            best_set = u
    
    # Phase 2: Diff-guided local search (allowing non-Golomb sets for fine-tuning)
    current_set = list(best_set)
    current_score = best_score
    
    def get_set_sizes(u):
        """Return (|sumset|, |diffset|) for set u."""
        n = len(u)
        if n <= 20:
            sumset = set()
            diffset = set()
            for a in u:
                for b in u:
                    sumset.add(a + b)
                    diffset.add(a - b)
            return len(sumset), len(diffset)
        else:
            max_u = max(u)
            size = 2 * max_u + 1
            fft_size = 1
            while fft_size < size:
                fft_size *= 2
            f = np.zeros(fft_size, dtype=np.float64)
            for val in u:
                f[val] = 1.0
            fft_f = np.fft.rfft(f)
            conv = np.fft.irfft(fft_f * fft_f, n=fft_size)
            sumset_size = int(np.sum(conv[:2*max_u+1] > 0.5))
            
            diffset = set()
            for a in u:
                for b in u:
                    diffset.add(a - b)
            return sumset_size, len(diffset)
    
    cur_sum_size, cur_diff_size = get_set_sizes(current_set)
    
    for iteration in range(500):
        if time.time() > deadline:
            break
        
        choice = random.random()
        
        if choice < 0.45 and len(current_set) < 15:
            # Try adding a new element - prioritize positions that add new differences
            max_pos = max(current_set) * 2 + 20 if current_set else 20
            
            best_candidate = None
            best_candidate_score = current_score + 1e-10
            best_candidate_guided = -1e10
            
            for _ in range(10):
                if time.time() > deadline:
                    break
                new_val = random.randint(1, max_pos)
                if new_val in current_set:
                    continue
                
                new_set = sorted(current_set + [new_val])
                score = compute_score(new_set)
                
                new_sum_size, new_diff_size = get_set_sizes(new_set)
                diff_gain = new_diff_size - cur_diff_size
                sum_gain = new_sum_size - cur_sum_size
                
                # Strong preference for Golomb-like properties
                golomb_bonus = 0.0
                if is_golomb(new_set):
                    golomb_bonus = 0.01
                
                guided_score = score + 0.005 * (diff_gain - sum_gain) + golomb_bonus
                
                if guided_score > best_candidate_guided:
                    best_candidate_guided = guided_score
                    best_candidate_score = score
                    best_candidate = new_set
            
            if best_candidate is not None:
                current_set = best_candidate
                current_score = best_candidate_score
                cur_sum_size, cur_diff_size = get_set_sizes(current_set)
                if current_score > best_score:
                    best_score = current_score
                    best_set = list(current_set)
        
        elif choice < 0.75 and len(current_set) > 3:
            # Try removing an element
            removable = [x for x in current_set if x != 0]
            if removable:
                idx = random.choice(removable)
                new_set = [x for x in current_set if x != idx]
                score = compute_score(new_set)
                
                new_sum_size, new_diff_size = get_set_sizes(new_set)
                diff_loss = cur_diff_size - new_diff_size
                sum_loss = cur_sum_size - new_sum_size
                
                # Prefer removals that lose fewer differences than sums
                guided_score = score + 0.005 * (sum_loss - diff_loss)
                
                if guided_score > current_score + 1e-10:
                    current_set = new_set
                    current_score = score
                    cur_sum_size, cur_diff_size = get_set_sizes(current_set)
                    if score > best_score:
                        best_score = score
                        best_set = list(current_set)
        
        else:
            # Try modifying an element
            modifiable = [x for x in current_set if x != 0]
            if modifiable:
                idx = random.choice(modifiable)
                best_mod = None
                best_mod_score = current_score + 1e-10
                best_mod_guided = -1e10
                
                for delta in [-3, -2, -1, 1, 2, 3, 5, 7]:
                    if time.time() > deadline:
                        break
                    new_val = idx + delta
                    if new_val < 0 or new_val in current_set:
                        continue
                    
                    new_set = sorted([x if x != idx else new_val for x in current_set])
                    score = compute_score(new_set)
                    
                    new_sum_size, new_diff_size = get_set_sizes(new_set)
                    diff_gain = new_diff_size - cur_diff_size
                    sum_gain = new_sum_size - cur_sum_size
                    
                    golomb_bonus = 0.0
                    if is_golomb(new_set):
                        golomb_bonus = 0.01
                    
                    guided_score = score + 0.005 * (diff_gain - sum_gain) + golomb_bonus
                    
                    if guided_score > best_mod_guided:
                        best_mod_guided = guided_score
                        best_mod_score = score
                        best_mod = new_set
                
                if best_mod is not None:
                    current_set = best_mod
                    current_score = best_mod_score
                    cur_sum_size, cur_diff_size = get_set_sizes(current_set)
                    if current_score > best_score:
                        best_score = current_score
                        best_set = list(current_set)
    
    # Phase 3: Final high-precision polish
    if best_set and time.time() < deadline - 10:
        polish_set = list(best_set)
        polish_score = best_score
        
        for _ in range(300):
            if time.time() > deadline - 5:
                break
            
            modifiable = [x for x in polish_set if x != 0]
            if not modifiable:
                break
            
            improved = False
            idx = random.choice(modifiable)
            for delta in [-2, -1, 1, 2]:
                if time.time() > deadline - 5:
                    break
                new_val = idx + delta
                if new_val < 0 or new_val in polish_set:
                    continue
                
                new_set = sorted([x if x != idx else new_val for x in polish_set])
                score = compute_score(new_set)
                if score > polish_score + 1e-12:
                    polish_score = score
                    polish_set = new_set
                    improved = True
                    break
            
            if not improved:
                # Try adding/removing
                if len(polish_set) < 14 and random.random() < 0.5:
                    max_pos = max(polish_set) + 10
                    for _ in range(5):
                        new_val = random.randint(1, max_pos)
                        if new_val in polish_set:
                            continue
                        new_set = sorted(polish_set + [new_val])
                        score = compute_score(new_set)
                        if score > polish_score + 1e-12:
                            polish_score = score
                            polish_set = new_set
                            break
                elif len(polish_set) > 4 and random.random() < 0.3:
                    rem = random.choice([x for x in polish_set if x != 0])
                    new_set = [x for x in polish_set if x != rem]
                    score = compute_score(new_set)
                    if score > polish_score + 1e-12:
                        polish_score = score
                        polish_set = new_set
        
        if polish_score > best_score:
            best_score = polish_score
            best_set = polish_set
    
    return best_set
