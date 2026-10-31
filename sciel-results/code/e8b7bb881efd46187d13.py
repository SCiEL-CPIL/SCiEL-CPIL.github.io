"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random
from itertools import combinations
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s limit
    
    def compute_score_exact(U):
        """Compute score using FFT for exact sumset/diffset sizes."""
        U = sorted(set(U))
        if not U or 0 not in U:
            return -1
        max_u = max(U)
        if max_u == 0:
            return 0
        
        n = len(U)
        # For small sets, use direct computation
        if n <= 30:
            sumset = set()
            diffset = set()
            for a in U:
                for b in U:
                    sumset.add(a + b)
                    diffset.add(a - b)
            n_sum = len(sumset)
            n_diff = len(diffset)
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
        
        # For larger sets, use FFT
        max_val = max(U)
        # Sumset: values range from 0 to 2*max_val
        # Diffset: values range from -max_val to max_val
        
        # Create indicator array
        size = 2 * max_val + 1
        arr = np.zeros(size, dtype=np.float64)
        for x in U:
            arr[x] = 1.0
        
        # Sumset via FFT
        # We need convolution of arr with itself
        # Result will be at indices 0 to 2*max_val
        n_fft = 1
        while n_fft < 2 * size:
            n_fft *= 2
        
        fft_arr = np.fft.fft(arr, n=n_fft)
        conv = np.fft.ifft(fft_arr * fft_arr)
        conv = np.real(conv)
        
        # Sumset size: count non-zero entries in conv[0 : size]
        n_sum = np.sum(np.abs(conv[:size]) > 1e-6)
        
        # Diffset: a - b where a, b in U
        # This is convolution of arr with reversed arr
        # Or equivalently, count distinct values of a-b
        # a-b ranges from -max_val to max_val
        # We can compute this by: for each possible diff d, count if exists a,b with a-b=d
        # This is equivalent to checking if conv_diff[d] > 0 where conv_diff is cross-correlation
        
        # Cross-correlation: sum_x arr[x] * arr[x-d] for each d
        # This can be done via FFT: fft(arr) * conj(fft(arr_reversed))
        # Or more simply: the difference set size equals the number of distinct differences
        
        # Let's compute directly for moderate sizes
        if max_val <= 10000:
            diffset = set()
            for a in U:
                for b in U:
                    diffset.add(a - b)
            n_diff = len(diffset)
        else:
            # Use FFT for large cases
            # Difference set: d = a - b
            # Count distinct d values
            # This is the support of the cross-correlation
            arr_rev = arr[::-1]
            fft_rev = np.fft.fft(arr_rev, n=n_fft)
            cross = np.fft.ifft(fft_arr * fft_rev)
            cross = np.real(cross)
            # The cross-correlation at lag d gives count of pairs (a,b) with a-b = d - max_val
            # We need to find the right offset
            # Actually, let's just do it directly for safety
            diffset = set()
            for a in U:
                for b in U:
                    diffset.add(a - b)
            n_diff = len(diffset)
        
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
    
    best_score = 0
    best_set = [0, 1, 3]
    best_score = compute_score_exact([0, 1, 3])
    
    def try_set(cand):
        nonlocal best_score, best_set
        cand = sorted(set(cand))
        if not cand or 0 not in cand:
            return
        if max(cand) == 0:
            return
        s = compute_score_exact(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)
    
    # Phase 1: Exhaustive search for small sets
    max_u_limit_small = 15
    for max_u in range(1, max_u_limit_small + 1):
        if time.time() > deadline - 5:
            break
        elements = list(range(1, max_u + 1))
        for size in range(3, 7):
            if len(elements) < size:
                continue
            for comb in combinations(elements, size):
                cand = sorted([0] + list(comb))
                try_set(cand)
    
    # Phase 2: Continuous optimization using gradient-based approach
    # We'll use a soft surrogate and optimize over real positions, then round
    
    def soft_score_continuous(positions):
        """
        Continuous surrogate for the score.
        positions: array of real values (including 0 at index 0)
        Uses soft counting via Gaussian kernels to approximate set cardinalities.
        """
        if len(positions) < 2:
            return 0.0
        
        max_u = max(positions)
        if max_u <= 0:
            return 0.0
        
        # Soft sumset: use histogram with Gaussian smoothing
        # For each pair (i,j), the sum is positions[i] + positions[j]
        # We approximate the cardinality by counting distinct sums
        
        n = len(positions)
        
        # Compute all sums and differences
        sums = []
        diffs = []
        for i in range(n):
            for j in range(n):
                sums.append(positions[i] + positions[j])
                diffs.append(positions[i] - positions[j])
        
        sums = np.array(sums)
        diffs = np.array(diffs)
        
        # Soft cardinality: use clustering via binning
        # Scale to a reasonable range
        sum_range = sums.max() - sums.min() if len(sums) > 0 else 1
        diff_range = diffs.max() - diffs.min() if len(diffs) > 0 else 1
        
        if sum_range < 1e-9:
            sum_range = 1.0
        if diff_range < 1e-9:
            diff_range = 1.0
        
        # Number of bins to use - proportional to range
        n_bins_sum = int(min(10000, max(100, sum_range * 2)))
        n_bins_diff = int(min(10000, max(100, diff_range * 2)))
        
        # Histogram-based soft count
        _, _ = np.histogram(sums, bins=n_bins_sum, range=(sums.min(), sums.max()))
        _, _ = np.histogram(diffs, bins=n_bins_diff, range=(diffs.min(), diffs.max()))
        
        # Count non-empty bins
        sum_hist, _ = np.histogram(sums, bins=n_bins_sum, range=(sums.min(), sums.max()))
        diff_hist, _ = np.histogram(diffs, bins=n_bins_diff, range=(diffs.min(), diffs.max()))
        
        n_sum_soft = np.sum(sum_hist > 0)
        n_diff_soft = np.sum(diff_hist > 0)
        
        if n_sum_soft == 0:
            return 0.0
        
        base = 2 * max_u + 1
        if base <= 1:
            return 1.0
        
        ratio = n_diff_soft / n_sum_soft
        if ratio <= 1:
            return 1.0
        
        score = 1 + math.log(ratio) / math.log(base)
        return score
    
    # Run continuous optimization from multiple starting points
    if time.time() < deadline - 20:
        # Starting points from best discrete set and random perturbations
        start_positions = [best_set.copy()]
        
        # Add some random starting points
        for _ in range(5):
            n_elem = random.randint(4, 10)
            max_u = random.randint(20, 200)
            pos = [0.0] + [random.uniform(1, max_u) for _ in range(n_elem - 1)]
            pos = sorted(pos)
            start_positions.append(pos)
        
        for init_pos in start_positions:
            if time.time() > deadline - 15:
                break
            
            positions = np.array(init_pos, dtype=np.float64)
            n = len(positions)
            
            # Simple gradient-free optimization: coordinate descent with perturbations
            current_score = soft_score_continuous(positions)
            
            for iteration in range(200):
                if time.time() > deadline - 15:
                    break
                
                improved = False
                for i in range(1, n):  # Keep 0 fixed
                    # Try perturbing position i
                    best_new_pos = positions[i]
                    best_new_score = current_score
                    
                    for delta in [-2.0, -1.0, -0.5, -0.2, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]:
                        new_pos = positions[i] + delta
                        if new_pos < 0.01:  # Keep positive
                            continue
                        if new_pos > 5000:  # Cap
                            continue
                        
                        positions[i] = new_pos
                        # Maintain sorted order
                        positions_sorted = sorted(positions)
                        new_score = soft_score_continuous(positions_sorted)
                        
                        if new_score > best_new_score:
                            best_new_score = new_score
                            best_new_pos = new_pos
                        positions[i] = positions_sorted[i] if positions_sorted[i] != positions[i] else positions[i]
                    
                    if best_new_pos != positions[i]:
                        positions[i] = best_new_pos
                        positions = sorted(positions)
                        current_score = best_new_score
                        improved = True
                
                if not improved:
                    break
            
            # Round to integers
            rounded = [int(round(p)) for p in positions]
            rounded = sorted(set(rounded))
            if 0 not in rounded:
                rounded.append(0)
                rounded = sorted(rounded)
            
            try_set(rounded)
    
    # Phase 3: Structured sets
    for max_u in range(10, 300):
        if time.time() > deadline - 10:
            break
        
        # Two blocks
        for a in range(1, 12):
            for b in range(a + 1, max_u - a + 1):
                if time.time() > deadline - 10:
                    break
                block1 = list(range(0, a + 1))
                block2 = list(range(b, b + a + 1))
                cand = sorted(set(block1 + block2))
                if max(cand) <= max_u:
                    try_set(cand)
        
        # Geometric
        for r in [2, 3, 4, 5]:
            geo = [0, 1]
            while geo[-1] * r <= max_u:
                geo.append(geo[-1] * r)
            try_set(geo)
        
        # Arithmetic progressions
        for d in range(1, 15):
            ap = []
            v = 0
            while v <= max_u:
                ap.append(v)
                v += d
            try_set(ap)
    
    # Phase 4: Local search around best set
    if time.time() < deadline - 10:
        current = list(best_set)
        no_improve = 0
        max_no_improve = 100
        
        while time.time() < deadline - 10 and no_improve < max_no_improve:
            improved = False
            
            # Try adding an element
            if len(current) < 20:
                max_cur = max(current)
                candidates = set()
                for x in current:
                    for dx in [-3, -2, -1, 1, 2, 3]:
                        candidates.add(x + dx)
                candidates.update(range(max_cur + 1, max_cur + 15))
                candidates.discard(0)
                
                for new_el in sorted(candidates):
                    if new_el in current or new_el < 0:
                        continue
                    cand = sorted(current + [new_el])
                    s = compute_score_exact(cand)
                    if s > best_score:
                        best_score = s
                        best_set = list(cand)
                        current = list(cand)
                        improved = True
                        break
            
            # Try replacing an element
            if len(current) >= 3:
                for i in range(len(current)):
                    if current[i] == 0:
                        continue
                    for new_el in range(max(1, current[i] - 10), current[i] + 11):
                        if new_el in current or new_el < 0:
                            continue
                        cand = list(current)
                        cand[i] = new_el
                        cand = sorted(set(cand))
                        if 0 not in cand:
                            continue
                        s = compute_score_exact(cand)
                        if s > best_score:
                            best_score = s
                            best_set = list(cand)
                            current = list(cand)
                            improved = True
                            break
                    if improved:
                        break
            
            # Try removing an element
            if len(current) > 3:
                for i in range(len(current)):
                    if current[i] == 0:
                        continue
                    cand = [x for j, x in enumerate(current) if j != i]
                    s = compute_score_exact(cand)
                    if s > best_score:
                        best_score = s
                        best_set = list(cand)
                        current = list(cand)
                        improved = True
                        break
            
            if not improved:
                no_improve += 1
            else:
                no_improve = 0
    
    return list(best_set)
# EVOLVE-BLOCK-END
