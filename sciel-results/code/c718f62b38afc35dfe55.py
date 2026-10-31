import time
import random
import math
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0  # Leave 10s safety margin for 150s limit
    
    def compute_score_fft(U):
        """Compute score using FFT for O(M log M) set size computation."""
        if not U or len(U) <= 1:
            return 1.0
        max_u = max(U)
        if max_u == 0:
            return 1.0
        
        n = len(U)
        
        # For small sets, direct computation is faster
        if n <= 30:
            sumset = set()
            diffset = set()
            for i in range(n):
                for j in range(i, n):
                    s = U[i] + U[j]
                    sumset.add(s)
                    if i != j:
                        d = U[i] - U[j]
                        diffset.add(d)
                        diffset.add(-d)
            diffset.add(0)
            size_sum = len(sumset)
            size_diff = len(diffset)
        else:
            # FFT-based computation
            # Sumset: convolution of indicator with itself
            # f = indicator of U, sumset size = number of nonzero entries in f*f
            size = 2 * max_u + 1
            # Next power of 2 for FFT
            fft_size = 1
            while fft_size < size:
                fft_size *= 2
            
            f = np.zeros(fft_size, dtype=np.float64)
            for u in U:
                f[u] = 1.0
            
            # FFT for sumset
            F = np.fft.fft(f)
            F2 = F * F
            conv = np.fft.ifft(F2).real
            
            # Count nonzero entries (sumset)
            # Due to floating point, use threshold
            threshold = 0.5
            size_sum = int(np.sum(conv[:size] > threshold))
            
            # Difference set: |U-U| = number of distinct differences
            # This is related to autocorrelation
            # autocorr[k] = sum_j f[j] * f[j+k] for k >= 0
            # |U-U| = 1 + 2 * sum_{k>0, autocorr[k]>0} 1
            # Actually: |U-U| = number of distinct values in {a-b : a,b in U}
            # = number of k such that exists a,b in U with a-b=k
            # = number of k in [-max_u, max_u] where autocorr is nonzero
            
            # Compute autocorrelation via FFT
            # autocorr[k] = sum_j f[j+k] * f[j]
            # This is the cross-correlation of f with itself
            F_conj = np.conj(F)
            autocorr = np.fft.ifft(F * F_conj).real
            
            # autocorr is circular; we need linear autocorrelation
            # For k in [0, fft_size), autocorr[k] gives sum_j f[j] * f[(j+k) % fft_size]
            # We want sum_j f[j+k] * f[j] for k in [-max_u, max_u]
            # Note: autocorr[-k] = autocorr[k] for real signals
            
            # The linear autocorrelation at lag k is:
            # For k >= 0: sum_{j=0}^{max_u-k} f[j+k]*f[j]
            # For k < 0: sum_{j=0}^{max_u+k} f[j+k]*f[j] = sum_{j=0}^{max_u-k} f[j]*f[j-k]
            
            # Using FFT: ifft(F * conj(F)) gives circular autocorrelation
            # For our case where fft_size > 2*max_u, circular = linear for |k| <= max_u
            
            # autocorr[k] for k in [0, fft_size/2] gives sum_j f[j]*f[j+k] (circular)
            # But we need to be careful about the indexing
            
            # Let's compute directly: diffset size = number of distinct (a-b) for a,b in U
            # This equals the number of nonzero entries in the "difference polynomial"
            # D(x) = (sum_{a in U} x^a) * (sum_{b in U} x^{-b})
            # = sum_{a,b in U} x^{a-b}
            # The number of distinct exponents with nonzero coefficient = |U-U|
            
            # Using FFT: let g[x] = f[-x mod fft_size] (time-reversed)
            # Then D = f * g (convolution)
            g = np.zeros(fft_size, dtype=np.float64)
            for u in U:
                if u == 0:
                    g[0] = 1.0
                else:
                    g[fft_size - u] = 1.0
            
            G = np.fft.fft(g)
            D = np.fft.ifft(F * G).real
            
            # D[k] for k in [0, fft_size) represents coefficient of x^k in D(x)
            # But we want exponents in [-max_u, max_u]
            # k=0 -> exponent 0
            # k in [1, max_u] -> exponent k
            # k in [fft_size-max_u, fft_size-1] -> exponent k - fft_size (negative)
            
            # Count nonzero entries
            diff_nonzero = 0
            for k in range(0, max_u + 1):
                if abs(D[k]) > threshold:
                    diff_nonzero += 1
            for k in range(fft_size - max_u, fft_size):
                if abs(D[k]) > threshold:
                    diff_nonzero += 1
            
            size_diff = diff_nonzero
        
        if size_sum == 0:
            return 1.0
            
        ratio = size_diff / size_sum
        if ratio <= 1.0:
            return 1.0
        
        denom = math.log(2 * max_u + 1)
        if denom == 0:
            return 1.0
            
        return 1.0 + math.log(ratio) / denom

    # Known optimal Golomb rulers (these have all distinct positive differences)
    # These maximize |U-U| for given max(U), which helps the ratio
    golomb_rulers = {
        1: [0],
        2: [0, 1],
        3: [0, 1, 3],
        4: [0, 1, 4, 6],
        5: [0, 1, 4, 9, 11],
        6: [0, 1, 4, 10, 12, 17],
        7: [0, 1, 4, 10, 12, 17, 25],
        8: [0, 1, 4, 10, 12, 17, 25, 29],
        9: [0, 1, 4, 10, 12, 17, 25, 29, 34],
        10: [0, 1, 4, 10, 12, 17, 25, 29, 34, 41],
        11: [0, 1, 4, 10, 12, 17, 25, 29, 34, 41, 47],
        12: [0, 1, 4, 10, 12, 17, 25, 29, 34, 41, 47, 55],
    }
    
    # Additional candidates: subsets of Golomb rulers, and some hand-crafted sets
    initial_sets = []
    for n in range(1, 13):
        if n in golomb_rulers:
            initial_sets.append(golomb_rulers[n])
    
    # Add more structured variations that may have better ratio
    extra_sets = [
        [0, 1, 3, 7],
        [0, 1, 4, 6, 10],
        [0, 2, 5, 11],
        [0, 1, 5, 11],
        [0, 3, 7, 12],
        [0, 1, 4, 9, 11, 18],
        [0, 2, 7, 12, 18],
        [0, 1, 5, 10, 16],
        [0, 1, 4, 9, 15, 20],
        [0, 1, 4, 10, 18, 23],
        [0, 2, 7, 14, 21],
        [0, 1, 5, 12, 19],
        [0, 3, 8, 14, 20],
        [0, 1, 4, 9, 16, 22],
        [0, 2, 6, 13, 19, 25],
        [0, 1, 5, 11, 17, 23],
        [0, 4, 9, 15, 21],
        [0, 1, 6, 11, 17, 22],
    ]
    initial_sets.extend(extra_sets)
    
    best_score = 0.0
    best_set = [0]
    
    for s in initial_sets:
        score = compute_score_fft(s)
        if score > best_score:
            best_score = score
            best_set = list(s)
    
    # Local search with multiple restarts
    max_restarts = 12
    best_overall_score = best_score
    best_overall_set = best_set
    
    for restart in range(max_restarts):
        if time.time() > deadline - 5:
            break
        
        # Start from best set or a random Golomb ruler
        if restart == 0:
            current_set = list(best_overall_set)
        elif restart % 3 == 1:
            # Pick a random Golomb ruler as starting point
            n = random.randint(3, 12)
            if n in golomb_rulers:
                current_set = list(golomb_rulers[n])
            else:
                current_set = list(best_overall_set)
        elif restart % 3 == 2:
            # Start from a random extra set
            current_set = list(random.choice(extra_sets))
        else:
            # Perturb the best set
            current_set = list(best_overall_set)
            for _ in range(random.randint(1, 4)):
                if len(current_set) > 1:
                    idx = random.randint(1, len(current_set) - 1)
                    current_set[idx] = max(1, current_set[idx] + random.randint(-6, 6))
            current_set = sorted(set(current_set))
            if 0 not in current_set:
                current_set.insert(0, 0)
        
        current_score = compute_score_fft(current_set)
        
        # Local search phase with adaptive acceptance
        no_improve_count = 0
        max_no_improve = 800
        temperature = 0.02
        
        while time.time() < deadline - 5 and no_improve_count < max_no_improve:
            # Generate a neighbor with weighted operations
            neighbor = list(current_set)
            r = random.random()
            
            if r < 0.20 and len(neighbor) > 1:
                # Remove a random element (not 0)
                idx = random.randint(1, len(neighbor) - 1)
                neighbor.pop(idx)
                if len(neighbor) == 0:
                    continue
                if 0 not in neighbor:
                    neighbor.append(0)
                    neighbor.sort()
            elif r < 0.50:
                # Add a random element - bias toward gaps in the set
                if not neighbor:
                    continue
                max_val = max(neighbor)
                # Try to fill gaps or extend
                if random.random() < 0.7:
                    # Fill a gap
                    gaps = []
                    for i in range(len(neighbor) - 1):
                        if neighbor[i+1] - neighbor[i] > 1:
                            gaps.append((neighbor[i], neighbor[i+1]))
                    if gaps:
                        gap_start, gap_end = random.choice(gaps)
                        new_val = random.randint(gap_start + 1, gap_end - 1)
                        if new_val not in neighbor:
                            neighbor.append(new_val)
                            neighbor.sort()
                    else:
                        new_val = random.randint(1, max_val + 15)
                        if new_val not in neighbor:
                            neighbor.append(new_val)
                            neighbor.sort()
                else:
                    # Extend or random
                    upper_bound = max(2 * max_val + 20, 30)
                    new_val = random.randint(1, upper_bound)
                    if new_val not in neighbor:
                        neighbor.append(new_val)
                        neighbor.sort()
            elif r < 0.75:
                # Modify a random element (not 0) to a nearby value
                if len(neighbor) > 1:
                    idx = random.randint(1, len(neighbor) - 1)
                    old_val = neighbor[idx]
                    # Try small perturbations with varying magnitudes
                    delta = random.choice([-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6])
                    new_val = old_val + delta
                    if new_val >= 1 and new_val not in neighbor:
                        neighbor[idx] = new_val
                        neighbor.sort()
                    else:
                        continue
            else:
                # Swap: remove one element and add another
                if len(neighbor) > 1:
                    idx = random.randint(1, len(neighbor) - 1)
                    removed_val = neighbor.pop(idx)
                    max_val = max(neighbor) if neighbor else 0
                    # Try to add a value that creates more distinct differences
                    upper_bound = max(2 * max_val + 20, 30)
                    new_val = random.randint(1, upper_bound)
                    if new_val not in neighbor:
                        neighbor.append(new_val)
                        neighbor.sort()
                    else:
                        # Revert
                        neighbor.append(removed_val)
                        neighbor.sort()
                        continue
            
            if not neighbor or 0 not in neighbor:
                continue
                
            new_score = compute_score_fft(neighbor)
            
            if new_score > current_score:
                current_set = neighbor
                current_score = new_score
                no_improve_count = 0
                if current_score > best_overall_score:
                    best_overall_score = current_score
                    best_overall_set = list(current_set)
            else:
                no_improve_count += 1
                # Simulated annealing: accept worse moves with decreasing probability
                delta_score = new_score - current_score
                if delta_score < 0 and random.random() < math.exp(delta_score / temperature):
                    current_set = neighbor
                    current_score = new_score
                temperature *= 0.993
    
    # Final polish: try all single-element modifications on the best set
    if time.time() < deadline - 2:
        current_set = list(best_overall_set)
        current_score = compute_score_fft(current_set)
        
        for _ in range(100):
            if time.time() > deadline - 2:
                break
            
            # Try removing each element
            improved = False
            for i in range(1, len(current_set)):
                neighbor = list(current_set)
                neighbor.pop(i)
                if len(neighbor) == 0:
                    continue
                if 0 not in neighbor:
                    neighbor.append(0)
                    neighbor.sort()
                new_score = compute_score_fft(neighbor)
                if new_score > current_score:
                    current_set = neighbor
                    current_score = new_score
                    improved = True
                    break
            
            if improved:
                if current_score > best_overall_score:
                    best_overall_score = current_score
                    best_overall_set = list(current_set)
                continue
            
            # Try adding elements
            max_val = max(current_set) if current_set else 0
            upper_bound = max(2 * max_val + 15, 25)
            for new_val in range(1, upper_bound + 1):
                if new_val in current_set:
                    continue
                neighbor = list(current_set)
                neighbor.append(new_val)
                neighbor.sort()
                new_score = compute_score_fft(neighbor)
                if new_score > current_score:
                    current_set = neighbor
                    current_score = new_score
                    improved = True
                    break
            
            if improved:
                if current_score > best_overall_score:
                    best_overall_score = current_score
                    best_overall_set = list(current_set)
                continue
            
            # Try modifying elements with wider range
            for i in range(1, len(current_set)):
                old_val = current_set[i]
                for delta in range(-5, 6):
                    if delta == 0:
                        continue
                    new_val = old_val + delta
                    if new_val < 1 or new_val in current_set:
                        continue
                    neighbor = list(current_set)
                    neighbor[i] = new_val
                    neighbor.sort()
                    new_score = compute_score_fft(neighbor)
                    if new_score > current_score:
                        current_set = neighbor
                        current_score = new_score
                        improved = True
                        break
                if improved:
                    break
            
            if improved:
                if current_score > best_overall_score:
                    best_overall_score = current_score
                    best_overall_set = list(current_set)
    
    return best_overall_set
# EVOLVE-BLOCK-END
