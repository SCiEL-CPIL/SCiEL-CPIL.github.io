import math
import time
import random
import numpy as np

_START_TIME = time.time()
_TIME_LIMIT = 120.0
_SAFETY_MARGIN = 10.0

def _elapsed():
    return time.time() - _START_TIME

def _remaining():
    return _TIME_LIMIT - _SAFETY_MARGIN - _elapsed()

def _score_fft(U):
    """
    FFT-based exact score computation.
    Uses convolution to compute sumset and difference set sizes.
    """
    U = sorted(set(int(x) for x in U if x >= 0))
    if len(U) < 2:
        return 0.0
    M = max(U)
    if M == 0:
        return 0.0
    
    # Sumset: compute autocorrelation via FFT
    # For sumset, we need indicator of U convolved with indicator of U
    n = len(U)
    max_val = M
    
    # Create indicator array
    ind = np.zeros(2 * max_val + 1, dtype=np.float64)
    for x in U:
        ind[x] = 1.0
    
    # Sumset: ind * ind (convolution)
    # FFT length needs to be at least 2*max_val+1
    fft_len = 1
    while fft_len < 2 * (max_val + 1):
        fft_len *= 2
    
    ind_fft = np.fft.fft(ind, n=fft_len)
    sum_fft = ind_fft * ind_fft
    sum_conv = np.fft.ifft(sum_fft).real
    
    # Sumset size: number of positions where sum_conv > 0.5
    sumset_size = int(np.sum(sum_conv > 0.5))
    
    # Difference set: need ind shifted and convolved
    # diff(a,b) = a - b, so we need correlation
    # Create reversed indicator for difference
    ind_rev = np.zeros(2 * max_val + 1, dtype=np.float64)
    for x in U:
        ind_rev[2 * max_val - x] = 1.0
    
    ind_rev_fft = np.fft.fft(ind_rev, n=fft_len)
    diff_fft = ind_fft * ind_rev_fft
    diff_conv = np.fft.ifft(diff_fft).real
    
    # Difference set: count positions where diff_conv > 0.5
    # The differences range from -M to M, so we look at the middle portion
    # In the convolution, the zero difference is at position max_val
    diffset_size = int(np.sum(diff_conv > 0.5))
    
    denom = math.log(2 * M + 1, 2)
    if denom == 0:
        return 0.0
    return 1.0 + math.log(diffset_size / sumset_size, 2) / denom

def _score_small(U):
    """Exact O(n^2) score for small sets (fallback)."""
    U = sorted(set(int(x) for x in U if x >= 0))
    if len(U) < 2:
        return 0.0
    M = max(U)
    if M == 0:
        return 0.0
    sumset = set()
    diffset = set()
    for a in U:
        for b in U:
            sumset.add(a + b)
            diffset.add(a - b)
    denom = math.log(2 * M + 1, 2)
    if denom == 0:
        return 0.0
    return 1.0 + math.log(len(diffset) / len(sumset), 2) / denom

def _is_golomb(U):
    """Check if U is a Golomb ruler (all pairwise differences are unique)."""
    diffs = set()
    n = len(U)
    for i in range(n):
        for j in range(i + 1, n):
            d = U[j] - U[i]
            if d in diffs:
                return False
            diffs.add(d)
    return True

def _gen_golomb_optimal(n):
    """Return known optimal Golomb rulers for small n."""
    optimal = {
        3: [0, 1, 3],
        4: [0, 1, 4, 6],
        5: [0, 1, 4, 9, 11],
        6: [0, 1, 4, 10, 12, 17],
        7: [0, 1, 4, 10, 18, 23, 25],
        8: [0, 1, 4, 10, 18, 23, 29, 34],
        9: [0, 1, 4, 10, 18, 23, 29, 37, 44],
        10: [0, 1, 4, 10, 18, 23, 29, 37, 44, 55],
        11: [0, 1, 4, 10, 18, 23, 29, 37, 44, 52, 67],
        12: [0, 1, 4, 10, 18, 23, 29, 37, 44, 52, 60, 80],
    }
    if n in optimal:
        return list(optimal[n])
    return None

def _simulated_annealing(U, time_budget=30.0):
    """
    Simulated annealing search over Golomb rulers.
    Moves: shift element, insert, delete, swap.
    Accepts worse moves with probability exp(-delta/T).
    """
    U = sorted(list(U))
    if 0 not in U:
        U.insert(0, 0)
    
    best_U = list(U)
    best_s = _score_fft(best_U)
    current_U = list(U)
    current_s = best_s
    
    start = time.time()
    rng = random.Random(42)
    
    T_init = 0.5
    T_min = 0.01
    cooling = 0.995
    
    while time.time() - start < time_budget and _remaining() > 2.0:
        n = len(current_U)
        if n < 2:
            break
        
        # Choose a move
        move_type = rng.random()
        
        new_U = list(current_U)
        valid = True
        
        if move_type < 0.5 and n > 1:
            # Shift an existing element
            i = rng.randint(1, n - 1)
            delta = rng.randint(-15, 15)
            if delta != 0:
                new_val = current_U[i] + delta
                if new_val >= 0 and new_val not in current_U:
                    new_U[i] = new_val
                    new_U = sorted(new_U)
                    if not _is_golomb(new_U):
                        valid = False
                else:
                    valid = False
        elif move_type < 0.75 and _remaining() > 3.0:
            # Insert a new element
            if n < 20:  # Cap size to keep computation manageable
                if rng.random() < 0.5:
                    # Near existing
                    j = rng.randint(1, n - 1)
                    gap = current_U[j] - current_U[j-1]
                    if gap > 1:
                        pos = current_U[j-1] + rng.randint(1, gap - 1)
                        if pos not in current_U:
                            new_U.append(pos)
                            new_U = sorted(new_U)
                            if not _is_golomb(new_U):
                                valid = False
                        else:
                            valid = False
                    else:
                        valid = False
                else:
                    # Near end or beginning
                    pos = current_U[-1] + rng.randint(1, 5)
                    if pos not in current_U:
                        new_U.append(pos)
                        new_U = sorted(new_U)
                        if not _is_golomb(new_U):
                            valid = False
                    else:
                        valid = False
            else:
                valid = False
        elif move_type < 0.85 and n > 3:
            # Delete an element
            idx = rng.randint(1, n - 1)
            new_U = [x for j, x in enumerate(current_U) if j != idx]
            # Always valid if current is Golomb
        elif move_type < 0.95 and n > 3:
            # Swap two elements
            i, j = rng.sample(range(1, n), 2)
            new_U[i], new_U[j] = new_U[j], new_U[i]
            new_U = sorted(new_U)
            if not _is_golomb(new_U):
                valid = False
        else:
            # Large random perturbation
            i = rng.randint(1, n - 1)
            delta = rng.randint(-30, 30)
            if delta != 0:
                new_val = current_U[i] + delta
                if new_val >= 0 and new_val not in current_U:
                    new_U[i] = new_val
                    new_U = sorted(new_U)
                    if not _is_golomb(new_U):
                        valid = False
                else:
                    valid = False
            else:
                valid = False
        
        if not valid:
            continue
        
        # Score the new state
        new_s = _score_fft(new_U)
        
        # Acceptance criterion
        delta_s = new_s - current_s
        if delta_s >= 0 or rng.random() < math.exp(delta_s / T_init):
            current_U = new_U
            current_s = new_s
            if current_s > best_s + 1e-10:
                best_s = current_s
                best_U = list(current_U)
        
        # Cool down
        T_init = max(T_min, T_init * cooling)
    
    return best_U, best_s

def construct():
    global _START_TIME
    _START_TIME = time.time()
    random.seed(42)

    candidates = []
    
    # 1. Known optimal Golomb rulers for small n
    for n in range(3, 13):
        g = _gen_golomb_optimal(n)
        if g:
            candidates.append(g)
            
    # 2. Add random variations of small Golomb rulers
    for n in range(3, 12):
        base = _gen_golomb_optimal(n)
        if base:
            for _ in range(8):
                perturbed = list(base)
                if len(perturbed) > 2:
                    idx = random.randint(1, len(perturbed) - 1)
                    delta = random.randint(-3, 3)
                    if delta != 0:
                        new_val = perturbed[idx] + delta
                        if new_val > 0 and new_val not in perturbed:
                            perturbed[idx] = new_val
                            perturbed = sorted(perturbed)
                            if _is_golomb(perturbed):
                                candidates.append(perturbed)

    # 3. Generate additional random Golomb rulers using greedy construction
    for trial in range(20):
        rng = random.Random(42 + trial)
        n = rng.randint(5, 15)
        U = [0]
        current_max = 0
        for _ in range(n - 1):
            # Try to add elements that maintain Golomb property
            found = False
            for attempt in range(50):
                if current_max > 500:
                    break
                # Try values in a range
                low = max(1, current_max - 30)
                high = current_max + rng.randint(5, 20)
                val = rng.randint(low, high)
                if val in U or val <= 0:
                    continue
                new_U = U + [val]
                new_U = sorted(new_U)
                if _is_golomb(new_U):
                    U = new_U
                    current_max = val
                    found = True
                    break
            if not found:
                break
        if len(U) >= 4:
            candidates.append(U)

    # Deduplicate
    seen = set()
    unique = []
    for c in candidates:
        key = tuple(sorted(c))
        if key not in seen:
            seen.add(key)
            unique.append(list(c))

    # Score all initial candidates using FFT (fast)
    scored = [(_score_fft(s), s) for s in unique]
    scored.sort(key=lambda x: -x[0])

    if not scored:
        return [0, 1]

    best_U = scored[0][1]
    best_s = scored[0][0]

    # Simulated annealing on top candidates
    top_candidates = [s for _, s in scored[:10]]
    
    # Allocate time for SA
    per_sa_time = 8.0
    
    for s in top_candidates:
        if _remaining() < 10.0:
            break
        result, rs = _simulated_annealing(s, time_budget=per_sa_time)
        if rs > best_s:
            best_s = rs
            best_U = result

    # Final polish with longer SA on the best found
    if _remaining() > 15.0:
        polished, ps = _simulated_annealing(best_U, time_budget=min(30.0, _remaining() - 3.0))
        if ps > best_s:
            best_s = ps
            best_U = polished

    # Ensure validity
    best_U = sorted(set(int(x) for x in best_U if x >= 0))
    if 0 not in best_U:
        best_U.insert(0, 0)
        
    return best_U
