import math
import time
import random
import itertools
import numpy as np

def _score(U):
    U = list(set(U))
    if 0 not in U:
        return 0.0
    mx = max(U)
    if mx == 0:
        return 0.0
    plus = set()
    minus = set()
    for a in U:
        for b in U:
            plus.add(a + b)
            minus.add(a - b)
    n_plus = len(plus)
    n_minus = len(minus)
    if n_plus == 0:
        return 0.0
    ratio = n_minus / n_plus
    if ratio <= 1:
        return 1.0
    return 1.0 + math.log(ratio) / math.log(2 * mx + 1)

def _fast_score(U):
    """Fast score using FFT convolution for sumset/difference set cardinality."""
    U = sorted(set(U))
    if 0 not in U:
        return 0.0
    mx = max(U)
    if mx == 0:
        return 0.0
    
    # Sumset: U+U = {a+b : a,b in U}
    # Use FFT: create indicator vector, convolve with itself
    # Size needs to be at least 2*mx + 1
    size = 2 * mx + 1
    # Next power of 2 for FFT efficiency
    nfft = 1
    while nfft < size:
        nfft *= 2
    
    vec = np.zeros(nfft)
    for x in U:
        vec[x] = 1.0
    
    # Sumset via convolution
    fft_vec = np.fft.fft(vec)
    conv_sum = np.fft.ifft(fft_vec * fft_vec)
    # Sumset cardinality: count positions where conv_sum > 0.5 (should be integer)
    n_plus = int(np.sum(conv_sum.real > 0.5))
    
    # Difference set: U-U = {a-b : a,b in U}
    # This is convolution of U with reversed U
    # rev_vec[k] = 1 if (mx - k) in U, i.e., rev_vec[i] = vec[mx - i]
    rev_vec = np.zeros(nfft)
    for x in U:
        rev_vec[mx - x] = 1.0
    
    fft_rev = np.fft.fft(rev_vec)
    conv_diff = np.fft.ifft(fft_vec * fft_rev)
    # Difference set: values range from -mx to mx, mapped to 0..2*mx
    # conv_diff[i] corresponds to value i - mx
    # Count non-zero entries
    n_minus = int(np.sum(conv_diff.real > 0.5))
    
    if n_plus == 0:
        return 0.0
    ratio = n_minus / n_plus
    if ratio <= 1:
        return 1.0
    return 1.0 + math.log(ratio) / math.log(2 * mx + 1)

def _simulated_annealing(U, max_iter=50000, time_limit=60.0, T_start=0.01, T_end=0.001):
    """Simulated annealing to optimize the score."""
    U = sorted(set(U))
    best_U = list(U)
    best_score = _fast_score(U)
    current_U = list(U)
    current_score = best_score
    start = time.time()
    
    for it in range(max_iter):
        if time.time() - start > time_limit:
            break
        
        # Update temperature (geometric cooling)
        progress = it / max_iter
        T = T_start * (T_end / T_start) ** progress
        
        # Generate perturbation
        mx = max(current_U) if current_U else 0
        r = random.random()
        
        if r < 0.35:
            # Add a random element
            new_elem = random.randint(1, max(2 * mx, 10))
            if new_elem not in current_U:
                candidate = sorted(current_U + [new_elem])
            else:
                continue
        elif r < 0.65:
            # Remove a random element (not 0)
            removable = [x for x in current_U if x != 0]
            if not removable:
                continue
            rem = random.choice(removable)
            candidate = [x for x in current_U if x != rem]
            candidate = sorted(candidate)
        elif r < 0.85:
            # Replace a random element (not 0)
            replaceable = [x for x in current_U if x != 0]
            if not replaceable:
                continue
            old = random.choice(replaceable)
            new = random.randint(1, max(2 * mx, 10))
            if new == old or new in current_U:
                continue
            candidate = sorted([x if x != old else new for x in current_U])
        else:
            # Shift a random element by a small amount
            shiftable = [x for x in current_U if x != 0]
            if not shiftable:
                continue
            old = random.choice(shiftable)
            delta = random.choice([-3, -2, -1, 1, 2, 3, 5, 7])
            new = old + delta
            if new < 1 or new in current_U:
                continue
            candidate = sorted([x if x != old else new for x in current_U])
        
        cand_score = _fast_score(candidate)
        delta_s = cand_score - current_score
        
        # Acceptance criterion
        if delta_s > 0 or random.random() < math.exp(delta_s / T):
            current_U = candidate
            current_score = cand_score
            if current_score > best_score:
                best_score = current_score
                best_U = list(current_U)
    
    return best_U, best_score

def _hill_climb(U, max_iter=5000, time_limit=2.0):
    """Local search: try adding/removing/swapping elements."""
    U = sorted(set(U))
    best_score = _fast_score(U)
    best_U = list(U)
    start = time.time()
    
    for it in range(max_iter):
        if time.time() - start > time_limit:
            break
        if random.random() < 0.3:
            # Try adding a random element
            mx = max(U)
            new_elem = random.randint(1, max(2 * mx, 10))
            if new_elem not in U:
                candidate = sorted(U + [new_elem])
                s = _fast_score(candidate)
                if s > best_score + 1e-12:
                    best_score = s
                    best_U = candidate
        elif random.random() < 0.5:
            # Try removing a random element (not 0)
            removable = [x for x in U if x != 0]
            if removable:
                rem = random.choice(removable)
                candidate = [x for x in U if x != rem]
                s = _fast_score(candidate)
                if s > best_score + 1e-12:
                    best_score = s
                    best_U = candidate
        else:
            # Try replacing a random element (not 0)
            replaceable = [x for x in U if x != 0]
            if replaceable:
                old = random.choice(replaceable)
                mx = max(U)
                new = random.randint(1, max(2 * mx, 10))
                if new != old and new not in U:
                    candidate = sorted([x if x != old else new for x in U])
                    s = _fast_score(candidate)
                    if s > best_score + 1e-12:
                        best_score = s
                        best_U = candidate
    return best_U, best_score

def _lacunary(n, base):
    """Lacunary sequence: 0, 1, base, base^2, ..."""
    U = [0, 1]
    for i in range(2, n):
        U.append(base ** (i - 1))
    return U

def _geometric_int(n, ratio):
    """Geometric progression with integer ratio."""
    U = [0, 1]
    for i in range(2, n):
        U.append(ratio ** (i - 1))
    return U

def construct():
    deadline = time.time() + 130  # Safety margin before 150s limit
    
    best_U = [0, 1, 2, 4]
    best_score = _fast_score(best_U)
    
    # Try known good starting points
    candidates = [
        [0, 1, 2, 4],
        [0, 1, 3, 7, 15],  # 2^k - 1
        [0, 1, 2, 4, 8, 16],  # powers of 2
        [0, 1, 3, 9, 27],  # powers of 3
        [0, 1, 4, 16, 64],  # powers of 4
        [0, 1, 2, 5, 10, 20],
        [0, 1, 3, 6, 10, 15],  # triangular numbers
    ]
    
    for cand in candidates:
        s = _fast_score(cand)
        if s > best_score:
            best_score = s
            best_U = list(cand)
    
    # Try lacunary sequences with different bases
    for base in [2, 3, 4, 5, 6, 7, 8, 10, 12, 16]:
        for n in range(3, 12):
            if time.time() > deadline - 60:
                break
            cand = _lacunary(n, base)
            s = _fast_score(cand)
            if s > best_score:
                best_score = s
                best_U = list(cand)
    
    # Try geometric with various ratios
    for ratio in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
        for n in range(3, 12):
            if time.time() > deadline - 60:
                break
            cand = _geometric_int(n, ratio)
            s = _fast_score(cand)
            if s > best_score:
                best_score = s
                best_U = list(cand)
    
    # Simulated annealing from current best
    if time.time() < deadline - 30:
        U_sa, s_sa = _simulated_annealing(best_U, max_iter=100000, time_limit=40, T_start=0.02, T_end=0.0005)
        if s_sa > best_score:
            best_score = s_sa
            best_U = U_sa
    
    # Simulated annealing from several random starts
    random.seed(42)
    for trial in range(5):
        if time.time() > deadline - 20:
            break
        # Random start: pick 3-8 elements
        n_elem = random.randint(3, 8)
        elems = random.sample(range(1, 25), n_elem - 1)
        U_start = sorted([0] + elems)
        U_sa, s_sa = _simulated_annealing(U_start, max_iter=50000, time_limit=10, T_start=0.02, T_end=0.001)
        if s_sa > best_score:
            best_score = s_sa
            best_U = U_sa
    
    # Final hill climb polish from best
    if time.time() < deadline - 5:
        U_climbed, s_climbed = _hill_climb(best_U, max_iter=3000, time_limit=8)
        if s_climbed > best_score:
            best_score = s_climbed
            best_U = U_climbed
    
    # Verify with exact score and ensure validity
    if 0 not in best_U:
        best_U = [0] + sorted(best_U)
    
    # Final exact score verification
    exact_score = _score(best_U)
    
    return sorted(set(best_U))
