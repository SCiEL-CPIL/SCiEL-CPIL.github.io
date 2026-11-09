import itertools
import math
import time
import random
import numpy as np

def _score(U):
    U = list(set(U))
    if 0 not in U:
        return 0.0
    if len(U) < 2:
        return 0.0
    mx = max(U)
    if mx == 0:
        return 0.0
    U_arr = np.array(U)
    
    # Use FFT-based approach for both sum and difference sets
    # This is faster for larger sets
    size = 2 * mx + 1
    fft_size = 1
    while fft_size < size:
        fft_size *= 2
    
    # Indicator vector for U
    ind = np.zeros(fft_size)
    ind[U_arr] = 1.0
    
    # Sumset via autocorrelation (convolution with itself)
    fft_ind = np.fft.fft(ind)
    conv_sum = np.real(np.fft.ifft(fft_ind * fft_ind))
    n_plus = np.sum(conv_sum >= 0.5)
    
    # Difference set: U-U = {u1-u2 : u1,u2 in U}
    # We can compute this by convolving ind with a flipped version
    # ind_flipped[i] = 1 if -i is in U, but U contains nonnegatives
    # So U-U has elements in [-mx, mx]
    # Let's shift: consider V = U + mx, then V-V = U-U
    # Actually, simpler: the difference set size can be computed as
    # the number of pairs (a,b) in U×U such that a-b is unique... no
    # Better: create an indicator for U in a range that allows negative indices
    # Place U at positions [mx, 2*mx] so that differences fall in [0, 2*mx]
    ind_shifted = np.zeros(fft_size)
    ind_shifted[U_arr + mx] = 1.0
    
    fft_ind_shifted = np.fft.fft(ind_shifted)
    conv_diff = np.real(np.fft.ifft(fft_ind_shifted * np.conj(fft_ind_shifted)))
    # The difference set U-U corresponds to positions where conv_diff > 0
    # in the range [0, 2*mx] (after shift)
    n_minus = np.sum(conv_diff >= 0.5)
    
    if n_plus == 0:
        return 0.0
    ratio = n_minus / n_plus
    if ratio <= 1:
        return 1.0
    return 1.0 + math.log(ratio) / math.log(2 * mx + 1)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    best_score = 0.0
    best_set = [0, 1, 3]

    def try_set(U):
        nonlocal best_score, best_set
        s = _score(U)
        if s > best_score:
            best_score = s
            best_set = list(U)

    # Phase 1: Structured sets
    structured_sets = []
    # Powers of 2
    for n in range(2, 12):
        structured_sets.append([0] + [2**i for i in range(n)])
    # Small arithmetic progressions
    for a in range(1, 10):
        for d in range(1, 10):
            structured_sets.append([0] + [a * i * d for i in range(1, 8)])
    # Mixed: small + large
    for small in range(1, 8):
        for large in range(10, 80, 3):
            structured_sets.append([0, small, large])
            structured_sets.append([0, small, large, 2*large])
    # Fibonacci-like
    fib = [0, 1]
    for i in range(1, 15):
        fib.append(fib[-1] + fib[-2])
    for n in range(3, len(fib)):
        structured_sets.append(fib[:n+1])

    for U in structured_sets:
        if time.time() > deadline:
            return best_set
        try_set(U)

    # Phase 2: Exhaustive search over small ranges
    max_val = 35
    candidates = list(range(1, max_val + 1))

    for size in range(2, 8):
        if time.time() > deadline:
            return best_set
        for combo in itertools.combinations(candidates, size - 1):
            if time.time() > deadline:
                return best_set
            U = [0] + list(combo)
            try_set(U)

    # Phase 3: Hill climbing from best set
    if time.time() < deadline - 10:
        current = list(best_set)
        current_score = best_score
        max_val_search = 150  # Extend search range with faster scoring
        for iteration in range(500):
            if time.time() > deadline - 5:
                break
            improved = False
            # Try adding a new element
            for x in range(1, max_val_search + 1):
                if x not in current:
                    new_U = sorted(current + [x])
                    s = _score(new_U)
                    if s > current_score:
                        current = new_U
                        current_score = s
                        improved = True
                        break
            # Try removing an element (keep 0)
            for i in range(1, len(current)):
                new_U = current[:i] + current[i+1:]
                s = _score(new_U)
                if s > current_score:
                    current = new_U
                    current_score = s
                    improved = True
                    break
            # Try swapping an element
            for i in range(1, len(current)):
                for x in range(1, max_val_search + 1):
                    if x not in current:
                        new_U = current[:i] + [x] + current[i+1:]
                        s = _score(new_U)
                        if s > current_score:
                            current = new_U
                            current_score = s
                            improved = True
                            break
                if improved:
                    break
            if not improved:
                break
        try_set(current)

    # Phase 4: Randomized local search from multiple restarts
    if time.time() < deadline - 10:
        max_val_restart = 200  # Extend search range
        for restart in range(10):
            if time.time() > deadline - 5:
                break
            # Start from a perturbation of best_set
            current = list(best_set)
            # Add 1-3 random elements
            for _ in range(random.randint(1, 3)):
                x = random.randint(1, max(best_set) + 50)
                if x not in current:
                    current.append(x)
            current = sorted(set(current))
            current_score = _score(current)

            # Local search
            for iteration in range(150):
                if time.time() > deadline - 5:
                    break
                improved = False
                # Try adding
                for x in range(1, max_val_restart + 1):
                    if x not in current:
                        new_U = sorted(current + [x])
                        s = _score(new_U)
                        if s > current_score:
                            current = new_U
                            current_score = s
                            improved = True
                            break
                # Try removing
                for i in range(1, len(current)):
                    new_U = current[:i] + current[i+1:]
                    s = _score(new_U)
                    if s > current_score:
                        current = new_U
                        current_score = s
                        improved = True
                        break
                # Try swapping
                for i in range(1, len(current)):
                    for x in range(1, max_val_restart + 1):
                        if x not in current:
                            new_U = current[:i] + [x] + current[i+1:]
                            s = _score(new_U)
                            if s > current_score:
                                current = new_U
                                current_score = s
                                improved = True
                                break
                    if improved:
                        break
                if not improved:
                    break
            try_set(current)

    return best_set
