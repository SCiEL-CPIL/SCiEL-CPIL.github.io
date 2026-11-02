"""Program that constructs a finite set U of nonnegative integers containing 0
to maximize the sumset/difference-set exponent lower bound.
"""
import time
import math
import random
from itertools import product

import numpy as np

# Deadline guard: kill before 120s wall-clock
_START = time.time()
_DEADLINE = 110.0  # safety margin


def _elapsed():
    return time.time() - _START


def _remaining():
    return _DEADLINE - _elapsed()


def _score_set(U):
    """Compute the lower bound 1 + log(|U-U|/|U+U|)/log(2 max U + 1)."""
    if not U or 0 not in U:
        return 0.0
    U = sorted(set(U))
    if U[0] != 0:
        return 0.0
    if len(U) < 2:
        return 0.0
    max_u = U[-1]
    if max_u == 0:
        return 0.0

    # Compute sumset and difference set sizes
    U_arr = np.array(U, dtype=np.int64)

    # Sumset: all a+b for a,b in U
    sums = np.add.outer(U_arr, U_arr)
    n_sums = len(np.unique(sums))

    # Difference set: all a-b for a,b in U
    diffs = np.subtract.outer(U_arr, U_arr)
    n_diffs = len(np.unique(diffs))

    if n_sums == 0:
        return 0.0
    ratio = n_diffs / n_sums
    if ratio <= 1.0:
        return 0.0
    denom = math.log2(2 * max_u + 1)
    if denom == 0:
        return 0.0
    return 1.0 + math.log2(ratio) / denom


def _mixed_radix_set(radices, digits):
    """Construct a set from mixed-radix representation.
    
    radices: list of radixes [r0, r1, ..., rk]
    digits: list of digit arrays, one per coordinate
    
    The set is {sum(d_i * prod_{j<i} r_j) for each combination}
    Actually we generate all combinations via Cartesian product.
    """
    coords = []
    for d in digits:
        coords.append(d)
    result = {0}
    for i, r in enumerate(radices):
        d = digits[i]
        base = 1
        for j in range(i):
            base *= radices[j]
        new_result = set()
        for val in result:
            for digit in d:
                new_result.add(val + digit * base)
        result = new_result
    return sorted(result)


def _score_mixed_radix(radices, digit_lists):
    """Score a mixed-radix construction."""
    U = _mixed_radix_set(radices, digit_lists)
    return _score_set(U)


def _local_search_mixed_radix(radices, digit_lists, time_budget):
    """Local search over mixed-radix digit choices."""
    best_score = _score_mixed_radix(radices, digit_lists)
    best_radices = list(radices)
    best_digits = [list(d) for d in digit_lists]
    
    if best_score <= 0:
        return best_radices, best_digits, best_score
    
    rng = random.Random(42)
    deadline = time.time() + time_budget
    
    while time.time() < deadline:
        # Pick a random coordinate and change its digit set
        k = len(radices)
        i = rng.randrange(k)
        
        # Try expanding or modifying the digit set
        op = rng.choice(['add', 'remove', 'replace'])
        new_digits = [list(d) for d in digit_lists]
        
        if op == 'add' and len(new_digits[i]) < radices[i]:
            # Add a random new digit
            available = set(range(radices[i])) - set(new_digits[i])
            if available:
                new_digits[i].append(rng.choice(list(available)))
                new_digits[i].sort()
        elif op == 'remove' and len(new_digits[i]) > 1:
            idx = rng.randrange(len(new_digits[i]))
            new_digits[i].pop(idx)
        elif op == 'replace' and len(new_digits[i]) > 1:
            idx = rng.randrange(len(new_digits[i]))
            available = set(range(radices[i])) - set(new_digits[i])
            if available:
                new_digits[i][idx] = rng.choice(list(available))
                new_digits[i].sort()
        else:
            continue
        
        s = _score_mixed_radix(radices, new_digits)
        if s > best_score:
            best_score = s
            best_digits = new_digits
    
    return best_radices, best_digits, best_score


def _random_mixed_radix_search(time_budget):
    """Random search over small mixed-radix constructions."""
    rng = random.Random(123)
    deadline = time.time() + time_budget
    best_score = 0.0
    best_radices = None
    best_digits = None
    
    # Try various small mixed-radix structures
    for total_elements in range(4, 60):
        if time.time() > deadline:
            break
        
        # Try product of 2-4 factors
        n_factors = min(4, max(2, int(math.log2(total_elements)) + 1))
        
        for trial in range(50):
            if time.time() > deadline:
                break
            
            # Generate random radices that multiply to roughly total_elements
            radices = []
            remaining = total_elements
            for i in range(n_factors):
                if i == n_factors - 1:
                    radices.append(remaining)
                else:
                    r = max(2, rng.randint(2, max(2, remaining // 2)))
                    radices.append(r)
                    remaining //= r
            
            if remaining < 1:
                continue
            
            # For each coordinate, choose a subset of digits
            digit_lists = []
            for r in radices:
                # Start with a random subset including 0
                n_digits = rng.randint(2, min(r, 6))
                digits = [0] + rng.sample(range(1, r), min(n_digits - 1, r - 1))
                digit_lists.append(sorted(digits))
            
            s = _score_mixed_radix(radices, digit_lists)
            if s > best_score:
                best_score = s
                best_radices = list(radices)
                best_digits = [list(d) for d in digit_lists]
    
    if best_radices is None:
        return [0, 1, 3], 0.0
    
    U = _mixed_radix_set(best_radices, best_digits)
    return U, best_score


def _fft_score(U):
    """Score using FFT-based convolution for larger sets."""
    if not U or 0 not in U:
        return 0.0
    U = sorted(set(U))
    if U[0] != 0 or len(U) < 2:
        return 0.0
    max_u = U[-1]
    if max_u == 0:
        return 0.0
    
    # For FFT we need the array size to be at least 2*max_u+1
    # This is only feasible for smaller max_u
    if max_u > 20000:
        # Fall back to direct
        return _score_set(U)
    
    size = 2 * max_u + 1
    # Next power of 2 for FFT efficiency
    fft_size = 1
    while fft_size < 2 * size:
        fft_size *= 2
    
    indicator = np.zeros(fft_size, dtype=np.float64)
    for u in U:
        indicator[u] = 1.0
    
    # Sumset: convolution of indicator with itself
    # The sumset is the support of the convolution
    conv = np.fft.fft(indicator)
    conv = conv * conv
    conv = np.fft.ifft(conv).real
    n_sums = int(np.sum(conv > 1e-10))
    
    # Difference set: convolution of indicator with reversed indicator
    # Actually: diff set support = |{a-b : a,b in U}|
    # This is the autocorrelation
    rev = indicator.copy()
    # For difference, we compute indicator * reverse(indicator)
    # The difference a-b ranges from -max_u to max_u
    # Autocorrelation: sum over x of f(x) f(x+b) gives count of pairs with diff b
    autocorr = np.fft.ifft(np.fft.fft(indicator) * np.conj(np.fft.fft(indicator))).real
    # autocorr[k] = number of pairs (a,b) with a-b = k (shifted)
    n_diffs = int(np.sum(autocorr > 1e-10))
    
    if n_sums == 0:
        return 0.0
    ratio = n_diffs / n_sums
    if ratio <= 1.0:
        return 0.0
    denom = math.log2(2 * max_u + 1)
    if denom == 0:
        return 0.0
    return 1.0 + math.log2(ratio) / denom


def _local_search_general(U, time_budget):
    """General local search on a set of integers."""
    rng = random.Random(999)
    deadline = time.time() + time_budget
    best_set = sorted(set(U))
    best_score = _score_set(best_set)
    
    max_val = max(best_set) if best_set else 10
    search_radius = max(max_val * 2, 20)
    
    while time.time() < deadline:
        new_set = list(best_set)
        op = rng.choice(['add', 'remove', 'modify'])
        
        if op == 'add':
            val = rng.randint(1, search_radius)
            if val not in new_set:
                new_set.append(val)
        elif op == 'remove':
            if len(new_set) > 3:
                idx = rng.choice(range(1, len(new_set)))  # keep 0
                new_set.pop(idx)
        elif op == 'modify':
            if len(new_set) > 2:
                idx = rng.choice(range(1, len(new_set)))
                new_set[idx] = rng.randint(1, search_radius)
        
        new_set = sorted(set(new_set))
        if 0 not in new_set or len(new_set) < 2:
            continue
        
        # Use FFT score for larger sets
        if max(new_set) <= 20000:
            s = _fft_score(new_set)
        else:
            s = _score_set(new_set)
        
        if s > best_score:
            best_score = s
            best_set = new_set
            max_val = max(best_set)
            search_radius = max(max_val * 2, 20)
    
    return best_set, best_score


def construct():
    """Construct a finite set U of nonnegative integers containing 0
    to maximize the sumset/difference-set exponent lower bound."""
    
    # Start with a good seed
    best_set = [0, 1, 3]
    best_score = _score_set(best_set)
    
    # Phase 1: Mixed-radix search (structural)
    if _remaining() > 5:
        mr_set, mr_score = _random_mixed_radix_search(min(30, _remaining() - 5))
        if mr_score > best_score:
            best_score = mr_score
            best_set = mr_set
    
    # Phase 2: Local search from the best found so far
    if _remaining() > 10:
        time_budget = min(_remaining() - 10, 60)
        best_set, best_score = _local_search_general(best_set, time_budget)
    
    # Phase 3: Try more mixed-radix structures with larger radices
    if _remaining() > 15:
        rng = random.Random(777)
        deadline = time.time() + min(_remaining() - 15, 30)
        
        for total_elements in range(20, 500):
            if time.time() > deadline:
                break
            
            for trial in range(20):
                if time.time() > deadline:
                    break
                
                n_factors = rng.randint(2, 4)
                radices = []
                remaining = total_elements
                for i in range(n_factors):
                    if i == n_factors - 1:
                        radices.append(remaining)
                    else:
                        r = max(2, rng.randint(2, max(2, remaining // 2)))
                        radices.append(r)
                        remaining //= r
                
                if remaining < 1:
                    continue
                
                digit_lists = []
                for r in radices:
                    n_digits = rng.randint(2, min(r, 8))
                    digits = [0] + rng.sample(range(1, r), min(n_digits - 1, r - 1))
                    digit_lists.append(sorted(digits))
                
                U = _mixed_radix_set(radices, digit_lists)
                if len(U) > 5000:
                    continue
                
                if max(U) <= 20000:
                    s = _fft_score(U)
                else:
                    s = _score_set(U)
                
                if s > best_score:
                    best_score = s
                    best_set = U
    
    # Final local search
    if _remaining() > 10:
        time_budget = min(_remaining() - 10, 40)
        best_set, best_score = _local_search_general(best_set, time_budget)
    
    # Ensure validity
    best_set = sorted(set(best_set))
    if 0 not in best_set:
        best_set.insert(0, 0)
    if len(best_set) < 2:
        best_set = [0, 1, 3]
    
    return best_set
