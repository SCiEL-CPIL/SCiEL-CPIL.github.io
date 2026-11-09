"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import itertools
import time

# EVOLVE-BLOCK-START
import numpy as np
import random

# Precompute FFT size for fast convolution
_FFT_SIZE = 128  # power of 2, enough for degree up to 40 (square has degree up to 80)

def _fft_square(c):
    """Fast square using FFT. Returns list of integer coefficients."""
    n = len(c)
    size = _FFT_SIZE
    while size < 2*n - 1:
        size *= 2
    # Pad and FFT
    c_pad = np.zeros(size, dtype=np.float64)
    c_pad[:n] = c
    cf = np.fft.fft(c_pad)
    cf_sq = cf * cf
    result = np.fft.ifft(cf_sq).real
    # Round to nearest integer
    result = np.round(result).astype(np.int64)
    # Trim trailing zeros
    nz = np.nonzero(result)[0]
    if len(nz) == 0:
        return [0]
    last = nz[-1]
    return result[:last+1].tolist()

def _nnz(c):
    return sum(1 for x in c if x != 0)

def _evaluate_poly(poly):
    """Evaluate a polynomial candidate: return (nnz_of_square, is_valid)."""
    if len(poly) > 41:
        return float('inf'), False
    if len(poly) == 0 or poly[-1] == 0:
        return float('inf'), False
    if sum(1 for x in poly if x != 0) != 12:
        return float('inf'), False
    if any(abs(x) > 64 for x in poly):
        return float('inf'), False
    sq = _fft_square(poly)
    return _nnz(sq), True

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s
    
    best_sq_nnz = float('inf')
    best_poly = [1]*12
    
    random.seed(42)
    
    # Phase 1: Try structured candidates
    candidates = [
        [1]*12,
        [1,-1,1,-1,1,-1,-1,1,-1,1,-1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1],
        [1,2,1,1,1,1,1,1,1,1,2,1],
        [1,1,1,1,1,1,1,1,1,1,1,1],
    ]
    
    for c in candidates:
        if time.time() > deadline:
            break
        nnz_sq, valid = _evaluate_poly(c)
        if valid and nnz_sq < best_sq_nnz:
            best_sq_nnz = nnz_sq
            best_poly = c[:]
    
    # Phase 2: Random search on palindromic degree-11 polynomials
    non_zero_vals = list(range(-8, 9))
    non_zero_vals.remove(0)
    
    attempts = 0
    max_attempts = 500000
    
    while time.time() < deadline and attempts < max_attempts:
        attempts += 1
        if attempts % 50000 == 0 and time.time() > deadline:
            break
        
        # Pick 6 nonzero coefficients
        free = [random.choice(non_zero_vals) for _ in range(6)]
        # Build palindromic degree 11
        poly = free + [free[5], free[4], free[3], free[2], free[1], free[0]]
        
        nnz_sq, valid = _evaluate_poly(poly)
        if valid and nnz_sq < best_sq_nnz:
            best_sq_nnz = nnz_sq
            best_poly = poly[:]
    
    # Phase 3: Local search - perturb best_poly coefficients by ±1, ±2
    if time.time() < deadline - 10:
        local_attempts = 0
        max_local = 200000
        while time.time() < deadline - 10 and local_attempts < max_local:
            local_attempts += 1
            if local_attempts % 20000 == 0 and time.time() > deadline - 10:
                break
            
            # Pick a random position and perturbation
            pos = random.randint(0, len(best_poly) - 1)
            delta = random.choice([-2, -1, 1, 2])
            new_val = best_poly[pos] + delta
            
            if abs(new_val) > 64:
                continue
            
            poly = best_poly[:]
            poly[pos] = new_val
            
            # Trim trailing zeros
            while len(poly) > 1 and poly[-1] == 0:
                poly.pop()
            
            nnz_sq, valid = _evaluate_poly(poly)
            if valid and nnz_sq < best_sq_nnz:
                best_sq_nnz = nnz_sq
                best_poly = poly[:]
    
    # Ensure output is valid
    while len(best_poly) > 1 and best_poly[-1] == 0:
        best_poly.pop()
    
    if len(best_poly) == 0 or best_poly[-1] == 0:
        best_poly = [1]*12
    
    if sum(1 for x in best_poly if x != 0) != 12:
        best_poly = [1]*12
    
    if len(best_poly) > 41:
        best_poly = best_poly[:41]
    
    return best_poly
# EVOLVE-BLOCK-END
