"""Sparse squares of twelve-term integer polynomials."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 85.0  # safety margin before 90s kill
    
    # A valid candidate: 12 nonzero terms, coefficients in [-64,64], length<=41
    # We want to minimize nonzero terms in the exact integer square.
    
    # Strategy: Use random restart local search with NumPy-accelerated convolution.
    # Start from random sparse polynomials and locally optimize to reduce square sparsity.
    
    def count_nonzero(coeffs):
        return np.count_nonzero(coeffs)
    
    def poly_square(coeffs):
        """Compute exact integer square using np.convolve (returns int64 array)."""
        return np.convolve(coeffs, coeffs)
    
    def evaluate(coeffs):
        """Return number of nonzero terms in the square."""
        sq = poly_square(coeffs)
        return count_nonzero(sq)
    
    def random_initial(max_deg=40):
        """Generate a random 12-term polynomial with coefficients in [-64,64]."""
        deg = random.randint(11, max_deg)
        coeffs = np.zeros(deg + 1, dtype=np.int64)
        positions = random.sample(range(deg + 1), 12)
        for pos in positions:
            coeffs[pos] = random.randint(1, 64) * (1 if random.random() < 0.5 else -1)
        # Ensure final coefficient is nonzero
        if coeffs[-1] == 0:
            coeffs[-1] = random.choice([1, -1])
        return coeffs
    
    def local_search(coeffs, max_iters=5000):
        """Local search: try flipping signs, changing coefficients, moving terms."""
        best = evaluate(coeffs)
        cur = coeffs.copy()
        cur_score = best
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            move_type = random.random()
            
            if move_type < 0.4:
                # Flip sign of one nonzero term
                nz = np.nonzero(cur)[0]
                if len(nz) == 0:
                    break
                idx = nz[random.randint(0, len(nz) - 1)]
                trial = cur.copy()
                trial[idx] = -trial[idx]
                if trial[-1] == 0:
                    continue
                score = evaluate(trial)
                if score <= cur_score:
                    cur = trial
                    cur_score = score
                    if score < best:
                        best = score
            elif move_type < 0.7:
                # Change magnitude of one nonzero term
                nz = np.nonzero(cur)[0]
                if len(nz) == 0:
                    break
                idx = nz[random.randint(0, len(nz) - 1)]
                trial = cur.copy()
                sign = 1 if trial[idx] > 0 else -1
                new_mag = random.randint(1, 64)
                trial[idx] = sign * new_mag
                if trial[-1] == 0:
                    continue
                score = evaluate(trial)
                if score <= cur_score:
                    cur = trial
                    cur_score = score
                    if score < best:
                        best = score
            elif move_type < 0.9:
                # Move one term to a new position
                nz = np.nonzero(cur)[0]
                if len(nz) < 2:
                    continue
                idx = nz[random.randint(0, len(nz) - 1)]
                val = cur[idx]
                new_pos = random.randint(0, len(cur) - 1)
                if new_pos == idx:
                    continue
                trial = cur.copy()
                trial[idx] = 0
                trial[new_pos] = val
                if trial[-1] == 0:
                    continue
                # Ensure exactly 12 nonzero terms
                if count_nonzero(trial) != 12:
                    continue
                score = evaluate(trial)
                if score <= cur_score:
                    cur = trial
                    cur_score = score
                    if score < best:
                        best = score
            else:
                # Swap two nonzero terms' positions
                nz = np.nonzero(cur)[0]
                if len(nz) < 2:
                    continue
                i1, i2 = random.sample(nz, 2)
                trial = cur.copy()
                trial[i1], trial[i2] = trial[i2], trial[i1]
                if trial[-1] == 0:
                    continue
                score = evaluate(trial)
                if score <= cur_score:
                    cur = trial
                    cur_score = score
                    if score < best:
                        best = score
        
        return cur, best
    
    # Random restart local search
    best_coeffs = None
    best_score = float('inf')
    
    num_restarts = 0
    while time.time() < deadline:
        coeffs = random_initial()
        coeffs, score = local_search(coeffs, max_iters=3000)
        num_restarts += 1
        
        if score < best_score:
            best_score = score
            best_coeffs = coeffs.copy()
        
        # If we find a very good solution, do more local refinement
        if best_score <= 15:
            coeffs, score = local_search(best_coeffs, max_iters=5000)
            if score < best_score:
                best_score = score
                best_coeffs = coeffs.copy()
    
    if best_coeffs is None:
        # Fallback: return a simple valid candidate
        best_coeffs = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=np.int64)
    
    # Convert to list, trim trailing zeros but keep nonzero final coefficient
    # Ensure length <= 41 and exactly 12 nonzero terms
    result = best_coeffs.tolist()
    
    # Trim trailing zeros
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    
    # Pad if needed (shouldn't happen)
    while len(result) < 2:
        result.append(0)
    
    # Ensure exactly 12 nonzero terms
    nz_count = sum(1 for x in result if x != 0)
    if nz_count != 12:
        # Fallback
        result = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    
    # Ensure coefficients in [-64, 64]
    result = [max(-64, min(64, x)) for x in result]
    
    # Ensure final coefficient is nonzero
    if result[-1] == 0:
        result[-1] = 1
    
    return result
# EVOLVE-BLOCK-END
