import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit ** 0.5) + 1):
            if sieve[i]:
                for j in range(i * i, limit + 1, i):
                    sieve[j] = False
        return [i for i in range(2, limit + 1) if sieve[i]]
    
    primes = get_primes(1000)
    n_primes = len(primes)
    L_max = 3000
    
    def compute_indices(p, r):
        """Return numpy array of positions in 1..L_max where x ≡ r (mod p)."""
        if r == 0:
            first = p
        else:
            first = r
        if first > L_max:
            return np.array([], dtype=np.int32)
        return np.arange(first, L_max + 1, p, dtype=np.int32)
    
    def evaluate(residues):
        coverage = np.zeros(L_max + 1, dtype=np.int32)
        for i, p in enumerate(primes):
            r = residues[i]
            idx = compute_indices(p, r)
            if len(idx) > 0:
                coverage[idx] += 1
        below = np.where(coverage[1:] < 2)[0]
        if len(below) == 0:
            return L_max
        return below[0]
    
    # Initialize
    residues = np.array([((i + 2) // 2) % primes[i] for i in range(n_primes)], dtype=np.int64)
    best_score = evaluate(residues)
    best_residues = residues.copy()
    
    # Precompute indices for all (p, r) pairs? Too much memory.
    # Instead, use incremental updates.
    
    improved = True
    max_passes = 50
    for _pass in range(max_passes):
        if time.time() > deadline - 5.0:
            break
        improved = False
        for i in range(n_primes):
            if time.time() > deadline - 5.0:
                break
            p = primes[i]
            current_r = residues[i]
            # Compute current coverage
            coverage = np.zeros(L_max + 1, dtype=np.int32)
            for j, pj in enumerate(primes):
                rj = residues[j]
                if rj == 0:
                    first = pj
                else:
                    first = rj
                if first <= L_max:
                    coverage[np.arange(first, L_max + 1, pj, dtype=np.int32)] += 1
            
            # Try changing residue i
            old_idx = compute_indices(p, current_r)
            best_r = current_r
            best_local_score = best_score
            
            for r in range(p):
                if r == current_r:
                    continue
                if time.time() > deadline - 5.0:
                    break
                new_idx = compute_indices(p, r)
                # Delta: -1 at old_idx, +1 at new_idx
                coverage[old_idx] -= 1
                coverage[new_idx] += 1
                
                below = np.where(coverage[1:] < 2)[0]
                score = L_max if len(below) == 0 else below[0]
                
                if score > best_local_score:
                    best_local_score = score
                    best_r = r
                else:
                    # Revert
                    coverage[new_idx] -= 1
                    coverage[old_idx] += 1
            
            # Apply best
            if best_r != current_r:
                # Revert current, apply best
                coverage[old_idx] -= 1
                new_idx_best = compute_indices(p, best_r)
                coverage[new_idx_best] += 1
                residues[i] = best_r
                best_residues[i] = best_r
                best_score = best_local_score
                improved = True
            else:
                # Ensure coverage is back to original
                # It should be, since we reverted all non-best changes
                pass
        
        if not improved:
            break
    
    return [int(r) for r in best_residues]
