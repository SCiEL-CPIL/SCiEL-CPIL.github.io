import numpy as np
import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # 1. Generate Primes
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(math.isqrt(n)) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]
    
    primes = sieve(1000)
    n_primes = len(primes)
    primes_np = np.array(primes)
    
    # Dynamic window expansion: start with a moderate window, expand as needed
    # Initial window size
    current_window = 5000
    
    # We need x_mod_p for the current window
    # x_values are 1..current_window
    x_values = np.arange(1, current_window + 1)
    x_mod_p = x_values[:, None] % primes_np[None, :]
    
    # Helper: Compute the current score (L) - first index where count < 2
    def compute_score(residues, x_mod_p, max_check):
        # residues shape: (n_primes,)
        # x_mod_p shape: (max_check, n_primes)
        match = (x_mod_p == residues[None, :])
        counts = match.sum(axis=1)
        invalid = counts < 2
        if not np.any(invalid):
            return max_check
        else:
            first_invalid_idx = np.argmax(invalid)
            return int(first_invalid_idx)
    
    # Initial Solution: random or heuristic?
    # Heuristic: try to cover small numbers well.
    # Let's start with r_p = 1, then improve.
    current_residues = np.ones(n_primes, dtype=np.int64)
    
    # Initial greedy pass to get a decent starting point (similar to seed but faster)
    # We do a quick greedy repair to extend L
    for _ in range(50):
        score = compute_score(current_residues, x_mod_p, current_window)
        if score >= current_window:
            # Expand window
            new_window = min(current_window * 2, 20000)
            if new_window == current_window:
                break
            old_window = current_window
            current_window = new_window
            x_values = np.arange(1, current_window + 1)
            x_mod_p = x_values[:, None] % primes_np[None, :]
            # Continue loop to repair new gaps
            continue
        
        gap_x = score + 1
        gap_idx = score
        
        # Find best single move to fix gap
        match_at_gap = (x_mod_p[gap_idx] == current_residues)
        candidates = np.where(~match_at_gap)[0]
        
        if len(candidates) == 0:
            break
        
        M = (x_mod_p == current_residues[None, :])
        old_counts = M.sum(axis=1)
        
        C = candidates
        R = x_mod_p[gap_idx, C]
        
        x_mod_p_cands = x_mod_p[:, C]
        new_matches_cands = (x_mod_p_cands == R[None, :])
        M_cands = M[:, C]
        
        delta_counts = new_matches_cands.astype(np.int8) - M_cands.astype(np.int8)
        new_counts = old_counts[:, None].astype(np.int8) + delta_counts
        
        invalid = new_counts < 2
        has_gap = invalid.any(axis=0)
        first_gap_idx = np.argmax(invalid, axis=0)
        scores = np.where(has_gap, first_gap_idx, current_window)
        
        best_idx_in_scores = np.argmax(scores)
        best_candidate_idx = C[best_idx_in_scores]
        best_new_score = scores[best_idx_in_scores]
        
        if best_new_score > score:
            current_residues[best_candidate_idx] = R[best_idx_in_scores]
        else:
            break

    best_residues = current_residues.copy()
    best_score = compute_score(best_residues, x_mod_p, current_window)
    
    # Simulated Annealing Phase
    # Temperature schedule
    T = 10.0
    T_min = 0.1
    cooling_rate = 0.995
    
    # Number of residues to change per step
    k_min = 1
    k_max = 5
    
    while time.time() < deadline - 2.0:
        # Dynamic window check
        current_score = compute_score(current_residues, x_mod_p, current_window)
        
        if current_score >= current_window:
            # Expand window
            new_window = min(current_window + 5000, 50000)
            if new_window > current_window:
                current_window = new_window
                x_values = np.arange(1, current_window + 1)
                x_mod_p = x_values[:, None] % primes_np[None, :]
                # Reset temperature slightly if we make big progress? Or keep cooling?
                # Let's keep cooling but allow a small bump if stuck
                continue
        
        if current_score > best_score:
            best_score = current_score
            best_residues = current_residues.copy()
        
        # Generate a random neighbor
        k = random.randint(k_min, k_max)
        # Pick k distinct primes to change
        changed_indices = np.random.choice(n_primes, size=k, replace=False)
        
        old_residues_subset = current_residues[changed_indices].copy()
        
        # New random residues
        for i, idx in enumerate(changed_indices):
            current_residues[idx] = np.random.randint(0, primes[idx])
        
        new_score = compute_score(current_residues, x_mod_p, current_window)
        
        # Acceptance criterion
        delta = new_score - current_score
        if delta > 0 or np.random.random() < np.exp(delta / T):
            # Accept
            pass
        else:
            # Reject
            current_residues[changed_indices] = old_residues_subset
        
        # Cool down
        T = max(T * cooling_rate, T_min)
        
        # Occasionally restart or reset temperature if stuck?
        # Simple heuristic: if no improvement in last 1000 steps, bump T
        # We don't track steps here easily, so we just let it cool.
        
    # Final check
    final_score = compute_score(best_residues, x_mod_p, current_window)
    if final_score > best_score:
        best_score = final_score
        best_residues = current_residues.copy()
        
    return [int(r) for r in best_residues]

np.random.seed(42)
random.seed(42)

if __name__ == "__main__":
    res = construct()
    print(len(res))
