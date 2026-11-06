import numpy as np
import time
import math

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
    
    # The range of integers to check.
    # We check up to 5000 to give the greedy algorithm room to push the frontier.
    # 5000 * 168 is ~840k elements, which is small for NumPy.
    MAX_CHECK = 5000
    x_values = np.arange(1, MAX_CHECK + 1)
    
    # Precompute x mod p for all x and p.
    # Shape: (MAX_CHECK, n_primes)
    x_mod_p = x_values[:, None] % primes_np[None, :]
    
    # Helper: Compute the current score (L)
    def compute_score(residues):
        match = (x_mod_p == residues[None, :])
        counts = match.sum(axis=1)
        invalid = counts < 2
        if not np.any(invalid):
            return MAX_CHECK
        else:
            first_invalid_idx = np.argmax(invalid)
            return int(first_invalid_idx)
    
    # Strategy: Multi-start Greedy Hill Climbing
    # We will run several independent "workers".
    # Each worker starts with a random set of residues.
    # Each worker performs a greedy loop:
    #   1. Find first gap L+1.
    #   2. Evaluate all single-residue changes that fix the gap (or improve coverage at gap).
    #   3. Pick the best change (maximizes new score).
    #   4. If no improvement, stop or randomize.
    
    # To make this efficient, we vectorize the evaluation of all candidate moves for a given gap.
    
    def run_greedy_worker(seed_offset, max_steps=2000):
        # Initialize with random residues
        rng = np.random.RandomState(seed_offset)
        residues = np.array([rng.randint(0, p) for p in primes], dtype=np.int64)
        
        current_score = compute_score(residues)
        best_score = current_score
        best_residues = residues.copy()
        
        # Greedy Loop
        for step in range(max_steps):
            if time.time() > deadline:
                break
            if current_score >= MAX_CHECK:
                break
            
            gap_x = current_score + 1
            gap_idx = current_score  # 0-based index in x_values
            
            # Find primes that do NOT currently match gap_x
            match_at_gap = (x_mod_p[gap_idx] == residues)
            candidates = np.where(~match_at_gap)[0]
            
            if len(candidates) == 0:
                # All primes match, but count < 2? Impossible for n_primes >= 2.
                # Or count >= 2? Then gap_x is covered, contradiction.
                # So if we are here, it's a logic error or MAX_CHECK edge case.
                break
            
            # Vectorized evaluation of all candidate moves
            C = candidates
            R = x_mod_p[gap_idx, C]  # New residue for each candidate
            
            # Get current match matrix M
            M = (x_mod_p == residues[None, :])
            old_counts = M.sum(axis=1)
            
            # Compute delta counts for all candidates
            # x_mod_p_cands: (MAX_CHECK, num_cands)
            x_mod_p_cands = x_mod_p[:, C]
            
            # new_matches_cands: (MAX_CHECK, num_cands)
            new_matches_cands = (x_mod_p_cands == R[None, :])
            
            # M_cands: (MAX_CHECK, num_cands)
            M_cands = M[:, C]
            
            # delta_counts: (MAX_CHECK, num_cands)
            delta_counts = new_matches_cands.astype(np.int8) - M_cands.astype(np.int8)
            
            # new_counts: (MAX_CHECK, num_cands)
            new_counts = old_counts[:, None].astype(np.int8) + delta_counts
            
            # Find scores for each candidate
            invalid = new_counts < 2
            has_gap = invalid.any(axis=0)
            first_gap_idx = np.argmax(invalid, axis=0)
            scores = np.where(has_gap, first_gap_idx, MAX_CHECK)
            
            # Select best candidate
            best_idx_in_scores = np.argmax(scores)
            best_candidate_idx = C[best_idx_in_scores]
            best_new_score = scores[best_idx_in_scores]
            
            if best_new_score > current_score:
                # Apply move
                residues[best_candidate_idx] = R[best_idx_in_scores]
                current_score = best_new_score
                
                if current_score > best_score:
                    best_score = current_score
                    best_residues = residues.copy()
            else:
                # Stuck. No single move improves the score.
                # Try a random perturbation to escape local minimum.
                # Change 1-3 random residues to random values.
                num_perturb = rng.randint(1, 4)
                for _ in range(num_perturb):
                    p_idx = rng.randint(0, n_primes)
                    old_r = residues[p_idx]
                    new_r = rng.randint(0, primes[p_idx])
                    if new_r != old_r:
                        residues[p_idx] = new_r
                        new_score = compute_score(residues)
                        if new_score >= current_score:
                            current_score = new_score
                            if current_score > best_score:
                                best_score = current_score
                                best_residues = residues.copy()
                        else:
                            # Revert if it gets worse
                            residues[p_idx] = old_r
                
                # If we are still stuck after perturbation, break to save time for other workers
                # Or continue with a few more steps?
                # Let's break to allow other workers to run.
                break
                
        return best_score, best_residues

    # Run multiple workers
    # How many? Depends on time.
    # Let's run 10 workers, each with a different seed.
    # We'll track the global best.
    
    global_best_score = 0
    global_best_residues = None
    
    num_workers = 10
    base_seed = 42
    
    for i in range(num_workers):
        if time.time() > deadline - 5.0:
            break
        
        seed = base_seed + i * 1000
        score, residues = run_greedy_worker(seed)
        
        if score > global_best_score:
            global_best_score = score
            global_best_residues = residues.copy()
        
        # Early exit if we reached the max possible score
        if global_best_score >= MAX_CHECK:
            break

    # If we didn't find a good solution, try a few more with different strategies?
    # For now, return the best found.
    
    if global_best_residues is None:
        # Fallback: all zeros
        return [0] * n_primes
        
    return [int(r) for r in global_best_residues]

if __name__ == "__main__":
    res = construct()
    print(len(res))
