# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 112.0  # Safety margin

    # Generate primes up to 1000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n ** 0.5) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]
    
    primes = sieve(1000)
    n_primes = len(primes)
    primes_list = primes

    # Fast evaluation for a given bound
    def evaluate(residues, bound):
        """Compute largest L <= bound such that counts[1..L] >= 2."""
        if bound < 1:
            return 0
        counts = np.zeros(bound + 1, dtype=np.int32)
        for i, p in enumerate(primes_list):
            r = residues[i]
            if r == 0:
                first = p
            else:
                first = r
            if first >= 1 and first <= bound:
                idx = np.arange(first, bound + 1, p)
                counts[idx] += 1
        
        bad_indices = np.where(counts[1:] < 2)[0]
        if len(bad_indices) == 0:
            return bound
        return int(bad_indices[0])

    # Dynamic Window Search
    initial_bound = 1000
    current_bound = initial_bound
    
    best_L = 0
    best_residues = [0] * n_primes
    
    random.seed(42)
    
    max_restarts = 200
    
    for restart in range(max_restarts):
        if time.time() > deadline - 5.0:
            break
            
        # Initialize current residues
        if best_L > 0 and random.random() < 0.7:
            current = best_residues[:]
            num_perturb = max(1, n_primes // 10)
            for _ in range(num_perturb):
                idx = random.randint(0, n_primes - 1)
                p = primes_list[idx]
                current[idx] = random.randint(0, p - 1)
        else:
            current = [random.randint(0, p - 1) for p in primes_list]
        
        # Compute initial counts and L
        current_L = evaluate(current, current_bound)
        
        if current_L > best_L:
            best_L = current_L
            best_residues = current[:]
            
        # Maintain counts array for incremental updates
        counts = np.zeros(current_bound + 1, dtype=np.int32)
        for i, p in enumerate(primes_list):
            r = current[i]
            if r == 0:
                first = p
            else:
                first = r
            if first >= 1 and first <= current_bound:
                idx = np.arange(first, current_bound + 1, p)
                counts[idx] += 1
        
        # Helper to get L from counts
        def get_L_from_counts(counts, bound):
            bad_indices = np.where(counts[1:] < 2)[0]
            if len(bad_indices) == 0:
                return bound
            return int(bad_indices[0])
        
        # Local Search: Hill Climbing
        steps_in_round = 0
        max_steps_in_round = 5000
        
        while current_L < current_bound and steps_in_round < max_steps_in_round:
            if time.time() > deadline - 2.0:
                break
                
            gap_pos = current_L + 1
            if gap_pos > current_bound:
                break
            
            # Find candidates
            candidates = []
            for i, p in enumerate(primes_list):
                if current[i] != gap_pos % p:
                    candidates.append((p, i))
            
            if not candidates:
                break
                
            candidates.sort(key=lambda x: x[0], reverse=True)
            
            best_candidate = -1
            best_new_L = current_L
            best_new_r = -1
            best_old_r = -1
            
            K = 20
            for p, i in candidates[:K]:
                if time.time() > deadline - 2.0:
                    break
                    
                old_r = current[i]
                new_r = gap_pos % p
                
                if new_r == old_r:
                    continue
                
                # Incremental update
                # Subtract old
                if old_r == 0:
                    first_old = p
                else:
                    first_old = old_r
                if first_old >= 1 and first_old <= current_bound:
                    idx_old = np.arange(first_old, current_bound + 1, p)
                    counts[idx_old] -= 1
                
                # Add new
                if new_r == 0:
                    first_new = p
                else:
                    first_new = new_r
                if first_new >= 1 and first_new <= current_bound:
                    idx_new = np.arange(first_new, current_bound + 1, p)
                    counts[idx_new] += 1
                
                # Get new L
                new_L = get_L_from_counts(counts, current_bound)
                
                if new_L > best_new_L:
                    best_new_L = new_L
                    best_candidate = i
                    best_new_r = new_r
                    best_old_r = old_r
                    # Store the counts state? No, we just keep the best one found.
                    # But we need to revert if it's not the best.
                    # Simpler: try, if better, keep. If not, revert.
                    # But we are looping. We need to revert after each try.
                    # So we revert immediately after checking.
                    # But if we revert, we lose the "best" state.
                    # So we should apply the best change after the loop.
                    # But we need to know which change was best.
                    # So we record it.
                    # However, the counts array is modified.
                    # We must revert the counts array after each trial.
                    
                # Revert counts
                if old_r == 0:
                    first_old = p
                else:
                    first_old = old_r
                if first_old >= 1 and first_old <= current_bound:
                    idx_old = np.arange(first_old, current_bound + 1, p)
                    counts[idx_old] += 1
                
                if new_r == 0:
                    first_new = p
                else:
                    first_new = new_r
                if first_new >= 1 and first_new <= current_bound:
                    idx_new = np.arange(first_new, current_bound + 1, p)
                    counts[idx_new] -= 1
                    
            if best_candidate != -1:
                # Apply the best change
                p = primes_list[best_candidate]
                old_r = best_old_r
                new_r = best_new_r
                
                # Update counts permanently
                if old_r == 0:
                    first_old = p
                else:
                    first_old = old_r
                if first_old >= 1 and first_old <= current_bound:
                    idx_old = np.arange(first_old, current_bound + 1, p)
                    counts[idx_old] -= 1
                
                if new_r == 0:
                    first_new = p
                else:
                    first_new = new_r
                if first_new >= 1 and first_new <= current_bound:
                    idx_new = np.arange(first_new, current_bound + 1, p)
                    counts[idx_new] += 1
                
                current[best_candidate] = new_r
                current_L = best_new_L
                steps_in_round += 1
                
                if current_L > best_L:
                    best_L = current_L
                    best_residues = current[:]
                    
                if current_L >= current_bound - 50:
                    break
            else:
                # No improvement
                if random.random() < 0.1:
                    idx = random.randint(0, n_primes - 1)
                    p = primes_list[idx]
                    old_r = current[idx]
                    new_r = random.randint(0, p - 1)
                    
                    # Update counts
                    if old_r == 0:
                        first_old = p
                    else:
                        first_old = old_r
                    if first_old >= 1 and first_old <= current_bound:
                        idx_old = np.arange(first_old, current_bound + 1, p)
                        counts[idx_old] -= 1
                    
                    if new_r == 0:
                        first_new = p
                    else:
                        first_new = new_r
                    if first_new >= 1 and first_new <= current_bound:
                        idx_new = np.arange(first_new, current_bound + 1, p)
                        counts[idx_new] += 1
                        
                    current[idx] = new_r
                    new_L = get_L_from_counts(counts, current_bound)
                    
                    if new_L >= current_L:
                        current_L = new_L
                        if current_L > best_L:
                            best_L = current_L
                            best_residues = current[:]
                    else:
                        # Revert
                        if new_r == 0:
                            first_new = p
                        else:
                            first_new = new_r
                        if first_new >= 1 and first_new <= current_bound:
                            idx_new = np.arange(first_new, current_bound + 1, p)
                            counts[idx_new] -= 1
                            
                        if old_r == 0:
                            first_old = p
                        else:
                            first_old = old_r
                        if first_old >= 1 and first_old <= current_bound:
                            idx_old = np.arange(first_old, current_bound + 1, p)
                            counts[idx_old] += 1
                            
                        current[idx] = old_r
                else:
                    break
            
        # Expand window
        if current_L >= current_bound - 100:
            current_bound = min(current_bound * 2, 100000)

    # Final evaluation
    if time.time() < deadline - 1.0:
        final_bound = max(current_bound, 10000)
        final_L = evaluate(best_residues, final_bound)
        if final_L > best_L:
            best_L = final_L

    return [int(r) for r in best_residues]
# EVOLVE-BLOCK-END
