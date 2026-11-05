import numpy as np
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit ** 0.5) + 1):
            if sieve[i]:
                sieve[i * i:limit + 1:i] = [False] * len(sieve[i * i:limit + 1:i])
        return [i for i, is_prime in enumerate(sieve) if is_prime]

    primes = get_primes(1000)
    n_primes = len(primes)
    MAX_L = 100000

    residues = np.zeros(n_primes, dtype=np.int32)
    counts = np.zeros(MAX_L + 1, dtype=np.int8)

    def build_counts():
        counts[:] = 0
        for i, p in enumerate(primes):
            r = residues[i]
            if r == 0:
                counts[p::p] += 1
            else:
                counts[r::p] += 1
        counts[0] = 100

    def find_L(current_L=0):
        if current_L >= MAX_L:
            return MAX_L
        rest = counts[current_L + 1:]
        bad_idx = np.where(rest < 2)[0]
        if len(bad_idx) == 0:
            return MAX_L
        return current_L + bad_idx[0]

    def apply_change(i, new_r):
        p = primes[i]
        old_r = residues[i]
        if old_r == new_r:
            return
        if old_r == 0:
            counts[p::p] -= 1
        else:
            counts[old_r::p] -= 1
        residues[i] = new_r
        if new_r == 0:
            counts[p::p] += 1
        else:
            counts[new_r::p] += 1

    def revert_change(i, old_r):
        p = primes[i]
        new_r = residues[i]
        if old_r == new_r:
            return
        if new_r == 0:
            counts[p::p] -= 1
        else:
            counts[new_r::p] -= 1
        residues[i] = old_r
        if old_r == 0:
            counts[p::p] += 1
        else:
            counts[old_r::p] += 1

    def is_safe_change(i, new_r, L):
        p = primes[i]
        old_r = residues[i]
        if old_r == new_r:
            return True
            
        if old_r == 0:
            old_indices = np.arange(p, L + 1, p)
        else:
            old_indices = np.arange(old_r, L + 1, p)
            
        if len(old_indices) == 0:
            return True

        min_old = np.min(counts[old_indices])
        if min_old >= 3:
            return True

        bad_mask = counts[old_indices] < 3
        if not np.any(bad_mask):
            return True
            
        bad_positions = old_indices[bad_mask]
        
        if new_r == 0:
            if old_r != 0:
                return False
            else:
                return True
        else:
            diffs = (bad_positions - new_r) % p
            if np.any(diffs != 0):
                return False
            if np.any(bad_positions < new_r):
                return False
            return True

    best_L = 0
    best_residues = residues.copy()

    # Multi-Start Greedy
    max_restarts = 50
    
    for restart in range(max_restarts):
        if time.time() > deadline - 5.0:
            break
            
        if restart == 0:
            residues[:] = 0
        else:
            # Perturb from best found so far
            residues[:] = best_residues
            # Randomly change 5-10 residues
            num_changes = random.randint(5, 10)
            for _ in range(num_changes):
                i = np.random.randint(n_primes)
                p = primes[i]
                residues[i] = np.random.randint(p)
        
        build_counts()
        current_L = find_L(0)
        
        if current_L > best_L:
            best_L = current_L
            best_residues = residues.copy()
            
        if best_L >= MAX_L:
            break

        # Greedy Extension Loop
        stuck_count = 0
        while time.time() < deadline - 5.0:
            x = current_L + 1
            if x > MAX_L:
                break
                
            found_fix = False
            perm = np.random.permutation(n_primes)
            
            for i in perm:
                p = primes[i]
                target_r = x % p
                old_r = residues[i]
                
                if old_r == target_r:
                    continue
                    
                if is_safe_change(i, target_r, current_L):
                    apply_change(i, target_r)
                    new_L = find_L(current_L)
                    if new_L > current_L:
                        current_L = new_L
                        if current_L > best_L:
                            best_L = current_L
                            best_residues = residues.copy()
                        found_fix = True
                        stuck_count = 0
                        break
                    else:
                        revert_change(i, old_r)
                        
            if not found_fix:
                stuck_count += 1
                if stuck_count > 100:
                    break

    return best_residues.tolist()
