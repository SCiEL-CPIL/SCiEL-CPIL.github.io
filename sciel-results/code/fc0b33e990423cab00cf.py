"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def _check_pairwise_coprime(values):
    """Check if all distinct values are pairwise coprime."""
    vals = list(set(values))
    for i in range(len(vals)):
        for j in range(i + 1, len(vals)):
            if gcd(vals[i], vals[j]) != 1:
                return False
    return True

def _evaluate(A, B):
    """Return (cardinality, is_valid) for the sumset A+B."""
    sums = [a + b for a in A for b in B]
    distinct = set(sums)
    if not _check_pairwise_coprime(distinct):
        return (0, False)
    return (len(distinct), True)

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin before 60s kill
    
    # Initialize with a known valid solution (the seed)
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_score = 5
    
    # Verify the seed
    score, valid = _evaluate(best_A, best_B)
    if valid:
        best_score = score
    else:
        best_score = 0
        best_A = [1, 2, 3]
        best_B = [4, 5, 6]

    # If we already found the maximum possible score (9), we can return early
    # but we'll allow a short search to ensure no issues, or just return.
    # However, the task is to maximize, and 9 is the max.
    # We will still run a short search to be safe against any subtle evaluation differences,
    # though the logic is standard.
    
    rng = random.Random(42)
    range_ = range(1, 129)
    
    iteration = 0
    while time.time() < deadline:
        iteration += 1
        
        # If we have reached the theoretical maximum, we can break early
        if best_score >= 9:
            break
            
        # Random search
        A = list(rng.sample(range_, 3))
        B = list(rng.sample(range_, 3))
        
        score, valid = _evaluate(A, B)
        if valid and score > best_score:
            best_score = score
            best_A = A[:]
            best_B = B[:]
            
        # Local perturbation
        if iteration % 5 == 0 and best_score < 9:
            # Perturb A
            A2 = best_A[:]
            idx = rng.randint(0, 2)
            A2[idx] = rng.randint(1, 128)
            if len(set(A2)) == 3:
                score2, valid2 = _evaluate(A2, best_B)
                if valid2 and score2 > best_score:
                    best_score = score2
                    best_A = A2[:]
                    
            # Perturb B
            B2 = best_B[:]
            idx = rng.randint(0, 2)
            B2[idx] = rng.randint(1, 128)
            if len(set(B2)) == 3:
                score3, valid3 = _evaluate(best_A, B2)
                if valid3 and score3 > best_score:
                    best_score = score3
                    best_B = B2[:]
                    
    return [best_A, best_B]
# EVOLVE-BLOCK-END
