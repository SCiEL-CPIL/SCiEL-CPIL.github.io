"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix
import time

def construct():
    """
    Construct a subset A of [0, 128] such that every integer in [128, 256]
    is a sum of two elements of A (unordered pairs, equal summands allowed).
    Minimize the largest representation multiplicity.
    
    Approach:
    1. Use a greedy/local search hybrid to find a set A with low max multiplicity.
    2. The seed program returns [64, 128], which has max multiplicity 64 (for sum 128: 64+64... wait no).
    
    Let's analyze the seed: A = {64, 65, ..., 128}.
    Sums range from 64+64=128 to 128+128=256.
    For sum s in [128, 256], the number of pairs (a,b) in A with a+b=s, a<=b:
    For s=128: only (64,64) -> 1
    For s=129: (64,65) -> 1
    ...
    For s=192: (64,128), (65,127), ..., (96,96) -> let's count: a from 64 to 96, b=s-a.
    a must be in [64,128] and b in [64,128] and a<=b.
    b = s - a >= a => s >= 2a => a <= s/2.
    Also a >= 64, b = s-a <= 128 => a >= s-128.
    So a in [max(64, s-128), min(128, s//2)].
    
    For s=192: a in [max(64, 64), min(128, 96)] = [64, 96]. Count = 96-64+1 = 33.
    For s=128: a in [max(64, 0), min(128, 64)] = [64, 64]. Count = 1.
    For s=256: a in [max(64, 128), min(128, 128)] = [128, 128]. Count = 1.
    
    So the seed has max multiplicity 33. We need to do better.
    
    A better approach: Use a set A that is more "spread out" or structured to balance multiplicities.
    The minimum possible max multiplicity is at least ceil(129 / max_possible_pairs_per_element).
    
    Let's try a local search approach:
    1. Start with a reasonable initial set.
    2. Iteratively remove elements that cause high multiplicity peaks and add elements that help cover under-covered sums.
    """
    n = 128
    target_low = 128
    target_high = 256
    targets = list(range(target_low, target_high + 1))
    
    # Initialize with a symmetric set around 64
    # Try A = {0, 1, ..., 63} U {65, 66, ..., 128} or similar
    # Actually, let's start with the seed and improve
    
    A = set(range(64, 129))
    
    def compute_multiplicities(A_set):
        """Compute the number of representations for each target sum."""
        mults = {s: 0 for s in targets}
        A_list = sorted(A_set)
        for i, a in enumerate(A_list):
            for j in range(i, len(A_list)):
                b = A_list[j]
                s = a + b
                if target_low <= s <= target_high:
                    mults[s] += 1
        return mults
    
    def max_mult(A_set):
        mults = compute_multiplicities(A_set)
        return max(mults.values())
    
    best_A = A
    best_score = max_mult(A)
    
    # Local search: try removing and adding elements
    import random
    random.seed(42)
    
    for iteration in range(500):
        A_list = sorted(A)
        
        # Try removing an element
        if len(A) > 10:
            a_to_remove = random.choice(A_list)
            A_new = A - {a_to_remove}
            # Check if all targets are still covered
            mults = compute_multiplicities(A_new)
            if all(m > 0 for m in mults.values()):
                new_score = max(mults.values())
                if new_score < best_score:
                    best_score = new_score
                    best_A = A_new
                    A = A_new
                    continue
        
        # Try adding an element
        candidates = [x for x in range(n + 1) if x not in A]
        if candidates:
            a_to_add = random.choice(candidates)
            A_new = A | {a_to_add}
            new_score = max_mult(A_new)
            if new_score < best_score:
                best_score = new_score
                best_A = A_new
                A = A_new
                continue
        
        # Try swapping: remove one, add one
        if len(A) > 10 and candidates:
            a_to_remove = random.choice(A_list)
            a_to_add = random.choice(candidates)
            A_new = (A - {a_to_remove}) | {a_to_add}
            mults = compute_multiplicities(A_new)
            if all(m > 0 for m in mults.values()):
                new_score = max(mults.values())
                if new_score < best_score:
                    best_score = new_score
                    best_A = A_new
                    A = A_new
                    continue
    
    # Convert to sorted list
    return sorted(best_A)
# EVOLVE-BLOCK-END
