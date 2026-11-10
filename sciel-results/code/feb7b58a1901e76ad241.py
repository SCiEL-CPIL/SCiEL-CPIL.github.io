"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations, product
import random
import time

def _compute_scores(A):
    """Compute max(|A+A|, |A*A|) for a set A of distinct positive integers."""
    A = list(A)
    n = len(A)
    sums = set()
    prods = set()
    # Optimization: use local variables for speed
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            aj = A[j]
            sums.add(ai + aj)
            prods.add(ai * aj)
    return max(len(sums), len(prods))

def _generate_geometric_cores():
    """
    Generate candidate core sets based on geometric progressions.
    We look for bases b and exponent ranges [start, end] such that 
    b^start ... b^end provides a good skeleton.
    Since we need 20 elements, we consider bases that allow 
    enough spread or density.
    """
    cores = []
    
    # Base 2 is special because it has many powers.
    # 2^0=1, ..., 2^13=8192. That's 14 powers.
    # We need 20 elements, so we must perturb or use a different base.
    # Base 3: 3^0=1, ..., 3^8=6561. That's 9 powers.
    # Base 4: 4^0=1, ..., 4^6=4096. That's 7 powers.
    # Base 5: 5^0=1, ..., 5^5=3125. That's 6 powers.
    # Base 10: 10^0=1, ..., 10^4=10000. That's 5 powers.
    
    # The directive suggests A = {b^e_i} then A' = {b^e_i * k_i}.
    # This implies we start with a geometric skeleton and add multipliers.
    
    # Strategy:
    # 1. Pick a base b.
    # 2. Pick a set of exponents E of size <= 20.
    # 3. If |E| < 20, we need to "split" some elements by multiplying by k_i.
    #    Or, we pick 20 exponents from a larger range if available.
    #    For b=2, max exp is 13. We can pick 20 exponents? No, only 14 distinct powers.
    #    So for b=2, we MUST use multipliers to get to 20 distinct values.
    #    For b=10, max exp is 4. We MUST use multipliers.
    
    # Let's generate cores for bases 2, 3, 4, 5, 10.
    
    # For each base, we will generate a set of 20 "slots" (power, multiplier).
    # The power is b^e. The multiplier k is small.
    # To maximize structural alignment, we want the multipliers to be 
    # "aligned" or small.
    
    return [] # Placeholder, we will do this in construct

def _build_candidate_set(base, exponents, multipliers):
    """
    Construct a set from base^e * k.
    Returns None if invalid (duplicates or > 10000).
    """
    A = []
    seen = set()
    for e, k in zip(exponents, multipliers):
        val = (base ** e) * k
        if val > 10000 or val <= 0:
            return None
        if val in seen:
            return None
        seen.add(val)
        A.append(val)
    if len(A) != 20:
        return None
    return sorted(A)

def _refine_set(A, max_iter=2000):
    """
    Local search refinement.
    Moves:
    1. Replace a value with a nearby value.
    2. Swap two values with nearby values.
    3. Specifically, if we track the structure (base, e, k), we can modify e or k.
       But since we lose structure after initial build, we do generic local search
       but biased towards "smooth" numbers (products of small primes) to maintain
       the geometric spirit.
    """
    A = list(A)
    best_score = _compute_scores(A)
    best_A = A[:]
    
    # Precompute neighbors? No, too expensive.
    # We will use a random walk with acceptance.
    
    # To speed up, we can check only if the change might reduce the score.
    # But _compute_scores is fast enough for ~20 elements.
    
    for _ in range(max_iter):
        # Pick an index to change
        i = random.randint(0, 19)
        old_val = A[i]
        
        # Generate a new candidate value
        # Bias towards numbers that are "similar" in magnitude or structure
        # 1. Random integer in [1, 10000]
        # 2. Random multiple of a small base
        # 3. Random power of a small base
        
        r = random.random()
        if r < 0.3:
            # Keep magnitude similar
            delta = random.randint(-50, 50)
            new_val = old_val + delta
        elif r < 0.6:
            # Small base power
            b = random.choice([2, 3, 4, 5])
            e = random.randint(0, 13)
            new_val = b ** e
            if new_val > 10000:
                new_val = 1
        else:
            # Random
            new_val = random.randint(1, 10000)
            
        if new_val < 1 or new_val > 10000:
            continue
            
        if new_val in A and new_val != old_val:
            continue
            
        A[i] = new_val
        A.sort()
        
        score = _compute_scores(A)
        if score < best_score:
            best_score = score
            best_A = A[:]
        else:
            A[i] = old_val
            A.sort()
            
    return best_A, best_score

def construct():
    start_time = time.time()
    best_score = float('inf')
    best_set = None
    
    # Define small multipliers to use for perturbation
    # These should be coprime to the base if possible, or just small integers.
    # Small integers: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12
    # To maintain "geometric" feel, we want the k_i to be small.
    
    multipliers_pool = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    
    # We will search over bases and exponent configurations.
    # Since we need 20 distinct values, and max power for base 2 is 14,
    # we must use multipliers for base 2.
    # For base 10, max power is 5, so we MUST use multipliers.
    
    # Approach:
    # 1. Iterate over bases.
    # 2. For each base, generate a set of 20 (exponent, multiplier) pairs.
    #    To ensure diversity and coverage, we can:
    #    - Pick a range of exponents.
    #    - Assign multipliers to fill the set.
    
    # Let's try a structured generation:
    # For base b, let's pick 20 exponents from [0, max_e].
    # If max_e >= 19, we can just use 20 distinct powers.
    # If max_e < 19, we must reuse exponents with different multipliers.
    
    # Base 2: max_e = 13. We need 20 values.
    # We can use exponents 0..13 (14 values) and add 6 more by using
    # smaller exponents with larger multipliers.
    # E.g., 2^0 * 1, 2^0 * 2, 2^0 * 3 ...
    
    # General Algorithm for Candidate Generation:
    # 1. Start with a "skeleton" of 20 exponents.
    #    If base^max_e has M powers, and M >= 20, pick 20 distinct exponents.
    #    If M < 20, we need to "branch" some powers.
    
    # Let's implement a specific search for Base 2 and Base 3, which are most likely to yield good results due to density.
    
    bases_to_try = [2, 3, 4, 5]
    
    for base in bases_to_try:
        # Find max exponent
        max_e = 0
        val = 1
        while val * base <= 10000:
            val *= base
            max_e += 1
        
        # We need 20 slots.
        # Let's create a list of 20 (e, k) pairs.
        # To keep it simple and effective, we will use a "grid" of exponents and multipliers.
        
        # If max_e >= 19:
        #   We can just pick 20 exponents. Multipliers are all 1.
        # Else:
        #   We distribute the 20 slots among the available exponents.
        
        # Strategy for M < 20:
        # Assign roughly 20/M slots to each exponent.
        # For each exponent e, we pick k values from multipliers_pool.
        
        num_powers = max_e + 1
        slots_per_power = 20 // num_powers
        remainder = 20 % num_powers
        
        # We will create a list of exponents to use
        exponents_to_use = []
        for e in range(num_powers):
            count = slots_per_power
            if e < remainder:
                count += 1
            exponents_to_use.extend([e] * count)
        
        # Now we have a list of 20 exponents.
        # We need to assign multipliers such that base^e * k are distinct.
        # To ensure distinctness, for a fixed e, k must be distinct.
        # Also base^e1 * k1 != base^e2 * k2.
        
        # We will try multiple random assignments of multipliers.
        
        # Sort exponents_to_use to group same exponents together
        exponents_to_use.sort()
        
        # We will use a backtracking or random assignment with validation.
        # Given the small size (20), random assignment with validation is fast.
        
        for trial in range(5000):
            if time.time() - start_time > 50:
                break
                
            # Assign multipliers
            # For each unique exponent, we need distinct multipliers.
            # We also need to ensure cross-exponent uniqueness.
            
            # Group indices by exponent
            exp_groups = {}
            for idx, e in enumerate(exponents_to_use):
                if e not in exp_groups:
                    exp_groups[e] = []
                exp_groups[e].append(idx)
            
            multipliers = [0] * 20
            valid = True
            
            for e, indices in exp_groups.items():
                # Pick distinct multipliers for these indices
                # From a pool of small multipliers
                # We want small multipliers to keep values small and "structured"
                # Let's pick from [1, 2, ..., 15]
                available_k = list(range(1, 16))
                random.shuffle(available_k)
                
                if len(indices) > len(available_k):
                    # Not enough small multipliers, extend pool?
                    # Or just fail this trial
                    valid = False
                    break
                    
                for i, idx in enumerate(indices):
                    multipliers[idx] = available_k[i]
            
            if not valid:
                continue
                
            # Check distinctness of values
            A = []
            seen = set()
            for i in range(20):
                e = exponents_to_use[i]
                k = multipliers[i]
                val = (base ** e) * k
                if val > 10000:
                    valid = False
                    break
                if val in seen:
                    valid = False
                    break
                seen.add(val)
                A.append(val)
            
            if not valid:
                continue
                
            A.sort()
            score = _compute_scores(A)
            
            if score < best_score:
                best_score = score
                best_set = A[:]
                
                # Try to refine this promising set
                refined, refined_score = _refine_set(A, max_iter=500)
                if refined_score < best_score:
                    best_score = refined_score
                    best_set = refined
    
    if best_set is None:
        # Fallback
        best_set = list(range(1, 21))
        
    return best_set

# EVOLVE-BLOCK-END
