"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _count_nonzero_square(terms):
    """
    Calculate the number of non-zero terms in the square of a polynomial 
    represented as a list of (coeff, exponent) tuples.
    terms: list of (c, e)
    Returns: count of non-zero coefficients in the square.
    """
    # Use a dictionary to accumulate coefficients of the square
    sq = {}
    n = len(terms)
    for i in range(n):
        c1, e1 = terms[i]
        for j in range(i, n):
            c2, e2 = terms[j]
            e_sum = e1 + e2
            if i == j:
                val = c1 * c1
            else:
                val = 2 * c1 * c2
            sq[e_sum] = sq.get(e_sum, 0) + val
    
    # Count non-zero values
    return sum(1 for v in sq.values() if v != 0)

def _validate_and_count(terms):
    """
    Check if terms form a valid polynomial for the task constraints and return non-zero count of square.
    Constraints:
    - Exactly 12 non-zero terms in the input.
    - Coefficients in [-64, 64], integers.
    - Exponents must be non-negative integers (implied by list index).
    - Max degree <= 40 (length <= 41).
    - Final coefficient (highest degree) must be non-zero.
    """
    if len(terms) != 12:
        return None
    
    # Check for duplicate exponents (should merge, but task says "12 non-zero terms", so exponents must be distinct)
    exps = [e for c, e in terms]
    if len(set(exps)) != 12:
        return None
        
    # Check coefficient bounds
    for c, e in terms:
        if c == 0:
            return None
        if not (-64 <= c <= 64):
            return None
        if e < 0 or e > 40:
            return None
            
    # The "nonzero final coefficient" constraint implies that if we return a list of length L, 
    # the last element must be non-zero. In our representation, the highest exponent term 
    # corresponds to the last non-zero element if we pad to that degree.
    # Since we return a list where index = degree, the length is max_exponent + 1.
    # The constraint "nonzero final coefficient" is automatically satisfied if the term with 
    # max exponent has a non-zero coefficient (which it does, by definition of non-zero terms).
    
    return _count_nonzero_square(terms)

def _random_terms():
    """Generate a random valid set of 12 terms."""
    # Choose 12 distinct exponents from 0..40
    exps = random.sample(range(41), 12)
    random.shuffle(exps) # Randomize order, though we'll sort later
    
    terms = []
    for e in exps:
        # Choose a random non-zero coefficient in [-64, 64]
        # Bias towards small coefficients for better cancellation chances? 
        # Let's try a mix.
        c = random.choice([-1, 1, -2, 2, -3, 3, -5, 5, -8, 8])
        # Occasionally pick a larger one
        if random.random() < 0.2:
            c = random.randint(-64, 64)
            if c == 0:
                c = 1
        terms.append((c, e))
        
    return terms

def _mutate_terms(terms, temperature):
    """
    Mutate a single term in the list.
    Returns a new list of terms.
    """
    new_terms = terms.copy()
    idx = random.randint(0, len(new_terms) - 1)
    c, e = new_terms[idx]
    
    # Mutation type: change coefficient, change exponent, or both
    r = random.random()
    
    if r < 0.5:
        # Change coefficient
        # Small step in coefficient
        delta_c = random.choice([-2, -1, 1, 2])
        new_c = c + delta_c
        # Keep in bounds and non-zero
        if new_c == 0:
            new_c = 1 if c > 0 else -1
        if not (-64 <= new_c <= 64):
            new_c = c
        new_terms[idx] = (new_c, e)
    elif r < 0.8:
        # Change exponent
        # Small step in exponent
        delta_e = random.choice([-2, -1, 1, 2])
        new_e = e + delta_e
        if new_e < 0 or new_e > 40:
            new_e = e
        # Check for collision with existing exponents
        other_exps = [e2 for i, (c2, e2) in enumerate(new_terms) if i != idx]
        if new_e in other_exps:
            # Try a larger jump
            new_e = e + random.randint(-5, 5)
            if new_e < 0 or new_e > 40 or new_e in other_exps:
                new_e = e
        new_terms[idx] = (c, new_e)
    else:
        # Change both
        delta_c = random.choice([-1, 1])
        new_c = c + delta_c
        if new_c == 0:
            new_c = 1
        if not (-64 <= new_c <= 64):
            new_c = c
            
        delta_e = random.choice([-1, 1])
        new_e = e + delta_e
        if new_e < 0 or new_e > 40:
            new_e = e
        other_exps = [e2 for i, (c2, e2) in enumerate(new_terms) if i != idx]
        if new_e in other_exps:
            new_e = e + random.randint(-3, 3)
            if new_e < 0 or new_e > 40 or new_e in other_exps:
                new_e = e
        new_terms[idx] = (new_c, new_e)
        
    return new_terms

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    best_score = 1000
    best_terms = None
    best_result_list = None
    
    # Initialize with a few random starts
    num_starts = 10
    current_terms = []
    current_score = 1000
    
    for s in range(num_starts):
        if time.time() > deadline - 5.0:
            break
            
        # Start with a random configuration
        current_terms = _random_terms()
        current_score = _validate_and_count(current_terms)
        
        if current_score is None:
            continue
            
        if current_score < best_score:
            best_score = current_score
            best_terms = current_terms.copy()
            
        # Local search (Simulated Annealing style, but mostly greedy with occasional jumps)
        # We want to minimize score.
        
        # Temperature schedule: start high, decrease
        temp = 10.0
        
        # Number of steps per start
        # Estimate cost: _count_nonzero_square is O(12^2) = 144 ops.
        # We can do many steps.
        steps = 2000
        
        for step in range(steps):
            if time.time() > deadline - 5.0:
                break
                
            new_terms = _mutate_terms(current_terms, temp)
            new_score = _validate_and_count(new_terms)
            
            if new_score is None:
                # Invalid mutation, revert or skip
                continue
                
            # Acceptance criterion
            if new_score < current_score:
                # Always accept improvement
                current_terms = new_terms
                current_score = new_score
                
                if current_score < best_score:
                    best_score = current_score
                    best_terms = current_terms.copy()
                    
                    # If we hit a very low score, we can stop early?
                    # Min possible is 1 (perfect square) or 2?
                    # If we find 1, we can't do better.
                    if best_score == 1:
                        break
            else:
                # Accept worse with probability exp(-delta/temp)
                delta = current_score - new_score
                # Since we minimize, delta is negative if new is better. 
                # Here new_score > current_score, so delta < 0.
                # prob = exp(delta / temp)
                prob = np.exp(delta / temp)
                if random.random() < prob:
                    current_terms = new_terms
                    current_score = new_score
                    
            # Cool down
            temp *= 0.999
            
            # If we found a very good solution, we can try to restart from there?
            # Or just continue.
            
        # If we found a good solution, we can try to polish it with a more aggressive local search?
        # For now, just move to next start.
        
    if best_terms is None:
        # Fallback: return a simple valid polynomial
        # 12 terms, e.g., 1 + x + ... + x^11
        return [1] * 12
        
    # Convert best_terms to coefficient list
    max_deg = max(e for c, e in best_terms)
    result = [0] * (max_deg + 1)
    for c, e in best_terms:
        result[e] = c
        
    # Ensure length <= 41
    if len(result) > 41:
        # This shouldn't happen because max e <= 40
        pass
        
    return result

# EVOLVE-BLOCK-END
