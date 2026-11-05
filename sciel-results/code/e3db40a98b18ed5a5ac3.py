import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin for 90s limit
    
    best_score = float('inf')
    best_poly = None
    
    def update_best(poly_dict, score):
        nonlocal best_score, best_poly
        if score < best_score:
            best_score = score
            # Convert dict to list
            max_exp = max(poly_dict.keys())
            lst = [0] * (max_exp + 1)
            for e, c in poly_dict.items():
                lst[e] = c
            best_poly = lst

    def get_score(d):
        """Exact count of nonzero terms in square."""
        square = {}
        items = list(d.items())
        for i in range(len(items)):
            ei, ci = items[i]
            for j in range(i, len(items)):
                ej, cj = items[j]
                key = ei + ej
                if i == j:
                    val = ci * ci
                else:
                    val = 2 * ci * cj
                square[key] = square.get(key, 0) + val
        return sum(1 for v in square.values() if v != 0)

    # ====================================================================
    # STRATEGY: Structural Decomposition P(x) = A(x)B(x)C(x)
    #
    # We model the 12-term polynomial as a product of three sparse factors.
    # Each factor has 2 or 3 terms. The total number of terms in P is 12.
    # Possible term counts for (A, B, C): (2, 2, 3), (2, 3, 2), (3, 2, 2).
    #
    # We enumerate small structures for A, B, C (exponents and coefficients)
    # and compute P = A*B*C. Then we evaluate the square of P.
    #
    # To maximize cancellation, we look for factors where A^2, B^2, C^2
    # have small support and where the exponents of A, B, C are chosen such
    # that the cross terms in (ABC)^2 = A^2 B^2 C^2 collide.
    #
    # We also allow randomization of exponents and coefficients to explore
    # the space efficiently.
    # ====================================================================

    # Define a function to generate a random sparse polynomial with k terms
    def random_poly(k, max_exp, coeff_range):
        # Choose k distinct exponents
        exps = random.sample(range(max_exp + 1), k)
        exps.sort()
        coeffs = [random.choice(coeff_range) for _ in range(k)]
        d = {}
        for e, c in zip(exps, coeffs):
            if c != 0:
                d[e] = c
        return d

    def multiply_poly(d1, d2):
        result = {}
        for e1, c1 in d1.items():
            for e2, c2 in d2.items():
                key = e1 + e2
                result[key] = result.get(key, 0) + c1 * c2
        # Remove zeros
        return {k: v for k, v in result.items() if v != 0}

    # Coefficient choices for factors
    coeff_choices = [-2, -1, 1, 2]
    
    # We will try different combinations of term counts for A, B, C
    term_counts = [(2, 2, 3), (2, 3, 2), (3, 2, 2)]
    
    max_exp_limit = 20  # Keep exponents small to ensure degree <= 40
    
    # Iterate through different random seeds to explore the space
    for restart in range(200):
        if time.time() > deadline:
            break
        
        random.seed(1000 + restart)
        
        for k1, k2, k3 in term_counts:
            # Generate random factors A, B, C
            A = random_poly(k1, max_exp_limit, coeff_choices)
            B = random_poly(k2, max_exp_limit, coeff_choices)
            C = random_poly(k3, max_exp_limit, coeff_choices)
            
            # Multiply them
            AB = multiply_poly(A, B)
            P = multiply_poly(AB, C)
            
            # Check if P has exactly 12 nonzero terms
            if len(P) != 12:
                continue
            
            # Check if coefficients are within bounds
            valid = True
            for c in P.values():
                if c < -64 or c > 64:
                    valid = False
                    break
            if not valid:
                continue
            
            # Check if max exponent is within bounds
            if max(P.keys()) > 40:
                continue
            
            # Evaluate the square
            score = get_score(P)
            
            if score < best_score:
                update_best(P, score)
    
    # If no valid polynomial was found, fall back to a simple one
    if best_poly is None:
        # Fallback: 12 consecutive terms with small coefficients
        best_poly = [1] * 12
        # Ensure it meets the criteria (it does)
    
    # Final validation
    final_dict = {i: best_poly[i] for i in range(len(best_poly)) if best_poly[i] != 0}
    if len(final_dict) != 12:
        best_poly = [1] * 12
    else:
        for c in final_dict.values():
            if not (-64 <= c <= 64):
                best_poly = [1] * 12
                break
        max_exp = max(final_dict.keys())
        if max_exp > 40:
            best_poly = [1] * 12

    return best_poly
