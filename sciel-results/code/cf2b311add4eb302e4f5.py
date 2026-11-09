"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    # We need to find 30 coefficients in {-1, 0, 1} such that
    # sum(c_i / i) is nonzero and has minimal absolute value.
    # Use local search to minimize |sum|
    
    random.seed(42)
    
    def evaluate(coeffs):
        total = Fraction(0, 1)
        for i, c in enumerate(coeffs, 1):
            if c != 0:
                total += Fraction(c, i)
        return total
    
    # Start with a random configuration
    best_coeffs = None
    best_abs = None
    
    # Try multiple random starts
    for attempt in range(50):
        if attempt == 0:
            # Start with the seed-like configuration
            coeffs = [0] * 28 + [1, -1]
        else:
            # Random initialization
            coeffs = [random.choice([-1, 0, 1]) for _ in range(30)]
        
        current = evaluate(coeffs)
        current_abs = abs(current)
        
        if best_abs is None or current_abs < best_abs:
            best_abs = current_abs
            best_coeffs = coeffs[:]
        
        # Local search: try flipping individual coefficients
        improved = True
        iterations = 0
        while improved and iterations < 200:
            improved = False
            iterations += 1
            for i in range(30):
                old_val = coeffs[i]
                for new_val in [-1, 0, 1]:
                    if new_val == old_val:
                        continue
                    coeffs[i] = new_val
                    new_total = evaluate(coeffs)
                    new_abs = abs(new_total)
                    if new_total != 0 and (best_abs is None or new_abs < best_abs):
                        best_abs = new_abs
                        best_coeffs = coeffs[:]
                        improved = True
                    else:
                        coeffs[i] = old_val
            # Also try swapping pairs
            for i in range(30):
                for j in range(i+1, 30):
                    old_i, old_j = coeffs[i], coeffs[j]
                    for new_i in [-1, 0, 1]:
                        for new_j in [-1, 0, 1]:
                            if new_i == old_i and new_j == old_j:
                                continue
                            coeffs[i] = new_i
                            coeffs[j] = new_j
                            new_total = evaluate(coeffs)
                            new_abs = abs(new_total)
                            if new_total != 0 and (best_abs is None or new_abs < best_abs):
                                best_abs = new_abs
                                best_coeffs = coeffs[:]
                                improved = True
                            else:
                                coeffs[i] = old_i
                                coeffs[j] = old_j
    
    # Verify best is nonzero
    final_val = evaluate(best_coeffs)
    if final_val == 0:
        # Fallback to seed
        return [0] * 28 + [1, -1]
    
    return best_coeffs
# EVOLVE-BLOCK-END
