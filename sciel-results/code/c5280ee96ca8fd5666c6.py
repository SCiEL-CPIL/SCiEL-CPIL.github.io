"""Construct 12 double-root locations to minimize B/(2*pi)."""
import time
import math
import random
import numpy as np
from fractions import Fraction

# We'll use sympy for exact arithmetic as required by the task
# But first, let's think about the structure

# The key insight: we need to find z_1..z_12 > 0, distinct, finite
# such that when we construct g(x) = sum_{k=0}^{25} c_k L_{2k}^{(-1/2)}(x)
# satisfying g(0)=0, g'(0)=1, and g(z_i)=g'(z_i)=0 for all i
# then q(x) = g(x) / (x * prod_i (x-z_i)^2) has a real root r
# with sign change at r±1e-198, and B = max(0, largest such r)
# Score = B/(2*pi), lower is better.

# Strategy: The double roots z_i are where g has both value 0 and derivative 0.
# This means (x-z_i)^2 divides g(x). Also x divides g(x) since g(0)=0.
# So g(x) = x * prod_i (x-z_i)^2 * q(x) where q is a polynomial.
# The degree of g is at most 50 (from L_50).
# deg(x * prod_i (x-z_i)^2) = 1 + 24 = 25
# So q(x) has degree at most 50-25 = 25.

# For B to be small, we want q to have no large positive real roots with sign changes.
# The best case would be if q has no positive real roots at all, giving B=0.
# But that might not be achievable with the Laguerre constraint.

# Let's try a heuristic approach:
# Start with some initial z_i values and see what happens.

# Actually, let's think more carefully. The constraint is that g must be in the span
# of {L_{2k}^{(-1/2)}(x) : k=0..25}. This is a 26-dimensional space.
# We have 26 constraints: g(0)=0, g'(0)=1, and 2*12=24 conditions g(z_i)=g'(z_i)=0.
# So the system is square (26x26).

# The question is: what choice of z_i minimizes B?

# Let me try a different approach. Since we need exact arithmetic and the system must
# be non-singular, let's try to find z_i that make the resulting q(x) have small roots.

# Heuristic: Try to make z_i cluster near 0, which might push the roots of q to be small
# or negative.

# Let's try z_i = i * small_value for i=1..12

# Actually, let me think about what makes the system well-conditioned and what minimizes B.

# For now, let's try several heuristics and pick the best one we can evaluate.

def construct():
    """Return 12 double-root locations."""
    # Deadline guard
    t0 = time.time()
    DEADLINE = 600  # seconds
    SAFETY_MARGIN = 30  # seconds
    
    # Try different configurations of z_i
    # Each configuration is a list of 12 positive distinct floats
    
    candidates = []
    
    # Configuration 1: evenly spaced, starting near 0
    for start in [0.01, 0.1, 0.5, 1.0, 2.0]:
        for spacing in [0.01, 0.1, 0.5, 1.0, 2.0]:
            z = [start + i * spacing for i in range(12)]
            if all(zi > 0 for zi in z) and len(set(z)) == 12:
                candidates.append(z)
    
    # Configuration 2: geometric progression
    for base in [0.1, 0.5, 1.5, 2.0, 3.0]:
        z = [base ** (i+1) for i in range(12)]
        if all(zi > 0 for zi in z) and len(set(z)) == 12:
            candidates.append(z)
    
    # Configuration 3: clustered near a point
    for center in [0.5, 1.0, 2.0, 5.0, 10.0]:
        for spread in [0.1, 0.5, 1.0, 2.0, 5.0]:
            z = [center + (i - 5.5) * spread / 12 for i in range(12)]
            if all(zi > 0 for zi in z) and len(set(z)) == 12:
                candidates.append(z)
    
    # Configuration 4: Original seed
    z_seed = [4.0 * (i + 1) for i in range(12)]
    candidates.append(z_seed)
    
    # Sort candidates by some heuristic - try smaller ones first
    # Actually, let's just try the first few and see which are valid
    
    best_score = float('inf')
    best_z = None
    best_b = None
    
    for z in candidates:
        if time.time() - t0 > DEADLINE - SAFETY_MARGIN:
            break
        
        try:
            b_val = _evaluate_z(z)
            if b_val is not None and b_val >= 0:
                score = b_val / (2 * math.pi)
                if score < best_score:
                    best_score = score
                    best_z = z
                    best_b = b_val
        except Exception:
            continue
    
    # If no valid configuration found, return the seed
    if best_z is None:
        return z_seed
    
    # Return as list of floats
    return [float(zi) for zi in best_z]


def _evaluate_z(z_list):
    """Evaluate B for a given list of z values. Returns B or None if invalid."""
    import sympy
    from sympy import Rational, symbols, expand, diff, solve, LUsolve, Matrix
    
    t0 = time.time()
    DEADLINE = 600
    SAFETY_MARGIN = 30
    
    def check_time():
        if time.time() - t0 > DEADLINE - SAFETY_MARGIN:
            raise TimeoutError("Deadline exceeded")
    
    # Convert z to rationals
    z_rat = [Rational(float(zi)) for zi in z_list]
    
    # Check distinctness and positivity
    if len(set(z_rat)) != 12:
        return None
    if any(zi <= 0 for zi in z_rat):
        return None
    
    # Build the Laguerre polynomials L_{2k}^{(-1/2)}(x) for k=0..25
    x = symbols('x')
    alpha = Rational(-1, 2)
    
    laguerres = []
    for k in range(26):
        d = 2 * k
        L = sympy.genlaguerre(d, alpha, x)
        laguerres.append(L)
    
    check_time()
    
    # Build the system
    # g(x) = sum_{k=0}^{25} c_k * L_{2k}(x)
    # Unknowns: c_0, c_1, ..., c_25
    c_syms = [symbols(f'c_{i}') for i in range(26)]
    
    # g(x)
    g = sum(c_syms[k] * laguerres[k] for k in range(26))
    
    # Constraints:
    # g(0) = 0
    # g'(0) = 1
    # g(z_i) = 0 for i=0..11
    # g'(z_i) = 0 for i=0..11
    
    constraints = []
    constraints.append(g.subs(x, 0))  # g(0) = 0
    constraints.append(diff(g, x).subs(x, 0) - 1)  # g'(0) = 1
    
    for zi in z_rat:
        constraints.append(g.subs(x, zi))  # g(z_i) = 0
        constraints.append(diff(g, x).subs(x, zi))  # g'(z_i) = 0
    
    # Build matrix A and vector b
    # A * c = b where c = [c_0, ..., c_25]
    # Move all to one side: sum c_k * (coefficient) = rhs
    
    A = Matrix(26, 26, 0)
    b = Matrix(26, 1, 0)
    
    for row_idx, constraint in enumerate(constraints):
        # Expand constraint
        constraint_expanded = expand(constraint)
        for col_idx, c_sym in enumerate(c_syms):
            coeff = constraint_expanded.coeff(c_sym)
            A[row_idx, col_idx] = coeff
        # Constant term (rhs)
        const_term = constraint_expanded.coeff(x, 0)
        # Actually, we need the part not involving any c_sym
        # Let me restructure: constraint = sum c_k * A[row,k] + const = 0
        # So sum c_k * A[row,k] = -const
        # But const here is the part of constraint_expanded that doesn't involve any c_sym
        # Since each constraint is linear in c's, we can extract the constant part
        
        # Get the constant part (part not multiplied by any c_sym)
        # This is the part that remains after setting all c's to 0
        const_val = constraint_expanded.subs(c_syms, [0]*26)
        b[row_idx] = -const_val
        # And A[row, col] is the coefficient of c_col
        for col_idx, c_sym in enumerate(c_syms):
            A[row_idx, col_idx] = constraint_expanded.coeff(c_sym)
    
    check_time()
    
    # Check if system is singular
    if A.det() == 0:
        return None
    
    # Solve
    c_sol = A.LUsolve(b)
    
    check_time()
    
    # Build g with solved coefficients
    g_sol = sum(c_sol[k] * laguerres[k] for k in range(26))
    g_sol = expand(g_sol)
    
    # Build denominator: x * prod_i (x - z_i)^2
    denom = x
    for zi in z_rat:
        denom = denom * (x - zi)**2
    denom = expand(denom)
    
    # q(x) = g(x) / denom
    # Do polynomial division
    q, remainder = sympy.div(g_sol, denom, x)
    
    check_time()
    
    # Check remainder is zero (should be if our setup is correct)
    if remainder != 0:
        # The division doesn't come out clean - this shouldn't happen if
        # g has double roots at z_i and g(0)=0
        # But due to the Laguerre constraint, it might not divide evenly
        # In that case, the problem statement says "an exact polynomial division"
        # which implies it should work out
        return None
    
    # Find real roots of q
    real_roots = sympy.real_roots(q)
    
    check_time()
    
    # Count roots with sign change at r ± 1e-198
    # 1e-198 as a rational
    eps = Rational(1, 10)**198
    
    counted_roots = []
    for r in real_roots:
        # Check sign at r - eps and r + eps
        val_left = q.subs(x, r - eps)
        val_right = q.subs(x, r + eps)
        
        if val_left * val_right < 0:
            # Strictly opposite signs
            counted_roots.append(r)
    
    # B = max(0, largest counted r)
    if counted_roots:
        B = max(Rational(0), max(counted_roots))
    else:
        B = Rational(0)
    
    return float(B)


# Pre-compute some candidate configurations at module load for speed
# Actually, let's just rely on the construct function

if __name__ == "__main__":
    # Test
    z = construct()
    print(f"Constructed z: {z}")
