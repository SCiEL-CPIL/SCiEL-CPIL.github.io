"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import sympy as sp

# EVOLVE-BLOCK-START

def construct():
    """
    Returns 12 double-root locations z_1..z_12 for the Laguerre double-root problem.
    
    Strategy: Use the zeros of the generalized Laguerre polynomial L_12^(alpha)(x)
    with alpha = -1/2 as the initial guess, then refine using a simple gradient-free
    optimization to minimize B = max(0, largest real root of q(x)).
    
    The key insight is that using orthogonal polynomial nodes gives a well-conditioned
    interpolation problem, and the resulting q(x) should have small positive roots.
    """
    start_time = time.time()
    DEADLINE = 600  # seconds
    MARGIN = 30     # safety margin
    
    # Step 1: Get zeros of L_12^(-1/2)(x) as initial placement
    # These are the nodes for Gaussian quadrature with weight x^(-1/2) exp(-x)
    # They are all positive and distinct.
    
    # Use numpy's roots for a polynomial representation, or use scipy
    # For L_n^(alpha)(x), we can use the explicit formula or scipy.special.roots_genlaguerre
    try:
        from scipy.special import roots_genlaguerre
        # roots_genlaguerre(n, alpha) returns nodes and weights for L_n^(alpha)
        # alpha = -1/2, n = 12
        nodes, weights = roots_genlaguerre(12, -0.5)
        z_init = sorted([float(x) for x in nodes])
    except:
        # Fallback: use equally spaced points on a reasonable scale
        z_init = [0.5 * (i + 1) for i in range(12)]
    
    # Verify z_init is valid
    if not all(np.isfinite(z) and z > 0 for z in z_init):
        z_init = [0.5 * (i + 1) for i in range(12)]
    
    # Check pairwise distinctness
    for i in range(len(z_init)):
        for j in range(i+1, len(z_init)):
            if abs(z_init[i] - z_init[j]) < 1e-10:
                z_init[j] += 1e-6
    
    best_z = z_init[:]
    best_score = float('inf')
    
    # Step 2: Evaluate the score for the initial placement
    def evaluate_score(z_list):
        """
        Given a list of 12 positive distinct doubles, compute the score B/(2*pi).
        Returns (score, valid_flag).
        """
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - MARGIN:
            return float('inf'), False
        
        try:
            # Convert to rationals
            z_rationals = [sp.Rational(float(z)) for z in z_list]
            z_sorted = sorted(z_rationals)
            
            # Build the system
            x = sp.Symbol('x')
            alpha = sp.Rational(-1, 2)
            
            # Generalized Laguerre polynomials L_0, L_2, ..., L_50
            # We need L_k^(alpha)(x) for k = 0, 2, 4, ..., 50
            laguerre_pols = []
            for k in range(0, 51, 2):
                Lk = sp.genlaguerre(k, alpha, x)
                laguerre_pols.append(Lk)
            
            # Unknowns: c_0, c_1, ..., c_25
            c = sp.symbols('c0:26')
            
            # g(x) = sum_{k=0}^{25} c_k * L_{2k}(x)
            g = sum(c[k] * laguerre_pols[k] for k in range(26))
            
            # Conditions:
            # g(0) = 0
            # g'(0) = 1
            # For each z_i: g(z_i) = 0 and g'(z_i) = 0
            # Total: 2 + 24 = 26 equations
            
            eqs = []
            eqs.append(g.subs(x, 0))  # g(0) = 0
            eqs.append(sp.diff(g, x).subs(x, 0) - 1)  # g'(0) = 1
            
            for zi in z_sorted:
                eqs.append(g.subs(x, zi))  # g(zi) = 0
                eqs.append(sp.diff(g, x).subs(x, zi))  # g'(zi) = 0
            
            # Solve the linear system
            # The system is linear in c_0..c_25
            # Build matrix and vector
            # Each equation is linear in c's
            
            # Extract coefficients
            matrix = sp.zeros(26, 26)
            rhs = sp.zeros(26, 1)
            
            for i, eq in enumerate(eqs):
                # eq = sum_j M[i,j] * c_j + rhs_i
                # Rearrange: sum_j M[i,j] * c_j = -rhs_i
                for j in range(26):
                    matrix[i, j] = eq.coeff(c[j])
                rhs[i] = -eq.subs({c[j]: 0 for j in range(26)})
            
            # Check if system is singular
            det = matrix.det()
            if det == 0:
                return float('inf'), False
            
            # Solve using LUsolve
            try:
                sol = sp.LUsolve(matrix, rhs)
            except:
                return float('inf'), False
            
            # Substitute solution into g
            g_sol = g
            for j in range(26):
                g_sol = g_sol.subs(c[j], sol[j])
            
            # q(x) = g(x) / (x * prod_i (x - z_i)^2)
            denom = x
            for zi in z_sorted:
                denom *= (x - zi)**2
            
            # Polynomial division
            try:
                q_poly = sp.div(g_sol, denom)
                if len(q_poly) == 2:
                    q, remainder = q_poly
                else:
                    q = q_poly
            except:
                return float('inf'), False
            
            # If there's a remainder, the division isn't exact
            if len(q_poly) == 2:
                _, remainder = q_poly
                if remainder != 0:
                    return float('inf'), False
            
            # Find real roots of q
            real_roots = sp.real_roots(q)
            
            # For each real root r, check if q has strictly opposite signs at r-1e-198 and r+1e-198
            # Since we're working with rationals, we need to be careful
            counted_roots = []
            for r in real_roots:
                if r <= 0:
                    continue
                # Evaluate q at r - 1e-198 and r + 1e-198
                # 1e-198 as a rational is approximately 10^-198
                eps = sp.Rational(1, 10**198)
                q_minus = q.subs(x, r - eps)
                q_plus = q.subs(x, r + eps)
                
                # Check if they have strictly opposite signs
                if q_minus * q_plus < 0:
                    counted_roots.append(r)
            
            if not counted_roots:
                B = 0
            else:
                B = max(counted_roots)
            
            score = float(B) / (2 * sp.pi)
            return score, True
            
        except Exception as e:
            return float('inf'), False
    
    # Evaluate initial score
    score_init, valid_init = evaluate_score(z_init)
    if valid_init:
        best_score = score_init
        best_z = z_init[:]
    
    # Step 3: Try to improve by perturbing the nodes
    # Use a simple coordinate descent or random perturbation approach
    
    if time.time() - start_time > DEADLINE - MARGIN:
        return best_z
    
    # Try small random perturbations
    np.random.seed(42)
    for iteration in range(50):
        if time.time() - start_time > DEADLINE - MARGIN:
            break
        
        # Create a perturbed version
        z_perturbed = best_z[:]
        for i in range(len(z_perturbed)):
            # Small random perturbation
            perturbation = np.random.uniform(-0.05, 0.05)
            z_perturbed[i] += perturbation
            # Ensure positivity
            if z_perturbed[i] <= 0:
                z_perturbed[i] = 0.01
        
        # Ensure distinctness
        for i in range(len(z_perturbed)):
            for j in range(i+1, len(z_perturbed)):
                if abs(z_perturbed[i] - z_perturbed[j]) < 1e-8:
                    z_perturbed[j] += 1e-5
        
        # Sort
        z_perturbed = sorted(z_perturbed)
        
        score_pert, valid_pert = evaluate_score(z_perturbed)
        if valid_pert and score_pert < best_score:
            best_score = score_pert
            best_z = z_perturbed[:]
    
    # Step 4: Try a few different initial scales
    if time.time() - start_time > DEADLINE - MARGIN:
        return best_z
    
    for scale in [0.1, 0.5, 2.0, 5.0]:
        if time.time() - start_time > DEADLINE - MARGIN:
            break
        
        z_scaled = [scale * z for z in z_init]
        z_scaled = sorted(z_scaled)
        
        # Ensure distinctness
        for i in range(len(z_scaled)):
            for j in range(i+1, len(z_scaled)):
                if abs(z_scaled[i] - z_scaled[j]) < 1e-8:
                    z_scaled[j] += 1e-5
        
        score_scaled, valid_scaled = evaluate_score(z_scaled)
        if valid_scaled and score_scaled < best_score:
            best_score = score_scaled
            best_z = z_scaled[:]
    
    return best_z

# EVOLVE-BLOCK-END
