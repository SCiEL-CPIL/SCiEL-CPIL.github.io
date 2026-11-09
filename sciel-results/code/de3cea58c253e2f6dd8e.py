"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.special import genlaguerre
from scipy.optimize import minimize, differential_evolution

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 590.0  # safety margin before 600s kill

    m = 12  # number of double roots
    n = 1   # dimension

    # The Cohn-Goncalves construction for the Fourier sign uncertainty
    # involves a polynomial P(x) = prod (x - r_i)^2 * L_m^{(n)}(x) or similar.
    # The "score" is the upper bound on C4, which is typically related to
    # the constant term or a specific evaluation of the polynomial.
    # 
    # For the Laguerre polynomial construction, a common formulation is that
    # the bound is proportional to the product of the roots or a function thereof.
    # The task states "score is the upper bound on C4" and "lower is better".
    #
    # A key insight from the "inspiration 2" comment: "fixed Laguerre roots (score 0.4862)".
    # This suggests that simply taking the roots of the generalized Laguerre polynomial
    # L_m^{(alpha)}(x) might yield a score of ~0.4862, which is an improvement over 1.0.
    #
    # Let's verify the roots of L_12^{(0)}(x).
    # The roots of L_n^{(alpha)}(x) are positive and distinct.
    
    # Compute roots of the generalized Laguerre polynomial L_12^{(0)}(x)
    # scipy.special.roots_genlaguerre(n, alpha) returns the roots of L_n^{(alpha)}
    try:
        roots_lag = np.sort(roots_genlaguerre(m, 0.0))
    except Exception:
        # Fallback if roots_genlaguerre fails
        roots_lag = np.array([4.0 * (i + 1) for i in range(m)])

    # The task mentions "double roots". In the Cohn-Goncalves method, one often
    # constructs a polynomial with double roots to satisfy sign-change conditions.
    # The "locations" returned are the positions of these double roots.
    
    # Hypothesis: The optimal locations are exactly the roots of the Laguerre polynomial.
    # The score 0.4862 mentioned in inspiration 2 corresponds to this baseline.
    # Can we do better?
    
    # The objective function for the "upper bound on C4" needs to be defined.
    # In many uncertainty principles, the constant C is determined by the norm
    # of the extremal function or the value of the polynomial at a specific point.
    # 
    # Let's look at the structure. If the bound is C4, and it's an upper bound,
    # minimizing it means finding the "tightest" bound.
    #
    # A common form for such bounds in Fourier analysis involves the product of
    # the roots or the value of the orthogonal polynomial.
    #
    # Let's assume the score is proportional to the product of the roots.
    # log(score) = sum(log(r_i)). Minimizing the product means minimizing the sum of logs.
    #
    # However, the roots of the Laguerre polynomial are fixed by the orthogonality
    # condition. We are "tuning" the construction. This might mean we are not
    # restricted to the standard Laguerre roots but can shift or scale them?
    # Or perhaps we are choosing which Laguerre polynomial to use?
    #
    # The task says "Tune the refined Cohn-Goncalves Laguerre-polynomial construction".
    # "Tune" implies we have free parameters.
    #
    # If the free parameters are the root locations themselves, and the constraint is
    # that they form a valid "Laguerre-like" configuration, what is the objective?
    #
    # Let's re-read: "score is the upper bound on C4".
    # In the Cohn-Goncalves paper, the bound is often derived from the value of a
    # specific polynomial at x=0 or similar.
    #
    # Let's try to optimize the root locations directly, assuming the score is
    # a function of the roots. A common candidate is the product of the roots.
    #
    # But wait, if we can just make the roots arbitrarily small, the product goes to 0.
    # There must be a constraint.
    #
    # The constraint might be that the polynomial P(x) = prod(x-r_i)^2 * L_m^{(n)}(x)
    # must satisfy certain sign-change properties or normalization.
    #
    # Alternatively, the "Laguerre construction" might imply that the roots r_i
    # are not free, but determined by a parameter alpha in L_m^{(alpha)}(x).
    # If we can tune alpha, we can change the roots.
    #
    # Let's test this hypothesis: The roots are those of L_m^{(alpha)}(x) for some alpha.
    # We need to find alpha that minimizes the score.
    #
    # What is the score in terms of alpha?
    # If the score is the product of the roots, for L_n^{(alpha)}, the product of roots
    # is related to the coefficients.
    #
    # Let's compute the product of roots for various alpha and see if there's a minimum.
    #
    # However, the task says "construct() -> list of 12 positive double-root locations".
    # This suggests the output is the locations, not alpha.
    #
    # Let's try a direct optimization of the root locations, with a constraint that
    # they are "close" to the Laguerre roots, or perhaps no constraint other than
    # positivity and ordering, but with a specific objective function.
    
    # Let's define the objective function.
    # Based on the "0.4862" score for Laguerre roots, and the seed score of 1.0,
    # the Laguerre roots are a good starting point.
    
    # Let's assume the objective is the product of the roots.
    # For L_12^{(0)}, the product of roots is:
    # The roots are approx: 0.44, 1.60, 3.41, 5.86, 8.93, 12.61, 16.88, 21.73, 27.15, 33.12, 39.63, 46.67
    # Product is huge.
    
    # Maybe the score is the sum of the roots? Or the last root?
    # "computes the last sign change" -> maybe the score is related to the largest root?
    
    # Let's try minimizing the largest root.
    # Or the sum of the roots.
    
    # Let's look at the seed program. It minimizes the log of the product.
    # The seed score is 1.0.
    # The Laguerre roots score is 0.4862.
    # This implies the Laguerre roots have a lower "score" than the seed's arithmetic progression.
    #
    # If the score is the product of roots, the Laguerre roots (smaller roots on average)
    # would have a smaller product than the arithmetic progression (4, 8, ..., 48).
    #
    # Let's calculate the product of the seed roots: 4 * 8 * ... * 48 = 4^12 * 12!
    # 4^12 is about 16 million. 12! is about 479 million. Product is ~7e15.
    #
    # Product of Laguerre roots:
    # r_i approx: 0.44, 1.6, 3.4, 5.9, 8.9, 12.6, 16.9, 21.7, 27.2, 33.1, 39.6, 46.7
    # Log product = sum(log(r_i))
    # log(0.44) = -0.82
    # log(1.6) = 0.47
    # log(3.4) = 1.22
    # log(5.9) = 1.77
    # log(8.9) = 2.19
    # log(12.6) = 2.53
    # log(16.9) = 2.83
    # log(21.7) = 3.08
    # log(27.2) = 3.30
    # log(33.1) = 3.50
    # log(39.6) = 3.68
    # log(46.7) = 3.84
    # Sum ~ 28.0
    # exp(28) ~ 1.4e12.
    #
    # So the product of Laguerre roots is much smaller.
    # If the score is the product, then Laguerre roots are better.
    #
    # Can we do even better?
    # If we can make the roots smaller, the product decreases.
    # But there must be a constraint preventing this.
    #
    # Perhaps the constraint is that the roots must be the roots of L_m^{(alpha)}(x)
    # for some alpha >= 0?
    #
    # Let's check how the product of roots changes with alpha.
    # For L_n^{(alpha)}, the roots move as alpha changes.
    # As alpha -> infinity, the roots spread out.
    # As alpha -> -1 (not allowed for standard Laguerre, but generalized), roots might cluster.
    #
    # Actually, the sum of roots of L_n^{(alpha)} is n(n+alpha+1).
    # The product of roots is related to the constant term.
    #
    # Let's try to optimize alpha.
    
    def score_from_alpha(alpha):
        try:
            if alpha < -1:
                return 1e10
            roots = np.sort(roots_genlaguerre(m, alpha))
            if np.any(roots <= 0):
                return 1e10
            # Objective: minimize the product of roots (log product)
            return np.sum(np.log(roots))
        except Exception:
            return 1e10

    # Optimize alpha
    best_alpha = 0.0
    best_score = score_from_alpha(0.0)
    
    # Use differential evolution or minimize to find optimal alpha
    # Alpha must be >= 0 for standard Laguerre polynomials to have positive roots.
    # Actually, roots are positive for alpha > -1.
    
    try:
        res = differential_evolution(
            score_from_alpha,
            bounds=[(-0.9, 10.0)],  # alpha in (-0.9, 10)
            maxiter=100,
            popsize=15,
            tol=1e-12,
            seed=42
        )
        if res.fun < best_score:
            best_score = res.fun
            best_alpha = res.x[0]
    except Exception:
        pass
    
    # Get the roots corresponding to the best alpha
    try:
        best_roots = np.sort(roots_genlaguerre(m, best_alpha))
    except Exception:
        best_roots = np.sort(roots_genlaguerre(m, 0.0))
    
    # Ensure roots are positive and ordered
    best_roots = np.maximum(best_roots, 1e-6)
    for i in range(1, m):
        if best_roots[i] <= best_roots[i-1]:
            best_roots[i] = best_roots[i-1] + 1e-6
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline:
        # Return the best we have so far
        pass
    
    return [float(x) for x in best_roots]
# EVOLVE-BLOCK-END
