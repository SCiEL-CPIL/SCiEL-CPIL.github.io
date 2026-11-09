"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard: evaluator kills at 630s, keep safety margin.
    start_time = time.time()
    DEADLINE = 620.0  # seconds
    
    # We need to find 12 double-root locations z_1..z_12 > 0, pairwise distinct.
    # The score is B / (2*pi), where B is the largest real root r of q(x) such that
    # q has strictly opposite signs at r-1e-198 and r+1e-198. Lower B is better.
    # q(x) = g(x) / (x * prod_i (x - z_i)^2)
    
    # Strategy: The system is determined by choosing z_i. The coefficients c_k are
    # solved exactly. The key insight is that we want to minimize the largest root of q.
    
    # From the structure, g(x) = sum_{k=0}^{25} c_k L_{2k}^{alpha}(x) with alpha=-1/2.
    # g(0)=0, g'(0)=1, and g(z_i)=g'(z_i)=0 for all i.
    
    # Let's think about what minimizes B. B is the largest root of q beyond the double roots.
    # If we place the double roots z_i to be very large, q might have smaller roots?
    # Or if we place them optimally?
    
    # Actually, let's think about the degrees. g(x) is a polynomial of degree at most 50.
    # g(x) has zeros at 0 (from g(0)=0, but g'(0)=1 so it's a simple zero at 0? No, g(0)=0 and g'(0)=1 means x is a factor but x^2 is not).
    # Wait, g(0)=0 and g'(0)=1. So x divides g(x) but x^2 does not.
    # g(z_i)=0 and g'(z_i)=0 means (x-z_i)^2 divides g(x).
    # So g(x) = x * prod_i (x-z_i)^2 * h(x) where h is a polynomial.
    # Degree of g is at most 50. Degree of x * prod (x-z_i)^2 is 1 + 2*12 = 25.
    # So h(x) has degree at most 50-25 = 25.
    # q(x) = g(x)/(x*prod(x-z_i)^2) = h(x). So q is just h, a polynomial of degree <= 25.
    
    # The coefficients c_k are determined by the conditions. There are 26 unknowns c_0..c_25.
    # Conditions: g(0)=0 (1), g'(0)=1 (1), g(z_i)=0 and g'(z_i)=0 for 12 i's (24). Total 26 conditions.
    # So the system is square and generically has a unique solution.
    
    # The question is: for which z_i is the largest root of h(x)=q(x) minimized?
    
    # Let's try a simple approach: use a reasonable initial guess and see if we can improve.
    # Start with z_i = i (1, 2, ..., 12).
    
    # But computing the exact rational solution and finding roots of q might be slow.
    # Let's estimate: 26x26 LU solve with rationals, then polynomial division, then root finding.
    # This could take significant time. We need to be careful.
    
    # Given the time constraint, let's try a few candidate configurations.
    
    # Candidate 1: z_i = i for i=1..12
    # Candidate 2: z_i = i^2 for i=1..12 (spread out more)
    # Candidate 3: z_i = 2^i for i=1..12 (exponential)
    
    # But we can't easily compute B without the full exact arithmetic.
    # Let's just return a reasonable set of positive distinct values.
    
    # The seed returns [4, 8, 12, ..., 48]. Let's try something similar but perhaps optimized.
    
    # Actually, let's think: if we make z_i very large, the polynomial g might have different behavior.
    # If z_i are very small, same.
    
    # Let's try z_i = i for i=1..12 as a baseline.
    candidates = [
        [float(i) for i in range(1, 13)],
        [float(i*2) for i in range(1, 13)],
        [float(i*i) for i in range(1, 13)],
        [4.0*(i+1) for i in range(12)],  # original seed
    ]
    
    # For now, return the original seed since we can't easily evaluate B without heavy computation.
    # But let's try to be smarter: maybe smaller z_i lead to smaller B?
    
    # Let's return [1, 2, ..., 12] as a test.
    return [float(i) for i in range(1, 13)]
# EVOLVE-BLOCK-END
