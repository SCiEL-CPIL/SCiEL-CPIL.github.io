"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
import numpy as np
from fractions import Fraction
import itertools
import sys
import time

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build the matrix M: for each triple (i,j,k), we have an equation
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # This is multilinear in U, V, W.
    
    # Strategy: For a given rank r, we can try to find a decomposition by:
    # 1. Fixing U and V (or some of them) and solving for W
    # 2. Or using a more systematic approach
    
    # For small problems, we can try to use the fact that the matrix multiplication
    # tensor has special structure.
    
    # Let's try a direct approach: for each rank r from a lower bound up,
    # try to find a valid decomposition.
    
    # Lower bound: rank >= max(n*m, m*p, p*n) is not correct.
    # Actually, rank >= n*m*p / max(n*m, m*p, p*n) * something...
    # For 3x5x7, let's start from r = 15 and go up.
    
    # A better lower bound: the rank of the matrix multiplication tensor
    # is at least the rank of any "slice" of the tensor.
    # But for now, let's just try increasing r.
    
    # For a given r, we have r*(nm + mp + pn) unknowns and nm*mp*p... no,
    # we have n*m*p = 105 equations (one for each (i,j,k)).
    
    # The system is:
    # For each (i,j,k): sum_{c=0}^{r-1} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # This is a system of 105 equations in r*(15+35+21) = 71r unknowns.
    # For r >= 2, we have more unknowns than equations, so solutions exist.
    # But we want integer/half-integer solutions, and we want the minimum r.
    
    # Let's try a constructive approach:
    # For r = n*m*p = 105, we have the schoolbook solution.
    # Can we do better?
    
    # Idea: Use the fact that some linear combinations might cancel.
    # For example, if we have two terms with the same (i,j) but different k,
    # we might be able to combine them.
    
    # Actually, let's try to use a known result: for matrix multiplication,
    # the rank is related to the number of "essential" multiplications.
    
    # For now, let's try to find a decomposition with rank < 105 by
    # solving a linear system for specific choices of U and V.
    
    # Approach: Fix U and V to have a specific structure, then solve for W.
    # For example, let U and V be identity-like matrices.
    
    # Let's try: for each c in range(r), set U[i*m+j, c] and V[j*p+k, c]
    # to be non-zero for specific (i,j) and (j,k), then solve for W.
    
    # A simpler approach: use the "Landsberg" style decomposition.
    
    # For now, let's try to return the schoolbook decomposition, but
    # check if we can reduce the rank by exploiting linear dependencies.
    
    # Let's try a numerical approach: use random U and V, then solve for W
    # using least squares, and check if the solution is valid.
    
    # But we need exact arithmetic. Let's use Fraction.
    
    # For a given r, let's try the following:
    # 1. Set U and V to be random (or structured) matrices with Fraction entries.
    # 2. For each (i,j,k), the equation is:
    #    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    #    This is linear in W.
    # 3. Solve for W using Gaussian elimination.
    
    # But the system is: for each (i,j,k), we have one equation in W.
    # The unknowns are W[k*n+i, c] for c in range(r).
    # For fixed (i,j), the equations over k give us a system in W[*, c].
    
    # Actually, for fixed (i,j) and fixed c, the equation is:
    # U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = contribution to (i,j,k)
    # But the sum is over c, so it's coupled.
    
    # Let's think differently. For fixed U and V, the system for W is:
    # For each (i,j,k): sum_c (U[i*m+j, c] * V[j*p+k, c]) * W[k*n+i, c] = 1
    # This is a linear system in W. The coefficient matrix is:
    # A[(i,j,k), (k',i',c)] = U[i*m+j, c] * V[j*p+k, c] if (k',i') == (k,i) else 0
    # Wait, W is indexed by (k*n+i, c), so the unknowns are W[row, c] where row = k*n+i.
    # For a given (i,j,k), the equation involves W[k*n+i, c] for all c.
    # So the system is: for each (i,j,k), sum_c coeff[(i,j,k),c] * W[k*n+i, c] = 1
    # where coeff[(i,j,k),c] = U[i*m+j, c] * V[j*p+k, c]
    
    # This means that for each (i,j,k), we have an equation that only involves
    # the W values for row = k*n+i. So the system decouples by (i,k) pairs!
    
    # For each (i,k), the equations over j are:
    # For j in range(m): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # This is a system of m equations in r unknowns (W[k*n+i, 0..r-1]).
    
    # So for each (i,k), we need to solve:
    # M[i,k] * W[k*n+i, :]^T = 1_m
    # where M[i,k] is an m x r matrix with M[i,k][j, c] = U[i*m+j, c] * V[j*p+k, c]
    
    # For this to be solvable, we need rank(M[i,k]) >= 1 (which is always true if m >= 1)
    # and the system must be consistent. Since we have m equations in r unknowns,
    # we need r >= 1 for a solution to exist (when m >= 1).
    
    # But we also need the same U and V to work for all (i,k) pairs.
    
    # Let's try: for a given r, choose U and V such that for each (i,k),
    # the system M[i,k] * W_row^T = 1 has a solution.
    
    # A simple choice: let U and V be such that for each (i,k), M[i,k] has
    # a row of all 1s (or some known pattern).
    
    # Actually, let's try the schoolbook decomposition first, then see if we
    # can reduce the rank.
    
    # Schoolbook: r = n*m*p = 105
    # U[i*m+j, i*m*p + j*p + k] = 1 (but this doesn't fit in r=105 columns)
    # Actually, in the schoolbook, each (i,j,k) triple gets its own column.
    # So U[i*m+j, idx] = 1, V[j*p+k, idx] = 1, W[k*n+i, idx] = 1 where idx = i*m*p + j*p + k
    
    # Can we reduce this? If two triples (i,j,k) and (i',j',k') have the same
    # (i,j) but different k, they share the same U row. If they have the same
    # (j,k) but different i, they share the same V row. If they have the same
    # (i,k) but different j, they share the same W row.
    
    # The key insight: we can combine terms if they share structure.
    
    # Let's try a different approach: use the fact that matrix multiplication
    # can be done with fewer than n*m*p multiplications using Strassen-like methods.
    
    # For 3x5x7, let's try to find a decomposition with rank < 105.
    
    # Given the complexity, let's try to use a heuristic:
    # 1. Start with the schoolbook decomposition (r=105).
    # 2. Try to find linear dependencies among the columns.
    # 3. If we find that some columns are linearly dependent, we can reduce the rank.
    
    # But this is complex. For now, let's return the schoolbook decomposition.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
