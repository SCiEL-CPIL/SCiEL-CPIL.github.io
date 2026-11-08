"""
Rank-minimal exact tensor decomposition for 3x3 * 3x4 matrix multiplication.

Strategy:
  - Start with the schoolbook decomposition (rank 36).
  - Use a randomized search over GF(2) and GF(3) to find a lower-rank decomposition.
  - The idea: if a rank-r decomposition exists over the integers, it also exists
    over any finite field. We search for rank-r decompositions over small fields,
    then attempt to lift them to integer/half-integer solutions.
  - We use a linear algebra approach: fix the structure of U and V, then solve for W.
  
  For matrix multiplication C = A * B where A is n x m, B is m x p, C is n x p:
  The tensor T[i, j, k] = 1 if and only if the (i,k) entry of C depends on A[i,j] * B[j,k].
  
  A rank-r decomposition has:
    U: (n*m) x r  matrix, U[i*m + j, c]
    V: (m*p) x r  matrix, V[j*p + k, c]  
    W: (p*n) x r  matrix, W[k*n + i, c]
  
  Such that for all i, j, k:
    sum_c U[i*m+j, c] * V[j*p+k, c] = W[k*n+i, c]  ... wait, that's not right.
  
  Actually, the tensor decomposition for matrix multiplication means:
    (A*B)[i,k] = sum_j A[i,j] * B[j,k]
  
  This can be written as a tensor:
    T[i, j, k] = delta (the multiplication tensor)
  
  A rank-r decomposition gives:
    T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]  -- no, that's not right either.
  
  Let me reconsider. The standard formulation:
  
  We want to compute C = A * B. The entries are C[i,k] = sum_j A[i,j] B[j,k].
  
  A rank-r decomposition of the matrix multiplication tensor means we can write:
    C[i,k] = sum_{c=1}^{r} alpha_c(i,k) * (sum_j U[i,j,c] A[i,j]) * (sum_j' V[j',k,c] B[j',k])
  
  Wait, that's not quite right either. Let me use the proper formulation.
  
  The matrix multiplication tensor T is a 3-tensor of size (n*m) x (m*p) x (n*p) where:
    T[(i,j), (j',k), (i',k')] = 1 if i=i' and j=j' and k=k', else 0
  
  Actually, the standard way: T[i,j,k] = 1 for the "multiplication" part.
  
  A rank-r decomposition means:
    T[i,j,k] = sum_{c=1}^r U[i,j,c] * V[j,k,c] * W[i,k,c]
  
  where U is n x m x r, V is m x p x r, W is n x p x r.
  
  In the flattened form used in the task:
    U_flat[i*m + j, c] = U[i,j,c]
    V_flat[j*p + k, c] = V[j,k,c]  
    W_flat[k*n + i, c] = W[i,k,c]
  
  So we need: for all i in [0,n), j in [0,m), k in [0,p):
    sum_c U_flat[i*m+j, c] * V_flat[j*p+k, c] * W_flat[k*n+i, c] = 1
  
  And for all other combinations (i,j,k) that don't correspond to a valid multiplication term,
  the sum should be 0. But wait, in matrix multiplication, every (i,j,k) triple with i in [0,n),
  j in [0,m), k in [0,p) contributes to C[i,k]. So actually ALL triples (i,j,k) with valid indices
  have T[i,j,k] = 1? No, that's not right.
  
  Let me think again. The matrix multiplication tensor has entries:
    T[i, j, k] = 1 if this triple (i,j,k) represents a multiplication term A[i,j]*B[j,k]
  
  In standard matrix multiplication C[i,k] = sum_j A[i,j] B[j,k], the tensor T has:
    T[i,j,k] = 1 for all valid (i,j,k), because each such term contributes.
  
  But that would make T all ones, which can't be right for a tensor decomposition problem.
  
  I think I'm confusing myself. Let me look at this differently.
  
  The correct formulation: The matrix multiplication tensor T is defined such that:
    sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] = sum_{i,k} C[i,k]
  
  where C[i,k] = sum_j A[i,j] B[j,k].
  
  So T[i,j,k] = 1 if and only if the term A[i,j]B[j,k] appears in C[i,k], which is always true
  for valid indices. But then T is just the all-ones tensor on the valid index set, which doesn't
  make sense.
  
  OK, I think the correct formulation is:
  
  T is a tensor of shape (n*p) x (n*m) x (m*p) where:
    T[c, a, b] = 1 if C[c] = A[a] * B[b] for some factorization
  
  No, that's not right either.
  
  Let me use the standard Strassen formulation. The matrix multiplication tensor T_{ijk} for
  multiplying an n x m matrix by an m x p matrix is:
    T_{ijk} = 1 if i < n, j < m, k < p (i.e., for all valid indices)
  
  And a rank-r decomposition gives matrices U (n*m x r), V (m*p x r), W (n*p x r) such that:
    T_{ijk} = sum_{c=1}^r U_{(i,j),c} * V_{(j,k),c} * W_{(i,k),c}
  
  Wait, but W is indexed by (i,k) not (k,i). Let me check the task description again.
  
  "The output factor follows the official trace convention W[k*n+i,r]."
  
  So W[k*n + i, c] corresponds to W[i,k,c] in my notation.
  
  So the condition is: for all i in [0,n), j in [0,m), k in [0,p):
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
  
  And this must hold for ALL such triples. There are no "other" triples because the tensor is only
  defined on the valid index set.
  
  So we need to find U, V, W with integer or half-integer entries such that:
    For all i, j, k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
  
  This is a system of n*m*p = 3*3*4 = 36 equations in r * (n*m + m*p + n*p) = r * (9 + 12 + 12) = 33r unknowns.
  
  For rank 36 (schoolbook), each equation has exactly one term that is 1, so it's trivially satisfied.
  
  For lower rank, we need to find non-trivial solutions.
  
  Approach:
  1. Try rank r from some lower bound up to 36.
  2. For each r, set up the system and try to solve it.
  
  The system is nonlinear (products of U, V, W entries), so it's hard to solve directly.
  
  Alternative approach: Fix U and V, then W is determined linearly? No, because each equation
  involves one W entry multiplied by a product of U and V entries.
  
  Let me write it out. For fixed i, j, k:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
  
  This is a sum over c of products of three variables. It's a cubic equation in the variables.
  
  This is a hard problem. Let me think of a different approach.
  
  Alternative: Use the fact that for small dimensions, we can do an exhaustive or randomized search.
  
  For n=3, m=3, p=4, the schoolbook rank is 36. The known minimal rank for 3x3 * 3x4 is not well-known,
  but it's at most 36 and at least some lower bound.
  
  Let me try a different strategy: Start with the schoolbook decomposition and try to "merge" pairs of
  terms to reduce the rank. This is a common technique in tensor rank reduction.
  
  Actually, let me try a simpler approach first: use NumPy to set up and solve a least-squares problem,
  then round to the nearest integer or half-integer.
  
  For a given rank r, we have 33r variables and 36 equations. If 33r >= 36, i.e., r >= 2 (since 33*1=33 < 36),
  we might have solutions. But the equations are nonlinear.
  
  Let me try a randomized approach:
  1. Initialize U and V randomly with entries from {-1, 0, 1}.
  2. For each (i,k) pair, solve for W[k*n+i, :] from the equations.
  
  For fixed i, k, the equations involving W[k*n+i, c] are:
    For each j: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
  
  This is a system of m equations in r unknowns (the W[k*n+i, c] for c=1..r).
  
  Specifically, for fixed i, k:
    sum_c W[k*n+i, c] * (U[i*m+j, c] * V[j*p+k, c]) = 1 for j = 0, 1, ..., m-1
  
  This is a linear system in W[k*n+i, :]:
    M[i,k] * W[k*n+i, :] = 1
  
  where M[i,k] is an m x r matrix with M[i,k][j, c] = U[i*m+j, c] * V[j*p+k, c].
  
  So the algorithm is:
  1. Randomly initialize U and V.
  2. For each (i, k), solve the linear system M[i,k] * w = 1 for w = W[k*n+i, :].
  3. Check if all solutions w are integers or half-integers.
  4. If yes, we have a valid decomposition.
  
  This is much more tractable! Let me implement this.
  
  The linear system M[i,k] * w = 1 has m equations and r unknowns. For a solution to exist,
  the rank of M[i,k] must be m (or the system must be consistent).
  
  For r >= m, we can use least squares or pseudo-inverse to find a solution. Then check if it's
  integer or half-integer.
  
  Let me implement this with a deadline guard.
"""

import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120  # 120 second deadline
    
    # Start with schoolbook as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = schoolbook()
    
    # Try to find a lower rank decomposition
    # We'll try ranks from a lower bound up to 35
    # Lower bound: at least max(n*m, m*p, n*p) = max(9, 12, 12) = 12? Not necessarily.
    # Actually, the rank must be at least the dimension of the space spanned by the slices.
    
    # For each rank r, try random initializations of U and V, then solve for W.
    
    def is_half_integer(x, tol=1e-9):
        """Check if x is close to an integer or half-integer."""
        x2 = 2 * x
        nearest = round(x2)
        return abs(x2 - nearest) < tol
    
    def try_rank(r, num_trials=1000):
        """Try to find a rank-r decomposition by random search."""
        for trial in range(num_trials):
            if time.time() > deadline - 10:
                return None
            
            # Randomly initialize U and V with entries from {-1, 0, 1}
            U = np.zeros((n * m, r), dtype=float)
            V = np.zeros((m * p, r), dtype=float)
            
            for i in range(n * m):
                for c in range(r):
                    U[i, c] = random.choice([-1, 0, 1])
            for i in range(m * p):
                for c in range(r):
                    V[i, c] = random.choice([-1, 0, 1])
            
            # For each (i, k), solve for W[k*n + i, :]
            W = np.zeros((p * n, r), dtype=float)
            valid = True
            
            for i in range(n):
                for k in range(p):
                    # Build M[i,k]: m x r matrix
                    M = np.zeros((m, r), dtype=float)
                    for j in range(m):
                        for c in range(r):
                            M[j, c] = U[i * m + j, c] * V[j * p + k, c]
                    
                    # Solve M * w = 1 (vector of ones)
                    try:
                        # Use least squares
                        w, residuals, rank_m, sv = np.linalg.lstsq(M, np.ones(m), rcond=None)
                    except:
                        valid = False
                        break
                    
                    # Check if w is half-integer
                    if not all(is_half_integer(w[c]) for c in range(r)):
                        valid = False
                        break
                    
                    W[k * n + i, :] = w
            
            if not valid:
                continue
            
            # Verify the full decomposition
            U_list = [[int(round(2 * U[i, c]) / 2) for c in range(r)] for i in range(n * m)]
            V_list = [[int(round(2 * V[i, c]) / 2) for c in range(r)] for i in range(m * p)]
            W_list = [[int(round(2 * W[i, c]) / 2) for c in range(r)] for i in range(p * n)]
            
            # Verify
            ok = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for c in range(r):
                            s += U_list[i * m + j][c] * V_list[j * p + k][c] * W_list[k * n + i][c]
                        if abs(s - 1) > 1e-9:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            
            if ok:
                return {"U": U_list, "V": V_list, "W": W_list}
        
        return None
    
    # Try ranks from 12 up to 35
    for r in range(12, 36):
        if time.time() > deadline - 30:
            break
        
        # Scale number of trials based on remaining time
        remaining = deadline - time.time()
        trials = min(5000, max(100, int(remaining * 50)))
        
        result = try_rank(r, trials)
        if result is not None:
            best_result = result
            break
    
    return best_result
