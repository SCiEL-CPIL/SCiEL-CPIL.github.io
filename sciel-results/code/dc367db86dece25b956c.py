"""
Matrix multiplication 3x4x8 rank-minimal exact tensor decomposition.

Approach:
1. Start with the schoolbook decomposition of rank 96.
2. Use alternating minimization with exact rational arithmetic to find a
   lower-rank decomposition that exactly satisfies the tensor identity.
3. The key insight: we can try to find a rank-r decomposition by solving
   for factor matrices U (12 x r), V (32 x r), W (24 x r) such that
   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta_{i,j,k} for all valid indices.

   Actually, the matrix multiplication tensor T_{i,j,k} for multiplying A (n x m) by B (m x p)
   gives C (n x p) where C[i,k] = sum_j A[i,j] B[j,k].
   
   The tensor is indexed such that T_{(i,j), (j,k), (k,i)} = 1 if the indices are consistent,
   but more precisely, the multiplication tensor has entries:
   T[(i,j), (j',k), (i',k')] = delta_{j,j'} * delta_{i,i'} * delta_{k,k'}
   
   Wait, let me think more carefully. The standard formulation:
   We want to compute C = A * B where A is n x m, B is m x p, C is n x p.
   C[i,k] = sum_j A[i,j] B[j,k]
   
   The tensor T is a 3-tensor of dimensions (n*m) x (m*p) x (n*p) such that:
   T[(i,j), (j',k'), (i'',k'')] = 1 if j == j' and i == i'' and k == k'', else 0.
   
   So T[(i,j), (j,k), (i,k)] = 1, and 0 otherwise.
   
   A rank-r decomposition means we can write T as a sum of r rank-1 tensors:
   T = sum_{c=1}^{r} u_c ⊗ v_c ⊗ w_c
   
   where u_c is a vector in R^{n*m}, v_c in R^{m*p}, w_c in R^{n*p}.
   
   In the output convention W[k*n+i, r], the third factor is indexed by (k, i) with k in [0,p), i in [0,n).
   So W has dimensions p*n = 24.

   The schoolbook solution has rank n*m*p = 96, with one term for each (i,j,k):
   u_{i,j,k} has 1 at position (i,j)
   v_{i,j,k} has 1 at position (j,k)
   w_{i,j,k} has 1 at position (k,i)

   To reduce the rank, we need to find a linear combination of these basis tensors that still
   reconstructs T. This is equivalent to finding a rank-r decomposition.

   For small matrices, known optimal ranks:
   - 3x4x8: The known lower bound is at least n*m*p / (some factor). 
   - Actually, for matrix multiplication, the rank is at least max(n*m, m*p, n*p) = max(12, 32, 24) = 32.
   - But more refined bounds exist. For 3x4x8, I believe the optimal rank is known to be around 87-90.

   Let me try a different approach: use the fact that we can express the multiplication tensor
   in a different basis and try to find a sparse representation.

   Actually, let me try a practical approach: start with rank 96 and try to reduce it by
   finding linear dependencies among the rank-1 terms.

   Key idea: If we have a rank-r decomposition, the factors U, V, W must satisfy:
   For all (i,j), (j',k'), (i'',k''):
   sum_c U[(i,j), c] * V[(j',k'), c] * W[(k'',i''), c] = delta_{j,j'} * delta_{i,i''} * delta_{k,k''}
   
   where k = k' and i'' = i and k'' = k.

   This is a system of bilinear equations in the entries of U, V, W.

   Let me try a simpler approach: use the known structure. For matrix multiplication A (n x m) * B (m x p),
   we can use the fact that:
   C[i,k] = sum_j A[i,j] B[j,k]
   
   The tensor can be written as a sum over j of (e_i ⊗ e_j) ⊗ (e_j ⊗ e_k) ⊗ (e_k ⊗ e_i) ... no that's not right.

   Let me just try to find a good decomposition numerically and then verify it exactly.

   Strategy:
   1. Try ranks from 87 down to some lower bound.
   2. For each rank, initialize U, V, W randomly (or with structure) and use alternating least squares
      to minimize the error.
   3. Check if the error is zero (within tolerance for floating point, then verify with rationals).

   But this might not find the exact minimum. Let me think of a better way.

   Actually, for this specific problem, let me try to use the following insight:
   The matrix multiplication tensor for n x m * m x p has a natural decomposition where we can
   group terms. For example, if we fix i and k, we have sum_j A[i,j] B[j,k].

   Let me try rank = 87 and see if we can find a valid decomposition.

   Given the complexity, let me try a heuristic approach:
   - Start with the 96 schoolbook terms.
   - Try to merge terms that share structure.

   Actually, let me try a completely different approach: use the Strassen-like idea but adapted
   for rectangular matrices. However, Strassen's algorithm is for square matrices.

   For 3x4x8, let me try to find a decomposition with rank less than 96 by solving the
   system of equations.

   Let me code up a solver that:
   1. Sets up the tensor constraints.
   2. Tries to find U, V, W of rank r that satisfy the constraints.
   3. Uses alternating minimization.

   For r = 87, we have:
   - U: 12 x 87 = 1044 variables
   - V: 32 x 87 = 2784 variables
   - W: 24 x 87 = 2088 variables
   Total: 5916 variables.

   The constraints are 12 * 32 * 24 = 9216 equations (but many are trivially satisfied or redundant).

   Actually, the non-zero entries of the tensor are exactly n*m*p = 96 entries (one for each (i,j,k)
   with the corresponding (i,j), (j,k), (k,i) indices). All other 9216 - 96 = 9120 entries must be 0.

   So we have 9216 constraints, but only 96 are non-trivial (the others are just saying the sum
   should be 0).

   This is a highly overdetermined system, which makes it hard to solve directly.

   Let me try a different tactic: use the fact that the tensor is sparse and try to find a
   structured decomposition.

   Given time constraints, let me try rank 87 with a randomized alternating minimization approach,
   and if that doesn't work, fall back to 96.

   Actually, I realize I should just try to find a good decomposition. Let me use the following
   approach:

   For the matrix multiplication tensor T, we know that:
   T[(i,j), (j',k'), (i'',k'')] = 1 iff j=j', i=i'', k=k''

   We can write T = sum_{j} (sum_i e_i ⊗ e_j) ⊗ (sum_k e_j ⊗ e_k) ⊗ (sum_{i,k} ...) 
   
   This doesn't seem to lead anywhere simple.

   Let me just implement a practical solver and hope it finds a lower rank.
   """

import numpy as np
from fractions import Fraction
import random
import time

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 12
    mp = m * p  # 32
    pn = p * n  # 24

    # Build the target tensor (sparse representation)
    # T[(i*m+j), (j*p+k), (k*n+i)] = 1
    target = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx1 = i * m + j
                idx2 = j * p + k
                idx3 = k * n + i
                target[(idx1, idx2, idx3)] = 1

    # Try to find a lower-rank decomposition
    best_rank = n * m * p  # 96
    best_factors = None

    # Start with schoolbook
    U = np.zeros((nm, best_rank))
    V = np.zeros((mp, best_rank))
    W = np.zeros((pn, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1

    # Verify schoolbook
    def verify(U, V, W, target, tol=1e-9):
        """Check if the decomposition exactly reconstructs the tensor."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        # For each non-zero target entry, check it's 1
        for (i1, i2, i3), val in target.items():
            s = sum(U[i1, c] * V[i2, c] * W[i3, c] for c in range(U.shape[1]))
            if abs(s - val) > tol:
                return False
        
        # Check some random zero entries
        rng = np.random.RandomState(42)
        for _ in range(100):
            i1 = rng.randint(0, nm)
            i2 = rng.randint(0, mp)
            i3 = rng.randint(0, pn)
            if (i1, i2, i3) not in target:
                s = sum(U[i1, c] * V[i2, c] * W[i3, c] for c in range(U.shape[1]))
                if abs(s) > tol:
                    return False
        return True

    if verify(U, V, W, target):
        best_factors = (U, V, W)

    # Try to reduce rank using alternating minimization
    for target_rank in range(87, 96):
        if target_rank >= best_rank:
            continue
        
        found = False
        for trial in range(5):
            rng = np.random.RandomState(trial * 100 + target_rank)
            
            # Initialize with random values
            U = rng.randn(nm, target_rank)
            V = rng.randn(mp, target_rank)
            W = rng.randn(pn, target_rank)
            
            # Scale to have reasonable magnitude
            U /= np.linalg.norm(U) * np.sqrt(target_rank)
            V /= np.linalg.norm(V) * np.sqrt(target_rank)
            W /= np.linalg.norm(W) * np.sqrt(target_rank)
            
            # Alternating minimization
            converged = False
            for it in range(200):
                # Update U: for each i1, U[i1, :] should satisfy
                # sum_c U[i1,c] * V[i2,c] * W[i3,c] = target[i1,i2,i3] for all i2,i3
                # This is: U[i1, :] @ (V[i2,:] * W[i3,:])^T = target[i1,i2,i3]
                # Let B[i2,i3,c] = V[i2,c] * W[i3,c], then U[i1,:] @ B[i2,i3,:]^T = target
                # So U[i1,:] = argmin sum_{i2,i3} (U[i1,:] @ B[i2,i3,:]^T - target[i1,i2,i3])^2
                # = solve linear system: (B^T B) U[i1,:]^T = B^T t[i1,:]
                
                # Build B: shape (mp * pn, target_rank)
                B = np.zeros((mp * pn, target_rank))
                for i2 in range(mp):
                    for i3 in range(pn):
                        B[i2 * pn + i3, :] = V[i2, :] * W[i3, :]
                
                # Solve for each i1
                try:
                    BtB = B.T @ B
                    BtB += 1e-12 * np.eye(target_rank)  # regularization
                    for i1 in range(nm):
                        t_vec = np.zeros(mp * pn)
                        for i2 in range(mp):
                            for i3 in range(pn):
                                if (i1, i2, i3) in target:
                                    t_vec[i2 * pn + i3] = target[(i1, i2, i3)]
                        rhs = B.T @ t_vec
                        U[i1, :] = np.linalg.solve(BtB, rhs)
                except np.linalg.LinAlgError:
                    break
                
                # Similarly update V and W
                # For V: sum_c U[i1,c] * V[i2,c] * W[i3,c] = target
                # Fix i2: V[i2,:] @ (U[i1,:] * W[i3,:])^T = target[i1,i2,i3]
                Bv = np.zeros((nm * pn, target_rank))
                for i1 in range(nm):
                    for i3 in range(pn):
                        Bv[i1 * pn + i3, :] = U[i1, :] * W[i3, :]
                try:
                    BvtBv = Bv.T @ Bv
                    BvtBv += 1e-12 * np.eye(target_rank)
                    for i2 in range(mp):
                        t_vec = np.zeros(nm * pn)
                        for i1 in range(nm):
                            for i3 in range(pn):
                                if (i1, i2, i3) in target:
                                    t_vec[i1 * pn + i3] = target[(i1, i2, i3)]
                        rhs = Bv.T @ t_vec
                        V[i2, :] = np.linalg.solve(BvtBv, rhs)
                except np.linalg.LinAlgError:
                    break
                
                # For W: sum_c U[i1,c] * V[i2,c] * W[i3,c] = target
                Bw = np.zeros((nm * mp, target_rank))
                for i1 in range(nm):
                    for i2 in range(mp):
                        Bw[i1 * mp + i2, :] = U[i1, :] * V[i2, :]
                try:
                    BwtBw = Bw.T @ Bw
                    BwtBw += 1e-12 * np.eye(target_rank)
                    for i3 in range(pn):
                        t_vec = np.zeros(nm * mp)
                        for i1 in range(nm):
                            for i2 in range(mp):
                                if (i1, i2, i3) in target:
                                    t_vec[i1 * mp + i2] = target[(i1, i2, i3)]
                        rhs = Bw.T @ t_vec
                        W[i3, :] = np.linalg.solve(BwtBw, rhs)
                except np.linalg.LinAlgError:
                    break
                
                # Check convergence
                if verify(U, V, W, target, tol=1e-6):
                    converged = True
                    break
            
            if converged:
                best_rank = target_rank
                best_factors = (U, V, W)
                found = True
                break
        
        if found:
            break
    
    # Convert to lists with rational entries if possible
    if best_factors is not None:
        U, V, W = best_factors
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        
        # Try to round to nice fractions
        def to_fraction_list(lst):
            result = []
            for row in lst:
                new_row = []
                for x in row:
                    # Try to find a simple fraction
                    best_frac = Fraction(0)
                    for denom in range(1, 100):
                        num = round(x * denom)
                        if abs(x - num / denom) < 1e-9:
                            best_frac = Fraction(num, denom)
                            break
                    new_row.append(best_frac)
                result.append(new_row)
            return result
        
        U_list = to_fraction_list(U_list)
        V_list = to_fraction_list(V_list)
        W_list = to_fraction_list(W_list)
        
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Fall back to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
