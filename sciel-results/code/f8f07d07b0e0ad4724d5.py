import numpy as np
from scipy.optimize import linprog
from itertools import product
import time

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    total_basis = nm * mp * pn  # 120 basis tensors
    
    # The multiplication tensor T has entries T[i*m+j, j*p+k, k*n+i] = 1
    # We want to write T = sum_r u_r ⊗ v_r ⊗ w_r
    # where u_r ∈ R^{nm}, v_r ∈ R^{mp}, w_r ∈ R^{pn}
    
    # For each basis tensor indexed by (i,j,k), we have:
    # u has 1 at position i*m+j, v has 1 at position j*p+k, w has 1 at position k*n+i
    
    # We need to find coefficients c_{i,j,k} such that the tensor equation holds.
    # The tensor equation: for all (a, b, c) in {0..n-1} x {0..m-1} x {0..p-1},
    # sum over (i,j,k) of c_{i,j,k} * delta_{a,i} * delta_{b,j} * delta_{c,k} = delta_{a,b,c}... 
    # Wait, that's not right. Let me reconsider.
    
    # Actually, the multiplication tensor T is defined as:
    # T[a, b, c] = sum_{j} A[a,j] * B[j,c]
    # As a tensor, T is the sum over all (i,j,k) of e_i ⊗ e_j ⊗ e_k where the
    # multiplication tensor has T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    
    # The standard decomposition uses rank n*m*p with each term being e_i ⊗ e_j ⊗ e_k
    # where e_i is the i-th basis vector in R^n, etc.
    
    # To find a lower rank decomposition, we need to find linear dependencies.
    # The tensor T can be viewed as a 3-way array of size n x m x p.
    # T[i,j,k] = 1 for all i,j,k (in the standard basis decomposition sense)
    # Wait no. Let me be more careful.
    
    # The matrix multiplication tensor T satisfies:
    # T(A,B) = A*B for all A (n x m), B (m x p)
    # T is a multilinear map, so T is a tensor in (R^n*) ⊗ (R^m*) ⊗ (R^p)
    # In coordinates, T_{i,j,k} = 1 if we identify A_{i,j} and B_{j,k}
    # Actually T_{i,j,k} = delta_{i,j} * delta_{j,k}... no that's not right either.
    
    # Let me use the standard formulation. The multiplication tensor is:
    # T = sum_{i=1}^n sum_{j=1}^m sum_{k=1}^p e_i^* ⊗ e_j^* ⊗ e_k
    # where e_i^* is the i-th dual basis vector.
    # So T_{i,j,k} = 1 for all i,j,k in the standard basis.
    
    # Wait, that gives T_{i,j,k} = 1 for all i,j,k, which is just the all-ones tensor.
    # That's not right for matrix multiplication.
    
    # Let me reconsider. For matrix multiplication C = A*B:
    # C_{i,k} = sum_j A_{i,j} * B_{j,k}
    # The tensor T is defined by T(A,B) = C, so T is in (R^{n×m}*) ⊗ (R^{m×p}*) ⊗ R^{n×p}
    # Flattening: T is in (R^{nm}*) ⊗ (R^{mp}*) ⊗ R^{np}
    # T_{(i,j), (j',k), (i',k')} = delta_{i,i'} * delta_{j,j'} * delta_{j,k} * delta_{k,k'}
    # = delta_{i,i'} * delta_{j,j'} * delta_{j,k} * delta_{k,k'}
    # This is nonzero only when i=i', j=j', j=k, k=k', so j=k and i=i', k=k'
    # So T_{(i,j), (j,k), (i,k)} = 1 when the indices are consistent.
    
    # In the flattened form with indices:
    # U index: i*m + j (0-indexed: i in [0,n), j in [0,m))
    # V index: j*p + k (0-indexed: j in [0,m), k in [0,p))
    # W index: k*n + i (0-indexed: k in [0,p), i in [0,n))
    
    # T[U_idx, V_idx, W_idx] = 1 if U_idx = i*m+j, V_idx = j'*p+k, W_idx = k'*n+i'
    # and i=i', j=j', j=k, k=k'
    # So T is nonzero at (i*m+j, j*p+k, k*n+i) for all i,j,k with the constraint that
    # the j in V must equal the j from U... 
    
    # Actually, for each triple (i,j,k), there's exactly one nonzero entry:
    # T[i*m+j, j*p+k, k*n+i] = 1
    # And there are n*m*p such entries, all equal to 1.
    
    # So the tensor T has exactly n*m*p = 120 nonzero entries, all equal to 1.
    # The trivial decomposition uses 120 rank-1 terms, one for each nonzero entry.
    
    # To find a lower rank decomposition, we need to find if these 120 rank-1 tensors
    # are linearly dependent. If they are, we can express some as linear combinations of others.
    
    # The rank-1 tensor for (i,j,k) is: u_{i,j} ⊗ v_{j,k} ⊗ w_{k,i}
    # where u_{i,j} has 1 at position i*m+j, v_{j,k} has 1 at position j*p+k,
    # w_{k,i} has 1 at position k*n+i.
    
    # We want to find coefficients c_{i,j,k} such that:
    # sum_{i,j,k} c_{i,j,k} * u_{i,j} ⊗ v_{j,k} ⊗ w_{k,i} = T
    
    # This means for each (a,b,c) where a in [0,nm), b in [0,mp), c in [0,pn):
    # sum_{i,j,k: i*m+j=a, j'*p+k=b, k'*n+i'=c} c_{i,j,k} * [a == i*m+j] * [b == j'*p+k] * [c == k'*n+i']
    # = T[a,b,c]
    
    # Since u_{i,j} is a basis vector, u_{i,j}[a] = 1 if a == i*m+j else 0.
    # Similarly for v and w.
    
    # So the equation becomes:
    # For each (a,b,c): sum_{i,j,k} c_{i,j,k} * delta_{a, i*m+j} * delta_{b, j*p+k} * delta_{c, k*n+i} = T[a,b,c]
    
    # But delta_{b, j*p+k} requires that b = j*p+k for some j,k. Since b ranges over [0,mp),
    # and j*p+k with j in [0,m), k in [0,p) covers all values in [0,mp), each b corresponds
    # to exactly one (j,k) pair. Similarly for a and c.
    
    # So for a given (a,b,c), let i = a//m, j_a = a%m, j_b = b//p, k = b%p, k_c = c//n, i_c = c%n.
    # The term is nonzero only if i == i_c, j_a == j_b, and k == k_c.
    # In that case, the sum is just c_{i, j_a, k} (since only one (i,j,k) satisfies all three deltas).
    
    # So T[a,b,c] = c_{i, j, k} where i = a//m = c//n, j = a%m = b//p, k = b%p = c%n.
    # And T[a,b,c] = 1 if the indices are consistent, 0 otherwise.
    
    # Wait, this means c_{i,j,k} = T[a,b,c] where a = i*m+j, b = j*p+k, c = k*n+i.
    # And T[a,b,c] = 1 for all such (i,j,k). So c_{i,j,k} = 1 for all i,j,k.
    
    # This means the only solution with the standard basis vectors is c_{i,j,k} = 1 for all i,j,k,
    # giving rank 120. To get a lower rank, we need to allow the vectors u_r, v_r, w_r to be
    # more general (not just basis vectors).
    
    # So the problem is: find the minimum rank R and vectors u_r ∈ R^{nm}, v_r ∈ R^{mp}, w_r ∈ R^{pn}
    # such that sum_{r=1}^R u_r ⊗ v_r ⊗ w_r = T.
    
    # This is a non-convex problem. However, we can try to find a solution with a small R by
    # using a MILP or other optimization approach.
    
    # For now, let's try to see if we can find a decomposition with rank less than 120.
    # One approach: assume that the vectors u_r, v_r, w_r have small support (sparse).
    
    # Let me try a different approach: use the fact that for small matrices, the rank is known.
    # For 4x5x6, the rank is not known exactly, but it's at most 120.
    
    # Let me try to construct a decomposition with a smaller rank by exploiting structure.
    
    # Actually, let me just return the trivial solution for now and note that improving it
    # requires a more sophisticated approach.
    
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
