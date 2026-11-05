"""Matrix multiplication 2,5,6 - rank-minimal tensor decomposition."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 5, 6]
START_TIME = time.time()
DEADLINE = START_TIME + 120

def _time_remaining():
    return DEADLINE - time.time()

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS  # n=2, m=5, p=6
    
    # Build the matrix multiplication tensor as a sparse linear system
    # The tensor T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # For each (i,j,k), we need: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
    
    # This is a non-linear system. Instead, we use the fact that for a rank-r
    # decomposition, each term r contributes a rank-1 tensor:
    # T_r[i,j,k] = U[i,j,r] * V[j,k,r] * W[k,i,r]
    # and we need sum_r T_r = T_mm where T_mm[i,j,k] = 1 for all valid (i,j,k)
    
    # For the ILP approach, we flatten and use the structure:
    # The matrix multiplication tensor has entries T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # A rank-r decomposition means: sum_{r=1}^{r} u_r[i,j] * v_r[j,k] * w_r[k,i] = 1
    
    # Let's build the linear system for a given rank r.
    # For each (i,j,k): sum_{r=1}^{R} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # This is still non-linear. 
    # 
    # Alternative: Use the fact that the tensor can be viewed as a linear map.
    # The multiplication tensor is a trilinear form. For rank r, we need:
    # For all (i,j,k): sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1
    #
    # Let's try a different approach: enumerate possible sparse rank-1 tensors
    # and find a combination that sums to the MM tensor.
    #
    # Actually, let's use the standard approach: the MM tensor for (n,m,p) 
    # can be decomposed. The key insight is that we can formulate this as:
    # Find the minimum r such that there exist matrices U (nm x r), V (mp x r), W (pn x r)
    # with integer/half-integer entries satisfying the tensor equation.
    #
    # For small dimensions, we can try to find the minimum rank by checking
    # ranks from a lower bound upward.
    #
    # Lower bound: max(n*m, m*p, p*n) = max(10, 30, 12) = 30
    # But the actual minimum is likely higher.
    
    # Let's use a smarter approach: build the constraint matrix for the ILP.
    # For a fixed rank r, we have r columns. Each column (rank-1 term) is defined
    # by three vectors: u in R^(nm), v in R^(mp), w in R^(pn).
    # The constraint is: for each (i,j,k), sum_{r=1}^R u[i*m+j]*v[j*p+k]*w[k*n+i] = 1
    #
    # This is a system of n*m*p = 60 equations with 3*r*r variables (if we allow
    # arbitrary real values). But we want integer/half-integer solutions.
    #
    # Let's try a different formulation. The tensor T has entries T[i,j,k]=1.
    # We can write T as a sum of rank-1 tensors. A rank-1 tensor is determined
    # by three vectors. The question is: what is the minimum number of rank-1
    # tensors (with integer/half-integer coefficients) that sum to T?
    #
    # For the 2x5x6 case, let's try to find the minimum rank by solving
    # a series of ILPs or using a greedy approach.
    
    # First, let's verify the Strassen-based rank-54 solution works, then try to improve.
    
    def verify(U, V, W):
        """Verify the decomposition is correct."""
        r = U.shape[1]
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        np.random.seed(42)
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_expected = A_test @ B_test
        
        U_reshaped = U_np.reshape(n, m, r)
        V_reshaped = V_np.reshape(m, p, r)
        W_reshaped = W_np.reshape(p, n, r)
        
        T = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
        C_computed = np.einsum('ijk,ij,jk->ik', T, A_test, B_test)
        
        return np.allclose(C_computed, C_expected, atol=1e-10)
    
    # Build the Strassen-based rank-54 solution
    def build_strassen_rank54():
        rank_54 = 7 * 2 * 3 + 4 * 3  # 42 + 12 = 54
        U = np.zeros((n * m, rank_54), dtype=int)
        V = np.zeros((m * p, rank_54), dtype=int)
        W = np.zeros((p * n, rank_54), dtype=int)
        
        col = 0
        
        def add_strassen(i_base, j_base, k_base):
            nonlocal col
            U[i_base*m + j_base, col] += 1
            U[(i_base+1)*m + (j_base+1), col] += 1
            V[j_base*p + k_base, col] += 1
            V[(j_base+1)*p + (k_base+1), col] += 1
            W[k_base*n + i_base, col] += 1
            W[(k_base+1)*n + (i_base+1), col] += 1
            col += 1
            
            U[(i_base+1)*m + j_base, col] += 1
            U[(i_base+1)*m + (j_base+1), col] += 1
            V[j_base*p + k_base, col] += 1
            W[k_base*n + (i_base+1), col] += 1
            W[(k_base+1)*n + (i_base+1), col] -= 1
            col += 1
            
            U[i_base*m + j_base, col] += 1
            V[j_base*p + (k_base+1), col] += 1
            V[(j_base+1)*p + (k_base+1), col] -= 1
            W[(k_base+1)*n + i_base, col] += 1
            W[(k_base+1)*n + (i_base+1), col] += 1
            col += 1
            
            U[(i_base+1)*m + (j_base+1), col] += 1
            V[(j_base+1)*p + k_base, col] += 1
            V[j_base*p + k_base, col] -= 1
            W[k_base*n + i_base, col] += 1
            W[k_base*n + (i_base+1), col] += 1
            col += 1
            
            U[i_base*m + j_base, col] += 1
            U[i_base*m + (j_base+1), col] += 1
            V[(j_base+1)*p + (k_base+1), col] += 1
            W[k_base*n + i_base, col] -= 1
            W[(k_base+1)*n + i_base, col] += 1
            col += 1
            
            U[(i_base+1)*m + j_base, col] += 1
            U[i_base*m + j_base, col] -= 1
            V[j_base*p + k_base, col] += 1
            V[j_base*p + (k_base+1), col] += 1
            W[(k_base+1)*n + (i_base+1), col] += 1
            col += 1
            
            U[i_base*m + (j_base+1), col] += 1
            U[(i_base+1)*m + (j_base+1), col] -= 1
            V[(j_base+1)*p + k_base, col] += 1
            V[(j_base+1)*p + (k_base+1), col] += 1
            W[k_base*n + i_base, col] += 1
            col += 1
        
        def add_naive_2x1x2(i_base, j_idx, k_base):
            nonlocal col
            for di in range(2):
                for dk in range(2):
                    U[(i_base+di)*m + j_idx, col] += 1
                    V[j_idx*p + (k_base+dk), col] += 1
                    W[(k_base+dk)*n + (i_base+di), col] += 1
                    col += 1
        
        for j_base in [0, 2]:
            for k_base in [0, 2, 4]:
                add_strassen(0, j_base, k_base)
        
        for k_base in [0, 2, 4]:
            add_naive_2x1x2(0, 4, k_base)
        
        return U, V, W
    
    # Try to find a better decomposition using ILP-like search
    # The key idea: the MM tensor for (2,5,6) can potentially be decomposed
    # with rank less than 54. Let's try to find the minimum.
    
    # Approach: Use the structure of the problem. The tensor T[i,j,k]=1 for all
    # valid indices. We can try to find a low-rank decomposition by:
    # 1. Starting from the rank-54 solution
    # 2. Trying to merge or eliminate terms
    
    # Alternative: Use a direct search for the minimum rank
    # For rank r, we need to find U, V, W with r columns such that
    # sum_{r'} U[i*m+j, r'] * V[j*p+k, r'] * W[k*n+i, r'] = 1 for all (i,j,k)
    
    # Let's try a systematic approach: build the constraint matrix and solve
    # using a branch-and-bound or similar method.
    
    # For now, let's try to improve the rank-54 solution by checking if
    # we can reduce the rank further using a greedy elimination approach.
    
    # Actually, let's try a completely different approach: use the fact that
    # the problem is equivalent to finding the tensor rank of the MM tensor.
    # For (2,5,6), the known bounds suggest the rank might be around 40-50.
    
    # Let's try to construct a rank-48 or rank-50 solution by modifying
    # the Strassen approach.
    
    # Idea: Instead of using 3 separate k-pairs for each j-pair, we can
    # try to share some computations across k-pairs.
    
    # For the j=4 row (single j index), we use 4 terms per k-pair = 12 terms total.
    # Can we reduce this? The 2x1x2 block has rank at most 4 (schoolbook),
    # but maybe we can do better by sharing with other blocks.
    
    # Let's try a different partitioning:
    # j indices: 0,1,2,3,4
    # k indices: 0,1,2,3,4,5
    # i indices: 0,1
    
    # What if we use Strassen for all j-pairs and handle j=4 differently?
    # Or what if we use a different algorithm for some blocks?
    
    # Let's try: use rank-7 Strassen for (j=0,1) x (k=0,1), (j=0,1) x (k=2,3),
    # (j=0,1) x (k=4,5), (j=2,3) x (k=0,1), (j=2,3) x (k=2,3), (j=2,3) x (k=4,5)
    # That's 6 * 7 = 42 terms.
    # For j=4: we need 2x1x6. We can use 3 pairs of k, each with rank 4 = 12 terms.
    # Total: 54.
    
    # Can we do better? What if we use a different decomposition for the j=4 part?
    # The 2x1x6 tensor (for fixed j=4) has rank at most 12 (schoolbook).
    # But we might be able to share some terms with the Strassen blocks.
    
    # Actually, let's try a completely different approach: use the ILP formulation
    # to find the minimum rank.
    
    # The ILP formulation:
    # Variables: x_{i,j,k,r} in {0, 1/2, 1} for each (i,j,k,r)
    # But this is too large. Instead, we can use the structure.
    
    # Let's try a heuristic: start from the rank-54 solution and try to
    # eliminate terms by checking if they can be represented as a linear
    # combination of other terms.
    
    # For now, let's just return the rank-54 solution, which is known to work.
    # We can try to improve it in future iterations.
    
    U, V, W = build_strassen_rank54()
    
    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook
        rank_school = n * m * p
        U_school = np.zeros((n * m, rank_school), dtype=int)
        V_school = np.zeros((m * p, rank_school), dtype=int)
        W_school = np.zeros((p * n, rank_school), dtype=int)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i * m * p + j * p + k
                    U_school[i*m+j, idx] = 1
                    V_school[j*p+k, idx] = 1
                    W_school[k*n+i, idx] = 1
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}

# EVOLVE-BLOCK-END
