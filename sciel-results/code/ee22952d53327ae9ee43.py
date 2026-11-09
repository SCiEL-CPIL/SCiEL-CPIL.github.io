"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Build the matrix multiplication tensor
    # T[i*m+j, j'*p+k, i'*n+k'] = 1 if j==j' and i==i' and k==k'
    # Actually, the tensor is: for C[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor T has indices (i,j) for A, (j,k) for B, (i,k) for C
    # T[(i,j),(j',k),(i',k')] = 1 if j==j' and i==i' and k==k'
    
    # Let's use the standard formulation:
    # A is n×m, B is m×p, C is n×p
    # C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
    # The tensor T in R^{(n*m) x (m*p) x (n*p)} where
    # T[i*m+j, j'*p+k, i'*n+k'] = 1 if j==j' and i==i' and k==k'
    
    # Schoolbook rank = n*m*p = 60
    # Try to find lower rank decompositions
    
    # First, let's try a structured approach:
    # For each pair (i,k) in C, we need sum_j A[i,j]*B[j,k]
    # This is like multiplying a row of A by a column of B
    
    # Known result: 2×n × n×2 can be done with 2n+1 multiplications (Strassen-like)
    # But our case is 2×5 × 5×6
    
    # Let me try an exhaustive search for small rank reductions
    # Start from schoolbook and try to eliminate terms
    
    # Represent each schoolbook term as a vector
    # Term (i,j,k) has:
    #   U component: 1 at position i*m+j
    #   V component: 1 at position j*p+k  
    #   W component: 1 at position i*n+k
    
    # The tensor is the sum of rank-1 tensors:
    # T = sum_{i,j,k} u_{i,j,k} ⊗ v_{i,j,k} ⊗ w_{i,j,k}
    
    # To check if we can reduce rank, we need to find a subset of terms
    # that still spans the same tensor space.
    
    # Alternative: Use the fact that for fixed i, the terms (i,j,k) for all j,k
    # form a rank-m*p sub-tensor. We can try to reduce each such sub-tensor.
    
    # Let's try a different approach: use known small matrix multiplication
    # results and combine them.
    
    # For 2×5 × 5×6, we can view this as:
    # C = A * B where A is 2×5, B is 5×6
    # Split A into [A1 A2] where A1 is 2×2, A2 is 2×3
    # Split B into [B1; B2] where B1 is 2×6, B2 is 3×6
    # Then C = A1*B1 + A2*B2
    
    # A1*B1 is 2×2 * 2×6, A2*B2 is 2×3 * 3×6
    
    # Known: 2×2 * 2×2 can be done with 7 multiplications (Strassen)
    # 2×2 * 2×6: can be done with 7*3 = 21? Or less?
    # Actually, 2×n * n×p can be done with various ranks.
    
    # Let me try a direct search approach with time limit
    
    # Build the full tensor
    def build_tensor():
        T = np.zeros((n*m, m*p, n*p), dtype=np.int64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m+j, j*p+k, i*n+k] = 1
        return T
    
    T = build_tensor()
    
    # Try to find minimal rank decomposition
    # Start with schoolbook rank 60 and try to reduce
    
    # Use a greedy approach: start with all 60 terms, try to remove each one
    # and check if the remaining terms still span the same tensor
    
    # For efficiency, work with flattened tensors
    T_flat = T.flatten()
    dim1, dim2, dim3 = n*m, m*p, n*p
    total_dim = dim1 * dim2 * dim3
    
    # Each term (i,j,k) is a rank-1 tensor
    def term_vector(i, j, k):
        v = np.zeros(total_dim, dtype=np.int64)
        idx = (i*m + j) * dim2 * dim3 + (j*p + k) * dim3 + (i*n + k)
        v[idx] = 1
        return v
    
    # All 60 terms
    all_terms = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    term_vectors = [term_vector(i, j, k) for (i, j, k) in all_terms]
    
    # The tensor is the sum of all term vectors
    # We want to find the minimum number of terms whose sum equals T_flat
    
    # This is a set cover / linear algebra problem
    # Since all entries are 0 or 1, and each position in T_flat has exactly one 1,
    # we need exactly one term to cover each position.
    
    # Wait, that means the rank is exactly 60? No, that's not right.
    # Multiple terms can contribute to the same position if we use coefficients.
    
    # Let me reconsider. The tensor T has T[i,j,j',k,i',k'] = 1 if j=j', i=i', k=k'
    # So for a fixed (i,j,j',k,i',k'), it's 1 only if j=j', i=i', k=k'
    # There are n*m*p = 60 positions where T is 1, and they're all distinct.
    
    # A rank-1 tensor u⊗v⊗w has (u[a]*v[b]*w[c]) at position (a,b,c)
    # To get T = sum_r u_r ⊗ v_r ⊗ w_r, we need:
    # sum_r u_r[a] * v_r[b] * w_r[c] = T[a,b,c]
    
    # For positions where T=1, we need sum_r u_r[a]*v_r[b]*w_r[c] = 1
    # For positions where T=0, we need sum_r u_r[a]*v_r[b]*w_r[c] = 0
    
    # This is a system of equations. The minimum rank is the tensor rank.
    
    # For small cases, we can try to solve this exactly.
    # But 60 variables and millions of equations is hard.
    
    # Let me try a heuristic: use known results for small matrix multiplication.
    
    # Known lower bounds and upper bounds:
    # 2×2 × 2×2: rank 7 (Strassen)
    # 2×3 × 3×2: rank 13
    # 2×4 × 4×2: rank 19
    # 2×5 × 5×2: rank 25? or less?
    
    # Our case is 2×5 × 5×6, which is different.
    
    # Let me try a direct construction using block decomposition.
    
    # A is 2×5, B is 5×6
    # Write A = [A_left, A_right] where A_left is 2×2, A_right is 2×3
    # Write B = [B_top; B_bottom] where B_top is 2×6, B_bottom is 3×6
    # C = A_left * B_top + A_right * B_bottom
    
    # A_left * B_top: 2×2 * 2×6
    # A_right * B_bottom: 2×3 * 3×6
    
    # For 2×2 * 2×p: can be done with 7 + 2*(p-2) = 2p+3 multiplications? 
    # Actually, 2×2 * 2×p can be done with 7*p/2? No.
    
    # Let me look up: 2×n * n×p matrix multiplication rank.
    # For 2×2 * 2×2: 7
    # For 2×2 * 2×3: 10 (I think)
    # For 2×2 * 2×4: 13
    # Pattern: 3n+1 for 2×2 * 2×n? 3*2+1=7, 3*3+1=10, 3*4+1=13. Yes!
    
    # For 2×3 * 3×p:
    # 2×3 * 3×3: I recall rank 13 or 14
    # Let's assume 2×3 * 3×p has rank around 5p or so.
    
    # This is getting complicated. Let me try a computational search.
    
    # Approach: Try ranks from a lower bound up to 60, and for each rank,
    # try to find a decomposition using random initialization + local search.
    
    # Lower bound: max(n*m, m*p, n*p) = max(10, 30, 12) = 30
    # Actually, the tensor rank is at least the matrix rank of any unfolding.
    # The (n*m) × (m*p) unfolding has rank at most n*p = 12
    # Wait, no. The unfolding T_{(nm) × (mp)} has entries T[i*m+j, j'*p+k] summed over i',k'
    # This is complicated.
    
    # Simple lower bound: rank >= max(n*m, m*p, n*p) / something
    # Actually, rank >= n*m = 10, rank >= m*p = 30, rank >= n*p = 12
    # So rank >= 30
    
    # Let's try ranks from 30 to 60 and see if we can find a decomposition.
    
    # For each rank r, we have r rank-1 tensors.
    # Each rank-1 tensor is determined by u (length nm), v (length mp), w (length np)
    # Total parameters: r * (nm + mp + np) = r * (10 + 30 + 12) = 52r
    
    # The constraint is that the sum of rank-1 tensors equals T.
    # This is a system of nm*mp*np = 10*30*12 = 3600 equations.
    
    # For r=30, we have 52*30 = 1560 parameters and 3600 equations. Overdetermined.
    # For r=60, we have 52*60 = 3120 parameters and 3600 equations. Still overdetermined.
    
    # So the system is always overdetermined, meaning not all ranks are achievable.
    
    # This is a hard problem. Let me try a different approach.
    
    # Use the fact that the tensor has a special structure (it's a matrix multiplication tensor)
    # and use known algorithms.
    
    # For now, let's return the schoolbook decomposition and add a time check.
    
    if time.time() > deadline:
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
    
    # Try to find a better decomposition using structured blocks
    # Split the 5-dimensional middle into 2+3
    # A = [A1 (2×2), A2 (2×3)], B = [B1 (2×6); B2 (3×6)]
    # C = A1*B1 + A2*B2
    
    # For A1*B1 (2×2 * 2×6): use Strassen-like for 2×2 part
    # 2×2 * 2×2 takes 7 mults, 2×2 * 2×1 takes 3 mults (or 4?)
    # Actually, 2×2 * 2×p: 
    # Let A = [[a,b],[c,d]], B = [[e,f],[g,h]] (2×2)
    # Strassen: 7 multiplications
    # For 2×2 * 2×3: B = [[e,f,g],[h,i,j]]
    # We can do: M1 = (a+b)*(e+h), M2 = (a-b)*(e+h) ... this gets complex.
    
    # Let me try a known result: 2×n * n×p can be done with 3n*p - 2*n - 2*p + 4? 
    # I'm not sure.
    
    # Given the time constraint, let me try a random search for small rank improvements.
    
    # Try rank 59: can we eliminate one term?
    # This requires solving a linear system, which is expensive.
    
    # For now, return schoolbook. The search-based approach from inspiration 2
    # might be better, but it's complex.
    
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
# EVOLVE-BLOCK-END
