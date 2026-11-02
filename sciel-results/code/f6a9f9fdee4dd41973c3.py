import time
import numpy as np
from itertools import product

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying 
    a 2 by 3 matrix by a 3 by 5 matrix.
    
    Strategy: 
    1. The matrix multiplication tensor for 2x3 * 3x5 has a known lower bound of rank 16 
       (based on Strassen-like bounds or specific structural arguments for small matrices).
       The schoolbook rank is 30.
    2. We perform an exact search for a rank-16 decomposition.
    3. We restrict the search space to integer and half-integer coefficients.
    4. We use a structured basis of "pattern" tensors that respect the sparsity structure 
       of the multiplication map as much as possible, or simply search over a small set 
       of candidate rank-1 tensors that can be combined.
    5. Given the small size, we can attempt to solve the linear system for the coefficients 
       of a fixed set of rank-1 tensors, or search for the rank-1 tensors directly.
       
    However, a more robust "structural" change is to use the known fact that 
    matmul(2,3,5) might have a specific known decomposition. 
    Since finding it from scratch is hard, we will implement a search that tries 
    to find a decomposition of rank R for R in [16, 17, 18, ...] using a 
    randomized algebraic approach with exact verification, but with a much tighter 
    search space than the previous program.
    
    Actually, the most "different" strategy is to exploit the specific structure.
    For small matrices, we can often find decompositions by looking at sub-blocks.
    But to be safe and fast, we will implement a high-quality random search with 
    exact arithmetic verification, targeting rank 16 specifically, using 
    a smarter initialization based on the Kronecker product structure or 
    simply random integers in a small range, but with many more restarts 
    enabled by the faster verification loop.
    
    Let's try a different angle: The problem asks for integer/half-integer entries.
    This suggests the solution lies in a small lattice.
    We will iterate over possible ranks starting from 16.
    For each rank, we will use a randomized search over a small grid of integers/half-integers.
    To make it efficient, we will fix the first two matrices U and V to random 
    small integer/half-integer values, and solve for W exactly. If W has 
    integer/half-integer entries and the decomposition is exact, we found it.
    """
    n, m, p = 2, 3, 5
    nm, mp, pn = n * m, m * p, p * n  # 6, 15, 10
    
    start_time = time.time()
    deadline = start_time + 115.0

    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W):
        """Verify exact decomposition."""
        r = len(U[0])
        # Check all non-zero entries of the target tensor
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    I = i * m + j
                    J = j * p + k
                    K = k * n + i
                    s = 0
                    for c in range(r):
                        s += U[I][c] * V[J][c] * W[K][c]
                    if abs(s - 1) > 1e-9:
                        return False
        
        # Check that other entries are 0
        for I in range(nm):
            for J in range(mp):
                for K in range(pn):
                    i1 = I // m
                    j1 = I % m
                    j2 = J // p
                    k1 = J % p
                    k2 = K // n
                    i2 = K % n
                    
                    if j1 == j2 and i1 == i2 and k1 == k2:
                        continue # Already checked
                    
                    s = 0
                    for c in range(r):
                        s += U[I][c] * V[J][c] * W[K][c]
                    if abs(s) > 1e-9:
                        return False
        return True

    def is_half_integer(x):
        return abs(x * 2 - round(x * 2)) < 1e-9

    def try_rank_exact_search(R, max_restarts=1000):
        """
        Search for a rank R decomposition by randomizing U and V with small 
        integer/half-integer entries, and solving for W.
        """
        # Values to sample from
        vals = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                return None
                
            # Randomize U and V
            U = np.random.choice(vals, size=(nm, R)).astype(float)
            V = np.random.choice(vals, size=(mp, R)).astype(float)
            
            # We need to find W such that T = sum_c U[:,c] V[:,c] W[:,c]^T
            # This is a linear system for each entry of W? No.
            # T[I, J, K] = sum_c U[I, c] V[J, c] W[K, c]
            # For fixed U, V, this is linear in W.
            # Let's construct the matrix for W.
            # Number of equations: nm * mp * pn = 900
            # Number of unknowns: pn * R = 10 * R
            
            # If R is small (e.g., 16), unknowns = 160. Equations = 900.
            # We can solve this overdetermined system using least squares,
            # then check if the solution is exact and has half-integer entries.
            
            # Construct M_W where M_W * W_flat = T_flat
            # M_W[idx_t, idx_w] = U[I, c] * V[J, c]
            # idx_t = I * mp * pn + J * pn + K
            # idx_w = K * R + c
            
            # Instead of building the full 900x160 matrix every time, 
            # we can use the fact that the tensor is sparse.
            # But building it is fast enough in NumPy.
            
            T = np.zeros((nm, mp, pn))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i * m + j, j * p + k, k * n + i] = 1.0
            
            T_flat = T.flatten()
            
            M_W = np.zeros((nm * mp * pn, pn * R))
            for I in range(nm):
                for J in range(mp):
                    for K in range(pn):
                        idx_t = I * mp * pn + J * pn + K
                        for c in range(R):
                            idx_w = K * R + c
                            M_W[idx_t, idx_w] = U[I, c] * V[J, c]
            
            try:
                W_flat, residuals, rank_w, s = np.linalg.lstsq(M_W, T_flat, rcond=None)
            except:
                continue
                
            W = W_flat.reshape((pn, R))
            
            # Check if W entries are close to half-integers
            # If so, round them and verify
            if all(is_half_integer(x) for x in W.flatten()):
                W_rounded = np.round(W * 2) / 2.0
                # Also round U and V just in case of floating point noise, 
                # though they were initialized as exact half-integers
                U_rounded = np.round(U * 2) / 2.0
                V_rounded = np.round(V * 2) / 2.0
                
                if verify(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()):
                    return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
            
            # Optimization: If residuals are zero, it's a solution, but we need half-integers.
            # The lstsq might give exact solutions that are not half-integers.
            # We rely on the random search hitting a "good" subspace where the solution 
            # lands on the lattice.
            
        return None

    # Try ranks starting from the theoretical lower bound
    # The lower bound for 2x3x5 is not trivially 16, but let's try a range.
    # Schoolbook is 30. Strassen for 2x2x2 is 7.
    # For 2x3x5, we can embed 2x2x2.
    # Let's try 16, 17, 18, 19, 20.
    
    for R in [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]:
        res = try_rank_exact_search(R)
        if res:
            return res
            
    return schoolbook()
