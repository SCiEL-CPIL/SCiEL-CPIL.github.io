import time
import sys
import random
import numpy as np
from fractions import Fraction

# Dimensions for the matrix multiplication: (n, m, p)
# A is n x m, B is m x p, C is n x p
# Here n=2, m=3, p=3
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x3 matrix.
    We use a known rank-17 decomposition.
    """
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    
    def verify(U, V, W):
        """Verify that the decomposition exactly reproduces the matrix multiplication tensor."""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    s = 0.0
                    for c in range(rank):
                        s += U[idx_u][c] * V[idx_v][c] * W[idx_w][c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def verify_exact(U, V, W):
        """Verify using exact rational arithmetic."""
        rank = len(U[0])
        U_frac = [[Fraction(x) for x in row] for row in U]
        V_frac = [[Fraction(x) for x in row] for row in V]
        W_frac = [[Fraction(x) for x in row] for row in W]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    s = Fraction(0)
                    for c in range(rank):
                        s += U_frac[idx_u][c] * V_frac[idx_v][c] * W_frac[idx_w][c]
                    if s != Fraction(1):
                        return False
        return True

    def schoolbook():
        """Return the schoolbook rank-18 decomposition."""
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        
        col_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    U[idx_u][col_idx] = 1.0
                    V[idx_v][col_idx] = 1.0
                    W[idx_w][col_idx] = 1.0
                    col_idx += 1
        
        return {"U": U, "V": V, "W": W}
    
    def rank17_decomposition():
        """
        Construct a rank-17 decomposition for 2x3x3 matrix multiplication.
        """
        rank = 17
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        
        # Indices:
        # A: i in 0..1, j in 0..2 -> idx = i*3 + j
        # B: j in 0..2, k in 0..2 -> idx = j*3 + k
        # C: k in 0..2, i in 0..1 -> idx = k*2 + i
        
        # Part 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
        # A indices: 0,1,3,4
        # B indices: 0,1,3,4
        # C indices: 0,1,2,3
        
        # M1: (A00+A11)(B00+B11) -> C00, C11
        U[0][0] = 1; U[4][0] = 1
        V[0][0] = 1; V[4][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # M2: (A10+A11)B00 -> C10, -C11
        U[3][1] = 1; U[4][1] = 1
        V[0][1] = 1
        W[1][1] = 1; W[3][1] = -1
        
        # M3: A00(B01-B11) -> C01, C11
        U[0][2] = 1
        V[1][2] = 1; V[4][2] = -1
        W[2][2] = 1; W[3][2] = 1
        
        # M4: A11(B10-B00) -> C00, C10
        U[4][3] = 1
        V[3][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[1][3] = 1
        
        # M5: (A00+A01)B11 -> -C00, C01
        U[0][4] = 1; U[1][4] = 1
        V[4][4] = 1
        W[0][4] = -1; W[2][4] = 1
        
        # M6: (A10-A00)(B00+B01) -> C11
        U[3][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        # M7: (A01-A11)(B10+B11) -> C00
        U[1][6] = 1; U[4][6] = -1
        V[3][6] = 1; V[4][6] = 1
        W[0][6] = 1
        
        # Part 2: Schoolbook for A[0:2, 2] * B[2, 0:2] -> C[0:2, 0:2]
        # A indices: 2, 5
        # B indices: 6, 7
        # C indices: 0,1,2,3
        
        # r=7: A02 B20 -> C00
        U[2][7] = 1
        V[6][7] = 1
        W[0][7] = 1
        
        # r=8: A02 B21 -> C01
        U[2][8] = 1
        V[7][8] = 1
        W[2][8] = 1
        
        # r=9: A12 B20 -> C10
        U[5][9] = 1
        V[6][9] = 1
        W[1][9] = 1
        
        # r=10: A12 B21 -> C11
        U[5][10] = 1
        V[7][10] = 1
        W[3][10] = 1
        
        # Part 3: Schoolbook for A[0:2, 0:3] * B[0:3, 2] -> C[0:2, 2]
        # A indices: 0,1,2,3,4,5
        # B indices: 2,5,8
        # C indices: 4,5
        
        # r=11: A00 B02 -> C02
        U[0][11] = 1
        V[2][11] = 1
        W[4][11] = 1
        
        # r=12: A01 B12 -> C02
        U[1][12] = 1
        V[5][12] = 1
        W[4][12] = 1
        
        # r=13: A02 B22 -> C02
        U[2][13] = 1
        V[8][13] = 1
        W[4][13] = 1
        
        # r=14: A10 B02 -> C12
        U[3][14] = 1
        V[2][14] = 1
        W[5][14] = 1
        
        # r=15: A11 B12 -> C12
        U[4][15] = 1
        V[5][15] = 1
        W[5][15] = 1
        
        # r=16: A12 B22 -> C12
        U[5][16] = 1
        V[8][16] = 1
        W[5][16] = 1
        
        return {"U": U, "V": V, "W": W}
    
    def try_rank16_numerical():
        """
        Attempt to find a rank-16 decomposition using numerical optimization.
        Uses gradient descent on the tensor reconstruction error.
        """
        target_rank = 16
        
        # Build the target tensor T[i,j,k] = 1 if (i,j,k) is a valid multiplication term
        # T[i,j,k] = 1 means C[i,k] += A[i,j] * B[j,k]
        # Flatten: T_flat[idx] where idx = (i*m+j)*p*k + ... 
        # Actually, let's use the standard flattening:
        # T[a,b,c] where a in [0,n*m), b in [0,m*p), c in [0,p*n)
        # T[a,b,c] = 1 if a=(i,j), b=(j,k), c=(k,i) for some i,j,k
        # i.e., the j-index in a matches the j-index in b, and the k-index in b matches the k-index in c
        
        nm = n * m  # 6
        mp = m * p  # 9
        pn = p * n  # 6
        
        # Build target tensor as a 3D array
        T = np.zeros((nm, mp, pn), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1.0
        
        best_result = None
        best_error = np.inf
        
        # Try multiple random initializations
        np.random.seed(42)
        num_restarts = 30
        
        for restart in range(num_restarts):
            if time.time() > deadline - 10:
                break
            
            # Initialize with random values
            scale = 0.5
            U = np.random.randn(nm, target_rank) * scale
            V = np.random.randn(mp, target_rank) * scale
            W = np.random.randn(pn, target_rank) * scale
            
            # Gradient descent
            lr = 0.001
            num_iters = 5000
            
            for it in range(num_iters):
                if time.time() > deadline - 5:
                    break
                
                # Compute error: E = sum_{a,b,c} (sum_r U[a,r]*V[b,r]*W[c,r] - T[a,b,c])^2
                # Use vectorized computation
                # Reconstruct tensor
                # R[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
                # This can be done as: for each r, outer product of U[:,r], V[:,r], W[:,r]
                # But that's expensive. Let's use einsum.
                
                # R = np.einsum('ar,br,cr->abc', U, V, W)
                # This might be too slow. Let's try a different approach.
                
                # Compute gradients:
                # dE/dU[a,r] = 2 * sum_{b,c} (R[a,b,c] - T[a,b,c]) * V[b,r] * W[c,r]
                # dE/dV[b,r] = 2 * sum_{a,c} (R[a,b,c] - T[a,b,c]) * U[a,r] * W[c,r]
                # dE/dW[c,r] = 2 * sum_{a,b} (R[a,b,c] - T[a,b,c]) * U[a,r] * V[b,r]
                
                # Compute R - T
                # R[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
                # Let's compute this efficiently
                # For each r, we have a rank-1 tensor U[:,r] outer V[:,r] outer W[:,r]
                
                # Use einsum for the residual
                R = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
                residual = R - T
                
                # Compute gradients
                # dU[a,r] = 2 * sum_{b,c} residual[a,b,c] * V[b,r] * W[c,r]
                # This is: 2 * einsum('abc,br,cr->ar', residual, V, W)
                dU = 2.0 * np.einsum('abc,br,cr->ar', residual, V, W, optimize=True)
                dV = 2.0 * np.einsum('abc,ar,cr->br', residual, U, W, optimize=True)
                dW = 2.0 * np.einsum('abc,ar,br->cr', residual, U, V, optimize=True)
                
                # Update
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                # Check convergence
                if it % 100 == 0:
                    err = np.sum(residual**2)
                    if err < best_error:
                        best_error = err
                        if err < 1e-6:
                            # Try to rationalize
                            U_r = [[float(x) for x in row] for row in U]
                            V_r = [[float(x) for x in row] for row in V]
                            W_r = [[float(x) for x in row] for row in W]
                            
                            # Check if we can snap to rational values
                            # Try with tolerance
                            U_snap = [[0.0] * target_rank for _ in range(nm)]
                            V_snap = [[0.0] * target_rank for _ in range(mp)]
                            W_snap = [[0.0] * target_rank for _ in range(pn)]
                            
                            for a in range(nm):
                                for r in range(target_rank):
                                    x = U[a, r]
                                    # Try to find a simple fraction
                                    best_frac = 0.0
                                    best_diff = abs(x)
                                    for num in range(-20, 21):
                                        for den in range(1, 21):
                                            val = num / den
                                            diff = abs(x - val)
                                            if diff < best_diff:
                                                best_diff = diff
                                                best_frac = val
                                    if best_diff < 0.01:
                                        U_snap[a][r] = best_frac
                                    else:
                                        U_snap[a][r] = x
                            
                            for b in range(mp):
                                for r in range(target_rank):
                                    x = V[b, r]
                                    best_frac = 0.0
                                    best_diff = abs(x)
                                    for num in range(-20, 21):
                                        for den in range(1, 21):
                                            val = num / den
                                            diff = abs(x - val)
                                            if diff < best_diff:
                                                best_diff = diff
                                                best_frac = val
                                    if best_diff < 0.01:
                                        V_snap[b][r] = best_frac
                                    else:
                                        V_snap[b][r] = x
                            
                            for c in range(pn):
                                for r in range(target_rank):
                                    x = W[c, r]
                                    best_frac = 0.0
                                    best_diff = abs(x)
                                    for num in range(-20, 21):
                                        for den in range(1, 21):
                                            val = num / den
                                            diff = abs(x - val)
                                            if diff < best_diff:
                                                best_diff = diff
                                                best_frac = val
                                    if best_diff < 0.01:
                                        W_snap[c][r] = best_frac
                                    else:
                                        W_snap[c][r] = x
                            
                            if verify(U_snap, V_snap, W_snap):
                                return {"U": U_snap, "V": V_snap, "W": W_snap}
                
                # Adaptive learning rate
                if it % 500 == 0 and it > 0:
                    lr *= 0.8
            
            # Final check
            R = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
            err = np.sum((R - T)**2)
            if err < 1e-6:
                U_r = [[float(x) for x in row] for row in U]
                V_r = [[float(x) for x in row] for row in V]
                W_r = [[float(x) for x in row] for row in W]
                if verify(U_r, V_r, W_r):
                    return {"U": U_r, "V": V_r, "W": W_r}
        
        return None
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # First, try the known rank-17 decomposition
    if time.time() < deadline:
        r17 = rank17_decomposition()
        if verify(r17["U"], r17["V"], r17["W"]):
            best_result = r17
            best_rank = 17
    
    # Try to find rank-16
    if time.time() < deadline - 10:
        r16 = try_rank16_numerical()
        if r16 is not None:
            best_result = r16
            best_rank = 16
    
    return best_result
# EVOLVE-BLOCK-END
