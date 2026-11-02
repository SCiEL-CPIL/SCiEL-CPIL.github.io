"""
Search for a rank-minimal exact tensor decomposition of the 3x4 @ 4x5 matrix product.

Strategy:
1. Define the tensor T for matrix multiplication.
2. For a given rank r, initialize U, V, W with small random real values.
3. Use gradient descent to minimize the squared error of the decomposition.
4. Round the resulting real matrices to the nearest half-integers.
5. Verify the rounded decomposition exactly using Fractions.
6. If valid, return the result.
7. Try multiple ranks and random seeds.
"""

import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [3, 4, 5]


def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    schoolbook_rank = n * m * p  # 60

    # Build the target tensor T as a numpy array
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k, else 0
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def compute_loss_and_grads(U, V, W):
        """
        Compute loss and gradients for U, V, W.
        U: (nm, r), V: (mp, r), W: (pn, r)
        """
        r = U.shape[1]
        
        # Reconstructed tensor R[a, b, c] = sum_k U[a,k] V[b,k] W[c,k]
        R = np.zeros_like(T)
        for k in range(r):
            R += U[:, k, None, None] * V[None, :, k, None] * W[None, None, :, k]
            
        E = R - T
        loss = np.sum(E * E)
        
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for k in range(r):
            Prod_VW = np.outer(V[:, k], W[:, k])
            dU[:, k] = 2.0 * np.tensordot(E, Prod_VW, axes=([1, 2], [0, 1]))
            
            Prod_UW = np.outer(U[:, k], W[:, k])
            dV[:, k] = 2.0 * np.tensordot(E, Prod_UW, axes=([0, 2], [0, 1]))
            
            Prod_UV = np.outer(U[:, k], V[:, k])
            dW[:, k] = 2.0 * np.tensordot(E, Prod_UV, axes=([0, 1], [0, 1]))
            
        return loss, dU, dV, dW

    def round_to_half_int(matrix):
        """Round each entry to the nearest half-integer."""
        return np.round(matrix * 2.0) / 2.0

    def verify_exact(U, V, W, n, m, p):
        """
        Verify if U, V, W (lists of lists of int/float) form an exact decomposition.
        """
        r = len(U[0])
        U_frac = [[Fraction(x) for x in row] for row in U]
        V_frac = [[Fraction(x) for x in row] for row in V]
        W_frac = [[Fraction(x) for x in row] for row in W]
        
        # Check all (i, j, k)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for c in range(r):
                        s += U_frac[i * m + j][c] * V_frac[j * p + k][c] * W_frac[k * n + i][c]
                    if s != 1:
                        return False
        
        # Check zero entries
        # Generate all indices
        indices = set()
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    indices.add((i*m+j, j*p+k, k*n+i))
                    
        # To speed up, we can just iterate through all 3600 and skip the 60 known ones
        # But generating the set is fast.
        # 3540 checks.
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    if (a, b, c) in indices:
                        continue
                    
                    s = Fraction(0)
                    for col in range(r):
                        s += U_frac[a][col] * V_frac[b][col] * W_frac[c][col]
                    if s != 0:
                        return False
                        
        return True

    def to_list(mat):
        return mat.tolist()

    def optimize_rank(r, seed, max_iters=200, lr=0.01):
        np.random.seed(seed)
        
        # Initialize with small random values
        U = np.random.randn(nm, r) * 0.5
        V = np.random.randn(mp, r) * 0.5
        W = np.random.randn(pn, r) * 0.5
        
        best_loss = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for i in range(max_iters):
            loss, dU, dV, dW = compute_loss_and_grads(U, V, W)
            
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
            if best_loss < 1e-8:
                break
                
            # Gradient step
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
        # Round the best found solution
        U_rounded = round_to_half_int(best_U)
        V_rounded = round_to_half_int(best_V)
        W_rounded = round_to_half_int(best_W)
        
        # Verify
        if verify_exact(to_list(U_rounded), to_list(V_rounded), to_list(W_rounded), n, m, p):
            return to_list(U_rounded), to_list(V_rounded), to_list(W_rounded)
            
        return None

    deadline = time.time() + 130
    
    # First, ensure we have the schoolbook solution as a fallback
    U_school = [[0] * schoolbook_rank for _ in range(nm)]
    V_school = [[0] * schoolbook_rank for _ in range(mp)]
    W_school = [[0] * schoolbook_rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    # Search for lower ranks
    # We will try ranks 59 down to 40.
    # If we find one, we return it.
    
    for r in range(59, 39, -1):
        if time.time() > deadline - 10:
            break
            
        # Try a few seeds for this rank
        for seed in range(5):
            if time.time() > deadline - 5:
                break
                
            res = optimize_rank(r, seed=1000 + r * 10 + seed, max_iters=100, lr=0.05)
            if res is not None:
                U, V, W = res
                return {"U": U, "V": V, "W": W}
                
    # If no lower rank found, return schoolbook
    return {"U": U_school, "V": V_school, "W": W_school}
