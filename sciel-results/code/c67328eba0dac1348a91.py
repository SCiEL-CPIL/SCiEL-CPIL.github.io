import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    start_time = time.time()
    deadline = start_time + 110  # 10 second safety margin before 120s limit

    # Precompute index mappings
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                indices.append((i * m + j, j * p + k, k * n + i))

    def verify_exact(U, V, W, rank):
        """Verify exact decomposition using Fraction arithmetic."""
        for (ui, vi, wi) in indices:
            s = Fraction(0)
            for r in range(rank):
                u_val = U[ui][r]
                v_val = V[vi][r]
                w_val = W[wi][r]
                # Convert to Fraction if not already
                if isinstance(u_val, float):
                    u_val = Fraction(u_val).limit_denominator(1000)
                if isinstance(v_val, float):
                    v_val = Fraction(v_val).limit_denominator(1000)
                if isinstance(w_val, float):
                    w_val = Fraction(w_val).limit_denominator(1000)
                s += u_val * v_val * w_val
            if s != 1:
                return False
        return True

    def try_algebraic_rank16():
        """
        Try to construct a rank-16 decomposition using structured patterns.
        For 2x3x3, we know rank 16 is achievable. We try several parameterized families.
        """
        # Family 1: Based on Strassen-like ideas adapted for rectangular case
        # C[i,k] = sum_j A[i,j] B[j,k]
        # We can group terms: for each pair (i,k), we need sum over j=0,1,2
        
        # Try a decomposition where we share some terms across j values
        # Let's try: use 2*3 = 6 "base" terms plus 10 shared terms = 16 total
        
        # Actually, let's try a direct approach:
        # C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0] + A[0,2]B[2,0]
        # C[0,1] = A[0,0]B[0,1] + A[0,1]B[1,1] + A[0,2]B[2,1]
        # C[0,2] = A[0,0]B[0,2] + A[0,1]B[1,2] + A[0,2]B[2,2]
        # C[1,0] = A[1,0]B[0,0] + A[1,1]B[1,0] + A[1,2]B[2,0]
        # C[1,1] = A[1,0]B[0,1] + A[1,1]B[1,1] + A[1,2]B[2,1]
        # C[1,2] = A[1,0]B[0,2] + A[1,1]B[1,2] + A[1,2]B[2,2]
        
        # We can factor: define M_j = A[:,j] * B[j,:] for j=0,1,2 (each is 2x3)
        # C = M_0 + M_1 + M_2
        # Each M_j has rank at most 1, so this is rank 3 in a different sense.
        # But for tensor decomposition, we need scalar rank.
        
        # Let's try a different approach: use the fact that we can combine
        # terms that share the same A or B index.
        
        # Try rank 16 with integer entries:
        # We'll use a systematic search over small integer patterns
        
        # Pattern: For each (i,j,k), we assign coefficients such that
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        
        # Try a structured approach with 16 terms:
        # Terms 0-5: one for each (i, k) pair, handling j=0
        # Terms 6-11: one for each (i, k) pair, handling j=1  
        # Terms 12-17: one for each (i, k) pair, handling j=2
        # But that's 18. We need to share some.
        
        # Alternative: use the identity that allows us to merge some terms.
        # For fixed i, the three terms for j=0,1,2 in C[i,k] can potentially
        # be expressed with fewer tensor terms if we use intermediate combinations.
        
        # Let's try: define for each i, k:
        # X[i,k,0] = A[i,0]*B[0,k]
        # X[i,k,1] = A[i,1]*B[1,k]  
        # X[i,k,2] = A[i,2]*B[2,k]
        # C[i,k] = X[i,k,0] + X[i,k,1] + X[i,k,2]
        
        # Now, can we express X[i,k,j] using shared tensor terms?
        # X[i,k,j] = A[i,j] * B[j,k]
        # This is already the schoolbook. To reduce rank, we need to find
        # linear dependencies or use algebraic identities.
        
        # For 2x3x3, a known result is that rank 16 is achievable.
        # Let's try to find it by solving a system of equations.
        
        # We'll set up the problem as finding U, V, W with 16 columns such that
        # for all (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        
        # This is a system of 18 equations (2*3*3) in 16*6 + 16*9 + 16*6 = 96 + 144 + 96 = 336 unknowns.
        # Highly underdetermined, so solutions should exist.
        
        # Let's try to find a solution with small integer entries.
        # We'll use a greedy/structured approach.
        
        rank = 16
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        
        # Strategy: assign one term per (i,j,k) but merge some.
        # There are 18 (i,j,k) tuples. We need to merge 2 of them.
        
        # Try merging: (0,0,0) and (1,1,1) can potentially share if we use
        # a clever coefficient assignment.
        
        # Actually, let's try a different structured approach:
        # Use 6 terms for the "diagonal" of each i-row, and 10 for cross terms.
        
        # Let me try a concrete construction:
        # For i=0:
        #   C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0] + A[0,2]B[2,0]
        #   C[0,1] = A[0,0]B[0,1] + A[0,1]B[1,1] + A[0,2]B[2,1]
        #   C[0,2] = A[0,0]B[0,2] + A[0,1]B[1,2] + A[0,2]B[2,2]
        # For i=1:
        #   C[1,0] = A[1,0]B[0,0] + A[1,1]B[1,0] + A[1,2]B[2,0]
        #   C[1,1] = A[1,0]B[0,1] + A[1,1]B[1,1] + A[1,2]B[2,1]
        #   C[1,2] = A[1,0]B[0,2] + A[1,1]B[1,2] + A[1,2]B[2,2]
        
        # Total 18 terms. To get to 16, we need to eliminate 2 terms.
        
        # One approach: use the identity that for fixed j, the matrix A[:,j] * B[j,:]
        # can be decomposed differently. But this doesn't directly help for scalar rank.
        
        # Let's try a numerical approach: solve for a rank-16 decomposition
        # with small rational entries using least squares, then round.
        
        if time.time() > deadline - 20:
            return None
        
        # Set up the linear system for a given rank
        # Variables: U (6x16), V (9x16), W (6x16)
        # Equations: for each (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
        
        # This is nonlinear. Let's use a different approach:
        # Fix W to be identity-like, then solve for U and V.
        
        # Actually, let's try a simpler structured approach:
        # Use the fact that we can write:
        # A[i,j] * B[j,k] = (1/2) * [A[i,j]*B[j,k] + A[i,j]*B[j,k]]
        # and group terms.
        
        # Let's try a concrete rank-16 construction based on known results:
        # Reference: The border rank of 2x3x3 is 16, and rank 16 is achievable.
        
        # I'll construct it by solving a small system.
        # Let's fix some entries and solve for others.
        
        # Try: use 16 terms where each term contributes to exactly one (i,k) output,
        # but some terms contribute to multiple j values.
        
        # Term r contributes to C[i,k] with coefficient sum_j U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
        # We want this to equal 1 for all (i,k).
        
        # Let's try: for each (i,k), assign 2-3 terms that together give 1.
        # With 6 (i,k) pairs and 16 terms, we can have ~2.67 terms per (i,k).
        
        # Construction:
        # Terms 0-2: contribute to C[0,0]
        # Terms 3-5: contribute to C[0,1]
        # Terms 6-8: contribute to C[0,2]
        # Terms 9-11: contribute to C[1,0]
        # Terms 12-14: contribute to C[1,1]
        # Terms 15: contributes to C[1,2]
        # But we need each term to also be consistent with the j-index.
        
        # This is getting complex. Let me just try a numerical optimization
        # with a better initialization and shorter runtime.
        
        return None  # Fall through to gradient-based search
    
    # Try gradient-based search for rank 16 with Adam optimizer
    def gradient_search(target_rank, max_steps=3000, seed=42):
        if time.time() > deadline - 15:
            return None
        
        np.random.seed(seed)
        
        # Initialize with small random values
        U = np.random.randn(nm, target_rank) * 0.5
        V = np.random.randn(mp, target_rank) * 0.5
        W = np.random.randn(pn, target_rank) * 0.5
        
        # Adam optimizer state
        mU = np.zeros_like(U)
        vU = np.zeros_like(U)
        mV = np.zeros_like(V)
        vV = np.zeros_like(V)
        mW = np.zeros_like(W)
        vW = np.zeros_like(W)
        t = 0
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        lr = 0.01
        best_err = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for step in range(max_steps):
            if time.time() > deadline - 5:
                break
            
            # Compute gradients
            dU = np.zeros((nm, target_rank))
            dV = np.zeros((mp, target_rank))
            dW = np.zeros((pn, target_rank))
            
            for (ui, vi, wi) in indices:
                VW = V[vi] * W[wi]
                UW = U[ui] * W[wi]
                UV = U[ui] * V[vi]
                T_val = np.dot(U[ui], VW)
                err = T_val - 1.0
                
                dU[ui] += err * VW
                dV[vi] += err * UW
                dW[wi] += err * UV
            
            # Adam update
            t += 1
            mU = beta1 * mU + (1 - beta1) * dU
            vU = beta2 * vU + (1 - beta2) * dU ** 2
            mU_hat = mU / (1 - beta1 ** t)
            vU_hat = vU / (1 - beta2 ** t)
            U -= lr * mU_hat / (np.sqrt(vU_hat) + eps)
            
            mV = beta1 * mV + (1 - beta1) * dV
            vV = beta2 * vV + (1 - beta2) * dV ** 2
            mV_hat = mV / (1 - beta1 ** t)
            vV_hat = vV / (1 - beta2 ** t)
            V -= lr * mV_hat / (np.sqrt(vV_hat) + eps)
            
            mW = beta1 * mW + (1 - beta1) * dW
            vW = beta2 * vW + (1 - beta2) * dW ** 2
            mW_hat = mW / (1 - beta1 ** t)
            vW_hat = vW / (1 - beta2 ** t)
            W -= lr * mW_hat / (np.sqrt(vW_hat) + eps)
            
            # Decay learning rate
            if step % 200 == 0:
                lr *= 0.95
            
            # Check error every 50 steps
            if step % 50 == 0:
                max_err = 0.0
                for (ui, vi, wi) in indices:
                    T_val = np.dot(U[ui], V[vi] * W[wi])
                    max_err = max(max_err, abs(T_val - 1.0))
                if max_err < best_err:
                    best_err = max_err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                if best_err < 1e-6:
                    break
        
        # Use best found solution
        U, V, W = best_U, best_V, best_W
        
        # Final error check
        max_err = 0.0
        for (ui, vi, wi) in indices:
            T_val = np.dot(U[ui], V[vi] * W[wi])
            max_err = max(max_err, abs(T_val - 1.0))
        
        if max_err > 1e-4:
            return None
        
        # High-precision polish pass with smaller learning rate
        for polish_step in range(500):
            if time.time() > deadline - 5:
                break
            
            dU = np.zeros((nm, target_rank))
            dV = np.zeros((mp, target_rank))
            dW = np.zeros((pn, target_rank))
            
            for (ui, vi, wi) in indices:
                VW = V[vi] * W[wi]
                UW = U[ui] * W[wi]
                UV = U[ui] * V[vi]
                T_val = np.dot(U[ui], VW)
                err = T_val - 1.0
                
                dU[ui] += err * VW
                dV[vi] += err * UW
                dW[wi] += err * UV
            
            U -= 0.001 * dU
            V -= 0.001 * dV
            W -= 0.001 * dW
        
        # Try to round to rationals with various denominators
        for denom in [1, 2, 4, 8, 16, 32, 64]:
            U_r = np.round(U * denom) / denom
            V_r = np.round(V * denom) / denom
            W_r = np.round(W * denom) / denom
            
            # Verify
            valid = True
            for (ui, vi, wi) in indices:
                s = 0.0
                for r in range(target_rank):
                    s += U_r[ui, r] * V_r[vi, r] * W_r[wi, r]
                if abs(s - 1.0) > 1e-10:
                    valid = False
                    break
            
            if valid:
                # Convert to lists
                U_out = [[float(U_r[i, j]) for j in range(target_rank)] for i in range(nm)]
                V_out = [[float(V_r[i, j]) for j in range(target_rank)] for i in range(mp)]
                W_out = [[float(W_r[i, j]) for j in range(target_rank)] for i in range(pn)]
                return U_out, V_out, W_out
        
        return None
    
    # Try rank 16 with multiple seeds and different initializations
    seeds = [42, 123, 456, 789, 1000, 2000, 3000, 4000]
    for seed in seeds:
        if time.time() > deadline - 10:
            break
        result = gradient_search(16, max_steps=2500, seed=seed)
        if result is not None:
            U_out, V_out, W_out = result
            return {"U": U_out, "V": V_out, "W": W_out}
    
    # Fallback: schoolbook decomposition (rank 18)
    rank = n * m * p
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * m * p + j * p + k
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
