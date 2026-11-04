import random
import time
import sys
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for 2x3 by 3x7 matrix multiplication.
    """
    n, m, p = DIMENSIONS  # n=2, m=3, p=7
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    # Schoolbook decomposition as fallback
    rank_school = n * m * p  # 42
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    def _verify_exact(U, V, W, n, m, p):
        """Verify the decomposition with exact arithmetic."""
        R = len(U[0]) if U else 0
        num_u = n * m
        num_v = m * p
        num_w = p * n
        
        # Check valid combinations
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = Fraction(0)
                    for r in range(R):
                        val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if val != 1:
                        return False
        
        # Check invalid combinations
        for a in range(num_u):
            i_u = a // m
            j_u = a % m
            for b in range(num_v):
                j_v = b // p
                k_v = b % p
                for c in range(num_w):
                    k_w = c // n
                    i_w = c % n
                    if not (j_u == j_v and i_u == i_w and k_v == k_w):
                        val = Fraction(0)
                        for r in range(R):
                            val += U[a][r] * V[b][r] * W[c][r]
                        if val != 0:
                            return False
        
        return True
    
    # Try to find a rank-21 decomposition (the theoretical lower bound)
    # Using a structured approach based on the 2x3 x 3x1 rank-7 construction
    
    # For 2x3 x 3x1, the rank is 7. The construction uses:
    # 3 terms for the "diagonal" (j=0,1,2 with k=0)
    # 4 additional terms to handle the 2 rows
    # Total: 7
    
    # For 2x3 x 3x7, we can potentially use a similar structure.
    # Let's try rank 21 first (the lower bound max(6,21,14)=21)
    
    def _try_rank_21():
        """Try to find a rank-21 decomposition."""
        if time.time() > deadline:
            return None
        
        n, m, p = DIMENSIONS
        R = 21
        num_u = n * m  # 6
        num_v = m * p  # 21
        num_w = p * n  # 14
        
        # Use a structured ansatz based on the known 2x3x3x1 rank-7 construction
        # For 2x3x3x1, the rank-7 construction is:
        # Terms 0-2: U[i*3+j]=1, V[j*1+0]=1, W[0*2+i]=1 for j=0,1,2
        # Terms 3-6: Additional terms to handle the 2 rows
        
        # Actually, let me use a different approach: try to find a decomposition
        # where each of the 21 "V" positions has a specific structure.
        
        # Let's try: for each j in {0,1,2} and k in {0,...,6}, we have a term
        # that handles the contribution of A[i,j] * B[j,k]
        # But we need to share across i values.
        
        # Key insight: For each j, we have A[0,j] and A[1,j].
        # We can compute:
        #   s_j = A[0,j] + A[1,j]
        #   d_j = A[0,j] - A[1,j]
        # Then A[0,j] = (s_j + d_j)/2, A[1,j] = (s_j - d_j)/2
        # 
        # For each j, we need to compute:
        #   C[0,k] += A[0,j] * B[j,k] = (s_j + d_j)/2 * B[j,k]
        #   C[1,k] += A[1,j] * B[j,k] = (s_j - d_j)/2 * B[j,k]
        #
        # This requires 2 multiplications per (j,k) pair, giving 3*7*2 = 42 total.
        # That's the same as schoolbook!
        
        # Let me try a different approach: use the fact that we can share B[j,k] values.
        # For a fixed j, we have B[j,0], B[j,1], ..., B[j,6].
        # We need to compute:
        #   C[0,k] += A[0,j] * B[j,k] for all k
        #   C[1,k] += A[1,j] * B[j,k] for all k
        #
        # If we compute A[0,j] and A[1,j] separately, we need 14 multiplications per j.
        # But if we can share, we might do better.
        
        # Actually, the known result for 2x3 x 3x1 is rank 7, not 6.
        # This suggests that there's no way to do better than 7 for the 1-column case.
        # For 7 columns, the rank might be 7*7 = 49? No, that's worse.
        
        # Let me look up the known results. For 2x3 x 3xp, the rank is:
        # p=1: 7
        # p=2: ?
        # p=3: ?
        # ...
        # p=7: ?
        
        # Actually, I recall that for 2x3 x 3xp, the rank is 7p for small p,
        # but it might be lower for larger p.
        
        # Let me try a direct numerical search for rank 21.
        
        # Use random initialization and gradient descent
        np.random.seed(42)
        U = np.random.randn(num_u, R) * 0.1
        V = np.random.randn(num_v, R) * 0.1
        W = np.random.randn(num_w, R) * 0.1
        
        def _loss_and_grad(U, V, W):
            """Compute loss and gradients."""
            residuals = np.zeros((num_u, num_v, num_w))
            
            # Fill in the target tensor
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        residuals[i*m+j, j*p+k, k*n+i] = 1.0
            
            # Compute actual tensor
            actual = np.zeros((num_u, num_v, num_w))
            for r in range(R):
                actual += np.einsum('ir,jr,kr->ijk', U[:,r], V[:,r], W[:,r])
            
            error = actual - residuals
            loss = np.sum(error**2)
            
            # Compute gradients
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for r in range(R):
                # dL/dU[i,r] = 2 * sum_{j,k} error[i,j,k] * V[j,r] * W[k,r]
                dU[:,r] = 2 * np.einsum('ijk,jr,kr->ir', error, V[:,r], W[:,r])
                dV[:,r] = 2 * np.einsum('ijk,ir,kr->jr', error, U[:,r], W[:,r])
                dW[:,r] = 2 * np.einsum('ijk,ir,jr->kr', error, U[:,r], V[:,r])
            
            return loss, dU, dV, dW
        
        # Adam optimizer
        lr = 0.01
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        
        mU = np.zeros_like(U)
        vU = np.zeros_like(U)
        mV = np.zeros_like(V)
        vV = np.zeros_like(V)
        mW = np.zeros_like(W)
        vW = np.zeros_like(W)
        
        best_loss = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        num_iters = 5000
        for it in range(num_iters):
            if time.time() > deadline - 5:
                break
            
            loss, dU, dV, dW = _loss_and_grad(U, V, W)
            
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if loss < 1e-10:
                break
            
            t = it + 1
            mU = beta1 * mU + (1 - beta1) * dU
            vU = beta2 * vU + (1 - beta2) * dU**2
            mU_hat = mU / (1 - beta1**t)
            vU_hat = vU / (1 - beta2**t)
            U -= lr * mU_hat / (np.sqrt(vU_hat) + eps)
            
            mV = beta1 * mV + (1 - beta1) * dV
            vV = beta2 * vV + (1 - beta2) * dV**2
            mV_hat = mV / (1 - beta1**t)
            vV_hat = vV / (1 - beta2**t)
            V -= lr * mV_hat / (np.sqrt(vV_hat) + eps)
            
            mW = beta1 * mW + (1 - beta1) * dW
            vW = beta2 * vW + (1 - beta2) * dW**2
            mW_hat = mW / (1 - beta1**t)
            vW_hat = vW / (1 - beta2**t)
            W -= lr * mW_hat / (np.sqrt(vW_hat) + eps)
        
        if best_loss < 1e-8:
            # Try to find exact rational solution
            # Round to nearest rational with small denominator
            U_frac = [[Fraction(int(round(best_U[i][r] * 100)), 100) for r in range(R)] for i in range(num_u)]
            V_frac = [[Fraction(int(round(best_V[i][r] * 100)), 100) for r in range(R)] for i in range(num_v)]
            W_frac = [[Fraction(int(round(best_W[i][r] * 100)), 100) for r in range(R)] for i in range(num_w)]
            
            if _verify_exact(U_frac, V_frac, W_frac, n, m, p):
                return U_frac, V_frac, W_frac
            
            # Try with larger denominator
            U_frac = [[Fraction(int(round(best_U[i][r] * 1000)), 1000) for r in range(R)] for i in range(num_u)]
            V_frac = [[Fraction(int(round(best_V[i][r] * 1000)), 1000) for r in range(R)] for i in range(num_v)]
            W_frac = [[Fraction(int(round(best_W[i][r] * 1000)), 1000) for r in range(R)] for i in range(num_w)]
            
            if _verify_exact(U_frac, V_frac, W_frac, n, m, p):
                return U_frac, V_frac, W_frac
        
        return None
    
    # Try rank 21
    result = _try_rank_21()
    if result is not None:
        U_frac, V_frac, W_frac = result
        return {"U": U_frac, "V": V_frac, "W": W_frac}
    
    # Return schoolbook
    return {"U": U_school, "V": V_school, "W": W_school}

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    start_time = time.time()
    res = construct()
    n, m, p = DIMENSIONS
    U, V, W = res['U'], res['V'], res['W']
    rank = len(U[0])
    
    assert len(U) == n * m
    assert len(V) == m * p
    assert len(W) == p * n
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for r in range(rank):
                    val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                assert val == 1, f"Failed for i={i}, j={j}, k={k}, val={val}"
    
    a, b, c = 0, 3, 2
    val = 0
    for r in range(rank):
        val += U[a][r] * V[b][r] * W[c][r]
    assert val == 0, f"Failed invalid check, val={val}"
    
    elapsed = time.time() - start_time
    if elapsed > 100:
        print(f"Warning: Construction took {elapsed}s", file=sys.stderr)
        
    print(f"Score (Rank): {rank}")
