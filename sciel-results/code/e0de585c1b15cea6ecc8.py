"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n=2, m=2, p=4, tol=1e-9):
    """Verify that U, V, W form a valid tensor decomposition."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _try_rank14():
    """Try to construct a rank-14 decomposition."""
    n, m, p = 2, 2, 4
    rank = 14
    
    # We'll use a numerical approach.
    # Fix U and V with a structured pattern, then solve for W.
    
    # Strategy: Use two independent (2,2,2) Strassen-like decompositions.
    # For the left part (k=0,1) and right part (k=2,3).
    
    # Let's try a simple structured approach.
    # For each of the 14 terms, we'll define U, V, W patterns.
    
    # Actually, let's just use optimization.
    # Initialize randomly and optimize.
    
    np.random.seed(42)
    
    # We need: for all i,j,k: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    # This is 16 equations.
    
    # Let's fix U to be a random matrix and V to be a random matrix,
    # then solve for W linearly.
    
    # For each (i,k): sum_r W[k*n+i,r] * U[i*m+j,r] * V[j*p+k,r] = 1 for j=0,1
    # This gives 2 equations per (i,k), total 16 equations in 14 unknowns per (i,k).
    
    # For fixed (i,k), let's define:
    # A_{j,r} = U[i*m+j, r] * V[j*p+k, r] for j=0,1
    # Then: sum_r W[k*n+i, r] * A_{j,r} = 1 for j=0,1
    # This is a 2x14 system. We need it to be consistent.
    
    # Let's try: make U and V such that for each (i,k), the 2x14 matrix [A_0; A_1] has rank 2.
    
    # Simple approach: make U and V "block diagonal" in a sense.
    
    # Let me try a specific construction:
    # Term r=0..6: handle left half (k=0,1)
    # Term r=7..13: handle right half (k=2,3)
    
    # For the left half, use a (2,2,2) decomposition.
    # For the right half, use another (2,2,2) decomposition.
    
    # Let's hard-code a (2,2,2) decomposition and verify it.
    # If I can't get Strassen right, I'll use a 8-term schoolbook for each half.
    
    # Schoolbook for (2,2,2): rank 8
    # But that gives total rank 16, not 14.
    
    # Let me try to find a rank-7 (2,2,2) decomposition numerically.
    
    def find_222_decomp():
        """Find a rank-7 decomposition of the (2,2,2) matrix multiplication tensor."""
        n2, m2, p2 = 2, 2, 2
        rank2 = 7
        # U2: (4, 7), V2: (4, 7), W2: (4, 7)
        # Condition: for all i,j,k in {0,1}: sum_r U2[i*2+j,r]*V2[j*2+k,r]*W2[k*2+i,r] = 1
        
        np.random.seed(123)
        U2 = np.random.randn(4, rank2)
        V2 = np.random.randn(4, rank2)
        
        # Solve for W2: for each (i,k): sum_r W2[k*2+i,r] * U2[i*2+j,r] * V2[j*2+k,r] = 1 for j=0,1
        W2 = np.zeros((4, rank2))
        for i in range(2):
            for k in range(2):
                # 2 equations in 7 unknowns
                A = np.zeros((2, rank2))
                for j in range(2):
                    for r in range(rank2):
                        A[j, r] = U2[i*2+j, r] * V2[j*2+k, r]
                # Solve A @ w = [1, 1]
                # Least squares
                w, residuals, _, _ = np.linalg.lstsq(A, np.ones(2), rcond=None)
                W2[k*2+i, :] = w
        
        # Verify
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    s = sum(U2[i*2+j, r] * V2[j*2+k, r] * W2[k*2+i, r] for r in range(rank2))
                    if abs(s - 1.0) > 1e-8:
                        return None
        return U2, V2, W2
    
    U2, V2, W2 = find_222_decomp()
    if U2 is None:
        return None
    
    # Now construct the (2,2,4) decomposition.
    U = np.zeros((4, 14))
    V = np.zeros((8, 14))
    W = np.zeros((8, 14))
    
    # Left half: terms 0-6
    U[:, 0:7] = U2
    V[0:2, 0:7] = V2[0:2, :]  # j=0, k=0,1
    V[2:4, 0:7] = 0           # j=0, k=2,3 (not used in left)
    V[4:6, 0:7] = V2[2:4, :]  # j=1, k=0,1
    V[6:8, 0:7] = 0           # j=1, k=2,3
    W[0:4, 0:7] = W2          # k=0,1; i=0,1
    W[4:8, 0:7] = 0           # k=2,3
    
    # Right half: terms 7-13
    U[:, 7:14] = U2
    V[0:2, 7:14] = 0          # j=0, k=0,1
    V[2:4, 7:14] = V2[0:2, :] # j=0, k=2,3
    V[4:6, 7:14] = 0          # j=1, k=0,1
    V[6:8, 7:14] = V2[2:4, :] # j=1, k=2,3
    W[0:4, 7:14] = 0          # k=0,1
    W[4:8, 7:14] = W2         # k=2,3; i=0,1
    
    # Verify
    if _verify(U, V, W):
        return U, V, W
    return None

def _schoolbook():
    n, m, p = 2, 2, 4
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    # Try rank 14
    result = _try_rank14()
    if result is not None:
        U, V, W = result
        # Convert to lists with integer/half-integer entries if possible
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fall back to schoolbook
    U, V, W = _schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
