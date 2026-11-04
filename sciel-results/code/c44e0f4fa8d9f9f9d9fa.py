"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import itertools

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s
    
    n, m, p = DIMENSIONS  # 2, 3, 6
    
    # Known rank-33 construction: 3x Strassen(2x2x2) + schoolbook(2x1x6)
    # Split middle dim 3 into 2+1, split output 6 into 3 pairs of 2
    # For each pair of output columns, do 2x2x2 Strassen on the first 2 middle indices
    # Plus schoolbook for the 3rd middle index across all 6 columns
    
    def strassen_2x2x2():
        """Return U, V, W for 2x2x2 Strassen, each shape (4,7) or similar."""
        # A is 2x2: A[i,j], B is 2x2: B[j,k], C is 2x2: C[i,k]
        # Strassen's 7 multiplications
        # U[i,j,c], V[j,k,c], W[k,i,c]
        # c=0..6
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A00+A11)(B00+B11)
        U[0,0] = 1; U[3,0] = 1
        V[0,0] = 1; V[3,0] = 1
        W[0,0] = 1; W[3,0] = 1  # C00 += M1, C11 += M1
        # Actually W[k,i,c]: W[0,0] contributes to C[0,0], W[3,0] contributes to C[1,1]
        # But M1 contributes to both C00 and C11
        # Let me redo this more carefully
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # M1 = (A00+A11)(B00+B11)
        U[0,0] = 1; U[3,0] = 1  # A00, A11
        V[0,0] = 1; V[3,0] = 1  # B00, B11
        W[0,0] = 1; W[3,0] = 1  # C00 += M1, C11 += M1
        
        # M2 = (A10+A11)B00
        U[2,1] = 1; U[3,1] = 1  # A10, A11
        V[0,1] = 1  # B00
        W[2,1] = 1  # C10 += M2
        
        # M3 = A00(B10-B11)
        U[0,2] = 1  # A00
        V[2,2] = 1; V[3,2] = -1  # B10, B11
        W[1,2] = 1; W[2,2] = 1  # C01 += M3, C10 += M3
        
        # M4 = A11(B01-B10)
        U[3,3] = 1  # A11
        V[1,3] = 1; V[2,3] = -1  # B01, B10
        W[0,3] = 1; W[2,3] = 1  # C00 += M4, C10 += M4
        
        # M5 = (A00+A01)B11
        U[0,4] = 1; U[1,4] = 1  # A00, A01
        V[3,4] = 1  # B11
        W[1,4] = 1; W[0,4] = -1  # C01 += M5, C00 -= M5
        
        # M6 = (A10-A00)(B00+B10)
        U[2,5] = 1; U[0,5] = -1  # A10, A00
        V[0,5] = 1; V[2,5] = 1  # B00, B10
        W[3,5] = 1  # C11 += M6
        
        # M7 = (A01-A11)(B01+B11)
        U[1,6] = 1; U[3,6] = -1  # A01, A11
        V[1,6] = 1; V[3,6] = 1  # B01, B11
        W[0,6] = 1  # C00 += M7
        
        return U, V, W
    
    def build_rank33():
        """Build the rank-33 decomposition."""
        rank = 33
        # U: 6x33 (2x3 matrix flattened), V: 18x33 (3x6 matrix flattened), W: 12x33 (6x2 matrix flattened)
        U = np.zeros((6, rank))
        V = np.zeros((18, rank))
        W = np.zeros((12, rank))
        
        col = 0
        # For each of 3 pairs of output columns (k_pair = 0,1,2):
        #   Use Strassen on A[:,0:2] (2x2) and B[0:2, 2*k_pair:2*k_pair+2] (2x2)
        #   This contributes to C[:, 2*k_pair:2*k_pair+2]
        for k_pair in range(3):
            sU, sV, sW = strassen_2x2x2()
            # sU is 4x7: indices are (i,j) where i in {0,1}, j in {0,1}
            # Map to U[i*m+j, col:c] where m=3, so U[i*3+j, col:c]
            for i in range(2):
                for j in range(2):
                    U[i*3 + j, col:col+7] = sU[i*2 + j, :]
            
            # sV is 4x7: indices are (j,k) where j in {0,1}, k in {0,1}
            # Map to V[j*p+k, col:c] where p=6, so V[j*6 + (2*k_pair + k), col:c]
            for j in range(2):
                for k in range(2):
                    V[j*6 + (2*k_pair + k), col:col+7] = sV[j*2 + k, :]
            
            # sW is 4x7: indices are (k,i) where k in {0,1}, i in {0,1}
            # Map to W[k*n+i, col:c] where n=2, so W[(2*k_pair+k)*2 + i, col:c]
            for k in range(2):
                for i in range(2):
                    W[(2*k_pair + k)*2 + i, col:col+7] = sW[k*2 + i, :]
            
            col += 7
        
        # Now schoolbook for the 3rd middle index (j=2) across all 6 columns
        # A[:,2] (2x1) * B[2,:] (1x6) -> C[:,:] (2x6)
        # For each i in {0,1}, k in {0..5}:
        for i in range(2):
            for k in range(6):
                U[i*3 + 2, col] = 1  # A[i,2]
                V[2*6 + k, col] = 1  # B[2,k]
                W[k*2 + i, col] = 1  # C[i,k]
                col += 1
        
        assert col == rank, f"Expected {rank} columns, got {col}"
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    def verify(U, V, W, n, m, p, tol=1e-9):
        """Verify that the decomposition is exact."""
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c]
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def to_half_integers(U, V, W):
        """Round to nearest half-integer and return as lists."""
        def round_half(x):
            return round(x * 2) / 2
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        return U_r.tolist(), V_r.tolist(), W_r.tolist()
    
    # First, try the known rank-33
    result33 = build_rank33()
    if verify(result33["U"], result33["V"], result33["W"], n, m, p):
        best = (33, result33)
    else:
        best = None
    
    # Try to find better decompositions
    # Strategy: Try rank r for r in [18, 19, ..., 32]
    # For each rank, try to find a decomposition using numerical methods
    
    def try_rank(r, max_iters=5000, seed=42):
        """Try to find a rank-r decomposition using gradient descent."""
        if time.time() > deadline - 5:
            return None
        
        np.random.seed(seed)
        # Initialize U, V, W randomly
        U = np.random.randn(n*m, r) * 0.1
        V = np.random.randn(m*p, r) * 0.1
        W = np.random.randn(p*n, r) * 0.1
        
        # Objective: minimize sum_{i,j,k} (sum_c U[i,j,c] V[j,k,c] W[k,i,c] - 1)^2
        # This is non-convex, use simple gradient descent
        
        lr = 0.01
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Compute current values
            vals = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        vals[i,j,k] = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
            
            loss = np.sum((vals - 1.0)**2)
            if loss < 1e-10:
                break
            
            # Compute gradients
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        err = 2 * (vals[i,j,k] - 1.0)
                        dU[i*m+j, :] += err * V[j*p+k, :] * W[k*n+i, :]
                        dV[j*p+k, :] += err * U[i*m+j, :] * W[k*n+i, :]
                        dW[k*n+i, :] += err * U[i*m+j, :] * V[j*p+k, :]
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
        
        # Check if we found a valid decomposition
        if time.time() > deadline - 5:
            return None
        
        # Round to half-integers
        U_r, V_r, W_r = to_half_integers(U, V, W)
        
        if verify(U_r, V_r, W_r, n, m, p, tol=1e-9):
            return {"U": U_r, "V": V_r, "W": W_r}
        
        # Also check without rounding
        if verify(U.tolist(), V.tolist(), W.tolist(), n, m, p, tol=1e-9):
            # Check if entries are close to half-integers
            max_dev = max(np.max(np.abs(U - np.round(U*2)/2)),
                         np.max(np.abs(V - np.round(V*2)/2)),
                         np.max(np.abs(W - np.round(W*2)/2)))
            if max_dev < 1e-6:
                return {"U": U_r, "V": V_r, "W": W_r}
        
        return None
    
    # Try ranks from 24 to 32 (skip some for speed)
    for r in [24, 25, 26, 27, 28, 29, 30, 31, 32]:
        if time.time() > deadline - 10:
            break
        
        # Try a few seeds
        found = None
        for seed in [42, 123, 456]:
            if time.time() > deadline - 10:
                break
            result = try_rank(r, max_iters=2000, seed=seed)
            if result is not None:
                found = result
                break
        
        if found is not None:
            if best is None or r < best[0]:
                best = (r, found)
    
    # Return the best result
    if best is not None:
        return best[1]
    
    # Fallback: return schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
