"""Matrix multiplication 2,4,5 — rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(T_recon, T, atol=1e-9)

def _build_strassen_2x2x2():
    """
    Build the rank-3 Strassen decomposition for 2x2 * 2x2.
    Returns U (4x3), V (4x3), W (4x3) for the 2x2x2 tensor.
    Convention: U[i*2+j], V[j*2+k], W[k*2+i]
    """
    # Strassen formulas:
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b10-b00)
    # M4 = a11*(b01-b11)
    # M5 = (a00+a01)*b11
    # M6 = (a10-a00)*(b00+b10)
    # M7 = (a01-a11)*(b01+b11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # U indices: i*2+j for i,j in {0,1} -> 0,1,2,3
    # V indices: j*2+k for j,k in {0,1} -> 0,1,2,3
    # W indices: k*2+i for k,i in {0,1} -> 0,1,2,3
    #
    # M1: U[0]=1,a00; U[3]=1,a11 -> U[0]=1,U[3]=1
    #     V[0]=1,b00; V[3]=1,b11 -> V[0]=1,V[3]=1
    #     W: C00=M1, C11=M1 -> W[0]=1 (k=0,i=0), W[3]=1 (k=1,i=1)
    # M2: U[2]=1,a10; U[3]=1,a11
    #     V[0]=1,b00
    #     W: C10=M2 -> W[2]=1 (k=1,i=0)... wait W[k*2+i]
    #     C10: k=1,i=0 -> W[1*2+0]=W[2]
    # M3: U[0]=1,a00
    #     V[2]=1,b10; V[0]=-1,b00
    #     W: C01=M3 -> W[1*2+0]=W[2]... no C01: i=0,k=1 -> W[1*2+0]=W[2]
    #     Wait, W index is k*n+i where n=2, so W[k*2+i]
    #     C01: i=0, k=1 -> W[1*2+0] = W[2]
    # M4: U[3]=1,a11
    #     V[1]=1,b01; V[3]=-1,b11
    #     W: C00=M4 -> W[0]=1, C10=M4 -> W[2]=1
    # M5: U[0]=1,a00; U[1]=1,a01
    #     V[3]=1,b11
    #     W: C00=-M5 -> W[0]=-1, C01=M5 -> W[2]=1
    # M6: U[2]=1,a10; U[0]=-1,a00
    #     V[0]=1,b00; V[2]=1,b10
    #     W: C11=M6 -> W[3]=1
    # M7: U[1]=1,a01; U[3]=-1,a11
    #     V[1]=1,b01; V[3]=1,b11
    #     W: C00=M7 -> W[0]=1

    U = np.zeros((4, 3))  # Only 3 multiplications for Strassen 2x2x2
    V = np.zeros((4, 3))
    W = np.zeros((4, 3))

    # M1 = (a00+a11)*(b00+b11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1  # C00 and C11

    # M2 = (a10+a11)*b00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1  # C10: k=1,i=0 -> W[2]

    # M3 = a00*(b10-b00)
    U[0, 2] = 1
    V[2, 2] = 1; V[0, 2] = -1
    W[2, 2] = 1  # C01: i=0,k=1 -> W[2]

    # Wait, I only have 3 columns but Strassen has 7 multiplications.
    # Strassen for 2x2x2 uses 7 multiplications, not 3.
    # The rank of M_{2,2,2} is 7 (Strassen), not 3.
    # I was confused. Let me redo this.
    
    # Actually, rank(M_{2,2,2}) = 7 by Strassen.
    # Let me build the full rank-7 Strassen decomposition.
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))

    # M1 = (a00+a11)*(b00+b11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1  # C00
    W[3, 0] = 1  # C11

    # M2 = (a10+a11)*b00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1  # C10: k=1,i=0 -> W[1*2+0]=W[2]

    # M3 = a00*(b10-b00)
    U[0, 2] = 1
    V[2, 2] = 1; V[0, 2] = -1
    W[2, 2] = 1  # C01: i=0,k=1 -> W[1*2+0]=W[2]

    # M4 = a11*(b01-b11)
    U[3, 3] = 1
    V[1, 3] = 1; V[3, 3] = -1
    W[0, 3] = 1  # C00
    W[2, 3] = 1  # C10: k=1,i=0 -> W[2]

    # M5 = (a00+a01)*b11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1  # C00
    W[2, 4] = 1   # C01: i=0,k=1 -> W[2]

    # M6 = (a10-a00)*(b00+b10)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[2, 5] = 1
    W[3, 5] = 1   # C11

    # M7 = (a01-a11)*(b01+b11)
    U[1, 6] = 1; U[3, 6] = -1
    V[1, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1   # C00

    return U, V, W

def _build_schoolbook(n, m, p):
    """Build the schoolbook decomposition of rank n*m*p."""
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
    return U, V, W

def construct():
    """
    Rank-minimal exact tensor decomposition for multiplying a 2x4 matrix by a 4x5 matrix.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    n, m, p = 2, 4, 5
    
    # Build target tensor
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Strategy: Try to find a low-rank decomposition.
    # Known lower bound: max(n*m, m*p, n*p) = max(8, 20, 10) = 20
    # Schoolbook: 40
    
    # Try structured decompositions at various ranks
    # Approach: split m=4 into two blocks of 2, use Strassen-like for each
    
    # For each block b (0,1), we have a (2x2) x (2x5) subproblem.
    # C_sub[i,k] = A_sub[i,0]*B_sub[0,k] + A_sub[i,1]*B_sub[1,k]
    
    # We can view this as: for each k, C_sub[:,k] = A_sub @ B_sub[:,k]
    # This is a 2x2 matrix times a 2x1 vector.
    
    # Alternatively, we can use the "bilinear" structure:
    # Let's try to find a decomposition using random initialization + ALS
    
    best_result = None
    best_rank = n * m * p  # Start with schoolbook rank
    
    # First, get the schoolbook as a baseline
    U_school, V_school, W_school = _build_schoolbook(n, m, p)
    if _verify_decomposition(U_school, V_school, W_school, n, m, p):
        best_result = (U_school, V_school, W_school)
        best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # Use randomized ALS with multiple restarts
    np.random.seed(42)
    
    # Try ranks from lower bound upward
    for rank_try in range(20, best_rank + 1):
        if time.time() > deadline - 20:
            break
        
        # Try multiple random initializations
        max_restarts = 5 if rank_try <= 25 else 3
        found = False
        
        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                break
            
            U = np.random.randn(n * m, rank_try)
            V = np.random.randn(m * p, rank_try)
            W = np.random.randn(p * n, rank_try)
            
            # Normalize
            for r in range(rank_try):
                nu = np.linalg.norm(U[:, r])
                nv = np.linalg.norm(V[:, r])
                nw = np.linalg.norm(W[:, r])
                if nu > 1e-12 and nv > 1e-12 and nw > 1e-12:
                    scale = (nu * nv * nw) ** (1.0 / 3.0)
                    U[:, r] /= nu
                    V[:, r] /= nv
                    W[:, r] /= nw
            
            best_err = float('inf')
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            for iteration in range(500):
                if time.time() > deadline - 10:
                    break
                
                # Update U
                for a in range(n * m):
                    M_mat = np.kron(W.T, V)
                    b_vec = T[a, :, :].flatten()
                    try:
                        sol, _, _, _ = np.linalg.lstsq(M_mat, b_vec, rcond=None)
                        U[a, :] = sol
                    except:
                        pass
                
                # Update V
                for b in range(m * p):
                    M_mat = np.kron(W.T, U)
                    b_vec = T[:, b, :].flatten()
                    try:
                        sol, _, _, _ = np.linalg.lstsq(M_mat, b_vec, rcond=None)
                        V[b, :] = sol
                    except:
                        pass
                
                # Update W
                for c in range(p * n):
                    M_mat = np.kron(V.T, U)
                    b_vec = T[:, :, c].flatten()
                    try:
                        sol, _, _, _ = np.linalg.lstsq(M_mat, b_vec, rcond=None)
                        W[c, :] = sol
                    except:
                        pass
                
                # Compute error
                T_recon = np.einsum('ar,br,cr->abc', U, V, W)
                err = np.linalg.norm(T_recon - T)
                if err < best_err:
                    best_err = err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if best_err < 1e-10:
                    break
            
            if best_err < 1e-8:
                # Verify
                if _verify_decomposition(best_U, best_V, best_W, n, m, p):
                    best_result = (best_U, best_V, best_W)
                    best_rank = rank_try
                    found = True
                    break
        
        if found:
            break
    
    # If we found a valid decomposition, return it
    if best_result is not None:
        U, V, W = best_result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
# EVOLVE-BLOCK-END
