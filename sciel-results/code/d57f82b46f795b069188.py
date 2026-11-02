"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n  # 4, 10, 10
    
    # Build target tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    def check_decomposition(U, V, W, tol=1e-9):
        R = np.einsum('ad,bd,cd->abc', U, V, W)
        return np.max(np.abs(R - T)) < tol
    
    def build_rank18():
        """Strassen for cols 0-1, Strassen for cols 2-3, schoolbook for col 4."""
        r = 18
        U = np.zeros((nm, r))
        V = np.zeros((mp, r))
        W = np.zeros((pn, r))
        
        # Strassen factors (standard 2x2x2x2)
        # U rows: A[0,0], A[0,1], A[1,0], A[1,1] -> indices 0,1,2,3
        S_U = np.array([
            [1, 0, 1, 0, 1, -1, 0],
            [0, 0, 0, 0, 1, 0, 1],
            [0, 1, 0, 0, 0, 1, 0],
            [1, 1, 0, 1, 0, 0, -1]
        ], dtype=float)
        
        # V rows: B[0,0], B[0,1], B[1,0], B[1,1]
        S_V = np.array([
            [1, 1, 0, -1, 0, 1, 0],
            [0, 0, 1, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 1],
            [1, 0, -1, 0, 1, 0, 1]
        ], dtype=float)
        
        # W rows: C[0,0], C[1,0], C[0,1], C[1,1]
        S_W = np.array([
            [1, 0, 0, 0, -1, 0, 1],
            [0, 1, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 1, 0, 0],
            [1, -1, 1, 0, 0, 0, 1]
        ], dtype=float)
        
        # Block 1: k=0,1
        # B indices: B[0,0]=0, B[0,1]=1, B[1,0]=5, B[1,1]=6
        # C indices: C[0,0]=0, C[1,0]=1, C[0,1]=2, C[1,1]=3
        U[:, 0:7] = S_U
        V[[0,1,5,6], 0:7] = S_V
        W[[0,1,2,3], 0:7] = S_W
        
        # Block 2: k=2,3
        # B indices: B[0,2]=2, B[0,3]=3, B[1,2]=7, B[1,3]=8
        # C indices: C[0,2]=4, C[1,2]=5, C[0,3]=6, C[1,3]=7
        U[:, 7:14] = S_U
        V[[2,3,7,8], 7:14] = S_V
        W[[4,5,6,7], 7:14] = S_W
        
        # Block 3: k=4 (schoolbook, rank 4)
        # A[0,0]*B[0,4] -> U[0,14]=1, V[4,14]=1, W[8,14]=1
        # A[0,1]*B[1,4] -> U[1,15]=1, V[9,15]=1, W[8,15]=1
        # A[1,0]*B[0,4] -> U[2,16]=1, V[4,16]=1, W[9,16]=1
        # A[1,1]*B[1,4] -> U[3,17]=1, V[9,17]=1, W[9,17]=1
        U[0, 14] = 1.0; V[4, 14] = 1.0; W[8, 14] = 1.0
        U[1, 15] = 1.0; V[9, 15] = 1.0; W[8, 15] = 1.0
        U[2, 16] = 1.0; V[4, 16] = 1.0; W[9, 16] = 1.0
        U[3, 17] = 1.0; V[9, 17] = 1.0; W[9, 17] = 1.0
        
        return U, V, W
    
    def als_solve(U, V, W, r, max_iters=100, tol=1e-12):
        """ALS to refine a rank-r decomposition."""
        for it in range(max_iters):
            # Solve for U
            TW = np.einsum('abc,cd->abd', T, W)
            U_num = np.einsum('abd,bd->ad', TW, V)
            U_den = np.einsum('bd,cd->d', V, W) ** 2
            U_den = np.where(U_den < 1e-15, 1.0, U_den)
            U_new = U_num / U_den[np.newaxis, :]
            
            # Solve for V
            TU = np.einsum('abc,ad->bcd', T, U)
            V_num = np.einsum('bcd,cd->bd', TU, W)
            V_den = np.einsum('ad,cd->d', U, W) ** 2
            V_den = np.where(V_den < 1e-15, 1.0, V_den)
            V_new = V_num / V_den[np.newaxis, :]
            
            # Solve for W
            TV = np.einsum('abc,bd->acd', T, V)
            W_num = np.einsum('acd,ad->cd', TV, U)
            W_den = np.einsum('ad,bd->d', U, V) ** 2
            W_den = np.where(W_den < 1e-15, 1.0, W_den)
            W_new = W_num / W_den[np.newaxis, :]
            
            err = np.max(np.abs(U_new - U)) + np.max(np.abs(V_new - V)) + np.max(np.abs(W_new - W))
            U, V, W = U_new, V_new, W_new
            if err < tol:
                break
        
        return U, V, W
    
    def try_rank_als(U0, V0, W0, r, max_iters=200, tol=1e-12):
        """Try to achieve exact decomposition at rank r using ALS."""
        U, V, W = U0.copy(), V0.copy(), W0.copy()
        U, V, W = als_solve(U, V, W, r, max_iters=max_iters, tol=tol)
        if check_decomposition(U, V, W, tol=1e-9):
            return U, V, W, True
        return U, V, W, False
    
    deadline = time.time() + 140
    
    # Start with rank-18 construction
    U18, V18, W18 = build_rank18()
    if not check_decomposition(U18, V18, W18, tol=1e-9):
        # Fallback to schoolbook rank 20
        U = np.zeros((nm, 20))
        V = np.zeros((mp, 20))
        W = np.zeros((pn, 20))
        for c in range(20):
            i = c // (m*p)
            rem = c % (m*p)
            j = rem // p
            k = rem % p
            U[i*m+j, c] = 1.0
            V[j*p+k, c] = 1.0
            W[k*n+i, c] = 1.0
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_U, best_V, best_W = U18, V18, W18
    best_rank = 18
    
    # Try to reduce to rank 17
    if time.time() < deadline:
        for attempt in range(30):
            if time.time() > deadline:
                break
            if attempt == 0:
                U17, V17, W17 = U18[:, :17].copy(), V18[:, :17].copy(), W18[:, :17].copy()
            else:
                np.random.seed(attempt * 104729 + 17)
                U17 = U18[:, :17] + np.random.randn(nm, 17) * 0.01
                V17 = V18[:, :17] + np.random.randn(mp, 17) * 0.01
                W17 = W18[:, :17] + np.random.randn(pn, 17) * 0.01
            
            U17, V17, W17, success = try_rank_als(U17, V17, W17, 17, max_iters=200)
            if success:
                best_U, best_V, best_W = U17, V17, W17
                best_rank = 17
                break
    
    # Try to reduce to rank 16
    if best_rank > 16 and time.time() < deadline:
        base_U, base_V, base_W = best_U, best_V, best_W
        for attempt in range(40):
            if time.time() > deadline:
                break
            if attempt == 0:
                U16, V16, W16 = base_U[:, :16].copy(), base_V[:, :16].copy(), base_W[:, :16].copy()
            elif attempt == 1:
                np.random.seed(16)
                U16 = base_U[:, :16] + np.random.randn(nm, 16) * 0.05
                V16 = base_V[:, :16] + np.random.randn(mp, 16) * 0.05
                W16 = base_W[:, :16] + np.random.randn(pn, 16) * 0.05
            else:
                np.random.seed(attempt * 104729 + 16)
                U16 = base_U[:, :16] + np.random.randn(nm, 16) * 0.1
                V16 = base_V[:, :16] + np.random.randn(mp, 16) * 0.1
                W16 = base_W[:, :16] + np.random.randn(pn, 16) * 0.1
            
            U16, V16, W16, success = try_rank_als(U16, V16, W16, 16, max_iters=300)
            if success:
                best_U, best_V, best_W = U16, V16, W16
                best_rank = 16
                break
    
    # Try rank 15
    if best_rank > 15 and time.time() < deadline:
        base_U, base_V, base_W = best_U, best_V, best_W
        for attempt in range(20):
            if time.time() > deadline:
                break
            np.random.seed(attempt * 104729 + 15)
            U15 = base_U[:, :15] + np.random.randn(nm, 15) * 0.1
            V15 = base_V[:, :15] + np.random.randn(mp, 15) * 0.1
            W15 = base_W[:, :15] + np.random.randn(pn, 15) * 0.1
            
            U15, V15, W15, success = try_rank_als(U15, V15, W15, 15, max_iters=300)
            if success:
                best_U, best_V, best_W = U15, V15, W15
                best_rank = 15
                break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
