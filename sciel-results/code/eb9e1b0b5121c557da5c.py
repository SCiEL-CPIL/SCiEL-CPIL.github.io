"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """Return U,V,W factor matrices (each 4x4) for the rank-7 Strassen
    decomposition of 2x2x2 matrix multiplication.
    Indexing: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r] with i,j,k in {0,1}.
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    # M1 = (A00+A11)*(B00+B11)
    U[0][0] = 1; U[3][0] = 1; V[0][0] = 1; V[3][0] = 1; W[0][0] = 1
    # M2 = (A10+A11)*B00
    U[2][1] = 1; U[3][1] = 1; V[0][1] = 1; W[2][1] = 1
    # M3 = A00*(B01-B11)
    U[0][2] = 1; V[1][2] = 1; V[3][2] = -1; W[1][2] = 1
    # M4 = A11*(B10-B00)
    U[3][3] = 1; V[2][3] = 1; V[0][3] = -1; W[3][3] = 1
    # M5 = (A00+A01)*B11
    U[0][4] = 1; U[1][4] = 1; V[3][4] = 1; W[1][4] = 1
    # M6 = (A10-A00)*(B00+B01)
    U[2][5] = 1; U[0][5] = -1; V[0][5] = 1; V[1][5] = 1; W[2][5] = 1
    # M7 = (A01-A11)*(B10+B11)
    U[1][6] = 1; U[3][6] = -1; V[2][6] = 1; V[3][6] = 1; W[3][6] = 1
    return np.array(U, dtype=np.float64), np.array(V, dtype=np.float64), np.array(W, dtype=np.float64)


def _build_structured_init(n, m, p):
    """Build a structured decomposition using Strassen on 2x2 blocks.
    Returns U (n*m x R), V (m*p x R), W (p*n x R) as float arrays.
    """
    # Split into 2x2 blocks
    # A is n x m = 4 x 4: 2x2 blocks, block (bi,bj) is 2x2
    # B is m x p = 4 x 7: split m into 2x2 blocks (bi), p into blocks of size 2,2,2,1
    # We'll handle p=7 as 4 blocks: sizes [2,2,2,1]
    
    # For simplicity, pad p to 8 (4 blocks of size 2), then we get rank 16*7=112
    # but with Strassen on the 2x2x2 blocks we can reduce.
    # Actually let's use a cleaner approach:
    
    # Block decomposition:
    # A = [[A00, A01], [A10, A11]] where each Ai is 2x2
    # B = [[B00, B01, B02, B03], [B10, B11, B12, B13]] where each Bi is 2x2 (for first 4 cols)
    #     plus B04, B14 which are 2x1
    
    # For the 4x4x4 part (first 4 columns of B):
    # C = A*B where both are 4x4
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    
    # Each C block is 2x2, computed as sum of two 2x2x2 products.
    # Using Strassen for each 2x2x2: rank 7 per product
    # Total for 4x4x4: 4 blocks * 2 products * 7 = 56
    # But we can share: use Strassen on the full 4x4x4? No, that's not standard.
    
    # Simpler: just use Strassen for each 2x2x2 sub-product
    # For 4x4x4: 8 sub-products (2 per C block, 4 C blocks) * 7 = 56
    # For 4x4x3 (last 3 cols): 
    #   B has 3 columns. Split into 2+1.
    #   C00 (2x2) = A00*B00 + A01*B10 where B00,B10 are 2x2
    #   C01 (2x1) = A00*B01 + A01*B11 where B01,B11 are 2x1
    #   C10 (2x2) = A10*B00 + A11*B10
    #   C11 (2x1) = A10*B01 + A11*B11
    #   For 2x2x2: rank 7 each (4 of them) = 28
    #   For 2x2x1: rank 4 each (schoolbook, 4 of them) = 16
    #   Total for 4x4x3: 44
    # Grand total: 56 + 44 = 100
    
    # This is better than 112 but not great. Let me just use schoolbook for now
    # and rely on ALS to find improvements.
    
    # Actually, let's use a simpler structured init: Strassen on 2x2x2 blocks
    # For the full 4x4x7, treat as 2x2 blocks in A and B:
    # A: 2x2 blocks of 2x2 (4 blocks)
    # B: 2x2 blocks of 2x2 for first 4 cols, then 2x1 for last 3 cols
    
    # Let me just build schoolbook and let ALS optimize
    R = n * m * p  # 112
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1.0
                V[j*p + k, col] = 1.0
                W[k*n + i, col] = 1.0
                col += 1
    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-10):
    """Check if the decomposition exactly represents the matrix multiplication tensor."""
    nm, mp, pn = n*m, m*p, p*n
    # Build target tensor
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    
    # Compute decomposition
    R = U.shape[1]
    T_dec = np.zeros((nm, mp, pn), dtype=np.float64)
    for r in range(R):
        T_dec += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    
    err = np.max(np.abs(T - T_dec))
    return err < tol, err


def _als_decompose(T, nm, mp, pn, target_rank, max_iter=50, tol=1e-8, seed=None):
    """Alternating least squares to find a rank-`target_rank` decomposition of T."""
    if seed is not None:
        np.random.seed(seed)
    
    U = np.random.randn(nm, target_rank) * 0.5
    V = np.random.randn(mp, target_rank) * 0.5
    W = np.random.randn(pn, target_rank) * 0.5
    
    T_flat = T.reshape(nm, mp*pn)
    
    for it in range(max_iter):
        # Update U: T_flat = U @ M_U where M_U[r, bc] = V[b,r]*W[c,r]
        M_U = np.zeros((target_rank, mp*pn))
        for r in range(target_rank):
            M_U[r, :] = np.outer(V[:, r], W[:, r]).reshape(-1)
        # Solve U @ M_U = T_flat
        # U = T_flat @ pinv(M_U)
        try:
            U_new = T_flat @ np.linalg.pinv(M_U)
        except:
            break
        U = U_new
        
        # Update V
        M_V = np.zeros((target_rank, nm*pn))
        for r in range(target_rank):
            M_V[r, :] = np.outer(U[:, r], W[:, r]).reshape(-1)
        T_flat_V = T.reshape(mp, nm*pn)  # T[b, a, c] -> reshape
        # Actually T is (nm, mp, pn), so T[b,:,:] for fixed b
        # T_flat_V[b, ac] = T[a, b, c]
        T_flat_V = np.transpose(T, (1, 0, 2)).reshape(mp, nm*pn)
        try:
            V_new = T_flat_V @ np.linalg.pinv(M_V)
        except:
            break
        V = V_new
        
        # Update W
        M_W = np.zeros((target_rank, nm*mp))
        for r in range(target_rank):
            M_W[r, :] = np.outer(U[:, r], V[:, r]).reshape(-1)
        T_flat_W = np.transpose(T, (2, 0, 1)).reshape(pn, nm*mp)
        try:
            W_new = T_flat_W @ np.linalg.pinv(M_W)
        except:
            break
        W = W_new
    
    # Check error
    T_dec = np.zeros((nm, mp, pn), dtype=np.float64)
    for r in range(target_rank):
        T_dec += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    err = np.max(np.abs(T - T_dec))
    
    if err < 1e-8:
        # Try to snap to rational values
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2
        T_dec_snap = np.zeros((nm, mp, pn), dtype=np.float64)
        for r in range(target_rank):
            T_dec_snap += np.einsum('a,b,c->abc', U_snap[:, r], V_snap[:, r], W_snap[:, r])
        err_snap = np.max(np.abs(T - T_dec_snap))
        if err_snap < 1e-10:
            return U_snap, V_snap, W_snap, err_snap
    return U, V, W, err


def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    deadline = time.time() + 140  # safety margin
    
    # Build target tensor
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    
    # Schoolbook baseline
    rank_school = n * m * p
    U_school = np.zeros((nm, rank_school))
    V_school = np.zeros((mp, rank_school))
    W_school = np.zeros((pn, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1.0
                V_school[j*p + k, col] = 1.0
                W_school[k*n + i, col] = 1.0
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    # Try structured init + ALS for lower ranks
    # Known: rank(4x4x4) = 21, so rank(4x4x7) might be around 21 + something
    # Let's try ranks from 42 upward
    
    for target_rank in range(42, rank_school + 1):
        if time.time() > deadline - 20:
            break
        
        found = False
        for restart in range(3):
            if time.time() > deadline - 20:
                break
            
            U, V, W, err = _als_decompose(T, nm, mp, pn, target_rank, 
                                           max_iter=30, seed=42 + restart)
            if err < 1e-10:
                # Verify
                valid, verify_err = _verify(U, V, W, n, m, p)
                if valid:
                    best_U, best_V, best_W = U, V, W
                    best_rank = target_rank
                    found = True
                    break
        
        if found:
            # Check if we can go lower
            if target_rank > 42:
                continue
            else:
                break
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
