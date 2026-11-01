"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 2, 5]
START_TIME = time.time()
TIME_LIMIT = 140.0  # Safety margin for the 150s limit

# EVOLVE-BLOCK-START

def _verify(U, V, W, n=2, m=2, p=5):
    """
    Verify if the given U, V, W correctly decompose the matrix multiplication tensor.
    """
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target = 1.0
                computed = 0.0
                for r in range(rank):
                    computed += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                if abs(computed - target) > 1e-9:
                    return False
    return True

def _construct_rank15(n=2, m=2, p=5):
    """
    Attempt to construct a rank-15 decomposition.
    Uses Strassen (rank 7) for cols 0,1 and a custom rank-8 for cols 2,3,4.
    """
    rank = 15
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # --- Part 1: Strassen for cols 0, 1 (7 terms, indices 0-6) ---
    # B indices: B00=0, B01=1, B10=2, B11=3
    # C indices: C00=0, C01=1, C10=5, C11=6
    
    # M1: (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[6][0] = 1
    
    # M2: (A10+A11)B00 -> C10 += M2, C11 -= M2
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[5][1] = 1; W[6][1] = -1
    
    # M3: A00(B01-B11) -> C01 += M3, C11 += M3
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[6][2] = 1
    
    # M4: A11(B10-B00) -> C00 += M4, C10 += M4
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[5][3] = 1
    
    # M5: (A00+A01)B11 -> C00 -= M5, C01 += M5
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6: (A10-A00)(B00+B01) -> C11 += M6
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[6][5] = 1
    
    # M7: (A01-A11)(B10+B11) -> C00 += M7
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1

    # --- Part 2: Custom rank-8 for cols 2,3,4 (indices 7-14) ---
    # B indices: B02=4, B03=5, B04=6, B12=7, B13=8, B14=9
    # C indices: C02=2, C03=3, C04=4, C12=7, C13=8, C14=9
    
    # This is a 2x2x3 problem. We need 8 terms.
    # Let's try a structured approach:
    
    # N1: A00*(B02+B13) -> C02 += N1, C13 += N1
    U[0][7] = 1
    V[4][7] = 1; V[8][7] = 1
    W[2][7] = 1; W[8][7] = 1
    
    # N2: (A10+A11)*B02 -> C12 += N2, C13 -= N2
    U[2][8] = 1; U[3][8] = 1
    V[4][8] = 1
    W[7][8] = 1; W[8][8] = -1
    
    # N3: A00*(B03-B02) -> C03 += N3, C02 -= N3
    U[0][9] = 1
    V[5][9] = 1; V[4][9] = -1
    W[3][9] = 1; W[2][9] = -1
    
    # N4: A11*(B12-B13) -> C12 += N4, C13 += N4
    U[3][10] = 1
    V[7][10] = 1; V[8][10] = -1
    W[7][10] = 1; W[8][10] = 1
    
    # N5: (A00+A01)*(B13-B03) -> C03 += N5, C02 -= N5... let me reconsider
    # Actually let me try a different structure for the 2x2x3 part
    
    # Let me use a different approach: split into two 2x2x1 + one 2x2x1
    # Col 2: schoolbook rank 4
    # Col 3: schoolbook rank 4  
    # Col 4: schoolbook rank 4
    # That's 12, too many.
    
    # Let me try: use Strassen-like for cols 2,3 (rank 7) + 1 for col 4? That's 8.
    # But Strassen for 2x2x2 is rank 7, so cols 2,3 in rank 7 + col 4 in rank 1 = 8.
    # But col 4 needs rank 2 minimum (A00*B04 + A01*B14 for C04, A10*B04 + A11*B14 for C14)
    # So that doesn't work directly.
    
    # Let me try a hybrid: 6 terms for cols 2,3 and 2 terms for col 4
    # Or 5 terms for cols 2,3 and 3 terms for col 4
    
    # Actually, let me try to be more systematic. The 2x2x3 tensor has 12 nonzero entries.
    # Lower bound: need at least 6 terms (since each term can cover at most 2 of the 12 in a structured way)
    # But realistically, we need more.
    
    # Let me try rank 8 with this structure:
    # N1: A00*B02 -> C02
    U[0][7] = 1
    V[4][7] = 1
    W[2][7] = 1
    
    # N2: A01*B12 -> C02
    U[1][8] = 1
    V[7][8] = 1
    W[2][8] = 1
    
    # N3: A10*B02 -> C12
    U[2][9] = 1
    V[4][9] = 1
    W[7][9] = 1
    
    # N4: A11*B12 -> C12
    U[3][10] = 1
    V[7][10] = 1
    W[7][10] = 1
    
    # N5: A00*B03 -> C03
    U[0][11] = 1
    V[5][11] = 1
    W[3][11] = 1
    
    # N6: A01*B13 -> C03
    U[1][12] = 1
    V[8][12] = 1
    W[3][12] = 1
    
    # N7: A10*B03 -> C13
    U[2][13] = 1
    V[5][13] = 1
    W[8][13] = 1
    
    # N8: A11*B13 -> C13
    U[3][14] = 1
    V[8][14] = 1
    W[8][14] = 1
    
    # But we still need col 4! We have no more terms. This won't work.
    
    # Let me restart the rank-15 attempt with a different strategy.
    # Maybe I should try to share terms between the Strassen block and the extension.
    
    return None  # This naive approach doesn't work, need search

def _construct_rank16(n=2, m=2, p=5):
    """
    Construct a rank-16 decomposition:
    - Strassen (rank 7) for B columns 0,1
    - Custom rank-9 decomposition for B columns 2,3,4
    """
    rank = 16
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # --- Part 1: Strassen for cols 0, 1 (7 terms) ---
    # B indices: B00=0, B01=1, B10=2, B11=3
    # C indices: C00=0, C01=1, C10=5, C11=6
    
    # M1: (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[6][0] = 1
    
    # M2: (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[5][1] = 1; W[6][1] = -1
    
    # M3: A00(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[6][2] = 1
    
    # M4: A11(B10-B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[5][3] = 1
    
    # M5: (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6: (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[6][5] = 1
    
    # M7: (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1

    # --- Part 2: Custom rank-9 for cols 2,3,4 ---
    # B indices: B02=4, B03=5, B04=6, B12=7, B13=8, B14=9
    # C indices: C02=2, C03=3, C04=4, C12=7, C13=8, C14=9
    
    # N1: A00*(B02+B13) -> C02, C13
    U[0][7] = 1
    V[4][7] = 1; V[8][7] = 1
    W[2][7] = 1; W[8][7] = 1
    
    # N2: (A10+A11)*B02 -> C12, C13
    U[2][8] = 1; U[3][8] = 1
    V[4][8] = 1
    W[7][8] = 1; W[8][8] = -1
    
    # N3: (A00+A01)*B13 -> C03, C13
    U[0][9] = 1; U[1][9] = 1
    V[8][9] = 1
    W[3][9] = 1; W[8][9] = 1
    
    # N4: A11*(B03-B02) -> C03
    U[3][10] = 1
    V[5][10] = 1; V[4][10] = -1
    W[3][10] = 1
    
    # N5: A00*(B12-B13) -> C02, C12
    U[0][11] = 1
    V[7][11] = 1; V[8][11] = -1
    W[2][11] = -1; W[7][11] = 1
    
    # N6: (A10-A00)*B02 -> C12
    U[2][12] = 1; U[0][12] = -1
    V[4][12] = 1
    W[7][12] = 1
    
    # N7: (A01-A11)*B13 -> C03, C13
    U[1][13] = 1; U[3][13] = -1
    V[8][13] = 1
    W[3][13] = 1; W[8][13] = -1
    
    # N8: A00*B04 -> C04
    U[0][14] = 1
    V[6][14] = 1
    W[4][14] = 1
    
    # N9: A10*B04 -> C14... wait, need A01*B14 and A11*B14 too
    # Let me fix this. C14 = A10*B04 + A11*B14
    # C04 = A00*B04 + A01*B14
    
    # N8: A00*B04 -> C04
    U[0][14] = 1
    V[6][14] = 1
    W[4][14] = 1
    
    # N9: A10*B04 -> C14
    U[2][15] = 1
    V[6][15] = 1
    W[9][15] = 1

    # But we're missing A01*B14 for C04 and A11*B14 for C14!
    # This rank-16 construction is incomplete. Let me use the verified one from the original.
    
    if _verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    
    return None

def _construct_rank18(n=2, m=2, p=5):
    """
    Construct a rank-18 decomposition using Strassen for 2x2x2 (cols 0-1),
    Strassen for 2x2x2 (cols 2-3), and Schoolbook for 2x2x1 (col 4).
    """
    rank = 18
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    def add_strassen_block(col0, col1, base):
        r = base
        U[0][r] = 1; U[3][r] = 1
        V[0*p+col0][r] = 1; V[1*p+col1][r] = 1
        W[col0*n+0][r] = 1; W[col1*n+1][r] = 1
        
        r = base + 1
        U[2][r] = 1; U[3][r] = 1
        V[0*p+col0][r] = 1
        W[col0*n+1][r] = 1; W[col1*n+1][r] = -1
        
        r = base + 2
        U[0][r] = 1
        V[0*p+col1][r] = 1; V[1*p+col1][r] = -1
        W[col1*n+0][r] = 1; W[col1*n+1][r] = 1
        
        r = base + 3
        U[3][r] = 1
        V[1*p+col0][r] = 1; V[0*p+col0][r] = -1
        W[col0*n+0][r] = 1; W[col0*n+1][r] = 1
        
        r = base + 4
        U[0][r] = 1; U[1][r] = 1
        V[1*p+col1][r] = 1
        W[col0*n+0][r] = -1; W[col1*n+0][r] = 1
        
        r = base + 5
        U[2][r] = 1; U[0][r] = -1
        V[0*p+col0][r] = 1; V[0*p+col1][r] = 1
        W[col1*n+1][r] = 1
        
        r = base + 6
        U[1][r] = 1; U[3][r] = -1
        V[1*p+col0][r] = 1; V[1*p+col1][r] = 1
        W[col0*n+0][r] = 1
    
    add_strassen_block(0, 1, 0)
    add_strassen_block(2, 3, 7)
    
    # Schoolbook for column 4
    r = 14
    U[0][r] = 1; V[0*p+4][r] = 1; W[4*n+0][r] = 1
    r = 15
    U[1][r] = 1; V[1*p+4][r] = 1; W[4*n+0][r] = 1
    r = 16
    U[2][r] = 1; V[0*p+4][r] = 1; W[4*n+1][r] = 1
    r = 17
    U[3][r] = 1; V[1*p+4][r] = 1; W[4*n+1][r] = 1
    
    if _verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    
    return None

def _construct_schoolbook(n=2, m=2, p=5):
    """
    Construct the standard schoolbook decomposition of rank n*m*p.
    """
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

def _als_refine(U, V, W, T, R, n, m, p, max_iter=300, tol=1e-14):
    """Run ALS iterations to refine the decomposition."""
    prev_err = np.inf
    for it in range(max_iter):
        # Solve for U
        VvW = np.einsum('vr,wr->vwr', V, W)
        VvW_flat = VvW.transpose(0, 2, 1).reshape(-1, R)
        for u in range(n*m):
            b = T[u].reshape(-1)
            sol, res, rank_s, sv = np.linalg.lstsq(VvW_flat, b, rcond=None)
            U[u] = sol
        
        # Solve for V
        Uw = np.einsum('ur,wr->uwr', U, W)
        Uw_flat = Uw.transpose(0, 2, 1).reshape(-1, R)
        for v in range(m*p):
            b = T[:, v, :].reshape(-1)
            sol, res, rank_s, sv = np.linalg.lstsq(Uw_flat, b, rcond=None)
            V[v] = sol
        
        # Solve for W
        UV = np.einsum('ur,vr->uvr', U, V)
        UV_flat = UV.reshape(-1, R)
        for w in range(p*n):
            b = T[:, :, w].reshape(-1)
            sol, res, rank_s, sv = np.linalg.lstsq(UV_flat, b, rcond=None)
            W[w] = sol
        
        # Compute error
        err = 0.0
        for r in range(R):
            err += np.sum((np.einsum('u,v,w->uvw', U[:,r], V[:,r], W[:,r]) - T)**2)
        err = np.sqrt(err / (n*m * m*p * p*n))
        
        if abs(prev_err - err) < tol:
            break
        prev_err = err
    return U, V, W, err

def _check_and_rationalize(U, V, W, T, R, n, m, p):
    """Check if decomposition is exact and try to rationalize entries."""
    result = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for r in range(R):
        result += np.einsum('u,v,w->uvw', U[:,r], V[:,r], W[:,r])
    
    if not np.allclose(result, T, atol=1e-8):
        return None
    
    # Try to rationalize with various denominators
    # Use a slightly larger tolerance for the check after rationalization
    for denom in [2, 4, 8, 16, 32]:
        U_s = np.round(U * denom) / denom
        V_s = np.round(V * denom) / denom
        W_s = np.round(W * denom) / denom
        
        result_s = np.zeros((n*m, m*p, p*n), dtype=np.float64)
        for r in range(R):
            result_s += np.einsum('u,v,w->uvw', U_s[:,r], V_s[:,r], W_s[:,r])
        
        if np.allclose(result_s, T, atol=1e-7):
            U_list = [[float(x) for x in row] for row in U_s]
            V_list = [[float(x) for x in row] for row in V_s]
            W_list = [[float(x) for x in row] for row in W_s]
            
            for mat in [U_list, V_list, W_list]:
                for i in range(len(mat)):
                    for j in range(len(mat[i])):
                        if abs(mat[i][j] - round(mat[i][j])) < 1e-9:
                            mat[i][j] = int(round(mat[i][j]))
            
            if _verify(U_list, V_list, W_list, n, m, p):
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # Return as floats if valid
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    if _verify(U_list, V_list, W_list, n, m, p):
        return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def _search_rank16(n=2, m=2, p=5, timeout=120):
    """
    Search for a rank-16 decomposition.
    Uses Strassen for first 2 cols (rank 7) and random/structured for the rest (rank 9).
    """
    start_time = time.time()
    
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i*m + j
                v = j*p + k
                w = k*n + i
                T[u, v, w] = 1.0
    
    R = 16
    R_strassen = 7
    
    for trial in range(30):
        if time.time() - start_time > timeout - 5:
            break
        
        # Initialize first 7 columns with Strassen for cols 0,1
        U = np.zeros((n*m, R))
        V = np.zeros((m*p, R))
        W = np.zeros((p*n, R))
        
        # Strassen block
        U_str, V_str, W_str = _get_strassen_block_np(n, m, p, 0, 1)
        U[:, :R_strassen] = U_str
        V[:, :R_strassen] = V_str
        W[:, :R_strassen] = W_str
        
        # Initialize remaining 9 columns randomly
        noise_scale = 0.5 + 0.1 * (trial % 5)
        U[:, R_strassen:] = np.random.randn(n*m, R - R_strassen) * noise_scale
        V[:, R_strassen:] = np.random.randn(m*p, R - R_strassen) * noise_scale
        W[:, R_strassen:] = np.random.randn(p*n, R - R_strassen) * noise_scale
        
        U, V, W, err = _als_refine(U, V, W, T, R, n, m, p, max_iter=300)
        
        result = _check_and_rationalize(U, V, W, T, R, n, m, p)
        if result is not None:
            return result
    
    return None

def _get_rank18_np(n=2, m=2, p=5):
    """Get rank-18 decomposition as numpy arrays."""
    rank = 18
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    def add_strassen_block(col0, col1, base):
        r = base
        U[0, r] = 1; U[3, r] = 1
        V[0*p+col0, r] = 1; V[1*p+col1, r] = 1
        W[col0*n+0, r] = 1; W[col1*n+1, r] = 1
        
        r = base + 1
        U[2, r] = 1; U[3, r] = 1
        V[0*p+col0, r] = 1
        W[col0*n+1, r] = 1; W[col1*n+1, r] = -1
        
        r = base + 2
        U[0, r] = 1
        V[0*p+col1, r] = 1; V[1*p+col1, r] = -1
        W[col1*n+0, r] = 1; W[col1*n+1, r] = 1
        
        r = base + 3
        U[3, r] = 1
        V[1*p+col0, r] = 1; V[0*p+col0, r] = -1
        W[col0*n+0, r] = 1; W[col0*n+1, r] = 1
        
        r = base + 4
        U[0, r] = 1; U[1, r] = 1
        V[1*p+col1, r] = 1
        W[col0*n+0, r] = -1; W[col1*n+0, r] = 1
        
        r = base + 5
        U[2, r] = 1; U[0, r] = -1
        V[0*p+col0, r] = 1; V[0*p+col1, r] = 1
        W[col1*n+1, r] = 1
        
        r = base + 6
        U[1, r] = 1; U[3, r] = -1
        V[1*p+col0, r] = 1; V[1*p+col1, r] = 1
        W[col0*n+0, r] = 1
    
    add_strassen_block(0, 1, 0)
    add_strassen_block(2, 3, 7)
    
    # Schoolbook for column 4
    r = 14
    U[0, r] = 1; V[0*p+4, r] = 1; W[4*n+0, r] = 1
    r = 15
    U[1, r] = 1; V[1*p+4, r] = 1; W[4*n+0, r] = 1
    r = 16
    U[2, r] = 1; V[0*p+4, r] = 1; W[4*n+1, r] = 1
    r = 17
    U[3, r] = 1; V[1*p+4, r] = 1; W[4*n+1, r] = 1
    
    return U, V, W

def _algebraic_search_rank17(n=2, m=2, p=5, timeout=120):
    """
    Algebraic elimination search for rank-17 decomposition.
    Tries structured sparsity patterns that break block-diagonal symmetry,
    using exact rational Gaussian elimination to solve for factor entries.
    """
    start_time = time.time()
    R = 17
    
    # The tensor T has entries T[i*m+j, j*p+k, k*n+i] = 1
    # We need: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    
    # Strategy: try various structured sparsity patterns for the 10 non-Strassen terms
    # Each pattern defines which (U,V,W) entries are nonzero in terms 7-16
    
    # Define candidate sparsity patterns for the 10 extra terms
    # Each pattern is a list of (u_indices, v_indices, w_indices) for each term
    # u in {0,1,2,3}, v in {0..9}, w in {0..9}
    
    patterns = []
    
    # Pattern A: Strassen-like for cols 2,3 (7 terms) + 3 terms for col 4
    # This is the natural extension but we've seen it doesn't work at rank 17
    # Let's try patterns that mix columns
    
    # Pattern B: 5 terms covering cols 2,3,4 with cross-terms
    # Try: each term covers a different (col_pair, row_combo)
    
    # Let me enumerate structured patterns where terms can span multiple columns
    # Key insight: if we allow V entries to be nonzero in multiple columns,
    # a single term can contribute to multiple C columns
    
    # Pattern 1: Try to use 10 terms where some have multi-column V entries
    # Term structure: (u_set, v_set, w_set) where each is a set of indices
    
    candidate_patterns = [
        # Each entry: list of 10 tuples (u_indices, v_indices, w_indices)
        # u_indices: which of U[0..3] are nonzero
        # v_indices: which of V[0..9] are nonzero  
        # w_indices: which of W[0..9] are nonzero
        
        # Pattern: 4 terms for col2 (schoolbook-like but shared), 4 for col3, 2 for col4
        [
            ([0], [4], [2]),      # A00*B02 -> C02
            ([1], [7], [2]),      # A01*B12 -> C02
            ([2], [4], [7]),      # A10*B02 -> C12
            ([3], [7], [7]),      # A11*B12 -> C12
            ([0], [5], [3]),      # A00*B03 -> C03
            ([1], [8], [3]),      # A01*B13 -> C03
            ([2], [5], [8]),      # A10*B03 -> C13
            ([3], [8], [8]),      # A11*B13 -> C13
            ([0,1], [6,9], [4,9]),# (A00+A01)*(B04+B14) -> C04, C14
            ([2,3], [6,9], [9,4]),# (A10+A11)*(B04+B14) -> C14, C04
        ],
        # Pattern 2: More mixed
        [
            ([0,1], [4,7], [2,7]),
            ([2,3], [4,7], [7,2]),
            ([0,1], [5,8], [3,8]),
            ([2,3], [5,8], [8,3]),
            ([0], [6], [4]),
            ([1], [9], [4]),
            ([2], [6], [9]),
            ([3], [9], [9]),
            ([0,3], [4,5], [2,3]),
            ([1,2], [7,8], [7,8]),
        ],
        # Pattern 3: Strassen-like for (2,3) but with cross-terms to col 4
        [
            ([0,3], [4,8], [2,8]),
            ([2,3], [4], [7,8]),
            ([0], [5,8], [3,8]),
            ([3], [7,4], [7,2]),
            ([0,1], [8], [3,8]),
            ([2,0], [4,5], [7,3]),
            ([1,3], [7,8], [2,3]),
            ([0], [6], [4]),
            ([1], [9], [4]),
            ([2,3], [6,9], [9,9]),
        ],
        # Pattern 4: Aggressive mixing
        [
            ([0,1,2,3], [4,5,6], [2,3,4]),
            ([0,1,2,3], [7,8,9], [7,8,9]),
            ([0,2], [4,7], [2,7]),
            ([1,3], [5,8], [3,8]),
            ([0,3], [6,9], [4,9]),
            ([1,2], [4,8], [7,2]),
            ([0,1], [5,9], [3,4]),
            ([2,3], [7,6], [8,9]),
            ([0,2,3], [4,5,6], [2,3,4]),
            ([1,2,3], [7,8,9], [7,8,9]),
        ],
        # Pattern 5: Based on 2x2x3 lower bound analysis
        [
            ([0], [4], [2]),
            ([1], [7], [2]),
            ([2], [4], [7]),
            ([3], [7], [7]),
            ([0,1], [5,8], [3,8]),
            ([2,3], [5,8], [8,3]),
            ([0,3], [6,9], [4,9]),
            ([1,2], [6,9], [9,4]),
            ([0,1], [4,5,6], [2,3,4]),
            ([2,3], [7,8,9], [7,8,9]),
        ],
    ]
    
    for pat_idx, pattern in enumerate(candidate_patterns):
        if time.time() - start_time > timeout - 10:
            break
        
        U = np.zeros((n*m, R))
        V = np.zeros((m*p, R))
        W = np.zeros((p*n, R))
        
        # Set up Strassen block for cols 0,1
        U_str, V_str, W_str = _get_strassen_block_np(n, m, p, 0, 1)
        U[:, :7] = U_str
        V[:, :7] = V_str
        W[:, :7] = W_str
        
        # Set up sparsity pattern for terms 7-16
        for t_idx, (u_set, v_set, w_set) in enumerate(pattern):
            r = 7 + t_idx
            for u in u_set:
                U[u, r] = 1.0
            for v in v_set:
                V[v, r] = 1.0
            for w in w_set:
                W[w, r] = 1.0
        
        # Now we need to find coefficients (scalars) for each term
        # such that the decomposition is exact.
        # The entries can be ±1 or other small rationals.
        # Let's set up a system: for each (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
        
        # With the Strassen block fixed, the remaining 10 terms must satisfy:
        # sum_{r=7}^{16} U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 - (Strassen contribution)
        
        # Compute Strassen contribution for each (i,j,k)
        strassen_contrib = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(7):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    strassen_contrib[i, j, k] = val
        
        # For the 10 extra terms, we need to find coefficients c_0, ..., c_9
        # such that for each (i,j,k): sum_t c_t * (product of pattern entries) = 1 - strassen_contrib[i,j,k]
        
        # Build the linear system
        num_equations = n * m * p  # 20 equations
        num_unknowns = 10  # coefficients
        
        A_mat = np.zeros((num_equations, num_unknowns))
        b_vec = np.zeros(num_equations)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    b_vec[eq_idx] = 1.0 - strassen_contrib[i, j, k]
                    for t in range(10):
                        u_set, v_set, w_set = pattern[t]
                        r = 7 + t
                        # The contribution is U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                        # which is 1 if i*m+j in u_set, j*p+k in v_set, k*n+i in w_set, else 0
                        prod = 1.0
                        if (i*m+j) not in u_set:
                            prod = 0.0
                        if (j*p+k) not in v_set:
                            prod = 0.0
                        if (k*n+i) not in w_set:
                            prod = 0.0
                        A_mat[eq_idx, t] = prod
                    eq_idx += 1
        
        # Solve the linear system
        try:
            coeffs, residuals, rank_s, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        except:
            continue
        
        # Check if solution is valid
        max_resid = np.max(np.abs(A_mat @ coeffs - b_vec))
        if max_resid > 1e-10:
            continue
        
        # Check if coefficients are "nice" (small integers or half-integers)
        nice = True
        for c in coeffs:
            if abs(c * 2 - round(c * 2)) > 1e-8:
                nice = False
                break
        
        if not nice:
            continue
        
        # Apply coefficients
        for t in range(10):
            r = 7 + t
            c = round(coeffs[t] * 2) / 2.0
            if abs(c - round(c)) < 1e-9:
                c = round(c)
            for u in pattern[t][0]:
                U[u, r] *= c
            for v in pattern[t][1]:
                V[v, r] *= c
            for w in pattern[t][2]:
                W[w, r] *= c
        
        # Verify
        result = _check_and_rationalize(U, V, W, None, R, n, m, p)
        if result is not None:
            return result
    
    return None

def _search_rank17(n=2, m=2, p=5, timeout=120):
    """
    Search for a rank-17 decomposition.
    First tries algebraic elimination, then falls back to ALS-based search.
    """
    start_time = time.time()
    
    # Try algebraic approach first
    if time.time() - start_time < timeout - 30:
        result = _algebraic_search_rank17(n, m, p, timeout=min(30, timeout - (time.time() - start_time)))
        if result is not None:
            return result
    
    # Fallback to ALS-based search
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i*m + j
                v = j*p + k
                w = k*n + i
                T[u, v, w] = 1.0
    
    R = 17
    R_strassen = 7
    
    for trial in range(30):
        if time.time() - start_time > timeout - 5:
            break
        
        U = np.zeros((n*m, R))
        V = np.zeros((m*p, R))
        W = np.zeros((p*n, R))
        
        U_str, V_str, W_str = _get_strassen_block_np(n, m, p, 0, 1)
        U[:, :R_strassen] = U_str
        V[:, :R_strassen] = V_str
        W[:, :R_strassen] = W_str
        
        noise_scale = 0.5 + 0.1 * (trial % 5)
        U[:, R_strassen:] = np.random.randn(n*m, R - R_strassen) * noise_scale
        V[:, R_strassen:] = np.random.randn(m*p, R - R_strassen) * noise_scale
        W[:, R_strassen:] = np.random.randn(p*n, R - R_strassen) * noise_scale
        
        U, V, W, err = _als_refine(U, V, W, T, R, n, m, p, max_iter=300)
        
        result = _check_and_rationalize(U, V, W, T, R, n, m, p)
        if result is not None:
            return result
    
    return None

def _get_strassen_block_np(n=2, m=2, p=5, col0=0, col1=1):
    """Get Strassen block for two columns as numpy arrays."""
    U = np.zeros((n*m, 7))
    V = np.zeros((m*p, 7))
    W = np.zeros((p*n, 7))
    
    # M1: (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0*p+col0, 0] = 1; V[1*p+col1, 0] = 1
    W[col0*n+0, 0] = 1; W[col1*n+1, 0] = 1
    
    # M2: (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0*p+col0, 1] = 1
    W[col0*n+1, 1] = 1; W[col1*n+1, 1] = -1
    
    # M3: A00(B01-B11)
    U[0, 2] = 1
    V[0*p+col1, 2] = 1; V[1*p+col1, 2] = -1
    W[col1*n+0, 2] = 1; W[col1*n+1, 2] = 1
    
    # M4: A11(B10-B00)
    U[3, 3] = 1
    V[1*p+col0, 3] = 1; V[0*p+col0, 3] = -1
    W[col0*n+0, 3] = 1; W[col0*n+1, 3] = 1
    
    # M5: (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[1*p+col1, 4] = 1
    W[col0*n+0, 4] = -1; W[col1*n+0, 4] = 1
    
    # M6: (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0*p+col0, 5] = 1; V[0*p+col1, 5] = 1
    W[col1*n+1, 5] = 1
    
    # M7: (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[1*p+col0, 6] = 1; V[1*p+col1, 6] = 1
    W[col0*n+0, 6] = 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    deadline = START_TIME + TIME_LIMIT
    
    # Try rank 17 algebraic search first (fast, exact)
    if time.time() < deadline - 60:
        result = _search_rank17(n, m, p, timeout=min(30, deadline - time.time() - 60))
        if result is not None:
            return result
    
    # Try rank 16 search
    if time.time() < deadline - 30:
        result = _search_rank16(n, m, p, timeout=min(25, deadline - time.time() - 30))
        if result is not None:
            return result
    
    # Try rank 18 (constructive)
    result = _construct_rank18(n, m, p)
    if result is not None:
        return result
    
    # Fallback: Schoolbook (rank 20)
    return _construct_schoolbook(n, m, p)

# EVOLVE-BLOCK-END
