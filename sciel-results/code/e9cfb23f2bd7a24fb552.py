"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from fractions import Fraction

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    recon = np.einsum('ar,br,cr->abc', U, V, W)
    if not np.allclose(recon, T, atol=1e-9):
        return False
    return True

def _build_tensor(n, m, p):
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _als_decompose(T, rank, max_iter=200, tol=1e-10, seed=42):
    """ALS decomposition of tensor T into rank-many rank-1 tensors."""
    rng = np.random.RandomState(seed)
    nA, nB, nC = T.shape
    U = rng.randn(nA, rank) * 0.1
    V = rng.randn(nB, rank) * 0.1
    W = rng.randn(nC, rank) * 0.1
    
    best_err = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        # Update U
        VW = np.einsum('br,cr->bcr', V, W)
        denom_U = np.einsum('bcr,bcr->r', VW, VW)
        for r in range(rank):
            if denom_U[r] > 1e-15:
                U[:, r] = np.einsum('abc,ab->c', T, VW[:,:,r]) / denom_U[r]
        
        # Update V
        UW = np.einsum('ar,cr->acr', U, W)
        denom_V = np.einsum('acr,acr->r', UW, UW)
        for r in range(rank):
            if denom_V[r] > 1e-15:
                V[:, r] = np.einsum('abc,ac->c', T, UW[:,:,r]) / denom_V[r]
        
        # Update W
        UV = np.einsum('ar,br->abr', U, V)
        denom_W = np.einsum('abr,abr->r', UV, UV)
        for r in range(rank):
            if denom_W[r] > 1e-15:
                W[:, r] = np.einsum('abc,ab->c', T, UV[:,:,r]) / denom_W[r]
        
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.linalg.norm(recon - T) / (np.linalg.norm(T) + 1e-15)
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        if err < tol:
            break
    
    return best_U, best_V, best_W, best_err

def _round_and_verify(U, V, W, n, m, p, max_denom=4):
    """Try rounding to rationals with small denominators and verify."""
    T = _build_tensor(n, m, p)
    for denom in [1, 2, 3, 4, max_denom]:
        U_r = np.round(U * denom) / denom
        V_r = np.round(V * denom) / denom
        W_r = np.round(W * denom) / denom
        recon = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
        if np.allclose(recon, T, atol=1e-9):
            return U_r, V_r, W_r, denom
    return None, None, None, None

def _build_rank44_decomposition(n, m, p):
    """Build a rank-44 decomposition for 2x3 x 3x8 matrix multiplication."""
    n, m, p = 2, 3, 8
    rank = 44
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    for s in range(4):
        base_rank = s * 11
        k0 = 2 * s
        k1 = 2 * s + 1
        
        # M1
        c = base_rank + 0
        U[0*3+0, c] = 1.0
        U[1*3+1, c] = 1.0
        V[0*8+k0, c] = 1.0
        V[1*8+k1, c] = 1.0
        W[k0*2+0, c] = 1.0
        W[k1*2+1, c] = 1.0
        
        # M2
        c = base_rank + 1
        U[1*3+0, c] = 1.0
        U[1*3+1, c] = 1.0
        V[0*8+k0, c] = 1.0
        W[k0*2+1, c] = 1.0
        
        # M3
        c = base_rank + 2
        U[0*3+0, c] = 1.0
        V[0*8+k1, c] = 1.0
        V[1*8+k1, c] = -1.0
        W[k1*2+0, c] = 1.0
        
        # M4
        c = base_rank + 3
        U[1*3+1, c] = 1.0
        V[1*8+k0, c] = 1.0
        V[0*8+k0, c] = -1.0
        W[k0*2+0, c] = 1.0
        W[k0*2+1, c] = 1.0
        
        # M5
        c = base_rank + 4
        U[0*3+0, c] = 1.0
        U[0*3+1, c] = 1.0
        V[1*8+k1, c] = 1.0
        W[k0*2+0, c] = -1.0
        W[k1*2+0, c] = 1.0
        
        # M6
        c = base_rank + 5
        U[1*3+0, c] = 1.0
        U[0*3+0, c] = -1.0
        V[0*8+k0, c] = 1.0
        V[0*8+k1, c] = 1.0
        W[k1*2+1, c] = 1.0
        
        # M7
        c = base_rank + 6
        U[0*3+1, c] = 1.0
        U[1*3+1, c] = -1.0
        V[1*8+k0, c] = 1.0
        V[1*8+k1, c] = 1.0
        W[k0*2+0, c] = 1.0

        # Schoolbook for j=2
        c = base_rank + 7
        U[0*3+2, c] = 1.0
        V[2*8+k0, c] = 1.0
        W[k0*2+0, c] = 1.0
        
        c = base_rank + 8
        U[0*3+2, c] = 1.0
        V[2*8+k1, c] = 1.0
        W[k1*2+0, c] = 1.0
        
        c = base_rank + 9
        U[1*3+2, c] = 1.0
        V[2*8+k0, c] = 1.0
        W[k0*2+1, c] = 1.0
        
        c = base_rank + 10
        U[1*3+2, c] = 1.0
        V[2*8+k1, c] = 1.0
        W[k1*2+1, c] = 1.0

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _build_rank40_decomposition(n, m, p):
    """
    Try to build a rank-40 decomposition using a more efficient structure.
    This uses the fact that for 2x3 x 3x8, we can potentially share more
    multiplications across the k-dimension.
    """
    n, m, p = 2, 3, 8
    rank = 40
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # For each pair of k values (k0, k1), use Strassen-like 7 multiplications
    # for the 2x2 sub-problems involving j=0 and j=1
    # Plus schoolbook for j=2
    
    # Actually, let's try a different approach: use 10 multiplications per 2 k-values
    # instead of 11, by finding a better decomposition for the 2x3 x 3x2 case
    
    # For 2x3 x 3x2, the schoolbook is 12. Can we do better?
    # Let's try rank 10 for 2x3 x 3x2
    
    for s in range(4):
        base = s * 10
        k0 = 2 * s
        k1 = 2 * s + 1
        
        # We need to find a rank-10 decomposition for 2x3 x 3x2
        # This is equivalent to multiplying a 2x3 matrix by a 3x2 matrix
        # Output is 2x2
        
        # Let's try to find this via a systematic search
        # For now, use schoolbook (12) as fallback
        # But let's try to be smarter
        
        # Actually, let's just use the known rank-11 structure but try to
        # see if we can do rank 10
        
        # For 2x3 x 3x2:
        # A is 2x3, B is 3x2, C is 2x2
        # C[i][k] = sum_j A[i][j] * B[j][k]
        
        # Let's try to find a rank-10 decomposition
        # This would save 4 multiplications total (1 per k-pair)
        
        # For now, let's use a heuristic approach
        # Use 7 for the 2x2 part (j=0,1) and 3 for j=2
        # That's 10 total
        
        # j=0,1 part (2x2 x 2x2 -> 2x2): use Strassen (rank 7)
        # j=2 part (2x1 x 1x2 -> 2x2): schoolbook (rank 2)
        # Total: 9
        
        # Wait, that's only 9. Let me recount.
        # A[i][0]*B[0][k] + A[i][1]*B[1][k] + A[i][2]*B[2][k]
        # The first two terms form a 2x2 x 2x2 multiplication
        # The third term is 2x1 x 1x2
        
        # Strassen for 2x2: 7 multiplications
        # 2x1 x 1x2: 2 multiplications (one for each output row)
        # Total: 9
        
        # So rank 9 per k-pair, total rank 36?
        # But Strassen gives us the 2x2 result from j=0,1
        # And we need to add the j=2 contribution
        
        # Let's build this
        # Strassen for M = [[A00, A01], [A10, A11]] * [[B00, B01], [B10, B11]]
        # where Aij = A[i][j], Bij = B[j][k]
        
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # For our case:
        # A00 = A[0][0], A01 = A[0][1]
        # A10 = A[1][0], A11 = A[1][1]
        # B00 = B[0][k0], B01 = B[0][k1]
        # B10 = B[1][k0], B11 = B[1][k1]
        
        # M1: (A[0][0]+A[1][1]) * (B[0][k0]+B[1][k0]) for k=k0
        #      (A[0][0]+A[1][1]) * (B[0][k1]+B[1][k1]) for k=k1
        
        # This is getting complex. Let me just use the rank-11 structure
        # and accept rank 44 for now, but try to find better via search.
        
        pass
    
    # Fallback: return the rank-44 decomposition
    return _build_rank44_decomposition(n, m, p)

def construct():
    """
    Find a low-rank decomposition of 2x3*3x8 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # First, get the rank-44 baseline
    result44 = _build_rank44_decomposition(n, m, p)
    if _verify(result44["U"], result44["V"], result44["W"], n, m, p):
        best_result = result44
        best_rank = 44
    else:
        # Fallback schoolbook
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
        best_result = {"U": U, "V": V, "W": W}
        best_rank = rank
        return best_result
    
    # Try to find lower rank via ALS
    T = _build_tensor(n, m, p)
    
    for target_rank in range(43, 30, -1):
        if time.time() > deadline:
            break
        
        found = False
        for seed in range(5):
            if time.time() > deadline:
                break
            
            U_a, V_a, W_a, err = _als_decompose(T, target_rank, max_iter=100, tol=1e-12, seed=seed)
            
            if err < 1e-8:
                # Try rounding
                U_r, V_r, W_r, denom = _round_and_verify(U_a, V_a, W_a, n, m, p)
                if U_r is not None:
                    result = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                    if _verify(result["U"], result["V"], result["W"], n, m, p):
                        best_result = result
                        best_rank = target_rank
                        found = True
                        break
        
        if found:
            break
    
    return best_result
# EVOLVE-BLOCK-END
