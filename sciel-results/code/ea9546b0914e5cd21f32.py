import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 4]

def _build_target_tensor(n, m, p):
    return np.ones((n, m, p), dtype=np.float64)

def _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
    rank = U.shape[1]
    T = np.zeros((n, m, p), dtype=np.float64)
    for r in range(rank):
        u = U[:, r].reshape(n, m)
        v = V[:, r].reshape(m, p)
        w = W[:, r].reshape(p, n)
        T += np.einsum('ij,jk,ki->ijk', u, v, w)
    return np.allclose(T, 1.0, atol=tol, rtol=0)

def _construct_laderman_4x4x4():
    """
    Construct the rank-23 Laderman decomposition for 4x4 x 4x4 matrix multiplication.
    
    The tensor T[i,j,k] = A[i,j] * B[j,k] for 4x4 matrices.
    We use the standard Laderman scheme which uses 23 scalar multiplications.
    
    The decomposition is:
    T = sum_{r=1}^{23} u_r ⊗ v_r ⊗ w_r
    
    where each term is a rank-1 tensor.
    
    Laderman's algorithm for 4x4x4:
    Partition into 2x2 blocks:
    A = [[A11, A12], [A21, A22]]  (each 2x2)
    B = [[B11, B12], [B21, B22]]  (each 2x2)
    
    Then use Strassen for the 2x2 block multiplication with 7 multiplications per block pair,
    but Laderman optimizes this to 23 total.
    
    The 23 terms are:
    M1 = (A11 + A22) * (B11 + B22)
    M2 = (A21 + A22) * B11
    M3 = A11 * (B12 - B22)
    M4 = A22 * (B21 - B11)
    M5 = (A11 + A12) * B22
    M6 = (A21 - A11) * (B11 + B12)
    M7 = (A12 - A22) * (B21 + B22)
    
    Plus 16 more terms for the other block combinations... 
    Actually, Laderman's full 23-term scheme is more complex.
    
    Let me use the explicit Laderman decomposition.
    """
    n, m, p = 4, 4, 4
    rank = 23
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Laderman's 23 terms for 4x4x4
    # Using the standard formulation with 2x2 blocks
    
    def idx_A(i, j):
        return i * m + j
    
    def idx_B(j, k):
        return j * p + k
    
    def idx_W(k, i):
        return k * n + i
    
    # Term 1: M1 = (A11 + A22)(B11 + B22)
    # A11 = A[0:2, 0:2], A22 = A[2:4, 2:4]
    # B11 = B[0:2, 0:2], B22 = B[2:4, 2:4]
    r = 0
    for i in range(2):
        for j in range(2):
            U[idx_A(i, j), r] = 1  # A11 part
            U[idx_A(i+2, j+2), r] = 1  # A22 part
    for j in range(2):
        for k in range(2):
            V[idx_B(j, k), r] = 1  # B11 part
            V[idx_B(j+2, k+2), r] = 1  # B22 part
    for k in range(2):
        for i in range(2):
            W[idx_W(k, i), r] = 1
            W[idx_W(k+2, i+2), r] = 1
    r += 1
    
    # Term 2: M2 = (A21 + A22)B11
    # A21 = A[2:4, 0:2], A22 = A[2:4, 2:4]
    # B11 = B[0:2, 0:2]
    r = 1
    for i in range(2, 4):
        for j in range(2):
            U[idx_A(i, j), r] = 1  # A21 part
            U[idx_A(i, j+2), r] = 1  # A22 part
    for j in range(2):
        for k in range(2):
            V[idx_B(j, k), r] = 1  # B11 part
    for k in range(2):
        for i in range(2, 4):
            W[idx_W(k, i), r] = 1
            W[idx_W(k+2, i), r] = 1
    r += 1
    
    # Term 3: M3 = A11(B12 - B22)
    # A11 = A[0:2, 0:2]
    # B12 = B[0:2, 2:4], B22 = B[2:4, 2:4]
    r = 2
    for i in range(2):
        for j in range(2):
            U[idx_A(i, j), r] = 1  # A11 part
    for j in range(2):
        for k in range(2):
            V[idx_B(j, k+2), r] = 1  # B12 part
            V[idx_B(j+2, k+2), r] = -1  # B22 part
    for k in range(2):
        for i in range(2):
            W[idx_W(k+2, i), r] = 1
    r += 1
    
    # Term 4: M4 = A22(B21 - B11)
    # A22 = A[2:4, 2:4]
    # B21 = B[2:4, 0:2], B11 = B[0:2, 0:2]
    r = 3
    for i in range(2, 4):
        for j in range(2, 4):
            U[idx_A(i, j), r] = 1  # A22 part
    for j in range(2):
        for k in range(2):
            V[idx_B(j+2, k), r] = 1  # B21 part
            V[idx_B(j, k), r] = -1  # B11 part
    for k in range(2):
        for i in range(2):
            W[idx_W(k, i+2), r] = 1
    r += 1
    
    # Term 5: M5 = (A11 + A12)B22
    # A11 = A[0:2, 0:2], A12 = A[0:2, 2:4]
    # B22 = B[2:4, 2:4]
    r = 4
    for i in range(2):
        for j in range(2):
            U[idx_A(i, j), r] = 1  # A11 part
            U[idx_A(i, j+2), r] = 1  # A12 part
    for j in range(2, 4):
        for k in range(2, 4):
            V[idx_B(j, k), r] = 1  # B22 part
    for k in range(2, 4):
        for i in range(2):
            W[idx_W(k, i), r] = 1
    r += 1
    
    # Term 6: M6 = (A21 - A11)(B11 + B12)
    # A21 = A[2:4, 0:2], A11 = A[0:2, 0:2]
    # B11 = B[0:2, 0:2], B12 = B[0:2, 2:4]
    r = 5
    for i in range(2, 4):
        for j in range(2):
            U[idx_A(i, j), r] = 1  # A21 part
    for i in range(2):
        for j in range(2):
            U[idx_A(i, j), r] = -1  # A11 part
    for j in range(2):
        for k in range(2):
            V[idx_B(j, k), r] = 1  # B11 part
            V[idx_B(j, k+2), r] = 1  # B12 part
    for k in range(2):
        for i in range(2, 4):
            W[idx_W(k, i), r] = 1
    r += 1
    
    # Term 7: M7 = (A12 - A22)(B21 + B22)
    # A12 = A[0:2, 2:4], A22 = A[2:4, 2:4]
    # B21 = B[2:4, 0:2], B22 = B[2:4, 2:4]
    r = 6
    for i in range(2):
        for j in range(2, 4):
            U[idx_A(i, j), r] = 1  # A12 part
    for i in range(2, 4):
        for j in range(2, 4):
            U[idx_A(i, j), r] = -1  # A22 part
    for j in range(2, 4):
        for k in range(2):
            V[idx_B(j, k), r] = 1  # B21 part
            V[idx_B(j, k+2), r] = 1  # B22 part
    for k in range(2):
        for i in range(2):
            W[idx_W(k, i), r] = 1
    r += 1
    
    # Terms 8-14: Similar Strassen terms for the (1,2) block of result
    # Result block (1,2) = A11*B12 + A12*B22
    # Using Strassen-style:
    # M8 = (A11 + A12)(B22 - B12) ... let me use a different approach
    
    # Actually, let me use the full Laderman scheme which has 23 terms.
    # The remaining 16 terms handle the off-diagonal blocks.
    
    # For the 3x4 x 4x4 case, we only need the first 3 rows of A.
    # So we can potentially eliminate terms that only involve A[3,:].
    
    # Let me construct the full 23-term Laderman and then see if we can
    # project it down to 3x4 x 4x4.
    
    # Terms 8-14: For computing C12 = A11*B12 + A12*B22
    # Using: M8 = A11*B12, M9 = A12*B22, M10 = (A11+A12)*(B12+B22) - M8 - M9... 
    # This is getting complex. Let me use a known Laderman decomposition.
    
    # I'll use a simpler approach: use Strassen for each 2x2 block pair.
    # 4 block pairs, each with 7 Strassen terms = 28 terms.
    # But Laderman optimizes to 23.
    
    # For now, let me just use the 7-term Strassen for each of the 4 block positions
    # and see if we can reduce.
    
    # Actually, let me just return the 7-term Strassen based decomposition for 3x4x4
    # which we know works, and try to optimize from there.
    
    # Let me restart with a cleaner approach.
    return None

def _construct_strassen_3x4x4():
    """
    Construct a decomposition for 3x4 x 4x4 using Strassen-like techniques.
    
    Split A (3x4) into:
    - A_top: rows 0-1 (2x4)
    - A_bottom: row 2 (1x4)
    
    Split B (4x4) into 2x2 blocks of 2x2.
    
    For A_top (2x4) x B (4x4):
    Split B into left (4x2) and right (4x2).
    A_top x B_left: 2x2 x 2x2 block multiplication (with 2x2 blocks)
    A_top x B_right: similar
    
    For A_bottom (1x4) x B (4x4):
    Just 4 scalar multiplications per output element, or use some optimization.
    """
    n, m, p = 3, 4, 4
    rank = 44
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    
    def add_strassen_2x2x2(rA, cA, rB, cB):
        """Add 7 Strassen terms for 2x2 x 2x2 block multiplication."""
        nonlocal col
        # M1 = (X11+X22)(Y11+Y22)
        U[(rA+0)*m + (cA+0), col] = 1
        U[(rA+1)*m + (cA+1), col] = 1
        V[(rB+0)*p + (cB+0), col] = 1
        V[(rB+1)*p + (cB+1), col] = 1
        W[(cB+0)*n + (rA+0), col] = 1
        W[(cB+1)*n + (rA+1), col] = 1
        col += 1
        # M2 = (X21+X22)Y11
        U[(rA+1)*m + (cA+0), col] = 1
        U[(rA+1)*m + (cA+1), col] = 1
        V[(rB+0)*p + (cB+0), col] = 1
        W[(cB+0)*n + (rA+1), col] = 1
        W[(cB+1)*n + (rA+1), col] = -1
        col += 1
        # M3 = X11(Y12-Y22)
        U[(rA+0)*m + (cA+0), col] = 1
        V[(rB+0)*p + (cB+1), col] = 1
        V[(rB+1)*p + (cB+1), col] = -1
        W[(cB+1)*n + (rA+0), col] = 1
        W[(cB+1)*n + (rA+1), col] = 1
        col += 1
        # M4 = X22(Y21-Y11)
        U[(rA+1)*m + (cA+1), col] = 1
        V[(rB+1)*p + (cB+0), col] = 1
        V[(rB+0)*p + (cB+0), col] = -1
        W[(cB+0)*n + (rA+0), col] = 1
        W[(cB+0)*n + (rA+1), col] = 1
        col += 1
        # M5 = (X11+X12)Y22
        U[(rA+0)*m + (cA+0), col] = 1
        U[(rA+0)*m + (cA+1), col] = 1
        V[(rB+1)*p + (cB+1), col] = 1
        W[(cB+1)*n + (rA+0), col] = 1
        W[(cB+0)*n + (rA+0), col] = -1
        col += 1
        # M6 = (X21-X11)(Y11+Y12)
        U[(rA+1)*m + (cA+0), col] = 1
        U[(rA+0)*m + (cA+0), col] = -1
        V[(rB+0)*p + (cB+0), col] = 1
        V[(rB+0)*p + (cB+1), col] = 1
        W[(cB+1)*n + (rA+1), col] = 1
        col += 1
        # M7 = (X12-X22)(Y21+Y22)
        U[(rA+0)*m + (cA+1), col] = 1
        U[(rA+1)*m + (cA+1), col] = -1
        V[(rB+1)*p + (cB+0), col] = 1
        V[(rB+1)*p + (cB+1), col] = 1
        W[(cB+0)*n + (rA+0), col] = 1
        col += 1
    
    def add_schoolbook_1x2x2(rA, cA, rB, cB):
        """Add 4 schoolbook terms for 1x2 x 2x2 block multiplication."""
        nonlocal col
        for di in range(1):
            for dj in range(2):
                for dk in range(2):
                    U[(rA+di)*m + (cA+dj), col] = 1
                    V[(rB+dj)*p + (cB+dk), col] = 1
                    W[(cB+dk)*n + (rA+di), col] = 1
                    col += 1
    
    # Top 2 rows of A (2x4) times B (4x4)
    # Split into 2x2 blocks: A_top = [[A00, A01], [A02, A03]] where each is 1x2
    # B = [[B00, B01], [B02, B03]] where each is 2x2
    
    # Result top-left 2x2: A00*B00 + A01*B02 (but A00 is 1x2, B00 is 2x2)
    # This doesn't fit nicely into 2x2x2 Strassen.
    
    # Let me use a different partitioning.
    # A is 3x4. Split rows as [2, 1].
    # B is 4x4. Split rows as [2, 2], cols as [2, 2].
    
    # For the top 2 rows of A (2x4) times B (4x4):
    # Split A_top into left (2x2) and right (2x2)
    # Split B into top (4x2) and bottom (4x2)
    
    # A_top_left (2x2) * B_top (4x2): split B_top into top (2x2) and bottom (2x2)
    # = A_top_left * B_top_top (2x2x2) + A_top_left * B_top_bottom (2x2x2)
    # Each 2x2x2 uses 7 Strassen terms.
    
    # A_top_right (2x2) * B_bottom (4x2): similar
    
    # For the bottom 1 row of A (1x4) times B (4x4):
    # Split A_bottom into left (1x2) and right (1x2)
    # Split B into top (4x2) and bottom (4x2)
    # Each 1x2 x 2x2 uses 4 schoolbook terms.
    
    # Total: 4 * 7 + 4 * 4 = 28 + 16 = 44 terms.
    
    # A_top_left: rows 0-1, cols 0-1
    # A_top_right: rows 0-1, cols 2-3
    # A_bottom_left: row 2, cols 0-1
    # A_bottom_right: row 2, cols 2-3
    
    # B_top_top: rows 0-1, cols 0-1
    # B_top_bottom: rows 2-3, cols 0-1
    # B_bottom_top: rows 0-1, cols 2-3
    # B_bottom_bottom: rows 2-3, cols 2-3
    
    # Result:
    # C[0:2, 0:2] = A_top_left * B_top_top + A_top_right * B_top_bottom... 
    # Wait, this isn't right. Let me think more carefully.
    
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # For i in 0,1 and k in 0,1:
    # C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    #        = sum_{j=0}^{1} A[i,j] * B[j,k] + sum_{j=2}^{3} A[i,j] * B[j,k]
    #        = (A[i,0:2] dot B[0:2,k]) + (A[i,2:4] dot B[2:4,k])
    
    # So for the top-left 2x2 block of C:
    # C[0:2, 0:2] = A[0:2, 0:2] * B[0:2, 0:2] + A[0:2, 2:4] * B[2:4, 0:2]
    
    # First term: 2x2 x 2x2 -> 7 Strassen terms
    # Second term: 2x2 x 2x2 -> 7 Strassen terms
    
    # Similarly for other blocks.
    
    # Let me just use the original approach which was working.
    add_strassen_2x2x2(0, 0, 0, 0)  # A[0:2,0:2] * B[0:2,0:2]
    add_strassen_2x2x2(0, 2, 2, 0)  # A[0:2,2:4] * B[2:4,0:2]
    add_strassen_2x2x2(0, 0, 0, 2)  # A[0:2,0:2] * B[0:2,2:4]
    add_strassen_2x2x2(0, 2, 2, 2)  # A[0:2,2:4] * B[2:4,2:4]
    add_schoolbook_1x2x2(2, 0, 0, 0)  # A[2,0:2] * B[0:2,0:2]
    add_schoolbook_1x2x2(2, 2, 2, 0)  # A[2,2:4] * B[2:4,0:2]
    add_schoolbook_1x2x2(2, 0, 0, 2)  # A[2,0:2] * B[0:2,2:4]
    add_schoolbook_1x2x2(2, 2, 2, 2)  # A[2,2:4] * B[2:4,2:4]
    
    return U, V, W

def _als_decomposition_vectorized(T, n, m, p, rank, max_iter=100, tol=1e-8):
    """Vectorized ALS decomposition of tensor T into rank-r sum of outer products."""
    np.random.seed(42)
    U = np.random.randn(n*m, rank) * 0.5
    V = np.random.randn(m*p, rank) * 0.5
    W = np.random.randn(p*n, rank) * 0.5
    
    for it in range(max_iter):
        # Update U
        for r in range(rank):
            Vr = V[:, r]
            Wr = W[:, r]
            num = np.zeros(n*m)
            den = np.zeros(n*m)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        num[i*m+j] += T[i,j,k] * Vr[j*p+k] * Wr[k*n+i]
                        den[i*m+j] += Vr[j*p+k]**2 * Wr[k*n+i]**2
            mask = den > 1e-12
            U[mask, r] = num[mask] / den[mask]
        
        # Update V
        for r in range(rank):
            Ur = U[:, r]
            Wr = W[:, r]
            num = np.zeros(m*p)
            den = np.zeros(m*p)
            for j in range(m):
                for k in range(p):
                    for i in range(n):
                        num[j*p+k] += T[i,j,k] * Ur[i*m+j] * Wr[k*n+i]
                        den[j*p+k] += Ur[i*m+j]**2 * Wr[k*n+i]**2
            mask = den > 1e-12
            V[mask, r] = num[mask] / den[mask]
        
        # Update W
        for r in range(rank):
            Ur = U[:, r]
            Vr = V[:, r]
            num = np.zeros(p*n)
            den = np.zeros(p*n)
            for k in range(p):
                for i in range(n):
                    for j in range(m):
                        num[k*n+i] += T[i,j,k] * Ur[i*m+j] * Vr[j*p+k]
                        den[k*n+i] += Ur[i*m+j]**2 * Vr[j*p+k]**2
            mask = den > 1e-12
            W[mask, r] = num[mask] / den[mask]
        
        # Check convergence
        T_recon = np.zeros((n, m, p))
        for r in range(rank):
            u = U[:, r].reshape(n, m)
            v = V[:, r].reshape(m, p)
            w = W[:, r].reshape(p, n)
            T_recon += np.einsum('ij,jk,ki->ijk', u, v, w)
        if np.max(np.abs(T_recon - T)) < tol:
            break
    
    return U, V, W

def construct():
    deadline = time.time() + 110
    n, m, p = DIMENSIONS
    T = _build_target_tensor(n, m, p)
    
    # Start with the structured rank-44 decomposition
    U44, V44, W44 = _construct_strassen_3x4x4()
    best_U, best_V, best_W = U44, V44, W44
    best_rank = 44
    
    # Verify the base decomposition
    if not _verify_decomposition(U44, V44, W44, n, m, p, tol=1e-6):
        # Fallback to simple schoolbook
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
        best_U, best_V, best_W = U, V, W
        best_rank = rank
        if not _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
            return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Try to find lower-rank decompositions using ALS
    for r in [43, 42, 41, 40, 39, 38]:
        if time.time() > deadline - 30:
            break
        
        found = False
        for trial in range(3):
            if time.time() > deadline - 25:
                break
            
            np.random.seed(42 + trial)
            U, V, W = _als_decomposition_vectorized(T, n, m, p, r, max_iter=50, tol=1e-6)
            
            if _verify_decomposition(U, V, W, n, m, p, tol=1e-4):
                U_r = np.round(U * 2) / 2.0
                V_r = np.round(V * 2) / 2.0
                W_r = np.round(W * 2) / 2.0
                if _verify_decomposition(U_r, V_r, W_r, n, m, p, tol=1e-4):
                    best_U, best_V, best_W = U_r, V_r, W_r
                    best_rank = r
                    found = True
                    break
                else:
                    best_U, best_V, best_W = U, V, W
                    best_rank = r
                    found = True
                    break
        
        if found:
            break
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
