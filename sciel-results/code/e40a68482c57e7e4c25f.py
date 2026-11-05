import numpy as np
import time

DIMENSIONS = [3, 4, 4]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
    """Check if U,V,W exactly decompose the tensor (within tol)."""
    rank = U.shape[1]
    T = np.zeros((n, m, p), dtype=np.float64)
    for r in range(rank):
        u = U[:, r].reshape(n, m)
        v = V[:, r].reshape(m, p)
        w = W[:, r].reshape(p, n)
        T += np.einsum('ij,jk,ki->ijk', u, v, w)
    return np.allclose(T, 1.0, atol=tol, rtol=0)

def _construct_optimized_decomposition():
    """
    Construct a low-rank decomposition using optimized block structure.
    Uses Strassen for 2x2x2 blocks where beneficial.
    """
    n, m, p = 3, 4, 4
    
    # We'll use a block structure:
    # Split A (3x4) into top 2 rows and bottom 1 row
    # Split B (4x4) into left 2 cols and right 2 cols
    # This gives us blocks:
    #   Top-Left: 2x2 x 2x2 -> use Strassen (rank 7)
    #   Top-Right: 2x2 x 2x2 -> use Strassen (rank 7)  
    #   Bottom-Left: 1x2 x 2x2 -> schoolbook (rank 4)
    #   Bottom-Right: 1x2 x 2x2 -> schoolbook (rank 4)
    # Total: 7 + 7 + 4 + 4 = 22... but we need to account for the full 4x4 B
    
    # Actually, let me reconsider. B is 4x4, so when we split columns:
    # B = [B_left (4x2), B_right (4x2)]
    # A = [A_top (2x4), A_bottom (1x4)]
    # A_top = [A_top_left (2x2), A_top_right (2x2)]
    # A_bottom = [A_bot_left (1x2), A_bot_right (1x2)]
    
    # C_top_left (2x2) = A_top_left * B_left + A_top_right * B_right
    # This doesn't decompose cleanly into independent 2x2x2 blocks.
    
    # Let me use a different approach: split both A and B into 2x2 blocks
    # A (3x4) -> rows 0-1, cols 0-1 | rows 0-1, cols 2-3 | row 2, cols 0-1 | row 2, cols 2-3
    # B (4x4) -> rows 0-1, cols 0-1 | rows 0-1, cols 2-3 | rows 2-3, cols 0-1 | rows 2-3, cols 2-3
    
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # For C[0:2, 0:2] (top-left 2x2 of C):
    # C[0:2, 0:2] = A[0:2, 0:2]*B[0:2, 0:2] + A[0:2, 2:4]*B[2:4, 0:2]
    # This is sum of two 2x2x2 multiplications -> rank 7+7 = 14
    
    # For C[0:2, 2:4] (top-right 2x2 of C):
    # C[0:2, 2:4] = A[0:2, 0:2]*B[0:2, 2:4] + A[0:2, 2:4]*B[2:4, 2:4]
    # This is sum of two 2x2x2 multiplications -> rank 7+7 = 14
    
    # For C[2, 0:2] (bottom-left 1x2 of C):
    # C[2, 0:2] = A[2, 0:2]*B[0:2, 0:2] + A[2, 2:4]*B[2:4, 0:2]
    # This is sum of two 1x2x2 multiplications -> rank 4+4 = 8
    
    # For C[2, 2:4] (bottom-right 1x2 of C):
    # C[2, 2:4] = A[2, 0:2]*B[0:2, 2:4] + A[2, 2:4]*B[2:4, 2:4]
    # This is sum of two 1x2x2 multiplications -> rank 4+4 = 8
    
    # Total: 14 + 14 + 8 + 8 = 44
    
    # This matches the inspiration. Let me try to do better.
    
    # Alternative: Use a 3x3x3 core with BCDL rank 19
    # Zero-pad A to 3x4 (already is), B to 4x4 (already is)
    # The issue is that BCDL is for 3x3x3, but we have 3x4x4x4
    
    # Let me just implement the rank-44 solution reliably and verify it.
    
    rank = 44
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    def add_strassen(rA, cA, rB, cB, rC):
        """Add Strassen decomposition for A[rA:rA+2, cA:cA+2] * B[rB:rB+2, cB:cB+2] -> C[rC:rC+2, cB:cB+2]"""
        nonlocal col
        # Strassen 7 terms for 2x2x2
        # A sub: rows rA:rA+2, cols cA:cA+2
        # B sub: rows rB:rB+2, cols cB:cB+2  
        # C sub: rows rC:rC+2, cols cB:cB+2
        
        # Global indices:
        # U[i*m + j] for A[i,j]
        # V[j*p + k] for B[j,k]
        # W[k*n + i] for C[i,k]
        
        # Term 1: M1 = (A11+A22)(B11+B22) -> contributes to C11, C22
        U[(rA+0)*m + (cA+0), col] = 1
        U[(rA+1)*m + (cA+1), col] = 1
        V[(rB+0)*p + (cB+0), col] = 1
        V[(rB+1)*p + (cB+1), col] = 1
        W[(cB+0)*n + (rC+0), col] = 1
        W[(cB+1)*n + (rC+1), col] = 1
        col += 1
        
        # Term 2: M2 = (A21+A22)B11 -> contributes to C21, C22(-1)
        U[(rA+1)*m + (cA+0), col] = 1
        U[(rA+1)*m + (cA+1), col] = 1
        V[(rB+0)*p + (cB+0), col] = 1
        W[(cB+0)*n + (rC+1), col] = 1
        W[(cB+1)*n + (rC+1), col] = -1
        col += 1
        
        # Term 3: M3 = A11(B12-B22) -> contributes to C12, C22
        U[(rA+0)*m + (cA+0), col] = 1
        V[(rB+0)*p + (cB+1), col] = 1
        V[(rB+1)*p + (cB+1), col] = -1
        W[(cB+1)*n + (rC+0), col] = 1
        W[(cB+1)*n + (rC+1), col] = 1
        col += 1
        
        # Term 4: M4 = A22(B21-B11) -> contributes to C11, C21
        U[(rA+1)*m + (cA+1), col] = 1
        V[(rB+1)*p + (cB+0), col] = 1
        V[(rB+0)*p + (cB+0), col] = -1
        W[(cB+0)*n + (rC+0), col] = 1
        W[(cB+0)*n + (rC+1), col] = 1
        col += 1
        
        # Term 5: M5 = (A11+A12)B22 -> contributes to C12, C11(-1)
        U[(rA+0)*m + (cA+0), col] = 1
        U[(rA+0)*m + (cA+1), col] = 1
        V[(rB+1)*p + (cB+1), col] = 1
        W[(cB+1)*n + (rC+0), col] = 1
        W[(cB+0)*n + (rC+0), col] = -1
        col += 1
        
        # Term 6: M6 = (A21-A11)(B11+B12) -> contributes to C22
        U[(rA+1)*m + (cA+0), col] = 1
        U[(rA+0)*m + (cA+0), col] = -1
        V[(rB+0)*p + (cB+0), col] = 1
        V[(rB+0)*p + (cB+1), col] = 1
        W[(cB+1)*n + (rC+1), col] = 1
        col += 1
        
        # Term 7: M7 = (A12-A22)(B21+B22) -> contributes to C11
        U[(rA+0)*m + (cA+1), col] = 1
        U[(rA+1)*m + (cA+1), col] = -1
        V[(rB+1)*p + (cB+0), col] = 1
        V[(rB+1)*p + (cB+1), col] = 1
        W[(cB+0)*n + (rC+0), col] = 1
        col += 1

    def add_schoolbook_1x2x2(rA, cA, rB, cB, rC):
        """Add schoolbook for A[rA, cA:cA+2] * B[rB:rB+2, cB:cB+2] -> C[rC, cB:cB+2]"""
        nonlocal col
        # A is 1x2, B is 2x2, C is 1x2
        # C0 = A0*B00 + A1*B10
        # C1 = A0*B01 + A1*B11
        
        # Term 1: A0*B00 -> C0
        U[(rA)*m + (cA+0), col] = 1
        V[(rB+0)*p + (cB+0), col] = 1
        W[(cB+0)*n + (rC), col] = 1
        col += 1
        
        # Term 2: A1*B10 -> C0
        U[(rA)*m + (cA+1), col] = 1
        V[(rB+1)*p + (cB+0), col] = 1
        W[(cB+0)*n + (rC), col] = 1
        col += 1
        
        # Term 3: A0*B01 -> C1
        U[(rA)*m + (cA+0), col] = 1
        V[(rB+0)*p + (cB+1), col] = 1
        W[(cB+1)*n + (rC), col] = 1
        col += 1
        
        # Term 4: A1*B11 -> C1
        U[(rA)*m + (cA+1), col] = 1
        V[(rB+1)*p + (cB+1), col] = 1
        W[(cB+1)*n + (rC), col] = 1
        col += 1

    # C[0:2, 0:2] = A[0:2, 0:2]*B[0:2, 0:2] + A[0:2, 2:4]*B[2:4, 0:2]
    add_strassen(0, 0, 0, 0, 0)  # A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    add_strassen(0, 2, 2, 0, 0)  # A[0:2,2:4] * B[2:4,0:2] -> C[0:2,0:2]
    
    # C[0:2, 2:4] = A[0:2, 0:2]*B[0:2, 2:4] + A[0:2, 2:4]*B[2:4, 2:4]
    add_strassen(0, 0, 0, 2, 0)  # A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4]
    add_strassen(0, 2, 2, 2, 0)  # A[0:2,2:4] * B[2:4,2:4] -> C[0:2,2:4]
    
    # C[2, 0:2] = A[2, 0:2]*B[0:2, 0:2] + A[2, 2:4]*B[2:4, 0:2]
    add_schoolbook_1x2x2(2, 0, 0, 0, 2)  # A[2,0:2] * B[0:2,0:2] -> C[2,0:2]
    add_schoolbook_1x2x2(2, 2, 2, 0, 2)  # A[2,2:4] * B[2:4,0:2] -> C[2,0:2]
    
    # C[2, 2:4] = A[2, 0:2]*B[0:2, 2:4] + A[2, 2:4]*B[2:4, 2:4]
    add_schoolbook_1x2x2(2, 0, 0, 2, 2)  # A[2,0:2] * B[0:2,2:4] -> C[2,2:4]
    add_schoolbook_1x2x2(2, 2, 2, 2, 2)  # A[2,2:4] * B[2:4,2:4] -> C[2,2:4]
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

def construct():
    deadline = time.time() + 110
    
    # Try the optimized decomposition first
    result = _construct_optimized_decomposition()
    
    n, m, p = DIMENSIONS
    U, V, W = result["U"], result["V"], result["W"]
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    if _verify_decomposition(U_np, V_np, W_np, n, m, p, tol=1e-6):
        return result
    else:
        # Fallback to schoolbook
        rank = n * m * p
        U_fb = [[0] * rank for _ in range(n * m)]
        V_fb = [[0] * rank for _ in range(m * p)]
        W_fb = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][col] = 1
                    V_fb[j * p + k][col] = 1
                    W_fb[k * n + i][col] = 1
                    col += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
