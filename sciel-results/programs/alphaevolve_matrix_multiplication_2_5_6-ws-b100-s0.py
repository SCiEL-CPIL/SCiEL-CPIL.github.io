"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

def _verify(U, V, W, n, m, p):
    """Verify the tensor decomposition"""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    total = np.zeros((n, m, p))
    for r in range(rank):
        u_col = U[:, r]
        v_col = V[:, r]
        w_col = W[:, r]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total[i,j,k] += u_col[i*m+j] * v_col[j*p+k] * w_col[k*n+i]
    expected = np.ones((n, m, p))
    if not np.allclose(total, expected, atol=1e-6):
        return False, np.max(np.abs(total - expected))
    return True, 0.0

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # Fallback: Rank 54 solution
    def _build_rank54():
        rank = 54
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        j_pairs = [(0, 1), (2, 3)]
        k_pairs = [(0, 1), (2, 3), (4, 5)]
        
        for j0, j1 in j_pairs:
            for k0, k1 in k_pairs:
                # P1
                U[0*m + j0][col] += 1
                U[1*m + j1][col] += 1
                V[j0*p + k0][col] += 1
                V[j1*p + k1][col] += 1
                W[k0*n + 0][col] += 1
                W[k1*n + 1][col] += 1
                col += 1
                # P2
                U[1*m + j0][col] += 1
                U[1*m + j1][col] += 1
                V[j0*p + k0][col] += 1
                W[k0*n + 1][col] += 1
                W[k1*n + 1][col] -= 1
                col += 1
                # P3
                U[0*m + j0][col] += 1
                V[j0*p + k1][col] += 1
                V[j1*p + k1][col] -= 1
                W[k1*n + 0][col] += 1
                W[k1*n + 1][col] += 1
                col += 1
                # P4
                U[1*m + j1][col] += 1
                V[j1*p + k0][col] += 1
                V[j0*p + k0][col] -= 1
                W[k0*n + 0][col] += 1
                W[k0*n + 1][col] += 1
                col += 1
                # P5
                U[0*m + j0][col] += 1
                U[0*m + j1][col] += 1
                V[j1*p + k1][col] += 1
                W[k0*n + 0][col] -= 1
                W[k1*n + 0][col] += 1
                col += 1
                # P6
                U[1*m + j0][col] += 1
                U[0*m + j0][col] -= 1
                V[j0*p + k0][col] += 1
                V[j0*p + k1][col] += 1
                W[k1*n + 1][col] += 1
                col += 1
                # P7
                U[0*m + j1][col] += 1
                U[1*m + j1][col] -= 1
                V[j1*p + k0][col] += 1
                V[j1*p + k1][col] += 1
                W[k0*n + 0][col] += 1
                col += 1
        
        for k in range(p):
            U[0*m + 4][col] += 1
            V[4*p + k][col] += 1
            W[k*n + 0][col] += 1
            col += 1
            U[1*m + 4][col] += 1
            V[4*p + k][col] += 1
            W[k*n + 1][col] += 1
            col += 1
        return {"U": U, "V": V, "W": W}

    # Try Rank 44 solution
    def _build_rank44():
        rank = 44
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        
        # Helper to add a 2x2x3 block (rank 8)
        def add_2x2x3(j_offset, k_offset):
            nonlocal col
            # 8 terms for 2x2x3
            # Using known decomposition
            # P1: A00 B00 -> C00
            U[0*m + j_offset + 0][col] += 1
            V[(j_offset + 0)*p + k_offset + 0][col] += 1
            W[(k_offset + 0)*n + 0][col] += 1
            col += 1
            # P2: A11 B11 -> C11
            U[1*m + j_offset + 1][col] += 1
            V[(j_offset + 1)*p + k_offset + 1][col] += 1
            W[(k_offset + 1)*n + 1][col] += 1
            col += 1
            # P3: A00 B01 -> C01
            U[0*m + j_offset + 0][col] += 1
            V[(j_offset + 0)*p + k_offset + 1][col] += 1
            W[(k_offset + 1)*n + 0][col] += 1
            col += 1
            # P4: A11 B10 -> C10
            U[1*m + j_offset + 1][col] += 1
            V[(j_offset + 1)*p + k_offset + 0][col] += 1
            W[(k_offset + 0)*n + 1][col] += 1
            col += 1
            # P5: A01 B10 -> C00
            U[0*m + j_offset + 1][col] += 1
            V[(j_offset + 1)*p + k_offset + 0][col] += 1
            W[(k_offset + 0)*n + 0][col] += 1
            col += 1
            # P6: A10 B01 -> C11
            U[1*m + j_offset + 0][col] += 1
            V[(j_offset + 0)*p + k_offset + 1][col] += 1
            W[(k_offset + 1)*n + 1][col] += 1
            col += 1
            # P7: (A00+A11)(B00+B11) - P1 - P2 -> handles A00B11 + A11B00
            # This is getting messy. I'll use a simpler fallback if this fails.
            # For now, just add 2 more terms for k=2
            # P7: A00 B02 -> C02
            U[0*m + j_offset + 0][col] += 1
            V[(j_offset + 0)*p + k_offset + 2][col] += 1
            W[(k_offset + 2)*n + 0][col] += 1
            col += 1
            # P8: A11 B12 -> C12
            U[1*m + j_offset + 1][col] += 1
            V[(j_offset + 1)*p + k_offset + 2][col] += 1
            W[(k_offset + 2)*n + 1][col] += 1
            col += 1
            # This is rank 8 but incomplete (missing cross terms for k=2).
            # It will fail verification.
            pass

        # Since the 44 solution is complex to construct correctly without a known explicit formula,
        # and the 54 solution is verified, I will return the 54 solution.
        return None

    baseline = _build_rank54()
    valid, diff = _verify(baseline["U"], baseline["V"], baseline["W"], n, m, p)
    
    return baseline
# EVOLVE-BLOCK-END
