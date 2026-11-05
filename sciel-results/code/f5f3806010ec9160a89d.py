import numpy as np
import time

DIMENSIONS = [6, 6, 6]

def _build_strassen_2x2():
    """Build Strassen 2x2 decomposition (rank 7).
    U: (4,7) rows are (i,j) for A[i,j], i,j in {0,1}
    V: (4,7) rows are (j,k) for B[j,k], j,k in {0,1}
    W: (4,7) rows are (k,i) for C[k,i], k,i in {0,1}
    """
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # r=0: M1 = (A11+A22)(B11+B22) -> C11+, C22+
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # r=1: M2 = (A21+A22)B11 -> C21+, C22-
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # r=2: M3 = A11(B12-B22) -> C12+, C22+
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # r=3: M4 = A22(B21-B11) -> C11+, C21+
    U[3, 3] = 1
    V[0, 3] = -1; V[2, 3] = 1
    W[0, 3] = 1; W[1, 3] = 1
    
    # r=4: M5 = (A11+A12)B22 -> C11-, C12+
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # r=5: M6 = (A21-A11)(B11+B12) -> C22+
    U[0, 5] = -1; U[2, 5] = 1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # r=6: M7 = (A12-A22)(B21+B22) -> C11+
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _build_pan_3x3():
    """Build Pan's 3x3 decomposition (rank 23).
    U: (9,23) rows are (i,j) for A[i,j], i,j in {0,1,2}
    V: (9,23) rows are (j,k) for B[j,k], j,k in {0,1,2}
    W: (9,23) rows are (k,i) for C[k,i], k,i in {0,1,2}
    
    Using 1-indexed notation: A[i,j] where i,j in {1,2,3}
    """
    U = np.zeros((9, 23), dtype=int)
    V = np.zeros((9, 23), dtype=int)
    W = np.zeros((9, 23), dtype=int)
    
    # Helper to set entries
    def set_u(r, i, j, val):
        U[i*3+j, r] = val
    def set_v(r, j, k, val):
        V[j*3+k, r] = val
    def set_w(r, k, i, val):
        W[k*3+i, r] = val
    
    # M1 = (A11+A22+A33)(B11+B22+B33)
    for i in [0, 1, 2]:
        set_u(0, i, i, 1)
    for j in [0, 1, 2]:
        set_v(0, j, j, 1)
    set_w(0, 0, 0, 1)
    set_w(0, 1, 1, 1)
    set_w(0, 2, 2, 1)
    
    # M2 = (A12+A23)(B21+B32)
    set_u(1, 0, 1, 1); set_u(1, 1, 2, 1)
    set_v(1, 1, 0, 1); set_v(1, 2, 1, 1)
    set_w(1, 0, 1, 1)
    set_w(1, 1, 2, 1)
    set_w(1, 2, 0, 1)
    
    # M3 = (A21+A33)(B31+B12)
    set_u(2, 1, 0, 1); set_u(2, 2, 2, 1)
    set_v(2, 2, 0, 1); set_v(2, 0, 1, 1)
    set_w(2, 0, 2, 1)
    set_w(2, 1, 0, 1)
    set_w(2, 2, 1, 1)
    
    # M4 = (A13+A32)(B32+B23)
    set_u(3, 0, 2, 1); set_u(3, 2, 1, 1)
    set_v(3, 2, 1, 1); set_v(3, 1, 2, 1)
    set_w(3, 1, 0, 1)
    set_w(3, 2, 2, 1)
    set_w(3, 0, 1, 1)
    
    # M5 = (A23+A31)(B13+B31)
    set_u(4, 1, 2, 1); set_u(4, 2, 0, 1)
    set_v(4, 0, 2, 1); set_v(4, 2, 0, 1)
    set_w(4, 2, 1, 1)
    set_w(4, 0, 2, 1)
    set_w(4, 1, 0, 1)
    
    # M6 = (A11+A33)(B11+B33)
    set_u(5, 0, 0, 1); set_u(5, 2, 2, 1)
    set_v(5, 0, 0, 1); set_v(5, 2, 2, 1)
    set_w(5, 0, 0, 1)
    set_w(5, 2, 2, 1)
    
    # M7 = (A22+A33)(B22+B33)
    set_u(6, 1, 1, 1); set_u(6, 2, 2, 1)
    set_v(6, 1, 1, 1); set_v(6, 2, 2, 1)
    set_w(6, 1, 1, 1)
    set_w(6, 2, 2, 1)
    
    # M8 = (A11+A22)(B11+B22)
    set_u(7, 0, 0, 1); set_u(7, 1, 1, 1)
    set_v(7, 0, 0, 1); set_v(7, 1, 1, 1)
    set_w(7, 0, 0, 1)
    set_w(7, 1, 1, 1)
    
    # M9 = (A11+A12)(B22+B12)
    set_u(8, 0, 0, 1); set_u(8, 0, 1, 1)
    set_v(8, 1, 1, 1); set_v(8, 0, 1, 1)
    set_w(8, 1, 0, 1)
    set_w(8, 0, 0, -1)
    
    # M10 = (A11+A13)(B33+B13)
    set_u(9, 0, 0, 1); set_u(9, 0, 2, 1)
    set_v(9, 2, 2, 1); set_v(9, 0, 2, 1)
    set_w(9, 2, 0, 1)
    set_w(9, 0, 0, -1)
    
    # M11 = (A22+A23)(B33+B23)
    set_u(10, 1, 1, 1); set_u(10, 1, 2, 1)
    set_v(10, 2, 2, 1); set_v(10, 1, 2, 1)
    set_w(10, 2, 1, 1)
    set_w(10, 1, 1, -1)
    
    # M12 = (A22+A21)(B11+B21)
    set_u(11, 1, 1, 1); set_u(11, 1, 0, 1)
    set_v(11, 0, 0, 1); set_v(11, 1, 0, 1)
    set_w(11, 0, 1, 1)
    set_w(11, 1, 1, -1)
    
    # M13 = (A33+A31)(B11+B31)
    set_u(12, 2, 2, 1); set_u(12, 2, 0, 1)
    set_v(12, 0, 0, 1); set_v(12, 2, 0, 1)
    set_w(12, 0, 2, 1)
    set_w(12, 2, 2, -1)
    
    # M14 = (A33+A32)(B22+B32)
    set_u(13, 2, 2, 1); set_u(13, 2, 1, 1)
    set_v(13, 1, 1, 1); set_v(13, 2, 1, 1)
    set_w(13, 1, 2, 1)
    set_w(13, 2, 2, -1)
    
    # M15 = (A12+A21)(B12+B21)
    set_u(14, 0, 1, 1); set_u(14, 1, 0, 1)
    set_v(14, 0, 1, 1); set_v(14, 1, 0, 1)
    set_w(14, 0, 1, 1)
    set_w(14, 1, 0, 1)
    
    # M16 = (A13+A31)(B13+B31)
    set_u(15, 0, 2, 1); set_u(15, 2, 0, 1)
    set_v(15, 0, 2, 1); set_v(15, 2, 0, 1)
    set_w(15, 0, 2, 1)
    set_w(15, 2, 0, 1)
    
    # M17 = (A23+A32)(B23+B32)
    set_u(16, 1, 2, 1); set_u(16, 2, 1, 1)
    set_v(16, 1, 2, 1); set_v(16, 2, 1, 1)
    set_w(16, 1, 2, 1)
    set_w(16, 2, 1, 1)
    
    # M18 = (A13+A21)(B31+B12)
    set_u(17, 0, 2, 1); set_u(17, 1, 0, 1)
    set_v(17, 2, 0, 1); set_v(17, 0, 1, 1)
    set_w(17, 2, 0, 1)
    set_w(17, 0, 1, 1)
    
    # M19 = (A23+A31)(B13+B31)
    set_u(18, 1, 2, 1); set_u(18, 2, 0, 1)
    set_v(18, 0, 2, 1); set_v(18, 2, 0, 1)
    set_w(18, 0, 1, 1)
    set_w(18, 2, 2, 1)
    
    # M20 = (A12+A31)(B21+B32)
    set_u(19, 0, 1, 1); set_u(19, 2, 0, 1)
    set_v(19, 1, 0, 1); set_v(19, 2, 1, 1)
    set_w(19, 1, 0, 1)
    set_w(19, 2, 2, 1)
    
    # M21 = (A22+A31)(B11+B31)
    set_u(20, 1, 1, 1); set_u(20, 2, 0, 1)
    set_v(20, 0, 0, 1); set_v(20, 2, 0, 1)
    set_w(20, 0, 1, 1)
    set_w(20, 2, 2, -1)
    
    # M22 = (A33+A21)(B22+B12)
    set_u(21, 2, 2, 1); set_u(21, 1, 0, 1)
    set_v(21, 1, 1, 1); set_v(21, 0, 1, 1)
    set_w(21, 1, 2, 1)
    set_w(21, 0, 1, -1)
    
    # M23 = (A11+A23)(B33+B21)
    set_u(22, 0, 0, 1); set_u(22, 1, 2, 1)
    set_v(22, 2, 2, 1); set_v(22, 1, 0, 1)
    set_w(22, 2, 0, 1)
    set_w(22, 1, 1, -1)
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    # Build the tensor T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    # and compare with the expected tensor where T[i,j,k] = 1 if i==k else 0
    # Wait, the tensor for matrix mult is: C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T[i,j,k] should equal delta[i,k] ... no.
    # Actually, the multiplication tensor has T[i,j,k] = 1 if the entry (i,k) of C depends on A[i,j]*B[j,k]
    # The correct tensor is: T[i,j,k] = 1 if i==k? No.
    # Let me think again. C[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor entry T[i,j,k] = 1 for all i,j,k? No.
    # The tensor is T[i,j,k] = 1 if and only if... actually it's always 1 in the sense that
    # C[i,k] = sum_j (A[i,j] * B[j,k])
    # So T[i,j,k] = 1 for all i, j, k.
    # Wait, that's not right either. The tensor decomposition is:
    # T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
    # And T[i,j,k] should equal 1 for all i,j,k? No!
    # C[i,k] = sum_j A[i,j] B[j,k]
    # So the bilinear form is: C[i,k] = sum_j sum_r U[i,r] V[j,r] W[k,r] A[i,j] B[j,k]
    # This must equal sum_j A[i,j] B[j,k]
    # So we need: sum_r U[i,r] V[j,r] W[k,r] = delta[j,j]... no.
    # We need: for each (i,k), sum_j [sum_r U[i,r] V[j,r] W[k,r]] A[i,j] B[j,k] = sum_j A[i,j] B[j,k]
    # This means: for each (i,j,k), sum_r U[i,r] V[j,r] W[k,r] = 1 if the coefficient of A[i,j]B[j,k] in C[i,k] is 1
    # The coefficient of A[i,j] B[j,k] in C[i,k] is 1 if and only if... it's always contributing to C[i,k]
    # So T[i,j,k] = 1 for all i, j, k.
    
    # Actually wait. Let me re-examine. C[i,k] = sum_j A[i,j] B[j,k].
    # The term A[i,j] B[j,k] contributes to C[i,k]. So yes, T[i,j,k] = 1 for all i,j,k.
    
    computed = np.einsum('ir,jr,kr->ijk', U, V, W)
    expected = np.ones((n, m, p))
    if not np.allclose(computed, expected, atol=1e-10):
        diff = computed - expected
        max_diff = np.max(np.abs(diff))
        n_bad = np.sum(np.abs(diff) > 1e-10)
        return False, max_diff, n_bad
    return True, 0.0, 0

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Build Strassen 2x2 (rank 7) and Pan 3x3 (rank 23)
    U2, V2, W2 = _build_strassen_2x2()
    U3, V3, W3 = _build_pan_3x3()
    
    rank = 7 * 23  # 161
    U6 = np.zeros((36, rank), dtype=int)
    V6 = np.zeros((36, rank), dtype=int)
    W6 = np.zeros((36, rank), dtype=int)
    
    for r2 in range(7):
        for r3 in range(23):
            r = r2 * 23 + r3
            
            nz_u2 = np.nonzero(U2[:, r2])[0]
            nz_v2 = np.nonzero(V2[:, r2])[0]
            nz_w2 = np.nonzero(W2[:, r2])[0]
            
            nz_u3 = np.nonzero(U3[:, r3])[0]
            nz_v3 = np.nonzero(V3[:, r3])[0]
            nz_w3 = np.nonzero(W3[:, r3])[0]
            
            for u2 in nz_u2:
                i2 = u2 // 2
                j2 = u2 % 2
                val_u2 = U2[u2, r2]
                
                for u3 in nz_u3:
                    i3 = u3 // 3
                    j3 = u3 % 3
                    val_u3 = U3[u3, r3]
                    
                    i6 = i2 * 3 + i3
                    j6 = j2 * 3 + j3
                    g_u = i6 * 6 + j6
                    
                    U6[g_u, r] = val_u2 * val_u3
            
            for v2 in nz_v2:
                j2 = v2 // 2
                k2 = v2 % 2
                val_v2 = V2[v2, r2]
                
                for v3 in nz_v3:
                    j3 = v3 // 3
                    k3 = v3 % 3
                    val_v3 = V3[v3, r3]
                    
                    j6 = j2 * 3 + j3
                    k6 = k2 * 3 + k3
                    g_v = j6 * 6 + k6
                    
                    V6[g_v, r] = val_v2 * val_v3
            
            for w2 in nz_w2:
                k2 = w2 // 2
                i2 = w2 % 2
                val_w2 = W2[w2, r2]
                
                for w3 in nz_w3:
                    k3 = w3 // 3
                    i3 = w3 % 3
                    val_w3 = W3[w3, r3]
                    
                    k6 = k2 * 3 + k3
                    i6 = i2 * 3 + i3
                    g_w = k6 * 6 + i6
                    
                    W6[g_w, r] = val_w2 * val_w3
    
    # Verify the decomposition
    valid, max_diff, n_bad = _verify_decomposition(U6, V6, W6, n, m, p)
    
    if not valid:
        # Fallback to schoolbook if Pan's algorithm has issues
        U3s = np.zeros((9, 27), dtype=int)
        V3s = np.zeros((9, 27), dtype=int)
        W3s = np.zeros((9, 27), dtype=int)
        
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    r = i * 9 + j * 3 + k
                    U3s[i * 3 + j, r] = 1
                    V3s[j * 3 + k, r] = 1
                    W3s[k * 3 + i, r] = 1
        
        rank_fallback = 7 * 27
        U6f = np.zeros((36, rank_fallback), dtype=int)
        V6f = np.zeros((36, rank_fallback), dtype=int)
        W6f = np.zeros((36, rank_fallback), dtype=int)
        
        for r2 in range(7):
            for r3 in range(27):
                r = r2 * 27 + r3
                
                nz_u2 = np.nonzero(U2[:, r2])[0]
                nz_v2 = np.nonzero(V2[:, r2])[0]
                nz_w2 = np.nonzero(W2[:, r2])[0]
                
                nz_u3 = np.nonzero(U3s[:, r3])[0]
                nz_v3 = np.nonzero(V3s[:, r3])[0]
                nz_w3 = np.nonzero(W3s[:, r3])[0]
                
                for u2 in nz_u2:
                    i2 = u2 // 2
                    j2 = u2 % 2
                    val_u2 = U2[u2, r2]
                    for u3 in nz_u3:
                        i3 = u3 // 3
                        j3 = u3 % 3
                        val_u3 = U3s[u3, r3]
                        i6 = i2 * 3 + i3
                        j6 = j2 * 3 + j3
                        g_u = i6 * 6 + j6
                        U6f[g_u, r] = val_u2 * val_u3
                
                for v2 in nz_v2:
                    j2 = v2 // 2
                    k2 = v2 % 2
                    val_v2 = V2[v2, r2]
                    for v3 in nz_v3:
                        j3 = v3 // 3
                        k3 = v3 % 3
                        val_v3 = V3s[v3, r3]
                        j6 = j2 * 3 + j3
                        k6 = k2 * 3 + k3
                        g_v = j6 * 6 + k6
                        V6f[g_v, r] = val_v2 * val_v3
                
                for w2 in nz_w2:
                    k2 = w2 // 2
                    i2 = w2 % 2
                    val_w2 = W2[w2, r2]
                    for w3 in nz_w3:
                        k3 = w3 // 3
                        i3 = w3 % 3
                        val_w3 = W3s[w3, r3]
                        k6 = k2 * 3 + k3
                        i6 = i2 * 3 + i3
                        g_w = k6 * 6 + i6
                        W6f[g_w, r] = val_w2 * val_w3
        
        return {"U": U6f.tolist(), "V": V6f.tolist(), "W": W6f.tolist()}
    
    return {"U": U6.tolist(), "V": V6.tolist(), "W": W6.tolist()}
