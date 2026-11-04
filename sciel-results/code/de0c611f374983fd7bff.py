import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0

    # We aim for rank 28 using Strassen for 2x2 blocks + schoolbook for the rest
    # 2x3 * 3x5
    # Split into:
    #   A_left (2x2) * B_left (2x2) -> C_left (2x2)  [Strassen, rank 7]
    #   A_left (2x2) * B_mid  (2x2) -> C_mid  (2x2)  [Strassen, rank 7]
    #   A_left (2x2) * B_right(2x1) -> C_right(2x1)  [Schoolbook, rank 4]
    #   A_right(2x1) * B_left (2x2) -> C_left2(2x2)  [Schoolbook, rank 4]
    #   A_right(2x1) * B_mid  (2x2) -> C_mid2 (2x2)  [Schoolbook, rank 4]
    #   A_right(2x1) * B_right(2x1) -> C_right2(2x1) [Schoolbook, rank 2]
    # Total: 7+7+4+4+4+2 = 28

    rank = 28
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    def add_strassen(i1, i2, j1, j2, k1, k2):
        nonlocal col
        # M1 = (A00 + A11)(B00 + B11)
        U[i1*m+j1, col] = 1; U[i2*m+j2, col] = 1
        V[j1*p+k1, col] = 1; V[j2*p+k2, col] = 1
        # C00 += M1, C11 += M1
        W[k1*n+i1, col] = 1; W[k2*n+i2, col] = 1
        col += 1
        
        # M2 = (A10 + A11) B00
        U[i2*m+j1, col] = 1; U[i2*m+j2, col] = 1
        V[j1*p+k1, col] = 1
        # C10 += M2, C11 -= M2
        W[k1*n+i2, col] = 1; W[k2*n+i2, col] = -1
        col += 1
        
        # M3 = A00 (B01 - B11)
        U[i1*m+j1, col] = 1
        V[j1*p+k2, col] = 1; V[j2*p+k2, col] = -1
        # C01 += M3, C11 += M3
        W[k2*n+i1, col] = 1; W[k2*n+i2, col] = 1
        col += 1
        
        # M4 = A11 (B10 - B00)
        U[i2*m+j2, col] = 1
        V[j2*p+k1, col] = 1; V[j1*p+k1, col] = -1
        # C00 += M4, C10 += M4
        W[k1*n+i1, col] = 1; W[k1*n+i2, col] = 1
        col += 1
        
        # M5 = (A00 + A01) B11
        U[i1*m+j1, col] = 1; U[i1*m+j2, col] = 1
        V[j2*p+k2, col] = 1
        # C00 -= M5, C01 += M5
        W[k1*n+i1, col] = -1; W[k2*n+i1, col] = 1
        col += 1
        
        # M6 = (A10 - A00)(B00 + B01)
        U[i2*m+j1, col] = 1; U[i1*m+j1, col] = -1
        V[j1*p+k1, col] = 1; V[j1*p+k2, col] = 1
        # C11 += M6
        W[k2*n+i2, col] = 1
        col += 1
        
        # M7 = (A01 - A11)(B10 + B11)
        U[i1*m+j2, col] = 1; U[i2*m+j2, col] = -1
        V[j2*p+k1, col] = 1; V[j2*p+k2, col] = 1
        # C00 += M7
        W[k1*n+i1, col] = 1
        col += 1

    def add_schoolbook(i_list, j_list, k_list):
        nonlocal col
        for i in i_list:
            for j in j_list:
                for k in k_list:
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1

    # Block 1: A[:,0:2] * B[0:2,0:2] -> C[0:2,0:2] (Strassen, rank 7)
    add_strassen(0, 1, 0, 1, 0, 1)

    # Block 2: A[:,0:2] * B[0:2,2:4] -> C[0:2,2:4] (Strassen, rank 7)
    add_strassen(0, 1, 0, 1, 2, 3)

    # Block 3: A[:,0:2] * B[0:2,4:5] -> C[0:2,4:5] (Schoolbook, rank 4)
    add_schoolbook([0, 1], [0, 1], [4])

    # Block 4: A[:,2:3] * B[2:3,0:2] -> C[0:2,0:2] (Schoolbook, rank 4)
    add_schoolbook([0, 1], [2], [0, 1])

    # Block 5: A[:,2:3] * B[2:3,2:4] -> C[0:2,2:4] (Schoolbook, rank 4)
    add_schoolbook([0, 1], [2], [2, 3])

    # Block 6: A[:,2:3] * B[2:3,4:5] -> C[0:2,4:5] (Schoolbook, rank 2)
    add_schoolbook([0, 1], [2], [4])

    if col != rank:
        # Should not happen, but if it does, we have a bug.
        # We can't easily fix it on the fly without knowing what went wrong.
        # Let's just proceed with verification.
        pass

    # Verify exactness
    # We want sum_c U_c[i,j] V_c[j,k] W_c[k,i] = 1 for all i,j,k
    # U_c is (n, m), V_c is (m, p), W_c is (p, n)
    # Note: W in the tensor product is typically the output coefficient.
    # The tensor T_{ijk} = sum_c U_{ic} V_{jc} W_{kc} ? No.
    # Standard form: T_{ijk} = sum_r A_{ir} B_{jr} C_{kr}
    # Here:
    # U is (nm, rank) -> U_{(i,j), r}
    # V is (mp, rank) -> V_{(j,k), r}
    # W is (pn, rank) -> W_{(k,i), r}
    # So T_{ijk} = sum_r U_{(i,j), r} V_{(j,k), r} W_{(k,i), r}
    
    # Let's verify this.
    # For Strassen M1:
    # U_{(i1,j1), 1} = 1, U_{(i2,j2), 1} = 1
    # V_{(j1,k1), 1} = 1, V_{(j2,k2), 1} = 1
    # W_{(k1,i1), 1} = 1, W_{(k2,i2), 1} = 1
    
    # Term for i=i1, j=j1, k=k1:
    # r=1: U=1, V=1, W=1 -> 1
    # Other r: 0
    # So T_{i1,j1,k1} gets +1. Correct (A00 B00 is part of C00).
    
    # Term for i=i1, j=j1, k=k2:
    # r=1: U=1, V=0 (V_{j1,k2}=0), W=0 -> 0
    # Wait.
    # C00 = M1 + M4 - M5 + M7.
    # M1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11.
    # So A00B11 contributes to C00.
    # For i=i1, j=j1, k=k2:
    # We need T_{i1,j1,k2} to have contribution from M1?
    # M1 involves A_{i1,j1} and B_{j1,k2}.
    # In our factorization:
    # U_{(i1,j1), 1} = 1
    # V_{(j1,k2), 1} = 0 (V has 1 at j1,k1 and j2,k2)
    # So M1 does NOT contribute to T_{i1,j1,k2} directly in this rank-1 term?
    # Ah, the rank-1 term is U_c V_c W_c.
    # If V_{(j1,k2), 1} = 0, then the term is 0.
    # But M1 SHOULD contribute to C_{i1,k2}?
    # No, C_{i1,k2} is C01.
    # C01 = M3 + M5.
    # M1 does not contribute to C01.
    # So it is correct that M1 (factor 1) does not contribute to T_{i1,j1,k2}.
    
    # Let's check M3.
    # M3 = A00(B01 - B11).
    # U_{(i1,j1), 3} = 1
    # V_{(j1,k2), 3} = 1
    # V_{(j2,k2), 3} = -1
    # W_{(k2,i1), 3} = 1
    # W_{(k2,i2), 3} = 1
    # For T_{i1,j1,k2}:
    # r=3: U=1, V=1, W=1 -> 1.
    # So M3 contributes 1 to C_{i1,k2} via A_{i1,j1} B_{j1,k2}. Correct.
    
    # Verification code:
    tensor = np.zeros((n, m, p))
    for r in range(rank):
        u = U[:, r].reshape(n, m)
        v = V[:, r].reshape(m, p)
        w = W[:, r].reshape(p, n)
        # T_{ijk} += u[i,j] * v[j,k] * w[k,i]
        # We can do this with broadcasting or loops.
        # For small dimensions, loops are fine.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    tensor[i, j, k] += u[i, j] * v[j, k] * w[k, i]

    if not np.allclose(tensor, 1.0, atol=1e-9):
        # Fallback: full schoolbook rank 30
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
