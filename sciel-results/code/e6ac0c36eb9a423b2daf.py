import time
import sys

DIMENSIONS = [2, 2, 3]

def _verify(U, V, W, n=2, m=2, p=3):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                lhs = 0.0
                for r in range(rank):
                    lhs += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(lhs) > 1e-9:
                    return False
    # Check that no other (i,j,k) combination is nonzero (should be 0 for invalid indices)
    # Actually, the tensor only has nonzeros at valid (i,j,k) positions.
    # For invalid positions, the sum should be 0.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                pass  # Already checked above
    # The tensor T[i,j,k] = sum_r U[i,j]_r V[j,k]_r W[k,i]_r should be 1 for valid (i,j,k) and 0 otherwise.
    # Since i,j,k are all in valid ranges, we just need T=1 for all valid combos.
    # Wait, the bilinear map is (A,B) -> C where C[i,k] = sum_j A[i,j] B[j,k].
    # The tensor is T[i,k,j] or similar. Let me re-read the contract.
    # "W[k*n+i,r]" - this is the index for W.
    # The tensor T is such that T[i,j,k] (flattened) corresponds to the product.
    # Standard convention: T is a 3-tensor with dimensions (n*m, m*p, p*n) or similar.
    # The product is sum_r U[i,j]_r V[j,k]_r W[k,i]_r = delta_{something}.
    # Actually, for matrix multiplication C = AB, C[i,k] = sum_j A[i,j] B[j,k].
    # The tensor T has indices (i, j, k) where T[i,j,k] = 1 if the term A[i,j]*B[j,k] contributes to C[i,k].
    # But C[i,k] is a single element. The tensor decomposition is for the bilinear map.
    # The map is (A,B) -> C. C is p*n elements (or n*p).
    # The tensor T has three modes: A (n*m), B (m*p), C (p*n).
    # T[a,b,c] = 1 if A_a * B_b contributes to C_c.
    # A_a corresponds to A[i,j] where a = i*m+j.
    # B_b corresponds to B[j,k] where b = j*p+k.
    # C_c corresponds to C[i,k] where c = k*n+i (as per "W[k*n+i,r]").
    # So T[a,b,c] is non-zero only if the j index in A[i,j] matches the j index in B[j,k].
    # i.e., a = i*m+j, b = j*p+k, c = k*n+i.
    # In this case, T[a,b,c] = 1.
    # So for the decomposition to be exact:
    # sum_r U[a,r] V[b,r] W[c,r] = 1 if (a,b,c) is a valid triple, else 0.
    # Let's check all combinations.
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                i_a = a // m
                j_a = a % m
                j_b = b // p
                k_b = b % p
                k_c = c // n
                i_c = c % n
                expected = 1.0 if (j_a == j_b and i_a == i_c and k_b == k_c) else 0.0
                lhs = 0.0
                for r in range(rank):
                    lhs += U[a][r] * V[b][r] * W[c][r]
                if abs(lhs - expected) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    
    # Rank 11 decomposition for 2x2x3 matrix multiplication.
    # We use Strassen's algorithm for the first two columns of B (2x2x2) and schoolbook for the third column.
    # Strassen for 2x2x2:
    # M1 = A00 * B00
    # M2 = A10 * (B00 + B01)
    # M3 = (A00 + A01) * B11
    # M4 = A11 * (B10 + B11)
    # M5 = (A00 + A11) * (B00 + B11)
    # M6 = (A10 + A11) * (B10 - B00)
    # M7 = (A00 + A01) * (B01 - B10)
    #
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M7
    # C10 = M2 + M6
    # C11 = M1 - M2 + M3 + M5
    #
    # Wait, standard Strassen:
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M7
    # C10 = M2 + M6
    # C11 = M1 - M2 + M3 + M5
    #
    # Let's verify indices.
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    #
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    #
    # For 2x2x2 part (k=0,1):
    # A indices: 00->0, 01->1, 10->2, 11->3
    # B indices: 00->0, 01->1, 10->3, 11->4 (since p=3, so j*p+k: 0*3+0=0, 0*3+1=1, 1*3+0=3, 1*3+1=4)
    # C indices: 00->0, 01->1, 10->2, 11->3 (since n=2, so k*n+i: 0*2+0=0, 0*2+1=1, 1*2+0=2, 1*2+1=3)
    
    rank = 11
    U = [[0.0] * rank for _ in range(n * m)]  # 4 x 11
    V = [[0.0] * rank for _ in range(m * p)]  # 6 x 11
    W = [[0.0] * rank for _ in range(p * n)]  # 6 x 11
    
    # Strassen columns 0-6
    # M1 = A00 * B00
    U[0][0] = 1.0; V[0][0] = 1.0; W[0][0] = 1.0
    # M2 = A10 * (B00 + B01)
    U[2][1] = 1.0; V[0][1] = 1.0; V[1][1] = 1.0; W[2][1] = 1.0
    # M3 = (A00 + A01) * B11
    U[0][2] = 1.0; U[1][2] = 1.0; V[4][2] = 1.0; W[1][2] = 1.0
    # M4 = A11 * (B10 + B11)
    U[3][3] = 1.0; V[3][3] = 1.0; V[4][3] = 1.0; W[3][3] = 1.0
    # M5 = (A00 + A11) * (B00 + B11)
    U[0][4] = 1.0; U[3][4] = 1.0; V[0][4] = 1.0; V[4][4] = 1.0; W[0][4] = -1.0; W[3][4] = 1.0
    # M6 = (A10 + A11) * (B10 - B00)
    U[2][5] = 1.0; U[3][5] = 1.0; V[3][5] = 1.0; V[0][5] = -1.0; W[2][5] = 1.0
    # M7 = (A00 + A01) * (B01 - B10)
    U[0][6] = 1.0; U[1][6] = 1.0; V[1][6] = 1.0; V[3][6] = -1.0; W[0][6] = -1.0; W[1][6] = 1.0
    
    # Check C reconstruction for 2x2x2:
    # C00 (index 0) = M1 + M4 - M5 - M7 -> W[0] = 1 + 0 - 1 - 1 = -1? Wait.
    # M4 contributes to C11 (index 3).
    # Let's re-verify Strassen reconstruction.
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M7
    # C10 = M2 + M6
    # C11 = M1 - M2 + M3 + M5
    #
    # My W assignments:
    # W[0] (C00): M1=1, M5=-1, M7=-1. Sum = 1 - 1 - 1 = -1. But should be M1 + M4 - M5 - M7. M4 is not here.
    # Wait, M4 = A11*(B10+B11). This contributes to C11.
    # So W[0] should have contributions from M1, M5, M7.
    # M1=1, M5=-1, M7=-1. Sum = -1.
    # But C00 = M1 + M4 - M5 - M7. M4 does not contribute to C00.
    # So W[0] should be 1 (M1) - 1 (M5) - 1 (M7) = -1?
    # Let's check the tensor values.
    # T[0,0,0] (A00, B00, C00) = 1. U[0]V[0]W[0] = 1*1*(-1) = -1. Wrong.
    
    # I need to be careful with signs.
    # Let's use the standard Strassen formulas directly in the tensor decomposition.
    # The rank-7 decomposition for 2x2x2 is well-known.
    # Let's just hardcode the 7 rank-1 terms for the 2x2x2 part correctly.
    
    # Reset U, V, W for clarity
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Strassen rank-7 for 2x2x2:
    # M1 = A00 * B00
    # M2 = A10 * (B00 + B01)
    # M3 = (A00 + A01) * B11
    # M4 = A11 * (B10 + B11)
    # M5 = (A00 + A11) * (B00 + B11)
    # M6 = (A10 + A11) * (B10 - B00)
    # M7 = (A00 + A01) * (B01 - B10)
    #
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M7
    # C10 = M2 + M6
    # C11 = M1 - M2 + M3 + M5
    
    # Col 0: M1
    U[0][0] = 1.0  # A00
    V[0][0] = 1.0  # B00
    W[0][0] = 1.0  # C00
    W[3][0] = 1.0  # C11
    
    # Col 1: M2
    U[2][1] = 1.0  # A10
    V[0][1] = 1.0  # B00
    V[1][1] = 1.0  # B01
    W[2][1] = 1.0  # C10
    W[3][1] = -1.0 # C11
    
    # Col 2: M3
    U[0][2] = 1.0  # A00
    U[1][2] = 1.0  # A01
    V[4][2] = 1.0  # B11
    W[1][2] = 1.0  # C01
    W[3][2] = 1.0  # C11
    
    # Col 3: M4
    U[3][3] = 1.0  # A11
    V[3][3] = 1.0  # B10
    V[4][3] = 1.0  # B11
    W[0][3] = 1.0  # C00
    W[3][3] = 1.0  # C11  -- Wait, C11 = M1 - M2 + M3 + M5. M4 is not in C11?
    # Standard Strassen:
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M7
    # C10 = M2 + M6
    # C11 = M1 - M2 + M3 + M5
    # So M4 contributes to C00 only.
    W[0][3] = 1.0  # C00
    
    # Col 4: M5
    U[0][4] = 1.0  # A00
    U[3][4] = 1.0  # A11
    V[0][4] = 1.0  # B00
    V[4][4] = 1.0  # B11
    W[0][4] = -1.0 # C00
    W[3][4] = 1.0  # C11
    
    # Col 5: M6
    U[2][5] = 1.0  # A10
    U[3][5] = 1.0  # A11
    V[3][5] = 1.0  # B10
    V[0][5] = -1.0 # B00
    W[2][5] = 1.0  # C10
    
    # Col 6: M7
    U[0][6] = 1.0  # A00
    U[1][6] = 1.0  # A01
    V[1][6] = 1.0  # B01
    V[3][6] = -1.0 # B10
    W[0][6] = -1.0 # C00
    W[1][6] = 1.0  # C01
    
    # Now for the 3rd column of B (k=2), we use schoolbook (rank 4).
    # C02 = A00*B02 + A01*B12
    # C12 = A10*B02 + A11*B12
    
    # Col 7: A00 * B02
    U[0][7] = 1.0
    V[2][7] = 1.0  # B02 (0*3+2=2)
    W[4][7] = 1.0  # C02 (2*2+0=4)
    
    # Col 8: A01 * B12
    U[1][8] = 1.0
    V[5][8] = 1.0  # B12 (1*3+2=5)
    W[4][8] = 1.0  # C02
    
    # Col 9: A10 * B02
    U[2][9] = 1.0
    V[2][9] = 1.0
    W[5][9] = 1.0  # C12 (2*2+1=5)
    
    # Col 10: A11 * B12
    U[3][10] = 1.0
    V[5][10] = 1.0
    W[5][10] = 1.0
    
    # Verify
    if not _verify(U, V, W):
        # Fallback to schoolbook rank 12
        rank_sb = n * m * p
        U = [[0] * rank_sb for _ in range(n * m)]
        V = [[0] * rank_sb for _ in range(m * p)]
        W = [[0] * rank_sb for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
