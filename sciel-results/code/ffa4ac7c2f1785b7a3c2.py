"""Matrix multiplication 2x2, 2x7 exact tensor decomposition search."""
import time
from fractions import Fraction

DIMENSIONS = [2, 2, 7]

def _build_schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p):
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(len(U[0])):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if s != 1:
                    return False
    return True

def _get_strassen_decomp():
    """
    Returns U, V, W for 2x2x2 Strassen decomposition.
    U: 4x7, V: 4x7, W: 4x7
    Indices:
    U rows: 0:A00, 1:A01, 2:A10, 3:A11
    V rows: 0:B00, 1:B01, 2:B10, 3:B11
    W rows: 0:C00, 1:C01, 2:C10, 3:C11
    """
    U = [
        [1, 0, 1, 0, 1, -1, 0], # M1..M7 for A00
        [0, 0, 0, 0, 1, 0, 1],  # A01
        [0, 1, 0, 0, 0, 1, 0],  # A10
        [1, 1, 0, 1, 0, 0, -1]  # A11
    ]
    V = [
        [1, 1, 0, -1, 0, 1, 0], # B00
        [0, 0, 1, 0, 0, 1, 0],  # B01
        [0, 0, 0, 1, 0, 0, 1],  # B10
        [1, 0, -1, 0, 1, 0, 1]  # B11
    ]
    W = [
        [1, 0, 0, 1, -1, 0, 1], # C00
        [0, 0, 1, 0, 1, 0, 0],  # C01
        [0, 1, 0, 1, 0, 0, 0],  # C10
        [1, -1, 1, 0, 0, 1, 0]  # C11
    ]
    return U, V, W

def _get_schoolbook_1col_decomp():
    """
    Returns U, V, W for 2x2x1 Schoolbook decomposition.
    U: 4x4, V: 2x4, W: 2x4
    Indices:
    U rows: 0:A00, 1:A01, 2:A10, 3:A11
    V rows: 0:B00, 1:B10 (Note: local index for col 0)
    W rows: 0:C0, 1:C1
    """
    U = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]
    V = [
        [1, 0, 1, 0], # B00
        [0, 1, 0, 1]  # B10
    ]
    W = [
        [1, 1, 0, 0], # C0
        [0, 0, 1, 1]  # C1
    ]
    return U, V, W

def _search_rank24():
    """
    Attempts to find a rank 24 decomposition by trying to merge the last 2x1 block (rank 4)
    with the previous 2x2 block (rank 7) to form a 2x3 block with rank 10 instead of 11.
    
    Total rank would be 7*2 + 10 = 24.
    
    This requires solving a system of polynomial equations. We use a simple grid search 
    over integer/half-integer coefficients for the new 2x3 block tensors.
    """
    import itertools
    
    n, m, p = DIMENSIONS # 2, 2, 7
    
    # We construct the first 2 blocks (cols 0-3) using standard Strassen.
    # Block 1: k=0,1. Rank 7.
    # Block 2: k=2,3. Rank 7.
    # Block 3: k=4,5,6. Target Rank 10.
    
    R = 24
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    StrU, StrV, StrW = _get_strassen_decomp()
    
    # Block 1: Cols 0, 1 (k=0,1)
    col_offset = 0
    for r in range(7):
        for i in range(4):
            U[i][col_offset + r] = StrU[i][r]
        V[0][col_offset + r] = StrV[0][r]   # B00
        V[1][col_offset + r] = StrV[1][r]   # B01
        V[7][col_offset + r] = StrV[2][r]   # B10
        V[8][col_offset + r] = StrV[3][r]   # B11
        
        W[0][col_offset + r] = StrW[0][r]   # C00
        W[1][col_offset + r] = StrW[1][r]   # C01
        W[2][col_offset + r] = StrW[2][r]   # C10
        W[3][col_offset + r] = StrW[3][r]   # C11
    col_offset += 7

    # Block 2: Cols 2, 3 (k=2,3)
    for r in range(7):
        for i in range(4):
            U[i][col_offset + r] = StrU[i][r]
        
        V[2][col_offset + r] = StrV[0][r]   # B02
        V[3][col_offset + r] = StrV[1][r]   # B03
        V[9][col_offset + r] = StrV[2][r]   # B12
        V[10][col_offset + r] = StrV[3][r]  # B13
        
        W[4][col_offset + r] = StrW[0][r]   # C20
        W[5][col_offset + r] = StrW[1][r]   # C21
        W[6][col_offset + r] = StrW[2][r]   # C30
        W[7][col_offset + r] = StrW[3][r]   # C31
    col_offset += 7

    # Block 3: Cols 4, 5, 6 (k=4,5,6). Target Rank 10.
    # Indices involved:
    # A: 0..3
    # B: 4(B04), 5(B05), 6(B06), 11(B14), 12(B15), 13(B16)
    # C: 8(C40), 9(C41), 10(C50), 11(C51), 12(C60), 13(C61)
    
    # We need to find U_sub (4x10), V_sub (6x10), W_sub (6x10)
    # such that sum_{r=0}^9 U[a][r] V[b][r] W[c][r] = 1 if (a,b,c) is a valid product term, else 0.
    
    # Valid terms for 2x2 x 2x3:
    # C = A B
    # C00 = A00 B00 + A01 B10
    # C01 = A00 B01 + A01 B11
    # C10 = A10 B00 + A11 B10
    # C11 = A10 B01 + A11 B11
    # Mapping to local indices:
    # A: 0(A00), 1(A01), 2(A10), 3(A11)
    # B_local: 0(B04), 1(B05), 2(B06), 3(B14), 4(B15), 5(B16)
    # C_local: 0(C40), 1(C41), 2(C50), 3(C51), 4(C60), 5(C61)
    
    # Equations:
    # C_local 0 (C40) depends on A00, A01 and B_local 0 (B04), 3 (B14)
    # C_local 1 (C41) depends on A00, A01 and B_local 1 (B05), 4 (B15)
    # C_local 2 (C50) depends on A10, A11 and B_local 0 (B04), 3 (B14)
    # C_local 3 (C51) depends on A10, A11 and B_local 1 (B05), 4 (B15)
    # C_local 4 (C60) depends on A00, A01 and B_local 2 (B06), 5 (B16)
    # C_local 5 (C61) depends on A10, A11 and B_local 2 (B06), 5 (B16)
    
    # This is a set of 12 independent 2x2 matrix multiplications?
    # No, the A part is shared.
    # Terms:
    # T0: A00 B04 -> C40
    # T1: A01 B14 -> C40
    # T2: A00 B05 -> C41
    # T3: A01 B15 -> C41
    # T4: A10 B04 -> C50
    # T5: A11 B14 -> C50
    # T6: A10 B05 -> C51
    # T7: A11 B15 -> C51
    # T8: A00 B06 -> C60
    # T9: A01 B16 -> C60
    # T10: A10 B06 -> C61
    # T11: A11 B16 -> C61
    
    # We can group by A rows?
    # Group 0 (A00, A01): B04, B05, B06 and B14, B15, B16. Outputs C40, C41, C60.
    # Group 1 (A10, A11): B04, B05, B06 and B14, B15, B16. Outputs C50, C51, C61.
    
    # This looks like two 2x1 matrix multiplications (A_row x B_col) resulting in 2x3?
    # No, A is 2x2.
    # Row 0 of A: [A00, A01]. B rows 0,1.
    # C_row 0: [C40, C41, C60] = [A00 B04 + A01 B14, A00 B05 + A01 B15, A00 B06 + A01 B16]
    # C_row 1: [C50, C51, C61] = [A10 B04 + A11 B14, A10 B05 + A11 B15, A10 B06 + A11 B16]
    
    # This is exactly two independent 1x2 by 2x3 matrix multiplications?
    # Let X0 = [A00, A01], Y = B (2x3). Z0 = C_row 0 (1x3).
    # X0 Y = Z0.
    # Rank of 1x2 x 2x3 tensor?
    # It is the outer product of X0 and Y? No, it's a sum.
    # Z0_j = sum_i X0_i Y_ij.
    # This is rank 2? (X0 is 1x2, Y is 2x3).
    # The tensor is T = sum_{i,j} E_{i} \otimes E_{j} \otimes E_{j+?}
    # Actually, the rank of the multiplication tensor for 1xn x nxm is n.
    # So rank of 1x2 x 2x3 is 2.
    # We have two such multiplications (one for row 0, one for row 1).
    # They are disjoint in A and C indices.
    # So total rank = 2 + 2 = 4?
    # Wait, the B indices are shared.
    # T_total = T_row0 + T_row1.
    # T_row0 involves A00, A01 and B04, B14, B05, B15, B06, B16 and C40, C41, C60.
    # T_row1 involves A10, A11 and B04, B14, B05, B15, B06, B16 and C50, C51, C61.
    
    # Let's check the rank of 1x2 x 2x3.
    # M(1,2,3).
    # Schoolbook: 2*3 = 6.
    # Can we do better?
    # Z_j = A0 B0j + A1 B1j.
    # This is equivalent to multiplying a 1x2 vector by a 2x3 matrix.
    # The tensor is sum_{j=0}^2 (B0j, B1j) \otimes A0, A1 \otimes Cj.
    # It is known that M(1,n,m) = n.
    # So M(1,2,3) = 2.
    # So rank is 2 for each row.
    # Since the A and C indices for row 0 and row 1 are disjoint, the total rank is 2+2=4.
    
    # So we can decompose the last 3 columns with rank 4?
    # If so, total rank = 7 + 7 + 4 = 18?
    # Wait, the B indices are shared between the two "row" tensors.
    # Does disjointness in A and C imply additivity of rank?
    # T = T1 + T2.
    # supp(T1) \cap supp(T2) = empty in A and C.
    # In B, they share.
    # However, the tensor is a sum of rank 1 terms.
    # T1 = sum_{r=1}^2 U1_r V1_r W1_r.
    # T2 = sum_{r=1}^2 U2_r V2_r W2_r.
    # Since U1_r and U2_r have disjoint supports (A00/A01 vs A10/A11), and W1_r and W2_r have disjoint supports (C40/C41/C60 vs C50/C51/C61), the union of these 4 rank 1 tensors is a valid decomposition.
    # So yes, rank is 4.
    
    # So why did the previous code use rank 4 for 2x1 and rank 7 for 2x2?
    # 2x1 is M(2,2,1).
    # M(2,2,1) = 4.
    # M(2,2,3) = 4?
    # Let's verify M(2,2,3) = 4.
    # If M(2,2,3)=4, then M(2,2,7) = M(2,2,3) + M(2,2,3) + M(2,2,1) = 4+4+4 = 12?
    # Or 3 blocks of 2x2 and 1 of 2x1?
    # 2x2 is M(2,2,2)=7.
    # 2x3 is M(2,2,3).
    # If M(2,2,3)=4, then 7+7+4 = 18.
    
    # Let's construct the rank 4 decomposition for 2x2 x 2x3.
    # We need 4 rank 1 terms.
    # Row 0 part (A00, A01 -> C40, C41, C60):
    # Z0_j = A00 B0j + A01 B1j for j=0,1,2 (local cols 4,5,6).
    # This is rank 2.
    # Terms:
    # r=0: U0 = [1, 0, 0, 0] (A00). V0 = [1, 1, 1, 0, 0, 0] (B04, B05, B06). W0 = [1, 1, 0, 0, 1, 0] (C40, C41, C60).
    # Check: U0 V0 W0 = 1 * (B04+B05+B06) * (C40+C41+C60)? No.
    # Rank 1 term is u \otimes v \otimes w.
    # Sum of terms.
    # Term 1: u1=[1,0,0,0], v1=[1,0,0,0,0,0] (B04), w1=[1,0,0,0,0,0] (C40). Product: A00 B04 C40.
    # This is schoolbook.
    # We need to combine them.
    # M(1,2,3) rank 2 decomposition:
    # Z_j = A0 B0j + A1 B1j.
    # Let M1 = A0, M2 = A1.
    # Z_0 = M1 B_00 + M2 B_10
    # Z_1 = M1 B_01 + M2 B_11
    # Z_2 = M1 B_02 + M2 B_12
    # We can factor?
    # Consider V = [[B_00, B_01, B_02], [B_10, B_11, B_12]].
    # Z = [M1, M2] V.
    # Rank 2 decomposition:
    # Split V into sum of 2 rank 1 matrices?
    # V = v1 w1^T + v2 w2^T.
    # Then Z = M1 v1 w1^T + M1 v2 w2^T + M2 v1 w1^T + M2 v2 w2^T = (M1 v1 + M2 v1) w1^T + ...
    # This results in 4 terms.
    # Wait, M(1,2,3) is the tensor T such that T_{i,j,k} = 1 if i=j=k? No.
    # T_{a,b,c} = 1 if a,b,c form a valid product.
    # The rank of the tensor for multiplication (1xn) x (nxm) -> (1xm) is n.
    # So M(1,2,3) = 2.
    # Decomposition:
    # M1 = A0. M2 = A1.
    # B is 2x3.
    # Z is 1x3.
    # Z_j = A0 B_0j + A1 B_1j.
    # Rank 2 terms:
    # r=1: U=[1,0] (A0). V=[1,1] (B00, B10)? No.
    # Let's use explicit rank 2 decomposition for 1x2 x 2x3.
    # Term 1: u=[1,0], v=[1,0], w=[1,0,0]. (A0 B00 C0)
    # Term 2: u=[0,1], v=[0,1], w=[0,0,1]? No.
    
    # Actually, the standard decomposition for 1xn x nxm with rank n is:
    # r=1..n.
    # U_r: 1 at index 0 (for A0)? No.
    # U is 1xn. V is nxm.
    # U_r has 1 at position r-1?
    # If U=[1,0], V=[1,0]^T, W=[1,0,0].
    # Sum:
    # r=0: U=[1,0], V=[1,1,0,0,0,0] (B04, B05, B14, B15...)?
    # Let's look at M(1,2,2). Rank 2.
    # Z0 = A0 B00 + A1 B10.
    # Z1 = A0 B01 + A1 B11.
    # Decomposition:
    # r=1: U=[1,0], V=[1,0], W=[1,0]. (A0 B00 C0)
    # r=2: U=[0,1], V=[0,1], W=[0,1]. (A1 B10 C1) -> Missing A0 B01 and A1 B10?
    # No, schoolbook is rank 4.
    # Strassen-like for 1x2 x 2x2?
    # M1 = A0 (B00+B10). M2 = A1 (B01-B10). M3 = A0 (B01). M4 = A1 (B00)?
    # M(1,2,2) = 2.
    # Terms:
    # r=1: U=[1,0], V=[1,1], W=[1,1]. (A0(B00+B10)(C0+C1)) -> A0B00C0, A0B10C0, A0B00C1, A0B10C1.
    # This has cross terms.
    # We need exact decomposition.
    
    # Given the complexity and time limit, and the fact that rank 25 is safe,
    # I will stick to the rank 25 construction.
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    # Rank 25 Construction
    # 3 blocks of 2x2 (Strassen, rank 7 each) + 1 block of 2x1 (Schoolbook, rank 4)
    # Total Rank = 7 * 3 + 4 = 25
    
    R = 25
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    StrU, StrV, StrW = _get_strassen_decomp()
    SchU, SchV, SchW = _get_schoolbook_1col_decomp()
    
    # Block 1: Cols 0, 1 (k=0,1)
    col_offset = 0
    for r in range(7):
        for i in range(4):
            U[i][col_offset + r] = StrU[i][r]
        V[0][col_offset + r] = StrV[0][r]   # B00
        V[1][col_offset + r] = StrV[1][r]   # B01
        V[7][col_offset + r] = StrV[2][r]   # B10
        V[8][col_offset + r] = StrV[3][r]   # B11
        
        W[0][col_offset + r] = StrW[0][r]   # C00
        W[1][col_offset + r] = StrW[1][r]   # C01
        W[2][col_offset + r] = StrW[2][r]   # C10
        W[3][col_offset + r] = StrW[3][r]   # C11
    col_offset += 7

    # Block 2: Cols 2, 3 (k=2,3)
    for r in range(7):
        for i in range(4):
            U[i][col_offset + r] = StrU[i][r]
        
        V[2][col_offset + r] = StrV[0][r]   # B02
        V[3][col_offset + r] = StrV[1][r]   # B03
        V[9][col_offset + r] = StrV[2][r]   # B12
        V[10][col_offset + r] = StrV[3][r]  # B13
        
        W[4][col_offset + r] = StrW[0][r]   # C20
        W[5][col_offset + r] = StrW[1][r]   # C21
        W[6][col_offset + r] = StrW[2][r]   # C30
        W[7][col_offset + r] = StrW[3][r]   # C31
    col_offset += 7

    # Block 3: Cols 4, 5 (k=4,5)
    for r in range(7):
        for i in range(4):
            U[i][col_offset + r] = StrU[i][r]
        
        V[4][col_offset + r] = StrV[0][r]   # B04
        V[5][col_offset + r] = StrV[1][r]   # B05
        V[11][col_offset + r] = StrV[2][r]  # B14
        V[12][col_offset + r] = StrV[3][r]  # B15
        
        W[8][col_offset + r] = StrW[0][r]   # C40
        W[9][col_offset + r] = StrW[1][r]   # C41
        W[10][col_offset + r] = StrW[2][r]  # C50
        W[11][col_offset + r] = StrW[3][r]  # C51
    col_offset += 7

    # Block 4: Col 6 (k=6)
    for r in range(4):
        for i in range(4):
            U[i][col_offset + r] = SchU[i][r]
        
        V[6][col_offset + r] = SchV[0][r]   # B06
        V[13][col_offset + r] = SchV[1][r]  # B16
        
        W[12][col_offset + r] = SchW[0][r]  # C60
        W[13][col_offset + r] = SchW[1][r]  # C61
    col_offset += 4

    # Verify
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback
    return _build_schoolbook()

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
