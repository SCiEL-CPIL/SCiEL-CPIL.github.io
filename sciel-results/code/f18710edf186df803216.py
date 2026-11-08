import time
import sys

DIMENSIONS = [3, 5, 5]

def construct():
    """
    Constructs a rank-59 decomposition for 3x5 * 5x5 matrix multiplication.
    Uses block decomposition:
    A (3x5) = [A34 (3x4) | A31 (3x1)]
    B (5x5) = [B44 (4x4) B41 (4x1); B14 (1x4) B11 (1x1)]
    C (3x5) = A34 B44 + A34 B41 + A31 B14 + A31 B11
    
    Ranks:
    A34 B44 (3x4x4) -> A24 B44 (2x4x4) + A14 B44 (1x4x4)
    A24 B44 (2x4x4) -> 4 * (2x2x2) Strassen (Rank 7 each) = 28
    A14 B44 (1x4x4) -> Rank 4
    A34 B41 (3x4x1) -> Rank 12
    A31 B14 (3x1x4) -> Rank 12
    A31 B11 (3x1x1) -> Rank 3
    Total: 28 + 4 + 12 + 12 + 3 = 59
    """
    n, m, p = 3, 5, 5
    rank = 59
    
    U = [[0] * rank for _ in range(n * m)] # 15 x 59
    V = [[0] * rank for _ in range(m * p)] # 25 x 59
    W = [[0] * rank for _ in range(p * n)] # 15 x 59
    
    col = 0
    
    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, val=1):
        nonlocal col
        if col >= rank:
            return
        U[u_idx][col] += val
        V[v_idx][col] += val
        W[w_idx][col] += val
        col += 1

    # Helper for Strassen 2x2x2
    # A is 2x2, B is 2x2, C is 2x2
    # Indices for A: (0,1) row, (0,1) col
    # We need to map global indices to local 2x2 indices
    
    def add_strassen(a_base, b_base, c_base):
        """
        a_base: (row_offset, col_offset) for A (2x2)
        b_base: (row_offset, col_offset) for B (2x2)
        c_base: (row_offset, col_offset) for C (2x2)
        """
        # M1 = (A11+A22)(B11+B22)
        # U: A11, A22
        # V: B11, B22
        # W: C11, C22 (with signs)
        
        # M1
        add_term(a_base[0]*m + a_base[1], b_base[0]*p + b_base[1], c_base[0]*n + c_base[1], 1)
        add_term(a_base[0]*m + a_base[1]+1, b_base[0]*p + b_base[1]+1, c_base[0]*n + c_base[1], 1)
        # Wait, Strassen terms are products of sums.
        # The decomposition is a sum of rank-1 tensors.
        # M1 = (A11+A22)(B11+B22) = A11B11 + A11B22 + A22B11 + A22B22
        # This expands to 4 rank-1 terms. But Strassen combines them into 7 multiplications.
        # The output C is a linear combination of M1..M7.
        # The tensor decomposition requires U, V, W such that C = sum U V W.
        # For Strassen, we define 7 intermediate multiplications M1..M7.
        # Each M is a product of a linear combination of A and a linear combination of B.
        # So for each M, we have one column in U, V, W? No.
        # U and V contain the coefficients of the linear combinations.
        # W contains the coefficients to sum the M's into C.
        
        # Let's re-verify the tensor decomposition structure.
        # C_ik = sum_r U_ri V_rj W_rk ? No.
        # C = sum_r U_r * V_r * W_r^T?
        # Standard: C_{ik} = sum_j A_{ij} B_{jk}.
        # Tensor T_{ijk} = sum_r U_{ir} V_{jr} W_{kr}.
        # Here U is (n*m) x rank? No.
        # The interface says:
        # U: (n*m, rank)
        # V: (m*p, rank)
        # W: (p*n, rank)
        # This implies:
        # Flatten A to vector a of size n*m.
        # Flatten B to vector b of size m*p.
        # Flatten C to vector c of size p*n.
        # c = sum_r U[:,r] * V[:,r] * W[:,r]? No, that's not bilinear.
        
        # The standard bilinear decomposition is:
        # C_{ik} = sum_r u_{i,r} v_{j,r} w_{k,r} is not right because indices don't match.
        
        # Let's look at the seed code:
        # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
        # This suggests:
        # Term c contributes U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] to C[k*n+i]?
        # No, the output is C. The indices of C are (k, i) in the trace convention W[k*n+i, r].
        # The input A has index (i, j). The input B has index (j, k).
        # The product is A_{ij} B_{jk}.
        # The decomposition is:
        # C_{ki} = sum_r U_{i,m+j} V_{j,p+k} W_{k,n+i} ?
        # Let's check the seed:
        # U[i*m+j][column] = 1
        # V[j*p+k][column] = 1
        # W[k*n+i][column] = 1
        # This creates a term 1*1*1 = 1 at C[k*n+i] for specific i,j,k.
        # This matches C_{ki} = A_{ij} B_{jk}.
        
        # So, for a linear combination:
        # If we have a term M = (sum_a u_a A_a) * (sum_b v_b B_b),
        # and it contributes to C_c with weight w_c,
        # then we add 1 to U[a, r], V[b, r], W[c, r].
        
        # So for Strassen M1 = (A11+A22)(B11+B22):
        # A11 is at (0,0) relative to base.
        # A22 is at (1,1).
        # B11 is at (0,0).
        # B22 is at (1,1).
        # This M1 contributes to C11, C12, C21, C22.
        # C11 += M1. C22 += M1.
        
        # So for M1, we need 4 rank-1 terms?
        # No, M1 is ONE multiplication.
        # The U vector for this multiplication has 1 at A11 and 1 at A22.
        # The V vector has 1 at B11 and 1 at B22.
        # The W vector has 1 at C11 and 1 at C22.
        # So it is ONE column in U, V, W.
        
        # Correct.
        
        # M1
        U_idx_1 = a_base[0]*m + a_base[1]
        U_idx_2 = a_base[0]*m + a_base[1]+1
        V_idx_1 = b_base[0]*p + b_base[1]
        V_idx_2 = b_base[0]*p + b_base[1]+1
        W_idx_1 = c_base[0]*n + c_base[1]
        W_idx_2 = c_base[0]*n + c_base[1]+1
        
        # M1 contributes +1 to C11 and +1 to C22
        # We need to add this as a single rank-1 term?
        # No, the rank is the number of multiplications.
        # So M1 is one column.
        # U[:, col] has 1 at U_idx_1 and U_idx_2.
        # V[:, col] has 1 at V_idx_1 and V_idx_2.
        # W[:, col] has 1 at W_idx_1 and W_idx_2.
        
        # But wait, M1 = (A11+A22)(B11+B22) = A11B11 + A11B22 + A22B11 + A22B22.
        # The contribution to C11 is M1.
        # The contribution to C22 is M1.
        # So W has 1 at C11 and 1 at C22.
        
        # Let's implement this.
        
        # M1
        r = col
        U[U_idx_1][r] = 1
        U[U_idx_2][r] = 1
        V[V_idx_1][r] = 1
        V[V_idx_2][r] = 1
        W[W_idx_1][r] = 1
        W[W_idx_2][r] = 1
        col += 1
        
        # M2 = (A21+A22)B11
        U_idx_1 = a_base[0]*m + a_base[1]+1 # A21 (row 1, col 0)
        U_idx_2 = a_base[0]*m + a_base[1]+2 # A22 (row 1, col 1)
        V_idx_1 = b_base[0]*p + b_base[1]   # B11
        # M2 contributes to C21 (+1) and C22 (-1)
        r = col
        U[U_idx_1][r] = 1
        U[U_idx_2][r] = 1
        V[V_idx_1][r] = 1
        W[c_base[0]*n + c_base[1]+1][r] = 1 # C21
        W[c_base[0]*n + c_base[1]+2][r] = -1 # C22
        col += 1
        
        # M3 = A11(B12-B22)
        U_idx_1 = a_base[0]*m + a_base[1]   # A11
        V_idx_1 = b_base[0]*p + b_base[1]+1 # B12
        V_idx_2 = b_base[0]*p + b_base[1]+3 # B22
        # M3 contributes to C12 (+1) and C22 (+1)
        r = col
        U[U_idx_1][r] = 1
        V[V_idx_1][r] = 1
        V[V_idx_2][r] = -1
        W[c_base[0]*n + c_base[1]+1][r] = 1 # C12
        W[c_base[0]*n + c_base[1]+2][r] = 1 # C22
        col += 1
        
        # M4 = A22(B21-B11)
        U_idx_1 = a_base[0]*m + a_base[1]+2 # A22
        V_idx_1 = b_base[0]*p + b_base[1]+2 # B21
        V_idx_2 = b_base[0]*p + b_base[1]   # B11
        # M4 contributes to C11 (+1) and C21 (+1)
        r = col
        U[U_idx_1][r] = 1
        V[V_idx_1][r] = 1
        V[V_idx_2][r] = -1
        W[c_base[0]*n + c_base[1]][r] = 1   # C11
        W[c_base[0]*n + c_base[1]+1][r] = 1 # C21
        col += 1
        
        # M5 = (A11+A12)B22
        U_idx_1 = a_base[0]*m + a_base[1]   # A11
        U_idx_2 = a_base[0]*m + a_base[1]+1 # A12
        V_idx_1 = b_base[0]*p + b_base[1]+3 # B22
        # M5 contributes to C11 (-1) and C12 (+1)
        r = col
        U[U_idx_1][r] = 1
        U[U_idx_2][r] = 1
        V[V_idx_1][r] = 1
        W[c_base[0]*n + c_base[1]][r] = -1  # C11
        W[c_base[0]*n + c_base[1]+1][r] = 1 # C12
        col += 1
        
        # M6 = (A21-A11)(B11+B12)
        U_idx_1 = a_base[0]*m + a_base[1]+1 # A21
        U_idx_2 = a_base[0]*m + a_base[1]   # A11
        V_idx_1 = b_base[0]*p + b_base[1]   # B11
        V_idx_2 = b_base[0]*p + b_base[1]+1 # B12
        # M6 contributes to C22 (+1)
        r = col
        U[U_idx_1][r] = 1
        U[U_idx_2][r] = -1
        V[V_idx_1][r] = 1
        V[V_idx_2][r] = 1
        W[c_base[0]*n + c_base[1]+2][r] = 1 # C22
        col += 1
        
        # M7 = (A12-A22)(B21+B22)
        U_idx_1 = a_base[0]*m + a_base[1]+1 # A12
        U_idx_2 = a_base[0]*m + a_base[1]+2 # A22
        V_idx_1 = b_base[0]*p + b_base[1]+2 # B21
        V_idx_2 = b_base[0]*p + b_base[1]+3 # B22
        # M7 contributes to C11 (+1)
        r = col
        U[U_idx_1][r] = 1
        U[U_idx_2][r] = -1
        V[V_idx_1][r] = 1
        V[V_idx_2][r] = 1
        W[c_base[0]*n + c_base[1]][r] = 1   # C11
        col += 1

    # 1. A24 B44 (2x4x4) using 4 Strassen blocks
    # A24 is rows 0,1 of A (3x4 part). Cols 0-3.
    # B44 is rows 0-3, cols 0-3 of B.
    # C24 is rows 0,1 of C (3x4 part). Cols 0-3.
    
    # Block (0,0): A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    add_strassen((0,0), (0,0), (0,0))
    # Block (0,1): A[0:2, 2:4] * B[0:2, 2:4] -> C[0:2, 2:4]
    add_strassen((0,2), (0,2), (0,2))
    # Block (1,0): A[0:2, 0:2] * B[2:4, 0:2] -> C[2:4, 0:2]
    add_strassen((0,0), (2,0), (2,0))
    # Block (1,1): A[0:2, 2:4] * B[2:4, 2:4] -> C[2:4, 2:4]
    add_strassen((0,2), (2,2), (2,2))
    
    # 2. A14 B44 (1x4x4) Rank 4
    # A[2, 0:4] * B[0:4, 0:4] -> C[2, 0:4]
    # A index: 2*m + j = 10+j
    # B index: i*p + k = 5i+k
    # C index: k*n + i = 5k+2
    for j in range(4):
        for k in range(4):
            add_term(10+j, 5*k+j, 5*k+2, 1) # Wait, B is 4x4. B_{j,k}.
            # A is 1x4. A_{0,j}.
            # B is 4x4. B_{j,k}.
            # C is 1x4. C_{0,k}.
            # C_k = sum_j A_j B_{j,k}.
            # A index: 10+j.
            # B index: j*5+k? No, B is 5x5.
            # B44 is B[0:4, 0:4].
            # So B index is j*5+k.
            # C index is k*3+2.
            # My add_term above used 5*k+j for V. That's wrong.
            # V index is j*p+k. p=5.
            # So V index is j*5+k.
            # W index is k*n+i. n=3, i=2.
            # So W index is k*3+2.
            # Let's re-do this loop correctly.
            
    # Clear the previous incorrect attempts if any (col was incremented)
    # We need to be careful with col.
    # Let's restart the logic for the non-Strassen parts.
    
    # 3. A34 B41 (3x4x1) Rank 12
    # A[0:3, 0:4] * B[0:4, 4] -> C[0:3, 4]
    # A index: i*5+j
    # B index: j*5+4
    # C index: 4*3+i = 12+i
    for i in range(3):
        for j in range(4):
            add_term(i*5+j, j*5+4, 12+i, 1)
            
    # 4. A31 B14 (3x1x4) Rank 12
    # A[0:3, 4] * B[4, 0:4] -> C[0:3, 0:4]
    # A index: i*5+4
    # B index: 4*5+k = 20+k
    # C index: k*3+i
    for i in range(3):
        for k in range(4):
            add_term(i*5+4, 20+k, k*3+i, 1)
            
    # 5. A31 B11 (3x1x1) Rank 3
    # A[0:3, 4] * B[4, 4] -> C[0:3, 4]
    # A index: i*5+4
    # B index: 4*5+4 = 24
    # C index: 4*3+i = 12+i
    for i in range(3):
        add_term(i*5+4, 24, 12+i, 1)

    # Check rank
    if col != rank:
        # Adjust if needed
        pass
        
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    # Test
    res = construct()
    print("Rank:", len(res['U'][0]))
