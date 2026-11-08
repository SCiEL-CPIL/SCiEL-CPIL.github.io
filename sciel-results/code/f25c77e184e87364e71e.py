"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    # Rank-55 decomposition using Strassen for 2x2x2 blocks and direct multiplication for 2x1x2 blocks.
    # 2x3x10 = (2x2x2) * 5 + (2x1x2) * 5 = 5*7 + 5*2 = 35 + 10 = 45? 
    # Actually: 2x3 = [2x2 | 2x1]. 
    # 2x2x10 uses 5 Strassen blocks of 2x2x2, rank 5*7 = 35.
    # 2x1x10 uses 5 direct 2x1x2 blocks, rank 5*2 = 10.
    # Total rank = 45.
    
    # Wait, the previous attempt said 55. Let me re-verify.
    # Schoolbook: 2*3*10 = 60.
    # Strassen on 2x2 part: For 2x2x2, rank 7. For 2x2x10, we have 5 disjoint 2x2x2 blocks? No.
    # B is 3x10. We can split B into 5 blocks of 3x2.
    # A is 2x3.
    # A @ B = C (2x10).
    # Split A into A1 (2x2) and A2 (2x1).
    # Split B into 5 blocks B_k (3x2) for k=0..4.
    # Split each B_k into B_k1 (2x2) and B_k2 (1x2).
    # C = A1 @ B1 + A2 @ B2.
    # Term 1: A1 (2x2) @ [B_k1 (2x2) for k=0..4]. This is 5 separate 2x2x2 multiplications.
    #   Rank per 2x2x2 is 7. Total rank for Term 1 = 5 * 7 = 35.
    # Term 2: A2 (2x1) @ [B_k2 (1x2) for k=0..4]. This is 5 separate 2x1x2 multiplications.
    #   Rank per 2x1x2 is 2 (schoolbook: 2*1*2=4? No, 2x1 times 1x2 is dot product of 2-vectors, result 2x2. Rank 4? Or can it be lower?
    #   2x1x2 tensor: (x_i, y_j, z_k) -> x_i y_j z_k.
    #   A2 is 2x1, B_k2 is 1x2.
    #   C_k = A2 * B_k2. A2 = [a; b], B_k2 = [b1, b2]. C_k = [[a b1, a b2], [b b1, b b2]].
    #   This is a 2x1x2 matrix multiplication.
    #   Rank of 2x1x2: Schoolbook is 4. Is it lower?
    #   T = sum_{i,j,k} T_{ijk} x_i y_j z_k.
    #   Here indices: i in {0,1} (A row), j in {0} (middle dim 1), k in {0,1} (B col).
    #   Wait, A is 2x1, B is 1x2.
    #   (AB)_{ik} = sum_j A_{ij} B_{jk}. Since j=0, (AB)_{ik} = A_{i0} B_{0k}.
    #   Tensor T_{ik0} = x_i z_k.
    #   T = x_0 z_0 + x_0 z_1 + x_1 z_0 + x_1 z_1 = (x_0+x_1)(z_0+z_1) - x_0 z_1 - x_1 z_0 + ... no.
    #   T = (x_0+x_1)(z_0+z_1) - x_1 z_0 - x_0 z_1 + x_0 z_0 + x_1 z_1 ... 
    #   Actually, T_{ik} = x_i z_k.
    #   This is the rank-4 tensor if we consider all 4 terms.
    #   Can we do it in rank 3?
    #   x_0 z_0, x_0 z_1, x_1 z_0, x_1 z_1.
    #   (x_0+x_1)(z_0+z_1) = x_0 z_0 + x_0 z_1 + x_1 z_0 + x_1 z_1.
    #   So T = (x_0+x_1)(z_0+z_1).
    #   Wait, the tensor is T_{ijk} x_i y_j z_k.
    #   Here j is fixed to 0. So T_{i0k} = 1 for all i,k.
    #   T = x_0 z_0 + x_0 z_1 + x_1 z_0 + x_1 z_1.
    #   This is exactly (x_0 + x_1) * (z_0 + z_1).
    #   So the rank is 1?
    #   Let's check. If we set u = x_0 + x_1, v = z_0 + z_1.
    #   Then u v = x_0 z_0 + x_0 z_1 + x_1 z_0 + x_1 z_1.
    #   Yes! The 2x1x2 multiplication (outer product of 2-vectors) has rank 1?
    #   No, wait. The result C is 2x2.
    #   C_{00} = x_0 z_0
    #   C_{01} = x_0 z_1
    #   C_{10} = x_1 z_0
    #   C_{11} = x_1 z_1
    #   If we use 1 rank-1 tensor: U_0 = x_0 + x_1, V_0 = 1 (since y_0=1? No, B has 1 row, so y_0 is the only entry of B).
    #   Let's be precise.
    #   A is 2x1. B is 1x2.
    #   A = [[a0], [a1]]. B = [[b0, b1]].
    #   C = [[a0 b0, a0 b1], [a1 b0, a1 b1]].
    #   Tensor T_{i j k} where i in {0,1}, j in {0}, k in {0,1}.
    #   T_{i 0 k} = 1.
    #   We want to express C_{ik} = sum_{c=0}^{r-1} U_{i c} V_{j c} W_{k c}.
    #   Since j is only 0, V_{0 c} is the only V entry.
    #   C_{ik} = sum_c U_{i c} V_{0 c} W_{k c}.
    #   Let M_c = U_c * W_c (vector outer product? No, U_c is 2x1, W_c is 1x2? No).
    #   U is (2*1, r) = (2, r). V is (1*2, r) = (2, r). W is (2*2, r) = (4, r).
    #   Wait, the interface says:
    #   U: (n*m, rank) -> (2*1, rank) = (2, rank)
    #   V: (m*p, rank) -> (1*2, rank) = (2, rank)
    #   W: (p*n, rank) -> (2*2, rank) = (4, rank)
    #   W index: W[k*n+i, c] = W[2k+i, c].
    #   C_{ik} = W[2k+i, c] ? No, C is 2x2. C_{i, k}.
    #   The decomposition is C_{i, j, k} = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c].
    #   For 2x1x2: n=2, m=1, p=2.
    #   C_{i, j, k} = sum_c U[i, c] V[j*2+k, c] W[k*2+i, c].
    #   j is always 0.
    #   C_{i, 0, k} = sum_c U[i, c] V[k, c] W[2k+i, c].
    #   We need C_{i0} = A_{i0} B_{0k} for all i,k.
    #   If we set r=1:
    #   U[0,0]=1, U[1,0]=1.
    #   V[0,0]=1, V[1,0]=1.
    #   W[0,0]=1, W[1,0]=1, W[2,0]=1, W[3,0]=1.
    #   Then C_{00} = 1*1*1 = 1. But we want A_{00} B_{00}.
    #   The variables are U[i,c], V[jp+k,c], W[kn+i,c].
    #   In the tensor, the "input" variables are A_{ij} and B_{jk}.
    #   U encodes A, V encodes B, W encodes C.
    #   For 2x1x2:
    #   A has indices i, j (j=0). A_{i0}.
    #   B has indices j, k (j=0). B_{0k}.
    #   C has indices i, k. C_{ik} = A_{i0} B_{0k}.
    #   T = A_{00} B_{00} C_{00} + A_{00} B_{01} C_{01} + A_{10} B_{00} C_{10} + A_{10} B_{01} C_{11}.
    #   Can we write this as sum of 1 rank-1 tensor?
    #   (A_{00} + A_{10}) * (B_{00} + B_{01}) * (C_{00} + C_{01} + C_{10} + C_{11})?
    #   No, the C part must match.
    #   Rank 1 means C_{ik} = u_i v_k.
    #   Here C_{ik} = A_i B_k. This is rank 1 as a matrix for fixed A, B.
    #   But as a tensor in variables A, B, C, is it rank 1?
    #   T = (A_0 + A_1) (B_0 + B_1) (C_00 + C_01 + C_10 + C_11) ... no.
    #   Let's try to construct it.
    #   We need sum_c U[i,c] V[k,c] W[2k+i,c] = A_i B_k.
    #   This is impossible for r=1 because LHS is product of 3 linear forms, RHS is product of 2.
    #   However, if we fix W, we can solve.
    #   Actually, the standard 2x1x2 multiplication has rank 4?
    #   Schoolbook: 4 terms.
    #   Can we do better?
    #   A_0 B_0 C_00 + A_0 B_1 C_01 + A_1 B_0 C_10 + A_1 B_1 C_11.
    #   This is just the 4 distinct monomials. No sharing.
    #   So rank is 4.
    #   
    #   So Term 2 rank is 5 * 4 = 20.
    #   Total rank = 35 + 20 = 55.
    
    n, m, p = DIMENSIONS
    # n=2, m=3, p=10
    # We split into 5 blocks of width 2 for B.
    # For each block, we do 2x2x2 (rank 7) and 2x1x2 (rank 4).
    # Total rank = 5 * (7 + 4) = 55.
    
    rank = 55
    U = [[0] * rank for _ in range(n * m)]  # (6, 55)
    V = [[0] * rank for _ in range(m * p)]  # (30, 55)
    W = [[0] * rank for _ in range(p * n)]  # (20, 55)
    
    col = 0
    for block in range(5):
        # Block k corresponds to columns 2*block and 2*block+1 of B.
        # A indices:
        # A00: i=0, j=0 -> U[0, c]
        # A01: i=0, j=1 -> U[1, c]
        # A10: i=1, j=0 -> U[3, c]
        # A11: i=1, j=1 -> U[4, c]
        # A02: i=0, j=2 -> U[2, c]
        # A12: i=1, j=2 -> U[5, c]
        
        # B indices for this block:
        # B0, 2k: j=0, k=2k -> V[0*10 + 2k, c] = V[2k, c]
        # B0, 2k+1: j=0, k=2k+1 -> V[2k+1, c]
        # B1, 2k: j=1, k=2k -> V[1*10 + 2k, c] = V[10+2k, c]
        # B1, 2k+1: j=1, k=2k+1 -> V[10+2k+1, c]
        # B2, 2k: j=2, k=2k -> V[2*10 + 2k, c] = V[20+2k, c]
        # B2, 2k+1: j=2, k=2k+1 -> V[20+2k+1, c]
        
        # C indices for this block:
        # C0, 2k: i=0, k=2k -> W[2k*2 + 0, c] = W[4k, c]
        # C0, 2k+1: i=0, k=2k+1 -> W[(2k+1)*2 + 0, c] = W[4k+2, c]
        # C1, 2k: i=1, k=2k -> W[2k*2 + 1, c] = W[4k+1, c]
        # C1, 2k+1: i=1, k=2k+1 -> W[(2k+1)*2 + 1, c] = W[4k+3, c]
        
        # --- Part 1: 2x2x2 Strassen ---
        # A1 = [[A00, A01], [A10, A11]]
        # B1 = [[B0,2k, B0,2k+1], [B1,2k, B1,2k+1]]
        # C1 = [[C0,2k, C0,2k+1], [C1,2k, C1,2k+1]]
        
        # Strassen 7 products:
        # P1 = (A00 + A11) * (B0,2k + B1,2k+1)
        # P2 = (A10 + A11) * B0,2k
        # P3 = A00 * (B0,2k+1 - B1,2k+1)
        # P4 = A11 * (B1,2k - B0,2k)
        # P5 = (A00 + A01) * B1,2k+1
        # P6 = (A10 - A00) * (B0,2k + B1,2k)
        # P7 = (A01 - A11) * (B1,2k + B0,2k+1)
        
        # Outputs:
        # C0,2k   = P1 + P4 - P5 + P7
        # C0,2k+1 = P3 + P5
        # C1,2k   = P2 + P4
        # C1,2k+1 = P1 - P2 + P3 + P6
        
        # We assign 7 columns for this.
        
        # P1
        U[0][col] = 1; U[4][col] = 1
        V[2*k][col] = 1; V[10+2*k+1][col] = 1
        W[4*k][col] = 1; W[4*k+1][col] = 1; W[4*k+2][col] = -1; W[4*k+3][col] = 1
        col += 1
        
        # P2
        U[3][col] = 1; U[4][col] = 1
        V[2*k][col] = 1
        W[4*k+1][col] = 1; W[4*k+3][col] = -1
        col += 1
        
        # P3
        U[0][col] = 1
        V[2*k+1][col] = 1; V[10+2*k+1][col] = -1
        W[4*k+2][col] = 1; W[4*k+3][col] = 1
        col += 1
        
        # P4
        U[4][col] = 1
        V[10+2*k][col] = 1; V[2*k][col] = -1
        W[4*k][col] = 1; W[4*k+1][col] = 1
        col += 1
        
        # P5
        U[0][col] = 1; U[1][col] = 1
        V[10+2*k+1][col] = 1
        W[4*k][col] = -1; W[4*k+2][col] = 1
        col += 1
        
        # P6
        U[3][col] = 1; U[0][col] = -1
        V[2*k][col] = 1; V[10+2*k][col] = 1
        W[4*k+3][col] = 1
        col += 1
        
        # P7
        U[1][col] = 1; U[4][col] = -1
        V[10+2*k][col] = 1; V[2*k+1][col] = 1
        W[4*k][col] = 1
        col += 1
        
        # --- Part 2: 2x1x2 Schoolbook ---
        # A2 = [[A02], [A12]]
        # B2 = [[B2,2k, B2,2k+1]]
        # C2 = [[C0,2k, C0,2k+1], [C1,2k, C1,2k+1]]
        # Wait, C is the sum of C1 and C2.
        # C1 comes from A1 @ B1. C2 comes from A2 @ B2.
        # So we add C2 to the same W entries.
        
        # C0,2k += A02 * B2,2k
        # C0,2k+1 += A02 * B2,2k+1
        # C1,2k += A12 * B2,2k
        # C1,2k+1 += A12 * B2,2k+1
        
        # 4 terms.
        
        # A02 * B2,2k
        U[2][col] = 1
        V[20+2*k][col] = 1
        W[4*k][col] = 1
        col += 1
        
        # A02 * B2,2k+1
        U[2][col] = 1
        V[20+2*k+1][col] = 1
        W[4*k+2][col] = 1
        col += 1
        
        # A12 * B2,2k
        U[5][col] = 1
        V[20+2*k][col] = 1
        W[4*k+1][col] = 1
        col += 1
        
        # A12 * B2,2k+1
        U[5][col] = 1
        V[20+2*k+1][col] = 1
        W[4*k+3][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
