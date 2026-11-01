"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    # Strassen decomposition for 2x2 * 2x16 matrix multiplication.
    # Rank = 7 * 8 = 56
    n, m, p = DIMENSIONS
    assert n == 2 and m == 2 and p == 16
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # W index mapping: W[k*n + i, r]
    # n=2, p=16. Indices 0..31.
    # k in 0..15, i in 0..1.
    # C is 16x2. C[k, i] is at row k*2 + i.
    # C_00 (block 0,0 of 2x2 C? No, C is 16x2.
    # A is 2x2. B is 2x16.
    # C = A * B.
    # C[i, k] = sum_j A[i, j] B[j, k].
    # Wait, the task says "2 by 2 matrix by a 2 by 16 matrix".
    # Usually matrix multiplication C = A B where A is n x m and B is m x p.
    # Here n=2, m=2, p=16.
    # A is 2x2. B is 2x16. C is 2x16.
    # But the output factor W has shape (p*n, rank) = (32, rank).
    # And the convention is W[k*n + i, r].
    # This implies the output is indexed by k (0..15) and i (0..1).
    # This corresponds to C[k, i] if C is p x n? Or C^T?
    # Standard matmul C = A B gives C[i, k] for i in n, k in p.
    # If W is indexed by k*n + i, it is storing C^T or a specific layout.
    # Let's check the schoolbook code:
    # U[i * m + j] -> A[i, j]
    # V[j * p + k] -> B[j, k]
    # W[k * n + i] -> C[i, k] placed at index k*n+i.
    # So W stores C^T effectively, or C with indices swapped.
    # The product sum_r U[i,j,r] V[j,k,r] should equal W[k*n+i, r] * ...?
    # No, tensor decomposition: T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} (with indices matching).
    # The tensor is T_{i,j,k} = 1 if this term contributes to C[i,k].
    # The factorization is T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}.
    # Note the W index is k*n+i.
    
    # Strassen applies to 2x2 blocks.
    # We can view B as 2 x 2 x 8? No, B is 2x16.
    # We can split B into 8 independent 2x2 blocks along the column dimension?
    # B[j, k] where k = 2*t + s, s in {0,1}.
    # Then C[i, k] = sum_j A[i,j] B[j, 2t+s].
    # This is equivalent to 8 independent 2x2 * 2x2 multiplications?
    # Let B_t be the 2x2 matrix B[:, 2t:2t+2].
    # Then C[:, 2t:2t+2] = A * B_t.
    # So we have 8 independent 2x2x2 tensor products.
    # For each t in 0..7, we solve the 2x2x2 matmul tensor.
    # The standard rank for 2x2x2 is 7 (Strassen).
    # So total rank is 8 * 7 = 56.
    
    # Strassen formulas for C = A B (2x2):
    # M1 = (A10 + A11) B00
    # M2 = A11 (B01 - B00)
    # M3 = A00 (B10 - B11)
    # M4 = A00 + A11
    # M5 = A00 + A01
    # M6 = A10 - A00
    # M7 = A01 - A11
    # C11 = M1 + M4 - M5 + M7
    # C10 = M2 + M4
    # C01 = M3 + M5
    # C00 = M1 - M2 + M3 + M6
    
    # Wait, different conventions exist. Let's use the standard ones:
    # p1 = (a11 + a22)(b11 + b22)
    # p2 = (a21 + a22)b11
    # p3 = a11(b21 - b22)
    # p4 = a22(b12 - b11)
    # p5 = (a11 + a12)b22
    # p6 = (a21 - a11)(b11 + b12)
    # p7 = (a12 - a22)(b21 + b22)
    # c11 = p1 - p4 + p5 - p7
    # c12 = p3 + p5
    # c21 = p2 + p4
    # c22 = p1 - p2 + p3 + p6
    
    # Let's map this to our indices.
    # A[i,j], B[j,k], C[i,k].
    # i,j,k in {0,1}.
    # For a specific t (column block of B), let B_t[j, s] = B[j, 2t+s].
    # We want C[i, 2t+s].
    
    # The 7 products for block t:
    # Let A be the full 2x2 matrix.
    # Let B_t be the 2x2 slice.
    
    # U vectors (coefficients for A):
    # p1: (A00 + A11) -> U1 = [1, 0, 0, 1]^T ? No, U is indexed by i*m+j.
    # i=0, j=0 -> idx 0. i=0, j=1 -> idx 1. i=1, j=0 -> idx 2. i=1, j=1 -> idx 3.
    # A00 is U[0], A01 is U[1], A10 is U[2], A11 is U[3].
    
    # p1 = (A00 + A11) * (B00 + B11)
    # U_p1 = [1, 0, 0, 1]
    # V_p1: B00 is V[2t], B11 is V[2*(2t+1)+1]?
    # B[j, k]. j in 0,1. k in 0,1 (local). Global k = 2t + s.
    # B00 (local) is B[0, 2t]. Global index in V: j*p + k = 0*16 + 2t = 2t.
    # B11 (local) is B[1, 2t+1]. Global index in V: 1*16 + 2t+1 = 16 + 2t + 1.
    # So V_p1 has 1 at 2t and 1 at 16+2t+1.
    
    # p2 = (A10 + A11) * B00
    # U_p2 = [0, 0, 1, 1]
    # V_p2: B00 (local) -> 2t.
    
    # p3 = A00 * (B10 - B11)
    # U_p3 = [1, 0, 0, 0]
    # V_p3: B10 (local) is B[1, 2t]. Index 16+2t. B11 (local) is B[1, 2t+1]. Index 16+2t+1.
    # So 1 at 16+2t, -1 at 16+2t+1.
    
    # p4 = A11 * (B01 - B00)
    # U_p4 = [0, 0, 0, 1]
    # V_p4: B01 (local) is B[0, 2t+1]. Index 2t+1. B00 (local) is B[0, 2t]. Index 2t.
    # So 1 at 2t+1, -1 at 2t.
    
    # p5 = (A00 + A01) * B11
    # U_p5 = [1, 1, 0, 0]
    # V_p5: B11 (local) is B[1, 2t+1]. Index 16+2t+1.
    
    # p6 = (A10 - A00) * (B00 + B01)
    # U_p6 = [-1, 0, 1, 0]
    # V_p6: B00 (2t) + B01 (2t+1). 1 at 2t, 1 at 2t+1.
    
    # p7 = (A01 - A11) * (B10 + B11)
    # U_p7 = [0, 1, 0, -1]
    # V_p7: B10 (16+2t) + B11 (16+2t+1). 1 at 16+2t, 1 at 16+2t+1.
    
    # W vectors (coefficients for C):
    # C indices: C[i, 2t+s]. Global W index: (2t+s)*n + i = (2t+s)*2 + i.
    # c11 (local C[1,1]) corresponds to i=1, s=1. Index 2*(2t+1) + 1 = 4t + 3.
    # c12 (local C[1,0]) corresponds to i=1, s=0. Index 2*(2t+0) + 1 = 4t + 1.
    # c21 (local C[0,1]) corresponds to i=0, s=1. Index 2*(2t+1) + 0 = 4t + 2.
    # c22 (local C[0,0]) corresponds to i=0, s=0. Index 2*(2t+0) + 0 = 4t + 0.
    
    # c11 = p1 - p4 + p5 - p7
    # c12 = p3 + p5
    # c21 = p2 + p4
    # c22 = p1 - p2 + p3 + p6
    
    # Let's construct the columns for each t.
    
    for t in range(8):
        # p1
        col = 7 * t + 0
        U[0][col] += 1; U[3][col] += 1
        V[2*t][col] += 1; V[16 + 2*t + 1][col] += 1
        W[4*t + 3][col] += 1  # c11
        W[4*t + 0][col] += 1  # c22
        
        # p2
        col = 7 * t + 1
        U[2][col] += 1; U[3][col] += 1
        V[2*t][col] += 1
        W[4*t + 1][col] += 1  # c12 (wait, c21? p2+p4=c21. c21 is C[1,0] -> i=1, s=0 -> 4t+1)
        # Correction: c21 = p2 + p4. c21 is C[1,0]?
        # c21 in standard notation is C[1,0] (0-indexed C[1,0]).
        # My mapping: c21 (local C[1,0]) -> i=1, s=0 -> 4t+1.
        # So W[4t+1] gets +1 from p2.
        
        # p3
        col = 7 * t + 2
        U[0][col] += 1
        V[16 + 2*t][col] += 1; V[16 + 2*t + 1][col] += -1
        W[4*t + 1][col] += 1  # c12 (p3+p5=c12. c12 is C[0,1]? No.
        # Standard: c12 = C[0,1]. c21 = C[1,0].
        # c12 = p3 + p5. c12 is C[0,1] -> i=0, s=1 -> 4t+2.
        # So W[4t+2] gets +1 from p3.
        # Let's re-verify standard Strassen output indices.
        # c11 = C[0,0]
        # c12 = C[0,1]
        # c21 = C[1,0]
        # c22 = C[1,1]
        
        # My W index mapping:
        # C[i, 2t+s] -> index (2t+s)*2 + i.
        # C[0,0] (s=0, i=0) -> 0
        # C[0,1] (s=1, i=0) -> 2
        # C[1,0] (s=0, i=1) -> 1
        # C[1,1] (s=1, i=1) -> 3
        # So:
        # c11 (C[0,0]) -> 4t + 0
        # c12 (C[0,1]) -> 4t + 2
        # c21 (C[1,0]) -> 4t + 1
        # c22 (C[1,1]) -> 4t + 3
        
        # Formulas:
        # c11 = p1 - p4 + p5 - p7
        # c12 = p3 + p5
        # c21 = p2 + p4
        # c22 = p1 - p2 + p3 + p6
        
        # Let's restart W assignments with correct indices.
        
        # Reset W for this t (we are adding to global, so just add)
        # p1
        # W[4t+0] += 1
        # W[4t+3] += 1
        
        # p2
        # W[4t+1] += 1
        
        # p3
        # W[4t+2] += 1
        
        # p4
        # W[4t+3] += -1
        # W[4t+1] += 1
        
        # p5
        # W[4t+0] += 1
        # W[4t+2] += 1
        
        # p6
        # W[4t+3] += 1
        
        # p7
        # W[4t+0] += -1
        
        # Let's re-apply carefully.
        
        # p1
        col = 7 * t + 0
        U[0][col] = 1; U[3][col] = 1
        V[2*t][col] = 1; V[16 + 2*t + 1][col] = 1
        W[4*t + 0][col] = 1
        W[4*t + 3][col] = 1
        
        # p2
        col = 7 * t + 1
        U[2][col] = 1; U[3][col] = 1
        V[2*t][col] = 1
        W[4*t + 1][col] = 1
        
        # p3
        col = 7 * t + 2
        U[0][col] = 1
        V[16 + 2*t][col] = 1; V[16 + 2*t + 1][col] = -1
        W[4*t + 2][col] = 1
        
        # p4
        col = 7 * t + 3
        U[3][col] = 1
        V[2*t + 1][col] = 1; V[2*t][col] = -1
        W[4*t + 3][col] = -1
        W[4*t + 1][col] = 1
        
        # p5
        col = 7 * t + 4
        U[0][col] = 1; U[1][col] = 1
        V[16 + 2*t + 1][col] = 1
        W[4*t + 0][col] = 1
        W[4*t + 2][col] = 1
        
        # p6
        col = 7 * t + 5
        U[2][col] = 1; U[0][col] = -1
        V[2*t][col] = 1; V[2*t + 1][col] = 1
        W[4*t + 3][col] = 1
        
        # p7
        col = 7 * t + 6
        U[1][col] = 1; U[3][col] = -1
        V[16 + 2*t][col] = 1; V[16 + 2*t + 1][col] = 1
        W[4*t + 0][col] = -1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
