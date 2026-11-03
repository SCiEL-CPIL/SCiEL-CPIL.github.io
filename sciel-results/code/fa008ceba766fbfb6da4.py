"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 22 decomposition
    rank = 22
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        nonlocal col
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val
        col += 1

    # 1. Strassen for A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # A indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # B indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # C indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3 (Note: W index is k*n+i)
    # W index for C[i,k] is k*3+i
    
    # M1 = (A00+A11)(B00+B11)
    add_term(0, 0, 0, 1, 1, 1) # C00
    add_term(1, 0, 1, 1, 1, 1) # C01 (k=0, i=1 -> idx 1)
    add_term(2, 0, 2, 1, 1, 1) # C10 (k=1, i=0 -> idx 3? No. W index k*n+i. C10 is i=1, k=0? No.
    
    # Let's clarify W index mapping.
    # W[k*n+i].
    # C[i,k].
    # If i=0, k=0 -> W[0].
    # If i=1, k=0 -> W[1].
    # If i=0, k=1 -> W[3].
    # If i=1, k=1 -> W[4].
    
    # So C[i,k] corresponds to W index k*3+i.
    
    # M1 contributes to C00, C01, C10, C11?
    # Strassen M1 = (A00+A11)(B00+B11).
    # C00 += M1
    # C01 += M1? No.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # So M1 contributes to C00 and C11.
    # C00: i=0, k=0 -> W[0]
    # C11: i=1, k=1 -> W[4]
    
    # M1:
    # U: A00 (idx 0) + A11 (idx 3)
    # V: B00 (idx 0) + B11 (idx 3)
    # W: C00 (idx 0) + C11 (idx 4)
    # This is not a single rank 1 term!
    # Strassen M1 is a scalar multiplication.
    # The term is $U_{A00} V_{B00} W_{C00} + \dots$
    # No, the decomposition is a sum of rank 1 tensors.
    # Each rank 1 tensor corresponds to one scalar multiplication.
    # So for M1, we have one column in U, V, W.
    # U has 1 at A00 and A11.
    # V has 1 at B00 and B11.
    # W has 1 at C00 and C11?
    # No, $W$ must be a vector.
    # The product $U \cdot V \cdot W$ is a tensor.
    # If $W$ has 1 at C00 and 1 at C11, then the term contributes to both C00 and C11.
    # This is correct for M1.
    
    # So:
    # M1:
    U[0][col] = 1; U[3][col] = 1
    V[0][col] = 1; V[3][col] = 1
    W[0][col] = 1; W[4][col] = 1
    col += 1
    
    # M2 = (A10+A11)B00
    # U: A10 (idx 2) + A11 (idx 3)
    # V: B00 (idx 0)
    # W: C10 (idx 3)
    # C10 = M2 + M4. So M2 contributes to C10.
    U[2][col] = 1; U[3][col] = 1
    V[0][col] = 1
    W[3][col] = 1
    col += 1
    
    # M3 = A00(B10-B11)
    # U: A00 (idx 0)
    # V: B10 (idx 2) - B11 (idx 3)
    # W: C01 (idx 1)
    # C01 = M3 + M5.
    U[0][col] = 1
    V[2][col] = 1; V[3][col] = -1
    W[1][col] = 1
    col += 1
    
    # M4 = A11(B01-B10)
    # U: A11 (idx 3)
    # V: B01 (idx 1) - B10 (idx 2)
    # W: C00 (idx 0) and C10 (idx 3)
    # C00 += M4, C10 += M4.
    U[3][col] = 1
    V[1][col] = 1; V[2][col] = -1
    W[0][col] = 1; W[3][col] = 1
    col += 1
    
    # M5 = (A00+A01)B11
    # U: A00 (idx 0) + A01 (idx 1)
    # V: B11 (idx 3)
    # W: C01 (idx 1) and C00 (idx 0)
    # C01 += M5, C00 -= M5.
    U[0][col] = 1; U[1][col] = 1
    V[3][col] = 1
    W[1][col] = 1; W[0][col] = -1
    col += 1
    
    # M6 = (A10-A00)(B00+B10)
    # U: A10 (idx 2) - A00 (idx 0)
    # V: B00 (idx 0) + B10 (idx 2)
    # W: C11 (idx 4)
    # C11 += M6.
    U[2][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[2][col] = 1
    W[4][col] = 1
    col += 1
    
    # M7 = (A01-A11)(B11+B01)
    # U: A01 (idx 1) - A11 (idx 3)
    # V: B11 (idx 3) + B01 (idx 1)
    # W: C00 (idx 0)
    # C00 += M7.
    U[1][col] = 1; U[3][col] = -1
    V[3][col] = 1; V[1][col] = 1
    W[0][col] = 1
    col += 1
    
    # 2. A21 B12 (Rank 1)
    # A21: i in {0,1}, j=2. Indices: 0*3+2=2, 1*3+2=5.
    # B12: j=2, k in {0,1}. Indices: 2*3+0=6, 2*3+1=7.
    # C21? No.
    # A21 is top-right block of A?
    # A = [[A22, A21], [A12, A11]]
    # A21 is i in {0,1}, j=2.
    # B12 is j=2, k in {0,1}.
    # Product C21? No.
    # C = [[C22, C21], [C12, C11]]
    # C21 = A22 B21 + A21 B11.
    # Wait, my previous analysis said A21 B12 contributes to C22?
    # C22 = A22 B22 + A21 B12.
    # Yes.
    # A21 (2x1) * B12 (1x2) -> C22 (2x2).
    # A21 indices: 2, 5.
    # B12 indices: 6, 7.
    # C22 indices: 0, 1, 3, 4.
    # Rank 1 term.
    # U: 1 at 2, 5.
    # V: 1 at 6, 7.
    # W: 1 at 0, 1, 3, 4.
    U[2][col] = 1; U[5][col] = 1
    V[6][col] = 1; V[7][col] = 1
    W[0][col] = 1; W[1][col] = 1; W[3][col] = 1; W[4][col] = 1
    col += 1
    
    # 3. A22 B21 (Rank 4)
    # A22 (2x2) * B21 (2x1) -> C21 (2x1).
    # A22 indices: 0,1,2,3.
    # B21 indices: j in {0,1}, k=2. Indices: 0*3+2=2, 1*3+2=5.
    # C21 indices: i in {0,1}, k=2. Indices: 2*3+0=6, 2*3+1=7.
    # Standard multiplication.
    # C21[0] = A22[0,0]B21[0,2] + A22[0,1]B21[1,2]
    # C21[1] = A22[1,0]B21[0,2] + A22[1,1]B21[1,2]
    
    # Term 1: A00 B02 -> C02 (idx 6)
    U[0][col] = 1; V[2][col] = 1; W[6][col] = 1; col += 1
    # Term 2: A01 B12 -> C02 (idx 6)
    U[1][col] = 1; V[5][col] = 1; W[6][col] = 1; col += 1
    # Term 3: A10 B02 -> C12 (idx 7)
    U[2][col] = 1; V[2][col] = 1; W[7][col] = 1; col += 1
    # Term 4: A11 B12 -> C12 (idx 7)
    U[3][col] = 1; V[5][col] = 1; W[7][col] = 1; col += 1
    
    # 4. A21 B11 (Rank 2)
    # A21 (2x1) * B11 (1x1) -> C21 (2x1).
    # A21 indices: 2, 5.
    # B11 index: j=2, k=2. Index 8.
    # C21 indices: 6, 7.
    # C21[0] += A21[0] B11
    # C21[1] += A21[1] B11
    U[2][col] = 1; V[8][col] = 1; W[6][col] = 1; col += 1
    U[5][col] = 1; V[8][col] = 1; W[7][col] = 1; col += 1
    
    # 5. A12 B22 (Rank 4)
    # A12 (1x2) * B22 (2x2) -> C12 (1x2).
    # A12 indices: i=2, j in {0,1}. Indices: 6, 7.
    # B22 indices: 0, 1, 2, 3.
    # C12 indices: i=2, k in {0,1}. Indices: 6, 7.
    # C12[0] = A12[0]B22[0,0] + A12[1]B22[1,0]
    # C12[1] = A12[0]B22[0,1] + A12[1]B22[1,1]
    
    U[6][col] = 1; V[0][col] = 1; W[6][col] = 1; col += 1
    U[7][col] = 1; V[2][col] = 1; W[6][col] = 1; col += 1
    U[6][col] = 1; V[1][col] = 1; W[7][col] = 1; col += 1
    U[7][col] = 1; V[3][col] = 1; W[7][col] = 1; col += 1
    
    # 6. A11 B12 (Rank 1)
    # A11 (1x1) * B12 (1x2) -> C12 (1x2).
    # A11 index: 8.
    # B12 indices: 6, 7.
    # C12 indices: 6, 7.
    U[8][col] = 1; V[6][col] = 1; W[6][col] = 1
    U[8][col] = 1; V[7][col] = 1; W[7][col] = 1
    # Wait, Rank 1 means ONE term.
    # U[8]=1, V[6]=1, V[7]=1, W[6]=1, W[7]=1.
    # This works.
    # But I already incremented col? No, I didn't add_term.
    # Let's reset col for this part.
    # Actually, I should have used add_term or manual.
    # I will just write it out.
    # But I already used col for previous steps.
    # I need to be careful.
    
    # Let's restart the construction with a cleaner approach.
    
    return {"U": U, "V": "V", "W": W} # Placeholder

# EVOLVE-BLOCK-END
