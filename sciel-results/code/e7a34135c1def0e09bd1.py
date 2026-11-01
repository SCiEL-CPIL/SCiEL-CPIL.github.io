import time

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target rank is 26
    rank = 26
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term: U[i,j], V[j,k], W[k,i]
    def add_term(i, j, k, u_val=1, v_val=1, w_val=1):
        nonlocal col
        if col >= rank:
            raise Exception("Rank overflow")
        U[i * m + j][col] = u_val
        V[j * p + k][col] = v_val
        W[k * n + i][col] = w_val
        col += 1

    # 1. C00 block (1x1 output, 1x3x1 input structure)
    # C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0] + A[0,2]B[2,0]
    add_term(0, 0, 0) # A00 B00 -> C00
    add_term(0, 1, 0) # A01 B10 -> C00
    add_term(0, 2, 0) # A02 B20 -> C00
    
    # 2. C01 block (1x2 output)
    # C[0,1] = A[0,0]B[0,1] + A[0,1]B[1,1] + A[0,2]B[2,1]
    # C[0,2] = A[0,0]B[0,2] + A[0,1]B[1,2] + A[0,2]B[2,2]
    add_term(0, 0, 1) # A00 B01 -> C01
    add_term(0, 1, 1) # A01 B11 -> C01
    add_term(0, 2, 1) # A02 B21 -> C01
    
    add_term(0, 0, 2) # A00 B02 -> C02
    add_term(0, 1, 2) # A01 B12 -> C02
    add_term(0, 2, 2) # A02 B22 -> C02
    
    # 3. C10 block (2x1 output)
    # C[1,0] = A[1,0]B[0,0] + A[1,1]B[1,0] + A[1,2]B[2,0]
    # C[2,0] = A[2,0]B[0,0] + A[2,1]B[1,0] + A[2,2]B[2,0]
    add_term(1, 0, 0) # A10 B00 -> C10
    add_term(1, 1, 0) # A11 B10 -> C10
    add_term(1, 2, 0) # A12 B20 -> C10
    
    add_term(2, 0, 0) # A20 B00 -> C20
    add_term(2, 1, 0) # A21 B10 -> C20
    add_term(2, 2, 0) # A22 B20 -> C20
    
    # 4. C11 block (2x2 output)
    # Part 1: A[1:3, 0] B[0, 1:3] (2x1 * 1x2) Rank 4
    add_term(1, 0, 1) # A10 B01 -> C11
    add_term(1, 0, 2) # A10 B02 -> C12
    add_term(2, 0, 1) # A20 B01 -> C21
    add_term(2, 0, 2) # A20 B02 -> C22
    
    # Part 2: A[1:3, 1:3] B[1:3, 1:3] (2x2 * 2x2) Strassen Rank 7
    # Local indices 0,1 map to Global 1,2
    # A11_local = A[1,1], A[1,2]
    # A12_local = A[1,1]... wait.
    # Global:
    # A11 (local 0,0) = A[1,1]
    # A12 (local 0,1) = A[1,2]
    # A21 (local 1,0) = A[2,1]
    # A22 (local 1,1) = A[2,2]
    
    # B11 (local 0,0) = B[1,1]
    # B12 (local 0,1) = B[1,2]
    # B21 (local 1,0) = B[2,1]
    # B22 (local 1,1) = B[2,2]
    
    # C11 (local 0,0) = C[1,1]
    # C12 (local 0,1) = C[1,2]
    # C21 (local 1,0) = C[2,1]
    # C22 (local 1,1) = C[2,2]
    
    # M1 = (A11 + A22) * (B11 + B22)
    # We cannot do sums in a single term. We must expand.
    # (A11 + A22) is a sum of 2 vectors. (B11 + B22) is sum of 2 vectors.
    # Product is 4 terms.
    # This would make Strassen rank 4+4+... which is not 7.
    
    # Wait, Strassen works on the *matrices* A and B.
    # The tensor decomposition requires linear combinations in U, V, W.
    # U, V, W entries can be -1, 0, 1.
    # So we CAN have linear combinations!
    
    # Let's implement Strassen properly with linear combinations.
    
    # M1: (A11 + A22) * (B11 + B22)
    # U terms: A11 and A22.
    # V terms: B11 and B22.
    # This is NOT a single term. A single term is $u \otimes v \otimes w$.
    # $u$ is a vector. $u$ can be $e_{1,1} + e_{2,2}$.
    # $v$ can be $e_{1,1} + e_{2,2}$.
    # $w$ is $e_{1,1}$.
    # Term: $(e_{1,1} + e_{2,2}) \otimes (e_{1,1} + e_{2,2}) \otimes e_{1,1}$
    # This contributes to C11.
    # It does NOT contribute to other C's in this specific local block calculation?
    # Wait, Strassen output:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # So each M_i is a product of two linear combinations of A and B entries.
    # And each M_i contributes to multiple C entries with coefficients.
    
    # Let's define the 7 terms for the 2x2 block (indices 1,2).
    
    # Term 7 (M1):
    # U: A[1,1] + A[2,2] -> U[1*3+1] += 1, U[2*3+2] += 1
    # V: B[1,1] + B[2,2] -> V[1*3+1] += 1, V[2*3+2] += 1
    # W: C[1,1] += 1, C[2,2] += 1
    
    def add_strassen_term(u_rows, u_cols, v_rows, v_cols, w_rows, w_cols, w_coeffs):
        nonlocal col
        if col >= rank:
            raise Exception("Rank overflow")
        for i in u_rows:
            for j in u_cols:
                U[i * m + j][col] = 1
        for i in v_rows:
            for j in v_cols:
                V[i * p + j][col] = 1
        for i in w_rows:
            for j in w_cols:
                W[j * n + i][col] = w_coeffs[i] # Note: W index is k*n+i? 
                # Task says W[k*n+i, r]. 
                # My W matrix is (p*n, rank). 
                # Row index is k*n + i.
                # So W[k*n + i][col] = coeff.
        col += 1

    # Re-verify W indexing.
    # W[k*n+i, r].
    # In my add_term: W[k*n+i][col] = w_val. Correct.
    
    # Strassen for indices 1,2 (0-based global 1,2)
    
    # M1 = (A11 + A22)(B11 + B22)
    # A11 is A[1,1], A22 is A[2,2].
    # B11 is B[1,1], B22 is B[2,2].
    # Contributes to C11 (coeff 1) and C22 (coeff 1).
    add_strassen_term([1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2], {1: 1, 2: 1})
    
    # M2 = (A21 + A22)(B21 - B11)
    # A21 is A[2,1], A22 is A[2,2].
    # B21 is B[2,1], B11 is B[1,1].
    # Contributes to C21 (coeff 1) and C22 (coeff -1).
    add_strassen_term([2], [1, 2], [2, 1], [1, 1], [2, 2], [1, 2], {2: 1, 2: -1})
    # Wait, V terms: B[2,1] and B[1,1]. 
    # V[2*3+1] += 1, V[1*3+1] += -1.
    # My helper does += 1. I need to handle signs.
    
    # Let's rewrite add_strassen_term to handle signs.
    
    # I will discard the helper and write explicit code for the 7 terms to be safe.
    
    # M1: (A11+A22)(B11+B22) -> C11+=1, C22+=1
    U[1*3+1][col] = 1; U[2*3+2][col] = 1
    V[1*3+1][col] = 1; V[2*3+2][col] = 1
    W[1*3+1][col] = 1; W[2*3+2][col] = 1
    col += 1
    
    # M2: (A21+A22)(B21-B11) -> C21+=1, C22+=-1
    U[2*3+1][col] = 1; U[2*3+2][col] = 1
    V[2*3+1][col] = 1; V[1*3+1][col] = -1
    W[2*3+1][col] = 1; W[2*3+2][col] = -1
    col += 1
    
    # M3: (A11+A12)(B22) -> C12+=1, C22+=1
    U[1*3+1][col] = 1; U[1*3+2][col] = 1
    V[2*3+2][col] = 1
    W[1*3+2][col] = 1; W[2*3+2][col] = 1
    col += 1
    
    # M4: (A22)(B22-B11) -> C11+=1, C21+=1
    U[2*3+2][col] = 1
    V[2*3+2][col] = 1; V[1*3+1][col] = -1
    W[1*3+1][col] = 1; W[2*3+1][col] = 1
    col += 1
    
    # M5: (A11+A21)(B11+B12) -> C11+=-1, C12+=1
    U[1*3+1][col] = 1; U[2*3+1][col] = 1
    V[1*3+1][col] = 1; V[1*3+2][col] = 1
    W[1*3+1][col] = -1; W[1*3+2][col] = 1
    col += 1
    
    # M6: (A22-A11)(B11+B22) -> C22+=1
    U[2*3+2][col] = 1; U[1*3+1][col] = -1
    V[1*3+1][col] = 1; V[2*3+2][col] = 1
    W[2*3+2][col] = 1
    col += 1
    
    # M7: (A12-A22)(B21+B22) -> C11+=1
    U[1*3+2][col] = 1; U[2*3+2][col] = -1
    V[2*3+1][col] = 1; V[2*3+2][col] = 1
    W[1*3+1][col] = 1
    col += 1
    
    # Total terms: 3 (C00) + 6 (C01) + 6 (C10) + 4 (C11 part1) + 7 (C11 part2) = 26.
    
    if col != rank:
        raise Exception(f"Rank mismatch: {col} != {rank}")
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    res = construct()
    print(f"Rank: {len(res['U'][0])}")
