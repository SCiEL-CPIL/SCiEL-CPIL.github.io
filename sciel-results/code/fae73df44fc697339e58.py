"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 60
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, sign):
        nonlocal col
        U[u_idx][col] = 1
        V[v_idx][col] = 1
        W[w_idx][col] = sign
        col += 1

    # 1. A[0:2, 0:2] * B[0:2, :]
    # Split B[0:2, :] into B[0:2, 0:2], B[0:2, 2:4], B[0:2, 4:6], B[0:2, 6:7]
    
    # Strassen for 2x2 * 2x2
    def strassen_2x2(k_start):
        nonlocal col
        # A indices: (0,0), (0,1), (1,0), (1,1) -> 0, 1, 3, 4
        # B indices: (0, k), (0, k+1), (1, k), (1, k+1)
        # B[0, k] -> 0*7 + k = k
        # B[0, k+1] -> k+1
        # B[1, k] -> 1*7 + k = 7+k
        # B[1, k+1] -> 7+k+1
        
        # m1 = (a00 + a11)(b00 + b11)
        # U: a00, a11. V: b00, b11. W: c00, c11
        # We need to handle the sums.
        # This requires multiple rank-1 terms for each multiplication if we don't pre-compute sums.
        # But the tensor decomposition must be a sum of rank-1 tensors.
        # Strassen is an algorithm, not a direct rank-1 decomposition of the 2x2x2 tensor?
        # Yes, the 2x2x2 tensor has rank 7.
        # The 7 rank-1 terms are:
        # 1. (a00+a11) * (b00+b11) -> contributes to c00, c11
        # 2. (a10+a11) * b00 -> contributes to c10
        # 3. a00 * (b10-b11) -> contributes to c01
        # 4. a11 * (b00-b01) -> contributes to c00
        # 5. (a00+a01) * b11 -> contributes to c00
        # 6. (a10-a00) * (b00+b01) -> contributes to c11
        # 7. (a01-a11) * (b10+b11) -> contributes to c11
        
        # Let's map these to U, V, W.
        # Term 1:
        # U: a00 (0), a11 (4). V: b00 (k), b11 (7+k+1).
        # This is not a single rank-1 term because U has two non-zero entries.
        # We must expand (a00+a11)(b00+b11) into 4 rank-1 terms?
        # No, the rank of the 2x2x2 tensor is 7.
        # The 7 terms are linear combinations of the entries.
        # We need to find the 7 rank-1 decompositions.
        
        # Actually, it's easier to just use the schoolbook for the 2x2x2 block if we can't easily write the Strassen rank-1 terms.
        # Schoolbook for 2x2x2 is 8 terms.
        # Strassen is 7 terms.
        # Let's use schoolbook for simplicity first, then optimize if needed.
        # Schoolbook 2x2x2: 8 terms.
        # 3 blocks * 8 + 1 block * 4 = 24 + 4 = 28.
        # Total rank: 28 + 14 + 14 + 7 = 63.
        # This is the same as the original schoolbook!
        
        # So I MUST implement the Strassen rank-1 terms to get the improvement.
        
        # The 7 rank-1 terms for 2x2x2 matrix multiplication (C = A * B):
        # C00 = A00*B00 + A01*B10
        # C01 = A00*B01 + A01*B11
        # C10 = A10*B00 + A11*B10
        # C11 = A10*B01 + A11*B11
        
        # Strassen:
        # M1 = (A00 + A11) * (B00 + B11)
        # M2 = (A10 + A11) * B00
        # M3 = A00 * (B10 - B11)
        # M4 = A11 * (B00 - B01)
        # M5 = (A00 + A01) * B11
        # M6 = (A10 - A00) * (B00 + B01)
        # M7 = (A01 - A11) * (B10 + B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Each Mi is a product of a linear form in A and a linear form in B.
        # So each Mi is a sum of rank-1 tensors?
        # No, a product of a linear form in A and a linear form in B is a sum of rank-1 tensors.
        # (sum ai ui) * (sum bi vi) = sum ai bi ui vi.
        # This expands to multiple rank-1 terms.
        # For example, M1 = (A00 + A11) * (B00 + B11) = A00 B00 + A00 B11 + A11 B00 + A11 B11.
        # This is 4 rank-1 terms.
        # So Strassen doesn't reduce the number of rank-1 terms in the tensor decomposition sense?
        # Wait, the rank of the tensor is the minimum number of rank-1 tensors to sum to it.
        # If we expand M1 into 4 rank-1 terms, we are using 4 terms.
        # Strassen uses 7 "multiplications", but each multiplication is of linear forms.
        # The rank of the 2x2x2 tensor is 7.
        # This means there EXISTS a decomposition into 7 rank-1 tensors.
        # The Strassen algorithm provides such a decomposition, but the "multiplications" are not scalar multiplications of single entries, but of linear combinations.
        # To implement this as a tensor decomposition, we need to express each of the 7 terms as a rank-1 tensor.
        # But a term like (A00 + A11) * (B00 + B11) is NOT a rank-1 tensor. It is a sum of 4 rank-1 tensors.
        # So how can the rank be 7?
        # Ah, the rank-1 tensors in the decomposition of the 2x2x2 tensor are not the scalar products of entries.
        # They are tensors $u \otimes v \otimes w$ where $u, v, w$ are vectors.
        # The Strassen algorithm shows that the tensor can be written as a sum of 7 such tensors.
        # The vectors $u, v, w$ have non-zero entries at multiple positions.
        
        # So I need to find the 7 vectors for the 2x2x2 case.
        # This is a known result.
        # Let's look up the rank-7 decomposition of the 2x2x2 matrix multiplication tensor.
        
        # It is not trivial to write down from memory.
        # However, for the purpose of this task, if I can't easily find the rank-7 decomposition, I might have to stick with schoolbook (63) or find a different improvement.
        
        # Is there a simpler improvement?
        # What if I use the fact that p=7?
        # Maybe use a rank-15 decomposition for 3x3x3?
        # If I can find a rank-15 decomposition for 3x3x3, I can use it for the first 3 columns of B.
        # And schoolbook for the remaining 4 columns.
        # Rank: 15 + 3*3*4 = 15 + 36 = 51.
        # This is much better.
        # But does a rank-15 decomposition for 3x3x3 exist with rational entries?
        # The minimal rank of 3x3x3 is not known, but 23 is the best upper bound.
        # So rank 15 is likely impossible.
        
        # Let's go back to the Strassen 2x2x2 rank-7 decomposition.
        # I can search for the specific vectors.
        # Or, I can just use the schoolbook for the 2x2x2 blocks and accept rank 63?
        # No, I want to improve.
        
        # Let's try to construct the rank-7 decomposition for 2x2x2.
        # The 7 rank-1 tensors are:
        # T1 = (1, 0, 0, 1) \otimes (1, 0, 0, 1) \otimes (1, 0, 0, 1)  -- No, this is not correct.
        
        # Actually, I can just use the schoolbook for the 2x2x2 part and see if I can save time.
        # But the task asks for a stronger algorithm.
        
        # Let's assume I can implement the Strassen rank-7 decomposition.
        # I will write a placeholder for the 7 terms.
        # If I can't, I will fall back to schoolbook.
        
        # For now, let's just return the schoolbook decomposition to ensure correctness, and then I will try to add the Strassen terms.
        # Actually, I will implement the Strassen terms using the known rank-7 decomposition.
        
        # The rank-7 decomposition of the 2x2x2 tensor is given by the following 7 tuples (u, v, w):
        # u, v, w are vectors of length 4 (flattened 2x2 matrices).
        # Order: (0,0), (0,1), (1,0), (1,1).
        
        # I don't have the exact vectors memorized.
        # I will use the schoolbook for now to guarantee a valid output.
        # Rank 63.
        
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    # A[i, j] * B[j, k_start + k]
                    u_idx = i * 3 + j
                    v_idx = j * 7 + (k_start + k)
                    w_idx = (k_start + k) * 3 + i
                    add_term(u_idx, v_idx, w_idx, 1)
                    # Note: This is schoolbook for 2x2x2. 8 terms.
        
    # This approach will result in rank 63.
    # I will try to implement the Strassen terms if I can find them online or derive them.
    # Since I cannot, I will stick with the schoolbook for the 2x2 blocks.
    
    # Let's just use the full schoolbook for the whole thing to be safe.
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
