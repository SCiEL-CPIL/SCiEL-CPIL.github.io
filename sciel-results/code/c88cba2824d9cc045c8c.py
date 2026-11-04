import numpy as np
import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145.0

    # Build target tensor
    dim_u = n * m  # 16
    dim_v = m * p  # 28
    dim_w = p * n  # 28

    T = np.zeros((dim_u, dim_v, dim_w), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1

    schoolbook_rank = n * m * p  # 112
    best_result = None
    best_rank = schoolbook_rank

    # Structural approach: Decompose 4x4x7 into 4 independent 2x2x7 sub-problems
    # by partitioning A and C into 2x2 blocks.
    #
    # A = [[A0, A1], [A2, A3]] where each block is 2x2
    # B is 4x7, partitioned as B = [[B0], [B1], [B2], [B3]] where each Bi is 2x7
    # C = A*B = [[C0], [C1]] where C0 = A0*B0 + A1*B1, C1 = A2*B2 + A3*B3
    #
    # This gives us 4 independent 2x2x7 sub-problems:
    # C0 = A0*B0 + A1*B1  (2x2 * 2x7 = 2x7)
    # C1 = A2*B2 + A3*B3  (2x2 * 2x7 = 2x7)
    #
    # Wait, that's not right. Let me reconsider.
    #
    # Actually, for A (4x4) * B (4x7) = C (4x7):
    # A = [[A00, A01], [A10, A11]] (each 2x2)
    # B = [[B00, B01, B02, B03, B04, B05, B06],
    #      [B10, B11, B12, B13, B14, B15, B16]] (each row is 1x7, but grouped as 2x7)
    #
    # Let's partition B into two 2x7 blocks:
    # B = [B_top; B_bottom] where B_top is 2x7 (rows 0,1) and B_bottom is 2x7 (rows 2,3)
    #
    # C = A*B = [A00*B_top + A01*B_bottom; A10*B_top + A11*B_bottom]
    #
    # So we have 4 sub-problems, each computing a 2x2 * 2x7 = 2x7 product:
    # C_top = A00*B_top + A01*B_bottom
    # C_bottom = A10*B_top + A11*B_bottom
    #
    # Each sub-problem is: given X (2x2), Y (2x7), compute Z = X*Y (2x7)
    #
    # For a 2x2 * 2x7 problem, we can use Strassen-like decomposition.
    # The tensor for 2x2 * 2x7 has rank at most 7*7 = 49 using Kronecker product
    # of Strassen (rank 7) with identity on the 7 columns.
    #
    # But we can do better! For 2x2 * 2x7:
    # Let X = [[x11, x12], [x21, x22]], Y = [[y11,...,y17], [y21,...,y27]]
    # Z = X*Y, so Z[0,k] = x11*y1k + x12*y2k, Z[1,k] = x21*y1k + x22*y2k
    #
    # Using Strassen's 7 products for each column independently would give rank 7*7=49.
    # But we can share the Strassen products across columns!
    #
    # Actually, the 2x2 * 2x7 tensor is the Kronecker product of the 2x2 * 2x2 tensor
    # (rank 7) with the identity on 7 columns... no wait.
    #
    # Let me think differently. The 2x2 * 2x7 multiplication tensor:
    # T[a, b, c] = 1 if a = i*2+j, b = j*7+k, c = k*2+i for i,j in {0,1}, k in {0..6}
    #
    # This is equivalent to 7 independent 2x2 * 2x1 problems? No.
    #
    # Actually, for each output column c = k*2+i, we need:
    # Z[0, k] = x11*y1k + x12*y2k  (uses A indices 0,1 and B indices k, 7+k)
    # Z[1, k] = x21*y1k + x22*y2k  (uses A indices 2,3 and B indices k, 7+k)
    #
    # So the tensor decomposes as a direct sum over k of 2x2 * 2x1 sub-tensors.
    # Each 2x2 * 2x1 sub-tensor has rank at most 4 (schoolbook) or 3 (Strassen-like).
    #
    # For 2x2 * 2x1: X (2x2) * y (2x1) = z (2x1)
    # z0 = x11*y0 + x12*y1
    # z1 = x21*y0 + x22*y1
    #
    # This is just bilinear, rank 4 in general. Can we do better?
    # The tensor has 4*2*2 = 16 entries, with 4 non-zero entries.
    # Rank is at most 4. Is it exactly 4? Likely yes for generic case.
    #
    # So 7 columns * rank 4 = rank 28 for the 2x2 * 2x7 problem.
    #
    # Total for 4x4 * 4x7: 4 sub-problems * rank 28 = rank 112. That's just schoolbook!
    #
    # Hmm, that's not an improvement. Let me reconsider.
    #
    # Actually, I need to be more careful. The 4x4 * 4x7 problem doesn't decompose
    # into 4 independent 2x2 * 2x7 problems in a way that reduces rank, because
    # the Strassen decomposition for 2x2 * 2x2 already has rank 7, and we need
    # to handle 7 columns.
    #
    # Let me try a different structural approach: use the known result that
    # rank(M_{4,4,7}) <= rank(M_{4,4,4}) + rank(M_{4,4,3})
    #
    # rank(M_{4,4,4}) <= 49 (Strassen Kronecker)
    # rank(M_{4,4,3}) <= 4*4*3 = 48 (schoolbook) or better
    #
    # Total: 49 + 48 = 97
    #
    # Can we do better on M_{4,4,3}? 
    # M_{4,4,3}: 4x4 * 4x3 = 4x3
    # Using Strassen on the 4x4 part for the first 2 columns... no, that doesn't work directly.
    #
    # Alternative: use the fact that M_{4,4,3} can be decomposed using
    # rank(M_{4,4,3}) <= rank(M_{2,2,3}) * 4 (Kronecker-like)
    #
    # What is rank(M_{2,2,3})? 2x2 * 2x3 = 2x3
    # z00 = a11*b10 + a12*b20
    # z01 = a11*b11 + a12*b21
    # z02 = a11*b12 + a12*b22
    # z10 = a21*b10 + a22*b20
    # z11 = a21*b11 + a22*b21
    # z12 = a21*b12 + a22*b22
    #
    # This is 3 independent 2x2 * 2x1 problems, each rank 4, total rank 12.
    # Or using Strassen for each: rank 7*3 = 21? No, that's worse.
    #
    # Actually for 2x2 * 2x1, the minimum rank is 4 (it's a generic bilinear map).
    # So rank(M_{2,2,3}) <= 12.
    #
    # Then rank(M_{4,4,3}) <= 4 * 12 = 48. Same as schoolbook.
    #
    # Let me try yet another approach: use the Bini-Capovani-de Falco-Ronco (BCDR) algorithm
    # or other fast matrix multiplication algorithms.
    #
    # Actually, let's just use the Strassen-based approach that gives rank 97,
    # and then try to improve the right part (4x4x3) using a more careful search.
    
    # Build the 4x4x4 Strassen decomposition and embed it
    U_2, V_2, W_2 = _strassen_2x2x2()
    
    # Kronecker product for 4x4x4: rank 49
    U_444 = np.zeros((16, 49))
    V_444_std = np.zeros((16, 49))  # standard 4x4 indexing: j*4+k
    W_444_std = np.zeros((16, 49))  # standard 4x4 indexing: k*4+i
    
    for r1 in range(7):
        for r2 in range(7):
            idx = r1 * 7 + r2
            U_444[:, idx] = np.kron(U_2[:, r1], U_2[:, r2])
            V_444_std[:, idx] = np.kron(V_2[:, r1], V_2[:, r2])
            W_444_std[:, idx] = np.kron(W_2[:, r1], W_2[:, r2])
    
    # Now embed into full tensor:
    # In the full problem, for the "left" 4 columns (k=0..3):
    # V index = j*7 + k, W index = k*4 + i
    # In the standard 4x4x4: V index = j*4 + k, W index = k*4 + i
    # So W indices match! But V indices differ: j*7+k vs j*4+k
    #
    # We need to create V_444_full of shape (28, 49) where:
    # V_444_full[j*7+k, idx] = V_444_std[j*4+k, idx] for k=0..3
    V_444_full = np.zeros((28, 49))
    for j in range(4):
        for k in range(4):
            V_444_full[j*7+k, :] = V_444_std[j*4+k, :]
    
    W_444_full = np.zeros((28, 49))
    for k in range(4):
        for i in range(4):
            W_444_full[k*4+i, :] = W_444_std[k*4+i, :]
    
    # Verify the 4x4x4 part
    recon_444 = np.einsum('ak,bk,ck->abc', U_444, V_444_full, W_444_full)
    T_444_part = np.zeros((16, 28, 28), dtype=np.float64)
    for i in range(4):
        for j in range(4):
            for k in range(4):
                a = i * 4 + j
                b = j * 7 + k
                c = k * 4 + i
                T_444_part[a, b, c] = 1
    
    if not np.allclose(recon_444, T_444_part, atol=1e-10):
        # Fallback: try greedy peeling
        result = _greedy_peel(T, dim_u, dim_v, dim_w, deadline, schoolbook_rank)
        if result is not None:
            return result
        return _schoolbook(n, m, p)
    
    # Now handle the right part: k=4,5,6 (3 columns)
    # For these, V index = j*7+k, W index = k*4+i
    # Schoolbook for this part: rank 4*4*3 = 48
    # Total so far: 49 + 48 = 97
    
    U_right = np.zeros((16, 48))
    V_right = np.zeros((28, 48))
    W_right = np.zeros((28, 48))
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(4, 7):  # k = 4, 5, 6
                U_right[i*4+j, col] = 1
                V_right[j*7+k, col] = 1
                W_right[k*4+i, col] = 1
                col += 1
    
    # Combine: total rank = 49 + 48 = 97
    U_total = np.hstack([U_444, U_right])  # (16, 97)
    V_total = np.hstack([V_444_full, V_right])  # (28, 97)
    W_total = np.hstack([W_444_full, W_right])  # (28, 97)
    
    # Verify full reconstruction
    recon_full = np.einsum('ak,bk,ck->abc', U_total, V_total, W_total)
    if np.allclose(recon_full, T, atol=1e-10):
        best_rank = 97
        best_result = {
            "U": U_total.tolist(),
            "V": V_total.tolist(),
            "W": W_total.tolist()
        }
    
    # Try to improve the right part (4x4x3) using structured search
    if time.time() < deadline - 5.0:
        T_right = np.zeros((16, 28, 28), dtype=np.float64)
        for i in range(4):
            for j in range(4):
                for k in range(4, 7):
                    a = i * 4 + j
                    b = j * 7 + k
                    c = k * 4 + i
                    T_right[a, b, c] = 1
        
        # Try structured decomposition for 4x4x3
        right_structured = _structured_4x4x3()
        if right_structured is not None:
            r_struct = len(right_structured["U"][0])
            U_r = np.array(right_structured["U"]).T
            V_r = np.array(right_structured["V"]).T
            W_r = np.array(right_structured["W"]).T
            
            U_combined = np.hstack([U_444, U_r])
            V_combined = np.hstack([V_444_full, V_r])
            W_combined = np.hstack([W_444_full, W_r])
            
            recon = np.einsum('ak,bk,ck->abc', U_combined, V_combined, W_combined)
            if np.allclose(recon, T, atol=1e-10):
                total_rank = 49 + r_struct
                if total_rank < best_rank:
                    best_rank = total_rank
                    best_result = {
                        "U": U_combined.tolist(),
                        "V": V_combined.tolist(),
                        "W": W_combined.tolist()
                    }
        
        # Also try greedy peeling on the right part
        if time.time() < deadline - 10.0:
            right_result = _greedy_peel(T_right, 16, 28, 28, min(deadline - 10.0, time.time() + 40.0), 48)
            if right_result is not None:
                r_right = len(right_result["U"][0])
                U_r = np.array(right_result["U"]).T
                V_r = np.array(right_result["V"]).T
                W_r = np.array(right_result["W"]).T
                
                U_combined = np.hstack([U_444, U_r])
                V_combined = np.hstack([V_444_full, V_r])
                W_combined = np.hstack([W_444_full, W_r])
                
                recon = np.einsum('ak,bk,ck->abc', U_combined, V_combined, W_combined)
                if np.allclose(recon, T, atol=1e-10):
                    total_rank = 49 + r_right
                    if total_rank < best_rank:
                        best_rank = total_rank
                        best_result = {
                            "U": U_combined.tolist(),
                            "V": V_combined.tolist(),
                            "W": W_combined.tolist()
                        }
    
    # Also try full greedy peeling as alternative
    if time.time() < deadline - 3.0 and best_rank >= 97:
        full_result = _greedy_peel(T, dim_u, dim_v, dim_w, deadline - 1.0, best_rank)
        if full_result is not None:
            r_full = len(full_result["U"][0])
            if r_full < best_rank:
                best_rank = r_full
                best_result = full_result
    
    # Try a different structural approach: use Strassen for all 7 columns
    # but with a more careful embedding.
    if time.time() < deadline - 10.0:
        # Alternative: decompose 4x4 * 4x7 as 7 independent 4x4 * 4x1 problems
        # Each 4x4 * 4x1 has rank at most 16 (schoolbook) or potentially less.
        # 7 * 16 = 112, same as schoolbook.
        #
        # Or use Strassen for 4x4 * 4x1: rank 7*2 = 14? No, that's not right either.
        #
        # Let's try: use the 4x4x4 Strassen (rank 49) for columns 0-3,
        # and for columns 4-6, use a 4x4 * 4x3 decomposition with rank 44
        # (Strassen for first 2 cols + schoolbook for 3rd col).
        #
        # Total: 49 + 44 = 93
        
        # Build 4x4 * 4x3 with rank 44
        U_443 = np.zeros((16, 44))
        V_443 = np.zeros((28, 44))
        W_443 = np.zeros((28, 44))
        
        col_idx = 0
        # For k=4,5 (first 2 of the right part): use Strassen-like
        for k in range(4, 6):
            for i_block in range(2):
                for j_block in range(2):
                    for s in range(7):
                        # A indices
                        for i_prime in range(2):
                            for j_prime in range(2):
                                a_row = i_block * 2 + i_prime
                                a_col = j_block * 2 + j_prime
                                U_443[a_row * 4 + a_col, col_idx] = U_2[i_prime * 2 + j_prime, s]
                        
                        # B indices
                        for j_prime in range(2):
                            b_row = j_block * 2 + j_prime
                            V_443[b_row * 7 + k, col_idx] = V_2[j_prime, s]  # Only 2 rows in B for this block
                            # Wait, V_2 has shape (4,7) for 2x2*2x2, indexed as j*2+k.
                            # For 2x2*2x1, we only need j' in 0,1 and k'=0.
                            # So V_2[j_prime * 2 + 0, s]
                        
                        # C indices
                        for i_prime in range(2):
                            c_row = k * 4 + i_block * 2 + i_prime
                            W_443[c_row, col_idx] = W_2[i_prime, s]  # W_2[i_prime * 2 + 0, s]
                        
                        col_idx += 1
        
        # For k=6 (3rd column): schoolbook, rank 16
        for i in range(4):
            for j in range(4):
                U_443[i * 4 + j, col_idx] = 1
                V_443[j * 7 + 6, col_idx] = 1
                W_443[6 * 4 + i, col_idx] = 1
                col_idx += 1
        
        # Verify
        recon_443 = np.einsum('ak,bk,ck->abc', U_443, V_443, W_443)
        T_443 = np.zeros((16, 28, 28), dtype=np.float64)
        for i in range(4):
            for j in range(4):
                for k in range(4, 7):
                    T_443[i * 4 + j, j * 7 + k, k * 4 + i] = 1
        
        if np.allclose(recon_443, T_443, atol=1e-10):
            U_combined = np.hstack([U_444, U_443])
            V_combined = np.hstack([V_444_full, V_443])
            W_combined = np.hstack([W_444_full, W_443])
            
            recon = np.einsum('ak,bk,ck->abc', U_combined, V_combined, W_combined)
            if np.allclose(recon, T, atol=1e-10):
                total_rank = 49 + 44
                if total_rank < best_rank:
                    best_rank = total_rank
                    best_result = {
                        "U": U_combined.tolist(),
                        "V": V_combined.tolist(),
                        "W": W_combined.tolist()
                    }
    
    if best_result is not None and best_rank < schoolbook_rank:
        return best_result
    
    return _schoolbook(n, m, p)


def _structured_4x4x3():
    """Try to find a structured decomposition of 4x4 * 4x3 = 4x3 with rank < 48.
    
    Uses the fact that we can apply Strassen-like decompositions to sub-blocks.
    Partition A into 2x2 blocks, B into 2x3 blocks.
    
    A = [[A00, A01], [A10, A11]] (each 2x2)
    B = [[B00, B01, B02], [B10, B11, B12]] (each 2x3)
    
    C = A*B = [[C00, C01, C02], [C10, C11, C12]] (each 2x3)
    C00 = A00*B00 + A01*B10
    C01 = A00*B01 + A01*B11
    C02 = A00*B02 + A01*B12
    C10 = A10*B00 + A11*B10
    C11 = A10*B01 + A11*B11
    C12 = A10*B02 + A11*B12
    
    Each "block product" is 2x2 * 2x3 = 2x3.
    For 2x2 * 2x3, we can use rank 12 (schoolbook) or potentially less.
    
    Actually, 2x2 * 2x3:
    z00 = a11*b10 + a12*b20
    z01 = a11*b11 + a12*b21
    z02 = a11*b12 + a12*b22
    z10 = a21*b10 + a22*b20
    z11 = a21*b11 + a22*b21
    z12 = a21*b12 + a22*b22
    
    This is 3 independent 2x2 * 2x1 problems, each rank 4, total rank 12.
    
    So total rank = 4 * 12 = 48. Same as schoolbook.
    
    Let me try a different partitioning or use a more clever approach.
    
    Alternative: use the 2x2 * 2x2 Strassen decomposition (rank 7) for the first
    2 columns, and schoolbook for the 3rd column.
    
    For columns 0,1: 2x2 * 2x2, rank 7 per block, 4 blocks = 28
    For column 2: 2x2 * 2x1, rank 4 per block, 4 blocks = 16
    Total: 28 + 16 = 44
    
    But wait, the blocks aren't independent in the way I'm thinking.
    Let me be more careful.
    
    For C00 and C01 (first two columns of top row):
    C00 = A00*B00 + A01*B10  (2x2 * 2x2)
    C01 = A00*B01 + A01*B11  (2x2 * 2x2)
    
    These share A00 and A01, but have different B blocks.
    We can't simply add their ranks because they're not independent.
    
    Actually, they ARE independent in the tensor sense: the output indices
    are different (C00 vs C01), so the tensor is a direct sum.
    
    So rank(C00 computation) + rank(C01 computation) = 7 + 7 = 14
    Similarly for C10, C11: 7 + 7 = 14
    Total for first 2 columns: 28
    
    For column 2:
    C02 = A00*B02 + A01*B12  (2x2 * 2x1, rank 4)
    C12 = A10*B02 + A11*B12  (2x2 * 2x1, rank 4)
    Total: 8
    
    Grand total: 28 + 8 = 36
    
    That's an improvement over 48!
    """
    # Build Strassen decomposition for 2x2 * 2x2
    U_s, V_s, W_s = _strassen_2x2x2()
    
    # Build schoolbook decomposition for 2x2 * 2x1
    U_sb = np.zeros((4, 4))
    V_sb = np.zeros((2, 4))  # Wait, B is 2x1, so V has 2*1=2 rows? No.
    # For 2x2 * 2x1: A is 2x2 (4 entries), B is 2x1 (2 entries), C is 2x1 (2 entries)
    # T[a, b, c] = 1 if a = i*2+j, b = j*1+k, c = k*2+i for i,j in {0,1}, k in {0}
    # So a = i*2+j, b = j, c = 2i + 0 = 2i... wait, k=0, so c = 0*2+i = i
    # T[i*2+j, j, i] = 1 for i,j in {0,1}
    # That's only 4 non-zero entries.
    
    # Actually for 2x2 * 2x1:
    # z0 = a00*b0 + a01*b1
    # z1 = a10*b0 + a11*b1
    # T[0,0,0] = 1 (a00, b0, z0)
    # T[1,1,0] = 1 (a01, b1, z0)
    # T[2,0,1] = 1 (a10, b0, z1)
    # T[3,1,1] = 1 (a11, b1, z1)
    
    U_sb = np.zeros((4, 4))
    V_sb = np.zeros((2, 4))
    W_sb = np.zeros((2, 4))
    
    # Term 0: a00 * b0 -> z0
    U_sb[0, 0] = 1
    V_sb[0, 0] = 1
    W_sb[0, 0] = 1
    # Term 1: a01 * b1 -> z0
    U_sb[1, 1] = 1
    V_sb[1, 1] = 1
    W_sb[0, 1] = 1
    # Term 2: a10 * b0 -> z1
    U_sb[2, 2] = 1
    V_sb[0, 2] = 1
    W_sb[1, 2] = 1
    # Term 3: a11 * b1 -> z1
    U_sb[3, 3] = 1
    V_sb[1, 3] = 1
    W_sb[1, 3] = 1
    
    # Now build the full 4x4 * 4x3 decomposition
    # A indices: i*4+j for i,j in 0..3
    # B indices: j*3+k for j in 0..3, k in 0..2
    # C indices: k*4+i for k in 0..2, i in 0..3
    
    # We'll use rank 36: 28 for first 2 columns (Strassen) + 8 for 3rd column (schoolbook)
    rank = 36
    U_full = np.zeros((16, rank))
    V_full = np.zeros((12, rank))  # 4x3 = 12
    W_full = np.zeros((12, rank))  # 3x4 = 12
    
    # First 2 columns (k=0,1): use Strassen for each 2x2 * 2x2 block
    # Block (i_block, j_block) where i_block, j_block in {0,1}
    # For column k, the B block is rows j_block*2 to j_block*2+1, column k
    
    col_idx = 0
    for i_block in range(2):
        for j_block in range(2):
            for k in range(2):  # k = 0, 1
                # This block computes: A[i_block*2:(i_block+1)*2, j_block*2:(j_block+1)*2] *
                #                     B[j_block*2:(j_block+1)*2, k] -> 
                #                     C[k, i_block*2:(i_block+1)*2]
                #
                # Using Strassen for 2x2 * 2x2:
                # A_local = A[i_block*2 + i', j_block*2 + j'] for i',j' in 0,1
                # B_local = B[j_block*2 + j', k] for j' in 0,1
                # C_local = C[k*4 + i_block*2 + i', :] for i' in 0,1
                #
                # Strassen terms:
                for s in range(7):
                    # U: A indices
                    for i_prime in range(2):
                        for j_prime in range(2):
                            a_idx = i_block * 2 + i_prime
                            a_col = j_block * 2 + j_prime
                            U_full[a_idx * 4 + a_col, col_idx] = U_s[i_prime * 2 + j_prime, s]
                    
                    # V: B indices
                    for j_prime in range(2):
                        b_idx = j_block * 2 + j_prime
                        V_full[b_idx * 3 + k, col_idx] = V_s[j_prime * 2 + 0, s]  # B is 2x1 here
                        # Wait, B_s has shape (4,7) for 2x2*2x2, but here B is 2x1.
                        # I need to adapt.
                    
                    # This is getting complicated. Let me reconsider.
                
                col_idx += 7
    
    # This approach is getting too complex. Let me just return None and fall back
    # to the greedy peeling approach.
    return None


def _greedy_peel(T, dim_u, dim_v, dim_w, deadline, max_terms):
    """Greedy rank-1 peeling with multiple restarts. Returns dict or None."""
    R = T.copy()
    U_list = []
    V_list = []
    W_list = []
    
    for term_idx in range(max_terms):
        if time.time() > deadline:
            break
        if not np.any(R):
            break
        
        best_score = 0
        best_u, best_v, best_w = None, None, None
        
        # Multiple restarts with different initializations
        n_restarts = 8
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            
            if restart == 0:
                cu = np.sign(np.sum(R, axis=(1, 2)))
                cv = np.sign(np.sum(R, axis=(0, 2)))
                cw = np.sign(np.sum(R, axis=(0, 1)))
            elif restart == 1:
                cu = np.random.choice([-1, 1], size=dim_u).astype(np.float64)
                cv = np.random.choice([-1, 1], size=dim_v).astype(np.float64)
                cw = np.random.choice([-1, 1], size=dim_w).astype(np.float64)
            else:
                cu = np.random.choice([-1, 0, 1], size=dim_u).astype(np.float64)
                cv = np.random.choice([-1, 0, 1], size=dim_v).astype(np.float64)
                cw = np.random.choice([-1, 0, 1], size=dim_w).astype(np.float64)
            
            # Alternating minimization
            n_iters = 15
            for it in range(n_iters):
                if time.time() > deadline:
                    break
                
                M_u = np.einsum('abc,b,c->a', R, cv, cw)
                new_u = np.sign(M_u)
                
                M_v = np.einsum('abc,a,c->b', R, new_u, cw)
                new_v = np.sign(M_v)
                
                M_w = np.einsum('abc,a,b->c', R, new_u, new_v)
                new_w = np.sign(M_w)
                
                cu, cv, cw = new_u, new_v, new_w
            
            score = np.einsum('abc,a,b,c', R, cu, cv, cw)
            if score > best_score:
                best_score = score
                best_u, best_v, best_w = cu.copy(), cv.copy(), cw.copy()
        
        if best_score <= 0 or best_u is None:
            break
        
        R -= np.einsum('a,b,c->abc', best_u, best_v, best_w)
        U_list.append(best_u)
        V_list.append(best_v)
        W_list.append(best_w)
        
        if np.allclose(R, 0, atol=1e-10):
            break
    
    if np.allclose(R, 0, atol=1e-10) and len(U_list) > 0:
        rank = len(U_list)
        U = np.array(U_list).T
        V = np.array(V_list).T
        W = np.array(W_list).T
        
        reconstructed = np.einsum('ak,bk,ck->abc', U, V, W)
        if np.allclose(reconstructed, T, atol=1e-10):
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    
    return None


def _strassen_2x2x2():
    """Return Strassen decomposition of 2x2x2 matrix multiplication.
    
    Returns U, V, W each of shape (4, 7) such that
    sum_k U[a,k]*V[b,k]*W[c,k] = T[a,b,c]
    where T[i*2+j, j*2+k, k*2+i] = 1
    """
    # Strassen's 7 products for 2x2 matrix multiplication
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)
    
    # M1 = (a11+a22)(b11+b22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (a21+a22)b11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3 = a11(b12-b22)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = a22(b21-b11)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (a11+a12)b22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6 = (a21-a11)(b11+b12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (a12-a22)(b21+b22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W


def _strassen_2x2x2_orig():
    """Return Strassen decomposition of 2x2x2 matrix multiplication.
    
    Returns U, V, W each of shape (4, 7) such that
    sum_k U[a,k]*V[b,k]*W[c,k] = T[a,b,c]
    where T[i*2+j, j*2+k, k*2+i] = 1
    """
    # Strassen's 7 products for 2x2 matrix multiplication
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # C = A*B
    
    # Strassen products:
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # U vectors (for A indices: a11=0, a12=1, a21=2, a22=3)
    # V vectors (for B indices: b11=0, b12=1, b21=2, b22=3)
    # W vectors (for C indices: c11=0, c12=1, c21=2, c22=3)
    
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)
    
    # M1 = (a11+a22)(b11+b22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1  # C11 and C22
    
    # M2 = (a21+a22)b11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1  # C21 += M2, C22 -= M2
    
    # M3 = a11(b12-b22)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1  # C12 += M3, C22 += M3
    
    # M4 = a22(b21-b11)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1  # C11 += M4, C21 += M4
    
    # M5 = (a11+a12)b22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1  # C11 -= M5, C12 += M5
    
    # M6 = (a21-a11)(b11+b12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1  # C22 += M6
    
    # M7 = (a12-a22)(b21+b22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1  # C11 += M7
    
    return U, V, W


def _schoolbook(n, m, p):
    """Return schoolbook decomposition of n x m times m x p."""
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

# EVOLVE-BLOCK-END
