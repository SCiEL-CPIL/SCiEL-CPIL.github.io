import numpy as np
import time
import itertools

def construct():
    n, m, p = 6, 6, 6
    # Target tensor T[u, v, w] where u=i*m+k, v=k*p+j, w=j*n+i
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(p):
            for k in range(m):
                u = i * m + k
                v = k * p + j
                w = j * n + i
                T[u, v, w] = 1.0

    deadline = time.time() + 110.0

    # We will use a block-Strassen approach with block size 2 (3x3 blocks of 2x2)
    # Standard: 27 rank-8 = 216
    # Block-Strassen 2x2: 9 blocks * 7 = 63 for diagonal, plus off-diagonal
    # 
    # Actually, for 6x6 = 3x3 blocks of 2x2:
    # Diagonal blocks (3): use Strassen rank 7 each -> 21
    # Off-diagonal blocks (6): use standard rank 8 each -> 48
    # Total: 21 + 48 = 69? No, that's not right.
    #
    # Let me think again. The matrix multiplication is C = A*B where A,B are 6x6.
    # If we partition into 3x3 blocks of size 2x2:
    # C_ij = sum_k A_ik * B_kj
    # Each C_ij is 2x2, A_ik is 2x2, B_kj is 2x2.
    # For each pair (i,j), C_ij = sum_k A_ik * B_kj
    #
    # Standard: each 2x2*2x2 takes rank 8. There are 3*3*3 = 27 such products.
    # Total rank: 27*8 = 216.
    #
    # Using Strassen for the 3x3 block structure:
    # We can apply Strassen to the block matrix itself.
    # 
    # Let me use the known result: for n=6, the best known bound is much better.
    # But let's just do a simple block approach with bs=3 (2x2 blocks):
    #
    # Actually, let me reconsider. The current program uses bs=3 (3x3 blocks) 
    # with Strassen on 2x2 block matrices, giving 9*7 = 63 for the "inner" 
    # Strassen, but the rank is 189.
    #
    # 189 = 27 * 7. So it's using 27 Strassen steps of rank 7 each.
    # That corresponds to: for each of the 27 block products (i,k,j), use Strassen.
    # Wait, no. 27*7 = 189.
    #
    # Let me just implement a clean block-Strassen with block size 2:
    # 3x3 grid of 2x2 blocks.
    # For each C_ij = sum_k A_ik * B_kj:
    #   Use standard 2x2*2x2 = rank 8 for each k, total 3*8 = 24 per C_ij
    #   Or use Strassen-like on the sum.
    #
    # Actually, the simplest improvement: use Strassen for 2x2 blocks directly.
    # Each 2x2*2x2 = rank 7 (Strassen) instead of 8.
    # 27 * 7 = 189. Same as before.
    #
    # To do better, we need to exploit the 3x3 structure more.
    # 
    # Let me try: for each (i,j) pair, C_ij = A_i0*B_0j + A_i1*B_1j + A_i2*B_2j
    # This is a sum of 3 products of 2x2 matrices.
    # We can precompute linear combinations.
    #
    # Actually, let me just keep the current approach but verify it works,
    # and then try to improve with a different block decomposition.
    
    bs = 2  # block size 2, so 3x3 blocks
    nb = n // bs  # 3 blocks
    
    # Strassen 2x2 definitions (7 multiplications)
    # A blocks indexed 0,1,2,3 = A00,A01,A10,A11
    # B blocks indexed 0,1,2,3 = B00,B01,B10,B11
    strassen_defs = [
        {'A': [(0, 1.0), (3, 1.0)], 'B': [(0, 1.0), (3, 1.0)]},
        {'A': [(2, 1.0), (3, 1.0)], 'B': [(0, 1.0)]},
        {'A': [(0, 1.0)], 'B': [(1, 1.0), (3, -1.0)]},
        {'A': [(3, 1.0)], 'B': [(2, 1.0), (0, -1.0)]},
        {'A': [(0, 1.0), (1, 1.0)], 'B': [(3, 1.0)]},
        {'A': [(2, 1.0), (0, -1.0)], 'B': [(0, 1.0), (1, 1.0)]},
        {'A': [(1, 1.0), (3, -1.0)], 'B': [(2, 1.0), (3, 1.0)]},
    ]
    
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    recon_coeffs = [
        [(0, 1.0), (3, 1.0), (4, -1.0), (6, 1.0)],
        [(2, 1.0), (4, 1.0)],
        [(1, 1.0), (3, 1.0)],
        [(0, 1.0), (1, -1.0), (2, 1.0), (5, 1.0)],
    ]
    
    # For each (i_block, k_block, j_block), we do a Strassen multiplication
    # Total: nb*nb*nb = 27 Strassen instances, each rank 7
    # Total rank: 27 * 7 = 189
    
    rank = 189
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    r_idx = 0
    for i_b in range(nb):
        for k_b in range(nb):
            for j_b in range(nb):
                for s in range(7):
                    s_def = strassen_defs[s]
                    # U: A[i_b * bs + u, k_b * bs + v]
                    for u in range(bs):
                        for v in range(bs):
                            for block_idx, coeff_a in s_def['A']:
                                ar = block_idx // 2
                                ac = block_idx % 2
                                i_glob = i_b * bs + u
                                k_glob = k_b * bs + v
                                # A_ik is at row i_glob, col k_glob in A
                                # But A_ik is a 2x2 block: A[i_b*bs:..., k_b*bs:...]
                                # The Strassen block indices refer to the 2x2 structure
                                # A_00 = A[i_b*bs, k_b*bs], A_01 = A[i_b*bs, k_b*bs+1]
                                # A_10 = A[i_b*bs+1, k_b*bs], A_11 = A[i_b*bs+1, k_b*bs+1]
                                # So for block_idx (ar, ac):
                                # row = i_b*bs + ar + u? No...
                                # 
                                # Actually, the 2x2 block A_ik has entries:
                                # A_ik[0,0] = A[i_b*bs + 0, k_b*bs + 0]
                                # A_ik[0,1] = A[i_b*bs + 0, k_b*bs + 1]
                                # A_ik[1,0] = A[i_b*bs + 1, k_b*bs + 0]
                                # A_ik[1,1] = A[i_b*bs + 1, k_b*bs + 1]
                                #
                                # Strassen combines these 4 entries into 7 linear combinations.
                                # The "u" and "v" in the current code are the positions
                                # within the 2x2 block.
                                #
                                # Wait, I'm confusing myself. Let me re-examine.
                                #
                                # In the current code with bs=3:
                                # for u in range(bs): for v in range(bs): for j in range(bs):
                                # This iterates over the 3x3x3 = 27 positions.
                                # For each position, it does one Strassen multiplication.
                                #
                                # The u,v,j are the "inner" indices within the blocks.
                                # i_glob = br * bs + u where br is from the Strassen block index.
                                #
                                # So the structure is:
                                # - Outer: i_b, k_b, j_b (which 2x2 block in the 3x3 grid)
                                # - Inner: u, v (position within the 2x2 A_ik block)
                                # - The Strassen definition operates on the 2x2 structure.
                                #
                                # Actually no. Let me re-read the current code more carefully.
                                #
                                # Current code: bs=3, so blocks are 3x3.
                                # for u in range(bs): for v in range(bs): for j in range(bs):
                                # That's 3*3*3 = 27 iterations.
                                # In each iteration, s_def is applied.
                                # block_idx is 0,1,2,3 (the 4 quadrants of the 3x3 block).
                                # br = block_idx // 2, bc = block_idx % 2
                                # i_glob = br * bs + u
                                # k_glob = bc * bs + v
                                #
                                # So for a 3x3 block A_ik, the 4 quadrants are:
                                # A_ik[0:3, 0:3] divided into 4 quadrants of size... 
                                # Wait, 3x3 can't be divided into 4 quadrants of integer size.
                                #
                                # I think the current code is actually treating the 3x3 block
                                # as having a 2x2 structure at a coarser level, and u,v are
                                # the fine indices.
                                #
                                # This is getting confusing. Let me just use bs=2 and be explicit.
                                #
                                # With bs=2, nb=3:
                                # A is 6x6, partitioned into 3x3 blocks of 2x2.
                                # A_ik for i_b in 0..2, k_b in 0..2 is a 2x2 matrix.
                                #
                                # For C_ij = sum_k A_ik * B_kj:
                                # Each term A_ik * B_kj is a 2x2 * 2x2 product.
                                # Using Strassen: 7 scalar multiplications.
                                # Total: 27 * 7 = 189.
                    
                    # U: coefficient for A[i_glob, k_glob] where i_glob, k_glob are global
                    # For Strassen step s, the A part is a linear combination of the 4 blocks
                    # of A_ik. Each block is a scalar (since we're doing scalar-level Strassen
                    # within the 2x2 block).
                    
                    # Actually, I realize the confusion: Strassen on 2x2 matrices treats
                    # each entry as a scalar. So the "rank" counts scalar multiplications.
                    # 
                    # Let me just directly build the tensor decomposition.
                    
                    pass
    
    # OK, let me take a completely different, cleaner approach.
    # Build the decomposition explicitly for the 6x6 case.
    
    # Reset
    U = np.zeros((n * m, 216), dtype=float)
    V = np.zeros((m * p, 216), dtype=float)
    W = np.zeros((p * n, 216), dtype=float)
    
    # First, let's just return the trivial decomposition to ensure correctness,
    # then improve.
    rank_trivial = n * m * p
    U = np.zeros((n * m, rank_trivial), dtype=float)
    V = np.zeros((m * p, rank_trivial), dtype=float)
    W = np.zeros((p * n, rank_trivial), dtype=float)
    
    idx = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                U[i * m + k, idx] = 1.0
                V[k * p + j, idx] = 1.0
                W[j * n + i, idx] = 1.0
                idx += 1
    
    # Verify
    T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
    if not np.allclose(T_recon, T, atol=1e-9):
        # Should not happen
        pass
    
    # Now try to improve using block-Strassen with block size 2
    # 3x3 blocks of 2x2
    # For each (i_b, k_b, j_b), do Strassen on the 2x2 block product
    # Total rank: 27 * 7 = 189
    
    rank_bs = 189
    U2 = np.zeros((n * m, rank_bs), dtype=float)
    V2 = np.zeros((m * p, rank_bs), dtype=float)
    W2 = np.zeros((p * n, rank_bs), dtype=float)
    
    # Strassen 2x2
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B10-B00)  [wait, let me use standard]
    # Standard Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B00)  -> actually M3 = A00*(B10-B00)? No.
    # Let me use the standard:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)(B00)
    # M3 = (A00)(B10-B00)  -> no, M3 = A00*(B10-B00)? 
    # 
    # Standard Strassen for C = AB:
    # C00 = M1 - M4 + M6 - M7  (using 1-indexed M1..M7)
    # Let me just use 0-indexed as in the current code.
    
    # From the current code's strassen_defs:
    # s=0: A=[(0,1),(3,1)] B=[(0,1),(3,1)]  -> (A00+A11)(B00+B11)
    # s=1: A=[(2,1),(3,1)] B=[(0,1)]         -> (A10+A11)B00
    # s=2: A=[(0,1)] B=[(1,1),(3,-1)]         -> A00(B01-B11)
    # s=3: A=[(3,1)] B=[(2,1),(0,-1)]         -> A11(B10-B00)
    # s=4: A=[(0,1),(1,1)] B=[(3,1)]          -> (A00+A01)B11
    # s=5: A=[(2,1),(0,-1)] B=[(0,1),(1,1)]   -> (A10-A00)(B00+B01)
    # s=6: A=[(1,1),(3,-1)] B=[(2,1),(3,1)]   -> (A01-A11)(B10+B11)
    
    # Reconstruction:
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    
    # Let me verify this is correct Strassen.
    # Standard Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B00)
    # M4 = A11(B01-B10)  [hmm, different conventions]
    # 
    # Let me just use the definitions from the current code and verify.
    
    # For the 2x2 block at position (i_b, k_b) in A and (k_b, j_b) in B:
    # A_ik is 2x2: A[i_b*2:(i_b+1)*2, k_b*2:(k_b+1)*2]
    # B_kj is 2x2: B[k_b*2:(k_b+1)*2, j_b*2:(j_b+1)*2]
    # 
    # The 4 "scalar" blocks of A_ik:
    # A_ik[0,0] = A[i_b*2, k_b*2]
    # A_ik[0,1] = A[i_b*2, k_b*2+1]
    # A_ik[1,0] = A[i_b*2+1, k_b*2]
    # A_ik[1,1] = A[i_b*2+1, k_b*2+1]
    #
    # Similarly for B_kj.
    #
    # For each Strassen step s (0..6), and for each (i_b, k_b, j_b):
    # The U entry: for each (u,v) in the 2x2 block, U[i_b*2+u, k_b*2+v, r] += coeff
    # Wait, no. U has shape (36, rank). U[row, r] where row = i*m+k = i*6+k.
    # 
    # For the A part of Strassen step s, the linear combination involves specific
    # entries of A_ik. Each such entry A[i_glob, k_glob] contributes to U[i_glob*6+k_glob, r].
    #
    # Let's define: for Strassen step s with A-def [(block_idx, coeff), ...]:
    #   For each (block_idx, coeff_a):
    #     ar = block_idx // 2, ac = block_idx % 2
    #     i_glob = i_b*2 + ar, k_glob = k_b*2 + ac
    #     U[i_glob*6 + k_glob, r] += coeff_a
    #
    # Similarly for V with B, and W with C reconstruction.
    
    r_idx = 0
    for i_b in range(nb):
        for k_b in range(nb):
            for j_b in range(nb):
                for s in range(7):
                    s_def = strassen_defs[s]
                    # U
                    for block_idx, coeff_a in s_def['A']:
                        ar = block_idx // 2
                        ac = block_idx % 2
                        i_glob = i_b * 2 + ar
                        k_glob = k_b * 2 + ac
                        U2[i_glob * m + k_glob, r_idx] += coeff_a
                    # V
                    for block_idx, coeff_b in s_def['B']:
                        br = block_idx // 2
                        bc = block_idx % 2
                        k_glob = k_b * 2 + br
                        j_glob = j_b * 2 + bc
                        V2[k_glob * p + j_glob, r_idx] += coeff_b
                    # W: reconstruction
                    for c_block_idx in range(4):
                        gamma = 0.0
                        for (s_idx, g) in recon_coeffs[c_block_idx]:
                            if s_idx == s:
                                gamma = g
                                break
                        if gamma != 0.0:
                            cr = c_block_idx // 2
                            cc = c_block_idx % 2
                            i_glob = i_b * 2 + cr
                            j_glob = j_b * 2 + cc
                            W2[j_glob * n + i_glob, r_idx] += gamma
                    r_idx += 1
    
    # Verify
    T_recon2 = np.einsum('ir,jr,kr->ijk', U2, V2, W2)
    if np.allclose(T_recon2, T, atol=1e-9):
        return {"U": U2.tolist(), "V": V2.tolist(), "W": W2.tolist()}
    else:
        # Fallback to trivial
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
