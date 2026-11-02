import numpy as np
import time

def construct():
    start_time = time.time()
    
    n, m, p = 2, 3, 6
    
    # The strategy is to decompose M_{2,3,6} into three M_{2,3,2} blocks.
    # The multiplication C = A B where A is 2x3 and B is 3x6.
    # Split B into B1, B2, B3 each of size 3x2.
    # C = [A B1, A B2, A B3].
    # Since the output indices for the three blocks are disjoint, the rank is the sum of the ranks.
    # We need a rank 11 decomposition for M_{2,3,2}.
    # 
    # Let's verify if rank 11 is possible for 2x3x2.
    # Schoolbook is 12.
    # Can we save 1 multiplication?
    # A = [[a11, a12, a13], [a21, a22, a23]]
    # B = [[b11, b12], [b21, b22], [b31, b32]]
    # C = [[c11, c12], [c21, c22]]
    #
    # c11 = a11*b11 + a12*b21 + a13*b31
    # c12 = a11*b12 + a12*b22 + a13*b32
    # c21 = a21*b11 + a22*b21 + a23*b31
    # c22 = a21*b12 + a22*b22 + a23*b32
    #
    # This is equivalent to multiplying a 2x3 matrix by a 3x2 matrix.
    # This can be viewed as two 2x3x1 multiplications (for col 1 and col 2 of B) but that's rank 12.
    # Or two 1x3x2? No.
    #
    # Let's look for a rank 11 solution.
    # Actually, R(2,3,2) is known to be 11?
    # Let's check if we can construct it.
    # If we use Strassen for 2x2x2 (rank 7), we can handle the first 2x2 block of A and B.
    # A = [A1, a2] where A1 is 2x2, a2 is 2x1.
    # B = [B1; b2] where B1 is 2x2, b2 is 1x2.
    # C = A1 B1 + a2 b2.
    # A1 B1 has rank 7 (Strassen).
    # a2 b2 is rank 1? No, it's an outer product of a 2x1 and 1x2, resulting in a 2x2 matrix.
    # The term a2 b2 contributes to all 4 entries of C.
    # We need to add this to the Strassen result.
    # The Strassen result computes A1 B1.
    # We need to add a2 b2.
    # a2 b2 = [a21, a22]^T [b21, b22].
    # This is 4 terms: a21*b21, a21*b22, a22*b21, a22*b22.
    # Can we fold these into the Strassen terms?
    # Strassen terms are linear combinations of entries of A1 and B1.
    # a2 and b2 are separate entries.
    # So we likely need extra multiplications for the a2*b2 part.
    # The rank of the outer product tensor (2x1x1x2) is 1? No, it's a bilinear map.
    # The map (a2, b2) -> a2 b2 is rank 1 in the sense that it's a single multiplication if we pre-compute? No, a2 and b2 are vectors.
    # Wait, a2 is 2x1, b2 is 1x2. Their product is 2x2.
    # The tensor for this part has rank 1? No.
    # T_{i,j,k,l} = 1 if i is row, j is col...
    # Actually, the product a2 b2 has 4 entries.
    # c11 += a21 b21
    # c12 += a21 b22
    # c21 += a22 b21
    # c22 += a22 b22
    # This is 4 independent multiplications if we don't share.
    # But we can share? No, a21*b21 is distinct from a21*b22.
    # So this part has rank 4.
    # Total rank = 7 (Strassen) + 4 (outer) = 11.
    # YES! Rank 11 is achievable for M_{2,3,2}.
    
    # Let's construct the rank 11 decomposition for M_{2,3,2}.
    # Indices for M_{2,3,2}:
    # A: 2x3 -> 6 entries. Index u = a*3 + b (a in 0..1, b in 0..2)
    # B: 3x2 -> 6 entries. Index v = c*2 + d (c in 0..2, d in 0..1)
    # C: 2x2 -> 4 entries. Index w = i*2 + k (i in 0..1, k in 0..1)
    
    # Strassen for 2x2x2 (A1, B1 -> C1):
    # A1 is 2x2 (entries a00, a01, a10, a11 in A)
    # B1 is 2x2 (entries b00, b01, b10, b11 in B)
    # Strassen uses 7 multiplications.
    # Let's define the 7 terms for the 2x2 block.
    # A1 indices in global A:
    # a00 -> u=0
    # a01 -> u=1
    # a10 -> u=3
    # a11 -> u=4
    # (a02, a12 are the "a2" part, indices u=2, u=5)
    
    # B1 indices in global B:
    # b00 -> v=0
    # b01 -> v=1
    # b10 -> v=2
    # b11 -> v=3
    # (b20, b21 are the "b2" part, indices v=4, v=5)
    
    # C1 indices in global C:
    # c00 -> w=0
    # c01 -> w=1
    # c10 -> w=2
    # c11 -> w=3
    
    # Strassen terms for A1 B1:
    # M1 = (a10 + a11) * (b00 + b01)
    # M2 = (a00 + a01) * b00
    # M3 = a00 * (b11 - b01)
    # M4 = a11 * (b10 - b00)
    # M5 = (a00 + a10) * b10
    # M6 = (a01 + a11) * b11
    # M7 = (a00 + a01 - a10 - a11) * (b10 + b11)
    
    # Result C1:
    # c00 = M1 + M4 - M5 + M7
    # c01 = M3 + M1 - M6 + M7
    # c10 = M2 - M1 + M5 - M7
    # c11 = M2 + M4 - M3 + M6
    
    # Let's build U, V, W for these 7 terms.
    U_strassen = np.zeros((6, 7))
    V_strassen = np.zeros((6, 7))
    W_strassen = np.zeros((4, 7))
    
    # Term 1: (a10 + a11) * (b00 + b01)
    U_strassen[3, 0] = 1; U_strassen[4, 0] = 1
    V_strassen[0, 0] = 1; V_strassen[1, 0] = 1
    W_strassen[0, 0] = 1; W_strassen[1, 0] = 1; W_strassen[2, 0] = -1; W_strassen[3, 0] = 1
    
    # Term 2: (a00 + a01) * b00
    U_strassen[0, 1] = 1; U_strassen[1, 1] = 1
    V_strassen[0, 1] = 1
    W_strassen[2, 1] = 1; W_strassen[3, 1] = 1
    
    # Term 3: a00 * (b11 - b01)
    U_strassen[0, 2] = 1
    V_strassen[3, 2] = 1; V_strassen[1, 2] = -1
    W_strassen[1, 2] = 1; W_strassen[3, 2] = -1
    
    # Term 4: a11 * (b10 - b00)
    U_strassen[4, 3] = 1
    V_strassen[2, 3] = 1; V_strassen[0, 3] = -1
    W_strassen[0, 3] = 1; W_strassen[3, 3] = 1
    
    # Term 5: (a00 + a10) * b10
    U_strassen[0, 4] = 1; U_strassen[3, 4] = 1
    V_strassen[2, 4] = 1
    W_strassen[0, 4] = -1; W_strassen[2, 4] = 1
    
    # Term 6: (a01 + a11) * b11
    U_strassen[1, 5] = 1; U_strassen[4, 5] = 1
    V_strassen[3, 5] = 1
    W_strassen[1, 5] = -1; W_strassen[3, 5] = 1
    
    # Term 7: (a00 + a01 - a10 - a11) * (b10 + b11)
    U_strassen[0, 6] = 1; U_strassen[1, 6] = 1; U_strassen[3, 6] = -1; U_strassen[4, 6] = -1
    V_strassen[2, 6] = 1; V_strassen[3, 6] = 1
    W_strassen[0, 6] = 1; W_strassen[1, 6] = 1; W_strassen[2, 6] = -1; W_strassen[3, 6] = -1
    
    # Now the 4 terms for a2 b2.
    # a2 is A[2] (a02) and A[5] (a12).
    # b2 is B[4] (b20) and B[5] (b21).
    # c00 += a02 * b20
    # c01 += a02 * b21
    # c10 += a12 * b20
    # c11 += a12 * b21
    
    U_outer = np.zeros((6, 4))
    V_outer = np.zeros((6, 4))
    W_outer = np.zeros((4, 4))
    
    # Term 8: a02 * b20
    U_outer[2, 0] = 1
    V_outer[4, 0] = 1
    W_outer[0, 0] = 1
    
    # Term 9: a02 * b21
    U_outer[2, 1] = 1
    V_outer[5, 1] = 1
    W_outer[1, 1] = 1
    
    # Term 10: a12 * b20
    U_outer[5, 2] = 1
    V_outer[4, 2] = 1
    W_outer[2, 2] = 1
    
    # Term 11: a12 * b21
    U_outer[5, 3] = 1
    V_outer[5, 3] = 1
    W_outer[3, 3] = 1
    
    # Combine Strassen and Outer for one 2x3x2 block.
    U_block = np.hstack([U_strassen, U_outer])
    V_block = np.hstack([V_strassen, V_outer])
    W_block = np.hstack([W_strassen, W_outer])
    
    # Verify the block decomposition locally.
    def verify_block(U, V, W):
        # U: 6x11, V: 6x11, W: 4x11
        for u in range(6):
            for v in range(6):
                for w in range(4):
                    # Target: 1 if A index u and B index v contribute to C index w
                    # u = a*3 + b
                    # v = c*2 + d
                    # w = i*2 + k
                    # Condition: a==i, c==k, b==d
                    a, b = divmod(u, 3)
                    c, d = divmod(v, 2)
                    i, k = divmod(w, 2)
                    target = 1.0 if (a == i and c == k and b == d) else 0.0
                    val = np.dot(U[u, :], np.multiply(V[v, :], W[w, :]))
                    if not np.isclose(val, target, atol=1e-9):
                        return False
        return True
    
    if not verify_block(U_block, V_block, W_block):
        # Fallback to schoolbook if verification fails (should not happen)
        U_school = np.zeros((6, 12))
        V_school = np.zeros((6, 12))
        W_school = np.zeros((4, 12))
        for u in range(6):
            for v in range(6):
                # b = u % 3, d = v % 2
                if (u % 3) == (v % 2):
                    a = u // 3
                    c = v // 2
                    i = a
                    k = c
                    w = i * 2 + k
                    idx = u * 6 + v # unique index? No, just count
                    # Let's just fill sequentially
                    pass
        # Simpler fallback: just use rank 12 schoolbook
        U_school = np.zeros((6, 12))
        V_school = np.zeros((6, 12))
        W_school = np.zeros((4, 12))
        idx = 0
        for a in range(2):
            for b in range(3):
                for c in range(3):
                    if b == c:
                        u = a * 3 + b
                        v = c * 2 + b
                        w = a * 2 + c
                        U_school[u, idx] = 1
                        V_school[v, idx] = 1
                        W_school[w, idx] = 1
                        idx += 1
        # This gives rank 9? No, 2*3*1/2?
        # Schoolbook is 12.
        # My loop: a in 2, b in 3, c in 3. if b==c.
        # b is 0,1,2. c is 0,1,2.
        # Matches: (0,0), (1,1), (2,2). 3 matches per a.
        # Total 6.
        # Wait, B is 3x2. d is 0,1.
        # v = c*2 + d.
        # Condition b==d.
        # b is 0,1,2. d is 0,1.
        # So b can only be 0 or 1.
        # If b=2, no match.
        # So for each a, b=0 matches d=0 (c=0), b=1 matches d=1 (c=1).
        # Wait, c is the row of B. d is the col of B.
        # v = c*2 + d.
        # Condition b == d.
        # So d is determined by b.
        # c can be anything? No, the term is A[a,b] * B[c,d].
        # It contributes to C[a,c] only if b==d.
        # So for each a, b, c, if b==d (where d is col of B), then term exists.
        # But d is part of v.
        # So for a fixed a, b, and c, we need d=b.
        # If b=0, d=0. v = c*2 + 0.
        # If b=1, d=1. v = c*2 + 1.
        # If b=2, no d in {0,1} matches.
        # So for b=0, c can be 0,1,2. 3 terms.
        # For b=1, c can be 0,1,2. 3 terms.
        # For b=2, 0 terms.
        # Total 6 terms per a. 12 terms total.
        idx = 0
        for a in range(2):
            for b in range(3):
                if b < 2:
                    d = b
                    for c in range(3):
                        u = a * 3 + b
                        v = c * 2 + d
                        w = a * 2 + c
                        U_school[u, idx] = 1
                        V_school[v, idx] = 1
                        W_school[w, idx] = 1
                        idx += 1
        U_block, V_block, W_block = U_school, V_school, W_school
        # Rank is 12. Total 36.
    
    # Now construct the full 2x3x6 decomposition.
    # Rank will be 3 * 11 = 33.
    R = U_block.shape[1]
    Total_Rank = 3 * R
    
    U_full = np.zeros((6, Total_Rank))
    V_full = np.zeros((18, Total_Rank))
    W_full = np.zeros((12, Total_Rank))
    
    # Block 0: columns 0,1 of B.
    # B indices 0..5 correspond to c=0,1,2 and d=0,1.
    # Block 0 uses d=0,1 for c=0,1,2? No.
    # My M_{2,3,2} assumed B is 3x2.
    # The full B is 3x6.
    # Block 0: B[:, 0:2] -> indices v = c*2 + d where d in 0,1.
    # This matches my V_block indices 0..5.
    # So V_full[0:6, 0:R] = V_block.
    # U_full is the same for all blocks.
    # W_full: Block 0 output C[:, 0:2].
    # C is 2x6. w = i*6 + k?
    # Task says W[k*n + i, r].
    # n=2. W index = k*2 + i.
    # k is 0..5. i is 0..1.
    # Block 0: k=0,1.
    # w indices for block 0:
    # k=0, i=0 -> 0
    # k=0, i=1 -> 1
    # k=1, i=0 -> 2
    # k=1, i=1 -> 3
    # My W_block indices were 0..3 corresponding to i*2 + k.
    # Wait, my W_block was defined with w = i*2 + k.
    # Task convention: W[k*n + i].
    # So I need to permute W_block indices.
    # My W_block row 0: i=0, k=0 -> Task W[0*2+0] = W[0].
    # My W_block row 1: i=0, k=1 -> Task W[1*2+0] = W[2].
    # My W_block row 2: i=1, k=0 -> Task W[0*2+1] = W[1].
    # My W_block row 3: i=1, k=1 -> Task W[1*2+1] = W[3].
    
    # Permutation for W:
    # My idx 0 -> Task idx 0
    # My idx 1 -> Task idx 2
    # My idx 2 -> Task idx 1
    # My idx 3 -> Task idx 3
    
    perm_W = [0, 2, 1, 3]
    
    for block in range(3):
        offset = block * R
        U_full[:, offset:offset+R] = U_block
        
        # V mapping
        # Block 0: B cols 0,1. v = c*2 + d. d in 0,1.
        # This is exactly indices 0..5 of V_full?
        # V_full index = c*6 + d?
        # Task: V index = c*p + d. p=6.
        # So V_full index = c*6 + d.
        # Block 0 uses d=0,1.
        # So V_full[c*6 + 0, ...] and V_full[c*6 + 1, ...]
        # My V_block has indices 0..5.
        # My V_block index v = c*2 + d.
        # If v=0 (c=0, d=0) -> V_full[0*6+0]
        # If v=1 (c=0, d=1) -> V_full[0*6+1]
        # If v=2 (c=1, d=0) -> V_full[1*6+0]
        # If v=3 (c=1, d=1) -> V_full[1*6+1]
        # If v=4 (c=2, d=0) -> V_full[2*6+0]
        # If v=5 (c=2, d=1) -> V_full[2*6+1]
        
        for c in range(3):
            for d in range(2):
                v_block_idx = c * 2 + d
                v_full_idx = c * 6 + d
                V_full[v_full_idx, offset:offset+R] = V_block[v_block_idx, :]
                
        # W mapping
        # Block 0: C cols 0,1. k=0,1.
        # W_full index = k*2 + i.
        # My W_block index w = i*2 + k.
        # We permute W_block rows to match task order.
        W_permuted = W_block[perm_W, :]
        
        for k in range(2):
            for i in range(2):
                w_full_idx = k * 2 + i
                w_block_idx = i * 2 + k # Original index in W_block
                # We already permuted, so we just map k, i to full index
                # W_permuted row corresponding to (i, k) is at index (i*2+k) in original, 
                # but we permuted it to be in task order?
                # Let's just build W_block in task order directly to avoid confusion.
                pass
        
        # Let's rebuild W_block in task order for clarity.
        # Task W index = k*2 + i.
        # My previous W_block was i*2 + k.
        # Let's just use the permutation.
        for k in range(2):
            for i in range(2):
                w_full_idx = (k + block*2) * 2 + i # k is global col 0..5
                # Wait, block 0 is k=0,1. Block 1 is k=2,3. Block 2 is k=4,5.
                # So global k = block*2 + local_k.
                local_k = k
                w_full_idx = (local_k + block*2) * 2 + i
                
                # Which row of W_block corresponds to this (i, local_k)?
                # Original W_block row idx = i*2 + local_k.
                W_full[w_full_idx, offset:offset+R] = W_block[i*2 + local_k, :]
                
    # Verify full decomposition
    def verify_full(U, V, W, n, m, p):
        for a in range(n):
            for b in range(m):
                for c in range(m):
                    for d in range(p):
                        for i in range(n):
                            for k in range(p):
                                target = 1.0 if (a == i and c == k and b == d) else 0.0
                                u_idx = a * m + b
                                v_idx = c * p + d
                                w_idx = k * n + i
                                val = np.dot(U[u_idx, :], np.multiply(V[v_idx, :], W[w_idx, :]))
                                if not np.isclose(val, target, atol=1e-9):
                                    return False
        return True
    
    if verify_full(U_full, V_full, W_full, n, m, p):
        return {
            "U": U_full.tolist(),
            "V": V_full.tolist(),
            "W": W_full.tolist()
        }
    else:
        # Fallback to rank 36 schoolbook if verification fails
        U_school = np.zeros((6, 36))
        V_school = np.zeros((18, 36))
        W_school = np.zeros((12, 36))
        idx = 0
        for a in range(n):
            for b in range(m):
                for c in range(m):
                    for d in range(p):
                        if b == d:
                            u_idx = a * m + b
                            v_idx = c * p + d
                            w_idx = c * n + a
                            U_school[u_idx, idx] = 1
                            V_school[v_idx, idx] = 1
                            W_school[w_idx, idx] = 1
                            idx += 1
        return {
            "U": U_school.tolist(),
            "V": V_school.tolist(),
            "W": W_school.tolist()
        }

if __name__ == "__main__":
    construct()
