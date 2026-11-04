import numpy as np
import time

def _strassen_2x2x2():
    """Return (U, V, W) for 2x2x2 matrix multiplication via Strassen, rank 7."""
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    # c0: (A00+A11)(B00+B11)
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
    # c1: (A10+A11)B00
    U[2,1]=1; U[3,1]=1; V[0,1]=1; W[2,1]=1; W[3,1]=-1
    # c2: A00(B01-B11)
    U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[1,2]=1; W[3,2]=1
    # c3: A11(B10-B00)
    U[3,3]=1; V[2,3]=1; V[0,3]=-1; W[0,3]=1; W[2,3]=1
    # c4: (A00+A01)B11
    U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[1,4]=1
    # c5: (A10-A00)(B00+B01)
    U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[3,5]=1
    # c6: (A01-A11)(B10+B11)
    U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U^T * tensor * V * W^T gives correct matmul tensor."""
    # Tensor T[a_idx, b_idx, w_idx] where:
    # a_idx = i*m + j  (row i, col j of A)
    # b_idx = j'*p + k  (row j', col k of B)
    # w_idx = k*n + i'  (col k, row i' of C)
    # T[a,b,w] = 1 if j == j' and i == i' and k == k, else 0
    # The decomposition says: T[a,b,w] = sum_r U[a,r]*V[b,r]*W[w,r]
    # Check: for each (a,b,w) where T should be nonzero, verify.
    # T[i*m+j, j*p+k, k*n+i] = 1
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1
    # For all other (a,b,w), sum should be 0.
    # Efficient check: build the tensor from U,V,W and compare to identity-like pattern.
    # rank = U.shape[1]
    # Compute: for each r, outer(U[:,r], V[:,r]) gives (n*m, m*p) matrix
    # Then contract with W: result[i*m+j, j'*p+k, k'*n+i'] = sum_r U[i*m+j,r]*V[j'*p+k,r]*W[k'*n+i',r]
    # We need this to equal delta_{j,j'} * delta_{i,i'} * delta_{k,k'}
    
    # Check a sample of entries for speed
    rank = U.shape[1]
    # Check all nonzero entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                w = k * n + i
                val = np.dot(U[a, :], V[b, :] * W[w, :])
                if abs(val - 1.0) > tol:
                    return False
    # Check some zero entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for j2 in range(m):
                    if j2 != j:
                        a = i * m + j
                        b = j2 * p + k
                        w = k * n + i
                        val = np.dot(U[a, :], V[b, :] * W[w, :])
                        if abs(val) > tol:
                            return False
    return True

def _reduce_rank(U, V, W, max_time=30.0):
    """Try to reduce rank by finding linear dependencies among the rank-1 slices.
    The decomposition is valid if for all (a,b,w): sum_r U[a,r]V[b,r]W[w,r] = T[a,b,w].
    We can eliminate a column r if there exist coefficients c_s (s != r) such that
    U[:,r] = sum_s c_s U[:,s], V[:,r] = sum_s c_s V[:,s], W[:,r] = sum_s c_s W[:,s].
    This is a strong condition. A weaker approach: check if the tensor of outer products
    has rank less than the number of columns.
    
    Alternative: use the fact that if two columns are identical in U, we can combine them.
    Or: perform Gaussian elimination on the set of vectors {u_r * v_r * w_r} in the tensor space.
    
    For now, try a simple approach: check for duplicate columns in U (after normalization).
    """
    start = time.time()
    rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank
    
    # Try to find columns that are linear combinations of others
    # For each column r, check if U[:,r] is in span of other U columns
    # AND V[:,r] is in span of other V columns  
    # AND W[:,r] is in span of other W columns
    # This is expensive; do it heuristically.
    
    # Simpler: check for exact duplicate columns
    used = [True] * rank
    for r in range(rank):
        if not used[r]:
            continue
        for s in range(r+1, rank):
            if not used[s]:
                continue
            if np.array_equal(U[:,r], U[:,s]) and np.array_equal(V[:,r], V[:,s]) and np.array_equal(W[:,r], W[:,s]):
                used[s] = False
    
    if any(not u for u in used):
        new_rank = sum(used)
        best_U = U[:, used]
        best_V = V[:, used]
        best_W = W[:, used]
        best_rank = new_rank
    
    # Try: if U[:,r] = c * U[:,s] for some c, and V[:,r] = (1/c) * V[:,s], and W[:,r] = W[:,s]
    # then we can eliminate one of them.
    for r in range(best_rank):
        if time.time() - start > max_time:
            break
        for s in range(r+1, best_rank):
            if time.time() - start > max_time:
                break
            # Check if U[:,r] is scalar multiple of U[:,s]
            nz_r = np.nonzero(U[:,r])[0]
            nz_s = np.nonzero(U[:,s])[0]
            if len(nz_r) == 0 or len(nz_s) == 0:
                continue
            if not np.array_equal(nz_r, nz_s):
                continue
            # Check scalar ratio
            c = U[nz_r[0], r] / U[nz_r[0], s]
            if abs(c) < 1e-15:
                continue
            if np.allclose(U[:, r] / c, U[:, s], atol=1e-12):
                # Check V
                nz_v_r = np.nonzero(V[:,r])[0]
                nz_v_s = np.nonzero(V[:,s])[0]
                if not np.array_equal(nz_v_r, nz_v_s):
                    continue
                c_v = V[nz_v_r[0], r] / V[nz_v_r[0], s]
                if abs(c_v) < 1e-15:
                    continue
                if np.allclose(V[:, r] / c_v, V[:, s], atol=1e-12):
                    # Check W
                    if np.allclose(W[:, r], W[:, s], atol=1e-12):
                        # Eliminate column s
                        mask = list(range(best_rank))
                        mask.pop(s)
                        best_U = U[:, mask]
                        best_V = V[:, mask]
                        best_W = W[:, mask]
                        best_rank = len(mask)
                        break
    
    return best_U, best_V, best_W, best_rank

def construct():
    n, m, p = 4, 4, 8
    start_time = time.time()
    deadline = start_time + 130.0
    
    # Strassen 2x2x2 base
    U2, V2, W2 = _strassen_2x2x2()
    
    # Compose: 4x4x8 = (2x2x2) x (2x2x4) where 2x2x4 = 2x2x2 x 2 (schoolbook in last dim)
    # Actually: 4x4x8 can be seen as:
    # A is 4x4, B is 4x8, C is 8x4
    # Split A into 2x2 blocks (each 2x2), B into 2x4 blocks (each 2x4), C into 4x2 blocks (each 4x2)
    # 
    # Better: use Kronecker structure.
    # 4x4x8 matmul tensor = (2x2x2 matmul) ⊗ (2x2x4 matmul) in some sense?
    # Not quite. Let me think differently.
    #
    # View 4x4x8 as: for each of 2 "outer" 2x2x2 problems (covering the 4x4 part),
    # and for each of 2 "inner" 2x2x2 problems (covering the 2x4 part of B and 4x2 of C).
    #
    # Actually, the cleanest way:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Write i = 2*i1 + i0, j = 2*j1 + j0, k = 2*k1 + k0
    # where i1,j1 in {0,1}, i0,j0,k0 in {0,1}, k1 in {0,1,2,3}
    #
    # C[2*i1+i0, 2*k1+k0] = sum_{j1,j0} A[2*i1+i0, 2*j1+j0] * B[2*j1+j0, 2*k1+k0]
    #
    # Group by (i1, k1):
    # C_block(i1, k1)[i0, k0] = sum_{j1,j0} A_block(i1,j1)[i0,j0] * B_block(j1,k1)[j0,k0]
    # where A_block(i1,j1) is 2x2, B_block(j1,k1) is 2x2.
    #
    # So C_block(i1,k1) = A_block(i1,0) * B_block(0,k1) + A_block(i1,1) * B_block(1,k1)
    # This is a 2x2x4 problem: (2x2) * (2x4) -> (2x4)... no, (2x2)*(2x2) + (2x2)*(2x2) = 2x2.
    # Wait, C_block is 2x2 (i0, k0), A_block is 2x2, B_block is 2x2.
    # C_block(i1,k1) = A_block(i1,0)*B_block(0,k1) + A_block(i1,1)*B_block(1,k1)
    # Each product is 2x2x2, rank 7 via Strassen.
    # Total: 2 * 4 * 2 * 7 = 112? No: i1 in {0,1}, k1 in {0,1,2,3}, so 2*4=8 blocks.
    # Each block: 2 products of 2x2x2, each rank 7. Total = 8*2*7 = 112.
    #
    # But we can do better by sharing. For fixed i1:
    # C_block(i1, k1) for all k1: 
    #   = A_block(i1,0) * B_block(0, k1) + A_block(i1,1) * B_block(1, k1)
    # The Strassen intermediates for A_block(i1,0) can be shared across k1!
    #
    # For fixed i1 and j1, A_block(i1,j1) is fixed. Apply Strassen to get 7 intermediate values.
    # Then for each k1, multiply by B_block(j1,k1) using the same Strassen structure.
    #
    # So: for each (i1, j1), compute 7 Strassen intermediates from A_block(i1,j1).
    # For each (i1, j1, k1), use those 7 intermediates with B_block(j1,k1) to get C contribution.
    #
    # Total rank: 2 * 2 * 4 * 7 = 112. Still not great.
    #
    # Alternative: apply Strassen in the (i1,j1) plane (2x2x2 over block indices)
    # AND in the (j0,k0) plane (2x2x2 over element indices within blocks).
    #
    # This gives rank 7 * 7 = 49 for the 4x4x4 part, then x2 for the extra 4 columns of B.
    # Total: 49 * 2 = 98.
    #
    # Let me build this properly using Kronecker products.
    #
    # The 4x4x4 matmul tensor can be decomposed as:
    # T_4x4x4[i,j,k] = delta_{i,k} delta_{j,j} ... 
    # Using Strassen at block level (2x2 blocks) and Strassen at element level:
    #
    # T_4x4x4 = sum_{r=0}^{6} sum_{s=0}^{6} U4[i,r,s] * V4[j,r,s] * W4[k,r,s]
    # where U4, V4, W4 are built from Kronecker products of U2, V2, W2.
    #
    # Specifically, for block index (i1,j1) and element index (i0,j0):
    # i = 2*i1 + i0, j = 2*j1 + j0
    # U4[i, (r,s)] = U2[i1, r] * U2[i0, s]  ... wait, this isn't quite right.
    #
    # The Kronecker composition of two tensor decompositions:
    # If T1 has rank r1 with factors (U1,V1,W1) and T2 has rank r2 with factors (U2,V2,W2),
    # then T1 ⊗ T2 has rank r1*r2 with factors:
    # U[i1*n1+i2, (r1,r2)] = U1[i1,r1] * U2[i2,r2]
    # V[j1*n2+j2, (r1,r2)] = V1[j1,r1] * V2[j2,r2]  
    # W[k1*n3+k2, (r1,r2)] = W1[k1,r1] * W2[k2,r2]
    #
    # For 4x4x4 = (2x2x2) ⊗ (2x2x2):
    # n1=n2=2, so:
    # U4[2*i1+i0, r*7+s] = U2[i1, r] * U2[i0, s]
    # V4[2*j1+j0, r*7+s] = V2[j1, r] * V2[j0, s]
    # W4[2*k1+k0, r*7+s] = W2[k1, r] * W2[k0, s]
    #
    # For 4x4x8: B has 8 columns instead of 4.
    # 4x4x8 = 4x4x4 ⊗ (1x1x2) in the last dimension? No.
    #
    # 4x4x8: A is 4x4, B is 4x8, C is 8x4.
    # Write k = 2*k1 + k0 where k1 in {0,1,2,3}, k0 in {0,1}.
    # C[2*k1+k0, i] = sum_j A[i,j] * B[j, 2*k1+k0]
    #
    # The 4x4x4 part (k in {0,1,2,3} treated as 2x2 blocks) gives rank 49.
    # But we have 4 k-blocks, not 2. So it's not a simple Kronecker.
    #
    # Let me just build 4x4x8 directly:
    # i = 2*i1 + i0, j = 2*j1 + j0, k = 2*k1 + k0
    # i1,j1 in {0,1}, i0,j0,k0 in {0,1}, k1 in {0,1,2,3}
    #
    # T[i,j,k] = delta_{i,k} delta_{j,j} (trivially 1 when indices match)
    # 
    # Decomposition: sum_{r,s,t} U[i,r,s,t] * V[j,r,s,t] * W[k,r,s,t]
    # where r in {0..6} (block-level Strassen for i1,j1),
    #       s in {0..6} (element-level Strassen for i0,j0),
    #       t in {0..3} (schoolbook for k1 dimension, since k1 has 4 values)
    #
    # Wait, this doesn't quite work because k1 and k0 are coupled in B.
    #
    # Let me just use the straightforward approach:
    # For each k1 in {0,1,2,3}:
    #   C[:, 2*k1:2*k1+2] = A * B[:, 2*k1:2*k1+2]
    #   This is a 4x4x2 problem.
    #   4x4x2 = (2x2x2) ⊗ (2x2x1)? No.
    #   4x4x2: A is 4x4, B is 4x2, C is 2x4.
    #   i = 2*i1+i0, j = 2*j1+j0, k = k0 (k0 in {0,1})
    #   C[k0, i] = sum_j A[i,j] B[j,k0]
    #   = sum_{j1,j0} A[2*i1+i0, 2*j1+j0] B[2*j1+j0, k0]
    #   
    #   Block: C_block(i1, k0) = A_block(i1,0)*B_block(0,k0) + A_block(i1,1)*B_block(1,k0)
    #   where A_block is 2x2, B_block is 2x1, C_block is 1x2.
    #   This is a 2x2x1 + 2x2x1 = 2x2x2 problem? No, it's 2x2 * 2x1 = 2x1, summed.
    #   Each 2x2*2x1 has rank 2 (schoolbook) or can't be Strassen'd.
    #
    # I think the cleanest is just rank 49 for 4x4x4 and double it for 4x4x8 = 98.
    # But let me try to get 49 for 4x4x8 directly.
    #
    # 4x4x8 tensor: T[i, j, k] where i in 0..3, j in 0..3, k in 0..7
    # T[i,j,k] = 1 if we're computing C[k,i] = sum_j A[i,j]B[j,k]
    # So T[i*m+j, j*p+k, k*n+i] = 1
    #
    # Using Strassen at two levels:
    # Level 1 (block): i1 = i//2, j1 = j//2, k1 = k//2 (k1 in 0..3)
    # Level 2 (element): i0 = i%2, j0 = j%2, k0 = k%2
    #
    # For the block level, we have a 2x2x4 problem (i1 in {0,1}, j1 in {0,1}, k1 in {0,1,2,3}).
    # For the element level, we have a 2x2x2 problem.
    #
    # The 2x2x4 block problem: C1[i1, k1] = sum_j1 A1[i1,j1] B1[j1,k1]
    # This has rank at most 2*4 = 8 (schoolbook) or 7+7=14 (two Strassen).
    # Actually 2x2x4: A is 2x2, B is 2x4, C is 4x2.
    # C1[k1, i1] = sum_j1 A1[i1, j1] * B1[j1, k1]
    # This is just a 2x2 matrix times a 2x4 matrix. Rank = 2*2*4 = 16 schoolbook.
    # Can we do better? The tensor is T[i1, j1, k1] = delta_{i1, i1} delta_{j1,j1}... 
    # It's the 2x2x4 matmul tensor. Known rank is at most 2*4=8? No.
    # Actually for 2x2x4, the rank is at most 8 (schoolbook: 2*2*4/2=8? No, 2*2*4=16 terms).
    # Wait, schoolbook rank for 2x2x4 is 2*2*4 = 16? No, it's 2*4 = 8 (sum over 2 j1 values for each of 4 k1 and 2 i1).
    # Actually schoolbook: for each (i1, j1, k1), one term. There are 2*2*4 = 16 terms.
    # But we can group: for fixed (i1, k1), sum over j1 (2 terms). So 2*4*2 = 16 rank-1 terms.
    # Can we do better? The 2x2x4 matmul tensor has rank at most 2*2+2*2 = 8? 
    # Hmm, I don't think there's a known sub-schoolbook decomposition for 2x2x4.
    #
    # Let me just go with rank 98 (49 * 2) and try to reduce.
    
    # Build 4x4x4 decomposition via Kronecker of two 2x2x2 Strassen
    rank_444 = 49
    U4 = np.zeros((16, rank_444), dtype=int)
    V4 = np.zeros((16, rank_444), dtype=int)
    W4 = np.zeros((16, rank_444), dtype=int)
    
    for r in range(7):
        for s in range(7):
            col = r * 7 + s
            for i1 in range(2):
                for i0 in range(2):
                    i = 2 * i1 + i0
                    U4[i, col] = U2[i1, r] * U2[i0, s]
            for j1 in range(2):
                for j0 in range(2):
                    j = 2 * j1 + j0
                    V4[j, col] = V2[j1, r] * V2[j0, s]
            for k1 in range(2):
                for k0 in range(2):
                    k = 2 * k1 + k0
                    W4[k, col] = W2[k1, r] * W2[k0, s]
    
    # Verify 4x4x4
    # T[i*m+j, j*p+k, k*n+i] = 1 for n=m=p=4
    # sum_r U4[i*4+j, r] * V4[j*4+k, r] * W4[k*4+i, r] should = 1
    valid_444 = True
    for i in range(4):
        for j in range(4):
            for k in range(4):
                val = np.dot(U4[i*4+j, :], V4[j*4+k, :] * W4[k*4+i, :])
                if abs(val - 1.0) > 1e-9:
                    valid_444 = False
                    break
            if not valid_444:
                break
        if not valid_444:
            break
    
    if not valid_444:
        # Fallback to full schoolbook 4x4x8 directly to avoid broken 4x4x4 extension
        rank = 128
        U = np.zeros((16, rank), dtype=int)
        V = np.zeros((32, rank), dtype=int)
        W = np.zeros((32, rank), dtype=int)
        for i in range(4):
            for j in range(4):
                for k in range(8):
                    col = i*32 + j*8 + k
                    U[i*4+j, col] = 1
                    V[j*8+k, col] = 1
                    W[k*4+i, col] = 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Now extend to 4x4x8:
    # C[k, i] = sum_j A[i,j] B[j,k] for k in 0..7
    # Split k = 2*k1 + k0, k1 in {0,1,2,3}, k0 in {0,1}
    # C[2*k1+k0, i] = sum_j A[i,j] B[j, 2*k1+k0]
    #
    # The 4x4x4 decomposition handles k in {0,1,2,3} (i.e., k0=0 part for k1=0,1 and k0=1 for k1=0,1)
    # We need to handle k in {0,1,2,3,4,5,6,7}.
    #
    # Actually, let me just build 4x4x8 directly.
    # Use the 4x4x4 Strassen-Kronecker for the "first half" (k=0..3) and "second half" (k=4..7).
    # Each half is a 4x4x4 problem (A is 4x4, B_half is 4x4, C_half is 4x4).
    # Total rank: 49 + 49 = 98.
    
    if time.time() > deadline - 5:
        # Fallback: return schoolbook
        rank = 128
        U = np.zeros((16, rank), dtype=int)
        V = np.zeros((32, rank), dtype=int)
        W = np.zeros((32, rank), dtype=int)
        for i in range(4):
            for j in range(4):
                for k in range(8):
                    col = i*32 + j*8 + k
                    U[i*4+j, col] = 1
                    V[j*8+k, col] = 1
                    W[k*4+i, col] = 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    rank = 98
    U = np.zeros((16, rank), dtype=int)
    V = np.zeros((32, rank), dtype=int)
    W = np.zeros((32, rank), dtype=int)
    
    # First half: k=0..3, use 4x4x4 decomposition
    for r in range(49):
        col = r
        U[:, col] = U4[:, r]
        # V: B[j, k] for k=0..3 -> V4[j, r]
        V[:16, col] = V4[:, r]
        # W: C[k, i] for k=0..3 -> W4[k, r] where k is col index of C
        # W index: k*n + i = k*4 + i
        W[:16, col] = W4[:, r]
    
    # Second half: k=4..7, same A, different B columns
    for r in range(49):
        col = 49 + r
        U[:, col] = U4[:, r]
        # V: B[j, 4+k] for k=0..3 -> same pattern, offset by 4 in k
        V[16:, col] = V4[:, r]
        # W: C[4+k, i] -> offset by 4 in k
        W[16:, col] = W4[:, r]
    
    # Verify
    if time.time() < deadline - 10:
        valid = _verify_decomposition(U, V, W, 4, 4, 8)
        if not valid:
            # Fallback to schoolbook
            rank = 128
            U = np.zeros((16, rank), dtype=int)
            V = np.zeros((32, rank), dtype=int)
            W = np.zeros((32, rank), dtype=int)
            for i in range(4):
                for j in range(4):
                    for k in range(8):
                        col = i*32 + j*8 + k
                        U[i*4+j, col] = 1
                        V[j*8+k, col] = 1
                        W[k*4+i, col] = 1
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try rank reduction
    if time.time() < deadline - 20:
        U, V, W, rank = _reduce_rank(U, V, W, max_time=30.0)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
