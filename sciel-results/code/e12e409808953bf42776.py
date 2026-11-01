import numpy as np
import time

_START = time.time()
_DEADLINE = _START + 110.0

def _time_left():
    return _DEADLINE - time.time()

def construct():
    # 2x4 * 4x8 matrix multiplication tensor decomposition
    # Using Strassen for 2x2 blocks, but exploiting structure to reduce rank.
    
    # Key insight: The 2x4 x 4x8 product can be viewed as 8 independent
    # 2x1 * 4x1 inner products (one per column of C), each with 8 terms.
    # But we can group: C is 8x2, so 16 output entries.
    
    # Better approach: Split B into four 4x2 blocks B_0..B_3.
    # C_g (2x2) = A (2x4) * B_g (4x2) for g=0,1,2,3
    # A * B_g = [A_left * B_g_top + A_right * B_g_bottom]
    # where A_left = A[:, 0:2] (2x2), A_right = A[:, 2:4] (2x2)
    # B_g_top = B_g[0:2, :] (2x2), B_g_bottom = B_g[2:4, :] (2x2)
    
    # So C_g = A_left * B_g_top + A_right * B_g_bottom
    # Each 2x2*2x2 uses Strassen rank 7, total per g: 14
    # 4 groups: 56. Same as before.
    
    # Can we do better? Let's think about the structure differently.
    # n=2, m=4, p=8. The tensor is C_{k,i} = sum_j A_{i,j} B_{j,k}
    # where i in {0,1}, j in {0,1,2,3}, k in {0,...,7}
    
    # This is equivalent to: for each k, c_k = A @ B[:,k] (2x4 @ 4x1 = 2x1)
    # 8 independent 2x4 @ 4x1 multiplications.
    # Each 2x4 @ 4x1 has tensor rank... it's a bilinear map R^4 x R^4 -> R^2.
    # The tensor is T_{i,j} = sum over... wait, it's just A_i0*B_0 + A_i1*B_1 + A_i2*B_2 + A_i3*B_3 for i=0,1.
    # So c_0 = A[0,0]*B[0] + A[0,1]*B[1] + A[0,2]*B[2] + A[0,3]*B[3]
    #    c_1 = A[1,0]*B[0] + A[1,1]*B[1] + A[1,2]*B[2] + A[1,3]*B[3]
    
    # The tensor for one column is T_{i,j} with i in {0,1}, j in {0,1,2,3}.
    # This is a 2x4 matrix, rank at most 2 (over the field).
    # The tensor rank of this bilinear form: it's the sum of 4 rank-1 terms
    # (A_i0 * B_0) for i=0,1 plus... actually each term A_{i,j} * B_j contributes
    # to both i=0 and i=1 outputs.
    
    # T = sum_{j=0}^{3} (a_j) * (b_j) where a_j is a 2-vector (A[0,j], A[1,j]) and b_j is B[j].
    # This is already a rank-4 decomposition! Each term is outer product of a 2-vector and scalar.
    # But wait, in the tensor decomposition framework:
    # T_{i,j,k} for the full 2x4*4x8 is: T_{i,j,k} = delta(i,j,k) where the product is A_{i,j} * B_{j,k}
    # No, the multiplication tensor is T_{i,j,k} = 1 if the indices align properly.
    # For C_{i,k} = sum_j A_{i,j} B_{j,k}, the tensor is T_{i,j,k} = 1 for all valid i,j,k.
    # So T is an 8x32x16 tensor (flattened).
    
    # The schoolbook decomposition has rank 64: for each (i,j,k) triple, one term.
    # Wait no: for each output (i,k) and each j, we have term A_{i,j} * B_{j,k} * W_{i,k,j}.
    # That's 2*4*8 = 64 terms.
    
    # Strassen approach groups these into 2x2 blocks to save.
    # For 2x2 x 2x2, Strassen gets rank 7 instead of 8, saving 1 per 2x2 block.
    # We have 4*2 = 8 such blocks (4 column groups x 2 row pairs in A), saving 8 total.
    # 64 - 8 = 56. Confirmed.
    
    # Can we do better than 56? The lower bound for 2x4 x 4x8 is unknown in general,
    # but for small sizes we can try to find better decompositions.
    
    # Let me try a different grouping. Instead of 2x2 blocks, what about exploiting
    # the fact that n=2 is small?
    
    # For each column k of B (4x1), we compute c_k = A @ B_k (2x4 @ 4x1).
    # This is a bilinear map (R^4 x R^4 -> R^2) with 8 input dimensions total.
    # The tensor is T_{i,j} = 1 for all i in {0,1}, j in {0,1,2,3}, which has matrix rank 2.
    # So the tensor rank is exactly 2? No, that's the rank of the matrix T, not the tensor rank.
    # The tensor here is actually a 3-way tensor of size 2x4x... wait.
    
    # Let me reconsider. The full multiplication tensor for 2x4 * 4x8:
    # Inputs: A (8 values), B (32 values). Output: C (16 values).
    # C_{i,k} = sum_{j=0}^{3} A_{i,j} * B_{j,k}
    # In tensor form: C_{i,k} = sum_{j} A_{i,j} B_{j,k}
    # The tensor T has entries T_{i,j,k} = 1 for all i,j,k (since every combination contributes).
    # This is a 2 x 4 x 8 tensor, all ones.
    # The tensor rank of the all-ones 2x4x8 tensor:
    # It can be written as sum of min(2,4,8) = 2 rank-1 tensors?
    # T_{i,j,k} = u_1(i) v_1(j) w_1(k) + u_2(i) v_2(j) w_2(k)
    # For all-ones: yes! T = (1,1) ⊗ (1,1,1,1) ⊗ (1,1,...,1) * ... 
    # Wait: (1,1) ⊗ (1,1,1,1) ⊗ (1,...,1) gives T_{i,j,k} = 1*1*1 = 1 for all.
    # So the tensor rank is 1?! That can't be right for matrix multiplication.
    
    # I'm confusing the multiplication tensor with something else.
    # The matrix multiplication tensor for m x n * n x p is:
    # T_{i,j,k} = delta_{j, n-i} ... no.
    # Standard: (AB)_{i,k} = sum_j A_{i,j} B_{j,k}
    # The tensor is T_{i,j,k} = 1 if we're summing over j, i.e., T is the 3-tensor where
    # T_{i,j,k} = 1 for all valid i,j,k, and the contraction is over j.
    # The "rank" of this tensor in the sense of bilinear complexity is the minimum number
    # of scalar multiplications needed.
    
    # For 2x4 x 4x8, the bilinear complexity is what we want to minimize.
    # Schoolbook: 2*4*8 = 64.
    # Strassen on 2x2 blocks: 56 as computed.
    
    # Let me verify: is there a known better bound?
    # For 2xn x nx(2n), the complexity might be lower.
    # Here 2x4 x 4x8, so n=4, and p=2n=8.
    
    # Actually, let me just try to construct a rank-48 or lower decomposition.
    # One approach: use the fact that for 2x4 x 4x8, we can write:
    # C = A @ B where A is 2x4, B is 4x8.
    # C is 2x8.
    
    # Think of A as [A1 | A2] where A1, A2 are 2x2.
    # B as [B1; B2] where B1, B2 are 2x8.
    # Then C = A1 @ B1 + A2 @ B2 (2x2 @ 2x8 + 2x2 @ 2x8 = 2x8).
    # Each 2x2 @ 2x8: use Strassen-like. 2x2 @ 2x8 can be seen as 4 independent 2x2 @ 2x2 (splitting the 8 columns into 4 pairs).
    # 4 * 7 = 28 per A-block, total 2 * 28 = 56. Same as before.
    
    # Alternative: 2x2 @ 2x8 directly. The tensor is 2x2 x 2x8 = 4x2x8 tensor of all ones.
    # Can we do better than 4*7=28?
    # For 2x2 x 2x8, schoolbook is 4*2*8/2 = ... 2*2*8 = 32. Strassen gives 28.
    # Is there a rank-24 decomposition for 2x2 x 2x8?
    # That would be 6 per 2x2 block on average, which is below Strassen's 7. Unlikely.
    
    # So 56 seems to be the best we can get with standard techniques.
    # Let me just return the rank-56 solution with a clean implementation.
    
    n, m, p = 2, 4, 8
    A_dim = n * m   # 8
    B_dim = m * p   # 32
    C_dim = p * n   # 16
    
    rank = 56
    U = np.zeros((A_dim, rank), dtype=float)
    V = np.zeros((B_dim, rank), dtype=float)
    W = np.zeros((C_dim, rank), dtype=float)
    
    col = 0
    for g in range(4):
        for ab in range(2):
            if ab == 0:
                x00, x01 = 0, 1
                x10, x11 = 4, 5
                y00, y01 = g*2, g*2+1
                y10, y11 = 8+g*2, 8+g*2+1
            else:
                x00, x01 = 2, 3
                x10, x11 = 6, 7
                y00, y01 = 16+g*2, 16+g*2+1
                y10, y11 = 24+g*2, 24+g*2+1
            
            c00 = 4*g + 0
            c01 = 4*g + 2
            c10 = 4*g + 1
            c11 = 4*g + 3
            
            # M1
            U[x00, col] = 1; U[x11, col] = 1
            V[y00, col] = 1; V[y11, col] = 1
            W[c00, col] = 1; W[c11, col] = 1
            col += 1
            # M2
            U[x10, col] = 1; U[x11, col] = 1
            V[y00, col] = 1
            W[c10, col] = 1; W[c11, col] = -1
            col += 1
            # M3
            U[x00, col] = 1
            V[y01, col] = 1; V[y11, col] = -1
            W[c01, col] = 1; W[c11, col] = 1
            col += 1
            # M4
            U[x11, col] = 1
            V[y10, col] = 1; V[y00, col] = -1
            W[c00, col] = 1; W[c10, col] = 1
            col += 1
            # M5
            U[x00, col] = 1; U[x01, col] = 1
            V[y11, col] = 1
            W[c00, col] = -1; W[c01, col] = 1
            col += 1
            # M6
            U[x10, col] = 1; U[x00, col] = -1
            V[y00, col] = 1; V[y01, col] = 1
            W[c11, col] = 1
            col += 1
            # M7
            U[x01, col] = 1; U[x11, col] = -1
            V[y10, col] = 1; V[y11, col] = 1
            W[c00, col] = 1
            col += 1
    
    assert col == rank
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
