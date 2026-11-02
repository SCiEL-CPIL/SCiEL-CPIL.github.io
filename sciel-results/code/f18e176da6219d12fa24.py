import time
import numpy as np

DIMENSIONS = [2, 2, 9]

def _verify(U, V, W, n, m, p, seed=12345):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = len(U[0]) if isinstance(U, list) else U.shape[1]
    Ua = np.array(U, dtype=float).reshape(n * m, rank)
    Va = np.array(V, dtype=float).reshape(m * p, rank)
    Wa = np.array(W, dtype=float).reshape(p * n, rank)
    
    rng = np.random.default_rng(seed)
    for _ in range(5):
        A = rng.standard_normal((n, m))
        B = rng.standard_normal((m, p))
        C_expected = A @ B
        
        # Compute C from decomposition
        # A_flat[i*m+j], B_flat[j*p+k]
        A_flat = A.reshape(n * m)
        B_flat = B.reshape(m * p)
        
        # C_flat[k*n+i] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Actually: C[i,k] = sum_j A[i,j]*B[j,k]
        # In tensor form: C[i,k] = sum_{i,j,k} A[i,j]*B[j,k]*T[i,j,k] where T is the multiplication tensor
        # Rank-1 term r contributes: sum_{i,j,k} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] * A[i,j] * B[j,k]
        # This equals sum_{i,k} (sum_j A[i,j]*B[j,k]*...) -- wait let me think again.
        
        # The tensor T[i,j,k] = delta_{i,i}*delta_{j,j}*delta_{k,k}... no.
        # Matrix multiplication tensor: T_{i,j,k} where the bilinear form is sum_{i,j,k} T_{i,j,k} * A_{i,j} * B_{j,k} = C_{i,k}
        # So T_{i,j,k} = 1 if the indices match, i.e., it's the identity in the appropriate sense.
        # Actually for C = AB, C_{i,k} = sum_j A_{i,j} B_{j,k}, so the tensor has T_{i,j,k} = 1 for all i,j,k (with appropriate indexing).
        # Wait no. The tensor is T[i,j,k] such that C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k].
        # So T[i,j,k] = delta_{j,j}... no, T[i,j,k] = 1 for all valid i,j,k since each term A[i,j]*B[j,k] contributes to C[i,k].
        
        # For a rank-1 decomposition term r:
        # C[i,k] += (sum_j A[i,j]*U[i*m+j, r]) * (sum_j B[j,k]*V[j*p+k, r]) ... no that's not right either.
        
        # Let me re-read the convention: U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
        # W[k*n+i, r] is the entry for output index (k, i).
        # The decomposition says: the multiplication tensor T = sum_r u_r ⊗ v_r ⊗ w_r
        # where u_r is a vector in R^{n*m}, v_r in R^{m*p}, w_r in R^{p*n}.
        # The bilinear form is: sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] = sum_r (sum_{i,j} U[i*m+j, r] A[i,j]) (sum_{j,k} V[j*p+k, r] B[j,k]) (sum_{i,k} W[k*n+i, r] ... )
        
        # Hmm, actually for a rank-1 tensor u ⊗ v ⊗ w, the bilinear form contribution is:
        # (sum_{i,j} u[i,j] A[i,j]) * (sum_{j,k} v[j,k] B[j,k]) * ... but that's not quite right for matrix mult.
        
        # Let me think differently. The multiplication tensor for C = AB where A is n×m, B is m×p:
        # T is a 3-tensor with indices (i, j, k) where i in [0,n), j in [0,m), k in [0,p).
        # T[i,j,k] = 1 (it's just 1 for all entries since C[i,k] = sum_j A[i,j]B[j,k]).
        # Wait no, T should be such that the trilinear form <T, A⊗B⊗C> gives the constraint.
        
        # Actually, the standard formulation: the matrix multiplication tensor T_{n,m,p} has entries T[i,j,k] = 1 for all i,j,k.
        # A rank-r decomposition is T = sum_{s=1}^r u_s ⊗ v_s ⊗ w_s where:
        # u_s ∈ R^{n×m}, v_s ∈ R^{m×p}, w_s ∈ R^{p×n}
        # and T[i,j,k] = sum_s u_s[i,j] * v_s[j,k] * w_s[k,i] ... 
        # No wait. Let me look at this more carefully.
        
        # The convention given: U has shape (n*m, rank), so U[i*m+j, r] corresponds to u_r[i,j].
        # V has shape (m*p, rank), so V[j*p+k, r] corresponds to v_r[j,k].
        # W has shape (p*n, rank), so W[k*n+i, r] corresponds to w_r[k,i].
        
        # For the decomposition to be exact: for all i,j,k, T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # where T[i,j,k] = 1 for all i,j,k (since in C=AB, every A[i,j]*B[j,k] product contributes to C[i,k]).
        
        # Wait, that can't be right because then T would be the all-ones tensor which has rank 1.
        # Let me reconsider.
        
        # Actually I think the tensor for matrix multiplication is defined differently. Let me reconsider.
        # C[i,k] = sum_j A[i,j] B[j,k]
        # So the "tensor" that encodes this is T where T[i,j,k] = δ_{j,j}... 
        # I think the correct statement is: T[i,j,k] = 1 for all i,j,k, and the bilinear form is
        # sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] = sum_{i,k} C[i,k]
        # But that's not a bilinear form in the right sense.
        
        # Let me just verify numerically. For a rank-1 term, the contribution to C is:
        # For each r, define:
        # a_r[i,j] = U[i*m+j, r]
        # b_r[j,k] = V[j*p+k, r]  
        # c_r[k,i] = W[k*n+i, r]
        # The rank-1 tensor is a_r ⊗ b_r ⊗ c_r.
        # The decomposition says: the multiplication tensor = sum_r a_r ⊗ b_r ⊗ c_r
        # The multiplication tensor T satisfies: T[i,j,k] * A[i,j] * B[j,k] summed over i,j,k gives sum_{i,k} C[i,k]... 
        # No, I think the right interpretation is:
        # C[i,k] = sum_{j} sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] * A[i,j] * B[j,k]
        # For this to equal sum_j A[i,j] B[j,k] for all i,k, we need:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
        # Wait no, that would mean the tensor is all ones.
        
        # I think I'm overcomplicating this. Let me just check: for the schoolbook decomposition,
        # U[i*m+j, i*m+j*p+k] = 1, V[j*p+k, i*m+j*p+k] = 1, W[k*n+i, i*m+j*p+k] = 1
        # Then sum_r U[i*m+j', r] V[j'*p+k', r] W[k'*n+i', r] A[i',j'] B[j',k']
        # Only non-zero when j'=j, k'=k, i'=i (from U and W), giving A[i,j]B[j,k]. ✓
        
        # So the verification is: for all i,k, C[i,k] = sum_{j} sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] A[i,j] B[j,k]
        # Let me just compute this directly.
        
        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    # sum over r
                    vals_u = [U[i*m+j][r] for r in range(rank)]
                    vals_v = [V[j*p+k][r] for r in range(rank)]
                    vals_w = [W[k*n+i][r] for r in range(rank)]
                    inner = sum(vals_u[r] * vals_v[r] * vals_w[r] for r in range(rank))
                    C_computed[i, k] += inner * A[i, j] * B[j, k]
        
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10 second safety margin
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 36
    
    # First, try to build a rank-33 decomposition using Strassen for 2x2 blocks
    # Split the 9 columns into three groups of 3: B = [B1, B2, B3] each 2x3
    # C = A @ B = [A@B1, A@B2, A@B3]
    # For each 2x2 @ 2x3 multiplication, we can use rank-11 (Strassen for 2x2 part + 3 scalar terms)
    
    # Actually, let me think about this differently.
    # 2x2 @ 2x3: C[i,k] = sum_j A[i,j] B[j,k] for i=0,1 and k=0,1,2
    # This is 6 output entries. Schoolbook rank = 2*2*3 = 12.
    # Using Strassen-like: treat it as 3 separate 2x2@2x1 multiplications plus cross terms?
    # Or: think of 2x2 @ 2x3 as computing 3 columns of 2x2@2x1.
    # Each 2x2@2x1 has schoolbook rank 4, Strassen rank... well it's not square.
    
    # Alternative: 2x2 @ 2x3 can be viewed as 2x2 @ (2x2 | 2x1) = (2x2@2x2) | (2x2@2x1)
    # 2x2@2x2 has rank 7 (Strassen), 2x2@2x1 has rank 4 (schoolbook, since it's just 4 scalar mults)
    # But they share the A matrix, so we can potentially do better.
    # Actually no, they're independent in terms of the B part.
    # Total: 7 + 4 = 11 for 2x2@2x3 if we split into 2x2@2x2 and 2x2@2x1.
    
    # Wait, but 2x2@2x1: C[i] = A[i,0]*B[0] + A[i,1]*B[1], 2 outputs. Rank 4 schoolbook.
    # Can we do better? The tensor has 2*2*1 = 4 entries, all = 1. Rank is 4 (can't do better since each output needs both products).
    # Actually for 2x2@2x1, it's just two independent sums of 2 products each. Min rank = 4.
    
    # So 2x2@2x3: split B into B_left (2x2) and B_right (2x1).
    # C_left = A @ B_left: rank 7 (Strassen)
    # C_right = A @ B_right: rank 4
    # Total: 11.
    
    # Now for 2x2@2x9: split into three 2x2@2x3 blocks.
    # Each block: rank 11. Total: 33.
    # But can we do better by not splitting? The full tensor is 2x2@2x9 with schoolbook rank 36.
    # If we can find a rank < 33 decomposition, that would be better.
    
    # Let me implement the rank-33 decomposition and verify it.
    
    def strassen_2x2(B2):
        """Return rank-1 tensors (u, v, w) for 2x2 @ 2x2 using Strassen.
        B2 is 2x2. Returns lists of (u_vec, v_vec, w_vec) where:
        u_vec is length 4 (i*m+j), v_vec is length 4 (j*p+k), w_vec is length 4 (k*n+i)
        with n=m=p=2.
        """
        # Strassen:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A00+A01)B11
        # M3 = (A10+A11)B00
        # M4 = A11(B10+B00)
        # M5 = (A00-A11)(B11+B10)
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        # C00 = M1+M4-M2+M6
        # C01 = M2+M5
        # C10 = M3+M7
        # C11 = M1+M5-M3-M7
        
        # For each rank-1 term, we need u (length 4: A indices 00,01,10,11), 
        # v (length 4: B indices 00,10,01,11 -- wait, V[j*p+k] with j=0,1 k=0,1 so order is 00,01,10,11),
        # w (length 4: C indices k*n+i, so 00,10,01,11 -- k=0,i=0; k=0,i=1; k=1,i=0; k=1,i=1)
        
        # Let me be careful with indexing:
        # U[i*m+j]: i=0,j=0 -> 0; i=0,j=1 -> 1; i=1,j=0 -> 2; i=1,j=1 -> 3
        # V[j*p+k]: j=0,k=0 -> 0; j=0,k=1 -> 1; j=1,k=0 -> 2; j=1,k=1 -> 3
        # W[k*n+i]: k=0,i=0 -> 0; k=0,i=1 -> 1; k=1,i=0 -> 2; k=1,i=1 -> 3
        
        terms = []
        
        # M1 = (A00+A11)(B00+B11)
        # u: A00=1, A11=1 -> [1,0,0,1]
        # v: B00=1, B11=1 -> [1,0,0,1]
        # w: contributes to C00 (+1) and C11 (+1) -> [1,0,0,1]
        terms.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))
        
        # M2 = (A00+A01)B11
        # u: A00=1, A01=1 -> [1,1,0,0]
        # v: B11=1 -> [0,0,0,1]
        # w: C01=+1, C00=-1 -> [ -1, 1, 0, 0]
        terms.append(([1,1,0,0], [0,0,0,1], [-1,1,0,0]))
        
        # M3 = (A10+A11)B00
        # u: A10=1, A11=1 -> [0,0,1,1]
        # v: B00=1 -> [1,0,0,0]
        # w: C10=+1, C11=-1 -> [0,0,1,-1]
        terms.append(([0,0,1,1], [1,0,0,0], [0,0,1,-1]))
        
        # M4 = A11(B10+B00)
        # u: A11=1 -> [0,0,0,1]
        # v: B10=1, B00=1 -> [1,0,1,0]
        # w: C00=+1 -> [1,0,0,0]
        terms.append(([0,0,0,1], [1,0,1,0], [1,0,0,0]))
        
        # M5 = (A00-A11)(B11+B10)
        # u: A00=1, A11=-1 -> [1,0,0,-1]
        # v: B11=1, B10=1 -> [0,0,1,1]
        # w: C01=+1, C11=+1 -> [0,1,0,1]
        terms.append(([1,0,0,-1], [0,0,1,1], [0,1,0,1]))
        
        # M6 = (A10-A00)(B00+B01)
        # u: A10=1, A00=-1 -> [-1,0,1,0]
        # v: B00=1, B01=1 -> [1,1,0,0]
        # w: C00=+1 -> [1,0,0,0]
        terms.append(([-1,0,1,0], [1,1,0,0], [1,0,0,0]))
        
        # M7 = (A01-A11)(B10+B11)
        # u: A01=1, A11=-1 -> [0,1,0,-1]
        # v: B10=1, B11=1 -> [0,0,1,1]
        # w: C10=+1, C11=-1 -> [0,0,1,-1]
        terms.append(([0,1,0,-1], [0,0,1,1], [0,0,1,-1]))
        
        return terms
    
    # Build rank-33 decomposition for 2x2@2x9
    # Split the 9 columns of B into 3 groups of 3: columns 0-1, 2-3, 4-5, ... wait, 9 is not divisible by 2.
    # Let me split into: first 2 columns (2x2), next 2 columns (2x2), next 2 columns (2x2), last 3 columns... 
    # Hmm, 9 = 2+2+2+3. That gives 3 blocks of 2x2@2x2 (rank 7 each) and 1 block of 2x2@2x3 (rank 11).
    # Total: 3*7 + 11 = 32. Even better!
    
    # Or: 9 = 4+5? No, let me think in terms of splitting B's columns.
    # Option 1: [2,2,2,3] -> 3*7 + 11 = 32
    # Option 2: [3,3,3] -> 3*11 = 33
    # Option 3: [2,2,5] -> 2*7 + rank(2x2@2x5)
    #   2x2@2x5: split as 2x2@2x2 + 2x2@2x3 = 7+11=18? Or can we do better?
    #   Actually 2x2@2x5 schoolbook rank = 20. Can we do rank 17? 2x2@2x4 = 2*7=14, 2x2@2x1=4, total 18.
    #   Hmm, 14+4=18. So 2*7+18 = 32. Same.
    
    # Let me go with [2,2,2,3] giving rank 32.
    
    def build_rank32():
        """Build rank-32 decomposition for 2x2@2x9."""
        rank = 32
        U = [[0]*rank for _ in range(n*m)]  # 4 x 32
        V = [[0]*rank for _ in range(m*p)]  # 18 x 32
        W = [[0]*rank for _ in range(p*n)]  # 18 x 32
        
        col = 0
        # Column groups: [0,1], [2,3], [4,5], [6,7,8]
        groups = [(0,2), (2,4), (4,6), (6,9)]
        
        for group_start, group_end in groups:
            group_width = group_end - group_start
            if group_width == 2:
                # Use Strassen 2x2@2x2
                terms = strassen_2x2(None)
                for u, v, w in terms:
                    for i in range(4):
                        U[i][col] = u[i]
                    for j in range(4):
                        # v[j] corresponds to B[j//2, j%2] in the local 2x2
                        # Global B index: j_local = j, global row = 0 or 1, global col = group_start + (j%2)
                        # V[j*p+k] where j is row of B (0 or 1), k is col of B
                        local_j = j // 2  # 0 or 1
                        local_k = j % 2   # 0 or 1
                        global_k = group_start + local_k
                        V[local_j * p + global_k][col] = v[j]
                    for idx in range(4):
                        # w[idx]: idx = k*n+i, local k=idx//2, local i=idx%2
                        local_k = idx // 2
                        local_i = idx % 2
                        global_k = group_start + local_k
                        W[global_k * n + local_i][col] = w[idx]
                    col += 1
            elif group_width == 3:
                # Use 2x2@2x3: split into 2x2@2x2 (cols 0,1) + 2x2@2x1 (col 2)
                # Strassen for first 2 cols
                terms = strassen_2x2(None)
                for u, v, w in terms:
                    for i in range(4):
                        U[i][col] = u[i]
                    for j in range(4):
                        local_j = j // 2
                        local_k = j % 2
                        global_k = group_start + local_k
                        V[local_j * p + global_k][col] = v[j]
                    for idx in range(4):
                        local_k = idx // 2
                        local_i = idx % 2
                        global_k = group_start + local_k
                        W[global_k * n + local_i][col] = w[idx]
                    col += 1
                # Schoolbook for 3rd col (col 2 within group)
                # 2x2@2x1: 4 terms
                for i in range(2):
                    for j in range(2):
                        U[i*m+j][col] = 1
                        V[j*p + (group_start+2)][col] = 1
                        W[(group_start+2)*n + i][col] = 1
                        col += 1
        
        assert col == rank, f"Expected rank {rank}, got {col}"
        return U, V, W
    
    U32, V32, W32 = build_rank32()
    
    if time.time() < deadline:
        if _verify(U32, V32, W32, n, m, p):
            return {"U": U32, "V": V32, "W": W32}
    
    # Fallback: schoolbook rank 36
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
