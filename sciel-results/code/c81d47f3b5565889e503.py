"""Matrix multiplication 4x5x6: search for rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 135.0  # 15s safety margin before 150s kill

    def elapsed():
        return time.time() - start_time

    def time_remaining():
        return DEADLINE - elapsed()

    n, m, p = DIMENSIONS
    N = n * m  # 20
    M = m * p  # 30
    P = p * n  # 24

    def schoolbook():
        rank = n * m * p
        U = np.zeros((N, rank))
        V = np.zeros((M, rank))
        W = np.zeros((P, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def strassen_4x4():
        """Return rank-7 Strassen decomposition for 4x4x4 matrix mult.
        Returns (U, V, W) each of shape (16, 7) such that
        T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r] computes C = A*B for 4x4 matrices.
        Index convention: a = i*4+j, b = j*4+k, c = k*4+i
        """
        # Strassen's 7 products for 2x2, then extend to 4x4 via block structure
        # For 4x4: split into four 2x2 blocks
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        # C00 = M1+M4-M5+M7
        # C01 = M3+M5
        # C10 = M2+M4
        # C11 = M1-M2+M3+M6
        
        # Build the tensor for 4x4x4 multiplication
        target_4 = np.zeros((16, 16, 16))
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    a = i * 4 + j
                    b = j * 4 + k
                    c = k * 4 + i
                    target_4[a, b, c] = 1.0
        
        # Use Strassen structure: 7 rank-1 tensors
        # Each rank-1 term: U[:,r], V[:,r], W[:,r]
        U = np.zeros((16, 7))
        V = np.zeros((16, 7))
        W = np.zeros((16, 7))
        
        # Define the 7 Strassen products in terms of block indices
        # Block (row, col) of 2x2 blocks: block row br in {0,1}, block col bc in {0,1}
        # Element (i,j) in block (br,bc) has global i = br*2 + li, j = bc*2 + lj
        
        # M1 = (A00+A11)(B00+B11)
        # U: rows for A00 (i in 0,1; j in 0,1) and A11 (i in 2,3; j in 2,3)
        # V: cols for B00 (j in 0,1; k in 0,1) and B11 (j in 2,3; k in 2,3)
        # W: result C00 (k in 0,1; i in 0,1) and C11 (k in 2,3; i in 2,3)
        for li in range(2):
            for lj in range(2):
                # A00 element: i=li, j=lj -> a = li*4+lj
                U[li * 4 + lj, 0] = 1  # A00
                # A11 element: i=2+li, j=2+lj -> a = (2+li)*4+(2+lj)
                U[(2 + li) * 4 + (2 + lj), 0] = 1  # A11
                # B00 element: j=lj, k=li (since in B, row=j, col=k) -> b = lj*4+li
                V[lj * 4 + li, 0] = 1  # B00
                # B11 element: j=2+lj, k=2+li -> b = (2+lj)*4+(2+li)
                V[(2 + lj) * 4 + (2 + li), 0] = 1  # B11
                # C00: k=li, i=lj -> c = li*4+lj (but wait, C index is c = k*4+i)
                # For C00 block: k in 0,1; i in 0,1
                # The result element C[i,j] where i is row, j is col
                # In our convention: c = k*4+i, and we need C00[i,j] = sum_k A[i,k]*B[k,j]
                # For the M1 contribution to C00: we add M1 to C00
                # C00[i,j] gets M1[i,j] where i,j are within 0,1
                # c index: k*4+i, but here we're computing the product, so:
                # The term contributes to C[i, j] = sum_k A[i,k] B[k,j]
                # In tensor form: T[a,b,c] where a=i*4+j (A row i, col j), b=j*4+k (B row j, col k), c=k*4+i (C col k, row i)
                # So for M1 contributing to C00[i,j]: a = i*4+j (A), b = j*4+k (B), c = k*4+i (C)
                # But M1 = (A00+A11)(B00+B11), so it's a sum over the 2x2 blocks
                # Actually, let me reconsider. The Strassen decomposition gives 7 rank-1 tensors.
                # Each rank-1 tensor T_r[a,b,c] = U[a,r]*V[b,r]*W[c,r]
                # For M1: U picks out A00 and A11 entries, V picks out B00 and B11 entries, W picks out C00 and C11 entries
                # Specifically: M1[i,j] = sum_k (A00[i,j]+A11[i,j]) * (B00[k,j]+B11[k,j]) ... no this is getting complicated.
                
                # Let me just use a direct approach: build the 4x4x4 tensor and use ALS to find rank-7
                pass
        
        # Actually, let me just use ALS to find a rank-7 decomposition of the 4x4x4 tensor
        # This is more reliable than trying to manually construct Strassen
        return None  # Will use ALS instead

    def to_lists(mat):
        return mat.tolist()

    def verify(result, atol=1e-6):
        U = np.array(result["U"])
        V = np.array(result["V"])
        W = np.array(result["W"])
        if U.ndim != 2 or V.ndim != 2 or W.ndim != 2:
            return False
        if U.shape[1] != V.shape[1] or V.shape[1] != W.shape[1]:
            return False
        if U.shape[0] != N or V.shape[0] != M or W.shape[0] != P:
            return False
        T = np.einsum('ar,br,cr->abc', U, V, W)
        target = np.zeros((N, M, P))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    target[a, b, c] = 1.0
        return np.allclose(T, target, atol=atol)

    # Build target tensor once
    target = np.zeros((N, M, P))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i * m + j, j * p + k, k * n + i] = 1.0

    # Start with schoolbook
    best = schoolbook()
    best_rank = n * m * p

    def build_4x4x4_target():
        """Build the 4x4x4 matrix multiplication tensor."""
        t = np.zeros((16, 16, 16))
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    a = i * 4 + j
                    b = j * 4 + k
                    c = k * 4 + i
                    t[a, b, c] = 1.0
        return t

    def als_decompose_target(target_t, rank, max_iter=100, tol=1e-12, num_restarts=3):
        """ALS decomposition of a given target tensor."""
        Nt, Mt, Pt = target_t.shape
        best_err = np.inf
        best_UVW = None
        
        for restart in range(num_restarts):
            U = np.random.randn(Nt, rank) * 0.1
            V = np.random.randn(Mt, rank) * 0.1
            W = np.random.randn(Pt, rank) * 0.1
            
            prev_err = np.inf
            for iteration in range(max_iter):
                # Update U
                H_U = np.einsum('br,cr->bcr', V, W).reshape(Mt * Pt, rank)
                target_flat = target_t.reshape(Nt, Mt * Pt)
                try:
                    U_new, _, _, _ = np.linalg.lstsq(H_U, target_flat.T, rcond=None)
                    U = U_new.T
                except:
                    pass
                
                # Update V
                H_V = np.einsum('ar,cr->acr', U, W).reshape(Nt * Pt, rank)
                target_v = target_t.transpose(1, 0, 2).reshape(Mt, Nt * Pt)
                try:
                    V_new, _, _, _ = np.linalg.lstsq(H_V, target_v.T, rcond=None)
                    V = V_new.T
                except:
                    pass
                
                # Update W
                H_W = np.einsum('ar,br->abr', U, V).reshape(Nt * Mt, rank)
                target_w = target_t.transpose(2, 0, 1).reshape(Pt, Nt * Mt)
                try:
                    W_new, _, _, _ = np.linalg.lstsq(H_W, target_w.T, rcond=None)
                    W = W_new.T
                except:
                    pass
                
                T = np.einsum('ar,br,cr->abc', U, V, W)
                err = np.linalg.norm(T - target_t)
                if abs(prev_err - err) < tol:
                    break
                prev_err = err
            
            if err < best_err:
                best_err = err
                best_UVW = (U.copy(), V.copy(), W.copy())
        
        return best_UVW, best_err

    def als_decompose(target, rank, max_iter=50, tol=1e-10, polish_iters=10):
        """ALS decomposition with better initialization, convergence check, and polish."""
        # Initialize with small random values
        U = np.random.randn(N, rank) * 0.5
        V = np.random.randn(M, rank) * 0.5
        W = np.random.randn(P, rank) * 0.5

        prev_err = np.inf
        for iteration in range(max_iter):
            if time_remaining() < 20:
                break

            # Update U: fix V, W
            # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            # For fixed V,W: minimize ||target - sum_r U[:,r] (V[:,r] ⊗ W[:,r])^T||_F
            # Build H_U efficiently: H_U[b*P+c, r] = V[b,r] * W[c,r]
            H_U = np.einsum('br,cr->bcr', V, W).reshape(M * P, rank)
            target_flat = target.reshape(N, M * P)
            try:
                U_new, _, _, _ = np.linalg.lstsq(H_U, target_flat.T, rcond=None)
                U = U_new.T  # (N, rank)
            except:
                pass

            # Update V: fix U, W
            # H_V[r, a*P+c] = U[a,r] * W[c,r]
            H_V = np.einsum('ar,cr->acr', U, W).reshape(N * P, rank)
            target_v = target.transpose(1, 0, 2).reshape(M, N * P)
            try:
                V_new, _, _, _ = np.linalg.lstsq(H_V, target_v.T, rcond=None)
                V = V_new.T  # (M, rank)
            except:
                pass

            # Update W: fix U, V
            # H_W[r, a*M+b] = U[a,r] * V[b,r]
            H_W = np.einsum('ar,br->abr', U, V).reshape(N * M, rank)
            target_w = target.transpose(2, 0, 1).reshape(P, N * M)
            try:
                W_new, _, _, _ = np.linalg.lstsq(H_W, target_w.T, rcond=None)
                W = W_new.T  # (P, rank)
            except:
                pass

            # Check convergence
            T = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.linalg.norm(T - target)
            if abs(prev_err - err) < tol:
                break
            prev_err = err

        # Final high-precision polish pass: tighten tolerance and do a few more iterations
        if time_remaining() > 25:
            for _ in range(polish_iters):
                if time_remaining() < 20:
                    break
                H_U = np.einsum('br,cr->bcr', V, W).reshape(M * P, rank)
                target_flat = target.reshape(N, M * P)
                try:
                    U_new, _, _, _ = np.linalg.lstsq(H_U, target_flat.T, rcond=None)
                    U = U_new.T
                except:
                    pass
                H_V = np.einsum('ar,cr->acr', U, W).reshape(N * P, rank)
                target_v = target.transpose(1, 0, 2).reshape(M, N * P)
                try:
                    V_new, _, _, _ = np.linalg.lstsq(H_V, target_v.T, rcond=None)
                    V = V_new.T
                except:
                    pass
                H_W = np.einsum('ar,br->abr', U, V).reshape(N * M, rank)
                target_w = target.transpose(2, 0, 1).reshape(P, N * M)
                try:
                    W_new, _, _, _ = np.linalg.lstsq(H_W, target_w.T, rcond=None)
                    W = W_new.T
                except:
                    pass

        return U, V, W

    def try_rank(r, num_trials=5):
        """Try to find a decomposition of rank r."""
        if time_remaining() < 25:
            return None
        for trial in range(num_trials):
            if time_remaining() < 20:
                break
            U, V, W = als_decompose(target, r, max_iter=40)
            result = {"U": U, "V": V, "W": W}
            if verify(result):
                return result
        return None

    # Try ranks below schoolbook, starting from 119 down
    for r in range(best_rank - 1, max(best_rank - 25, 95), -1):
        if time_remaining() < 35:
            break
        result = try_rank(r, num_trials=3)
        if result is not None:
            best = result
            best_rank = r
            print(f"Found decomposition with rank {r}")
            continue

    # Block partitioning strategy: split middle index j into 4+1
    # A is 4x5, B is 5x6, C is 4x6
    # Split j: j in 0..3 (4 values) and j=4 (1 value)
    # C = A[:,:4]*B[:,:4,:] + A[:,4:5]*B[:,4:5,:]
    # Block 1: 4x4x4 multiplication (A[:,:4] * B[:,:4,:]) -> rank 7 via Strassen/ALS
    # Block 2: 4x1x6 multiplication (A[:,4:5] * B[:,4:5,:]) -> rank 4*1*6=24 via schoolbook
    # Total: 7 + 24 = 31? No, that's not right because the tensors don't simply add.
    
    # Actually, the full tensor is:
    # T[a,b,c] where a = i*5+j (A row i, col j), b = j*6+k (B row j, col k), c = k*4+i (C col k, row i)
    # T[i*5+j, j*6+k, k*4+i] = 1
    
    # Split j into two parts: j1 in 0..3 and j2=4
    # Part 1: j1 in 0..3: T[i*5+j1, j1*6+k, k*4+i] = 1 for j1=0,1,2,3
    # Part 2: j2=4: T[i*5+4, 4*6+k, k*4+i] = 1 for j2=4
    
    # Part 1 is a 4x4x4 tensor (with indices remapped)
    # Part 2 is a 4x1x6 tensor (with indices remapped)
    
    # Let's build these sub-tensors and decompose them separately, then combine
    
    def build_subtensor_4x4():
        """Build the 4x4x4 sub-tensor for j=0,1,2,3."""
        # Original indices: a = i*5+j, b = j*6+k, c = k*4+i
        # For j in 0..3: a ranges over {0..3, 5..8, 10..13, 15..18} (i*5+j for i=0..3, j=0..3)
        # b ranges over {0..3, 6..9, 12..15, 18..21} (j*6+k for j=0..3, k=0..5)
        # c ranges over {0..3, 4..7, 8..11, 12..15, 16..19, 20..23} (k*4+i for k=0..5, i=0..3)
        
        # Remap to compact indices:
        # a_compact = i*4 + j (for i=0..3, j=0..3) -> 0..15
        # b_compact = j*6 + k (for j=0..3, k=0..5) -> but this isn't compact
        # Actually, let's just use the original indices and zero out the unused ones
        
        sub = np.zeros((N, M, P))
        for i in range(4):
            for j in range(4):  # j = 0,1,2,3
                for k in range(6):
                    a = i * 5 + j
                    b = j * 6 + k
                    c = k * 4 + i
                    sub[a, b, c] = 1.0
        return sub
    
    def build_subtensor_1x6():
        """Build the 4x1x6 sub-tensor for j=4."""
        sub = np.zeros((N, M, P))
        for i in range(4):
            for k in range(6):
                a = i * 5 + 4  # j=4
                b = 4 * 6 + k  # j=4
                c = k * 4 + i
                sub[a, b, c] = 1.0
        return sub
    
    # Decompose sub-tensor 1 (4x4x4 part) with rank 7
    if time_remaining() > 40:
        sub1 = build_subtensor_4x4()
        # Use ALS to find rank-7 decomposition of sub1
        U1, V1, W1 = None, None, None
        for trial in range(5):
            if time_remaining() < 30:
                break
            U1, V1, W1 = als_decompose(sub1, 7, max_iter=80)
            T1 = np.einsum('ar,br,cr->abc', U1, V1, W1)
            err1 = np.linalg.norm(T1 - sub1)
            if err1 < 1e-6:
                break
        
        # Decompose sub-tensor 2 (4x1x6 part) with rank 24 (schoolbook)
        sub2 = build_subtensor_1x6()
        # Schoolbook for this part: rank = 4*1*6 = 24
        U2 = np.zeros((N, 24))
        V2 = np.zeros((M, 24))
        W2 = np.zeros((P, 24))
        col = 0
        for i in range(4):
            for k in range(6):
                a = i * 5 + 4
                b = 4 * 6 + k
                c = k * 4 + i
                U2[a, col] = 1
                V2[b, col] = 1
                W2[c, col] = 1
                col += 1
        
        # Combine: U = [U1, U2], V = [V1, V2], W = [W1, W2]
        if U1 is not None:
            U_combined = np.hstack([U1, U2])
            V_combined = np.hstack([V1, V2])
            W_combined = np.hstack([W1, W2])
            
            result_combined = {"U": U_combined, "V": V_combined, "W": W_combined}
            if verify(result_combined):
                rank_combined = U_combined.shape[1]
                if rank_combined < best_rank:
                    best = result_combined
                    best_rank = rank_combined
                    print(f"Block partitioning found rank {rank_combined}")

    # Final verification with tighter tolerance
    if not verify(best, atol=1e-5):
        best = schoolbook()
        best_rank = n * m * p

    return {"U": to_lists(best["U"]), "V": to_lists(best["V"]), "W": to_lists(best["W"])}
# EVOLVE-BLOCK-END
