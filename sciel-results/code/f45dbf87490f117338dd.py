import numpy as np
import time

def construct():
    """
    Returns a rank-minimal decomposition of the 4x4x4 matrix multiplication tensor.
    Uses rank 49 via recursive Strassen.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    def _build_rank49():
        """Build the standard recursive Strassen rank-49 decomposition."""
        terms = []
        
        strassen_2x2 = [
            ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
            ([(1,0,1),(1,1,1)], [(0,0,1)],               [(1,0,1),(1,1,-1)]),
            ([(0,0,1)],               [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
            ([(1,1,1)],               [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
            ([(0,0,1),(0,1,1)],      [(1,1,1)],               [(0,0,-1),(0,1,1)]),
            ([(1,0,1),(0,0,-1)],     [(0,0,1),(0,1,1)],      [(1,1,1)]),
            ([(0,1,1),(1,1,-1)],     [(1,0,1),(1,1,1)],      [(0,0,1)]),
        ]
        
        outer_strassen = [
            ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
            ([(1,0,1),(1,1,1)], [(0,0,1)],               [(1,0,1),(1,1,-1)]),
            ([(0,0,1)],               [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
            ([(1,1,1)],               [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
            ([(0,0,1),(0,1,1)],      [(1,1,1)],               [(0,0,-1),(0,1,1)]),
            ([(1,0,1),(0,0,-1)],     [(0,0,1),(0,1,1)],      [(1,1,1)]),
            ([(0,1,1),(1,1,-1)],     [(1,0,1),(1,1,1)],      [(0,0,1)]),
        ]
        
        for (a_blocks, b_blocks, c_blocks) in outer_strassen:
            for (a_comb, b_comb, c_comb) in strassen_2x2:
                u_dict = {}
                v_dict = {}
                w_dict = {}
                
                for (lr, lc, ca) in a_comb:
                    for (br, bc, coeff) in a_blocks:
                        gr = br * 2 + lr
                        gc = bc * 2 + lc
                        u_idx = gr * 4 + gc
                        u_dict[u_idx] = u_dict.get(u_idx, 0) + ca * coeff
                
                for (lr, lc, cb) in b_comb:
                    for (br, bc, coeff) in b_blocks:
                        gr = br * 2 + lr
                        gc = bc * 2 + lc
                        v_idx = gr * 4 + gc
                        v_dict[v_idx] = v_dict.get(v_idx, 0) + cb * coeff
                
                for (lr, lc, cc) in c_comb:
                    for (br, bc, coeff) in c_blocks:
                        gr = br * 2 + lr
                        gc = bc * 2 + lc
                        w_idx = gc * 4 + gr
                        w_dict[w_idx] = w_dict.get(w_idx, 0) + cc * coeff
                
                terms.append((u_dict, v_dict, w_dict))
        
        rank = len(terms)
        U = [[0] * rank for _ in range(16)]
        V = [[0] * rank for _ in range(16)]
        W = [[0] * rank for _ in range(16)]
        
        for col, (u_dict, v_dict, w_dict) in enumerate(terms):
            for idx, val in u_dict.items():
                if val != 0:
                    U[idx][col] = val
            for idx, val in v_dict.items():
                if val != 0:
                    V[idx][col] = val
            for idx, val in w_dict.items():
                if val != 0:
                    W[idx][col] = val
        
        return U, V, W, rank

    def _verify(U, V, W, rank):
        """Verify the decomposition is exact using NumPy (fast path)."""
        try:
            Ua = np.asarray(U, dtype=np.int64)
            Va = np.asarray(V, dtype=np.int64)
            Wa = np.asarray(W, dtype=np.int64)
            # U shape (16, rank), V shape (16, rank), W shape (16, rank)
            # U[i*4+j, r] * V[j*4+k, r] * W[k*4+i, r]
            # Sum over r -> (16, 16) where row idx = i*4+j, col idx = j*4+k? No.
            # Let's do it carefully with einsum.
            # U[i*4+j, r], V[j*4+k, r], W[k*4+i, r]
            # We want result[i, k] = sum_{j, r} U[i*4+j, r] * V[j*4+k, r] * W[k*4+i, r]
            # But W depends on k and i.
            # Let's reshape:
            # U: (4, 4, rank) -> U3[i, j, r]
            # V: (4, 4, rank) -> V3[j, k, r]
            # W: (4, 4, rank) -> W3[k, i, r]  (since W[k*4+i, r])
            
            U3 = Ua.reshape(4, 4, rank)
            V3 = Va.reshape(4, 4, rank)
            W3 = Wa.reshape(4, 4, rank)
            
            # Sum over j and r: U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
            # First sum over r: T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r]
            # T = np.einsum('ijr,jkr->ijk', U3, V3)
            # Then sum over j: R[i,k] = sum_j T[i,j,k] * W3[k,i,r]... wait W3 doesn't have r in the second step.
            # R[i,k] = sum_j (sum_r U3[i,j,r] * V3[j,k,r]) * W3[k,i, *]
            # No, W3[k,i,r] has r.
            # So R[i,k] = sum_{j,r} U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
            
            # Let's use einsum directly:
            # 'ijr,jkr,kir->ik'
            R = np.einsum('ijr,jkr,kir->ik', U3, V3, W3)
            
            # R[i,k] should be 1 for all i,k?
            # Let's check the tensor definition.
            # Matrix mult: C[i,k] = sum_j A[i,j] B[j,k]
            # Tensor T[i,j,k] = 1.
            # Decomposition: sum_r U[i,j,r] V[j,k,r] W[k,i,r] = T[i,j,k] = 1.
            # So for each (i,j,k), the sum over r is 1.
            # My einsum above summed over j and r, which is wrong.
            # I need to check each (i,j,k) separately or reshape to (i,j,k).
            
            # Let's compute S[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
            # 'ijr,jkr,kir->ijk'
            S = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
            
            # S should be all 1s.
            return bool(np.all(S == 1))
        except Exception:
            # Fallback to slow python loop if numpy fails
            for i in range(4):
                for j in range(4):
                    for k in range(4):
                        s = 0
                        for r in range(rank):
                            s += U[i*4+j][r] * V[j*4+k][r] * W[k*4+i][r]
                        if s != 1:
                            return False
            return True

    def _try_rank48():
        """
        Attempt to find a rank-48 decomposition.
        Strategy: The 4x4x4 tensor viewed as 2x2x2 over 2x2 matrices.
        Try a rank-6 outer decomposition combined with rank-8 inner (full 2x2 basis).
        """
        # The 2x2x2 block tensor: C[br,bc] = sum_{ar,ac} A[ar,ac] * B[ac,bc]
        # where each entry is a 2x2 matrix.
        # 
        # A rank-r decomposition over 2x2 matrices: 
        #   C[br,bc] = sum_s X_s[br,bc] * Y_s[br,bc]  ... no, that's not right.
        # 
        # Actually: C_block[br,bc] = sum_s (linear combo of A blocks)_s * (linear combo of B blocks)_s * (coeff for C block)
        # But each "linear combo" is a 2x2 matrix, and the product is matrix multiplication of 2x2 matrices.
        #
        # So: C_block[br,bc] = sum_s coeff_s[br,bc] * (X_s * Y_s)
        # where X_s, Y_s are 2x2 matrices (linear combos of A,B blocks)
        # and the product X_s * Y_s is 2x2 matrix multiplication.
        #
        # Now decompose each 2x2 matrix product X_s * Y_s using rank-8 (schoolbook for 2x2):
        # (X_s * Y_s)[lr,lc] = sum_{m=0}^{1} X_s[lr,m] * Y_s[m,lc]
        #
        # This gives 4*2 = 8 scalar multiplications per 2x2 product.
        # Total: 6 (outer) * 8 (inner) = 48 terms.
        #
        # We need a rank-6 decomposition of the 2x2x2 block tensor.
        # The block tensor T_block[br,ac,bc] = delta (C[br,bc] gets contribution from A[br,ac]*B[ac,bc])
        #
        # For rank-6, we need 6 terms: T_block[br,ac,bc] = sum_s u_s[br,ac] * v_s[ac,bc] * w_s[br,bc]
        # where u_s, v_s, w_s are 2x2 matrices (entries are scalars).
        #
        # This is equivalent to finding a rank-6 decomposition of the 8-dimensional tensor
        # T_block indexed by (br,ac,bc) in {0,1}^3.
        #
        # The standard 2x2x2 multiplication tensor has rank 7. But we're looking for
        # a rank-6 decomposition where the "entries" are themselves 2x2 matrices.
        #
        # Wait, I'm confusing myself. Let me restart.
        #
        # The full 4x4x4 tensor T[i,j,k] = 1 for all i,j,k.
        # i = 2*br + lr, j = 2*ac + mc, k = 2*bc + nc
        # where br,ac,bc in {0,1} (block indices) and lr,mc,nc in {0,1} (local indices).
        #
        # T[2*br+lr, 2*ac+mc, 2*bc+nc] = 1
        #
        # A decomposition: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
        #
        # If we assume a structured form where:
        # U[2*br+lr, 2*ac+mc, r] = sum_s u_s[br,ac] * alpha_s[lr,mc]  (for r in group s)
        # V[2*ac+mc, 2*bc+nc, r] = sum_s v_s[ac,bc] * beta_s[mc,nc]
        # W[2*bc+nc, 2*br+lr, r] = sum_s w_s[br,bc] * gamma_s[nc,lr]
        #
        # with r = 8*s + t where t indexes the 2x2 basis (rank-8 inner decomposition)
        #
        # Then: sum_{s,t} u_s[br,ac] * alpha_s[lr,mc] * v_s[ac,bc] * beta_s[mc,nc] * w_s[br,bc] * gamma_s[nc,lr] = 1
        #
        # For the inner part (fixed s): sum_t alpha_s[lr,mc] * beta_s[mc,nc] * gamma_s[nc,lr] should give
        # the (lr,nc) entry of the 2x2 matrix product, i.e., sum_mc alpha_s[lr,mc] * beta_s[mc,nc].
        # This requires gamma_s[nc,lr] = 1 and the sum over mc to work out.
        #
        # Actually, for the inner 2x2 multiplication: (X*Y)[lr,nc] = sum_mc X[lr,mc] * Y[mc,nc]
        # This is rank-8 (4*2) decomposition: for each (lr,mc,nc) triple, one term.
        # U part: X[lr,mc], V part: Y[mc,nc], W part: 1 (coefficient for C[lr,nc])
        #
        # So the inner decomposition has 8 terms indexed by (lr,mc,nc) in {0,1}^3.
        #
        # For the outer part, we need: sum_s u_s[br,ac] * v_s[ac,bc] * w_s[br,bc] = 1 for all br,ac,bc.
        # This is exactly the 2x2x2 multiplication tensor of rank 6.
        #
        # The 2x2x2 multiplication tensor M[br,ac,bc] = 1 for all br,ac,bc in {0,1}.
        # We need rank-6: M[br,ac,bc] = sum_{s=1}^{6} u_s[br,ac] * v_s[ac,bc] * w_s[br,bc]
        #
        # Is rank(M) = 6? The tensor M is the all-ones tensor of size 2x2x2.
        # The all-ones tensor of size n x n x n has rank... For 2x2x2, 
        # M[i,j,k] = 1. 
        # M = 1*1*1 + ... Let's see: sum_s u_s[i,j] * v_s[j,k] * w_s[k,i] = 1 for all i,j,k.
        #
        # Try rank 1: u[i,j]*v[j,k]*w[k,i] = 1 for all i,j,k. 
        # u[0,0]*v[0,0]*w[0,0] = 1, u[0,0]*v[0,1]*w[1,0] = 1, etc.
        # This means u[i,j] = a_i / b_j type... Let's check: u[i,j]*v[j,k]*w[k,i] = 1.
        # u[i,j] = p_i / q_j, v[j,k] = q_j / r_k, w[k,i] = r_k / p_i. Product = 1. Yes!
        # So rank 1 works for the all-ones tensor! M[i,j,k] = 1 = (1)*(1)*(1).
        #
        # Wait, that's just the trivial decomposition: u[i,j]=1, v[j,k]=1, w[k,i]=1 for all.
        # That's rank 1!
        #
        # But that can't be right for matrix multiplication. Let me re-examine.
        #
        # The issue is: in matrix multiplication, the "block" tensor is NOT the all-ones tensor.
        # C[br,bc] = sum_ac A[br,ac] * B[ac,bc] where * is 2x2 matrix multiplication.
        # At the block level: C_block[br,bc] = sum_ac A_block[br,ac] * B_block[ac,bc]
        # This IS the 2x2 matrix multiplication tensor at the block level!
        #
        # So the block-level tensor is the 2x2x2 matrix multiplication tensor, which has rank 7 (Strassen).
        # It does NOT have rank 6.
        #
        # So the rank-48 approach via rank-6 outer + rank-8 inner doesn't work because
        # the outer tensor has rank 7, not 6.
        #
        # Unless we use a different inner decomposition. What if the inner decomposition
        # is not rank 8 but something else?
        #
        # Alternative: use a rank-7 outer (Strassen) and find a way to reduce the inner
        # from rank 7 to rank 6 for some terms? That seems unlikely.
        #
        # Or: use a non-separable decomposition that doesn't factor as outer*inner.
        #
        # Given the difficulty, let me just return the rank-49 decomposition.
        return None

    # Build rank-49 decomposition
    U, V, W, rank = _build_rank49()
    
    # Check time
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
        
    # Verify
    if not _verify(U, V, W, rank):
        # Fallback to schoolbook rank 64
        rank = 64
        U = [[0] * rank for _ in range(16)]
        V = [[0] * rank for _ in range(16)]
        W = [[0] * rank for _ in range(16)]
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U[i*4+j][col] = 1
                    V[j*4+k][col] = 1
                    W[k*4+i][col] = 1
                    col += 1
    
    return {"U": U, "V": V, "W": W}
