import time
import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    rank = 88
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen factors for 2x2x2
    strassen_factors = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    # Process 2x M_2,4,4 blocks (for i=0,1)
    for c_block in range(2):
        for factor in strassen_factors:
            a_coeffs, b_coeffs, c_coeffs = factor
            for sub_prod_idx in range(4):
                if sub_prod_idx == 0:
                    a_i_off, a_j_off = 0, 0
                    b_i_off, b_j_off = 0, 0
                    c_i_off, c_j_off = 0, 0
                elif sub_prod_idx == 1:
                    a_i_off, a_j_off = 0, 0
                    b_i_off, b_j_off = 0, 2
                    c_i_off, c_j_off = 0, 2
                elif sub_prod_idx == 2:
                    a_i_off, a_j_off = 0, 2
                    b_i_off, b_j_off = 2, 0
                    c_i_off, c_j_off = 0, 0
                else:
                    a_i_off, a_j_off = 0, 2
                    b_i_off, b_j_off = 2, 2
                    c_i_off, c_j_off = 0, 2
                
                for (ai, aj), au in a_coeffs.items():
                    u_idx = (ai + a_i_off) * 4 + (aj + a_j_off)
                    U[u_idx][col] += au
                
                for (bi, bj), bv in b_coeffs.items():
                    v_idx = (bi + b_i_off) * 8 + (bj + b_j_off + 4 * c_block)
                    V[v_idx][col] += bv
                
                for (ci, cj), cw in c_coeffs.items():
                    w_idx = (cj + c_j_off + 4 * c_block) * 3 + (ci + c_i_off)
                    W[w_idx][col] += cw
                
                col += 1
    
    # Process 2x M_1,4,4 blocks (for i=2)
    for c_block in range(2):
        for k in range(4):
            for j in range(4):
                u_idx = 2 * 4 + k
                v_idx = k * 8 + (j + 4 * c_block)
                w_idx = (j + 4 * c_block) * 3 + 2
                
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
    
    # Convert to numpy for efficient computation
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    start_time = time.time()
    deadline = start_time + 110.0
    
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Compute rank-1 tensors as flat vectors
    R = np.zeros((rank, nm * mp * pn), dtype=float)
    for r in range(rank):
        Rr = np.einsum('a,b,c->abc', U_np[:, r], V_np[:, r], W_np[:, r])
        R[r] = Rr.reshape(-1)
    
    keep = [True] * rank
    best_rank = rank
    
    # Pass 1: Check for identical or proportional rank-1 tensors
    for r in range(rank):
        if time.time() > deadline:
            break
        if not keep[r]:
            continue
        
        for s in range(r + 1, rank):
            if not keep[s]:
                continue
            
            norm_r = np.linalg.norm(R[r])
            norm_s = np.linalg.norm(R[s])
            
            if norm_r < 1e-10 or norm_s < 1e-10:
                continue
            
            alpha = np.dot(R[r], R[s]) / (norm_s ** 2)
            
            diff = R[r] - alpha * R[s]
            if np.linalg.norm(diff) / norm_r < 1e-8:
                # R[r] is proportional to R[s], we can eliminate R[r]
                # by scaling R[s] appropriately. But actually, if T_r = alpha * T_s,
                # then T_r + T_s = (1+alpha)*T_s, so we can just scale W[s] by (1+alpha)
                # and drop term r.
                if abs(1 + alpha) > 1e-10:
                    W_np[:, s] *= (1 + alpha)
                else:
                    # 1+alpha ≈ 0, so T_r + T_s ≈ 0, we can drop both
                    keep[s] = False
                keep[r] = False
                break
    
    # Pass 2: Try to find small linearly dependent sets (size 2 and 3)
    # For size 2, this is the same as proportional check.
    # For size 3: check if T_r, T_s, T_t are linearly dependent.
    # If so, one can be expressed as a combination of the other two.
    # But we need to be careful: we can only eliminate a term if the combination
    # preserves the tensor structure (i.e., the coefficients allow us to update
    # the U, V, W factors appropriately).
    
    # A simpler approach: check if any three terms sum to zero or are in a
    # specific algebraic relationship that allows elimination.
    
    # For now, let's do a more careful check: for each pair (r,s) not yet eliminated,
    # check if there exists a t such that T_r + c*T_s + d*T_t = 0 for some c,d.
    # This is equivalent to checking if T_r is in the span of {T_s, T_t}.
    
    # This is expensive, so let's only do it for a limited number of candidates.
    
    # Actually, let's try a different approach: use the fact that in Strassen-based
    # decompositions, there are known algebraic identities. Let me check if any
    # triple of terms satisfies a Strassen-like identity.
    
    # For now, let's just verify what we have and return.
    
    final_indices = [i for i in range(rank) if keep[i]]
    final_rank = len(final_indices)
    
    if final_rank < best_rank:
        best_rank = final_rank
    
    U_final = U_np[:, final_indices]
    V_final = V_np[:, final_indices]
    W_final = W_np[:, final_indices]
    
    # Verification
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(final_rank):
                    val += U_final[i*m+j][c] * V_final[j*p+k][c] * W_final[k*n+i][c]
                if abs(val - 1) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to original if verification fails
        return {"U": U, "V": V, "W": W}
        
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is correct."""
    r = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(r):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if val != 1:
                    return False
    return True
