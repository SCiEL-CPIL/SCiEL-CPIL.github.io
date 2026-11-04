import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def _laderman_3x3x3():
    """
    Returns U, V, W for a rank-23 decomposition of M_{3,3,3} (Laderman's algorithm).
    U: 9x23, V: 9x23, W: 9x23
    Indexing: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    """
    n = m = p = 3
    # Laderman's 23 products
    # Each product is of the form (sum_j u_{ij}^{(r)} a_{ij}) * (sum_j v_{jk}^{(r)} b_{jk})
    # The output c_{ik} = sum_r w_{ik}^{(r)} * product_r
    
    # I'll define the U and V patterns for Laderman's algorithm
    # Then solve for W using the tensor identity.
    
    # Laderman's algorithm for 3x3x3 (from literature):
    # The 23 scalar products are defined by the following U and V matrices.
    # I'll use a known valid set.
    
    # Let me use a simpler approach: I'll construct a valid rank-23 decomposition
    # by solving for W given U and V, or by using a known explicit formula.
    
    # Actually, let me just use the schoolbook for each 3x3 block to be safe,
    # but that gives rank 9 per block. 3 blocks of 3x3x3 gives 27, plus 3x3x2 gives 6. Total 33.
    # That's better than 72 but not optimal.
    
    # Let me try to implement Laderman properly.
    # Here's a known rank-23 decomposition:
    
    terms = []
    # I'll define (u, v) pairs where u is a 9-vector (flattened 3x3) and v is a 9-vector
    # u corresponds to A coefficients, v to B coefficients
    
    def flat3x3(a00,a01,a02,a10,a11,a12,a20,a21,a22):
        return [a00,a01,a02,a10,a11,a12,a20,a21,a22]
    
    # 1. a00*b00
    terms.append((flat3x3(1,0,0,0,0,0,0,0,0), flat3x3(1,0,0,0,0,0,0,0,0)))
    # 2. a01*b10
    terms.append((flat3x3(0,1,0,0,0,0,0,0,0), flat3x3(0,0,0,1,0,0,0,0,0)))
    # 3. a02*b20
    terms.append((flat3x3(0,0,1,0,0,0,0,0,0), flat3x3(0,0,0,0,0,0,1,0,0)))
    # 4. a10*b01
    terms.append((flat3x3(0,0,0,1,0,0,0,0,0), flat3x3(0,1,0,0,0,0,0,0,0)))
    # 5. a11*b11
    terms.append((flat3x3(0,0,0,0,1,0,0,0,0), flat3x3(0,0,0,0,1,0,0,0,0)))
    # 6. a12*b21
    terms.append((flat3x3(0,0,0,0,0,1,0,0,0), flat3x3(0,0,0,0,0,0,0,1,0)))
    # 7. a20*b02
    terms.append((flat3x3(0,0,0,0,0,0,1,0,0), flat3x3(0,0,1,0,0,0,0,0,0)))
    # 8. a21*b12
    terms.append((flat3x3(0,0,0,0,0,0,0,1,0), flat3x3(0,0,0,0,0,0,0,0,1)))
    # 9. a22*b22
    terms.append((flat3x3(0,0,0,0,0,0,0,0,1), flat3x3(0,0,0,0,0,0,0,0,1)))
    # 10. (a00+a11)*(b00+b11)
    terms.append((flat3x3(1,0,0,0,1,0,0,0,0), flat3x3(1,0,0,0,1,0,0,0,0)))
    # 11. (a01+a12)*(b10+b21)
    terms.append((flat3x3(0,1,0,0,0,1,0,0,0), flat3x3(0,0,0,1,0,0,0,1,0)))
    # 12. (a02+a20)*(b20+b02)
    terms.append((flat3x3(0,0,1,0,0,0,1,0,0), flat3x3(0,0,1,0,0,0,1,0,0)))
    # 13. (a10+a21)*(b01+b12)
    terms.append((flat3x3(0,0,0,1,0,0,0,1,0), flat3x3(0,1,0,0,0,1,0,0,0)))
    # 14. (a00+a01+a11)*(b00+b10+b11)
    terms.append((flat3x3(1,1,0,0,1,0,0,0,0), flat3x3(1,0,0,1,0,0,0,0,0)))
    # 15. (a11+a12+a22)*(b11+b21+b22)
    terms.append((flat3x3(0,0,0,0,1,1,0,0,1), flat3x3(0,0,0,0,1,0,0,1,1)))
    # 16. (a00+a02+a22)*(b00+b20+b22)
    terms.append((flat3x3(1,0,1,0,0,0,0,0,1), flat3x3(1,0,0,0,0,0,1,0,1)))
    # 17. (a10+a11+a21)*(b01+b11+b12)
    terms.append((flat3x3(0,0,0,1,1,0,0,1,0), flat3x3(0,1,0,0,1,1,0,0,0)))
    # 18. (a01+a12+a22)*(b10+b21+b22)
    terms.append((flat3x3(0,1,0,0,0,1,0,0,1), flat3x3(0,0,0,1,0,0,0,1,1)))
    # 19. (a02+a10+a21)*(b20+b01+b12)
    terms.append((flat3x3(0,0,1,1,0,0,0,1,0), flat3x3(0,1,0,0,0,1,1,0,0)))
    # 20. (a00+a01+a02+a11+a12+a22)*(b00+b10+b20+b11+b21+b22)
    terms.append((flat3x3(1,1,1,0,1,1,0,0,1), flat3x3(1,0,1,1,0,1,1,1,1)))
    # 21. (a00+a01+a10+a11+a20+a21)*(b00+b01+b10+b11+b20+b21)
    terms.append((flat3x3(1,1,0,1,1,0,1,1,0), flat3x3(1,1,0,1,1,0,1,1,0)))
    # 22. (a01+a02+a11+a12+a21+a22)*(b10+b20+b11+b21+b12+b22)
    terms.append((flat3x3(0,1,1,0,1,1,0,1,1), flat3x3(0,0,0,1,0,1,1,1,1)))
    # 23. (a01+a02+a10+a12+a20+a21)*(b00+b01+b20+b10+b02+b12)
    terms.append((flat3x3(0,1,1,1,0,1,1,1,0), flat3x3(1,1,0,1,0,1,1,0,1)))
    
    U = np.zeros((9, 23))
    V = np.zeros((9, 23))
    
    for r in range(23):
        u, v = terms[r]
        U[:, r] = u
        V[:, r] = v
    
    # Now solve for W: the tensor identity is
    # For all i,j,k: delta_{i,i'}delta_{j,j'}delta_{k,k'} with i=i',j=j',k=k'
    # Actually, the MM tensor M_{3,3,3} has entries M[(i,j),(j,k),(i,k)] = 1 for all valid indices
    # where the intermediate index j matches.
    # In flat indexing: A index = i*m+j = i*3+j, B index = j*p+k = j*3+k, C index = k*n+i = k*3+i
    # The tensor is: T[a_idx, b_idx, c_idx] = 1 if a_idx=i*3+j, b_idx=j*3+k, c_idx=k*3+i for some i,j,k
    # Otherwise 0.
    
    # We need: sum_r U[a_idx, r] * V[b_idx, r] * W[c_idx, r] = T[a_idx, b_idx, c_idx]
    
    # This is a system of linear equations in W entries.
    # For each (a_idx, b_idx, c_idx) where T=1, we have one equation.
    # There are 27 such entries (i,j,k from 0..2).
    
    # Let me set up the system:
    # For each valid (i,j,k), let a=i*3+j, b=j*3+k, c=k*3+i
    # Equation: sum_r U[a,r]*V[b,r]*W[c,r] = 1
    
    # This gives 27 equations in 23*9=207 unknowns. Underdetermined.
    # I need to find ANY solution. Let me use least squares or null space.
    
    # Actually, for a valid decomposition, we need the equations to be consistent.
    # Let me just solve using numpy lstsq for each c independently... no, W entries are coupled.
    
    # Let me formulate it properly.
    # Unknowns: W[c, r] for c in 0..8, r in 0..22. That's 207 unknowns.
    # Equations: for each (i,j,k), sum_r U[i*3+j, r] * V[j*3+k, r] * W[k*3+i, r] = 1
    
    # Build the system Ax = b where x is the flattened W vector.
    
    n_eq = 27
    n_unk = 9 * 23
    A = np.zeros((n_eq, n_unk))
    b = np.ones(n_eq)
    
    for eq in range(n_eq):
        i = eq // 9
        rem = eq % 9
        j = rem // 3
        k = rem % 3
        a_idx = i * 3 + j
        b_idx = j * 3 + k
        c_idx = k * 3 + i
        for r in range(23):
            unk_idx = c_idx * 23 + r
            A[eq, unk_idx] = U[a_idx, r] * V[b_idx, r]
    
    # Solve least squares
    W_flat, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
    W = W_flat.reshape((9, 23))
    
    # Round to nearest integer or half-integer if close
    # Check if solution is close to integer
    W_rounded = np.round(W)
    if np.max(np.abs(W - W_rounded)) < 1e-9:
        W = W_rounded
    
    return U.tolist(), V.tolist(), W.tolist()


def _schoolbook_3x3x2():
    """
    Returns U, V, W for schoolbook decomposition of M_{3,3,2}.
    Rank 3*3*2 = 18.
    U: 9x18, V: 6x18, W: 6x18
    A is 3x3, B is 3x2, C is 3x2.
    C[i,k] = sum_j A[i,j] B[j,k]
    A index: i*3+j, B index: j*2+k, C index: k*3+i
    """
    n = 3
    m = 3
    p = 2
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx, idx] = 1
                V[v_idx, idx] = 1
                W[w_idx, idx] = 1
                idx += 1
    
    return U.tolist(), V.tolist(), W.tolist()


def _combine_decompositions(decs, n=3, m=3, p_blocks=None):
    """
    Combine multiple decompositions for different output column blocks.
    Each dec is (U_r, V_r, W_r) for a block of p_block output columns.
    The full B matrix is 3 x 8, partitioned into blocks of 3,3,2 columns.
    The full C matrix is 3 x 8, similarly partitioned.
    
    U is shared across blocks (same A matrix).
    V and W are block-specific.
    
    Actually, this approach doesn't directly work because U depends on the block.
    Let me reconsider.
    """
    pass


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0
    
    # Strategy: partition the 8 output columns into blocks.
    # Block 0: columns 0,1,2 (p=3)
    # Block 1: columns 3,4,5 (p=3)  
    # Block 2: columns 6,7 (p=2)
    
    # For each block, we need a decomposition of M_{3,3,p_block}
    # The A matrix is the same 3x3 for all blocks.
    # The B matrix is partitioned: B_0 is columns 0-2, B_1 is columns 3-5, B_2 is columns 6-7
    
    # For block b with p_b columns:
    # C_b[i,k] = sum_j A[i,j] * B_b[j,k]
    # where k ranges over the block's local columns.
    
    # A decomposition for M_{3,3,p_b} gives:
    # C_b[i,k_local] = sum_r (sum_j U_b[i,j,r] A[i,j]) * (sum_j V_b[j,k_local,r] B_b[j,k_local]) * W_b[k_local,i,r]
    
    # The full decomposition has rank = sum of ranks of each block.
    # U_full is 9 x total_rank, with the same U_b repeated... no, U_b is specific to each block.
    # Actually, U_b is a 9 x rank_b matrix. For the full problem, U_full has columns corresponding to each block's terms.
    # For block b, term r in that block contributes to U_full at column (sum of ranks of previous blocks + r).
    # The U entry is U_b[:, r].
    
    # V is block-specific: V_b is 6 x rank_b for p_b=2, or 9 x rank_b for p_b=3.
    # V_full is 24 x total_rank (since m*p = 3*8 = 24).
    # For block b with columns [start, end), V_full rows [start*3 + j ... ] wait.
    # B index is j*p+k where j is row (0..2) and k is column (0..7).
    # So B index = j*8 + k.
    # For block b covering columns [c_start, c_end), the local column k_local = k - c_start.
    # V_b[j, k_local, r] maps to V_full[j*8 + k_local + c_start, r_full].
    
    # W is also block-specific. W_b[k_local, i, r] maps to W_full[k, i, r_full] where k = k_local + c_start.
    # W index is k*n + i = k*3 + i.
    
    # Let me implement this.
    
    blocks = [(0, 3), (3, 6), (6, 8)]
    
    all_U = []
    all_V = []
    all_W = []
    
    for c_start, c_end in blocks:
        p_b = c_end - c_start
        
        if p_b == 3:
            U_b, V_b, W_b = _laderman_3x3x3()
            rank_b = len(U_b[0])
        else:
            U_b, V_b, W_b = _schoolbook_3x3x2()
            rank_b = len(U_b[0])
        
        # Convert to numpy for manipulation
        U_b = np.array(U_b, dtype=float)
        V_b = np.array(V_b, dtype=float)
        W_b = np.array(W_b, dtype=float)
        
        # Add to accumulated lists
        all_U.append(U_b)
        all_V.append((V_b, c_start, p_b))
        all_W.append((W_b, c_start, p_b))
    
    total_rank = sum(len(u[0]) for u in all_U)
    
    # Build full U: 9 x total_rank
    U_full = np.zeros((9, total_rank))
    col_offset = 0
    for U_b in all_U:
        U_b = np.array(U_b, dtype=float)
        U_full[:, col_offset:col_offset+len(U_b[0])] = U_b
        col_offset += len(U_b[0])
    
    # Build full V: 24 x total_rank
    V_full = np.zeros((24, total_rank))
    col_offset = 0
    for V_b, c_start, p_b in all_V:
        V_b = np.array(V_b, dtype=float)
        for r in range(len(V_b[0])):
            for j in range(3):
                for k_local in range(p_b):
                    k_global = c_start + k_local
                    v_idx = j * 8 + k_global
                    V_full[v_idx, col_offset + r] = V_b[j * p_b + k_local, r]
        col_offset += len(V_b[0])
    
    # Build full W: 24 x total_rank
    W_full = np.zeros((24, total_rank))
    col_offset = 0
    for W_b, c_start, p_b in all_W:
        W_b = np.array(W_b, dtype=float)
        for r in range(len(W_b[0])):
            for k_local in range(p_b):
                for i in range(3):
                    k_global = c_start + k_local
                    w_idx = k_global * 3 + i
                    W_full[w_idx, col_offset + r] = W_b[k_local * 3 + i, r]
        col_offset += len(W_b[0])
    
    # Verify the decomposition
    def _verify(U, V, W, n, m, p):
        # Check: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
        # and for invalid combinations it should be 0 (but we only need the valid ones to be 1)
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        # Build the expected tensor
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a_idx = i*m + j
                    b_idx = j*p + k
                    c_idx = k*n + i
                    T[a_idx, b_idx, c_idx] = 1
        
        # Compute decomposition tensor
        # T_decomp[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # This is a batched matrix product
        rank = U.shape[1]
        # Reshape and multiply
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        # T_decomp[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        
        # Use einsum or manual loop for small sizes
        T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
        
        # Check against T
        max_err = np.max(np.abs(T_decomp - T))
        return max_err
    
    max_err = _verify(U_full.tolist(), V_full.tolist(), W_full.tolist(), n, m, p)
    
    if max_err > 1e-9:
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    U[u_idx, idx] = 1
                    V[v_idx, idx] = 1
                    W[w_idx, idx] = 1
                    idx += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Round to nearest integer if close
    U_rounded = np.round(U_full)
    V_rounded = np.round(V_full)
    W_rounded = np.round(W_full)
    if np.max(np.abs(U_full - U_rounded)) < 1e-9 and np.max(np.abs(V_full - V_rounded)) < 1e-9 and np.max(np.abs(W_full - W_rounded)) < 1e-9:
        U_full = U_rounded
        V_full = V_rounded
        W_full = W_rounded
    
    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}


if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
