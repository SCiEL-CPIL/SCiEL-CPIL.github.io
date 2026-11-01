"""Matrix multiplication 2x2x15 - rank minimal decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS
    t_start = time.time()
    deadline = t_start + 140  # 20s safety margin before 160s limit
    
    # We'll use a block-based approach.
    # Split p=15 into blocks. We can use:
    # - 7 blocks of size 2 (Strassen, rank 7 each) + 1 block of size 1 (schoolbook, rank 4)
    #   Total: 7*7 + 4 = 53
    # - 3 blocks of size 5 (need to find optimal for 2x2x5)
    # - 2 blocks of size 4 + 1 block of size 7
    # - 5 blocks of size 3
    # - etc.
    
    # First, let's establish a baseline with the known construction
    def _strassen_2x2x2():
        """Return (U, V, W) for 2x2x2 Strassen, rank 7."""
        U = np.zeros((4, 7), dtype=float)
        V = np.zeros((4, 7), dtype=float)
        W = np.zeros((4, 7), dtype=float)
        
        # Strassen's 7 products for 2x2x2
        # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
        # C = A*B
        # 
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-f)
        # M5 = (a+b)h
        # M6 = (c-d)(e+f)
        # M7 = (b-d)(g+h)
        #
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        #
        # U maps (i,j) -> coefficient for A_ij in each product
        # V maps (j,k) -> coefficient for B_jk in each product
        # W maps (k,i) -> coefficient for C_ki in each product
        #
        # Product 1: M1 = (a+d)(e+h)
        U[0, 0] = 1  # A[0,0]=a
        U[3, 0] = 1  # A[1,1]=d
        V[0, 0] = 1  # B[0,0]=e
        V[3, 0] = 1  # B[1,1]=h
        W[0, 0] = 1  # C[0,0]
        W[3, 0] = 1  # C[1,1]
        
        # Product 2: M2 = (c+d)e
        U[2, 1] = 1  # A[1,0]=c
        U[3, 1] = 1  # A[1,1]=d
        V[0, 1] = 1  # B[0,0]=e
        W[1, 1] = 1  # C[1,0]
        W[3, 1] = -1 # C[1,1]
        
        # Product 3: M3 = a(f-h)
        U[0, 2] = 1  # A[0,0]=a
        V[1, 2] = 1  # B[0,1]=f
        V[3, 2] = -1 # B[1,1]=h
        W[0, 2] = 1  # C[0,1]
        W[3, 2] = 1  # C[1,1]
        
        # Product 4: M4 = d(g-f)
        U[3, 3] = 1  # A[1,1]=d
        V[2, 3] = 1  # B[1,0]=g
        V[0, 3] = -1 # B[0,0]=f... wait
        # B indices: (j,k) where j is row, k is col
        # B[0,0]=e, B[0,1]=f, B[1,0]=g, B[1,1]=h
        # V[j,k] = V[j*2+k]
        # M4 = d*(g-f): d is A[1,1], g is B[1,0], f is B[0,1]
        V[2, 3] = 1  # B[1,0]=g
        V[1, 3] = -1 # B[0,1]=f
        W[0, 3] = 1  # C[0,0]
        W[1, 3] = 1  # C[1,0]
        
        # Product 5: M5 = (a+b)h
        U[0, 4] = 1  # A[0,0]=a
        U[1, 4] = 1  # A[0,1]=b
        V[3, 4] = 1  # B[1,1]=h
        W[0, 4] = -1 # C[0,0]
        W[0, 5] = 0  # placeholder
        # C01 += M5
        W[1, 4] = 1  # C[0,1]
        
        # Product 6: M6 = (c-d)(e+f)
        U[2, 5] = 1  # A[1,0]=c
        U[3, 5] = -1 # A[1,1]=d
        V[0, 5] = 1  # B[0,0]=e
        V[1, 5] = 1  # B[0,1]=f
        W[3, 5] = 1  # C[1,1]
        
        # Product 7: M7 = (b-d)(g+h)
        U[1, 6] = 1  # A[0,1]=b
        U[3, 6] = -1 # A[1,1]=d
        V[2, 6] = 1  # B[1,0]=g
        V[3, 6] = 1  # B[1,1]=h
        W[0, 6] = 1  # C[0,0]
        
        return U, V, W
    
    def _schoolbook_2x2x1():
        """Return (U, V, W) for 2x2x1 schoolbook, rank 4."""
        U = np.zeros((4, 4), dtype=float)
        V = np.zeros((2, 4), dtype=float)
        W = np.zeros((2, 4), dtype=float)
        
        # C00 = a*e + b*g
        # C10 = c*e + d*g
        # (only column 0 of B)
        # U[i,j] where i,j in {0,1} -> index i*2+j
        # V[j,k] where j in {0,1}, k=0 -> index j
        # W[k,i] where k=0, i in {0,1} -> index i
        
        # r=0: a*e -> C00
        U[0, 0] = 1
        V[0, 0] = 1
        W[0, 0] = 1
        
        # r=1: b*g -> C00
        U[1, 1] = 1
        V[1, 1] = 1
        W[0, 1] = 1
        
        # r=2: c*e -> C10
        U[2, 2] = 1
        V[0, 2] = 1
        W[1, 2] = 1
        
        # r=3: d*g -> C10
        U[3, 3] = 1
        V[1, 3] = 1
        W[1, 3] = 1
        
        return U, V, W
    
    def _assemble_blocks(block_specs):
        """
        Assemble U, V, W from block specifications.
        block_specs: list of (p_size, U_block, V_block, W_block, rank_block)
        """
        total_rank = sum(spec[4] for spec in block_specs)
        U_full = np.zeros((n * m, total_rank), dtype=float)
        V_full = np.zeros((m * p, total_rank), dtype=float)
        W_full = np.zeros((p * n, total_rank), dtype=float)
        
        r_offset = 0
        k_offset = 0
        for p_size, U_b, V_b, W_b, rank_b in block_specs:
            # Copy U
            U_full[:, r_offset:r_offset+rank_b] = U_b
            
            # Copy V with row offset
            for j in range(m):
                for k in range(p_size):
                    V_full[j*p + k_offset + k, r_offset:r_offset+rank_b] = V_b[j, k]
            
            # Copy W with row offset
            for k in range(p_size):
                for i in range(n):
                    W_full[(k_offset + k)*n + i, r_offset:r_offset+rank_b] = W_b[k, i]
            
            r_offset += rank_b
            k_offset += p_size
        
        return U_full.tolist(), V_full.tolist(), W_full.tolist()
    
    # Build baseline: 7 Strassen blocks of size 2 + 1 schoolbook block of size 1
    U_s, V_s, W_s = _strassen_2x2x2()
    U_1, V_1, W_1 = _schoolbook_2x2x1()
    
    blocks = []
    for _ in range(7):
        blocks.append((2, U_s, V_s, W_s, 7))
    blocks.append((1, U_1, V_1, W_1, 4))
    
    best_U, best_V, best_W = _assemble_blocks(blocks)
    best_rank = 53
    
    # Now try to improve by using different block sizes
    # Key insight: if we can find a 2x2x3 decomposition with rank < 7+4=11 (per 3 cols),
    # or a 2x2x4 with rank < 14, etc.
    
    # For 2x2x3: schoolbook is 8, can we do better?
    # Let's try rank 6 for 2x2x3 via search
    def _search_2x2x3(max_rank, max_iter=5000):
        """Try to find rank-r decomposition for 2x2x3."""
        n_, m_, p_ = 2, 2, 3
        # Tensor T[i,j,k] = delta_{i,k} (since C_i = sum_j A_ij B_jk, 
        # for this to work as identity-like: T[i,j,k] = 1 if i==k else 0... 
        # Actually the multiplication tensor has T[i,j,k] = delta_{i,k}... no.
        # The matrix multiplication tensor M_{i,j,k} for (i,j) in A, (j,k) in B, (i,k) in C
        # is: M_{(i,j),(j,k),(i,k)} = 1
        # In flattened form: T[a, b, c] where a=i*m+j, b=j*p+k, c=k*n+i
        # T[a,b,c] = 1 iff b's j-part == a's j-part AND c's k-part == b's k-part AND c's i-part == a's i-part
        # i.e., a=i*m+j, b=j*p+k, c=k*n+i
        
        # Build target tensor
        T = np.zeros((n_*m_, m_*p_, p_*n_), dtype=float)
        for i in range(n_):
            for j in range(m_):
                for k in range(p_):
                    a = i*m_ + j
                    b = j*p_ + k
                    c = k*n_ + i
                    T[a, b, c] = 1
        
        # Search for rank decomposition
        best_found = None
        for rank in range(4, max_rank+1):
            # Try random initialization
            for attempt in range(max_iter // max_rank):
                if best_found is not None:
                    break
                U_r = np.random.randn(n_*m_, rank)
                V_r = np.random.randn(m_*p_, rank)
                W_r = np.random.randn(p_*n_, rank)
                
                # Verify: T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
                # Check sparsity: T is sparse, so most entries should be 0
                # Compute reconstruction error on non-zero entries
                errors = []
                for i in range(n_):
                    for j in range(m_):
                        for k in range(p_):
                            a = i*m_ + j
                            b = j*p_ + k
                            c = k*n_ + i
                            recon = 0.0
                            for r in range(rank):
                                recon += U_r[a, r] * V_r[b, r] * W_r[c, r]
                            errors.append(abs(recon - 1.0))
                
                # Also check zero entries are approximately 0
                max_err = max(errors) if errors else 0
                if max_err < 0.01:
                    # Check zero entries
                    zero_max = 0
                    for a in range(n_*m_):
                        for b in range(m_*p_):
                            for c in range(p_*n_):
                                if T[a,b,c] == 0:
                                    recon = 0.0
                                    for r in range(rank):
                                        recon += U_r[a, r] * V_r[b, r] * W_r[c, r]
                                    zero_max = max(zero_max, abs(recon))
                    if zero_max < 0.01:
                        best_found = (U_r.copy(), V_r.copy(), W_r.copy(), rank)
                        break
        return best_found
    
    # Try to find better 2x2x3
    result_3 = _search_2x2x3(max_rank=6, max_iter=2000)
    
    if time.time() > deadline - 10:
        # Time's up, return best
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # Try 2x2x4
    def _search_2x2x4(max_rank, max_iter=3000):
        n_, m_, p_ = 2, 2, 4
        T = np.zeros((n_*m_, m_*p_, p_*n_), dtype=float)
        for i in range(n_):
            for j in range(m_):
                for k in range(p_):
                    a = i*m_ + j
                    b = j*p_ + k
                    c = k*n_ + i
                    T[a, b, c] = 1
        
        best_found = None
        for rank in range(6, max_rank+1):
            for attempt in range(max_iter // max_rank):
                if best_found is not None:
                    break
                U_r = np.random.randn(n_*m_, rank)
                V_r = np.random.randn(m_*p_, rank)
                W_r = np.random.randn(p_*n_, rank)
                
                errors = []
                for i in range(n_):
                    for j in range(m_):
                        for k in range(p_):
                            a = i*m_ + j
                            b = j*p_ + k
                            c = k*n_ + i
                            recon = 0.0
                            for r in range(rank):
                                recon += U_r[a, r] * V_r[b, r] * W_r[c, r]
                            errors.append(abs(recon - 1.0))
                
                max_err = max(errors) if errors else 0
                if max_err < 0.01:
                    zero_max = 0
                    for a in range(n_*m_):
                        for b in range(m_*p_):
                            for c in range(p_*n_):
                                if T[a,b,c] == 0:
                                    recon = 0.0
                                    for r in range(rank):
                                        recon += U_r[a, r] * V_r[b, r] * W_r[c, r]
                                    zero_max = max(zero_max, abs(recon))
                    if zero_max < 0.01:
                        best_found = (U_r.copy(), V_r.copy(), W_r.copy(), rank)
                        break
        return best_found
    
    result_4 = _search_2x2x4(max_rank=10, max_iter=1500)
    
    if time.time() > deadline - 10:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # Try combinations
    # Option 1: original 7x2 + 1x1 = 53
    # Option 2: if 2x2x3 has rank 6: 5x3 = 30, then remaining 0... 5*3=15, so 5 blocks of 3
    #   Total rank = 5*6 = 30
    # Option 3: if 2x2x4 has rank 8: 3x4 + 1x3: 3*8 + 6 = 30
    # Option 4: if 2x2x4 has rank 9: 3x4 + 1x3: 3*9 + 6 = 33
    # Option 5: 2x2x5 search
    
    # Try 2x2x5
    def _search_2x2x5(max_rank, max_iter=2000):
        n_, m_, p_ = 2, 2, 5
        T = np.zeros((n_*m_, m_*p_, p_*n_), dtype=float)
        for i in range(n_):
            for j in range(m_):
                for k in range(p_):
                    a = i*m_ + j
                    b = j*p_ + k
                    c = k*n_ + i
                    T[a, b, c] = 1
        
        best_found = None
        for rank in range(8, max_rank+1):
            for attempt in range(max_iter // max_rank):
                if best_found is not None:
                    break
                U_r = np.random.randn(n_*m_, rank)
                V_r = np.random.randn(m_*p_, rank)
                W_r = np.random.randn(p_*n_, rank)
                
                errors = []
                for i in range(n_):
                    for j in range(m_):
                        for k in range(p_):
                            a = i*m_ + j
                            b = j*p_ + k
                            c = k*n_ + i
                            recon = 0.0
                            for r in range(rank):
                                recon += U_r[a, r] * V_r[b, r] * W_r[c, r]
                            errors.append(abs(recon - 1.0))
                
                max_err = max(errors) if errors else 0
                if max_err < 0.01:
                    zero_max = 0
                    for a in range(n_*m_):
                        for b in range(m_*p_):
                            for c in range(p_*n_):
                                if T[a,b,c] == 0:
                                    recon = 0.0
                                    for r in range(rank):
                                        recon += U_r[a, r] * V_r[b, r] * W_r[c, r]
                                    zero_max = max(zero_max, abs(recon))
                    if zero_max < 0.01:
                        best_found = (U_r.copy(), V_r.copy(), W_r.copy(), rank)
                        break
        return best_found
    
    result_5 = _search_2x2x5(max_rank=12, max_iter=1000)
    
    if time.time() > deadline - 5:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # Now try to construct better decompositions based on findings
    # If we have a good 2x2x3 with rank 6:
    if result_3 is not None:
        U3, V3, W3, r3 = result_3
        # 5 blocks of size 3
        blocks_new = []
        for _ in range(5):
            blocks_new.append((3, U3, V3, W3, r3))
        U_new, V_new, W_new = _assemble_blocks(blocks_new)
        rank_new = 5 * r3
        if rank_new < best_rank:
            best_U, best_V, best_W = U_new, V_new, W_new
            best_rank = rank_new
    
    # If we have a good 2x2x4 with rank 8:
    if result_4 is not None:
        U4, V4, W4, r4 = result_4
        # 3 blocks of size 4 + 1 block of size 3 (schoolbook or searched)
        if result_3 is not None:
            U3b, V3b, W3b, r3b = result_3
            blocks_new = []
            for _ in range(3):
                blocks_new.append((4, U4, V4, W4, r4))
            blocks_new.append((3, U3b, V3b, W3b, r3b))
            U_new, V_new, W_new = _assemble_blocks(blocks_new)
            rank_new = 3 * r4 + r3b
            if rank_new < best_rank:
                best_U, best_V, best_W = U_new, V_new, W_new
                best_rank = rank_new
        else:
            # 3 blocks of size 4 + schoolbook 2x2x3 (rank 8)
            blocks_new = []
            for _ in range(3):
                blocks_new.append((4, U4, V4, W4, r4))
            # Schoolbook 2x2x3
            U_sb3 = np.zeros((4, 8), dtype=float)
            V_sb3 = np.zeros((6, 8), dtype=float)
            W_sb3 = np.zeros((6, 8), dtype=float)
            # C[i,k] = sum_j A[i,j] B[j,k] for 2x2x3
            for i in range(2):
                for j in range(2):
                    for k in range(3):
                        r = i*4 + j*2 + k  # 0..7
                        U_sb3[i*2+j, r] = 1
                        V_sb3[j*3+k, r] = 1
                        W_sb3[k*2+i, r] = 1
            blocks_new.append((3, U_sb3, V_sb3, W_sb3, 8))
            U_new, V_new, W_new = _assemble_blocks(blocks_new)
            rank_new = 3 * r4 + 8
            if rank_new < best_rank:
                best_U, best_V, best_W = U_new, V_new, W_new
                best_rank = rank_new
    
    # If we have a good 2x2x5 with rank 10:
    if result_5 is not None:
        U5, V5, W5, r5 = result_5
        # 3 blocks of size 5
        blocks_new = []
        for _ in range(3):
            blocks_new.append((5, U5, V5, W5, r5))
        U_new, V_new, W_new = _assemble_blocks(blocks_new)
        rank_new = 3 * r5
        if rank_new < best_rank:
            best_U, best_V, best_W = U_new, V_new, W_new
            best_rank = rank_new
    
    # Final: also try mixed blocks
    # 2x2x2 (Strassen r=7) + 2x2x2 + 2x2x2 + 2x2x2 + 2x2x2 + 2x2x2 + 2x2x2 + 2x2x1
    # That's our original 53.
    
    # Try: 2x2x4 (if found) + 2x2x4 + 2x2x4 + 2x2x3 (if found)
    if result_4 is not None and result_3 is not None:
        U4, V4, W4, r4 = result_4
        U3, V3, W3, r3 = result_3
        blocks_new = []
        for _ in range(3):
            blocks_new.append((4, U4, V4, W4, r4))
        blocks_new.append((3, U3, V3, W3, r3))
        U_new, V_new, W_new = _assemble_blocks(blocks_new)
        rank_new = 3*r4 + r3
        if rank_new < best_rank:
            best_U, best_V, best_W = U_new, V_new, W_new
            best_rank = rank_new
    
    # Try: 2x2x5 + 2x2x5 + 2x2x5
    if result_5 is not None:
        U5, V5, W5, r5 = result_5
        blocks_new = []
        for _ in range(3):
            blocks_new.append((5, U5, V5, W5, r5))
        U_new, V_new, W_new = _assemble_blocks(blocks_new)
        rank_new = 3*r5
        if rank_new < best_rank:
            best_U, best_V, best_W = U_new, V_new, W_new
            best_rank = rank_new
    
    # Try: 2x2x4 + 2x2x4 + 2x2x4 + 2x2x3 (schoolbook r=8)
    if result_4 is not None:
        U4, V4, W4, r4 = result_4
        U_sb3 = np.zeros((4, 8), dtype=float)
        V_sb3 = np.zeros((6, 8), dtype=float)
        W_sb3 = np.zeros((6, 8), dtype=float)
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    r = i*4 + j*2 + k
                    U_sb3[i*2+j, r] = 1
                    V_sb3[j*3+k, r] = 1
                    W_sb3[k*2+i, r] = 1
        blocks_new = []
        for _ in range(3):
            blocks_new.append((4, U4, V4, W4, r4))
        blocks_new.append((3, U_sb3, V_sb3, W_sb3, 8))
        U_new, V_new, W_new = _assemble_blocks(blocks_new)
        rank_new = 3*r4 + 8
        if rank_new < best_rank:
            best_U, best_V, best_W = U_new, V_new, W_new
            best_rank = rank_new
    
    # Try: 2x2x5 + 2x2x4 + 2x2x4 + 2x2x2
    if result_5 is not None and result_4 is not None:
        U5, V5, W5, r5 = result_5
        U4, V4, W4, r4 = result_4
        blocks_new = []
        blocks_new.append((5, U5, V5, W5, r5))
        for _ in range(2):
            blocks_new.append((4, U4, V4, W4, r4))
        blocks_new.append((2, U_s, V_s, W_s, 7))
        U_new, V_new, W_new = _assemble_blocks(blocks_new)
        rank_new = r5 + 2*r4 + 7
        if rank_new < best_rank:
            best_U, best_V, best_W = U_new, V_new, W_new
            best_rank = rank_new
    
    # Convert to lists of lists
    U_out = [[float(x) for x in row] for row in best_U]
    V_out = [[float(x) for x in row] for row in best_V]
    W_out = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_out, "V": V_out, "W": W_out}

# EVOLVE-BLOCK-END
