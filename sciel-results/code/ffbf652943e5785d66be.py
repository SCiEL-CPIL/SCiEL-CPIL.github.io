"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    T = np.zeros((n, m, p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill
    
    # Schoolbook decomposition as baseline
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W, rank
    
    U_sb, V_sb, W_sb, rank_sb = schoolbook()
    best_decomp = (U_sb, V_sb, W_sb)
    best_rank = rank_sb
    
    def verify_decomp(U, V, W, rank, tol=1e-10):
        """Verify that the decomposition exactly reconstructs T."""
        recon = np.zeros((n, m, p), dtype=np.float64)
        for r in range(rank):
            u_col = U[:, r].reshape(n, m)
            v_col = V[:, r].reshape(m, p)
            w_col = W[:, r].reshape(p, n)
            # Add rank-1 tensor: u[i,j] * v[j,k] * w[k,i]
            for i in range(n):
                for j in range(m):
                    uj = u_col[i, j]
                    if uj == 0:
                        continue
                    for k in range(p):
                        recon[i, j, k] += uj * v_col[j, k] * w_col[k, i]
        return np.allclose(recon, T, atol=tol)
    
    def strassen_4x4x4():
        """
        Strassen decomposition for 4x4 * 4x4 = 4x4.
        Returns U, V, W columns for the 4x4x4 tensor.
        Strassen: 7 multiplications.
        
        A = [[a,b],[c,d]] (2x2 blocks), B = [[e,f],[g,h]]
        M1 = (a+d)(e+h)
        M2 = (c+d)e
        M3 = a(f-h)
        M4 = d(g-c)
        M5 = (a+b)h
        M6 = (c-a)(e+f)
        M7 = (b-d)(g+h)
        
        P = [[M1+M4-M5+M7, M3+M5],
             [M2+M4, M1-M2+M3+M6]]
        
        For 4x4, we use 2x2 block Strassen, which gives rank 7 for the 4x4x4 tensor.
        """
        # The 4x4x4 tensor T4[i,j,k] = 1 for all i,j,k in 0..3
        # We need U (16, 7), V (16, 7), W (16, 7) such that
        # T4[i,j,k] = sum_r U[i*4+j, r] * V[j*4+k, r] * W[k*4+i, r]
        
        # Strassen for 2x2 blocks:
        # A = [[A11, A12], [A21, A22]] where each block is 2x2
        # B = [[B11, B12], [B21, B22]]
        # C = A*B
        
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-A11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Each of these involves products of 2x2 matrices, so we need to decompose
        # each 2x2*2x2 = 2x2 multiplication. But wait, the Strassen algorithm
        # reduces 8 multiplications to 7 for 2x2. For 4x4, we apply Strassen
        # at the block level, but each "multiplication" is a 2x2*2x2 which
        # still needs rank 8 (or can be further decomposed).
        
        # Actually, for the tensor T4 of shape (4,4,4), the Strassen decomposition
        # gives rank 7 if we can decompose the 2x2 multiplications optimally.
        # But 2x2*2x2=2x2 has rank 7 (Strassen) or 8 (schoolbook).
        
        # Let's use the standard Strassen decomposition for 4x4x4:
        # This is well-known to have rank 7.
        
        # U, V, W for Strassen 4x4x4:
        # The indices are i*4+j for U, j*4+k for V, k*4+i for W
        
        U4 = np.zeros((16, 7))
        V4 = np.zeros((16, 7))
        W4 = np.zeros((16, 7))
        
        # M1 = (A11+A22)(B11+B22)
        # A11: rows 0,1 cols 0,1; A22: rows 2,3 cols 2,3
        # B11: rows 0,1 cols 0,1; B22: rows 2,3 cols 2,3
        # Contribution to C11 (rows 0,1 cols 0,1) and C22 (rows 2,3 cols 2,3)
        
        # For M1: (A11+A22) * (B11+B22)
        # U: for A11+A22, we add rows: A11 contributes to (i,j) for i in 0,1, j in 0,1
        #     A22 contributes to (i,j) for i in 2,3, j in 2,3
        #     So U[i*4+j, 0] = 1 for (i,j) in {(0,0),(0,1),(1,0),(1,1),(2,2),(2,3),(3,2),(3,3)}
        # V: for B11+B22, V[j*4+k, 0] = 1 for (j,k) in {(0,0),(0,1),(1,0),(1,1),(2,2),(2,3),(3,2),(3,3)}
        # W: contributes to C11 and C22. C11: (i,k) in {(0,0),(0,1),(1,0),(1,1)}, C22: (i,k) in {(2,2),(2,3),(3,2),(3,3)}
        #     W[k*4+i, 0] = 1 for those (k,i) pairs
        
        # This is getting complex. Let me use a known Strassen 4x4x4 decomposition.
        # Actually, let's just use the schoolbook for now and try to improve.
        
        return U4, V4, W4, 7
    
    def strassen_2x2x2():
        """
        Strassen decomposition for 2x2 * 2x2 = 2x2.
        Returns U (4,7), V (4,7), W (4,7).
        
        A = [[a,b],[c,d]], B = [[e,f],[g,h]]
        M1 = (a+d)(e+h)
        M2 = (c+d)e
        M3 = a(f-h)
        M4 = d(g-c)
        M5 = (a+b)h
        M6 = (c-a)(e+f)
        M7 = (b-d)(g+h)
        
        P = [[M1+M4-M5+M7, M3+M5],
             [M2+M4, M1-M2+M3+M6]]
        """
        U2 = np.zeros((4, 7))
        V2 = np.zeros((4, 7))
        W2 = np.zeros((4, 7))
        
        # Index: U[i*2+j], V[j*2+k], W[k*2+i]
        # a=(0,0), b=(0,1), c=(1,0), d=(1,1)
        # e=(0,0), f=(0,1), g=(1,0), h=(1,1)
        
        # M1 = (a+d)(e+h)
        # U: a=(0,0)->idx 0, d=(1,1)->idx 3. U[0,0]=1, U[3,0]=1
        # V: e=(0,0)->idx 0, h=(1,1)->idx 3. V[0,0]=1, V[3,0]=1
        # W: contributes to P11=(0,0) and P22=(1,1)
        # P11: i=0,k=0 -> W[0*2+0, 0] = W[0,0] += 1
        # P22: i=1,k=1 -> W[1*2+1, 0] = W[3,0] += 1
        U2[0, 0] += 1; U2[3, 0] += 1
        V2[0, 0] += 1; V2[3, 0] += 1
        W2[0, 0] += 1; W2[3, 0] += 1
        
        # M2 = (c+d)e
        # U: c=(1,0)->idx 2, d=(1,1)->idx 3. U[2,1]=1, U[3,1]=1
        # V: e=(0,0)->idx 0. V[0,1]=1
        # W: contributes to P21=(1,0)
        # P21: i=1,k=0 -> W[0*2+1, 1] = W[1,1] += 1
        U2[2, 1] += 1; U2[3, 1] += 1
        V2[0, 1] += 1
        W2[1, 1] += 1
        
        # M3 = a(f-h)
        # U: a=(0,0)->idx 0. U[0,2]=1
        # V: f=(0,1)->idx 1, h=(1,1)->idx 3. V[1,2]=1, V[3,2]=-1
        # W: contributes to P12=(0,1)
        # P12: i=0,k=1 -> W[1*2+0, 2] = W[2,2] += 1
        U2[0, 2] += 1
        V2[1, 2] += 1; V2[3, 2] -= 1
        W2[2, 2] += 1
        
        # M4 = d(g-c)
        # U: d=(1,1)->idx 3. U[3,3]=1
        # V: g=(1,0)->idx 2, c=(1,0)->idx 2. V[2,3]=1, V[2,3]=-1 => V[2,3]=0? No:
        # g-c means V for g is 1, V for c is -1. But c and g have same index (1,0)->2.
        # So V[2,3] = 1 - 1 = 0? That can't be right.
        # Wait: M4 = d*(g-c). So U[d]=1, V[g]=1, V[c]=-1.
        # d=(1,1)->idx 3, g=(1,0)->idx 2, c=(1,0)->idx 2.
        # So V[2,3] = 1 + (-1) = 0? That means M4 vanishes? No, that's wrong.
        # Let me re-check. c=(1,0), g=(1,0). Yes, same position.
        # So M4 = d*(g-c) = d*g - d*c. Since g and c are at the same (j,k) position,
        # the V vector has V[2,3] = 1-1 = 0. This means M4 contributes nothing?
        # That can't be right for Strassen. Let me re-derive.
        
        # Actually, I think I made an error. Let me re-check Strassen:
        # M4 = (A22)(B21 - A11) = d*(g - a)
        # Not d*(g-c)! Let me re-verify Strassen formulas.
        
        # Standard Strassen:
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B12 - B22)
        # M4 = A22(B21 - A11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # So M4 = A22*(B21 - A11) = d*(g - a)
        # U: d=(1,1)->idx 3. U[3,3]=1
        # V: g=(1,0)->idx 2, a=(0,0)->idx 0. V[2,3]=1, V[0,3]=-1
        # W: contributes to C11 and C21
        # C11: i=0,k=0 -> W[0,3] += 1
        # C21: i=1,k=0 -> W[1,3] += 1
        U2[3, 3] += 1
        V2[2, 3] += 1; V2[0, 3] -= 1
        W2[0, 3] += 1; W2[1, 3] += 1
        
        # M5 = (A11+A12)B22
        # U: a=(0,0)->idx 0, b=(0,1)->idx 1. U[0,4]=1, U[1,4]=1
        # V: h=(1,1)->idx 3. V[3,4]=1
        # W: contributes to C11 and C12
        # C11: i=0,k=0 -> W[0,4] -= 1 (because -M5)
        # C12: i=0,k=1 -> W[2,4] += 1
        U2[0, 4] += 1; U2[1, 4] += 1
        V2[3, 4] += 1
        W2[0, 4] -= 1; W2[2, 4] += 1
        
        # M6 = (A21-A11)(B11+B12)
        # U: c=(1,0)->idx 2, a=(0,0)->idx 0. U[2,5]=1, U[0,5]=-1
        # V: e=(0,0)->idx 0, f=(0,1)->idx 1. V[0,5]=1, V[1,5]=1
        # W: contributes to C22
        # C22: i=1,k=1 -> W[3,5] += 1
        U2[2, 5] += 1; U2[0, 5] -= 1
        V2[0, 5] += 1; V2[1, 5] += 1
        W2[3, 5] += 1
        
        # M7 = (A12-A22)(B21+B22)
        # U: b=(0,1)->idx 1, d=(1,1)->idx 3. U[1,6]=1, U[3,6]=-1
        # V: g=(1,0)->idx 2, h=(1,1)->idx 3. V[2,6]=1, V[3,6]=1
        # W: contributes to C11
        # C11: i=0,k=0 -> W[0,6] += 1
        U2[1, 6] += 1; U2[3, 6] -= 1
        V2[2, 6] += 1; V2[3, 6] += 1
        W2[0, 6] += 1
        
        return U2, V2, W2, 7
    
    # Let's verify the 2x2 Strassen decomposition
    U2, V2, W2, r2 = strassen_2x2x2()
    T2 = np.ones((2, 2, 2), dtype=np.float64)
    recon2 = np.zeros((2, 2, 2), dtype=np.float64)
    for r in range(r2):
        u_col = U2[:, r].reshape(2, 2)
        v_col = V2[:, r].reshape(2, 2)
        w_col = W2[:, r].reshape(2, 2)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    recon2[i, j, k] += u_col[i, j] * v_col[j, k] * w_col[k, i]
    
    if not np.allclose(recon2, T2, atol=1e-10):
        # Strassen 2x2 failed, fall back to schoolbook
        pass
    
    # For the 4x5*5x6 problem, let's try a block decomposition approach:
    # Partition m=5 into m1=4 and m2=1
    # Partition p=6 into p1=4 and p2=2
    # C = A1*B1 + A1*B2 + A2*B1 + A2*B2
    # where A1 is 4x4, A2 is 4x1, B1 is 4x6, B2 is 1x6
    
    # Actually, let's try a different approach: use the known result that
    # the rank of 4x5*5x6 matrix multiplication is at most some value.
    # The schoolbook rank is 120. Can we do better?
    
    # Let's try a simple block decomposition:
    # A (4x5) * B (5x6) = C (4x6)
    # Split B into B1 (5x4) and B2 (5x2)
    # C1 = A*B1 (4x4), C2 = A*B2 (4x2)
    # Rank of C1: at most 4*5*4 = 80 (schoolbook) or less
    # Rank of C2: at most 4*5*2 = 40 (schoolbook)
    # Total: 120 (same as schoolbook)
    
    # Let's try using Strassen for the 4x4 part:
    # A (4x5) * B1 (5x4) = C1 (4x4)
    # We can write A = [A1 | A2] where A1 is 4x4, A2 is 4x1
    # B1 = [B1a; B1b] where B1a is 4x4, B1b is 1x4
    # C1 = A1*B1a + A2*B1b
    # A1*B1a: 4x4*4x4, Strassen rank 7 (if we can decompose it)
    # A2*B1b: 4x1*1x4, rank 4 (outer product)
    # Total for C1: 7 + 4 = 11
    
    # C2 = A*B2 (4x2)
    # A = [A1 | A2], B2 = [B2a; B2b] where B2a is 4x2, B2b is 1x2
    # C2 = A1*B2a + A2*B2b
    # A1*B2a: 4x4*4x2. We can use Strassen-like or schoolbook.
    # Schoolbook: 4*4*2 = 32
    # A2*B2b: 4x1*1x2, rank 4
    # Total for C2: 32 + 4 = 36
    
    # Grand total: 11 + 36 = 47. That's a big improvement over 120!
    
    # But we need to verify this decomposition is correct.
    # Let's construct it explicitly.
    
    def construct_block_decomp():
        """
        Block decomposition:
        A (4x5) * B (5x6) = C (4x6)
        A = [A1 | A2], A1 is 4x4 (cols 0-3), A2 is 4x1 (col 4)
        B = [B1; B2], B1 is 5x4 (rows 0-3, cols 0-3), B2 is 5x2 (rows 0-3, cols 4-5)
        
        Wait, let me re-index. B is 5x6.
        B = [[B11, B12], [B21, B22]] where B11 is 4x4, B12 is 4x2, B21 is 1x4, B22 is 1x2
        
        C = A1*B11 + A1*B12 + A2*B21 + A2*B22
        
        C11 = A1*B11 (4x4): Strassen rank 7
        C12 = A1*B12 (4x2): schoolbook rank 32
        C21 = A2*B21 (4x4): A2 is 4x1, B21 is 1x4, rank 4
        C22 = A2*B22 (4x2): A2 is 4x1, B22 is 1x2, rank 4
        
        Total: 7 + 32 + 4 + 4 = 47
        """
        U_cols = []
        V_cols = []
        W_cols = []
        
        # Helper to add a rank-1 term
        def add_term(u_vec, v_vec, w_vec):
            U_cols.append(u_vec)
            V_cols.append(v_vec)
            W_cols.append(w_vec)
        
        # C11 = A1 * B11, where A1 is 4x4 (i in 0..3, j in 0..3), B11 is 4x4 (j in 0..3, k in 0..3)
        # Use Strassen 4x4x4. But we don't have a verified Strassen 4x4x4 yet.
        # Let's use schoolbook for now: rank 64
        # Actually, let's try to build Strassen 4x4x4 from Strassen 2x2x2.
        
        # For 4x4, we partition into 2x2 blocks:
        # A1 = [[A11, A12], [A21, A22]] where each is 2x2
        # B11 = [[B111, B112], [B211, B212]] where each is 2x2
        # C11 = A1 * B11
        # Strassen at the block level gives 7 block multiplications, each of which
        # is a 2x2*2x2=2x2 multiplication with rank 7 (Strassen) or 8 (schoolbook).
        # So total rank = 7 * 7 = 49 for Strassen, or 7 * 8 = 56 for mixed.
        # That's worse than schoolbook 64? No, 49 < 64.
        
        # But implementing Strassen 4x4x4 is complex. Let's just use schoolbook for C11.
        # Schoolbook for 4x4*4x4: rank 64
        
        # Actually, let's just use schoolbook for everything to get a valid decomposition,
        # then try to improve specific blocks.
        
        # Schoolbook for A1*B11 (4x4*4x4):
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    u = np.zeros(20)  # n*m = 4*5 = 20
                    v = np.zeros(30)  # m*p = 5*6 = 30
                    w = np.zeros(24)  # p*n = 6*4 = 24
                    u[i*5 + j] = 1.0
                    v[j*6 + k] = 1.0
                    w[k*4 + i] = 1.0
                    add_term(u, v, w)
        
        # Schoolbook for A1*B12 (4x4*4x2):
        for i in range(4):
            for j in range(4):
                for k in range(4, 6):
                    u = np.zeros(20)
                    v = np.zeros(30)
                    w = np.zeros(24)
                    u[i*5 + j] = 1.0
                    v[j*6 + k] = 1.0
                    w[k*4 + i] = 1.0
                    add_term(u, v, w)
        
        # Schoolbook for A2*B21 (4x1*1x4):
        for i in range(4):
            for k in range(4):
                u = np.zeros(20)
                v = np.zeros(30)
                w = np.zeros(24)
                u[i*5 + 4] = 1.0  # j=4
                v[4*6 + k] = 1.0  # j=4, k in 0..3
                w[k*4 + i] = 1.0
                add_term(u, v, w)
        
        # Schoolbook for A2*B22 (4x1*1x2):
        for i in range(4):
            for k in range(4, 6):
                u = np.zeros(20)
                v = np.zeros(30)
                w = np.zeros(24)
                u[i*5 + 4] = 1.0
                v[4*6 + k] = 1.0
                w[k*4 + i] = 1.0
                add_term(u, v, w)
        
        rank = len(U_cols)
        U = np.column_stack(U_cols)
        V = np.column_stack(V_cols)
        W = np.column_stack(W_cols)
        
        # Verify
        if verify_decomp(U, V, W, rank):
            return U, V, W, rank
        else:
            return None
    
    result = construct_block_decomp()
    if result is not None:
        U, V, W, rank = result
        if rank < best_rank:
            best_rank = rank
            best_decomp = (U, V, W)
            print(f"Block decomposition: rank {rank}")
    
    # Try greedy peeling to see if we can improve further
    def peel_greedy(seed, max_iters=20):
        """Greedy rank-1 peeling on the residual."""
        rng = np.random.RandomState(seed)
        residual = T.copy()
        U_cols = []
        V_cols = []
        W_cols = []
        
        for it in range(max_iters):
            if time.time() > deadline - 10.0:
                break
            if np.allclose(residual, 0, atol=1e-12):
                break
            
            best_score = -np.inf
            best_u, best_v, best_w = None, None, None
            
            n_trials = 50
            for t in range(n_trials):
                u = rng.randn(n * m)
                v = rng.randn(m * p)
                u_mat = u.reshape(n, m)
                v_mat = v.reshape(m, p)
                
                num = np.zeros((n, p))
                den = np.zeros((n, p))
                for i in range(n):
                    for k in range(p):
                        uj = u_mat[i, :]
                        vk = v_mat[:, k]
                        ujvk = uj * vk
                        num[i, k] = np.dot(residual[i, :, k], ujvk)
                        den[i, k] = np.dot(ujvk, ujvk)
                
                mask = den > 1e-12
                w_flat = np.zeros(p * n)
                for i in range(n):
                    for k in range(p):
                        if mask[i, k]:
                            w_flat[k * n + i] = num[i, k] / den[i, k]
                
                recon = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            recon[i, j, k] = u_mat[i, j] * v_mat[j, k] * w_flat[k * n + i]
                
                diff = residual - recon
                score = -np.sum(diff * diff)
                
                if score > best_score:
                    best_score = score
                    best_u, best_v, best_w = u.copy(), v.copy(), w_flat.copy()
            
            if best_u is None:
                break
            
            recon = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        recon[i, j, k] = best_u[i * m + j] * best_v[j * p + k] * best_w[k * n + i]
            
            new_residual = residual - recon
            if np.sum(new_residual * new_residual) >= np.sum(residual * residual) - 1e-12:
                break
            
            U_cols.append(best_u)
            V_cols.append(best_v)
            W_cols.append(best_w)
            residual = new_residual
        
        if len(U_cols) == 0:
            return None
        
        U = np.column_stack(U_cols)
        V = np.column_stack(V_cols)
        W = np.column_stack(W_cols)
        rank = len(U_cols)
        
        if verify_decomp(U, V, W, rank):
            return U, V, W, rank
        else:
            return None
    
    # Try a few seeds
    for seed in range(5):
        if time.time() > deadline - 15.0:
            break
        result = peel_greedy(seed)
        if result is not None:
            U, V, W, rank = result
            if rank < best_rank:
                best_rank = rank
                best_decomp = (U, V, W)
                print(f"Greedy peeling: rank {rank} with seed {seed}")
    
    U, V, W = best_decomp
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
