import time
import numpy as np
import random

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit

    # Precompute target tensor T of shape (n*m, m*p, p*n)
    # T[u, v, w] = 1 if u=i*m+j, v=j*p+k, w=k*n+i, else 0
    U_size = n * m
    V_size = m * p
    W_size = p * n
    T = np.zeros((U_size, V_size, W_size), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0

    def verify(U, V, W):
        """Check if U @ V^T @ W^T == T (tensor equality)"""
        rank = U.shape[1]
        # Compute tensor: T_comp[u, v, w] = sum_r U[u,r] V[v,r] W[w,r]
        # Use einsum: 'ur,vr,wr->uvw'
        T_comp = np.einsum('ur,vr,wr->uvw', U, V, W)
        return np.allclose(T_comp, T, atol=1e-8)

    def random_decomposition(rank):
        """Generate a random valid decomposition of rank `rank`."""
        # Start with a valid schoolbook-like decomposition but with random coefficients
        # Actually, easiest is to start with a known valid one and perturb, or just use random and hope?
        # No, random won't be valid.
        # Let's construct a valid one using the "trivial" decomposition but expanded.
        # Trivial rank is n*m*p = 112.
        # We can start with rank 112 and reduce.
        
        # Let's just return a valid rank-112 decomposition (schoolbook)
        # This is deterministic and valid.
        U = np.zeros((U_size, rank), dtype=float)
        V = np.zeros((V_size, rank), dtype=float)
        W = np.zeros((W_size, rank), dtype=float)
        
        if rank >= n*m*p:
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u = i * m + j
                        v = j * p + k
                        w = k * n + i
                        U[u, col] = 1.0
                        V[v, col] = 1.0
                        W[w, col] = 1.0
                        col += 1
            return U, V, W
        
        # If rank < n*m*p, we can't easily generate a valid one randomly.
        # So we must start with a valid one and reduce.
        # The caller should ensure rank is sufficient or handle it.
        # For this function, let's assume we are called with rank >= n*m*p
        # or we just return zeros (invalid) if not, and let the search fix it?
        # No, search needs a valid starting point.
        # We will always start with the full schoolbook decomposition.
        pass

    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((U_size, rank), dtype=float)
        V = np.zeros((V_size, rank), dtype=float)
        W = np.zeros((W_size, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    U[u, col] = 1.0
                    V[v, col] = 1.0
                    W[w, col] = 1.0
                    col += 1
        return U, V, W

    # Initialize with schoolbook
    U, V, W = get_schoolbook()
    current_rank = U.shape[1]
    
    # We want to reduce rank.
    # Strategy:
    # 1. Try to remove columns (terms) one by one.
    # 2. If removal fails, try to replace a term with a new one that allows removing another?
    #    Or use SVD to find a better basis?
    
    # Let's try a more aggressive reduction:
    # Instead of just removing, we can try to "merge" terms or find linear dependencies.
    
    # A powerful method:
    # Flatten the decomposition matrices.
    # The condition is that the tensor T is in the span of the rank-1 tensors.
    
    # Let's try iterative removal with a twist:
    # If we can't remove a term, maybe we can modify it to make another term removable?
    # This is complex.
    
    # Let's stick to the greedy removal but do it smarter.
    # The previous program did this. Why did it get 1.0?
    # It started with Strassen (97) and removed down to... ?
    # The prompt says "score 1.0000" for the current program.
    # The inspiration says "reduces total rank from 97 to 93, improving score from 1 to ~1.155".
    # This implies the seed program (score 1.0) likely ended up with rank 97 or higher, or the seed was different.
    # Wait, the "Current Program" in the prompt is the one with score 1.0000.
    # It uses Strassen 4x4x4 (49) + Schoolbook 4x4x3 (48) = 97.
    # It then does greedy elimination.
    # If it finished with 97, it removed 0 terms.
    # If it removed some, it would be better.
    # Maybe the greedy elimination is too slow or fails to find removable terms?
    # Or maybe the verification is too strict/slow?
    
    # My new strategy:
    # 1. Start with a better initial decomposition.
    #    Use Strassen for 4x4x4 (49) and Strassen for 4x4x2 (28) + Schoolbook for 4x4x1 (16).
    #    Total = 49 + 28 + 16 = 93.
    #    This matches the inspiration's "93".
    # 2. Perform a more robust reduction search.
    #    - Try removing terms.
    #    - If stuck, try "swapping" a term with a random vector and checking if we can then remove another term?
    #      Or just try to reduce rank by solving linear systems?
    
    # Let's implement the "Strassen 4x4x2" and "Schoolbook 4x4x1" parts.
    
    def strassen_2x2():
        U = np.zeros((4, 7), dtype=float)
        V = np.zeros((4, 7), dtype=float)
        W = np.zeros((4, 7), dtype=float)
        
        # p1 = (a+d)*(e+h)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # p2 = (c+d)*e
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = 1
        
        # p3 = a*(f-h)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1
        
        # p4 = d*(g-e)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        # p5 = (a+b)*h
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        # p6 = (c-a)*(e+f)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # p7 = (b-d)*(g+h)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W

    def block_indices(br, bc, size=4):
        indices = []
        for r in range(2):
            for c in range(2):
                row = br * 2 + r
                col = bc * 2 + c
                indices.append(row * size + col)
        return indices

    def strassen_4x4x4():
        rank = 49
        U = np.zeros((16, rank), dtype=float)
        V = np.zeros((16, rank), dtype=float)
        W = np.zeros((16, rank), dtype=float)
        
        U2, V2, W2 = strassen_2x2()
        
        A11_idx = block_indices(0, 0)
        A12_idx = block_indices(0, 1)
        A21_idx = block_indices(1, 0)
        A22_idx = block_indices(1, 1)
        
        B11_idx = block_indices(0, 0)
        B12_idx = block_indices(0, 1)
        B21_idx = block_indices(1, 0)
        B22_idx = block_indices(1, 1)
        
        C11_idx = block_indices(0, 0)
        C12_idx = block_indices(0, 1)
        C21_idx = block_indices(1, 0)
        C22_idx = block_indices(1, 1)
        
        col = 0
        # P1 = (A11 + A22) * (B11 + B22)
        for r in range(7):
            for idx in A11_idx: U[idx, col] += U2[idx % 4, r]
            for idx in A22_idx: U[idx, col] += U2[idx % 4, r]
            for idx in B11_idx: V[idx, col] += V2[idx % 4, r]
            for idx in B22_idx: V[idx, col] += V2[idx % 4, r]
            for idx in C11_idx: W[idx, col] += W2[idx % 4, r]
            for idx in C22_idx: W[idx, col] -= W2[idx % 4, r]
            col += 1
        
        # P2 = (A21 + A22) * B11
        for r in range(7):
            for idx in A21_idx: U[idx, col] += U2[idx % 4, r]
            for idx in A22_idx: U[idx, col] += U2[idx % 4, r]
            for idx in B11_idx: V[idx, col] += V2[idx % 4, r]
            for idx in C21_idx: W[idx, col] += W2[idx % 4, r]
            for idx in C22_idx: W[idx, col] -= W2[idx % 4, r]
            col += 1
        
        # P3 = A11 * (B12 - B22)
        for r in range(7):
            for idx in A11_idx: U[idx, col] += U2[idx % 4, r]
            for idx in B12_idx: V[idx, col] += V2[idx % 4, r]
            for idx in B22_idx: V[idx, col] -= V2[idx % 4, r]
            for idx in C12_idx: W[idx, col] += W2[idx % 4, r]
            for idx in C22_idx: W[idx, col] += W2[idx % 4, r]
            col += 1
        
        # P4 = A22 * (B21 - B11)
        for r in range(7):
            for idx in A22_idx: U[idx, col] += U2[idx % 4, r]
            for idx in B21_idx: V[idx, col] += V2[idx % 4, r]
            for idx in B11_idx: V[idx, col] -= V2[idx % 4, r]
            for idx in C11_idx: W[idx, col] += W2[idx % 4, r]
            for idx in C21_idx: W[idx, col] += W2[idx % 4, r]
            col += 1
        
        # P5 = (A11 + A12) * B22
        for r in range(7):
            for idx in A11_idx: U[idx, col] += U2[idx % 4, r]
            for idx in A12_idx: U[idx, col] += U2[idx % 4, r]
            for idx in B22_idx: V[idx, col] += V2[idx % 4, r]
            for idx in C11_idx: W[idx, col] -= W2[idx % 4, r]
            for idx in C12_idx: W[idx, col] += W2[idx % 4, r]
            col += 1
        
        # P6 = (A21 - A11) * (B11 + B12)
        for r in range(7):
            for idx in A21_idx: U[idx, col] += U2[idx % 4, r]
            for idx in A11_idx: U[idx, col] -= U2[idx % 4, r]
            for idx in B11_idx: V[idx, col] += V2[idx % 4, r]
            for idx in B12_idx: V[idx, col] += V2[idx % 4, r]
            for idx in C22_idx: W[idx, col] += W2[idx % 4, r]
            col += 1
        
        # P7 = (A12 - A22) * (B21 + B22)
        for r in range(7):
            for idx in A12_idx: U[idx, col] += U2[idx % 4, r]
            for idx in A22_idx: U[idx, col] -= U2[idx % 4, r]
            for idx in B21_idx: V[idx, col] += V2[idx % 4, r]
            for idx in B22_idx: V[idx, col] += V2[idx % 4, r]
            for idx in C11_idx: W[idx, col] += W2[idx % 4, r]
            col += 1
        
        return U, V, W

    def strassen_4x4x2():
        """
        Decompose 4x4 * 4x2 into 4x2.
        Use Strassen on 2x2 blocks.
        A is 4x4, B is 4x2.
        Split A into 2x2 blocks: A11, A12, A21, A22 (each 2x2).
        Split B into 2x1 blocks: B1 (top 2x1), B2 (bottom 2x1).
        Wait, B is 4x2. So B1 is 2x2? No, B is 4 rows, 2 cols.
        Let's split rows of B into top 2 and bottom 2.
        B_top: 2x2, B_bot: 2x2.
        
        We want C = A * B. C is 4x2.
        C_top = A11*B_top + A12*B_bot
        C_bot = A21*B_top + A22*B_bot
        
        This is just 4 multiplications of 2x2 * 2x2.
        Rank = 4 * 7 = 28.
        """
        rank = 28
        U = np.zeros((16, rank), dtype=float)
        V = np.zeros((8, rank), dtype=float) # 4*2
        W = np.zeros((8, rank), dtype=float) # 2*4
        
        U2, V2, W2 = strassen_2x2()
        
        A11_idx = block_indices(0, 0)
        A12_idx = block_indices(0, 1)
        A21_idx = block_indices(1, 0)
        A22_idx = block_indices(1, 1)
        
        # B indices:
        # B is 4x2.
        # B_top is rows 0,1. B_bot is rows 2,3.
        # In V (4*2=8), rows 0-3 are B_top, rows 4-7 are B_bot.
        # V2 is (4, 7) for 2x2.
        # So for B_top, V rows 0-3 correspond to V2 rows 0-3.
        # For B_bot, V rows 4-7 correspond to V2 rows 0-3.
        
        # C indices:
        # C is 4x2.
        # C_top is rows 0,1. C_bot is rows 2,3.
        # In W (2*4=8), rows 0-3 are C_top (cols 0,1 of C? No, W[k*n+i]).
        # W is (p*n, rank). p=2, n=4.
        # W index = k*4 + i.
        # C_top corresponds to i=0,1.
        # C_bot corresponds to i=2,3.
        # W2 is (4, 7) for 2x2 output (p=2, n=2).
        # W2 index = k*2 + i.
        
        col = 0
        # C_top = A11*B_top + A12*B_bot
        
        # Term 1: A11 * B_top
        for r in range(7):
            for idx in A11_idx: U[idx, col] += U2[idx % 4, r]
            for idx in range(4): V[idx, col] += V2[idx, r] # B_top
            # W: C_top. i=0,1.
            # W2 gives 2x2 output.
            # We need to map W2 (2x2) to W (2x4) rows 0-3.
            # W2[k*2+i] -> W[k*4+i] for i in 0,1.
            for k in range(2):
                for i in range(2):
                    W[k*4+i, col] += W2[k*2+i, r]
            col += 1
            
        # Term 2: A12 * B_bot
        for r in range(7):
            for idx in A12_idx: U[idx, col] += U2[idx % 4, r]
            for idx in range(4): V[4+idx, col] += V2[idx, r] # B_bot
            for k in range(2):
                for i in range(2):
                    W[k*4+i, col] += W2[k*2+i, r]
            col += 1
            
        # C_bot = A21*B_top + A22*B_bot
        
        # Term 3: A21 * B_top
        for r in range(7):
            for idx in A21_idx: U[idx, col] += U2[idx % 4, r]
            for idx in range(4): V[idx, col] += V2[idx, r] # B_top
            # W: C_bot. i=2,3.
            for k in range(2):
                for i in range(2):
                    W[k*4+2+i, col] += W2[k*2+i, r]
            col += 1
            
        # Term 4: A22 * B_bot
        for r in range(7):
            for idx in A22_idx: U[idx, col] += U2[idx % 4, r]
            for idx in range(4): V[4+idx, col] += V2[idx, r] # B_bot
            for k in range(2):
                for i in range(2):
                    W[k*4+2+i, col] += W2[k*2+i, r]
            col += 1
            
        assert col == 28
        return U, V, W

    def schoolbook_4x4x1():
        """
        4x4 * 4x1 -> 4x1.
        Rank = 4*4*1 = 16.
        """
        rank = 16
        U = np.zeros((16, rank), dtype=float)
        V = np.zeros((4, rank), dtype=float) # 4*1
        W = np.zeros((4, rank), dtype=float) # 1*4
        
        col = 0
        for i in range(4):
            for j in range(4):
                U[i*4+j, col] = 1.0
                V[j, col] = 1.0
                W[i, col] = 1.0
                col += 1
                
        assert col == 16
        return U, V, W

    # Construct the 93-rank decomposition
    U1, V1, W1 = strassen_4x4x4() # 49
    U2, V2, W2 = strassen_4x4x2() # 28
    U3, V3, W3 = schoolbook_4x4x1() # 16
    
    rank_total = 49 + 28 + 16
    U_full = np.zeros((16, rank_total), dtype=float)
    V_full = np.zeros((28, rank_total), dtype=float)
    W_full = np.zeros((28, rank_total), dtype=float)
    
    # Part 1: 4x4x4 (B cols 0-3, C cols 0-3)
    # V1 is (16, 49). V_full rows 0-15 are B[0:4, 0:4].
    # W1 is (16, 49). W_full rows 0-15 are C[0:4, 0:4].
    U_full[:, :49] = U1
    V_full[:16, :49] = V1
    W_full[:16, :49] = W1
    
    # Part 2: 4x4x2 (B cols 4-5, C cols 4-5)
    # V2 is (8, 28). V_full rows 16-23 are B[0:4, 4:6].
    # W2 is (8, 28). W_full rows 16-23 are C[4:6, 0:4].
    U_full[:, 49:77] = U2
    V_full[16:24, 49:77] = V2
    W_full[16:24, 49:77] = W2
    
    # Part 3: 4x4x1 (B col 6, C col 6)
    # V3 is (4, 16). V_full rows 24-27 are B[0:4, 6].
    # W3 is (4, 16). W_full rows 24-27 are C[6, 0:4].
    U_full[:, 77:] = U3
    V_full[24:28, 77:] = V3
    W_full[24:28, 77:] = W3
    
    # Verify
    if not verify(U_full, V_full, W_full):
        # Fallback
        U, V, W = get_schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    current_U = U_full.copy()
    current_V = V_full.copy()
    current_W = W_full.copy()
    current_rank = rank_total
    
    # Reduction Search
    # 1. Greedy Removal
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for term_idx in range(current_rank - 1, -1, -1):
            if time.time() > deadline - 5:
                break
            
            mask = [i for i in range(current_rank) if i != term_idx]
            U_new = current_U[:, mask]
            V_new = current_V[:, mask]
            W_new = current_W[:, mask]
            
            if verify(U_new, V_new, W_new):
                current_U = U_new
                current_V = V_new
                current_W = W_new
                current_rank -= 1
                improved = True
                # Continue loop (indices < term_idx are unaffected)
                
    # 2. Randomized Local Search (Swap/Replace)
    # If we are stuck, try to replace a term with a new one that might allow further reduction?
    # Or just try random perturbations of the vectors and see if we can drop rank?
    # This is risky. Let's stick to greedy removal which is safe.
    
    # If time remains, try a few random restarts of the removal process?
    # Or try removing pairs?
    
    # Final Check
    if not verify(current_U, current_V, current_W):
        U, V, W = get_schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    return {"U": current_U.tolist(), "V": current_V.tolist(), "W": current_W.tolist()}
