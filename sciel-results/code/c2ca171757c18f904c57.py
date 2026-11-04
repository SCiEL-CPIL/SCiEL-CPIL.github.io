import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """Find rank-minimal exact tensor decomposition for 2x3 * 3x6 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    n, m, p = DIMENSIONS  # 2, 3, 6
    schoolbook_rank = n * m * p  # 36
    
    def verify_decomposition(U, V, W, rank):
        """Verify that the decomposition exactly represents matrix multiplication."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def schoolbook():
        rank = schoolbook_rank
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
        return U, V, W, rank
    
    def strassen_2x2():
        """
        Strassen decomposition for 2x2 * 2x2 matrix multiplication.
        Rank 7.
        U has shape (4, 7), V has shape (4, 7), W has shape (4, 7)
        Indexing: U[i*2+j], V[j*2+k], W[k*2+i]
        """
        # Strassen's 7 products:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        #
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # A indices: A00=0, A01=1, A10=2, A11=3
        # B indices: B00=0, B01=1, B10=2, B11=3
        # C indices: C00=0, C01=1, C10=2, C11=3
        
        # M1: (A00+A11)(B00+B11)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1  # C00 += M1, C11 += M1
        
        # M2: (A10+A11)B00
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1; W[3][1] = -1  # C10 += M2, C11 -= M2
        
        # M3: A00(B01-B11)
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[1][2] = 1; W[3][2] = 1  # C01 += M3, C11 += M3
        
        # M4: A11(B10-B00)
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[2][3] = 1  # C00 += M4, C10 += M4
        
        # M5: (A00+A01)B11
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[1][4] = 1  # C00 -= M5, C01 += M5
        
        # M6: (A10-A00)(B00+B01)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1  # C11 += M6
        
        # M7: (A01-A11)(B10+B11)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1  # C00 += M7
        
        return U, V, W
    
    def strassen_2x2x2():
        """
        Tensor decomposition for multiplying three 2x2 matrices: (2x2)*(2x2)*(2x2) -> scalar
        Actually, we need a decomposition for the tensor T[i,j,k] where
        sum over middle index. Let me reconsider.
        
        For 2x2 * 2x2, we have n=2, m=2, p=2, schoolbook rank = 8, Strassen rank = 7.
        U shape (4,7), V shape (4,7), W shape (4,7)
        """
        return strassen_2x2()
    
    def try_strassen_blocks():
        """
        Exploit block structure: view the 3x6 matrix B as having 3 blocks of size 3x2.
        For each block b (b=0,1,2), B_b is 3x2.
        A is 2x3.
        
        We can decompose A into 2x2 and 2x1 blocks:
        A = [A_top | A_right] where A_top is 2x2 (columns 0,1) and A_right is 2x1 (column 2)
        
        For each block b, we compute A_top * B_b_top (2x2 * 2x2) using Strassen (rank 7)
        and A_right * B_b_bottom (2x1 * 1x2) using schoolbook (rank 2).
        
        Wait, B_b is 3x2. Let me partition B_b:
        B_b = [B_b_top; B_b_bottom] where B_b_top is 2x2 (rows 0,1) and B_b_bottom is 1x2 (row 2)
        
        So for block b:
        Result_b = A_top * B_b_top + A_right * B_b_bottom
        = (2x2)*(2x2) + (2x1)*(1x2)
        
        The (2x2)*(2x2) part has Strassen rank 7.
        The (2x1)*(1x2) part: A_right is 2x1, B_b_bottom is 1x2.
        This is just outer product scaled: A_right[i] * B_b_bottom[j] for i in {0,1}, j in {0,1}
        Rank 4 (schoolbook) or can we do better?
        Actually (2x1)*(1x2) is rank 4 in the tensor sense (n=2,m=1,p=2, schoolbook=4).
        
        Total per block: 7 + 4 = 11
        For 3 blocks: 3 * 11 = 33
        
        But we can share the A_top Strassen factors across blocks!
        The A_top part is the same for all blocks.
        
        Let me think more carefully. The full problem is:
        C = A * B where A is 2x3, B is 3x6.
        
        Partition A into A_left (2x2, cols 0-1) and A_right (2x1, col 2).
        Partition B into B_top (2x6, rows 0-1) and B_bottom (1x6, row 2).
        
        C = A_left * B_top + A_right * B_bottom
        
        Now B_top is 2x6. Partition B_top into 3 blocks of 2x2: B_top_0, B_top_1, B_top_2.
        B_bottom is 1x6. Partition into 3 blocks of 1x2: B_bottom_0, B_bottom_1, B_bottom_2.
        
        C[:, 2b:2b+2] = A_left * B_top_b + A_right * B_bottom_b
        
        For the A_left * B_top_b part (2x2 * 2x2), use Strassen (rank 7).
        For the A_right * B_bottom_b part (2x1 * 1x2), use schoolbook (rank 4).
        
        Key insight: The A_left factors in Strassen are shared across all 3 blocks!
        So we only need 7 rank terms for the A_left part total, not 3*7.
        
        Wait, that's not right either. In Strassen, the rank terms involve both A and B factors.
        The A factors are shared, but the B factors differ per block.
        
        Let me reconsider. In a tensor decomposition, each rank term c has:
        U[i,j,c] for A's (i,j)
        V[j,k,c] for B's (j,k)  
        W[k,i,c] for C's (k,i)
        
        For the Strassen part on block b:
        U[i,j,c] is the same for all b (depends only on A's structure)
        V[j, 2b+k', c] depends on b and k'
        W[2b+k', i, c] depends on b and k'
        
        So actually the rank terms are NOT shared - each block needs its own set of 7 terms
        because V and W have different k indices.
        
        Total rank = 3 * 7 (Strassen) + 3 * 4 (schoolbook for right part) = 21 + 12 = 33
        """
        if time.time() > deadline - 10:
            return None
        
        # Build the decomposition with rank 33
        rank = 33
        U = [[0] * rank for _ in range(n * m)]  # (6, 33)
        V = [[0] * rank for _ in range(m * p)]  # (18, 33)
        W = [[0] * rank for _ in range(p * n)]  # (12, 33)
        
        # Get Strassen factors for 2x2 * 2x2
        U_s, V_s, W_s = strassen_2x2()
        
        # Strassen part: 3 blocks, 7 terms each = 21 terms
        for b in range(3):
            offset_s = b * 7
            for c in range(7):
                # U: A_left[i, j] for i in {0,1}, j in {0,1}
                for i in range(2):
                    for j in range(2):
                        U[i*m + j][offset_s + c] = U_s[i*2 + j][c]
                
                # V: B_top[j, 2b+k'] for j in {0,1}, k' in {0,1}
                for j in range(2):
                    for kp in range(2):
                        k = 2*b + kp
                        V[j*p + k][offset_s + c] = V_s[j*2 + kp][c]
                
                # W: C[2b+k', i] for k' in {0,1}, i in {0,1}
                for kp in range(2):
                    k = 2*b + kp
                    for i in range(2):
                        W[k*n + i][offset_s + c] = W_s[kp*2 + i][c]
        
        # Schoolbook part for A_right * B_bottom: 3 blocks, 4 terms each = 12 terms
        # A_right is column 2 of A (2x1), B_bottom is row 2 of B (1x6)
        # For block b: A_right[i] * B_bottom[2b+k'] for i in {0,1}, k' in {0,1}
        for b in range(3):
            offset_sb = 21 + b * 4
            for c in range(4):
                i = c // 2
                kp = c % 2
                k = 2*b + kp
                U[i*m + 2][offset_sb + c] = 1  # A[i, 2]
                V[2*p + k][offset_sb + c] = 1  # B[2, k]
                W[k*n + i][offset_sb + c] = 1  # C[k, i]
        
        if verify_decomposition(U, V, W, rank):
            return U, V, W, rank
        
        return None
    
    def try_kronecker_structure(rank):
        """
        Try to construct a decomposition using Kronecker product structure.
        """
        if time.time() > deadline - 10:
            return None
        
        for sub_rank in [8, 9, 10, 11, 12]:
            if time.time() > deadline - 10:
                break
            
            total_rank = 3 * sub_rank
            
            if total_rank >= schoolbook_rank:
                continue
            
            result = _find_2x3x3x2_decomposition(sub_rank, deadline)
            if result is not None:
                U_sub, V_sub, W_sub = result
                U_full = [[0] * total_rank for _ in range(n * m)]
                V_full = [[0] * total_rank for _ in range(m * p)]
                W_full = [[0] * total_rank for _ in range(p * n)]
                
                for sub_idx in range(3):
                    offset = sub_idx * sub_rank
                    for i in range(n):
                        for j in range(m):
                            for c in range(sub_rank):
                                U_full[i*m+j][offset+c] = U_sub[i*m+j][c]
                    
                    for j in range(m):
                        for kb in range(2):
                            k = sub_idx * 2 + kb
                            for c in range(sub_rank):
                                V_full[j*p+k][offset+c] = V_sub[j*2+kb][c]
                    
                    for kb in range(2):
                        k = sub_idx * 2 + kb
                        for i in range(n):
                            for c in range(sub_rank):
                                W_full[k*n+i][offset+c] = W_sub[kb*n+i][c]
                
                if verify_decomposition(U_full, V_full, W_full, total_rank):
                    return U_full, V_full, W_full, total_rank
        
        return None
    
    def _find_2x3x3x2_decomposition(rank, deadline):
        """Find rank decomposition for 2x3 x 3x2 matrix multiplication."""
        n2, m2, p2 = 2, 3, 2
        schoolbook = n2 * m2 * p2  # 12
        
        def verify_sub(U, V, W, r):
            for i in range(n2):
                for j in range(m2):
                    for k in range(p2):
                        s = 0.0
                        for c in range(r):
                            s += U[i*m2+j][c] * V[j*p2+k][c] * W[k*n2+i][c]
                        if abs(s - 1.0) > 1e-9:
                            return False
            return True
        
        for trial in range(5):
            if time.time() > deadline - 5:
                return None
            
            np.random.seed(1000 + trial * 11)
            
            W = np.random.choice([-1.0, 0.0, 1.0], size=(p2*n2, rank))
            for i in range(p2*n2):
                if all(W[i] == 0):
                    W[i][trial % rank] = 1.0
            
            U = np.random.randn(n2*m2, rank) * 0.1
            V = np.random.randn(m2*p2, rank) * 0.1
            
            for iteration in range(150):
                if time.time() > deadline - 5:
                    return None
                
                A_v = np.zeros((n2*m2*p2, m2*p2*rank))
                b_v = np.ones(n2*m2*p2)
                eq_idx = 0
                for i in range(n2):
                    for j in range(m2):
                        for k in range(p2):
                            for c in range(rank):
                                w_val = W[k*n2+i][c]
                                if abs(w_val) > 1e-12:
                                    v_idx = (j*p2 + k) * rank + c
                                    A_v[eq_idx, v_idx] = U[i*m2+j][c] * w_val
                            eq_idx += 1
                V_sol, _, _, _ = np.linalg.lstsq(A_v, b_v, rcond=None)
                V = V_sol.reshape(m2*p2, rank)
                
                A_u = np.zeros((n2*m2*p2, n2*m2*rank))
                b_u = np.ones(n2*m2*p2)
                eq_idx = 0
                for i in range(n2):
                    for j in range(m2):
                        for k in range(p2):
                            for c in range(rank):
                                w_val = W[k*n2+i][c]
                                if abs(w_val) > 1e-12:
                                    u_idx = (i*m2 + j) * rank + c
                                    A_u[eq_idx, u_idx] = V[j*p2+k][c] * w_val
                            eq_idx += 1
                U_sol, _, _, _ = np.linalg.lstsq(A_u, b_u, rcond=None)
                U = U_sol.reshape(n2*m2, rank)
                
                if iteration % 5 == 0 or iteration == 149:
                    max_err = 0.0
                    for i in range(n2):
                        for j in range(m2):
                            for k in range(p2):
                                s = 0.0
                                for c in range(rank):
                                    s += U[i*m2+j][c] * V[j*p2+k][c] * W[k*n2+i][c]
                                max_err = max(max_err, abs(s - 1.0))
                    
                    if max_err < 1e-10:
                        U_snap = np.round(U * 2) / 2
                        V_snap = np.round(V * 2) / 2
                        W_snap = np.round(W * 2) / 2
                        if verify_sub(U_snap.tolist(), V_snap.tolist(), W_snap.tolist(), rank):
                            return U_snap.tolist(), V_snap.tolist(), W_snap.tolist()
                        
                        U_int = np.round(U)
                        V_int = np.round(V)
                        W_int = np.round(W)
                        if verify_sub(U_int.tolist(), V_int.tolist(), W_int.tolist(), rank):
                            return U_int.tolist(), V_int.tolist(), W_int.tolist()
        
        return None
    
    def try_direct_als(rank, max_iters=200, trials=3):
        """Direct ALS for the full 2x3x3x6 problem."""
        for trial in range(trials):
            if time.time() > deadline - 5:
                return None
                
            np.random.seed(3000 + trial * 7)
            
            W = np.random.choice([-1.0, 0.0, 1.0], size=(p*n, rank))
            for i in range(p*n):
                if all(W[i] == 0):
                    W[i][trial % rank] = 1.0
            
            U = np.random.randn(n*m, rank) * 0.1
            V = np.random.randn(m*p, rank) * 0.1
            
            prev_error = float('inf')
            
            for iteration in range(max_iters):
                if time.time() > deadline - 5:
                    return None
                
                A_v = np.zeros((n*m*p, m*p*rank))
                b_v = np.ones(n*m*p)
                eq_idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            for c in range(rank):
                                w_val = W[k*n+i][c]
                                if abs(w_val) > 1e-12:
                                    v_idx = (j*p + k) * rank + c
                                    A_v[eq_idx, v_idx] = U[i*m+j][c] * w_val
                            eq_idx += 1
                V_sol, _, _, _ = np.linalg.lstsq(A_v, b_v, rcond=None)
                V = V_sol.reshape(m*p, rank)
                
                A_u = np.zeros((n*m*p, n*m*rank))
                b_u = np.ones(n*m*p)
                eq_idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            for c in range(rank):
                                w_val = W[k*n+i][c]
                                if abs(w_val) > 1e-12:
                                    u_idx = (i*m + j) * rank + c
                                    A_u[eq_idx, u_idx] = V[j*p+k][c] * w_val
                            eq_idx += 1
                U_sol, _, _, _ = np.linalg.lstsq(A_u, b_u, rcond=None)
                U = U_sol.reshape(n*m, rank)
                
                if iteration % 5 == 0 or iteration == max_iters - 1:
                    max_err = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                s = 0.0
                                for c in range(rank):
                                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                                max_err = max(max_err, abs(s - 1.0))
                    
                    if max_err < 1e-10:
                        U_snap = np.round(U * 2) / 2
                        V_snap = np.round(V * 2) / 2
                        W_snap = np.round(W * 2) / 2
                        if verify_decomposition(U_snap.tolist(), V_snap.tolist(), W_snap.tolist(), rank):
                            return U_snap.tolist(), V_snap.tolist(), W_snap.tolist()
                        
                        U_int = np.round(U)
                        V_int = np.round(V)
                        W_int = np.round(W)
                        if verify_decomposition(U_int.tolist(), V_int.tolist(), W_int.tolist(), rank):
                            return U_int.tolist(), V_int.tolist(), W_int.tolist()
                    
                    if max_err > prev_error * 1.01 and iteration > 50:
                        break
                    prev_error = max_err
                    
        return None
    
    # First try the Strassen block approach (rank 33)
    strassen_result = try_strassen_blocks()
    if strassen_result is not None:
        U, V, W, rank = strassen_result
        return {"U": U, "V": V, "W": W}
    
    # Try the Kronecker structure approach
    kronecker_result = try_kronecker_structure(24)
    if kronecker_result is not None:
        U, V, W, rank = kronecker_result
        return {"U": U, "V": V, "W": W}
    
    # Try direct ALS for various ranks
    for rank in range(18, 36):
        if time.time() > deadline - 15:
            break
        
        res = try_direct_als(rank, max_iters=150, trials=2)
        if res:
            U, V, W = res
            return {"U": U, "V": V, "W": W}
            
    # Fallback to schoolbook
    U_fb, V_fb, W_fb, rank_fb = schoolbook()
    return {"U": U_fb, "V": V_fb, "W": W_fb}
# EVOLVE-BLOCK-END
