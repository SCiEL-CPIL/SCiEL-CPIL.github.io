"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import sys
from itertools import product

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 145  # Leave 5s margin before 150s kill
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 72
    NM, MP, PN = n*m, m*p, p*n
    
    # Strassen decomposition for 2x2x2: rank 7
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # For 2x2x2, Strassen gives rank 7 decomposition
    # Using the standard Strassen formulas for matrix multiplication
    # C = A*B where A is 2x2, B is 2x2
    # M1 = (A11-A22)*(B21-B22)
    # M2 = (A11+A22)*B22
    # M3 = A11*(B11-B22)
    # M4 = A22*(B11-B12)
    # M5 = (A11+A12)*B11
    # M6 = (A21-A22)*(B12+B22)
    # M7 = (A11-A12)*(B11+B21)
    # C11 = M1+M4+M5-M7
    # C12 = M3+M5
    # C21 = M1+M2
    # C22 = M2-M4+M6+M7
    # 
    # For tensor decomposition T[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]:
    # U[i,j,c] corresponds to A[i,j] coefficient in M_c
    # V[j,k,c] corresponds to B[j,k] coefficient in M_c
    # W[k,i,c] corresponds to C[k,i]... wait, let me re-derive
    #
    # Actually for matrix mult: (AB)[i,k] = sum_j A[i,j] B[j,k]
    # Tensor: T[i,j,k] = 1 if we're computing (AB)[i,k] = sum_j A[i,j]*B[j,k]
    # The decomposition is: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # where W[k,i,c] provides the coefficient for C[i,k] from M_c
    #
    # Let me index properly. In Strassen:
    # A = [[A11, A12],[A21, A22]], B = [[B11, B12],[B21, B22]]
    # C = [[C11, C12],[C21, C22]]
    # 
    # C11 = M1 + M4 + M5 - M7
    # C12 = M3 + M5
    # C21 = M1 + M2
    # C22 = M2 - M4 + M6 + M7
    #
    # M1 = (A11 - A22)*(B21 - B22)
    # M2 = (A11 + A22)*B22
    # M3 = A11*(B11 - B22)
    # M4 = A22*(B11 - B12)
    # M5 = (A11 + A12)*B11
    # M6 = (A21 - A22)*(B12 + B22)
    # M7 = (A11 - A12)*(B11 + B21)
    #
    # For the tensor T[i,j,k] where i=row of A, j=col of A/row of B, k=col of B:
    # U[i,j,c] = coefficient of A[i,j] in M_c
    # V[j,k,c] = coefficient of B[j,k] in M_c  
    # W[k,i,c] = coefficient that contributes to C[i,k] from M_c
    #
    # Wait, the convention is W[k*n+i, r] for the output factor.
    # Let me re-read: "The output factor follows the official trace convention W[k*n+i,r]"
    # So W has shape (p*n, rank) and W[k*n+i, c] is the (k,i)-th entry.
    # In our tensor T[i,j,k], the "output" index pair is (i,k) where i is row of result, k is col.
    # So W[k*n + i, c] gives the weight for result position (i,k) from term c.
    #
    # Let me define for 2x2x2 (indices 0,1):
    # A indices: (0,0)=A11, (0,1)=A12, (1,0)=A21, (1,1)=A22
    # B indices: (0,0)=B11, (0,1)=B12, (1,0)=B21, (1,1)=B22
    # C indices: (0,0)=C11, (0,1)=C12, (1,0)=C21, (1,1)=C22
    
    def strassen_2x2x2():
        """Return U(2,2,7), V(2,2,7), W(2,2,7) for 2x2x2 matrix mult tensor."""
        U = np.zeros((2, 2, 7))
        V = np.zeros((2, 2, 7))
        W = np.zeros((2, 2, 7))  # W[k,i,c] = coefficient for C[i,k]
        
        # M1 = (A11 - A22)*(B21 - B22)
        # A: A11=(0,0), A22=(1,1)
        # B: B21=(1,0), B22=(1,1)
        # C: contributes to C11 with +1
        U[0,0,0] = 1; U[1,1,0] = -1
        V[1,0,0] = 1; V[1,1,0] = -1
        W[0,0,0] = 1  # C[0,0] += M1
        
        # M2 = (A11 + A22)*B22
        # A: A11=(0,0), A22=(1,1)
        # B: B22=(1,1)
        # C: contributes to C21 with +1, C22 with -1
        U[0,0,1] = 1; U[1,1,1] = 1
        V[1,1,1] = 1
        W[0,1,1] = 1  # C[1,0] += M2  (i=1, k=0)
        W[1,1,1] = -1 # C[1,1] -= M2  (i=1, k=1)
        
        # M3 = A11*(B11 - B22)
        # A: A11=(0,0)
        # B: B11=(0,0), B22=(1,1)
        # C: contributes to C12 with +1
        U[0,0,2] = 1
        V[0,0,2] = 1; V[1,1,2] = -1
        W[1,0,2] = 1  # C[0,1] += M3  (i=0, k=1)
        
        # M4 = A22*(B11 - B12)
        # A: A22=(1,1)
        # B: B11=(0,0), B12=(0,1)
        # C: contributes to C11 with +1, C22 with -1
        U[1,1,3] = 1
        V[0,0,3] = 1; V[0,1,3] = -1
        W[0,0,3] = 1  # C[0,0] += M4
        W[1,1,3] = -1 # C[1,1] -= M4
        
        # M5 = (A11 + A12)*B11
        # A: A11=(0,0), A12=(0,1)
        # B: B11=(0,0)
        # C: contributes to C11 with +1, C12 with +1
        U[0,0,4] = 1; U[0,1,4] = 1
        V[0,0,4] = 1
        W[0,0,4] = 1  # C[0,0] += M5
        W[1,0,4] = 1  # C[0,1] += M5
        
        # M6 = (A21 - A22)*(B12 + B22)
        # A: A21=(1,0), A22=(1,1)
        # B: B12=(0,1), B22=(1,1)
        # C: contributes to C22 with +1
        U[1,0,5] = 1; U[1,1,5] = -1
        V[0,1,5] = 1; V[1,1,5] = 1
        W[1,1,5] = 1  # C[1,1] += M6
        
        # M7 = (A11 - A12)*(B11 + B21)
        # A: A11=(0,0), A12=(0,1)
        # B: B11=(0,0), B21=(1,0)
        # C: contributes to C11 with -1
        U[0,0,6] = 1; U[0,1,6] = -1
        V[0,0,6] = 1; V[1,0,6] = 1
        W[0,0,6] = -1 # C[0,0] -= M7
        
        return U, V, W
    
    def verify_strassen():
        """Verify the Strassen 2x2x2 decomposition is correct."""
        Us, Vs, Ws = strassen_2x2x2()
        T_rec = np.einsum('ijc,jkc,kic->ijk', Us, Vs, Ws)
        T_true = np.ones((2, 2, 2))
        return np.allclose(T_rec, T_true, atol=1e-10)
    
    def build_strassen_decomposition():
        """
        Build full 3x4x6 decomposition using Strassen on p=2 blocks.
        
        Split the 4x6 matrix B into three 4x2 blocks: B[:, 0:2], B[:, 2:4], B[:, 4:6]
        Split the result C (3x6) into three 3x2 blocks: C[:, 0:2], C[:, 2:4], C[:, 4:6]
        
        For each 2-column block, we need to compute A(3x4) * B_block(4x2).
        We can split A into 2x2 blocks of rows: A_top(2x4), A_bottom(1x4)
        And B_block into 2x2 blocks of rows: B_top(2x2), B_bottom(2x2)
        
        Actually, let's think differently. We have A(3x4) * B(4x6).
        Split the 4 middle dimension into 2+2: A = [A1; A2] where A1(3x2), A2(3x2)
        B = [B1; B2] where B1(2x6), B2(2x6)
        C = A1*B1 + A2*B2
        
        Now for each term A1*B1 (3x2 * 2x6), we can further split.
        Actually, let's use the p=2 block approach from the finding.
        
        The finding says: "Splitting p=6 into three p=2 blocks allows using Strassen (rank 7) for each 2x2x2 subproblem"
        
        So we split the output dimension p=6 into three groups of 2.
        For each group of 2 output columns, we compute a 3x2 sub-result.
        
        For a 3x4 * 4x2 multiplication:
        Split the 4 middle dimension: A(3x4) = [A_top(3x2), A_bot(3x2)], B(4x2) = [B_top(2x2), B_bot(2x2)]
        C = A_top * B_top + A_bot * B_bot
        
        Now A_top(3x2) * B_top(2x2): split the 3 rows into 2+1.
        A_top = [A1(2x2); A2(1x2)], B_top(2x2)
        A1*B_top is 2x2*2x2 -> use Strassen (rank 7)
        A2*B_top is 1x2*2x2 -> rank 2 (schoolbook)
        
        Similarly A_bot(3x2) * B_bot(2x2):
        A_bot = [A3(2x2); A4(1x2)], B_bot(2x2)
        A3*B_bot -> Strassen (rank 7)
        A4*B_bot -> rank 2
        
        Total per p=2 block: 7 + 2 + 7 + 2 = 18
        Three blocks: 18 * 3 = 54? That seems too optimistic.
        
        Wait, let me reconsider. The finding says rank goes from 68 to 66.
        Let me think about what structure gives 66.
        
        Actually, re-reading: "Splitting p=6 into three p=2 blocks allows using Strassen (rank 7) for each 2x2x2 subproblem instead of the less efficient M(2,2,3) (rank 11) approach, reducing total rank from 68 to 66"
        
        So the baseline was using some M(2,2,3) decomposition of rank 11, and switching to three independent 2x2x2 Strassen decompositions of rank 7 each saves 3*(11-7) = 12, going from 68+12=80... no that doesn't work either.
        
        Let me think about this differently. Perhaps the approach is:
        - Split n=3 into 2+1
        - Split m=4 into 2+2  
        - Split p=6 into 2+2+2
        
        For each (n_block, p_block) pair, we have a 2x2x2 or 1x2x2 subproblem.
        
        Actually, I think the simplest correct approach is:
        
        For the full 3x4x6 problem, split p into three 2s.
        For each 2-column output block, compute 3x4 * 4x2.
        
        For 3x4 * 4x2:
        Split m=4 into 2+2: A(3x4)=[A1(3x2), A2(3x2)], B(4x2)=[B1(2x2), B2(2x2)]
        C = A1*B1 + A2*B2
        
        For A1(3x2)*B1(2x2): split n=3 into 2+1
        A1 = [A1a(2x2); A1b(1x2)]
        A1a*B1: 2x2x2 -> Strassen rank 7
        A1b*B1: 1x2x2 -> schoolbook rank 2
        Subtotal: 9
        
        For A2(3x2)*B2(2x2): similarly
        A2 = [A2a(2x2); A2b(1x2)]
        A2a*B2: 2x2x2 -> Strassen rank 7
        A2b*B2: 1x2x2 -> schoolbook rank 2
        Subtotal: 9
        
        Total per p=2 block: 18
        Three p=2 blocks: 54
        
        Hmm, that gives 54 which is better than 66. Let me just implement this and verify.
        
        Actually wait - I need to be more careful. Let me just build it systematically.
        """
        n, m, p = 3, 4, 6
        NM, MP, PN = n*m, m*p, p*n
        
        U = np.zeros((n, m, 54))
        V = np.zeros((m, p, 54))
        W = np.zeros((p, n, 54))
        
        col_offset = 0
        
        # Split p=6 into three blocks of 2: [0,1], [2,3], [4,5]
        p_blocks = [(0, 2), (2, 4), (4, 6)]
        
        # Split m=4 into two blocks of 2: [0,1], [2,3]
        m_blocks = [(0, 2), (2, 4)]
        
        # Split n=3 into (2, 1): [0,1], [2]
        n_blocks = [(0, 2), (2, 3)]
        
        for p_start, p_end in p_blocks:
            for m_start, m_end in m_blocks:
                for n_start, n_end in n_blocks:
                    n_sub = n_end - n_start
                    m_sub = m_end - m_start
                    p_sub = p_end - p_start  # always 2
                    
                    if n_sub == 2 and m_sub == 2 and p_sub == 2:
                        # 2x2x2: use Strassen
                        Us, Vs, Ws = strassen_2x2x2()
                        for i in range(2):
                            for j in range(2):
                                for c in range(7):
                                    U[n_start+i, m_start+j, col_offset+c] = Us[i, j, c]
                        for j in range(2):
                            for k in range(2):
                                for c in range(7):
                                    V[m_start+j, p_start+k, col_offset+c] = Vs[j, k, c]
                        for k in range(2):
                            for i in range(2):
                                for c in range(7):
                                    W[p_start+k, n_start+i, col_offset+c] = Ws[k, i, c]
                        col_offset += 7
                    elif n_sub == 1 and m_sub == 2 and p_sub == 2:
                        # 1x2x2: schoolbook, rank 2
                        for j in range(2):
                            for k in range(2):
                                U[n_start, m_start+j, col_offset] = 1
                                V[m_start+j, p_start+k, col_offset] = 1
                                W[p_start+k, n_start, col_offset] = 1
                                col_offset += 1
                        # This gives rank 4 for 1x2x2, but we only need rank 2?
                        # No, 1x2x2 tensor has 4 nonzero entries, minimum rank is 4? 
                        # Actually T[i,j,k] for i in {0}, j in {0,1}, k in {0,1}
                        # T[0,j,k] = 1 for all j,k. This is outer product of (1,1,1) with... 
                        # No, it's a 1x2x2 tensor with all ones. 
                        # Rank = 4? Let's check: can we do it in fewer?
                        # T[0,0,0]=1, T[0,0,1]=1, T[0,1,0]=1, T[0,1,1]=1
                        # This is the all-ones 1x2x2 tensor. Its rank is 4 (it's a rank-4 tensor).
                        # Wait no. A 1x2x2 tensor with all ones: 
                        # T[i,j,k] = 1. We need sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1 for all valid (i,j,k).
                        # Since i is fixed at 0, we need sum_c V[j,k,c]*W[k,0,c] = 1 for all j,k.
                        # U[0,j,c] can be anything, let's set U[0,j,c] = 1.
                        # Then we need sum_c V[j,k,c]*W[k,0,c] = 1 for all j,k.
                        # This is like decomposing the 2x2 all-ones matrix as sum of rank-1 matrices.
                        # All-ones 2x2 matrix has rank 1, so we can do it in 1 term!
                        # V[0,0,0]=1, V[0,1,0]=1, V[1,0,0]=1, V[1,1,0]=1, W[0,0,0]=1, W[1,0,0]=1
                        # Then sum_c V[j,k,c]*W[k,0,c] = V[j,k,0]*W[k,0,0] = 1*1 = 1. Yes!
                        # So rank 1 suffices for 1x2x2 all-ones tensor.
                        # Let me redo:
                        pass  # We'll handle this below
                    else:
                        # Should not happen with our splits
                        pass
        
        # Let me redo this more carefully
        U = np.zeros((n, m, 54))
        V = np.zeros((m, p, 54))
        W = np.zeros((p, n, 54))
        col_offset = 0
        
        for p_start, p_end in p_blocks:
            for m_start, m_end in m_blocks:
                for n_start, n_end in n_blocks:
                    n_sub = n_end - n_start
                    m_sub = m_end - m_start
                    p_sub = p_end - p_start
                    
                    if n_sub == 2 and m_sub == 2 and p_sub == 2:
                        # 2x2x2: Strassen, rank 7
                        Us, Vs, Ws = strassen_2x2x2()
                        for i in range(2):
                            for j in range(2):
                                for c in range(7):
                                    U[n_start+i, m_start+j, col_offset+c] = Us[i, j, c]
                        for j in range(2):
                            for k in range(2):
                                for c in range(7):
                                    V[m_start+j, p_start+k, col_offset+c] = Vs[j, k, c]
                        for k in range(2):
                            for i in range(2):
                                for c in range(7):
                                    W[p_start+k, n_start+i, col_offset+c] = Ws[k, i, c]
                        col_offset += 7
                    elif n_sub == 1 and m_sub == 2 and p_sub == 2:
                        # 1x2x2 all-ones tensor: rank 1
                        # U[0,j,c]=1, V[j,k,c]=1, W[k,0,c]=1
                        # With one term c=0:
                        # U[n_start, m_start+0, col_offset] = 1
                        # U[n_start, m_start+1, col_offset] = 1
                        # V[m_start+0, p_start+0, col_offset] = 1
                        # V[m_start+0, p_start+1, col_offset] = 1
                        # V[m_start+1, p_start+0, col_offset] = 1
                        # V[m_start+1, p_start+1, col_offset] = 1
                        # W[p_start+0, n_start, col_offset] = 1
                        # W[p_start+1, n_start, col_offset] = 1
                        for j in range(2):
                            U[n_start, m_start+j, col_offset] = 1
                        for j in range(2):
                            for k in range(2):
                                V[m_start+j, p_start+k, col_offset] = 1
                        for k in range(2):
                            W[p_start+k, n_start, col_offset] = 1
                        col_offset += 1
                    else:
                        # Fallback: schoolbook
                        for i in range(n_sub):
                            for j in range(m_sub):
                                for k in range(p_sub):
                                    U[n_start+i, m_start+j, col_offset] = 1
                                    V[m_start+j, p_start+k, col_offset] = 1
                                    W[p_start+k, n_start+i, col_offset] = 1
                                    col_offset += 1
        
        # col_offset should be 54
        # Let's verify
        actual_rank = col_offset
        U = U[:, :, :actual_rank]
        V = V[:, :, :actual_rank]
        W = W[:, :, :actual_rank]
        
        return U, V, W, actual_rank
    
    # Build target tensor T[i,j,k] = 1 for all valid indices
    T = np.ones((n, m, p))
    T_flat = T.flatten()
    
    def check_exact(U, V, W, rank):
        """Check if U, V, W with given rank exactly reconstruct T"""
        U_r = np.array(U, dtype=np.float64).reshape(n, m, rank)
        V_r = np.array(V, dtype=np.float64).reshape(m, p, rank)
        W_r = np.array(W, dtype=np.float64).reshape(p, n, rank)
        T_rec = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        return np.allclose(T_rec.flatten(), T_flat, atol=1e-10)
    
    # First, try the Strassen-based structured decomposition
    try:
        U_str, V_str, W_str, str_rank = build_strassen_decomposition()
        T_rec = np.einsum('ijc,jkc,kic->ijk', U_str, V_str, W_str)
        if np.allclose(T_rec.flatten(), T_flat, atol=1e-10):
            # We have a valid decomposition of rank str_rank
            # Try to reduce further with ALS
            best_U, best_V, best_W, best_rank = U_str, V_str, W_str, str_rank
            
            # Try to reduce rank with ALS
            for target_rank in range(max(1, str_rank - 10), str_rank):
                if time.time() > deadline - 30:
                    break
                result = als_decompose(target_rank, max_iters=100, tol=1e-14, seed=0, 
                                       init=(U_str[:, :, :target_rank].copy(), 
                                             V_str[:, :, :target_rank].copy(), 
                                             W_str[:, :, :target_rank].copy()))
                if result is not None:
                    best_U, best_V, best_W = [np.array(x) for x in result]
                    best_rank = target_rank
                    break
                # Try with random init
                for seed in range(5):
                    if time.time() > deadline - 30:
                        break
                    result = als_decompose(target_rank, max_iters=80, tol=1e-14, seed=seed)
                    if result is not None:
                        best_U, best_V, best_W = [np.array(x) for x in result]
                        best_rank = target_rank
                        break
                    if result is not None:
                        break
            
            return {"U": best_U.reshape(NM, best_rank).tolist(),
                    "V": best_V.reshape(MP, best_rank).tolist(),
                    "W": best_W.reshape(PN, best_rank).tolist()}
    except Exception as e:
        pass
    
    def schoolbook_init():
        """Create the schoolbook (trivial) decomposition"""
        U = np.zeros((n, m, rank_schoolbook))
        V = np.zeros((m, p, rank_schoolbook))
        W = np.zeros((p, n, rank_schoolbook))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i, j, col] = 1
                    V[j, k, col] = 1
                    W[k, i, col] = 1
                    col += 1
        return U, V, W
    
    def als_decompose(rank, max_iters=100, tol=1e-14, seed=0, init=None):
        """Alternating Least Squares for 3rd order tensor decomposition"""
        np.random.seed(seed)
        
        if init is not None:
            U, V, W = [x.copy() for x in init]
        else:
            U = np.random.randn(n, m, rank) * 0.5
            V = np.random.randn(m, p, rank) * 0.5
            W = np.random.randn(p, n, rank) * 0.5
        
        prev_err = np.inf
        for it in range(max_iters):
            if time.time() > deadline - 5:
                return None
            
            # Solve for U: fix V, W
            for i in range(n):
                for j in range(m):
                    A = np.einsum('kc,kc->ck', V[j], W[:, i])
                    b = T[i, j, :]
                    try:
                        u_sol, _, _, _ = np.linalg.lstsq(A.T, b, rcond=None)
                        U[i, j, :] = u_sol
                    except:
                        pass
            
            # Solve for V: fix U, W
            for j in range(m):
                for k in range(p):
                    A = np.einsum('ic,ic->ci', U[:, j, :], W[k, :, :])
                    b = T[:, j, k]
                    try:
                        v_sol, _, _, _ = np.linalg.lstsq(A.T, b, rcond=None)
                        V[j, k, :] = v_sol
                    except:
                        pass
            
            # Solve for W: fix U, V
            for k in range(p):
                for i in range(n):
                    A = np.einsum('jc,jc->cj', U[i, :, :], V[:, k, :])
                    b = T[i, :, k]
                    try:
                        w_sol, _, _, _ = np.linalg.lstsq(A.T, b, rcond=None)
                        W[k, i, :] = w_sol
                    except:
                        pass
            
            # Check convergence
            T_rec = np.einsum('ijc,jkc,kic->ijk', U, V, W)
            err = np.sum((T_rec - T)**2)
            if err < tol:
                break
            if abs(prev_err - err) < tol * 0.1:
                break
            prev_err = err
        
        # Final check
        T_rec = np.einsum('ijc,jkc,kic->ijk', U, V, W)
        final_err = np.sum((T_rec - T)**2)
        
        if final_err < 1e-6:
            # Try rounding to various grids
            for scale in [1, 2, 4, 8, 16, 32, 64]:
                U_rounded = np.round(U * scale) / scale
                V_rounded = np.round(V * scale) / scale
                W_rounded = np.round(W * scale) / scale
                if check_exact(U_rounded, V_rounded, W_rounded, rank):
                    return (U_rounded.reshape(NM, rank).tolist(), 
                           V_rounded.reshape(MP, rank).tolist(), 
                           W_rounded.reshape(PN, rank).tolist())
            
            # Try integer rounding
            U_int = np.round(U)
            V_int = np.round(V)
            W_int = np.round(W)
            if check_exact(U_int, V_int, W_int, rank):
                return (U_int.reshape(NM, rank).tolist(), 
                       V_int.reshape(MP, rank).tolist(), 
                       W_int.reshape(PN, rank).tolist())
        
        return None
    
    def try_rank_als(rank, num_restarts=15):
        """Try to find decomposition of given rank using ALS"""
        for seed in range(num_restarts):
            if time.time() > deadline - 10:
                return None
            result = als_decompose(rank, max_iters=80, tol=1e-14, seed=seed)
            if result is not None:
                return result
        
        # Try with schoolbook-inspired initialization
        U_sb, V_sb, W_sb = schoolbook_init()
        for seed in range(10):
            if time.time() > deadline - 10:
                return None
            np.random.seed(seed + 1000)
            selected = np.random.choice(rank_schoolbook, size=min(rank, rank_schoolbook), replace=False)
            U_init = U_sb[:, :, selected]
            V_init = V_sb[:, :, selected]
            W_init = W_sb[:, :, selected]
            
            result = als_decompose(rank, max_iters=80, tol=1e-14, seed=seed, init=(U_init, V_init, W_init))
            if result is not None:
                return result
        
        return None
    
    def try_rank_als(rank, num_restarts=15):
        """Try to find decomposition of given rank using ALS"""
        for seed in range(num_restarts):
            if time.time() > deadline - 10:
                return None
            result = als_decompose(rank, max_iters=80, tol=1e-14, seed=seed)
            if result is not None:
                return result
        
        # Try with schoolbook-inspired initialization
        U_sb, V_sb, W_sb = schoolbook_init()
        for seed in range(10):
            if time.time() > deadline - 10:
                return None
            np.random.seed(seed + 1000)
            selected = np.random.choice(rank_schoolbook, size=min(rank, rank_schoolbook), replace=False)
            U_init = U_sb[:, :, selected]
            V_init = V_sb[:, :, selected]
            W_init = W_sb[:, :, selected]
            
            result = als_decompose(rank, max_iters=80, tol=1e-14, seed=seed, init=(U_init, V_init, W_init))
            if result is not None:
                return result
        
        return None
    
    # (Removed unused helper functions to reduce code size)
    
    # Strategy:
    # 1. Use Strassen-based structured decomposition (rank 54)
    # 2. Try to reduce further with ALS
    # 3. Fall back to schoolbook if nothing works
    
    best_result = None
    best_rank = rank_schoolbook
    
    # Phase 1: Strassen-based decomposition
    try:
        U_str, V_str, W_str, str_rank = build_strassen_decomposition()
        T_rec = np.einsum('ijc,jkc,kic->ijk', U_str, V_str, W_str)
        if np.allclose(T_rec.flatten(), T_flat, atol=1e-10):
            best_U, best_V, best_W, best_rank = U_str, V_str, W_str, str_rank
            
            # Try to reduce rank with ALS from the Strassen init
            for target_rank in range(max(1, str_rank - 15), str_rank):
                if time.time() > deadline - 30:
                    break
                
                # Try with truncated Strassen init
                if target_rank <= str_rank:
                    result = als_decompose(target_rank, max_iters=100, tol=1e-14, seed=0,
                                           init=(U_str[:, :, :target_rank].copy(),
                                                 V_str[:, :, :target_rank].copy(),
                                                 W_str[:, :, :target_rank].copy()))
                    if result is not None:
                        best_U = np.array(result[0]).reshape(n, m, target_rank)
                        best_V = np.array(result[1]).reshape(m, p, target_rank)
                        best_W = np.array(result[2]).reshape(p, n, target_rank)
                        best_rank = target_rank
                        break
                
                # Try random init
                for seed in range(5):
                    if time.time() > deadline - 30:
                        break
                    result = als_decompose(target_rank, max_iters=80, tol=1e-14, seed=seed)
                    if result is not None:
                        best_U = np.array(result[0]).reshape(n, m, target_rank)
                        best_V = np.array(result[1]).reshape(m, p, target_rank)
                        best_W = np.array(result[2]).reshape(p, n, target_rank)
                        best_rank = target_rank
                        break
                    if result is not None:
                        break
            
            best_result = {"U": best_U.reshape(NM, best_rank).tolist(),
                          "V": best_V.reshape(MP, best_rank).tolist(),
                          "W": best_W.reshape(PN, best_rank).tolist()}
    except Exception as e:
        pass
    
    # Phase 2: If Strassen didn't work, try ALS search
    if best_result is None:
        for rank in [50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70]:
            if time.time() > deadline - 20:
                break
            
            result = try_rank_als(rank, num_restarts=12)
            if result is not None:
                best_result = {"U": result[0], "V": result[1], "W": result[2]}
                best_rank = rank
                break
    
    # Phase 3: Fall back to schoolbook
    if best_result is None:
        U = [[0] * rank_schoolbook for _ in range(NM)]
        V = [[0] * rank_schoolbook for _ in range(MP)]
        W = [[0] * rank_schoolbook for _ in range(PN)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        best_result = {"U": U, "V": V, "W": W}
        best_rank = rank_schoolbook
    
    return best_result
# EVOLVE-BLOCK-END
