import time
import numpy as np

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120
    
    # Try 64-rank solution: M(3,4,6) = M(3,3,3)+M(3,3,3)+M(3,1,3)+M(3,1,3)
    # with m split 3+1, p split 3+3
    # M(3,3,3) uses 23-term decomposition, M(3,1,3) uses schoolbook (9 terms)
    
    try:
        result = _construct_64()
        # Validate
        U, V, W = result['U'], result['V'], result['W']
        rank = len(U[0])
        # Check shapes
        assert len(U) == n*m
        assert len(V) == m*p
        assert len(W) == p*n
        assert all(len(row) == rank for row in U)
        assert all(len(row) == rank for row in V)
        assert all(len(row) == rank for row in W)
        # Validate: sum_r U[i,j,r] * V[j,k,r] == delta for C[i,k]
        # C[i,k] = sum_j A[i,j]*B[j,k], so coefficient of A[i,j]*B[j,k] in C[i,k] is 1
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should be 1 if k*n+i is the output index
        # Actually: sum_r U[i*m+j, r] * V[j*p+k, r] = W_coeff for C[i,k]
        # The W matrix gives the coefficient for each (i,k) pair
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
        # Wait, that's not right. The decomposition is:
        # C[i,k] = sum_j A[i,j] * B[j,k] = sum_r (sum_j U[i*m+j,r]*A[i,j]) * (sum_j V[j*p+k,r]*B[j,k]) * W[k*n+i,r]
        # No, that's not right either. The correct form is:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
        # Let me verify with numpy
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        # For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should be 1
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = sum(U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] for r in range(rank))
                    if abs(s - 1.0) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        if valid:
            return result
    except Exception:
        pass
    
    # Fallback: 66-rank solution
    return _construct_66()

def _construct_64():
    n, m, p = DIMENSIONS
    # m split: 3+1, p split: 3+3
    # Blocks:
    # (m=3, p=3): M(3,3,3) at m_off=0, p_off=0
    # (m=3, p=3): M(3,3,3) at m_off=0, p_off=3
    # (m=1, p=3): M(3,1,3) at m_off=3, p_off=0
    # (m=1, p=3): M(3,1,3) at m_off=3, p_off=3
    
    rank = 23 + 23 + 9 + 9  # 64
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    current_rank = 0
    
    def add_rank1(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals):
        nonlocal current_rank
        for idx, val in zip(u_indices, u_vals):
            U[idx][current_rank] += val
        for idx, val in zip(v_indices, v_vals):
            V[idx][current_rank] += val
        for idx, val in zip(w_indices, w_vals):
            W[idx][current_rank] += val
        current_rank += 1
    
    def add_3_3_3(i_off, j_off, k_off):
        # 23-term decomposition of M(3,3,3)
        # This is the complex part. I'll use a known formulation.
        # Due to complexity, I'll use a placeholder that may not be correct.
        # Actually, let me use schoolbook (27 terms) as a safe fallback within this function
        # No, that would give 27+27+9+9=72 which is worse.
        # I need to implement the 23-term decomposition.
        # For now, I'll mark this as incomplete and let validation catch it.
        # Using schoolbook as placeholder:
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    u_idx = (i_off + i) * m + (j_off + j)
                    v_idx = (j_off + j) * p + (k_off + k)
                    w_idx = (k_off + k) * n + (i_off + i)
                    add_rank1([u_idx], [1], [v_idx], [1], [w_idx], [1])
    
    def add_3_1_3(i_off, j_off, k_off):
        # M(3,1,3): schoolbook, 9 terms
        for i in range(3):
            for k in range(3):
                u_idx = (i_off + i) * m + (j_off)
                v_idx = (j_off) * p + (k_off + k)
                w_idx = (k_off + k) * n + (i_off + i)
                add_rank1([u_idx], [1], [v_idx], [1], [w_idx], [1])
    
    # This will use schoolbook for 3x3x3, giving 27+27+9+9=72, which is wrong.
    # I need the actual 23-term decomposition.
    # For now, this function is incomplete.
    
    # Actually, let me just return the 66-rank solution directly.
    raise ValueError("64-rank not implemented")

def _construct_66():
    n, m, p = DIMENSIONS
    rank = 66
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    current_rank = 0
    
    def add_rank1(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals):
        nonlocal current_rank
        if current_rank >= rank:
            return
        for idx, val in zip(u_indices, u_vals):
            U[idx][current_rank] += val
        for idx, val in zip(v_indices, v_vals):
            V[idx][current_rank] += val
        for idx, val in zip(w_indices, w_vals):
            W[idx][current_rank] += val
        current_rank += 1
    
    def add_1_4_2(i_off, j_off, k_off):
        for j in range(4):
            for k in range(2):
                u_idx = (i_off) * m + (j_off + j)
                v_idx = (j_off + j) * p + (k_off + k)
                w_idx = (k_off + k) * n + i_off
                add_rank1([u_idx], [1], [v_idx], [1], [w_idx], [1])
    
    def add_2_2_2(i_off, j_off, k_off):
        def u(i, j): return (i_off + i) * m + (j_off + j)
        def v(j, k): return (j_off + j) * p + (k_off + k)
        def w(i, k): return (k_off + k) * n + (i_off + i)
        
        add_rank1([u(0,0), u(1,1)], [1, 1], [v(0,0), v(1,1)], [1, 1], [w(0,0), w(1,1)], [1, 1])
        add_rank1([u(1,0), u(1,1)], [1, 1], [v(0,0)], [1], [w(1,0), w(1,1)], [1, -1])
        add_rank1([u(0,0)], [1], [v(0,1), v(1,1)], [1, -1], [w(0,1), w(1,1)], [1, 1])
        add_rank1([u(1,1)], [1], [v(1,0), v(0,0)], [1, -1], [w(0,0), w(1,0)], [1, 1])
        add_rank1([u(0,0), u(0,1)], [1, 1], [v(1,1)], [1], [w(0,0), w(0,1)], [-1, 1])
        add_rank1([u(1,0), u(0,0)], [1, -1], [v(0,0), v(0,1)], [1, 1], [w(1,1)], [1])
        add_rank1([u(0,1), u(1,1)], [1, -1], [v(1,0), v(1,1)], [1, 1], [w(0,0)], [1])
    
    def add_2_4_2(i_off, j_off, k_off):
        add_2_2_2(i_off, j_off, k_off)
        add_2_2_2(i_off, j_off + 2, k_off)
    
    for block_k in range(3):
        k_off = block_k * 2
        add_1_4_2(0, 0, k_off)
        add_2_4_2(1, 0, k_off)
    
    return {"U": U, "V": V, "W": W}
