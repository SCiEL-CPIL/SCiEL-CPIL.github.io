"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    
    columns = []  # Each entry is (u_dict, v_dict, w_dict) as sparse dicts
    
    def add_term(u_dict, v_dict, w_dict):
        columns.append((u_dict, v_dict, w_dict))
    
    def add_schoolblock(n_start, m_start, p_start, ni, mi, pi):
        for i in range(ni):
            for j in range(mi):
                for k in range(pi):
                    u_idx = (n_start + i) * m + (m_start + j)
                    v_idx = (m_start + j) * p + (p_start + k)
                    w_idx = (p_start + k) * n + (n_start + i)
                    add_term({u_idx: 1}, {v_idx: 1}, {w_idx: 1})
    
    def verify_laderman_2x3x4():
        """Verify a candidate Laderman decomposition for 2x3x4."""
        # Local indices: A: i*3+j (6 elems), B: i*4+j (12 elems), C: i*4+j (8 elems)
        candidate_terms = [
            # Each term: (u_dict, v_dict, w_dict) in local coords
            ({0: 1}, {0: 1}, {0: 1}),
            ({0: 1}, {1: 1}, {1: 1}),
            ({1: 1}, {4: 1}, {0: 1}),
            ({1: 1}, {5: 1}, {1: 1}),
            ({2: 1}, {8: 1}, {0: 1}),
            ({2: 1}, {9: 1}, {1: 1}),
            ({0: 1, 3: 1}, {2: 1, 6: 1}, {2: 1, 6: 1}),
            ({1: 1, 4: 1}, {6: 1, 10: 1}, {2: 1, 6: 1}),
            ({2: 1, 5: 1}, {10: 1, 2: 1}, {2: 1, 6: 1}),
            ({0: 1, 3: 1}, {3: 1, 7: 1}, {3: 1, 7: 1}),
            ({1: 1, 4: 1}, {7: 1, 11: 1}, {3: 1, 7: 1}),
            ({2: 1, 5: 1}, {11: 1, 3: 1}, {3: 1, 7: 1}),
            ({3: 1}, {0: 1}, {4: 1}),
            ({4: 1}, {4: 1}, {4: 1}),
        ]
        
        # Build local U, V, W matrices
        local_rank = len(candidate_terms)
        local_U = np.zeros((6, local_rank))
        local_V = np.zeros((12, local_rank))
        local_W = np.zeros((8, local_rank))
        
        for col_idx, (ud, vd, wd) in enumerate(candidate_terms):
            for idx, val in ud.items():
                local_U[idx, col_idx] = val
            for idx, val in vd.items():
                local_V[idx, col_idx] = val
            for idx, val in wd.items():
                local_W[idx, col_idx] = val
        
        # Compute the tensor: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[i,k,r]
        # But we need to check it equals the matmul tensor.
        # The matmul tensor for 2x3x4: T[i,j,k] = 1 if the (i,j) of A and (j,k) of B produce (i,k) of C
        # Actually the tensor T[i,j,k] = 1 for all valid (i,j,k) since C[i,k] = sum_j A[i,j]*B[j,k]
        
        # Let's compute: for each (i,k), C[i,k] = sum_j A[i,j]*B[j,k]
        # The decomposition says: C[i,k] = sum_r (sum_j U[i,j,r]*B[j,k]) * W[i,k,r]
        # No wait, the tensor decomposition is:
        # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[i,k,r]
        # where T[i,j,k] = 1 for all i in [0,2), j in [0,3), k in [0,4)
        
        # Actually for matrix multiplication tensor:
        # The tensor T has T[i,j,k] = 1 if the product A[i,j]*B[j,k] contributes to C[i,k]
        # So T[i,j,k] = 1 for all valid indices (since every A[i,j]*B[j,k] contributes to C[i,k])
        
        # Let's verify: compute sum_r U[i,j,r]*V[j,k,r]*W[i,k,r] and check it equals 1 for all (i,j,k)
        
        for i in range(2):
            for j in range(3):
                for k in range(4):
                    val = 0
                    for r in range(local_rank):
                        val += local_U[i*3+j, r] * local_V[j*4+k, r] * local_W[i*4+k, r]
                    if abs(val - 1) > 1e-10:
                        return False, candidate_terms
        return True, candidate_terms
    
    def add_laderman(n_start, m_start, p_start):
        """Add Laderman's rank-14 decomposition for 2x3x4 block, with verification."""
        # Local to global index mapping
        def u_glob(li, lj):  # li in {0,1}, lj in {0,1,2}
            return (n_start + li) * m + (m_start + lj)
        def v_glob(li, lj):  # li in {0,1,2}, lj in {0,1,2,3}
            return (m_start + li) * p + (p_start + lj)
        def w_glob(li, lj):  # li in {0,1}, lj in {0,1,2,3}
            return (p_start + lj) * n + (n_start + li)
        
        # Try the candidate Laderman decomposition
        valid, terms = verify_laderman_2x3x4()
        
        if valid:
            # Use the 14-term Laderman decomposition
            for ud, vd, wd in terms:
                gu = {}
                gv = {}
                gw = {}
                for li, val in ud.items():
                    i, j = divmod(li, 3)
                    gu[u_glob(i, j)] = val
                for li, val in vd.items():
                    i, j = divmod(li, 4)
                    gv[v_glob(i, j)] = val
                for li, val in wd.items():
                    i, j = divmod(li, 4)
                    gw[w_glob(i, j)] = val
                add_term(gu, gv, gw)
        else:
            # Fallback to schoolbook (24 terms)
            for i in range(2):
                for j in range(3):
                    for k in range(4):
                        u = {u_glob(i, j): 1}
                        v = {v_glob(j, k): 1}
                        w = {w_glob(i, k): 1}
                        add_term(u, v, w)
    
    # Block 1: n=[0], m=[0,1,2], p=[0,1,2,3]
    add_schoolblock(0, 0, 0, 1, 3, 4)  # 12 terms
    
    # Block 2: n=[0], m=[3], p=[0,1,2,3]
    add_schoolblock(0, 3, 0, 1, 1, 4)  # 4 terms
    
    # Block 3: n=[0], m=[0,1,2], p=[4,5,6,7]
    add_schoolblock(0, 0, 4, 1, 3, 4)  # 12 terms
    
    # Block 4: n=[0], m=[3], p=[4,5,6,7]
    add_schoolblock(0, 3, 4, 1, 1, 4)  # 4 terms
    
    # Block 5: n=[1,2], m=[0,1,2], p=[0,1,2,3] - Laderman
    add_laderman(1, 0, 0)  # 14 or 24 terms
    
    # Block 6: n=[1,2], m=[3], p=[0,1,2,3]
    add_schoolblock(1, 3, 0, 2, 1, 4)  # 8 terms
    
    # Block 7: n=[1,2], m=[0,1,2], p=[4,5,6,7] - Laderman
    add_laderman(1, 0, 4)  # 14 or 24 terms
    
    # Block 8: n=[1,2], m=[3], p=[4,5,6,7]
    add_schoolblock(1, 3, 4, 2, 1, 4)  # 8 terms
    
    rank = len(columns)
    
    # Build U, V, W matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col_idx, (u_dict, v_dict, w_dict) in enumerate(columns):
        for idx, val in u_dict.items():
            U[idx][col_idx] = val
        for idx, val in v_dict.items():
            V[idx][col_idx] = val
        for idx, val in w_dict.items():
            W[idx][col_idx] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
