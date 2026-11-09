import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def construct():
    start_time = time.time()
    deadline = 120  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # We will construct the rank 99 decomposition.
    # Part 1: A[0:4, 0:4] * B[0:4, 0:3] (padded to 4x4) -> C[0:4, 0:3]
    # Part 2: A[0:4, 0:4] * B[0:4, 3:6] (padded to 4x4) -> C[0:4, 3:6]
    # Part 3: A[0:4, 4] * B[4, 0:6] -> C[0:4, 0:6]
    
    # Helper to generate Strassen decomposition for n x n
    def get_strassen_decomp(n):
        if n == 1:
            return [[1]], [[1]], [[1]]
        
        m_half = n // 2
        U_m, V_m, W_m = get_strassen_decomp(m_half)
        R_m = len(U_m[0])
        R = 7 * R_m
        
        U = [[0] * R for _ in range(n * n)]
        V = [[0] * R for _ in range(n * n)]
        W = [[0] * R for _ in range(n * n)]
        
        # Define block index ranges
        # Block (r, c) of size m_half x m_half
        # Global index: (r * m_half + i) * n + (c * m_half + j)
        
        def block_indices(r, c):
            rows = range(r * m_half, (r + 1) * m_half)
            cols = range(c * m_half, (c + 1) * m_half)
            indices = []
            for i in rows:
                for j in cols:
                    indices.append(i * n + j)
            return indices
        
        # Strassen terms definitions
        # (A_blocks, B_blocks, C_blocks)
        # A_blocks: list of (block_id, coeff)
        # B_blocks: list of (block_id, coeff)
        # C_blocks: list of (block_id, coeff)
        
        terms = [
            # M1
            ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, -1)]),
            # M2
            ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, 1)]),
            # M3
            ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
            # M4
            ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
            # M5
            ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
            # M6
            ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
            # M7
            ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1), (3, 1)])
        ]
        
        for t in range(7):
            a_blocks, b_blocks, c_blocks = terms[t]
            offset = t * R_m
            
            # Fill U
            for b_id, coeff in a_blocks:
                r, c = divmod(b_id, 2)
                indices = block_indices(r, c)
                for idx in indices:
                    for k in range(R_m):
                        U[idx][offset + k] += coeff * U_m[idx % (m_half * m_half)][k] # Wait, U_m indices are local to m x m
                        # Actually, U_m row index corresponds to the local i,j.
                        # We need to map local (i,j) to global (r*m+i, c*m+j).
                        # The loop above does that.
                        
            # Fill V
            for b_id, coeff in b_blocks:
                r, c = divmod(b_id, 2)
                indices = block_indices(r, c)
                for idx in indices:
                    for k in range(R_m):
                        V[idx][offset + k] += coeff * V_m[idx % (m_half * m_half)][k]

            # Fill W
            for b_id, coeff in c_blocks:
                r, c = divmod(b_id, 2)
                indices = block_indices(r, c)
                for idx in indices:
                    for k in range(R_m):
                        W[idx][offset + k] += coeff * W_m[idx % (m_half * m_half)][k]

        return U, V, W

    U_4x4, V_4x4, W_4x4 = get_strassen_decomp(4)
    rank_4x4 = len(U_4x4[0]) # Should be 49
    
    total_rank = rank_4x4 * 2 + 1
    U = [[0] * total_rank for _ in range(n * m)] # 4x5 = 20
    V = [[0] * total_rank for _ in range(m * p)] # 5x6 = 30
    W = [[0] * total_rank for _ in range(p * n)] # 6x4 = 24
    
    # Part 1: A[0:4, 0:4] * B[0:4, 0:3]
    # U rows: 0..19 (A indices)
    # V rows: 0..29 (B indices)
    # W rows: 0..23 (C indices, k*4+i)
    
    for i in range(4):
        for j in range(4):
            u_idx = i * m + j # A index
            for k in range(rank_4x4):
                U[u_idx][k] = U_4x4[i * 4 + j][k]
                
    for i in range(4):
        for j in range(3): # B cols 0,1,2
            v_idx = i * p + j # B index
            for k in range(rank_4x4):
                V[v_idx][k] = V_4x4[i * 4 + j][k]
                
    for i in range(4):
        for j in range(3): # C cols 0,1,2
            w_idx = j * n + i # C index (k*4+i)
            for k in range(rank_4x4):
                W[w_idx][k] = W_4x4[i * 4 + j][k]

    # Part 2: A[0:4, 0:4] * B[0:4, 3:6]
    offset2 = rank_4x4
    
    for i in range(4):
        for j in range(4):
            u_idx = i * m + j
            for k in range(rank_4x4):
                U[u_idx][offset2 + k] = U_4x4[i * 4 + j][k]
                
    for i in range(4):
        for j in range(3): # B cols 3,4,5 -> local 0,1,2
            v_idx = i * p + (j + 3)
            for k in range(rank_4x4):
                V[v_idx][offset2 + k] = V_4x4[i * 4 + j][k]
                
    for i in range(4):
        for j in range(3): # C cols 3,4,5 -> local 0,1,2
            w_idx = (j + 3) * n + i
            for k in range(rank_4x4):
                W[w_idx][offset2 + k] = W_4x4[i * 4 + j][k]

    # Part 3: A[0:4, 4] * B[4, 0:6]
    offset3 = 2 * rank_4x4
    
    for i in range(4):
        u_idx = i * m + 4
        U[u_idx][offset3] = 1
        
    for j in range(6):
        v_idx = 4 * p + j
        V[v_idx][offset3] = 1
        
    for i in range(4):
        for j in range(6):
            w_idx = j * n + i
            W[w_idx][offset3] = 1

    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
