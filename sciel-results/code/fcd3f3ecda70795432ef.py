import time
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    global_n, global_m, global_p = n, m, p
    
    # We will determine the total rank first to allocate matrices
    # Or we can just use lists and convert to numpy at the end.
    # Lists are easier to append to.
    
    U_list = []
    V_list = []
    W_list = []
    
    def add_term(u_row, u_val, v_row, v_val, w_row, w_val, rank_idx):
        # Ensure lists are long enough
        while len(U_list) <= rank_idx:
            U_list.append([0] * (global_n * global_m))
            V_list.append([0] * (global_m * global_p))
            W_list.append([0] * (global_p * global_n))
            
        U_list[rank_idx][u_row] += u_val
        V_list[rank_idx][v_row] += v_val
        W_list[rank_idx][w_row] += w_val

    # Helper to add a sub-decomposition
    # u_off, v_off, w_off are row offsets for U, V, W
    # local U, V, W are 2D lists
    def add_sub(u_off, v_off, w_off, n_sub, m_sub, p_sub, U_sub, V_sub, W_sub, rank_start):
        rank = len(U_sub[0]) if U_sub else 0
        for r in range(rank):
            # U_sub is n_sub*m_sub x rank
            for i in range(n_sub * m_sub):
                val = U_sub[i][r]
                if val != 0:
                    # Global row index for U: 
                    # U index is (i_global, j_global).
                    # i_global = n_off + i_local_row_n
                    # j_global = m_off + i_local_row_m
                    # i_local_row is i.
                    # i = i_n * m_sub + i_m
                    i_n = i // m_sub
                    i_m = i % m_sub
                    u_row = (u_off + i_n) * global_m + (v_off + i_m)
                    add_term(u_row, val, -1, 0, -1, 0, rank_start + r) # Placeholder, need to do V and W too
            
            # This is inefficient. 
            # I'll just iterate over all non-zeros in U_sub, V_sub, W_sub simultaneously.
            pass

    # Better approach: 
    # Generate the full U, V, W for each base case locally, 
    # then map indices and add to global lists.
    
    def get_base_1x4x4():
        n, m, p = 1, 4, 4
        rank = 4
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        for r in range(4):
            U[0*m + r][r] = 1
            for k in range(4):
                V[r*p + k][r] = 1
                W[k*n + 0][r] = 1
        return U, V, W

    def get_base_2x2x2():
        n, m, p = 2, 2, 2
        rank = 7
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        
        # M1
        U[0][0]=1; U[3][0]=1
        V[0][0]=1; V[3][0]=1
        W[0][0]=1; W[3][0]=1
        # M2
        U[2][1]=1; U[3][1]=1
        V[0][1]=1
        W[1][1]=1; W[3][1]=-1
        # M3
        U[0][2]=1
        V[1][2]=1; V[3][2]=-1
        W[2][2]=1; W[3][2]=1
        # M4
        U[3][3]=1
        V[2][3]=1; V[0][3]=-1
        W[0][3]=1; W[1][3]=1
        # M5
        U[0][4]=1; U[1][4]=1
        V[3][4]=1
        W[0][4]=-1; W[2][4]=1
        # M6
        U[2][5]=1; U[0][5]=-1
        V[0][5]=1; V[1][5]=1
        W[3][5]=1
        # M7
        U[1][6]=1; U[3][6]=-1
        V[2][6]=1; V[3][6]=1
        W[0][6]=1
        
        return U, V, W

    def add_decomp(n_off, m_off, p_off, n_sub, m_sub, p_sub, U_sub, V_sub, W_sub, rank_start):
        rank = len(U_sub[0])
        for r in range(rank):
            # Process U
            for i in range(n_sub * m_sub):
                val = U_sub[i][r]
                if val != 0:
                    i_n = i // m_sub
                    i_m = i % m_sub
                    u_row = (n_off + i_n) * global_m + (m_off + i_m)
                    while len(U_list) <= rank_start + r:
                        U_list.append([0] * (global_n * global_m))
                        V_list.append([0] * (global_m * global_p))
                        W_list.append([0] * (global_p * global_n))
                    U_list[rank_start + r][u_row] += val
            
            # Process V
            for i in range(m_sub * p_sub):
                val = V_sub[i][r]
                if val != 0:
                    i_m = i // p_sub
                    i_p = i % p_sub
                    v_row = (m_off + i_m) * global_p + (p_off + i_p)
                    V_list[rank_start + r][v_row] += val
                    
            # Process W
            for i in range(p_sub * n_sub):
                val = W_sub[i][r]
                if val != 0:
                    i_p = i // n_sub
                    i_n = i % n_sub
                    w_row = (p_off + i_p) * global_n + (n_off + i_n)
                    W_list[rank_start + r][w_row] += val

    # Build 1x4x4
    U1, V1, W1 = get_base_1x4x4()
    # Build 2x2x2
    U2, V2, W2 = get_base_2x2x2()
    
    # Compose 2x2x4 from two 2x2x2
    # 2x2x4 is split p into 2+2
    # Sub 1: 2x2x2 at (0,0,0)
    # Sub 2: 2x2x2 at (0,0,2)
    # We need to "paste" them into a 2x2x4 local frame?
    # No, we can just add them to the global frame directly with correct offsets.
    
    # Hierarchy:
    # 3x4x8
    # -> 1x4x4 (n=0, m=0, p=0)
    # -> 1x4x4 (n=0, m=0, p=4)
    # -> 2x4x4 (n=1, m=0, p=0)
    #    -> 2x2x4 (n=0, m=0, p=0) [Local to 2x4x4]
    #       -> 2x2x2 (n=0, m=0, p=0)
    #       -> 2x2x2 (n=0, m=0, p=2)
    #    -> 2x2x4 (n=0, m=2, p=0)
    #       -> 2x2x2 (n=0, m=2, p=0)
    #       -> 2x2x2 (n=0, m=2, p=2)
    # -> 2x4x4 (n=1, m=0, p=4)
    #    -> ... similar
    
    rank_idx = 0
    
    # 1. 1x4x4 at (0,0,0)
    add_decomp(0, 0, 0, 1, 4, 4, U1, V1, W1, rank_idx)
    rank_idx += 4
    
    # 2. 1x4x4 at (0,0,4)
    add_decomp(0, 0, 4, 1, 4, 4, U1, V1, W1, rank_idx)
    rank_idx += 4
    
    # 3. 2x4x4 at (1,0,0)
    #   3a. 2x2x4 at (1,0,0) -> 2x2x2 at (1,0,0) and (1,0,2)
    add_decomp(1, 0, 0, 2, 2, 2, U2, V2, W2, rank_idx)
    rank_idx += 7
    add_decomp(1, 0, 2, 2, 2, 2, U2, V2, W2, rank_idx)
    rank_idx += 7
    
    #   3b. 2x2x4 at (1,2,0) -> 2x2x2 at (1,2,0) and (1,2,2)
    add_decomp(1, 2, 0, 2, 2, 2, U2, V2, W2, rank_idx)
    rank_idx += 7
    add_decomp(1, 2, 2, 2, 2, 2, U2, V2, W2, rank_idx)
    rank_idx += 7
    
    # 4. 2x4x4 at (1,0,4)
    #   4a. 2x2x4 at (1,0,4) -> 2x2x2 at (1,0,4) and (1,0,6)
    add_decomp(1, 0, 4, 2, 2, 2, U2, V2, W2, rank_idx)
    rank_idx += 7
    add_decomp(1, 0, 6, 2, 2, 2, U2, V2, W2, rank_idx)
    rank_idx += 7
    
    #   4b. 2x2x4 at (1,2,4) -> 2x2x2 at (1,2,4) and (1,2,6)
    add_decomp(1, 2, 4, 2, 2, 2, U2, V2, W2, rank_idx)
    rank_idx += 7
    add_decomp(1, 2, 6, 2, 2, 2, U2, V2, W2, rank_idx)
    rank_idx += 7
    
    # Convert to numpy arrays or return lists
    # The interface expects {'U', 'V', 'W'} with shapes (n*m, rank) etc.
    # My lists are already in that format (list of rows).
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
