import numpy as np
import time

def construct():
    n, m, p = 2, 4, 5
    
    # The tensor for 2x4 * 4x5 matrix multiplication
    # A is 2x4, B is 4x5, C is 2x5
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # W index: k*n + i = 5*k + i (since n=2, p=5, so W is 10x)
    # U index: i*m + j = 4*i + j (since m=4, so U is 8x)
    # V index: j*p + k = 5*j + k (since p=5, so V is 20x)
    
    # Let's try to find a rank-minimal decomposition
    # The naive rank is 2*4*5 = 40
    # Using Strassen-like methods, we can potentially do better
    
    # Strategy: Use Strassen for the 2x4 * 4x4 part (rank 28) 
    # and handle the last column separately
    # 2x4 * 4x4: split B into B[:, :4] (4x4)
    #   Split A into A[:, :2] and A[:, 2:] (each 2x2)
    #   Split B[:, :4] into B[:2, :4] and B[2:, :4] (each 2x4)
    #   Each 2x2 * 2x4 can be done with Strassen-like approach
    #   Actually, 2x2 * 2x4: treat as two 2x2*2x2 blocks
    #   So 2x4*4x4 = 4 blocks of 2x2*2x2, each rank 7 = 28
    # 2x4 * 4x1 (last column): rank 8 naive, but maybe less?
    #   2x2 * 2x1: rank 4 naive
    #   So total = 28 + 8 = 36
    
    # Let's try to see if we can do better on the 2x4 * 4x1 part
    # 2x4 * 4x1 = [A1 A2] * [B1; B2] where A1,A2 are 2x2, B1,B2 are 2x1
    # = A1*B1 + A2*B2
    # Each 2x2 * 2x1 is rank 4 (can't really do better for rectangular)
    # So 8 for this part seems optimal
    
    # Total rank 36. Let's construct it.
    
    rank = 36
    U = np.zeros((n*m, rank))  # 8 x 36
    V = np.zeros((m*p, rank))  # 20 x 36
    W = np.zeros((p*n, rank))  # 10 x 36
    
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    col = 0
    
    def strassen_2x2_2x2(y, z, w_map):
        """
        y: 4 indices (i,j) for the 2x2 A-block: (y00, y01, y10, y11)
        z: 4 indices (j,k) for the 2x2 B-block: (z00, z01, z10, z11)
        w_map: dict from (local_r, local_c) to (global_k, global_i)
        """
        nonlocal col
        y00, y01, y10, y11 = y
        z00, z01, z10, z11 = z
        
        k0, i0 = w_map[0,0]
        k1, i1 = w_map[0,1]
        k2, i2 = w_map[1,0]
        k3, i3 = w_map[1,1]
        
        # M1 = (y00+y11)*(z00+z11)
        U[u_idx(*y00)][col] += 1
        U[u_idx(*y11)][col] += 1
        V[v_idx(*z00)][col] += 1
        V[v_idx(*z11)][col] += 1
        W[w_idx(k0,i0)][col] += 1
        W[w_idx(k3,i3)][col] += 1
        col += 1
        
        # M2 = (y10+y11)*z00
        U[u_idx(*y10)][col] += 1
        U[u_idx(*y11)][col] += 1
        V[v_idx(*z00)][col] += 1
        W[w_idx(k2,i2)][col] += 1
        W[w_idx(k3,i3)][col] -= 1
        col += 1
        
        # M3 = y00*(z01-z11)
        U[u_idx(*y00)][col] += 1
        V[v_idx(*z01)][col] += 1
        V[v_idx(*z11)][col] -= 1
        W[w_idx(k1,i1)][col] += 1
        W[w_idx(k3,i3)][col] += 1
        col += 1
        
        # M4 = y11*(z10-z00)
        U[u_idx(*y11)][col] += 1
        V[v_idx(*z10)][col] += 1
        V[v_idx(*z00)][col] -= 1
        W[w_idx(k0,i0)][col] += 1
        W[w_idx(k2,i2)][col] += 1
        col += 1
        
        # M5 = (y00+y01)*z11
        U[u_idx(*y00)][col] += 1
        U[u_idx(*y01)][col] += 1
        V[v_idx(*z11)][col] += 1
        W[w_idx(k0,i0)][col] -= 1
        W[w_idx(k1,i1)][col] += 1
        col += 1
        
        # M6 = (y10-y00)*(z00+z01)
        U[u_idx(*y10)][col] += 1
        U[u_idx(*y00)][col] -= 1
        V[v_idx(*z00)][col] += 1
        V[v_idx(*z01)][col] += 1
        W[w_idx(k3,i3)][col] += 1
        col += 1
        
        # M7 = (y01-y11)*(z10+z11)
        U[u_idx(*y01)][col] += 1
        U[u_idx(*y11)][col] -= 1
        V[v_idx(*z10)][col] += 1
        V[v_idx(*z11)][col] += 1
        W[w_idx(k0,i0)][col] += 1
        col += 1
    
    # 2x4 * 4x4 part: split into 4 blocks of 2x2 * 2x2
    # A = [A_left | A_right] where A_left = A[:,0:2], A_right = A[:,2:4]
    # B = [B_top; B_bottom] where B_top = B[0:2,:4], B_bottom = B[2:4,:4]
    
    # Block 1: A_left * B_top -> C[:, 0:2]
    # A_left indices: (0,0),(0,1),(1,0),(1,1)
    # B_top indices: (0,0),(0,1),(1,0),(1,1)
    # C[:,0:2] -> w_map: (0,0)->(0,0), (0,1)->(1,0), (1,0)->(0,1), (1,1)->(1,1)
    y1 = [(0,0),(0,1),(1,0),(1,1)]
    z1 = [(0,0),(0,1),(1,0),(1,1)]
    w1 = {(0,0):(0,0), (0,1):(1,0), (1,0):(0,1), (1,1):(1,1)}
    strassen_2x2_2x2(y1, z1, w1)
    
    # Block 2: A_right * B_top -> C[:, 0:2]
    # A_right indices: (0,2),(0,3),(1,2),(1,3)
    # B_top indices: (0,0),(0,1),(1,0),(1,1)
    y2 = [(0,2),(0,3),(1,2),(1,3)]
    z2 = [(0,0),(0,1),(1,0),(1,1)]
    w2 = {(0,0):(0,0), (0,1):(1,0), (1,0):(0,1), (1,1):(1,1)}
    strassen_2x2_2x2(y2, z2, w2)
    
    # Block 3: A_left * B_bottom -> C[:, 2:4]
    # A_left indices: (0,0),(0,1),(1,0),(1,1)
    # B_bottom indices: (2,0),(2,1),(3,0),(3,1)
    y3 = [(0,0),(0,1),(1,0),(1,1)]
    z3 = [(2,0),(2,1),(3,0),(3,1)]
    w3 = {(0,0):(2,0), (0,1):(3,0), (1,0):(2,1), (1,1):(3,1)}
    strassen_2x2_2x2(y3, z3, w3)
    
    # Block 4: A_right * B_bottom -> C[:, 2:4]
    # A_right indices: (0,2),(0,3),(1,2),(1,3)
    # B_bottom indices: (2,0),(2,1),(3,0),(3,1)
    y4 = [(0,2),(0,3),(1,2),(1,3)]
    z4 = [(2,0),(2,1),(3,0),(3,1)]
    w4 = {(0,0):(2,0), (0,1):(3,0), (1,0):(2,1), (1,1):(3,1)}
    strassen_2x2_2x2(y4, z4, w4)
    
    # 2x4 * 4x1 part (column 4 of B)
    # A_left * B_top_col4: 2x2 * 2x1
    # A_left: (0,0),(0,1),(1,0),(1,1)
    # B_top_col4: (0,4),(1,4)
    # C[:,4]: (4,0),(4,1)
    
    # Direct 2x2 * 2x1: C[0,4] = A[0,0]*B[0,4] + A[0,1]*B[1,4]
    #                 C[1,4] = A[1,0]*B[0,4] + A[1,1]*B[1,4]
    U[u_idx(0,0)][col] = 1; V[v_idx(0,4)][col] = 1; W[w_idx(4,0)][col] = 1; col+=1
    U[u_idx(0,1)][col] = 1; V[v_idx(1,4)][col] = 1; W[w_idx(4,0)][col] = 1; col+=1
    U[u_idx(1,0)][col] = 1; V[v_idx(0,4)][col] = 1; W[w_idx(4,1)][col] = 1; col+=1
    U[u_idx(1,1)][col] = 1; V[v_idx(1,4)][col] = 1; W[w_idx(4,1)][col] = 1; col+=1
    
    # A_right * B_bottom_col4: 2x2 * 2x1
    # A_right: (0,2),(0,3),(1,2),(1,3)
    # B_bottom_col4: (2,4),(3,4)
    U[u_idx(0,2)][col] = 1; V[v_idx(2,4)][col] = 1; W[w_idx(4,0)][col] = 1; col+=1
    U[u_idx(0,3)][col] = 1; V[v_idx(3,4)][col] = 1; W[w_idx(4,0)][col] = 1; col+=1
    U[u_idx(1,2)][col] = 1; V[v_idx(2,4)][col] = 1; W[w_idx(4,1)][col] = 1; col+=1
    U[u_idx(1,3)][col] = 1; V[v_idx(3,4)][col] = 1; W[w_idx(4,1)][col] = 1; col+=1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
