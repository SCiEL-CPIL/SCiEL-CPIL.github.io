import numpy as np
import time
import random

DIMENSIONS = [2, 3, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 135
    
    n, m, p = DIMENSIONS
    
    def find_2x2x3_rank10():
        # Search for rank 10 decomp of 2x2 by 2x3
        # U: 4x10, V: 6x10, W: 6x10
        # Indices: U[i*2+j, c], V[j*3+k, c], W[k*2+i, c]
        # Wait, V is (2,3) so size 6. W is (3,2) so size 6.
        
        # Try random integer solutions
        for _ in range(1000):
            if time.time() > deadline - 10:
                return None
            
            U = np.zeros((4, 10), dtype=np.int64)
            V = np.zeros((6, 10), dtype=np.int64)
            W = np.zeros((6, 10), dtype=np.int64)
            
            # Random sparse initialization
            for c in range(10):
                # U
                idx_u = random.randint(0, 3)
                U[idx_u, c] = random.choice([-1, 1])
                
                # V
                idx_v = random.randint(0, 5)
                V[idx_v, c] = random.choice([-1, 1])
                
                # W
                idx_w = random.randint(0, 5)
                W[idx_w, c] = random.choice([-1, 1])
            
            # Check
            valid = True
            for i in range(2):
                for j in range(2):
                    for k in range(3):
                        # Sum over c of U[i,j,c] V[j,k,c] W[k,i,c]
                        # Note: V index is j*3+k, W index is k*2+i
                        val = 0
                        for c in range(10):
                            val += U[i*2+j, c] * V[j*3+k, c] * W[k*2+i, c]
                        if val != 1:
                            valid = False
                            break
                    if not valid: break
                if not valid: break
            
            if valid:
                return U, V, W
                
        return None

    def build_solution(sub_decomp):
        # sub_decomp: U (4,10), V (6,10), W (6,10) for 2x2 by 2x3
        # Total rank: 3 * 10 + 18 = 48
        rank = 48
        U_full = [[0]*rank for _ in range(6)]
        V_full = [[0]*rank for _ in range(27)]
        W_full = [[0]*rank for _ in range(18)]
        
        # Part 1: 3 blocks of 2x2 by 2x3
        # A_L is 2x2 (rows 0,1 of A, cols 0,1 of A) -> indices 0,1,2,3 in U_full
        # B_0:2, 0:3 is 2x3 -> indices 0..5 in V_full
        # C_0:2, 0:3 is 2x3 -> indices 0..5 in W_full (k*2+i)
        
        if sub_decomp:
            U_sub, V_sub, W_sub = sub_decomp
            for block in range(3):
                col_offset = block * 10
                # Map sub indices to full indices
                # U: i in 0..1, j in 0..1 -> i*3+j? No, A is 2x3.
                # A_L is cols 0,1. So j_sub 0->0, 1->1.
                # U_full index: i*m + j = i*3 + j.
                # U_sub index: i*2 + j_sub.
                
                for i in range(2):
                    for j_sub in range(2):
                        j_full = j_sub # j is 0 or 1
                        u_idx = i*3 + j_full
                        u_sub_idx = i*2 + j_sub
                        U_full[u_idx][col_offset + 0] = 0 # Placeholder, will copy whole col
                        # Copy all 10 cols
                        for c in range(10):
                            U_full[u_idx][col_offset + c] = U_sub[u_sub_idx, c]
                
                # V: j_sub in 0..1, k_sub in 0..2
                # V_full index: j*m? No, V is (3,9). j is row of B (0..2).
                # Here j_sub is 0,1 (rows of B_0:2).
                # V_full index: j_sub * 9 + (block*3 + k_sub)
                for j_sub in range(2):
                    for k_sub in range(3):
                        v_idx = j_sub * 9 + (block*3 + k_sub)
                        v_sub_idx = j_sub * 3 + k_sub
                        for c in range(10):
                            V_full[v_idx][col_offset + c] = V_sub[v_sub_idx, c]
                
                # W: k_sub in 0..2, i in 0..1
                # W_full index: (block*3 + k_sub) * 2 + i
                for k_sub in range(3):
                    for i in range(2):
                        w_idx = (block*3 + k_sub) * 2 + i
                        w_sub_idx = k_sub * 2 + i
                        for c in range(10):
                            W_full[w_idx][col_offset + c] = W_sub[w_sub_idx, c]
        else:
            # Fallback: Rank 50 logic (Strassen 7 for 2x2x2, etc)
            # Just return the original rank 50
            return None

        # Part 2: A_R (2x1) * B_B (1x9)
        # A_R is col 2 of A. Indices in U: i*3 + 2.
        # B_B is row 2 of B. Indices in V: 2*9 + k.
        # C is all cols. Indices in W: k*2 + i.
        
        col_offset = 30
        for k in range(9):
            # Term 1: A_{0,2} B_{2,k} -> C_{0,k}
            U_full[0*3+2][col_offset] = 1
            V_full[2*9+k][col_offset] = 1
            W_full[k*2+0][col_offset] = 1
            
            # Term 2: A_{1,2} B_{2,k} -> C_{1,k}
            U_full[1*3+2][col_offset+1] = 1
            V_full[2*9+k][col_offset+1] = 1
            W_full[k*2+1][col_offset+1] = 1
            
            col_offset += 2
            
        return {"U": U_full, "V": V_full, "W": W_full}
        
    def verify_solution(sol):
        if not sol: return False
        U = np.array(sol["U"])
        V = np.array(sol["V"])
        W = np.array(sol["W"])
        r = len(U[0])
        for i in range(2):
            for j in range(3):
                for k in range(9):
                    val = np.sum(U[i*3+j, :] * V[j*9+k, :] * W[k*2+i, :])
                    if val != 1:
                        return False
        return True

    # Try to find rank 10 sub-decomp
    sub_decomp = find_2x2x3_rank10()
    
    if sub_decomp:
        sol = build_solution(sub_decomp)
        if verify_solution(sol):
            return sol
            
    # Fallback to Rank 50
    return current_rank_50_solution()
    
def current_rank_50_solution():
    # Copy from original
    pass
