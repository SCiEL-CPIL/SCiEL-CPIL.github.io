"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 4, 6
    
    # Build the rank-42 Strassen decomposition
    rank = 42
    U = [[0] * rank for _ in range(n * m)]  # 8 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 12 x 42
    
    def add_strassen_block(a_idx, b_idx, w_idx, start_col):
        """
        a_idx: [a00, a01, a10, a11] indices into U
        b_idx: [b00, b01, b10, b11] indices into V
        w_idx: [w00, w01, w10, w11] indices into W for the 2x2 output block
        """
        c = start_col
        # M1 = (a00+a11)(b00+b11)
        U[a_idx[0]][c] += 1; U[a_idx[3]][c] += 1
        V[b_idx[0]][c] += 1; V[b_idx[3]][c] += 1
        W[w_idx[0]][c] += 1; W[w_idx[3]][c] += 1
        # M2 = (a10+a11)b00
        U[a_idx[2]][c+1] += 1; U[a_idx[3]][c+1] += 1
        V[b_idx[0]][c+1] += 1
        W[w_idx[2]][c+1] += 1; W[w_idx[3]][c+1] -= 1
        # M3 = a00(b01-b11)
        U[a_idx[0]][c+2] += 1
        V[b_idx[1]][c+2] += 1; V[b_idx[3]][c+2] -= 1
        W[w_idx[1]][c+2] += 1; W[w_idx[3]][c+2] += 1
        # M4 = a11(b10-b00)
        U[a_idx[3]][c+3] += 1
        V[b_idx[2]][c+3] += 1; V[b_idx[0]][c+3] -= 1
        W[w_idx[0]][c+3] += 1; W[w_idx[2]][c+3] += 1
        # M5 = (a00+a01)b11
        U[a_idx[0]][c+4] += 1; U[a_idx[1]][c+4] += 1
        V[b_idx[3]][c+4] += 1
        W[w_idx[0]][c+4] -= 1; W[w_idx[1]][c+4] += 1
        # M6 = (a10-a00)(b00+b01)
        U[a_idx[2]][c+5] += 1; U[a_idx[0]][c+5] -= 1
        V[b_idx[0]][c+5] += 1; V[b_idx[1]][c+5] += 1
        W[w_idx[3]][c+5] += 1
        # M7 = (a01-a11)(b10+b11)
        U[a_idx[1]][c+6] += 1; U[a_idx[3]][c+6] -= 1
        V[b_idx[2]][c+6] += 1; V[b_idx[3]][c+6] += 1
        W[w_idx[0]][c+6] += 1
    
    # A is 2x4: indices i*m+j
    # A[0,0]=0, A[0,1]=1, A[0,2]=2, A[0,3]=3
    # A[1,0]=4, A[1,1]=5, A[1,2]=6, A[1,3]=7
    a_L = [0, 1, 4, 5]  # a00, a01, a10, a11
    a_R = [2, 3, 6, 7]
    
    # B is 4x6: indices j*p+k
    # B col block 0: cols 0,1
    b_1_T = [0, 1, 6, 7]
    b_1_B = [12, 13, 18, 19]
    w_1 = [0, 2, 1, 3]
    
    add_strassen_block(a_L, b_1_T, w_1, 0)
    add_strassen_block(a_R, b_1_B, w_1, 7)
    
    # B col block 1: cols 2,3
    b_2_T = [2, 3, 8, 9]
    b_2_B = [14, 15, 20, 21]
    w_2 = [4, 6, 5, 7]
    
    add_strassen_block(a_L, b_2_T, w_2, 14)
    add_strassen_block(a_R, b_2_B, w_2, 21)
    
    # B col block 2: cols 4,5
    b_3_T = [4, 5, 10, 11]
    b_3_B = [16, 17, 22, 23]
    w_3 = [8, 10, 9, 11]
    
    add_strassen_block(a_L, b_3_T, w_3, 28)
    add_strassen_block(a_R, b_3_B, w_3, 35)
    
    # Verify the decomposition
    def verify_decomp(U, V, W, rank):
        """Check if sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i,j,k"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(rank):
                        total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(total - 1) > 1e-9:
                        return False
        return True
    
    if not verify_decomp(U, V, W, rank):
        # Fallback: return schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U[i*m+j][r] = 1
                    V[j*p+k][r] = 1
                    W[k*n+i][r] = 1
        return {"U": U, "V": V, "W": W}
    
    # Convert to numpy for linear algebra
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Attempt to reduce rank by finding linear dependencies among the 42 terms
    # We'll try to find a subset of columns that can be removed while adjusting
    # the remaining columns to still satisfy the tensor equation.
    
    # The key insight: if we have a valid decomposition, we can try to find
    # linear combinations of the rank-1 tensors that sum to zero.
    
    # For each pair of columns (r1, r2), check if there exists a non-trivial
    # combination alpha * T_{r1} + beta * T_{r2} = 0 where T_r is the rank-1 tensor
    # defined by (U[:,r], V[:,r], W[:,r]).
    
    # A simpler approach: try removing one column at a time and see if the remaining
    # 41 columns can be scaled to still work. This is a linear system in the scaling factors.
    
    # Actually, let's try a more systematic approach: use the fact that the tensor
    # has symmetries. The 2x4x6 tensor can be viewed as 3 copies of 2x2x2 tensors
    # with shared structure.
    
    # Let's try to find if rank 40 is achievable by checking all pairs of columns
    # that might be removable.
    
    best_rank = rank
    best_U, best_V, best_W = U, V, W
    
    # Try to find rank-40 decomposition by removing 2 columns
    # For each pair of columns to remove, check if the remaining 40 columns
    # can satisfy the tensor equation with appropriate scaling.
    
    if time.time() < deadline - 30:
        # For each pair of columns, try to solve for scaling factors
        # that make the remaining columns work.
        
        # We'll use a least-squares approach: for each (i,j,k), we need
        # sum_{r in active} U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
        # This is nonlinear, but if we fix two factors, it's linear in the third.
        
        # Let's fix U and V, and solve for W. For each row of W (indexed by k*n+i),
        # we have multiple equations (one for each j).
        
        for remove_pair in combinations(range(rank), 2):
            if time.time() > deadline - 25:
                break
            
            active = [r for r in range(rank) if r not in remove_pair]
            R = len(active)
            
            # Set up the linear system for W
            # Unknowns: W[k*n+i, r] for r in active, for each k*n+i
            # Equations: for each (i,j,k): sum_{r in active} U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
            
            # For a fixed (k, i), the unknowns are W[k*n+i, r] for r in active
            # The equations involving these unknowns are for all j:
            # sum_{r in active} U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
            
            # This gives m=4 equations for R=40 unknowns. Underdetermined.
            # But we have the constraint that the original W values are close to the solution.
            
            # Let's use a different approach: try to find if the removed columns
            # are in the span of the remaining ones in the tensor space.
            
            # For now, let's just check if zeroing out the removed columns still works
            # (i.e., if they contribute 0 to the tensor).
            
            U_test = U_np.copy()
            V_test = V_np.copy()
            W_test = W_np.copy()
            
            U_test[:, list(remove_pair)] = 0
            V_test[:, list(remove_pair)] = 0
            W_test[:, list(remove_pair)] = 0
            
            # Check if this still satisfies the tensor
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        total = 0
                        for r in active:
                            total += U_test[i*m+j, r] * V_test[j*p+k, r] * W_test[k*n+i, r]
                        if abs(total - 1) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                best_rank = R
                best_U = U_test.tolist()
                best_V = V_test.tolist()
                best_W = W_test.tolist()
                break  # Found rank 40, stop
    
    # If we found rank 40, try rank 38
    if best_rank == 40 and time.time() < deadline - 20:
        for remove_pair in combinations(range(best_rank), 2):
            if time.time() > deadline - 15:
                break
            
            active = [r for r in range(best_rank) if r not in remove_pair]
            R = len(active)
            
            U_test = np.array(best_U, dtype=float)
            V_test = np.array(best_V, dtype=float)
            W_test = np.array(best_W, dtype=float)
            
            U_test[:, list(remove_pair)] = 0
            V_test[:, list(remove_pair)] = 0
            W_test[:, list(remove_pair)] = 0
            
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        total = 0
                        for r in active:
                            total += U_test[i*m+j, r] * V_test[j*p+k, r] * W_test[k*n+i, r]
                        if abs(total - 1) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                best_rank = R
                best_U = U_test.tolist()
                best_V = V_test.tolist()
                best_W = W_test.tolist()
                break
    
    if best_rank < rank:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
