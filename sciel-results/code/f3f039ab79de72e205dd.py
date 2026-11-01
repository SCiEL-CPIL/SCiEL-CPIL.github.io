import numpy as np
import time
import itertools

DIMENSIONS = [2, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin for 120s limit
    
    n, m, p = DIMENSIONS
    # n=2, m=4, p=5
    
    rank_initial = 36
    
    U = np.zeros((n * m, rank_initial), dtype=float)
    V = np.zeros((m * p, rank_initial), dtype=float)
    W = np.zeros((p * n, rank_initial), dtype=float)
    
    def u_idx(i, j):
        return i * m + j

    def v_idx(j, k):
        return j * p + k

    def w_idx(k, i):
        return k * n + i

    col = 0
    
    def strassen_block(y_indices, z_indices, w_offsets):
        nonlocal col
        
        y00, y01, y10, y11 = y_indices
        z00, z01, z10, z11 = z_indices
        
        # M1: (y00+y11)(z00+z11)
        U[u_idx(y00[0], y00[1])][col] = 1
        U[u_idx(y11[0], y11[1])][col] = 1
        V[v_idx(z00[0], z00[1])][col] = 1
        V[v_idx(z11[0], z11[1])][col] = 1
        W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] += 1
        col += 1
        
        # M2: (y10+y11)z00
        U[u_idx(y10[0], y10[1])][col] = 1
        U[u_idx(y11[0], y11[1])][col] = 1
        V[v_idx(z00[0], z00[1])][col] = 1
        W[w_idx(w_offsets[2][0], w_offsets[2][1])][col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] -= 1
        col += 1
        
        # M3: y00(z01-z11)
        U[u_idx(y00[0], y00[1])][col] = 1
        V[v_idx(z01[0], z01[1])][col] = 1
        V[v_idx(z11[0], z11[1])][col] = -1
        W[w_idx(w_offsets[1][0], w_offsets[1][1])][col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] += 1
        col += 1
        
        # M4: y11(z10-z00)
        U[u_idx(y11[0], y11[1])][col] = 1
        V[v_idx(z10[0], z10[1])][col] = 1
        V[v_idx(z00[0], z00[1])][col] = -1
        W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] += 1
        W[w_idx(w_offsets[2][0], w_offsets[2][1])][col] += 1
        col += 1
        
        # M5: (y00+y01)z11
        U[u_idx(y00[0], y00[1])][col] = 1
        U[u_idx(y01[0], y01[1])][col] = 1
        V[v_idx(z11[0], z11[1])][col] = 1
        W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] -= 1
        W[w_idx(w_offsets[1][0], w_offsets[1][1])][col] += 1
        col += 1
        
        # M6: (y10-y00)(z00+z01)
        U[u_idx(y10[0], y10[1])][col] = 1
        U[u_idx(y00[0], y00[1])][col] = -1
        V[v_idx(z00[0], z00[1])][col] = 1
        V[v_idx(z01[0], z01[1])][col] = 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] += 1
        col += 1
        
        # M7: (y01-y11)(z10+z11)
        U[u_idx(y01[0], y01[1])][col] = 1
        U[u_idx(y11[0], y11[1])][col] = -1
        V[v_idx(z10[0], z10[1])][col] = 1
        V[v_idx(z11[0], z11[1])][col] = 1
        W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] += 1
        col += 1

    # Block 1: A_L (cols 0,1) * B_11 (rows 0,1; cols 0,1)
    y1 = [(0,0), (0,1), (1,0), (1,1)]
    z1 = [(0,0), (0,1), (1,0), (1,1)]
    w1 = [(0,0), (1,0), (0,1), (1,1)]
    strassen_block(y1, z1, w1)
    
    # Block 2: A_R (cols 2,3) * B_21 (rows 2,3; cols 0,1)
    y2 = [(0,2), (0,3), (1,2), (1,3)]
    z2 = [(2,0), (2,1), (3,0), (3,1)]
    w2 = [(0,0), (1,0), (0,1), (1,1)]
    strassen_block(y2, z2, w2)
    
    # Block 3: A_L (cols 0,1) * B_12 (rows 0,1; cols 2,3)
    y3 = [(0,0), (0,1), (1,0), (1,1)]
    z3 = [(0,2), (0,3), (1,2), (1,3)]
    w3 = [(2,0), (3,0), (2,1), (3,1)]
    strassen_block(y3, z3, w3)
    
    # Block 4: A_R (cols 2,3) * B_22 (rows 2,3; cols 2,3)
    y4 = [(0,2), (0,3), (1,2), (1,3)]
    z4 = [(2,2), (2,3), (3,2), (3,3)]
    w4 = [(2,0), (3,0), (2,1), (3,1)]
    strassen_block(y4, z4, w4)
    
    # Block 5: A_L (cols 0,1) * B_15 (rows 0,1; col 4) - 2x2x1: rank 4
    U[u_idx(0,0)][col] = 1
    V[v_idx(0,4)][col] = 1
    W[w_idx(4,0)][col] = 1
    col += 1
    
    U[u_idx(0,1)][col] = 1
    V[v_idx(1,4)][col] = 1
    W[w_idx(4,0)][col] = 1
    col += 1
    
    U[u_idx(1,0)][col] = 1
    V[v_idx(0,4)][col] = 1
    W[w_idx(4,1)][col] = 1
    col += 1
    
    U[u_idx(1,1)][col] = 1
    V[v_idx(1,4)][col] = 1
    W[w_idx(4,1)][col] = 1
    col += 1
    
    # Block 6: A_R (cols 2,3) * B_25 (rows 2,3; col 4) - 2x2x1: rank 4
    U[u_idx(0,2)][col] = 1
    V[v_idx(2,4)][col] = 1
    W[w_idx(4,0)][col] = 1
    col += 1
    
    U[u_idx(0,3)][col] = 1
    V[v_idx(3,4)][col] = 1
    W[w_idx(4,0)][col] = 1
    col += 1
    
    U[u_idx(1,2)][col] = 1
    V[v_idx(2,4)][col] = 1
    W[w_idx(4,1)][col] = 1
    col += 1
    
    U[u_idx(1,3)][col] = 1
    V[v_idx(3,4)][col] = 1
    W[w_idx(4,1)][col] = 1
    col += 1
    
    # Now try to find linear dependencies among the 36 terms to reduce rank
    # We'll check if any term can be expressed as a linear combination of others
    # This is equivalent to finding the rank of the 3D tensor's "slice" matrices
    
    # Strategy: For each term r, check if the 3D vector (U[:,r], V[:,r], W[:,r])
    # lies in the span of the other 35 terms. We can do this by checking if
    # the matrix formed by all terms has a column that can be removed.
    
    # A more practical approach: try to find a term that is a linear combination
    # of others by checking if removing it still gives a valid decomposition.
    
    # For efficiency, we'll check subsets of terms that might be dependent.
    # The Strassen blocks are independent by design, but the schoolbook blocks
    # for the 5th column might have dependencies with the Strassen blocks.
    
    # Let's check if we can merge some of the schoolbook terms with Strassen terms.
    # Actually, let's try a systematic approach: check if any single term is redundant.
    
    def check_decomposition(U_mat, V_mat, W_mat, rank):
        """Check if the decomposition is valid."""
        if rank > 36:
            return False
        try:
            D = np.einsum('ar,br,cr->abc', U_mat, V_mat, W_mat, optimize=True)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if abs(D[i*m+j, j*p+k, k*n+i] - 1.0) > 1e-9:
                            return False
        except Exception:
            return False
        return True
    
    # Try removing each term one at a time
    best_rank = rank_initial
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Check single term removal
    for r_to_remove in range(rank_initial):
        if time.time() > deadline:
            break
        U_test = U.copy()
        V_test = V.copy()
        W_test = W.copy()
        
        # Remove term r_to_remove
        U_test = np.delete(U_test, r_to_remove, axis=1)
        V_test = np.delete(V_test, r_to_remove, axis=1)
        W_test = np.delete(W_test, r_to_remove, axis=1)
        
        if check_decomposition(U_test, V_test, W_test, rank_initial - 1):
            best_rank = rank_initial - 1
            best_U, best_V, best_W = U_test, V_test, W_test
            break
    
    # If no single removal works, try pairs
    if best_rank == rank_initial and time.time() < deadline:
        for r1, r2 in itertools.combinations(range(rank_initial), 2):
            if time.time() > deadline:
                break
            U_test = U.copy()
            V_test = V.copy()
            W_test = W.copy()
            
            # Remove terms r1 and r2
            U_test = np.delete(U_test, [r1, r2], axis=1)
            V_test = np.delete(V_test, [r1, r2], axis=1)
            W_test = np.delete(W_test, [r1, r2], axis=1)
            
            if check_decomposition(U_test, V_test, W_test, rank_initial - 2):
                best_rank = rank_initial - 2
                best_U, best_V, best_W = U_test, V_test, W_test
                break
    
    # Try triples if needed
    if best_rank == rank_initial and time.time() < deadline - 10:
        for r1, r2, r3 in itertools.combinations(range(rank_initial), 3):
            if time.time() > deadline:
                break
            U_test = U.copy()
            V_test = V.copy()
            W_test = W.copy()
            
            # Remove terms r1, r2, r3
            U_test = np.delete(U_test, [r1, r2, r3], axis=1)
            V_test = np.delete(V_test, [r1, r2, r3], axis=1)
            W_test = np.delete(W_test, [r1, r2, r3], axis=1)
            
            if check_decomposition(U_test, V_test, W_test, rank_initial - 3):
                best_rank = rank_initial - 3
                best_U, best_V, best_W = U_test, V_test, W_test
                break
    
    U, V, W = best_U, best_V, best_W
    
    # Convert to lists of ints/floats
    U_list = [[int(val) if val == int(val) else val for val in row] for row in U.tolist()]
    V_list = [[int(val) if val == int(val) else val for val in row] for row in V.tolist()]
    W_list = [[int(val) if val == int(val) else val for val in row] for row in W.tolist()]
    
    return {"U": U_list, "V": V_list, "W": W_list}
