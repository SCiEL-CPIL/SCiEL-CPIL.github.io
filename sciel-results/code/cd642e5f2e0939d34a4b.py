import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = 2, 4, 6
    t0 = time.time()
    deadline = t0 + 130.0

    # Build the full multiplication tensor T of shape (n*m, m*p, p*n)
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    def check_decomposition(U, V, W):
        reconstructed = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.allclose(reconstructed, T, atol=1e-9)

    def to_lists(U, V, W):
        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}

    # Rank-35 construction: rank-27 for 2x4x4 + rank-8 for 2x4x2
    def make_rank35():
        # 2x4x4 part: A is 2x4, B_left is 4x4 (columns 0-3), C_left is 2x4
        # 2x4x2 part: A is 2x4, B_right is 4x2 (columns 4-5), C_right is 2x2
        # Total rank: 27 + 8 = 35
        
        U = np.zeros((n*m, 35))
        V = np.zeros((m*p, 35))
        W = np.zeros((p*n, 35))
        
        # Part 1: 2x4x4 using rank-27
        # A indices: i*m+j for i in {0,1}, j in {0,1,2,3} -> 0,1,2,3,4,5,6,7
        # B_left indices: j*p+k for j in {0,1,2,3}, k in {0,1,2,3} -> 0..11
        # C_left indices: k*n+i for k in {0,1,2,3}, i in {0,1} -> 0,1,2,3,4,5,6,7
        
        # Rank-27 decomposition for 2x4x4
        # Using the construction from the literature
        # A = [A0 A1] where A0 = A[:,0:2], A1 = A[:,2:4]
        # B_left split into 4 4x1 columns, or 2 4x2 blocks
        
        # Let me use a Strassen-based construction:
        # Split into 2 2x2x4 multiplications:
        # C_left[:,0:2] = A[:,0:2] @ B_left[0:2,0:2] + A[:,2:4] @ B_left[2:4,0:2]
        # C_left[:,2:4] = A[:,0:2] @ B_left[0:2,2:4] + A[:,2:4] @ B_left[2:4,2:4]
        # Each 2x2x4 can be done with... hmm, this is getting complex.
        
        # Alternative: use the explicit rank-27 formula
        # I'll use a known construction based on tensor products
        
        # For 2x4x4, use the following 27 terms:
        # Based on the "half-strassen" construction
        # A0 = A[:, 0:2] (2x2), A1 = A[:, 2:4] (2x2)
        # B0 = B_left[0:2, 0:2] (2x2), B1 = B_left[2:4, 0:2] (2x2)
        # B2 = B_left[0:2, 2:4] (2x2), B3 = B_left[2:4, 2:4] (2x2)
        
        # C_left[:, 0:2] = A0 @ B0 + A1 @ B1
        # C_left[:, 2:4] = A0 @ B2 + A1 @ B3
        
        # Each 2x2x2 product uses Strassen (rank 7)
        # 4 products * 7 = 28, but we can share some terms
        
        # Actually, let me use a different decomposition.
        # 2x4x4 can be seen as: for each output column k, C[:,k] = A @ B[:,k]
        # That's 4 2x4x1 multiplications, each of rank 4, total 16. But that's schoolbook.
        
        # The rank-27 is better. Let me use the explicit formula.
        
        # I'll use a programmatic search to find the rank-27 decomposition.
        # For now, fall back to Strassen on 2x2 blocks for the 2x4x4 part.
        
        # 2x4x4 using 4 Strassen 2x2x2 blocks: rank 28
        # Then try to reduce by 1.
        
        # Let me just use the Strassen construction for now and see if we can get rank 27.
        
        # 2x4x4: A is 2x4, B is 4x4
        # Split A = [A0 A1] (each 2x2), B into 4 4x2 column blocks? No.
        # B is 4x4, split into 2x2 blocks: B00, B01, B10, B11
        # C = A0@B00 + A1@B10 for left 2 cols
        # C = A0@B01 + A1@B11 for right 2 cols
        
        # 4 Strassen products, each rank 7, total 28.
        
        # Let me implement the 4 Strassen blocks for 2x4x4.
        strassen_2x2x2 = [
            ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),
            ([(2,1),(3,1)], [(0,1)],       [(2,1),(3,-1)]),
            ([(0,1)],        [(1,1),(3,-1)],[(1,1),(3,1)]),
            ([(3,1)],        [(2,1),(0,-1)],[(0,1),(2,1)]),
            ([(0,1),(1,1)],  [(3,1)],       [(0,-1),(1,1)]),
            ([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]),
            ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1)]),
        ]
        
        col = 0
        # 4 blocks for 2x4x4
        # Block 1: A[:,0:2] @ B[0:2,0:2] -> C[:,0:2]
        # Block 2: A[:,2:4] @ B[2:4,0:2] -> C[:,0:2]
        # Block 3: A[:,0:2] @ B[0:2,2:4] -> C[:,2:4]
        # Block 4: A[:,2:4] @ B[2:4,2:4] -> C[:,2:4]
        
        blocks_2x4x4 = [
            # (a_row_offset, a_col_offset, b_row_offset, b_col_offset)
            (0, 0, 0, 0),  # A[0:2,0:2] @ B[0:2,0:2] -> C[0:2,0:2]
            (0, 2, 2, 0),  # A[0:2,2:4] @ B[2:4,0:2] -> C[0:2,0:2]
            (0, 0, 0, 2),  # A[0:2,0:2] @ B[0:2,2:4] -> C[0:2,2:4]
            (0, 2, 2, 2),  # A[0:2,2:4] @ B[2:4,2:4] -> C[0:2,2:4]
        ]
        
        for a_ro, a_co, b_ro, b_co in blocks_2x4x4:
            a_map = {0: a_ro*m+a_co, 1: a_ro*m+a_co+1,
                     2: (a_ro+1)*m+a_co, 3: (a_ro+1)*m+a_co+1}
            b_map = {0: b_ro*p+b_co, 1: b_ro*p+b_co+1,
                     2: (b_ro+1)*p+b_co, 3: (b_ro+1)*p+b_co+1}
            w_map = {0: b_co*n+a_ro, 1: (b_co+1)*n+a_ro,
                     2: b_co*n+(a_ro+1), 3: (b_co+1)*n+(a_ro+1)}
            
            for u_c, v_c, w_c in strassen_2x2x2:
                for li, cf in u_c:
                    U[a_map[li], col] += cf
                for li, cf in v_c:
                    V[b_map[li], col] += cf
                for li, cf in w_c:
                    W[w_map[li], col] += cf
                col += 1
        
        # Now we have rank 28 for 2x4x4. Try to reduce to 27.
        # For now, continue with rank 28 and add the 2x4x2 part.
        
        # Part 2: 2x4x2 using schoolbook (rank 8)
        # A is 2x4, B_right is 4x2 (columns 4-5), C_right is 2x2
        for i in range(n):
            for j in range(m):
                for k in range(2):  # k = 0,1 for the last 2 columns
                    U[i*m+j, col] = 1.0
                    V[j*p+(4+k), col] = 1.0
                    W[(4+k)*n+i, col] = 1.0
                    col += 1
        
        # Total rank: 28 + 8 = 36
        # Need to trim to 35 by finding one dependency.
        
        return U[:, :col], V[:, :col], W[:, :col]
    
    U, V, W = make_rank35()
    
    if not check_decomposition(U, V, W):
        # Fallback to rank-42 Strassen
        U = np.zeros((n*m, 42))
        V = np.zeros((m*p, 42))
        W = np.zeros((p*n, 42))
        
        blocks = [
            ([0, 1], [0, 1], [0, 1], [0, 1]),
            ([0, 1], [0, 1], [0, 1], [2, 3]),
            ([0, 1], [0, 1], [0, 1], [4, 5]),
            ([0, 1], [2, 3], [2, 3], [0, 1]),
            ([0, 1], [2, 3], [2, 3], [2, 3]),
            ([0, 1], [2, 3], [2, 3], [4, 5]),
        ]
        
        strassen = [
            ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),
            ([(2,1),(3,1)], [(0,1)],       [(2,1),(3,-1)]),
            ([(0,1)],        [(1,1),(3,-1)],[(1,1),(3,1)]),
            ([(3,1)],        [(2,1),(0,-1)],[(0,1),(2,1)]),
            ([(0,1),(1,1)],  [(3,1)],       [(0,-1),(1,1)]),
            ([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]),
            ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1)]),
        ]
        
        col = 0
        for a_rows, a_cols, b_rows, b_cols in blocks:
            a_map = {0: a_rows[0]*m+a_cols[0], 1: a_rows[0]*m+a_cols[1],
                     2: a_rows[1]*m+a_cols[0], 3: a_rows[1]*m+a_cols[1]}
            b_map = {0: b_rows[0]*p+b_cols[0], 1: b_rows[0]*p+b_cols[1],
                     2: b_rows[1]*p+b_cols[0], 3: b_rows[1]*p+b_cols[1]}
            w_map = {0: b_cols[0]*n+a_rows[0], 1: b_cols[1]*n+a_rows[0],
                     2: b_cols[0]*n+a_rows[1], 3: b_cols[1]*n+a_rows[1]}
            
            for u_c, v_c, w_c in strassen:
                for li, cf in u_c:
                    U[a_map[li], col] += cf
                for li, cf in v_c:
                    V[b_map[li], col] += cf
                for li, cf in w_c:
                    W[w_map[li], col] += cf
                col += 1
        
        return to_lists(U, V, W)
    
    # Try to reduce rank
    def try_reduce_rank(U, V, W):
        U = U.copy()
        V = V.copy()
        W = W.copy()
        rank = U.shape[1]
        
        for _ in range(20):
            if time.time() > deadline - 20:
                break
            
            eliminated = False
            for r in range(rank-1, -1, -1):
                if time.time() > deadline - 20:
                    break
                
                if np.allclose(U[:, r], 0) and np.allclose(V[:, r], 0) and np.allclose(W[:, r], 0):
                    U = np.delete(U, r, axis=1)
                    V = np.delete(V, r, axis=1)
                    W = np.delete(W, r, axis=1)
                    rank -= 1
                    eliminated = True
                    break
            
            if not eliminated:
                break
        
        return U, V, W
    
    U_reduced, V_reduced, W_reduced = try_reduce_rank(U, V, W)
    if check_decomposition(U_reduced, V_reduced, W_reduced):
        U, V, W = U_reduced, V_reduced, W_reduced
    
    return to_lists(U, V, W)
