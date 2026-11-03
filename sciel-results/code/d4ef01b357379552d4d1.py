import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = 2, 4, 6
    deadline = time.time() + 135.0

    # Build the full multiplication tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    def check_decomposition(U, V, W):
        """Check if sum_r U[:,r] * V[:,r] * W[:,r] == T (up to tolerance)"""
        reconstructed = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.allclose(reconstructed, T, atol=1e-9)

    def to_lists(U, V, W):
        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}

    # Strategy: Use rank-20 decomposition for 2x3x3 matrix multiplication
    # Partition the 4x6 right matrix into two 4x3 blocks:
    #   A (2x4) * B (4x6) = A * [B1 B2] where B1, B2 are 4x3
    #   = [A*B1, A*B2]
    # Each A*B1 and A*B2 is a 2x3x3 multiplication.
    # If rank(2x3x3) = 20, then total rank = 40.
    #
    # But wait - we need to be careful. The tensor for 2x3x3 is different from
    # what we need. Let me think about this more carefully.
    #
    # Actually, the standard approach:
    # 2x4 * 4x6: We can view this as two 2x2*2x6 or 2x4*2x3 blocks, etc.
    #
    # Let me use the block Strassen approach but try to find a better decomposition.
    #
    # Alternative: Use the known result that rank(2x3x3) <= 20 (from literature).
    # If we split the 4x6 into two 4x3 blocks, we get two 2x3x3 multiplications.
    # But the tensor structure needs to match exactly.
    #
    # Let me reconsider. The matrix multiplication tensor for (n,m,p) has entries
    # T[i*m+j, j*p+k, k*n+i] = 1.
    #
    # For 2x4x6: n=2, m=4, p=6
    # If we split m=4 into m1=2, m2=2:
    #   A (2x4) = [A1 A2] where A1, A2 are 2x2
    #   B (4x6) = [B1; B2] where B1, B2 are 2x6
    #   A*B = A1*B1 + A2*B2
    #   Each term is 2x2*2x6, which can be further decomposed.
    #
    # For 2x2*2x6: split p=6 into p1=3, p2=3:
    #   B1 (2x6) = [B1a B1b] where B1a, B1b are 2x3
    #   A1*B1 = [A1*B1a, A1*B1b]
    #   Each term is 2x2*2x3.
    #
    # So 2x4x6 = 4 blocks of 2x2*2x3.
    #
    # What's the rank of 2x2*2x3?
    # This is equivalent to multiplying a 2x2 matrix by a 2x3 matrix.
    # The tensor has dimensions (2*2, 2*3, 3*2) = (4, 6, 6).
    #
    # Actually, let me just use the known rank-42 block Strassen and try to improve it.
    #
    # Let me try a different approach: direct construction using a known low-rank 
    # decomposition for 2x3x3.
    #
    # Known: rank(2x3x3) <= 20 (Landsberg, 1995)
    # If we can use this, then:
    # 2x4x6: split m=4 into 2+2, giving two 2x2x6 multiplications
    # 2x2x6: split p=6 into 3+3, giving two 2x2x3 multiplications
    # 2x2x3: this is a 2x2 matrix times a 2x3 matrix
    #
    # Hmm, this is getting complicated. Let me just use the rank-42 decomposition
    # and try to find linear dependencies to reduce the rank.
    
    def make_rank42():
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
        
        return U, V, W
    
    U, V, W = make_rank42()
    
    # Verify the rank-42 decomposition is correct
    if not check_decomposition(U, V, W):
        # Fallback: schoolbook decomposition (rank = n*m*p = 48)
        U = np.zeros((n*m, n*m*p))
        V = np.zeros((m*p, n*m*p))
        W = np.zeros((p*n, n*m*p))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, idx] = 1.0
                    V[j*p+k, idx] = 1.0
                    W[k*n+i, idx] = 1.0
                    idx += 1
        if not check_decomposition(U, V, W):
            # This should never happen, but just in case
            return to_lists(U, V, W)
    
    # Try to reduce rank by finding linear dependencies
    # Use a more systematic approach: check if any subset of columns is linearly dependent
    # in a way that preserves the tensor decomposition.
    
    def try_reduce_rank(U, V, W):
        """Try to reduce the rank by finding and eliminating redundant columns."""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        rank = U.shape[1]
        
        for _ in range(50):  # Try up to 50 eliminations
            if time.time() > deadline - 15:
                break
            
            eliminated = False
            for r in range(rank-1, -1, -1):
                if time.time() > deadline - 15:
                    break
                
                # Check if column r is zero
                if np.allclose(U[:, r], 0) and np.allclose(V[:, r], 0) and np.allclose(W[:, r], 0):
                    U = np.delete(U, r, axis=1)
                    V = np.delete(V, r, axis=1)
                    W = np.delete(W, r, axis=1)
                    rank -= 1
                    eliminated = True
                    break
                
                # Check if column r is identical to another column
                for s in range(r):
                    if np.allclose(U[:, r], U[:, s]) and np.allclose(V[:, r], V[:, s]) and np.allclose(W[:, r], W[:, s]):
                        # Column r is identical to s, remove r
                        U = np.delete(U, r, axis=1)
                        V = np.delete(V, r, axis=1)
                        W = np.delete(W, r, axis=1)
                        rank -= 1
                        eliminated = True
                        break
                if eliminated:
                    break
            
            if not eliminated:
                break
        
        return U, V, W
    
    # Apply reduction
    U_reduced, V_reduced, W_reduced = try_reduce_rank(U, V, W)
    
    # Verify the reduced decomposition
    if check_decomposition(U_reduced, V_reduced, W_reduced):
        U, V, W = U_reduced, V_reduced, W_reduced
    
    return to_lists(U, V, W)
