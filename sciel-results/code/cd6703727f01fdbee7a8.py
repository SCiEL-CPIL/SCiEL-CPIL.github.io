"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 5, 6
    deadline = time.time() + 140  # Safety margin

    def validate(U, V, W, rank):
        """Check if U, V, W correctly decompose the matrix multiplication tensor."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def strassen_2x2x2(U, V, W, col, i0, i1, j0, j1, k0, k1):
        """Add Strassen decomposition for 2x2x2 block. Returns new col."""
        # P1 = (A[i0,j0]+A[i1,j1]) * (B[j0,k0]+B[j1,k1])
        U[i0*m+j0, col] += 1; U[i1*m+j1, col] += 1
        V[j0*p+k0, col] += 1; V[j1*p+k1, col] += 1
        W[k0*n+i0, col] += 1; W[k1*n+i1, col] += 1
        col += 1
        # P2 = (A[i1,j0]+A[i1,j1]) * B[j0,k0]
        U[i1*m+j0, col] += 1; U[i1*m+j1, col] += 1
        V[j0*p+k0, col] += 1
        W[k0*n+i1, col] += 1; W[k1*n+i1, col] -= 1
        col += 1
        # P3 = A[i0,j0] * (B[j0,k1]-B[j1,k1])
        U[i0*m+j0, col] += 1
        V[j0*p+k1, col] += 1; V[j1*p+k1, col] -= 1
        W[k1*n+i0, col] += 1; W[k1*n+i1, col] += 1
        col += 1
        # P4 = A[i1,j1] * (B[j1,k0]-B[j0,k0])
        U[i1*m+j1, col] += 1
        V[j1*p+k0, col] += 1; V[j0*p+k0, col] -= 1
        W[k0*n+i0, col] += 1; W[k0*n+i1, col] += 1
        col += 1
        # P5 = (A[i0,j0]+A[i0,j1]) * B[j1,k1]
        U[i0*m+j0, col] += 1; U[i0*m+j1, col] += 1
        V[j1*p+k1, col] += 1
        W[k0*n+i0, col] -= 1; W[k1*n+i0, col] += 1
        col += 1
        # P6 = (A[i1,j0]-A[i0,j0]) * (B[j0,k0]+B[j0,k1])
        U[i1*m+j0, col] += 1; U[i0*m+j0, col] -= 1
        V[j0*p+k0, col] += 1; V[j0*p+k1, col] += 1
        W[k1*n+i1, col] += 1
        col += 1
        # P7 = (A[i0,j1]-A[i1,j1]) * (B[j1,k0]+B[j1,k1])
        U[i0*m+j1, col] += 1; U[i1*m+j1, col] -= 1
        V[j1*p+k0, col] += 1; V[j1*p+k1, col] += 1
        W[k0*n+i0, col] += 1
        col += 1
        return col

    def schoolbook_block(U, V, W, col, i_range, j_range, k_range):
        """Add schoolbook terms for a block. Returns new col."""
        for i in i_range:
            for j in j_range:
                for k in k_range:
                    U[i*m+j, col] += 1
                    V[j*p+k, col] += 1
                    W[k*n+i, col] += 1
                    col += 1
        return col

    def build_base_decomposition(j_pairs, k_pairs, remaining_j):
        """Build a base decomposition using Strassen on 2x2x2 blocks + schoolbook for rest."""
        # Estimate rank
        rank = 0
        for j0, j1 in j_pairs:
            for k0, k1 in k_pairs:
                rank += 7
        for j in remaining_j:
            rank += 2 * p  # schoolbook for each k
        # Also need to account for any j in j_pairs that might overlap
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for j0, j1 in j_pairs:
            for k0, k1 in k_pairs:
                col = strassen_2x2x2(U, V, W, col, 0, 1, j0, j1, k0, k1)
        for j in remaining_j:
            for k in range(p):
                U[0*m+j, col] += 1
                V[j*p+k, col] += 1
                W[k*n+0, col] += 1
                col += 1
                U[1*m+j, col] += 1
                V[j*p+k, col] += 1
                W[k*n+1, col] += 1
                col += 1
        return U, V, W, rank

    def try_rank_reduction(U, V, W, rank):
        """
        Try to reduce the rank by finding linear dependencies.
        The tensor T is fixed. If we have rank r decomposition, we can try
        to find a rank r-1 decomposition by checking if one term is in the
        span of the others in a specific way.
        
        Approach: For each column c, check if we can express the contribution
        of column c as a linear combination of other columns, then eliminate it.
        
        More practically: use the fact that the tensor has 60 non-zero entries.
        Build the constraint matrix and check if we can satisfy it with fewer terms.
        """
        # The tensor T[i*m+j, j*p+k, k*n+i] = 1 for all valid (i,j,k)
        # We need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        
        # For rank reduction, we try to find a null vector in the "product space"
        # This is equivalent to checking if the set of rank-1 tensors is linearly dependent
        # in a way that preserves the sum.
        
        # Practical approach: try removing each column and see if the remaining
        # columns can still satisfy the constraints (with adjusted values).
        # This is a linear system for the removed column's contribution.
        
        best_U, best_V, best_W, best_rank = U.copy(), V.copy(), W.copy(), rank
        
        # Try to eliminate columns one by one
        for c_remove in range(rank):
            if time.time() > deadline - 30:
                break
            # Remove column c_remove
            r_new = rank - 1
            U_new = np.delete(U, c_remove, axis=1)
            V_new = np.delete(V, c_remove, axis=1)
            W_new = np.delete(W, c_remove, axis=1)
            
            # Check if the remaining decomposition still works
            # (It won't, but we can try to adjust)
            # For now, just check if the sum is close to 1
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.sum(U_new[i*m+j, :] * V_new[j*p+k, :] * W_new[k*n+i, :])
                        if abs(s - 1.0) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if valid and r_new < best_rank:
                best_U, best_V, best_W, best_rank = U_new, V_new, W_new, r_new
        
        return best_U, best_V, best_W, best_rank

    # Strategy 1: Standard Strassen split
    def try_strassen_split():
        j_pairs = [(0,1), (2,3)]
        k_pairs = [(0,1), (2,3), (4,5)]
        remaining_j = [4]
        return build_base_decomposition(j_pairs, k_pairs, remaining_j)
    
    U, V, W, rank = try_strassen_split()
    if not validate(U, V, W, rank):
        rank = n*m*p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = schoolbook_block(U, V, W, 0, range(n), range(m), range(p))
    
    best_U, best_V, best_W, best_rank = U.copy(), V.copy(), W.copy(), rank
    
    # Strategy 2: Different block splits
    if time.time() < deadline - 20:
        # m=1+4, p=2+4
        def try_split_m14_p24():
            rank = 2*1*2 + 2*1*4 + 2*4*2 + 2*14
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            col = 0
            col = schoolbook_block(U, V, W, col, [0,1], [0], [0,1])
            col = schoolbook_block(U, V, W, col, [0,1], [0], [2,3,4,5])
            col = schoolbook_block(U, V, W, col, [0,1], [1,2,3,4], [0,1])
            for j0, j1 in [(1,2), (3,4)]:
                for k0, k1 in [(2,3), (4,5)]:
                    col = strassen_2x2x2(U, V, W, col, 0, 1, j0, j1, k0, k1)
            return U, V, W, rank
        
        U2, V2, W2, rank2 = try_split_m14_p24()
        if validate(U2, V2, W2, rank2) and rank2 < best_rank:
            best_U, best_V, best_W, best_rank = U2, V2, W2, rank2
    
    # Strategy 3: Try rank reduction on the best decomposition found so far
    if time.time() < deadline - 15:
        U3, V3, W3, rank3 = try_rank_reduction(best_U, best_V, best_W, best_rank)
        if rank3 < best_rank:
            best_U, best_V, best_W, best_rank = U3, V3, W3, rank3
    
    # Strategy 4: Try more aggressive block splits
    if time.time() < deadline - 10:
        # Try m=2+3, p=3+3 with mixed strategies
        def try_split_m23_p33():
            # 2x2x3: 2x2x2 (Strassen=7) + 2x2x1 (schoolbook=4) = 11
            # 2x2x3: 11
            # 2x3x3: schoolbook = 18
            # 2x3x3: 18
            # Total: 58 (worse than 54)
            # Try 2x3x3 with better method
            # R(2,3,3) <= R(2,2,3)+R(2,1,3) = 11+6 = 17
            # So total: 11+11+17+17 = 56 (still worse)
            return None
        
        # Try m=2+1+2, p=2+2+2 with shared structure
        def try_m212_p222():
            # j=0,1: Strassen with k=0,1 -> 7; k=2,3 -> 7; k=4,5 -> 7 => 21
            # j=2: schoolbook with all k => 12
            # j=3,4: Strassen with k=0,1 -> 7; k=2,3 -> 7; k=4,5 -> 7 => 21
            # Total: 54 (same as before)
            # Try to share terms between j=2 and the Strassen blocks
            # This requires a more sophisticated approach
            rank = 54
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            col = 0
            for j0, j1 in [(0,1), (3,4)]:
                for k0, k1 in [(0,1), (2,3), (4,5)]:
                    col = strassen_2x2x2(U, V, W, col, 0, 1, j0, j1, k0, k1)
            col = schoolbook_block(U, V, W, col, [0,1], [2], range(p))
            return U, V, W, rank
        
        U4, V4, W4, rank4 = try_m212_p222()
        if validate(U4, V4, W4, rank4) and rank4 < best_rank:
            best_U, best_V, best_W, best_rank = U4, V4, W4, rank4
    
    # Convert to lists for the return interface
    U_list = [[int(best_U[i, c]) if abs(best_U[i, c] - round(best_U[i, c])) < 1e-9 else float(best_U[i, c]) for c in range(best_rank)] for i in range(n*m)]
    V_list = [[int(best_V[i, c]) if abs(best_V[i, c] - round(best_V[i, c])) < 1e-9 else float(best_V[i, c]) for c in range(best_rank)] for i in range(m*p)]
    W_list = [[int(best_W[i, c]) if abs(best_W[i, c] - round(best_W[i, c])) < 1e-9 else float(best_W[i, c]) for c in range(best_rank)] for i in range(p*n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
