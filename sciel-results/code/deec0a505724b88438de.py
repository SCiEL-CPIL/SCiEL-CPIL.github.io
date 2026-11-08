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
    
    def try_reduce_rank(U, V, W, rank):
        """Try to reduce rank by finding linearly dependent rank-1 terms."""
        # Check for zero columns
        for r in range(rank):
            if np.allclose(U[:, r], 0) and np.allclose(V[:, r], 0) and np.allclose(W[:, r], 0):
                keep = [c for c in range(rank) if c != r]
                return try_reduce_rank(U[:, keep], V[:, keep], W[:, keep], rank - 1)
        
        # Check for duplicate columns (up to scaling)
        for r in range(rank):
            for s in range(r+1, rank):
                if not np.allclose(U[:, r], 0) and not np.allclose(V[:, r], 0) and not np.allclose(W[:, r], 0):
                    if not np.allclose(U[:, s], 0) and not np.allclose(V[:, s], 0) and not np.allclose(W[:, s], 0):
                        # Check proportionality
                        try:
                            alpha_u = U[0, s] / U[0, r] if U[0, r] != 0 else None
                            if alpha_u is not None:
                                if (np.allclose(U[:, s], alpha_u * U[:, r], atol=1e-10) and
                                    np.allclose(V[:, s], alpha_u * V[:, r], atol=1e-10) and
                                    np.allclose(W[:, s], alpha_u * W[:, r], atol=1e-10)):
                                    keep = [c for c in range(rank) if c != s]
                                    return try_reduce_rank(U[:, keep], V[:, keep], W[:, keep], rank - 1)
                        except:
                            pass
        
        return U, V, W, rank
    
    # Build the rank-54 decomposition (original approach)
    rank = 54
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for j0, j1 in [(0,1), (2,3)]:
        for k0, k1 in [(0,1), (2,3), (4,5)]:
            col = strassen_2x2x2(U, V, W, col, 0, 1, j0, j1, k0, k1)
    col = schoolbook_block(U, V, W, col, [0,1], [4], range(p))
    
    # Verify
    if not validate(U, V, W, rank):
        # Fallback to pure schoolbook
        rank = n*m*p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = schoolbook_block(U, V, W, 0, range(n), range(m), range(p))
    
    best_U, best_V, best_W, best_rank = U.copy(), V.copy(), W.copy(), rank
    
    # Try alternative block partitions
    if time.time() < deadline - 60:
        # Partition: m={0,1,2,3},{4}, p={0,1,2},{3,4,5}
        # 2x4x3 + 2x1x3 + 2x4x3 + 2x1x3
        # Use Strassen for 2x2x2 sub-blocks within 2x4x3
        alt_rank = 60
        alt_U = np.zeros((n*m, alt_rank))
        alt_V = np.zeros((m*p, alt_rank))
        alt_W = np.zeros((p*n, alt_rank))
        col = 0
        # m={0,1,2,3}, p={0,1,2}: split m into {0,1} and {2,3}
        for k0, k1 in [(0,1), (1,2)]:  # 2x2x2 blocks for p={0,1,2}
            if k0 == 0 and k1 == 1:
                col = strassen_2x2x2(alt_U, alt_V, alt_W, col, 0, 1, 0, 1, k0, k1)
                col = strassen_2x2x2(alt_U, alt_V, alt_W, col, 0, 1, 2, 3, k0, k1)
            else:
                # For k1=2, we need 2x2x1 blocks
                col = schoolbook_block(alt_U, alt_V, alt_W, col, [0,1], [0,1], [2])
                col = schoolbook_block(alt_U, alt_V, alt_W, col, [0,1], [2,3], [2])
        # m={4}, p={0,1,2}: schoolbook
        col = schoolbook_block(alt_U, alt_V, alt_W, col, [0,1], [4], range(3))
        # m={0,1,2,3}, p={3,4,5}: similar
        for k0, k1 in [(3,4), (4,5)]:
            col = strassen_2x2x2(alt_U, alt_V, alt_W, col, 0, 1, 0, 1, k0, k1)
            col = strassen_2x2x2(alt_U, alt_V, alt_W, col, 0, 1, 2, 3, k0, k1)
        # m={4}, p={3,4,5}: schoolbook
        col = schoolbook_block(alt_U, alt_V, alt_W, col, [0,1], [4], range(3,6))
        
        alt_rank = col
        if validate(alt_U, alt_V, alt_W, alt_rank) and alt_rank < best_rank:
            best_U, best_V, best_W, best_rank = alt_U.copy(), alt_V.copy(), alt_W.copy(), alt_rank
    
    # Try to reduce rank using linear algebra
    if time.time() < deadline - 60:
        U_red, V_red, W_red, rank_red = try_reduce_rank(U.copy(), V.copy(), W.copy(), rank)
        if rank_red < best_rank and validate(U_red, V_red, W_red, rank_red):
            best_U, best_V, best_W, best_rank = U_red, V_red, W_red, rank_red
    
    # Try a different approach: use the fact that we can combine some terms
    if time.time() < deadline - 30:
        # Try to find if any 2x2x2 Strassen block can be combined with adjacent blocks
        # by looking for shared structure
        
        # Alternative: try m={0,1,2},{3,4} and p={0,1,2,3},{4,5}
        alt_rank = 60
        alt_U = np.zeros((n*m, alt_rank))
        alt_V = np.zeros((m*p, alt_rank))
        alt_W = np.zeros((p*n, alt_rank))
        col = 0
        # m={0,1,2}, p={0,1,2,3}: 2x3x4
        # Split into 2x2x4 + 2x1x4
        for k0, k1 in [(0,1), (2,3)]:
            col = strassen_2x2x2(alt_U, alt_V, alt_W, col, 0, 1, 0, 1, k0, k1)
        col = schoolbook_block(alt_U, alt_V, alt_W, col, [0,1], [2], range(4))
        # m={3,4}, p={0,1,2,3}: 2x2x4
        for k0, k1 in [(0,1), (2,3)]:
            col = strassen_2x2x2(alt_U, alt_V, alt_W, col, 0, 1, 3, 4, k0, k1)
        # m={0,1,2}, p={4,5}: 2x3x2
        col = strassen_2x2x2(alt_U, alt_V, alt_W, col, 0, 1, 0, 1, 4, 5)
        col = schoolbook_block(alt_U, alt_V, alt_W, col, [0,1], [2], [4,5])
        # m={3,4}, p={4,5}: 2x2x2
        col = strassen_2x2x2(alt_U, alt_V, alt_W, col, 0, 1, 3, 4, 4, 5)
        
        alt_rank = col
        if validate(alt_U, alt_V, alt_W, alt_rank) and alt_rank < best_rank:
            best_U, best_V, best_W, best_rank = alt_U.copy(), alt_V.copy(), alt_W.copy(), alt_rank
    
    # Final: try to see if we can do better with a different block structure
    if time.time() < deadline - 20:
        # Try: split m as {0,1},{2,3,4}, p as {0,1,2},{3,4,5}
        # 2x2x3 + 2x3x3 + 2x2x3 + 2x3x3
        # R(2,2,3) = ? At most 2*2*3=12, possibly 11
        # R(2,3,3) = ? At most 2*3*3=18
        # This might not be better than 54
        
        # Try to use the known result that R(2,5,6) might be less than 54
        # by checking if there's a rank-53 or lower decomposition
        
        # For now, return the best solution found
        pass
    
    # Return the best solution found
    U_list = [[int(U[i, c]) if abs(U[i, c] - round(U[i, c])) < 1e-9 else float(U[i, c]) for c in range(best_rank)] for i in range(n*m)]
    V_list = [[int(V[i, c]) if abs(V[i, c] - round(V[i, c])) < 1e-9 else float(V[i, c]) for c in range(best_rank)] for i in range(m*p)]
    W_list = [[int(W[i, c]) if abs(W[i, c] - round(W[i, c])) < 1e-9 else float(W[i, c]) for c in range(best_rank)] for i in range(p*n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
