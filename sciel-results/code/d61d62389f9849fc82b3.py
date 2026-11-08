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

    def build_strassen_split(j_pairs, k_pairs, remaining_j):
        """Build a decomposition using Strassen on 2x2x2 blocks + schoolbook for rest."""
        rank = 0
        for j0, j1 in j_pairs:
            for k0, k1 in k_pairs:
                rank += 7
        for j in remaining_j:
            rank += 2 * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for j0, j1 in j_pairs:
            for k0, k1 in k_pairs:
                col = strassen_2x2x2(U, V, W, col, 0, 1, j0, j1, k0, k1)
        for j in remaining_j:
            col = schoolbook_block(U, V, W, col, [0,1], [j], range(p))
        return U, V, W, rank

    def try_rank_reduction(U, V, W, rank):
        """
        Try to reduce the rank by finding linear dependencies.
        For each column c, check if removing it still satisfies constraints.
        """
        best_U, best_V, best_W, best_rank = U.copy(), V.copy(), W.copy(), rank
        
        for c_remove in range(rank):
            if time.time() > deadline - 30:
                break
            r_new = rank - 1
            U_new = np.delete(U, c_remove, axis=1)
            V_new = np.delete(V, c_remove, axis=1)
            W_new = np.delete(W, c_remove, axis=1)
            
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
    # j in {0,1},{2,3} and k in {0,1},{2,3},{4,5}, plus j=4
    U, V, W, rank = build_strassen_split([(0,1), (2,3)], [(0,1), (2,3), (4,5)], [4])
    if not validate(U, V, W, rank):
        rank = n*m*p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = schoolbook_block(U, V, W, 0, range(n), range(m), range(p))
    
    best_U, best_V, best_W, best_rank = U.copy(), V.copy(), W.copy(), rank

    # Strategy 2: Try different block splits
    if time.time() < deadline - 20:
        # Split m=2+3, p=2+4
        # j in {0,1}, k in {0,1}: Strassen = 7
        # j in {0,1}, k in {2,3,4,5}: schoolbook = 2*1*4 = 8
        # j in {2,3,4}, k in {0,1}: schoolbook = 2*3*2 = 12
        # j in {2,3,4}, k in {2,3,4,5}: split into 2x2x2 blocks
        #   j={2,3}, k={2,3}: 7
        #   j={2,3}, k={4,5}: 7
        #   j={4}, k={2,3,4,5}: 2*1*4 = 8
        # Total: 7+8+12+7+7+8 = 49
        def try_split_m23_p24():
            rank = 49
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            col = 0
            # j={0,1}, k={0,1}: Strassen
            col = strassen_2x2x2(U, V, W, col, 0, 1, 0, 1, 0, 1)
            # j={0,1}, k={2,3,4,5}: schoolbook
            col = schoolbook_block(U, V, W, col, [0,1], [0,1], [2,3,4,5])
            # j={2,3,4}, k={0,1}: schoolbook
            col = schoolbook_block(U, V, W, col, [0,1], [2,3,4], [0,1])
            # j={2,3}, k={2,3}: Strassen
            col = strassen_2x2x2(U, V, W, col, 0, 1, 2, 3, 2, 3)
            # j={2,3}, k={4,5}: Strassen
            col = strassen_2x2x2(U, V, W, col, 0, 1, 2, 3, 4, 5)
            # j={4}, k={2,3,4,5}: schoolbook
            col = schoolbook_block(U, V, W, col, [0,1], [4], [2,3,4,5])
            return U, V, W, rank
        
        U2, V2, W2, rank2 = try_split_m23_p24()
        if validate(U2, V2, W2, rank2) and rank2 < best_rank:
            best_U, best_V, best_W, best_rank = U2, V2, W2, rank2

    # Strategy 3: Try m=3+2, p=4+2
    # j={0,1,2}, k={0,1,2,3}: schoolbook = 2*3*4 = 24
    # j={0,1,2}, k={4,5}: schoolbook = 2*3*2 = 12
    # j={3,4}, k={0,1,2,3}: schoolbook = 2*2*4 = 16
    # j={3,4}, k={4,5}: Strassen = 7
    # Total: 24+12+16+7 = 59 (worse)
    
    # Strategy 4: Try m=1+2+2, p=2+2+2 with mixed
    # j={0}, k={0,1,2,3,4,5}: 2*1*6 = 12
    # j={1,2}, k={0,1}: 7
    # j={1,2}, k={2,3}: 7
    # j={1,2}, k={4,5}: 7
    # j={3,4}, k={0,1}: 7
    # j={3,4}, k={2,3}: 7
    # j={3,4}, k={4,5}: 7
    # Total: 12+7*6 = 54 (same as before)

    # Strategy 5: Try m=2+1+2, p=3+3
    # j={0,1}, k={0,1,2}: 2x2x3 = 2x2x2(7) + 2x2x1(4) = 11
    # j={0,1}, k={3,4,5}: 11
    # j={2}, k={0,1,2,3,4,5}: 2*1*6 = 12
    # j={3,4}, k={0,1,2}: 11
    # j={3,4}, k={3,4,5}: 11
    # Total: 11+11+12+11+11 = 56 (worse)

    # Strategy 6: Try to use the 2x3x4 rank-12 algorithm if available
    # For now, just try more combinations of Strassen blocks

    # Strategy 7: Try m=2+3, p=3+3
    # j={0,1}, k={0,1,2}: 11
    # j={0,1}, k={3,4,5}: 11
    # j={2,3,4}, k={0,1,2}: 2*3*3 = 18
    # j={2,3,4}, k={3,4,5}: 18
    # Total: 11+11+18+18 = 58 (worse)

    # Strategy 8: Try m=2+1+2, p=2+2+2 but with j=2 handled differently
    # Already tried, same as Strategy 4

    # Strategy 9: Try to find a better decomposition using linear algebra
    # Build the constraint matrix and try to find a lower-rank solution

    # For now, use the best decomposition found so far
    if time.time() < deadline - 15:
        U3, V3, W3, rank3 = try_rank_reduction(best_U, best_V, best_W, best_rank)
        if rank3 < best_rank:
            best_U, best_V, best_W, best_rank = U3, V3, W3, rank3

    # Convert to lists for the return interface
    U_list = [[int(best_U[i, c]) if abs(best_U[i, c] - round(best_U[i, c])) < 1e-9 else float(best_U[i, c]) for c in range(best_rank)] for i in range(n*m)]
    V_list = [[int(best_V[i, c]) if abs(best_V[i, c] - round(best_V[i, c])) < 1e-9 else float(best_V[i, c]) for c in range(best_rank)] for i in range(m*p)]
    W_list = [[int(best_W[i, c]) if abs(best_W[i, c] - round(best_W[i, c])) < 1e-9 else float(best_W[i, c]) for c in range(best_rank)] for i in range(p*n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
