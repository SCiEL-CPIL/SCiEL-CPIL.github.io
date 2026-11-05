import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 148.0

    def try_rank_28():
        """Build the known rank-28 decomposition: two Strassen 2x2x2 blocks + schoolbook for rest."""
        rank = 28
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0

        def add_strassen(i1, i2, j1, j2, k1, k2):
            nonlocal col
            # M1 = (A00+A11)(B00+B11)
            U[i1*m+j1, col] = 1; U[i2*m+j2, col] = 1
            V[j1*p+k1, col] = 1; V[j2*p+k2, col] = 1
            W[k1*n+i1, col] = 1; W[k2*n+i2, col] = 1
            col += 1
            # M2 = (A10+A11)B00
            U[i2*m+j1, col] = 1; U[i2*m+j2, col] = 1
            V[j1*p+k1, col] = 1
            W[k1*n+i2, col] = 1; W[k2*n+i2, col] = -1
            col += 1
            # M3 = A00(B01-B11)
            U[i1*m+j1, col] = 1
            V[j1*p+k2, col] = 1; V[j2*p+k2, col] = -1
            W[k2*n+i1, col] = 1; W[k2*n+i2, col] = 1
            col += 1
            # M4 = A11(B10-B00)
            U[i2*m+j2, col] = 1
            V[j2*p+k1, col] = 1; V[j1*p+k1, col] = -1
            W[k1*n+i1, col] = 1; W[k1*n+i2, col] = 1
            col += 1
            # M5 = (A00+A01)B11
            U[i1*m+j1, col] = 1; U[i1*m+j2, col] = 1
            V[j2*p+k2, col] = 1
            W[k1*n+i1, col] = -1; W[k2*n+i1, col] = 1
            col += 1
            # M6 = (A10-A00)(B00+B01)
            U[i2*m+j1, col] = 1; U[i1*m+j1, col] = -1
            V[j1*p+k1, col] = 1; V[j1*p+k2, col] = 1
            W[k2*n+i2, col] = 1
            col += 1
            # M7 = (A01-A11)(B10+B11)
            U[i1*m+j2, col] = 1; U[i2*m+j2, col] = -1
            V[j2*p+k1, col] = 1; V[j2*p+k2, col] = 1
            W[k1*n+i1, col] = 1
            col += 1

        def add_schoolbook(i_list, j_list, k_list):
            nonlocal col
            for i in i_list:
                for j in j_list:
                    for k in k_list:
                        U[i*m+j, col] = 1
                        V[j*p+k, col] = 1
                        W[k*n+i, col] = 1
                        col += 1

        add_strassen(0, 1, 0, 1, 0, 1)      # A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
        add_strassen(0, 1, 0, 1, 2, 3)      # A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4]
        add_schoolbook([0,1],[0,1],[4])      # A[0:2,0:2] * B[0:2,4:5] -> C[0:2,4:5]
        add_schoolbook([0,1],[2],[0,1])      # A[0:2,2:3] * B[2:3,0:2] -> C[0:2,0:2]
        add_schoolbook([0,1],[2],[2,3])      # A[0:2,2:3] * B[2:3,2:4] -> C[0:2,2:4]
        add_schoolbook([0,1],[2],[4])        # A[0:2,2:3] * B[2:3,4:5] -> C[0:2,4:5]

        assert col == rank, f"col={col}, expected {rank}"
        return U, V, W, rank

    def verify(U, V, W, rank):
        Ur = U.T.reshape(rank, n, m)
        Vr = V.T.reshape(rank, m, p)
        Wr = W.T.reshape(rank, p, n)
        tensor = np.einsum('rij,rjk,rki->ijk', Ur, Vr, Wr)
        return np.allclose(tensor, 1.0, atol=1e-9)

    # First, get the rank-28 solution
    U28, V28, W28, rank28 = try_rank_28()
    
    if not verify(U28, V28, W28, rank28):
        # Shouldn't happen, but fall back to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try to find rank 27 by removing one rank-1 term and adding a compensating term
    # The schoolbook terms (columns 14-27) are the most likely candidates for removal
    # since they're simple rank-1 terms that might be partially covered by other terms
    
    if time.time() < deadline - 10:
        rank_target = 27
        best_solution = None
        
        # Try removing each schoolbook term (cols 14-27) and see if we can compensate
        for remove_col in range(14, 28):
            if time.time() > deadline - 15:
                break
            
            U_new = U28.copy()
            V_new = V28.copy()
            W_new = W28.copy()
            
            # Remove the term
            U_new[:, remove_col] = 0
            V_new[:, remove_col] = 0
            W_new[:, remove_col] = 0
            
            # The removed term contributes to some (i,j,k) entries
            # We need to find a new rank-1 term that compensates
            # This is hard, so instead let's try a different approach:
            # Look for pairs of terms that can be merged
            
            # Actually, let's try a smarter approach: 
            # Check if any schoolbook term is redundant (its contribution is already covered)
            # by the linear span of other terms
            
            # For now, just check if removing a term still gives a valid decomposition
            # (unlikely, but worth checking)
            if verify(U_new, V_new, W_new, rank_target + 1):
                # We can remove this term, so we have rank 27
                # (remove the column)
                idx = [i for i in range(28) if i != remove_col]
                U_final = U28[:, idx]
                V_final = V28[:, idx]
                W_final = W28[:, idx]
                if verify(U_final, V_final, W_final, rank_target):
                    best_solution = (U_final, V_final, W_final, rank_target)
                    break
    
    if best_solution is not None:
        U, V, W, rank = best_solution
        if verify(U, V, W, rank):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: rank 28
    return {"U": U28.tolist(), "V": V28.tolist(), "W": W28.tolist()}
