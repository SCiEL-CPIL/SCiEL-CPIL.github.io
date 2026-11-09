"""Bootstrap MODEL-AND-OPTIMIZE rewrite: parameterize the tensor decomposition
and verify it exactly, then attempt a small search to reduce rank.

Dimensions: A is (n x m) = (2 x 3), B is (m x p) = (3 x 4), C = A B is (n x p) = (2 x 4).
Convention: U index = i*m+j (A entry), V index = j*p+k (B entry), W index = k*n+i (C entry).
Decomposition: C = sum_r U[:,r] * V[:,r] * W[:,r]  (outer product along r).
Score = rank (number of terms). Lower is better.
"""
import time
import numpy as np
from itertools import product, combinations

DIMENSIONS = [2, 3, 4]
START_TIME = time.time()
TIME_BUDGET = 140.0  # evaluator kills at 150s; keep 10s margin


def _tensor_index(n, m, p):
    """Build the multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1."""
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T


def _verify(U, V, W, T, tol=1e-9):
    """Check that sum_r U[:,r] V[:,r] W[:,r] == T exactly (within tol)."""
    rank = U.shape[1]
    reconstructed = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(reconstructed, T, atol=tol, rtol=0.0)


def _try_reduce(U, V, W, T, max_attempts=200):
    """Attempt to find a lower-rank decomposition by:
    1. Merging proportional terms.
    2. Trying to remove one term at a time, solving for the remaining terms
       by fixing two of the three factor matrices and solving for the third.
    3. Trying to remove two terms at a time.
    4. Trying to replace one term with a linear combination of others.
    """
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    n, m, p = T.shape

    # Step 1: Try merging proportional terms
    merged = True
    while merged and rank > 1:
        merged = False
        for r in range(rank):
            for s in range(r + 1, rank):
                if np.linalg.norm(U[:, r]) < 1e-12 or np.linalg.norm(U[:, s]) < 1e-12:
                    continue
                if np.linalg.norm(V[:, r]) < 1e-12 or np.linalg.norm(V[:, s]) < 1e-12:
                    continue
                ratio_u = np.dot(U[:, r], U[:, s]) / np.dot(U[:, s], U[:, s])
                if (np.allclose(U[:, r], ratio_u * U[:, s], atol=1e-12) and
                        np.allclose(V[:, r], ratio_u * V[:, s], atol=1e-12) and
                        np.allclose(W[:, r], ratio_u * W[:, s], atol=1e-12)):
                    U = np.delete(U, r, axis=1)
                    V = np.delete(V, r, axis=1)
                    W = np.delete(W, r, axis=1)
                    rank -= 1
                    merged = True
                    break
            if merged:
                break

    def _build_system_for_W(U_r, V_r, nr):
        """Build linear system to solve for W given U_r and V_r."""
        A_mat = np.zeros((n * m * p, p * nr))
        b_vec = T.flatten()
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for r_idx in range(nr):
                        A_mat[idx, k * nr + r_idx] = U_r[i, r_idx] * V_r[j, r_idx]
                    idx += 1
        return A_mat, b_vec

    def _build_system_for_U(U_r, V_r, W_r, nr):
        """Build linear system to solve for U given V_r and W_r."""
        A_mat = np.zeros((n * m * p, n * nr))
        b_vec = T.flatten()
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for r_idx in range(nr):
                        A_mat[idx, i * nr + r_idx] = V_r[j, r_idx] * W_r[k, r_idx]
                    idx += 1
        return A_mat, b_vec

    def _build_system_for_V(U_r, V_r, W_r, nr):
        """Build linear system to solve for V given U_r and W_r."""
        A_mat = np.zeros((n * m * p, m * nr))
        b_vec = T.flatten()
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for r_idx in range(nr):
                        A_mat[idx, j * nr + r_idx] = U_r[i, r_idx] * W_r[k, r_idx]
                    idx += 1
        return A_mat, b_vec

    def _check_solution(U_new, V_new, W_new):
        """Check if the new decomposition is valid."""
        reconstructed = np.einsum('ir,jr,kr->ijk', U_new, V_new, W_new)
        return np.allclose(reconstructed, T, atol=1e-9, rtol=0.0)

    # Step 2: Try removing one term at a time, solving for W (fixing U and V)
    for attempt in range(max_attempts):
        if time.time() - START_TIME > TIME_BUDGET - 10:
            break
        if rank <= 1:
            break
        removed = False
        for r_to_remove in range(rank):
            mask = [i for i in range(rank) if i != r_to_remove]
            U_r = U[:, mask]
            V_r = V[:, mask]
            nr = len(mask)
            
            A_mat, b_vec = _build_system_for_W(U_r, V_r, nr)
            try:
                W_sol, residuals, rank_sol, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                W_sol = W_sol.reshape(p, nr)
                if _check_solution(U_r, V_r, W_sol):
                    U = U_r
                    V = V_r
                    W = W_sol
                    rank = nr
                    removed = True
                    break
            except np.linalg.LinAlgError:
                continue
        
        if not removed:
            break
    
    # Step 3: Try removing two terms at a time
    for r1 in range(rank):
        if time.time() - START_TIME > TIME_BUDGET - 10:
            break
        for r2 in range(r1 + 1, rank):
            if time.time() - START_TIME > TIME_BUDGET - 10:
                break
            mask = [i for i in range(rank) if i != r1 and i != r2]
            U_r = U[:, mask]
            V_r = V[:, mask]
            W_r = W[:, mask]
            nr = len(mask)
            
            # Try solving for W
            A_mat, b_vec = _build_system_for_W(U_r, V_r, nr)
            try:
                W_sol, residuals, rank_sol, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                W_sol = W_sol.reshape(p, nr)
                if _check_solution(U_r, V_r, W_sol):
                    U = U_r
                    V = V_r
                    W = W_sol
                    rank = nr
                    break
            except np.linalg.LinAlgError:
                pass
            
            # Try solving for U
            A_mat, b_vec = _build_system_for_U(U_r, V_r, W_r, nr)
            try:
                U_sol, residuals, rank_sol, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                U_sol = U_sol.reshape(n, nr)
                if _check_solution(U_sol, V_r, W_r):
                    U = U_sol
                    V = V_r
                    W = W_r
                    rank = nr
                    break
            except np.linalg.LinAlgError:
                pass
            
            # Try solving for V
            A_mat, b_vec = _build_system_for_V(U_r, V_r, W_r, nr)
            try:
                V_sol, residuals, rank_sol, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                V_sol = V_sol.reshape(m, nr)
                if _check_solution(U_r, V_sol, W_r):
                    U = U_r
                    V = V_sol
                    W = W_r
                    rank = nr
                    break
            except np.linalg.LinAlgError:
                pass
        if rank < rank:  # if we reduced, break outer loop
            break
    
    # Step 4: Try replacing one term with a linear combination of others
    # This is more complex; for now, try a simple approach
    # For each term r, try to express it as a linear combination of others
    for r_target in range(rank):
        if time.time() - START_TIME > TIME_BUDGET - 10:
            break
        others = [i for i in range(rank) if i != r_target]
        # Try to find coefficients c_s such that:
        # U[:,r_target] = sum_s c_s * U[:,s]
        # V[:,r_target] = sum_s c_s * V[:,s]  
        # W[:,r_target] = sum_s c_s * W[:,s]
        # This is only possible if all three are simultaneously linear combinations
        
        # Check if U[:,r_target] is in span of U[:,others]
        U_others = U[:, others]
        try:
            coeff_U, res_U, rank_U, sv_U = np.linalg.lstsq(U_others, U[:, r_target], rcond=None)
            if np.allclose(U_others @ coeff_U, U[:, r_target], atol=1e-10):
                # Check V
                V_others = V[:, others]
                coeff_V, res_V, rank_V, sv_V = np.linalg.lstsq(V_others, V[:, r_target], rcond=None)
                if np.allclose(V_others @ coeff_V, V[:, r_target], atol=1e-10):
                    # Check W
                    W_others = W[:, others]
                    coeff_W, res_W, rank_W, sv_W = np.linalg.lstsq(W_others, W[:, r_target], rcond=None)
                    if np.allclose(W_others @ coeff_W, W[:, r_target], atol=1e-10):
                        # All three are linear combinations, but with different coefficients
                        # This means the term is redundant
                        U = np.delete(U, r_target, axis=1)
                        V = np.delete(V, r_target, axis=1)
                        W = np.delete(W, r_target, axis=1)
                        rank -= 1
        except np.linalg.LinAlgError:
            pass
    
    return U, V, W


def _build_rank22_decomposition(n, m, p):
    """Build the rank-22 Strassen-based decomposition for 2x3 x 3x4.

    Structure:
      C = A_L * B_T1 + A_L * B_T2 + A_R * B_B
    where:
      A_L = A[:, 0:2] (2x2), A_R = A[:, 2:3] (2x1)
      B_T1 = B[0:2, 0:2] (2x2), B_T2 = B[0:2, 2:4] (2x2), B_B = B[2:3, :] (1x4)

    Strassen (rank 7) for each 2x2 block, plus rank-8 schoolbook for 2x1 x 1x4.
    Total: 7 + 7 + 8 = 22.
    """
    rank = 22
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)

    def _add_strassen_block(a_idx, b_idx, c_idx, col_start):
        """Add a Strassen rank-7 decomposition for a 2x2x2 block.

        a_idx: dict with keys '00','01','10','11' -> U row indices
        b_idx: dict with keys '00','01','10','11' -> V row indices
        c_idx: dict with keys '00','01','10','11' -> W row indices
        col_start: starting column in U,V,W
        """
        # Strassen formulas:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)(B00)
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        #
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6

        col = col_start
        # M1
        U[a_idx['00']][col] += 1; U[a_idx['11']][col] += 1
        V[b_idx['00']][col] += 1; V[b_idx['11']][col] += 1
        W[c_idx['00']][col] += 1
        W[c_idx['11']][col] += 1
        col += 1
        # M2
        U[a_idx['10']][col] += 1; U[a_idx['11']][col] += 1
        V[b_idx['00']][col] += 1
        W[c_idx['10']][col] += 1
        W[c_idx['11']][col] += -1
        col += 1
        # M3
        U[a_idx['00']][col] += 1
        V[b_idx['01']][col] += 1; V[b_idx['11']][col] += -1
        W[c_idx['01']][col] += 1
        W[c_idx['11']][col] += 1
        col += 1
        # M4
        U[a_idx['11']][col] += 1
        V[b_idx['10']][col] += 1; V[b_idx['00']][col] += -1
        W[c_idx['00']][col] += 1
        W[c_idx['10']][col] += 1
        col += 1
        # M5
        U[a_idx['00']][col] += 1; U[a_idx['01']][col] += 1
        V[b_idx['11']][col] += 1
        W[c_idx['00']][col] += -1
        W[c_idx['01']][col] += 1
        col += 1
        # M6
        U[a_idx['10']][col] += 1; U[a_idx['00']][col] += -1
        V[b_idx['00']][col] += 1; V[b_idx['01']][col] += 1
        W[c_idx['11']][col] += 1
        col += 1
        # M7
        U[a_idx['01']][col] += 1; U[a_idx['11']][col] += -1
        V[b_idx['10']][col] += 1; V[b_idx['11']][col] += 1
        W[c_idx['00']][col] += 1
        return col

    # Block 1: A_L (2x2) * B_T1 (2x2) -> C[:, 0:2]
    # A indices: A[i,j] -> i*m+j. m=3.
    # A_L[0,0]=A[0,0]->idx 0, A_L[0,1]=A[0,1]->idx 1
    # A_L[1,0]=A[1,0]->idx 3, A_L[1,1]=A[1,1]->idx 4
    a1 = {'00': 0, '01': 1, '10': 3, '11': 4}
    # B_T1[j,k] -> j*p+k. p=4.
    # B_T1[0,0]=B[0,0]->0, B_T1[0,1]=B[0,1]->1
    # B_T1[1,0]=B[1,0]->4, B_T1[1,1]=B[1,1]->5
    b1 = {'00': 0, '01': 1, '10': 4, '11': 5}
    # C[i,k] -> k*n+i. n=2.
    # C[0,0]->0, C[0,1]->2, C[1,0]->1, C[1,1]->3
    c1 = {'00': 0, '01': 2, '10': 1, '11': 3}
    col = _add_strassen_block(a1, b1, c1, 0)

    # Block 2: A_L (2x2) * B_T2 (2x2) -> C[:, 2:4]
    # B_T2[j,k] where k in {2,3}: B_T2[0,2]=B[0,2]->2, B_T2[0,3]=B[0,3]->3
    # B_T2[1,2]=B[1,2]->6, B_T2[1,3]=B[1,3]->7
    b2 = {'00': 2, '01': 3, '10': 6, '11': 7}
    # C[0,2]->4, C[0,3]->6, C[1,2]->5, C[1,3]->7
    c2 = {'00': 4, '01': 6, '10': 5, '11': 7}
    col = _add_strassen_block(a1, b2, c2, col)

    # Block 3: A_R (2x1) * B_B (1x4) -> C (2x4), rank-8 schoolbook
    # A_R[0]=A[0,2]->idx 2, A_R[1]=A[1,2]->idx 5
    # B_B[k]=B[2,k]->idx 8+k for k=0..3
    # C[i,k] -> k*2+i
    for i in range(2):
        for k in range(4):
            U[i * m + 2][col] = 1.0  # A[i,2]
            V[2 * p + k][col] = 1.0  # B[2,k]
            W[k * n + i][col] = 1.0  # C[i,k]
            col += 1

    return U.tolist(), V.tolist(), W.tolist()


def construct():
    n, m, p = DIMENSIONS
    T = _tensor_index(n, m, p)

    # Start with rank-22 Strassen decomposition
    U_list, V_list, W_list = _build_rank22_decomposition(n, m, p)
    U = np.array(U_list, dtype=np.float64)
    V = np.array(V_list, dtype=np.float64)
    W = np.array(W_list, dtype=np.float64)

    if not _verify(U, V, W, T):
        # Fallback: schoolbook rank n*m*p = 2*3*4 = 24
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=np.float64)
        V = np.zeros((m * p, rank), dtype=np.float64)
        W = np.zeros((p * n, rank), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1.0
                    V[j * p + k][col] = 1.0
                    W[k * n + i][col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Attempt rank reduction
    U, V, W = _try_reduce(U, V, W, T)

    # Verify after reduction
    if not _verify(U, V, W, T):
        # Revert to rank 22
        U = np.array(U_list, dtype=np.float64)
        V = np.array(V_list, dtype=np.float64)
        W = np.array(W_list, dtype=np.float64)

    # Time guard: if we're close to the limit, return what we have
    if time.time() - START_TIME > TIME_BUDGET - 5:
        U = np.round(U, 10)
        V = np.round(V, 10)
        W = np.round(W, 10)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Convert to list format with clean values
    # Round to handle floating point noise
    U = np.round(U, 10)
    V = np.round(V, 10)
    W = np.round(W, 10)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


if __name__ == "__main__":
    result = construct()
    rank = len(result["U"][0]) if result["U"] else 0
    print(f"Rank: {rank}")
    T = _tensor_index(2, 3, 4)
    U = np.array(result["U"])
    V = np.array(result["V"])
    W = np.array(result["W"])
    print(f"Valid: {_verify(U, V, W, T)}")
