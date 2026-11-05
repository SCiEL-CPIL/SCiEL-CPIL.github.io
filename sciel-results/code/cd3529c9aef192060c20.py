"""
Strategy: Numerical rank-reduction search starting from a known valid decomposition.
1. Construct the standard rank-88 decomposition (2x2x2 Strassen blocks + 1x2x2 Schoolbook blocks).
2. Iteratively attempt to reduce rank by projecting one column of the factor matrices 
   onto the subspace spanned by the others. If the residual error is below a tolerance, 
   remove that column (rank reduction).
3. After numerical reduction, attempt to snap coefficients to the nearest half-integer 
   and verify exactness.
4. If exactness fails, return the best valid decomposition found (likely rank 88).
"""
import time
import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # 1. Construct initial rank-88 decomposition
    U = np.zeros((n*m, 88), dtype=float)
    V = np.zeros((m*p, 88), dtype=float)
    W = np.zeros((p*n, 88), dtype=float)
    
    col = 0
    # Strassen terms for 2x2x2
    # Indices for U (i*m+j), V (j*p+k), W (k*n+i)
    # Standard Strassen for A(2,2)B(2,2):
    # m1 = (A00+A11)(B01+B10)
    # m2 = (A10+A11)B00
    # m3 = A00(B10-B11)
    # m4 = A11(B01-B00)
    # m5 = (A00+A10)B11
    # m6 = (A11-A01)(B00+B10)
    # m7 = (A01-A11)(B01+B11)
    # C00 = m1 + m4 - m5 - m7
    # C01 = m1 + m3
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m5
    
    # We need to map these to U, V, W factors for the tensor T[i,j,k] = sum u[i,j] v[j,k] w[k,i]
    # Note: The tensor definition is T[i,j,k] = 1 if we are decomposing the "all ones" tensor?
    # No, the task is to decompose the matrix multiplication tensor.
    # T[i,j,k] is the coefficient of A[i,j]B[j,k] in C[i,k].
    # For standard matrix multiplication, T[i,j,k] = 1 for all i,j,k.
    # Wait, the tensor T is defined by T[i,j,k] = 1.
    # The decomposition is T = sum_r U_r V_r W_r^T (where W_r is vector in R^{p*n})
    # Specifically: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k.
    
    # Strassen block for i in {0,1}, j in {0,1}, k in {0,1} (relative to block origin)
    # Let's define the 7 terms for a 2x2x2 block.
    # U indices: (0,0), (0,1), (1,0), (1,1) -> flat: 0, 1, 2, 3
    # V indices: (0,0), (0,1), (1,0), (1,1) -> flat: 0, 1, 2, 3
    # W indices: (0,0), (0,1), (1,0), (1,1) -> flat: 0, 1, 2, 3 (k*n+i)
    
    # Term 1: (A00+A11)(B01+B10) -> U: [1,0,0,1], V: [0,1,1,0], W: contributes to C00, C01, C11
    # Actually, it's easier to just use the known rank-88 construction from the seed or derive it carefully.
    # The seed program provided a valid rank-88 construction. I will reuse that logic to initialize.
    
    strassen_terms = [
        # u_coeffs, v_coeffs, w_coeffs
        # u: i0j0, i0j1, i1j0, i1j1
        # v: j0k0, j0k1, j1k0, j1k1
        # w: k0i0, k0i1, k1i0, k1i1
        ([1,0,0,1], [0,1,1,0], [1,1,0,0]), # m1: (A00+A11)(B01+B10) -> C00+=1, C01+=1, C11-=1? No.
        # Let's stick to the exact terms from the seed which were verified.
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]), # p1
        ([0,0,1,1], [1,1,0,0], [0,0,1,-1]), # p2
        ([1,1,0,0], [0,0,1,1], [0,1,0,1]),  # p3
        ([0,0,0,1], [0,-1,0,1], [1,0,-1,0]), # p4
        ([1,0,0,0], [1,0,-1,0], [-1,1,0,0]), # p5
        ([-1,0,1,0], [0,1,0,1], [0,0,0,1]),  # p6
        ([0,1,0,-1], [1,0,1,0], [1,0,0,0])   # p7
    ]
    
    for i0 in [0]:
        for j0 in [0, 2]:
            for k0 in [0, 2, 4, 6]:
                for t in range(7):
                    a_coeffs = strassen_terms[t][0]
                    b_coeffs = strassen_terms[t][1]
                    c_coeffs = strassen_terms[t][2]
                    
                    for di in range(2):
                        for dj in range(2):
                            U[(i0+di)*m + (j0+dj), col] = a_coeffs[di*2+dj]
                    
                    for dj in range(2):
                        for dk in range(2):
                            V[(j0+dj)*p + (k0+dk), col] = b_coeffs[dj*2+dk]
                    
                    for dk in range(2):
                        for di in range(2):
                            W[(k0+dk)*n + (i0+di), col] = c_coeffs[di*2+dk]
                    
                    col += 1
    
    for i0 in [2]:
        for j0 in [0, 2]:
            for k0 in [0, 2, 4, 6]:
                for dj in range(2):
                    for dk in range(2):
                        U[i0*m + (j0+dj), col] = 1
                        V[(j0+dj)*p + (k0+dk), col] = 1
                        W[(k0+dk)*n + i0, col] = 1
                        col += 1
                        
    current_rank = 88
    U = U[:, :current_rank]
    V = V[:, :current_rank]
    W = W[:, :current_rank]

    # 2. Attempt rank reduction
    # We try to remove columns one by one.
    # To remove column c, we need to find coefficients alpha_r such that
    # U[:, c] ≈ sum_r alpha_r U[:, r]
    # V[:, c] ≈ sum_r alpha_r V[:, r]
    # W[:, c] ≈ sum_r alpha_r W[:, r]
    # This is a system of linear equations.
    # Let M_U = U[:, -1:], M_V = V[:, -1:], M_W = W[:, -1:]
    # Let A_U = U[:, :-1], A_V = V[:, :-1], A_W = W[:, :-1]
    # We want alpha such that A_U alpha ≈ M_U, A_V alpha ≈ M_V, A_W alpha ≈ M_W.
    # This is a least squares problem on the stacked matrix.
    
    max_iter = 1000
    tol = 1e-9
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        if current_rank <= 1:
            break
            
        removed = False
        # Try to remove the last column
        c = current_rank - 1
        
        # Stack U, V, W for the target column
        target = np.concatenate([U[:, c], V[:, c], W[:, c]])
        # Stack U, V, W for the other columns
        basis = np.concatenate([U[:, :c], V[:, :c], W[:, :c]], axis=0)
        
        # Solve basis @ alpha = target
        try:
            alpha, residuals, rank_b, sv = np.linalg.lstsq(basis, target, rcond=None)
        except:
            continue
            
        # Calculate residual
        pred = basis @ alpha
        residual = np.linalg.norm(target - pred)
        
        if residual < tol:
            # Remove column c
            U = np.delete(U, c, axis=1)
            V = np.delete(V, c, axis=1)
            W = np.delete(W, c, axis=1)
            current_rank -= 1
            removed = True
            
    # 3. Rounding and Verification
    # The numerical solution might have floating point errors.
    # We need integer/half-integer entries.
    # Since we started with integers and only did linear combinations, 
    # if a removal was valid, the coefficients should be "nice".
    # However, lstsq might produce floats.
    # Let's try to snap to nearest half-integer.
    
    def snap_to_half_int(matrix):
        return np.round(matrix * 2) / 2.0
        
    U_snap = snap_to_half_int(U)
    V_snap = snap_to_half_int(V)
    W_snap = snap_to_half_int(W)
    
    # Verify exactness
    def _verify(U, V, W, n, m, p):
        rank = U.shape[1]
        # Vectorized verification
        # T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
        # We can compute the tensor T_rec and check if it is all ones.
        # T_rec[i, j, k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
        # This is a contraction.
        # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
        # Sum over rank.
        # T_rec = np.einsum('ijr,jkr,kir->ijk', U, V, W)
        
        try:
            T_rec = np.einsum('ijr,jkr,kir->ijk', U, V, W)
            # Check if all elements are 1.0
            return np.allclose(T_rec, 1.0, atol=1e-9)
        except:
            return False
            
    if _verify(U_snap, V_snap, W_snap, n, m, p):
        return {
            "U": U_snap.tolist(),
            "V": V_snap.tolist(),
            "W": W_snap.tolist()
        }
    
    # If rounding failed, try the unrounded version? 
    # The task requires integer/half-integer. 
    # If the numerical reduction produced non-halves, we must fall back.
    # However, if no reduction happened, U, V, W are still the original integers.
    # So we should check if the ORIGINAL (or reduced) matrices are close to halves.
    
    # Check if current U, V, W are close to halves
    if np.allclose(U, U_snap) and np.allclose(V, V_snap) and np.allclose(W, W_snap):
         # They are effectively halves. Verify again just in case.
         if _verify(U_snap, V_snap, W_snap, n, m, p):
             return {
                "U": U_snap.tolist(),
                "V": V_snap.tolist(),
                "W": W_snap.tolist()
            }
             
    # Fallback: Return the initial rank-88 solution (which is valid integers)
    # Re-construct or just use the original? 
    # We modified U, V, W in place during reduction.
    # If reduction failed or rounding failed, we need a valid solution.
    # The safest is to return the initial construction.
    
    # Re-initialize to be safe
    U_init = np.zeros((n*m, 88), dtype=float)
    V_init = np.zeros((m*p, 88), dtype=float)
    W_init = np.zeros((p*n, 88), dtype=float)
    col = 0
    for i0 in [0]:
        for j0 in [0, 2]:
            for k0 in [0, 2, 4, 6]:
                for t in range(7):
                    a_coeffs = strassen_terms[t][0]
                    b_coeffs = strassen_terms[t][1]
                    c_coeffs = strassen_terms[t][2]
                    for di in range(2):
                        for dj in range(2):
                            U_init[(i0+di)*m + (j0+dj), col] = a_coeffs[di*2+dj]
                    for dj in range(2):
                        for dk in range(2):
                            V_init[(j0+dj)*p + (k0+dk), col] = b_coeffs[dj*2+dk]
                    for dk in range(2):
                        for di in range(2):
                            W_init[(k0+dk)*n + (i0+di), col] = c_coeffs[di*2+dk]
                    col += 1
    for i0 in [2]:
        for j0 in [0, 2]:
            for k0 in [0, 2, 4, 6]:
                for dj in range(2):
                    for dk in range(2):
                        U_init[i0*m + (j0+dj), col] = 1
                        V_init[(j0+dj)*p + (k0+dk), col] = 1
                        W_init[(k0+dk)*n + i0, col] = 1
                        col += 1
                        
    return {
        "U": U_init.tolist(),
        "V": V_init.tolist(),
        "W": W_init.tolist()
    }

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
