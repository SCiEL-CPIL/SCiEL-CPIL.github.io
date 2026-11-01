import time
import numpy as np

DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin

    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = rank_schoolbook

    def verify_decomposition(U, V, W):
        """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
        R = U.shape[1]
        U_r = U.reshape(n, m, R)
        V_r = V.reshape(m, p, R)
        W_r = W.reshape(p, n, R)
        
        # T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        
        return np.allclose(T, 1.0, atol=1e-9)

    def solve_for_U(V, W, R):
        """
        Given fixed V and W of shape (m*p, R) and (p*n, R),
        find U of shape (n*m, R) such that the tensor equation holds.
        
        The equation is:
        sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1 for all i,j,k.
        
        This is linear in U. We can set up a linear system A * u = b.
        u is the flattened U (size n*m*R).
        b is the flattened target tensor (size n*m*p), all ones.
        
        For each (i,j,k), the coefficient of U[i', j', c] is:
        delta_{i,i'} delta_{j,j'} V[j', k, c] W[k, i', c]
        
        So for a fixed (i,j,k), the row in A corresponds to:
        A[row, idx(i,j,c)] = V[j, k, c] * W[k, i, c]
        
        We construct A as a matrix of size (n*m*p) x (n*m*R).
        """
        num_vars = n * m * R
        num_eqs = n * m * p
        
        A = np.zeros((num_eqs, num_vars))
        b = np.ones(num_eqs)
        
        idx_eq = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for c in range(R):
                        # Coefficient for U[i, j, c] in equation (i, j, k)
                        # The term is U[i,j,c] * V[j,k,c] * W[k,i,c]
                        # Note: Only U[i,j,c] appears in the sum for equation (i,j,k)
                        # Wait, let's re-read the tensor definition.
                        # T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
                        # Yes, for a fixed (i,j,k), the sum is over c.
                        # The variables are U[i,j,c] for all i,j,c.
                        # But in the sum for T[i,j,k], only terms where the first two indices of U match i,j contribute.
                        # So the row for (i,j,k) has non-zero entries only for columns corresponding to U[i,j,c].
                        
                        val = V[j * p + k, c] * W[k * n + i, c]
                        A[idx_eq, i * m * R + j * R + c] = val
                    idx_eq += 1
        
        # Solve A * u = b using least squares
        try:
            u, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            return None, np.inf

        # Check residual
        if residuals is None or len(residuals) == 0:
            # Rank deficient, compute residual manually
            res = np.linalg.norm(A @ u - b)
        else:
            res = residuals[0]
            
        U_flat = u
        U = U_flat.reshape(n, m, R).reshape(n * m, R)
        
        return U, res

    def snap_and_verify(U, V, W):
        """Snap U, V, W to half-integers and verify."""
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2
        
        if verify_decomposition(U_snap, V_snap, W_snap):
            return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
        
        U_snap2 = np.round(U)
        V_snap2 = np.round(V)
        W_snap2 = np.round(W)
        if verify_decomposition(U_snap2, V_snap2, W_snap2):
            return {"U": U_snap2.tolist(), "V": V_snap2.tolist(), "W": W_snap2.tolist()}
            
        return None

    # Strategy:
    # 1. Try ranks from theoretical lower bound (30) up to 53 (Strassen+Naive).
    # 2. For each rank R, try several structured initializations of V and W.
    # 3. Solve for U via least squares.
    # 4. If residual is small, snap and verify.

    # Theoretical lower bound is max(n*m, m*p, p*n) = 30.
    # Known result for M_{2,2,p} is often cited around 2p+1 or similar.
    # Let's try R from 30 to 40 first, then up to 53 if time permits.
    
    max_R_to_try = 53
    
    # Structured initializations for V and W
    # We need V (2*15, R) and W (15*2, R).
    # Let's generate a few different "bases" for the columns.
    
    def generate_structured_VW(R, seed):
        np.random.seed(seed)
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        
        # Strategy A: Random sparse
        # Each column has a few non-zeros
        for c in range(R):
            # Pick random non-zeros for V
            non_zero_V = np.random.choice(m * p, size=min(3, m * p), replace=False)
            V[non_zero_V, c] = np.random.choice([-1, 1], size=len(non_zero_V))
            
            non_zero_W = np.random.choice(p * n, size=min(3, p * n), replace=False)
            W[non_zero_W, c] = np.random.choice([-1, 1], size=len(non_zero_W))
            
        return V, W

    def generate_strassen_hybrid_VW(R, seed):
        """
        Use Strassen structure for the first 2x2x2 block (k=0,1) 
        and simple structure for the rest.
        """
        np.random.seed(seed)
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        
        if R < 7:
            # Cannot fit full Strassen, fallback to random
            return generate_structured_VW(R, seed)
            
        # Strassen factors for 2x2x2
        # U_strassen: 4x7, V_strassen: 4x7, W_strassen: 4x7
        # Indices: U[i,j,c], V[j,k,c], W[k,i,c]
        # Standard Strassen:
        # m1 = (a11-a22)(b21+b22)
        # m2 = (a11+a12)b21
        # m3 = a12(b22-b12)
        # m4 = (a21+a22)b12
        # m5 = (a11-a21)(b11+b21)
        # m6 = (a12-a22)(b12+b22)
        # m7 = (a11+a22)(b11-b22)
        
        U_s = np.zeros((4, 7))
        V_s = np.zeros((4, 7))
        W_s = np.zeros((4, 7))
        
        # c=0
        U_s[0, 0] = 1; U_s[3, 0] = -1
        V_s[2, 0] = 1; V_s[3, 0] = 1
        W_s[1, 0] = 1; W_s[0, 0] = 1
        
        # c=1
        U_s[0, 1] = 1; U_s[1, 1] = 1
        V_s[2, 1] = 1
        W_s[1, 1] = 1
        
        # c=2
        U_s[1, 2] = 1
        V_s[3, 2] = 1; V_s[1, 2] = -1
        W_s[0, 2] = 1
        
        # c=3
        U_s[2, 3] = 1; U_s[3, 3] = 1
        V_s[1, 3] = 1
        W_s[0, 3] = 1
        
        # c=4
        U_s[0, 4] = 1; U_s[2, 4] = -1
        V_s[0, 4] = 1; V_s[2, 4] = 1
        W_s[0, 4] = 1
        
        # c=5
        U_s[1, 5] = 1; U_s[3, 5] = -1
        V_s[1, 5] = 1; V_s[3, 5] = 1
        W_s[1, 5] = 1
        
        # c=6
        U_s[0, 6] = 1; U_s[3, 6] = 1
        V_s[0, 6] = 1; V_s[3, 6] = -1
        W_s[0, 6] = 1; W_s[1, 6] = -1
        
        # Map Strassen indices to full dimensions for k=0,1
        # V index: j*p + k. For p=15, j=0 -> k=0,1; j=1 -> k=0,1
        # W index: k*n + i. For n=2, k=0 -> i=0,1; k=1 -> i=0,1
        
        for c in range(7):
            for j in range(2):
                for k in range(2):
                    if V_s[j, c] != 0:
                        V[j * p + k, c] = V_s[j, c]
            for k in range(2):
                for i in range(2):
                    if W_s[k, c] != 0:
                        W[k * n + i, c] = W_s[k, c]
                        
        # Fill remaining columns (7 to R-1) with simple structure for k=2..14
        # We have R-7 columns left.
        # We need to handle k=2..14 (13 values).
        # For each k, we need 2 multiplications (i=0, j=0 and i=1, j=1? No, 4 mults per k).
        # Actually, for each k, we have a 2x2 matrix multiplication A * B_k.
        # We can use rank 4 per k (schoolbook) -> 13*4 = 52. Total 7+52=59.
        # We want to share.
        # Let's just use random sparse for the rest if R > 7.
        
        for c in range(7, R):
            # Random sparse
            non_zero_V = np.random.choice(m * p, size=min(4, m * p), replace=False)
            V[non_zero_V, c] = np.random.choice([-1, 1], size=len(non_zero_V))
            
            non_zero_W = np.random.choice(p * n, size=min(4, p * n), replace=False)
            W[non_zero_W, c] = np.random.choice([-1, 1], size=len(non_zero_W))
            
        return V, W

    # Try ranks
    for R in range(30, max_R_to_try + 1):
        if time.time() > deadline:
            break
            
        # Try a few different seeds/structures
        found = None
        for seed in range(5):
            if time.time() > deadline:
                break
                
            # Alternate between random sparse and Strassen hybrid
            if seed % 2 == 0:
                V, W = generate_structured_VW(R, seed)
            else:
                V, W = generate_strassen_hybrid_VW(R, seed)
                
            U, res = solve_for_U(V, W, R)
            
            if U is None:
                continue
                
            # If residual is very small, we likely have a solution
            if res < 1e-6:
                result = snap_and_verify(U, V, W)
                if result:
                    return result
        # If we found a low residual but snapping failed, we might need to adjust V,W slightly
        # But for now, we stick to integer/half-integer solutions which are more likely to snap.
        
    # Fallback to schoolbook
    return best_result
# EVOLVE-BLOCK-END
