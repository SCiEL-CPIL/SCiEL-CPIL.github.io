import numpy as np
import time
import itertools

DIMENSIONS = [2, 3, 6]

def check_decomposition(U, V, W, n, m, p):
    """
    Verify that U, V, W exactly decompose the n x m x p matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    """
    r = U.shape[1]
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    
    # For each basis pair (A_basis, B_basis), check that the computed C matches expected C.
    # A_basis has 1 at (a, b), B_basis has 1 at (c, d).
    # Expected C[i, k] = 1 if i==a, k==d, b==c; else 0.
    # Computed C[i, k] = sum_l W[k*n+i, l] * (sum_b' U[a*m+b', l] * A_basis[a,b']) * (sum_d' V[c*p+d', l] * B_basis[c,d'])
    # Since A_basis and B_basis are basis vectors:
    # sum_b' U[a*m+b', l] * A_basis[a,b'] = U[a*m+b, l] (since A_basis[a,b']=1 only if b'=b)
    # sum_d' V[c*p+d', l] * B_basis[c,d'] = V[c*p+d, l] (since B_basis[c,d']=1 only if d'=d)
    # So Computed C[i, k] = sum_l W[k*n+i, l] * U[a*m+b, l] * V[c*p+d, l]
    
    for a in range(n):
        for b in range(m):
            for c in range(m):
                for d in range(p):
                    for i in range(n):
                        for k in range(p):
                            target = 1.0 if (a == i and d == k and b == c) else 0.0
                            s = 0.0
                            for l in range(r):
                                s += W[k*n+i, l] * U[a*m+b, l] * V[c*p+d, l]
                            if not np.isclose(s, target, atol=1e-9):
                                return False
    return True

def solve_for_rank(n, m, p, rank):
    """
    Attempt to find a rank-`rank` decomposition for n x m x p multiplication.
    Uses alternating least squares (ALS) on a randomly initialized dense support.
    Returns (U, V, W) if successful, else None.
    """
    np.random.seed(42 + rank)
    
    # Dimensions
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    U = np.random.randn(dim_U, rank) * 0.1
    V = np.random.randn(dim_V, rank) * 0.1
    W = np.random.randn(dim_W, rank) * 0.1
    
    # We want to minimize || T - sum_l u_l v_l^T w_l^T || where T is the multiplication tensor.
    # In index form: T[i, b, k] = delta_{i, b} delta_{b, k} ... wait.
    # T_{i, b, k} is the coefficient of A_{i, b} B_{b, k} in C_{i, k}.
    # So T_{i, b, k} = 1 if the indices match the multiplication structure.
    # Actually, the tensor is defined such that:
    # C_{i, k} = sum_{b} T_{i, b, k} A_{i, b} B_{b, k}
    # Here T_{i, b, k} = 1 for all valid i, b, k? No.
    # The multiplication is C_{i, k} = sum_{b} A_{i, b} B_{b, k}.
    # So the tensor T has entries T_{i, b, k} = 1 if we view it as a map from A, B to C.
    # But in the decomposition:
    # C_{i, k} = sum_{l} W_{i, k, l} (sum_{b} U_{i, b, l} A_{i, b}) (sum_{b'} V_{b', k, l} B_{b', k})
    # Note: The summation indices in U and V must be consistent.
    # Usually, U is (n*m, r), V is (m*p, r), W is (p*n, r).
    # Let's flatten indices:
    # A index: a = i*m + b (0..n*m-1)
    # B index: c = b'*p + k (0..m*p-1) -- wait, standard is row-major or column-major?
    # Task says: U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank).
    # W[k*n+i, r]. This implies W index is k*n + i.
    # Let's assume standard row-major for U and V as well?
    # U[i*m + b, l], V[b'*p + k, l].
    # Then term l contributes to C[i, k]:
    # W[k*n+i, l] * (sum_b U[i*m+b, l] A[i,b]) * (sum_b' V[b'*p+k, l] B[b',k])
    # We need this sum to equal sum_b A[i,b] B[b,k].
    # This requires that for each l, the vector u_l (in A space) and v_l (in B space) and w_l (in C space)
    # form a rank-1 tensor that sums to the multiplication tensor.
    # The multiplication tensor T has T[i, b, k] = 1 if ... no.
    # T is a 3-way tensor. T_{i, b, k} = 1 if the indices correspond to a valid multiplication term?
    # No, the tensor IS the multiplication map.
    # T_{i, b, k} = 1 if we consider the basis.
    # Let's construct the target tensor T as a 3D array of shape (n, m, p) or flattened.
    # T_flat[a, c, w_idx] ? No.
    # The equation is: for all i, k: sum_l W[k*n+i, l] * U[i*m + b, l] * V[b*p + k, l] ... wait.
    # The indices b in U and b in V must be the SAME b for the term to contribute to C[i,k] via A[i,b]B[b,k].
    # So, for a fixed l:
    # u_l is a vector of length n*m.
    # v_l is a vector of length m*p.
    # w_l is a vector of length p*n.
    # The rank-1 tensor is u_l \otimes v_l \otimes w_l? No.
    # It's a bilinear form.
    # Let's rewrite:
    # C[i,k] = sum_l W[k*n+i, l] * (sum_b U[i*m+b, l] A[i,b]) * (sum_b V[b*p+k, l] B[b,k])
    # Note: In the second sum, the index is b. In the first sum, the index is also b.
    # This implies that U and V are aligned on the b index.
    # So, for a fixed i, k, and l:
    # The term is W[k*n+i, l] * (sum_b U[i*m+b, l] V[b*p+k, l] A[i,b] B[b,k])
    # We want this to equal sum_b A[i,b] B[b,k].
    # This means:
    # sum_l W[k*n+i, l] U[i*m+b, l] V[b*p+k, l] = delta_{b, b} ? No.
    # It must equal 1 for all b? No.
    # The RHS is sum_b A[i,b] B[b,k].
    # So the coefficient of A[i,b] B[b,k] in the LHS must be 1 for all b, and 0 for b' != b.
    # Wait, A[i,b] and B[b,k] are coupled by b.
    # So for a fixed i, k, the LHS is a sum over b of (coeff_{i,b,k}) A[i,b] B[b,k].
    # We need coeff_{i,b,k} = 1 for all b.
    # So: sum_l W[k*n+i, l] U[i*m+b, l] V[b*p+k, l] = 1 for all i, k, b.
    
    # Let's verify this logic.
    # LHS = sum_l W[k*n+i, l] (sum_{b1} U[i*m+b1, l] A[i,b1]) (sum_{b2} V[b2*p+k, l] B[b2,k])
    # LHS = sum_{b1, b2} (sum_l W[k*n+i, l] U[i*m+b1, l] V[b2*p+k, l]) A[i,b1] B[b2,k]
    # RHS = sum_b A[i,b] B[b,k]
    # For LHS = RHS for all A, B, we must have:
    # sum_l W[k*n+i, l] U[i*m+b1, l] V[b2*p+k, l] = 1 if b1 == b2, else 0.
    # i.e., = delta_{b1, b2}.
    
    # So the condition is:
    # For all i, k, b1, b2:
    # sum_l W[k*n+i, l] U[i*m+b1, l] V[b2*p+k, l] = delta_{b1, b2}
    
    # This is a system of equations.
    # Number of equations: n * p * m * m = 2 * 6 * 3 * 3 = 108.
    # Number of unknowns: n*m*rank + m*p*rank + p*n*rank = 6r + 18r + 12r = 36r.
    # For r=11, 396 unknowns, 108 equations. Underdetermined.
    # We can use ALS.
    
    # Let's define the target tensor T of shape (n, m, m, p) ?
    # T[i, b1, b2, k] = delta_{b1, b2}.
    
    # We can solve for W given U, V.
    # Eq: sum_l W[k*n+i, l] U[i*m+b1, l] V[b2*p+k, l] = delta_{b1, b2}
    # Let's fix i, k, b1, b2.
    # Let M_{(i,k,b1,b2), l} = U[i*m+b1, l] V[b2*p+k, l].
    # Then sum_l M_{(i,k,b1,b2), l} W[k*n+i, l] = delta_{b1, b2}.
    # Note that W depends on (k, i, l).
    # Let's group by (k, i).
    # For fixed i, k, we have m*m equations (for b1, b2).
    # Unknowns are W[k*n+i, 0..r-1].
    # Let's form a matrix A_{ik} of size (m*m) x r.
    # A_{ik}[b1*m + b2, l] = U[i*m+b1, l] V[b2*p+k, l].
    # Target vector t_{ik} of size m*m.
    # t_{ik}[b1*m + b2] = delta_{b1, b2}.
    # Solve A_{ik} w_{ik} = t_{ik} for w_{ik} (which is W[k*n+i, :]).
    
    # Similarly for U and V.
    
    # ALS loop:
    max_iter = 200
    tol = 1e-9
    
    for it in range(max_iter):
        # Update W
        W_new = np.zeros((dim_W, rank))
        for i in range(n):
            for k in range(p):
                # Construct A_{ik}
                A_ik = np.zeros((m*m, rank))
                t_ik = np.zeros(m*m)
                for b1 in range(m):
                    for b2 in range(m):
                        A_ik[b1*m + b2, :] = U[i*m + b1, :] * V[b2*p + k, :]
                        if b1 == b2:
                            t_ik[b1*m + b2] = 1.0
                # Solve least squares
                # A_ik is (m*m, rank). We want W_row (1, rank) such that A_ik @ W_row.T = t_ik.
                # W_row is a row of W.
                # So we solve A_ik^T W_row^T = t_ik? No.
                # sum_l A_ik[idx, l] W_row[l] = t_ik[idx].
                # This is A_ik @ W_row.T = t_ik.
                # W_row.T = np.linalg.lstsq(A_ik, t_ik, rcond=None)[0]
                W_row = np.linalg.lstsq(A_ik, t_ik, rcond=None)[0]
                W_new[k*n + i, :] = W_row
        W = W_new
        
        # Update U
        U_new = np.zeros((dim_U, rank))
        for i in range(n):
            for b in range(m):
                # Equations involving U[i*m+b, l].
                # sum_l W[k*n+i, l] U[i*m+b, l] V[b2*p+k, l] = delta_{b, b2}
                # For fixed i, b, we have p * m equations (for k, b2).
                # Unknowns: U[i*m+b, 0..r-1].
                # A_ub[k*m + b2, l] = W[k*n+i, l] V[b2*p+k, l].
                # t_ub[k*m + b2] = delta_{b, b2}.
                A_ub = np.zeros((p*m, rank))
                t_ub = np.zeros(p*m)
                for k in range(p):
                    for b2 in range(m):
                        A_ub[k*m + b2, :] = W[k*n + i, :] * V[b2*p + k, :]
                        if b == b2:
                            t_ub[k*m + b2] = 1.0
                U_row = np.linalg.lstsq(A_ub, t_ub, rcond=None)[0]
                U_new[i*m + b, :] = U_row
        U = U_new
        
        # Update V
        V_new = np.zeros((dim_V, rank))
        for b in range(m):
            for k in range(p):
                # Equations involving V[b*p+k, l].
                # sum_l W[k*n+i, l] U[i*m+b1, l] V[b*p+k, l] = delta_{b1, b}
                # For fixed b, k, we have n * m equations (for i, b1).
                # Unknowns: V[b*p+k, 0..r-1].
                # A_vk[i*m + b1, l] = W[k*n+i, l] U[i*m+b1, l].
                # t_vk[i*m + b1] = delta_{b1, b}.
                A_vk = np.zeros((n*m, rank))
                t_vk = np.zeros(n*m)
                for i in range(n):
                    for b1 in range(m):
                        A_vk[i*m + b1, :] = W[k*n + i, :] * U[i*m + b1, :]
                        if b1 == b:
                            t_vk[i*m + b1] = 1.0
                V_row = np.linalg.lstsq(A_vk, t_vk, rcond=None)[0]
                V_new[b*p + k, :] = V_row
        V = V_new
        
        # Check convergence
        # Compute residual
        # We can check one random set of equations or all.
        # For speed, check all 108.
        residual = 0.0
        for i in range(n):
            for k in range(p):
                for b1 in range(m):
                    for b2 in range(m):
                        target = 1.0 if b1 == b2 else 0.0
                        s = 0.0
                        for l in range(rank):
                            s += W[k*n+i, l] * U[i*m+b1, l] * V[b2*p+k, l]
                        residual += (s - target)**2
        if residual < tol:
            # Verify full decomposition
            if check_decomposition(U, V, W, n, m, p):
                return U, V, W
            else:
                # Should not happen if equations are correct, but just in case
                return None
        
        # If residual is not decreasing, break?
        # Keep it simple.
        
    # Final check
    if check_decomposition(U, V, W, n, m, p):
        return U, V, W
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    
    # Try ranks from 11 up to 34
    # 11 is the lower bound for 2x3x3, but for 2x3x6 it might be higher.
    # However, ALS might find a solution if one exists.
    # We start with 11 and go up.
    
    best_U, best_V, best_W = None, None, None
    best_rank = 35 # Fallback
    
    for rank in range(11, 35):
        if time.time() > deadline - 10:
            break
            
        U, V, W = solve_for_rank(n, m, p, rank)
        if U is not None:
            best_U, best_V, best_W = U, V, W
            best_rank = rank
            break # Found minimal rank (in the range tried)
            
    if best_U is None:
        # Fallback to Rank 34 (Seed)
        # Reconstruct the seed solution from the current program
        U_comb = np.zeros((6, 34))
        V_comb = np.zeros((18, 34))
        W_comb = np.zeros((12, 34))
        
        U233_seed = np.zeros((6, 17))
        V233_seed = np.zeros((9, 17))
        W233_seed = np.zeros((6, 17))
        
        U233_seed[0, 0] = 1; U233_seed[4, 0] = 1
        V233_seed[0, 0] = 1; V233_seed[4, 0] = 1
        W233_seed[0, 0] = 1; W233_seed[3, 0] = 1
        
        U233_seed[3, 1] = 1; U233_seed[4, 1] = 1
        V233_seed[0, 1] = 1
        W233_seed[1, 1] = 1; W233_seed[3, 1] = -1
        
        U233_seed[0, 2] = 1
        V233_seed[1, 2] = 1; V233_seed[4, 2] = -1
        W233_seed[2, 2] = 1; W233_seed[3, 2] = 1
        
        U233_seed[4, 3] = 1
        V233_seed[3, 3] = 1; V233_seed[0, 3] = -1
        W233_seed[0, 3] = 1; W233_seed[1, 3] = 1
        
        U233_seed[0, 4] = 1; U233_seed[1, 4] = 1
        V233_seed[4, 4] = 1
        W233_seed[0, 4] = -1; W233_seed[2, 4] = 1
        
        U233_seed[3, 5] = 1; U233_seed[0, 5] = -1
        V233_seed[0, 5] = 1; V233_seed[1, 5] = 1
        W233_seed[3, 5] = 1
        
        U233_seed[1, 6] = 1; U233_seed[4, 6] = -1
        V233_seed[3, 6] = 1; V233_seed[4, 6] = 1
        W233_seed[0, 6] = 1
        
        U233_seed[0, 7] = 1
        V233_seed[2, 7] = 1
        W233_seed[4, 7] = 1
        
        U233_seed[1, 8] = 1
        V233_seed[5, 8] = 1
        W233_seed[4, 8] = 1
        
        U233_seed[3, 9] = 1
        V233_seed[2, 9] = 1
        W233_seed[5, 9] = 1
        
        U233_seed[4, 10] = 1
        V233_seed[5, 10] = 1
        W233_seed[5, 10] = 1
        
        U233_seed[2, 11] = 1
        V233_seed[6, 11] = 1
        W233_seed[0, 11] = 1
        
        U233_seed[2, 12] = 1
        V233_seed[7, 12] = 1
        W233_seed[2, 12] = 1
        
        U233_seed[2, 13] = 1
        V233_seed[8, 13] = 1
        W233_seed[4, 13] = 1
        
        U233_seed[5, 14] = 1
        V233_seed[6, 14] = 1
        W233_seed[1, 14] = 1
        
        U233_seed[5, 15] = 1
        V233_seed[7, 15] = 1
        W233_seed[3, 15] = 1
        
        U233_seed[5, 16] = 1
        V233_seed[8, 16] = 1
        W233_seed[5, 16] = 1
        
        U_comb[:, :17] = U233_seed
        U_comb[:, 17:] = U233_seed
        
        for j in range(3):
            for k in range(3):
                V_comb[j * 6 + k, :17] = V233_seed[j * 3 + k]
                V_comb[j * 6 + 3 + k, 17:] = V233_seed[j * 3 + k]
                
        for k in range(3):
            for i in range(2):
                W_comb[k * 2 + i, :17] = W233_seed[k * 2 + i]
                W_comb[(k + 3) * 2 + i, 17:] = W233_seed[k * 2 + i]
                
        best_U, best_V, best_W = U_comb, V_comb, W_comb
        best_rank = 34

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }

if __name__ == "__main__":
    construct()
