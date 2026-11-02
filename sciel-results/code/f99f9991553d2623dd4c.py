import numpy as np
import time
import itertools

DIMENSIONS = [2, 3, 6]

def _tensor_to_matrix(tensor, mode):
    """
    Reshape the 3D tensor (n, m, p) into a matrix where 'mode' is the row index
    and the other two modes are flattened into columns.
    mode=0: rows n, cols m*p
    mode=1: rows m, cols n*p
    mode=2: rows p, cols n*m
    """
    n, m, p = tensor.shape
    if mode == 0:
        return tensor.reshape(n, m * p)
    elif mode == 1:
        return np.transpose(tensor, (1, 0, 2)).reshape(m, n * p)
    elif mode == 2:
        return np.transpose(tensor, (2, 0, 1)).reshape(p, n * m)

def _matrix_to_tensor(mat, mode, n, m, p):
    """
    Reshape a matrix back into the 3D tensor.
    """
    if mode == 0:
        return mat.reshape(n, m, p)
    elif mode == 1:
        return np.transpose(mat.reshape(m, n, p), (1, 0, 2))
    elif mode == 2:
        return np.transpose(mat.reshape(p, n, m), (2, 0, 1))

def _rank1_product(u, v, w, n, m, p):
    """
    Compute the outer product of three vectors u (len n), v (len m), w (len p).
    Returns a 3D tensor of shape (n, m, p).
    """
    return np.einsum('i,j,k->ijk', u, v, w)

def _greedy_decompose(target_tensor, max_rank, time_limit):
    """
    Greedy algorithm to find a rank decomposition.
    At each step, find the best rank-1 approximation to the residual,
    subtract it, and repeat until rank is reached or residual is zero.
    """
    n, m, p = target_tensor.shape
    residual = target_tensor.copy()
    U_list = []
    V_list = []
    W_list = []
    
    start_time = time.time()
    
    for r in range(max_rank):
        if time.time() - start_time > time_limit:
            break
            
        # Check if residual is effectively zero
        if np.allclose(residual, 0, atol=1e-9):
            break
            
        # Find best rank-1 approximation
        # We can use SVD on the matrix unfolding.
        # The best rank-1 approximation of a tensor is NOT simply the top SVD vector of one unfolding,
        # but for a greedy step, we can iterate over the three modes to refine.
        # However, a simpler and effective greedy step for small tensors is:
        # 1. Pick a mode (e.g., 0).
        # 2. Compute SVD of the unfolding M0.
        # 3. Let u be the first left singular vector.
        # 4. Update residual R = R - u * (something).
        
        # Actually, the standard greedy step is:
        # Approximate residual T by u v^T w^T.
        # We can find u, v, w by alternating minimization starting from random or SVD init.
        
        # Let's initialize u, v, w from SVD of mode-0 unfolding.
        M0 = _tensor_to_matrix(residual, 0)
        try:
            U_svd, S, Vt = np.linalg.svd(M0, full_matrices=False)
            u = U_svd[:, 0]
            # The remaining part is S[0] * Vt[0, :]
            # We need to decompose the matrix S[0] * Vt[0, :] into v and w.
            # Vt[0, :] has shape (m*p,). Reshape to (m, p).
            vw_mat = (S[0] * Vt[0, :]).reshape(m, p)
            # SVD of vw_mat
            U_vw, S_vw, Vt_vw = np.linalg.svd(vw_mat, full_matrices=False)
            v = U_vw[:, 0] * np.sqrt(S_vw[0])
            w = Vt_vw[0, :] * np.sqrt(S_vw[0])
        except:
            break
            
        # Refine u, v, w using alternating least squares (ALS) on the residual
        # Maximize <residual, u v w>
        # This is equivalent to finding the rank-1 approximation that minimizes ||residual - u v w||_F
        # We can do a few iterations of ALS.
        
        for _ in range(10):
            # Update u: fix v, w. u = argmax <residual, u v w> = (residual * (v w)) flattened
            # <T, u v w> = sum_{i,j,k} T[i,j,k] u[i] v[j] w[k]
            # For fixed v, w, this is linear in u.
            # u[i] should be proportional to sum_{j,k} T[i,j,k] v[j] w[k]
            # This is the mode-1 unfolding of T multiplied by v, then by w?
            # Let's use einsum.
            # u_new[i] = sum_{j,k} T[i,j,k] v[j] w[k]
            u_new = np.einsum('ijk,j,k->i', residual, v, w)
            if np.linalg.norm(u_new) < 1e-12: break
            u = u_new / np.linalg.norm(u_new)
            
            v_new = np.einsum('ijk,i,k->j', residual, u, w)
            if np.linalg.norm(v_new) < 1e-12: break
            v = v_new / np.linalg.norm(v_new)
            
            w_new = np.einsum('ijk,i,j->k', residual, u, v)
            if np.linalg.norm(w_new) < 1e-12: break
            w = w_new / np.linalg.norm(w_new)
            
        # Compute the scalar coefficient alpha for the term u v w
        # We want to subtract the projection of residual onto u v w.
        # alpha = <residual, u v w> / <u v w, u v w>
        # Since u, v, w are normalized, <u v w, u v w> = 1
        alpha = np.einsum('ijk,ijk->', residual, u, v, w)
        
        # Subtract
        residual -= alpha * _rank1_product(u, v, w, n, m, p)
        
        U_list.append(u * np.sqrt(abs(alpha))) # We can absorb signs
        V_list.append(v * np.sign(alpha) * np.sqrt(abs(alpha))) # Wait, better to keep alpha separate or distribute
        # Let's just store the vectors and the scalar, or distribute the scalar into one vector.
        # To keep entries integer/half-integer later, it's better to keep the product as is.
        # The term is alpha * u v w.
        # We can store U= u, V= v, W= alpha*w.
        
        U_list.append(u)
        V_list.append(v)
        W_list.append(alpha * w)
        
    return U_list, V_list, W_list, residual

def _snap_to_rational(arr, tol=1e-6):
    """
    Try to snap values to nearest 0.5 or integer.
    """
    snapped = arr.copy()
    for i in range(len(snapped)):
        val = snapped[i]
        # Check nearest 0.5
        nearest_half = np.round(val * 2) / 2
        if abs(val - nearest_half) < tol:
            snapped[i] = nearest_half
    return snapped

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    
    # Construct the target tensor M_{2,3,6}
    # T[a, b, c] = 1 if a==i and c==k and b==d? No.
    # The tensor for matrix multiplication C = A B.
    # C[i, k] = sum_j A[i, j] B[j, k].
    # The tensor T has indices (i, j, k).
    # T[i, j, k] = 1 if this term contributes to C[i, k] from A[i, j] and B[j, k].
    # So T[i, j, k] = 1.
    # Wait, the tensor is defined such that C[i, k] = sum_{j, l} T[i, j, k, l] A[i, j] B[l, k]?
    # Standard definition: The matrix multiplication tensor M_{n,m,p} is in R^{n x m x m x p}?
    # No, the task says U (n*m, rank), V (m*p, rank), W (p*n, rank).
    # This corresponds to the decomposition of the bilinear map.
    # The tensor T is in R^{(n*m) x (m*p) x (p*n)}?
    # Let's check the indices in the check function of the seed:
    # U[a*m+b, l] * V[c*p+d, l] * W[k*n+i, l]
    # Target is 1 if (a==i and c==k and b==d).
    # This means the tensor T is indexed by (a, b, c, d, k, i)? No.
    # The sum is over l.
    # The term is U_{ab, l} V_{cd, l} W_{ki, l}.
    # The target is delta_{a,i} delta_{c,k} delta_{b,d}.
    # This implies that the "tensor" we are decomposing is actually the structure of the multiplication.
    # Let's construct the 3D tensor T of shape (n*m, m*p, p*n) such that
    # T_{(a,b), (c,d), (k,i)} = 1 if a==i, c==k, b==d else 0.
    
    T = np.zeros((n*m, m*p, p*n))
    for a in range(n):
        for b in range(m):
            for c in range(m):
                for d in range(p):
                    for k in range(p):
                        for i in range(n):
                            if a == i and c == k and b == d:
                                T[a*m+b, c*p+d, k*n+i] = 1.0
                                
    # Now we want to decompose T into a sum of rank-1 tensors.
    # A rank-1 tensor is U_col * V_col * W_col (outer product).
    # T = sum_l U[:, l] V[:, l] W[:, l]^T?
    # The check function computes:
    # sum_l U[a*m+b, l] * V[c*p+d, l] * W[k*n+i, l]
    # This is exactly the entry of the tensor sum_l U_l V_l W_l.
    
    # We will use the greedy algorithm to find U, V, W.
    # The greedy algorithm works on the 3D tensor T.
    
    # However, the greedy algorithm on the full tensor might be slow or get stuck.
    # Let's try a specific known structure:
    # The seed program used a block decomposition.
    # 2x3x6 can be seen as 2x3x(3+3) = 2x3x3 + 2x3x3.
    # If we have a rank R decomposition for 2x3x3, we can use 2R for 2x3x6.
    # The seed used Rank 17 for 2x3x3 (which is high, schoolbook is 18).
    # Actually, the rank of 2x3x3 multiplication is known to be 16? Or 17?
    # Schoolbook is 2*3*3 = 18.
    # Strassen-like improvements might exist.
    
    # Let's try to find a rank 16 or 17 decomposition for 2x3x3 using the greedy method,
    # then double it for 2x3x6.
    
    # Construct T_233
    n2, m2, p2 = 2, 3, 3
    T_233 = np.zeros((n2*m2, m2*p2, p2*n2))
    for a in range(n2):
        for b in range(m2):
            for c in range(m2):
                for d in range(p2):
                    for k in range(p2):
                        for i in range(n2):
                            if a == i and c == k and b == d:
                                T_233[a*m2+b, c*p2+d, k*n2+i] = 1.0
                                
    # Try greedy decomposition of T_233 with max rank 17
    U_list, V_list, W_list, residual = _greedy_decompose(T_233, 17, 60.0)
    
    # Check if residual is small
    if np.linalg.norm(residual) > 1e-6:
        # Try higher rank or fallback
        # If greedy fails, try rank 18 (schoolbook)
        U_list, V_list, W_list, residual = _greedy_decompose(T_233, 18, 60.0)
        
    if np.linalg.norm(residual) > 1e-6:
        # Fallback to a known valid decomposition (Schoolbook)
        # Schoolbook for 2x3x3:
        U_sc = np.zeros((6, 18))
        V_sc = np.zeros((9, 18))
        W_sc = np.zeros((6, 18))
        idx = 0
        for a in range(2):
            for b in range(3):
                for c in range(3):
                    # Term: A[a,b] * B[c,b? No, B is 3x3. B[j,k].
                    # C[i,k] = sum_j A[i,j] B[j,k].
                    # The term A[a,b] B[b,k] contributes to C[a,k].
                    # So U index (a,b), V index (b,k), W index (k,a).
                    # Wait, V index in seed is c*p+d. c is row of B, d is col of B.
                    # So V index (b, k).
                    # W index (k, a).
                    U_sc[a*3+b, idx] = 1
                    V_sc[b*3+k, idx] = 1
                    W_sc[k*2+a, idx] = 1
                    idx += 1
        # This is rank 18.
        U_list = [U_sc[:, i] for i in range(18)]
        V_list = [V_sc[:, i] for i in range(18)]
        W_list = [W_sc[:, i] for i in range(18)]
        
    # Now construct the 2x3x6 decomposition
    # We have U_233 (6 x r), V_233 (9 x r), W_233 (6 x r).
    # For 2x3x6, we split the last dimension (p=6) into two blocks of 3.
    # The V matrix has indices (c, d) where c in 0..2, d in 0..5.
    # Block 1: d in 0..2. Block 2: d in 3..5.
    # The W matrix has indices (k, i) where k in 0..5, i in 0..1.
    # Block 1: k in 0..2. Block 2: k in 3..5.
    
    r = len(U_list)
    U_233 = np.array(U_list).T # Shape (6, r)
    V_233 = np.array(V_list).T # Shape (9, r)
    W_233 = np.array(W_list).T # Shape (6, r)
    
    # Snap to half-integers if possible
    U_233 = _snap_to_rational(U_233)
    V_233 = _snap_to_rational(V_233)
    W_233 = _snap_to_rational(W_233)
    
    # Build U_236 (6 x 2r)
    U_236 = np.zeros((6, 2*r))
    U_236[:, :r] = U_233
    U_236[:, r:] = U_233
    
    # Build V_236 (18 x 2r)
    V_236 = np.zeros((18, 2*r))
    # V index: c*6 + d
    # For d in 0..2 (block 1), V[c*6+d, :r] = V_233[c*3+d, :]
    # For d in 3..5 (block 2), V[c*6+d, r:] = V_233[c*3+(d-3), :]
    for c in range(3):
        for d in range(3):
            V_236[c*6+d, :r] = V_233[c*3+d, :]
            V_236[c*6+3+d, r:] = V_233[c*3+d, :]
            
    # Build W_236 (12 x 2r)
    W_236 = np.zeros((12, 2*r))
    # W index: k*2 + i
    # For k in 0..2 (block 1), W[k*2+i, :r] = W_233[k*2+i, :]
    # For k in 3..5 (block 2), W[k*2+i, r:] = W_233[(k-3)*2+i, :]
    for k in range(3):
        for i in range(2):
            W_236[k*2+i, :r] = W_233[k*2+i, :]
            W_236[(k+3)*2+i, r:] = W_233[k*2+i, :]
            
    # Verify the decomposition
    # Check a few random entries
    valid = True
    for _ in range(100):
        a = np.random.randint(0, 2)
        b = np.random.randint(0, 3)
        c = np.random.randint(0, 3)
        d = np.random.randint(0, 6)
        k = np.random.randint(0, 6)
        i = np.random.randint(0, 2)
        
        target = 1.0 if (a == i and c == k and b == d) else 0.0
        s = np.dot(U_236[a*3+b, :], np.dot(V_236[c*6+d, :], W_236[k*2+i, :]))
        if not np.isclose(s, target, atol=1e-6):
            valid = False
            break
            
    if not valid:
        # Fallback to the seed's rank 34 if greedy failed verification
        # Reconstruct seed's rank 34
        U_comb = np.zeros((6, 34))
        V_comb = np.zeros((18, 34))
        W_comb = np.zeros((12, 34))
        
        U233_seed = np.zeros((6, 17))
        V233_seed = np.zeros((9, 17))
        W233_seed = np.zeros((6, 17))
        
        # Copy seed data
        U233_seed[0, 0] = 1; U233_seed[4, 0] = 1
        V233_seed[0, 0] = 1; V233_seed[4, 0] = 1
        W233_seed[0, 0] = 1
        W233_seed[3, 0] = 1
        
        U233_seed[3, 1] = 1; U233_seed[4, 1] = 1
        V233_seed[0, 1] = 1
        W233_seed[1, 1] = 1
        W233_seed[3, 1] = -1
        
        U233_seed[0, 2] = 1
        V233_seed[1, 2] = 1; V233_seed[4, 2] = -1
        W233_seed[2, 2] = 1
        W233_seed[3, 2] = 1
        
        U233_seed[4, 3] = 1
        V233_seed[3, 3] = 1; V233_seed[0, 3] = -1
        W233_seed[0, 3] = 1
        W233_seed[1, 3] = 1
        
        U233_seed[0, 4] = 1; U233_seed[1, 4] = 1
        V233_seed[4, 4] = 1
        W233_seed[0, 4] = -1
        W233_seed[2, 4] = 1
        
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
                
        return {
            "U": U_comb.tolist(),
            "V": V_comb.tolist(),
            "W": W_comb.tolist()
        }
        
    return {
        "U": U_236.tolist(),
        "V": V_236.tolist(),
        "W": W_236.tolist()
    }

if __name__ == "__main__":
    construct()
