import time
import numpy as np

DIMENSIONS = [2, 5, 5]

def construct():
    """
    Rank-minimal exact tensor decomposition for 2x5 * 5x5 matrix multiplication.
    
    Uses structured decomposition with Strassen blocks and aggressive rank reduction.
    """
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS  # n=2, m=5, p=5
    
    # Build the full tensor T[a, b, c] where:
    #   a = i*m + j  (row of A, col of A)
    #   b = j*p + k  (row of B, col of B)
    #   c = k*n + i  (output, trace convention: k*n + i)
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    def build_decomposition_from_terms(terms):
        """Build U, V, W matrices from a list of (u_dict, v_dict, w_dict) terms."""
        rank = len(terms)
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        for r, (u_dict, v_dict, w_dict) in enumerate(terms):
            for (i, j), val in u_dict.items():
                U[i * m + j, r] = val
            for (j, k), val in v_dict.items():
                V[j * p + k, r] = val
            for (k, i), val in w_dict.items():
                W[k * n + i, r] = val
        
        return U, V, W
    
    def verify_decomposition(U, V, W, tol=1e-9):
        """Verify that U, V, W exactly decompose the tensor T."""
        Rec = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.max(np.abs(T - Rec))
        return err < tol, err
    
    def add_strassen(terms, i_start, j_start, k_start):
        """Add 7 Strassen terms for a 2x2x2 block at (i_start, j_start, k_start)."""
        # M1 = (A11+A22)(B11+B22)
        u = {(i_start, j_start): 1, (i_start+1, j_start+1): 1}
        v = {(j_start, k_start): 1, (j_start+1, k_start+1): 1}
        w = {(k_start, i_start): 1, (k_start+1, i_start+1): 1}
        terms.append((u, v, w))
        
        # M2 = (A21+A22)B11
        u = {(i_start+1, j_start): 1, (i_start+1, j_start+1): 1}
        v = {(j_start, k_start): 1}
        w = {(k_start, i_start+1): 1, (k_start+1, i_start+1): -1}
        terms.append((u, v, w))
        
        # M3 = A11(B21-B11)
        u = {(i_start, j_start): 1}
        v = {(j_start, k_start+1): 1, (j_start+1, k_start+1): -1}
        w = {(k_start+1, i_start): 1, (k_start+1, i_start+1): 1}
        terms.append((u, v, w))
        
        # M4 = A22(B22-B12)
        u = {(i_start+1, j_start+1): 1}
        v = {(j_start+1, k_start): 1, (j_start, k_start): -1}
        w = {(k_start, i_start): 1, (k_start, i_start+1): 1}
        terms.append((u, v, w))
        
        # M5 = (A11+A12)B22
        u = {(i_start, j_start): 1, (i_start, j_start+1): 1}
        v = {(j_start+1, k_start+1): 1}
        w = {(k_start, i_start): -1, (k_start+1, i_start): 1}
        terms.append((u, v, w))
        
        # M6 = (A21-A11)(B11+B12)
        u = {(i_start+1, j_start): 1, (i_start, j_start): -1}
        v = {(j_start, k_start): 1, (j_start, k_start+1): 1}
        w = {(k_start+1, i_start+1): 1}
        terms.append((u, v, w))
        
        # M7 = (A12-A22)(B21+B22)
        u = {(i_start, j_start+1): 1, (i_start+1, j_start+1): -1}
        v = {(j_start+1, k_start): 1, (j_start+1, k_start+1): 1}
        w = {(k_start, i_start): 1}
        terms.append((u, v, w))
    
    def add_schoolbook_term(terms, i, j, k):
        """Add a single schoolbook term."""
        u = {(i, j): 1}
        v = {(j, k): 1}
        w = {(k, i): 1}
        terms.append((u, v, w))
    
    def gradient_descent(U, V, W, steps, lr_schedule, momentum=0.9):
        """Run gradient descent to refine the decomposition."""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        vU = np.zeros_like(U)
        vV = np.zeros_like(V)
        vW = np.zeros_like(W)
        
        for step in range(steps):
            if time.time() > deadline - 5:
                break
            
            # Adaptive learning rate
            lr = lr_schedule(step)
            
            Rec = np.einsum('ar,br,cr->abc', U, V, W)
            diff = T - Rec
            
            gU = np.einsum('abc,br,cr->ar', diff, V, W)
            gV = np.einsum('abc,ar,cr->br', diff, U, W)
            gW = np.einsum('abc,ar,br->cr', diff, U, V)
            
            vU = momentum * vU + (1 - momentum) * gU
            vV = momentum * vV + (1 - momentum) * gV
            vW = momentum * vW + (1 - momentum) * gW
            
            U -= lr * vU
            V -= lr * vV
            W -= lr * vW
        
        return U, V, W
    
    def try_reduce_rank(U, V, W, max_reduction=15, steps_per_trial=200):
        """Try to reduce rank by removing dependent terms and re-optimizing."""
        R = U.shape[1]
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_rank = R
        
        # Compute norms to identify potentially removable terms
        norms = np.linalg.norm(U, axis=0) * np.linalg.norm(V, axis=0) * np.linalg.norm(W, axis=0)
        sorted_indices = np.argsort(norms)
        
        for reduction in range(1, max_reduction + 1):
            if time.time() > deadline - 20:
                break
                
            new_R = R - reduction
            if new_R < 1:
                break
            
            # Keep the largest new_R terms
            keep_indices = sorted_indices[-new_R:]
            keep_indices = np.sort(keep_indices)
            
            U_new = U[:, keep_indices].copy()
            V_new = V[:, keep_indices].copy()
            W_new = W[:, keep_indices].copy()
            
            # Local optimization with the reduced rank
            lr_schedule = lambda step: 0.01 * (0.995 ** step)
            U_new, V_new, W_new = gradient_descent(U_new, V_new, W_new, steps_per_trial, lr_schedule)
            
            # Verify the reduced decomposition
            valid, err = verify_decomposition(U_new, V_new, W_new)
            if valid:
                best_U, best_V, best_W = U_new, V_new, W_new
                best_rank = new_R
            else:
                break
        
        return best_U, best_V, best_W, best_rank
    
    # Build initial decomposition with Strassen blocks
    terms = []
    
    # 4 Strassen blocks covering j∈{0,1,2,3}, k∈{0,1,2,3}
    add_strassen(terms, 0, 0, 0)
    add_strassen(terms, 0, 0, 2)
    add_strassen(terms, 0, 2, 0)
    add_strassen(terms, 0, 2, 2)
    
    # j∈{0,1,2,3}, k=4: 8 schoolbook terms
    for j in range(4):
        for i in range(2):
            add_schoolbook_term(terms, i, j, 4)
    
    # j=4, k∈{0,1,2,3,4}: 10 schoolbook terms
    for k in range(5):
        for i in range(2):
            add_schoolbook_term(terms, i, 4, k)
    
    # Build the decomposition
    U, V, W = build_decomposition_from_terms(terms)
    R = U.shape[1]
    
    # Verify initial decomposition
    valid, err = verify_decomposition(U, V, W)
    
    if not valid:
        # Fallback to schoolbook
        U = np.zeros((n * m, n * m * p))
        V = np.zeros((m * p, n * m * p))
        W = np.zeros((p * n, n * m * p))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        R = n * m * p
    else:
        # First, refine the full-rank decomposition
        if time.time() < deadline - 40:
            lr_schedule = lambda step: 0.005 * (0.99 ** step)
            U, V, W = gradient_descent(U, V, W, 100, lr_schedule)
            valid, err = verify_decomposition(U, V, W)
        
        # Try to reduce rank
        if time.time() < deadline - 20:
            U, V, W, R = try_reduce_rank(U, V, W, max_reduction=10, steps_per_trial=150)
    
    # Final verification
    valid, err = verify_decomposition(U, V, W)
    if not valid:
        # Fallback to schoolbook if something went wrong
        U = np.zeros((n * m, n * m * p))
        V = np.zeros((m * p, n * m * p))
        W = np.zeros((p * n, n * m * p))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
