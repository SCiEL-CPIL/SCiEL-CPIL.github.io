import time
import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for 2x5 * 5x5 matrix multiplication.
    
    Uses Strassen's algorithm on 2x2x2 sub-blocks to reduce rank from 50 to 46,
    then attempts further reduction via alternating minimization and local search.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
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
    
    def verify_decomposition(U, V, W, tol=1e-10):
        """Verify that U, V, W exactly decompose the tensor T."""
        Rec = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.max(np.abs(T - Rec))
        return err < tol, err
    
    def alternating_minimize(U, V, W, max_iter=500, tol=1e-8):
        """
        Alternating minimization: fix two factors, solve for the third.
        This is more effective than gradient descent for tensor decomposition.
        """
        U = U.copy()
        V = V.copy()
        W = W.copy()
        prev_err = np.inf
        
        for it in range(max_iter):
            if time.time() > deadline - 10:
                break
            
            # Solve for U: for each column r, U[:,r] = (V[:,r] * W[:,r])^T * T_resid
            # More precisely: minimize ||T - sum_r U[:,r] outer(V[:,r], W[:,r])||
            # For fixed V,W: U[:,r] = argmin ||T - sum_s ... + U[:,r] outer(V[:,r], W[:,r])||
            # This gives: U[:,r] = (T contracted with V[:,r] and W[:,r]) / (||V[:,r]||^2 * ||W[:,r]||^2)
            
            for r in range(U.shape[1]):
                # Compute residual without term r
                Rec = np.einsum('ar,br,cr->abc', U, V, W)
                diff = T - Rec + np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
                
                # Solve for U[:,r]: minimize ||diff - U[:,r] outer(V[:,r], W[:,r])||
                # Gradient: -2 * diff * outer(V[:,r], W[:,r]) + 2 * U[:,r] * (||V[:,r]||^2 * ||W[:,r]||^2) = 0
                # U[:,r] = (diff * V[:,r] * W[:,r]).sum(axis=(1,2)) / (||V[:,r]||^2 * ||W[:,r]||^2)
                
                num = np.einsum('abc,b,c->a', diff, V[:,r], W[:,r])
                den = np.dot(V[:,r], V[:,r]) * np.dot(W[:,r], W[:,r])
                
                if den > 1e-12:
                    U[:,r] = num / den
            
            # Solve for V
            for r in range(V.shape[1]):
                Rec = np.einsum('ar,br,cr->abc', U, V, W)
                diff = T - Rec + np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
                
                num = np.einsum('abc,a,c->b', diff, U[:,r], W[:,r])
                den = np.dot(U[:,r], U[:,r]) * np.dot(W[:,r], W[:,r])
                
                if den > 1e-12:
                    V[:,r] = num / den
            
            # Solve for W
            for r in range(W.shape[1]):
                Rec = np.einsum('ar,br,cr->abc', U, V, W)
                diff = T - Rec + np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
                
                num = np.einsum('abc,a,b->c', diff, U[:,r], V[:,r])
                den = np.dot(U[:,r], U[:,r]) * np.dot(V[:,r], V[:,r])
                
                if den > 1e-12:
                    W[:,r] = num / den
            
            # Check convergence
            Rec = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.max(np.abs(T - Rec))
            if abs(prev_err - err) < tol:
                break
            prev_err = err
        
        return U, V, W
    
    def try_reduce_rank(U, V, W, max_reduction=8):
        """Try to reduce rank by removing terms and using alternating minimization."""
        R = U.shape[1]
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_rank = R
        
        # Compute norms to identify potentially removable terms
        norms = np.linalg.norm(U, axis=0) * np.linalg.norm(V, axis=0) * np.linalg.norm(W, axis=0)
        sorted_indices = np.argsort(norms)
        
        for reduction in range(1, max_reduction + 1):
            if time.time() > deadline - 15:
                break
                
            new_R = R - reduction
            if new_R < 1:
                break
                
            # Try multiple strategies for which terms to remove
            found_valid = False
            
            # Strategy 1: Remove smallest norm terms
            keep_indices = np.sort(sorted_indices[-new_R:])
            
            for attempt in range(3):
                if time.time() > deadline - 15:
                    break
                
                if attempt == 0:
                    # Remove smallest
                    keep_indices = np.sort(sorted_indices[-new_R:])
                elif attempt == 1:
                    # Remove random subset
                    all_indices = np.arange(R)
                    remove_count = reduction
                    remove_indices = np.random.choice(all_indices, size=remove_count, replace=False)
                    keep_indices = np.sort(np.setdiff1d(all_indices, remove_indices))
                else:
                    # Remove middle-norm terms (keep extremes)
                    keep_indices = np.sort(np.concatenate([sorted_indices[:new_R//2], sorted_indices[-(new_R - new_R//2):]]))
                
                U_new = U[:, keep_indices].copy()
                V_new = V[:, keep_indices].copy()
                W_new = W[:, keep_indices].copy()
                
                # Run alternating minimization
                U_new, V_new, W_new = alternating_minimize(U_new, V_new, W_new, max_iter=300)
                
                # Verify
                valid, err = verify_decomposition(U_new, V_new, W_new)
                if valid:
                    best_U, best_V, best_W = U_new, V_new, W_new
                    best_rank = new_R
                    found_valid = True
                    break
            
            if not found_valid:
                # If not valid, try one more time with different random seed
                np.random.seed(int(time.time() * 1000) % 10000)
                all_indices = np.arange(R)
                remove_indices = np.random.choice(all_indices, size=reduction, replace=False)
                keep_indices = np.sort(np.setdiff1d(all_indices, remove_indices))
                
                U_new = U[:, keep_indices].copy()
                V_new = V[:, keep_indices].copy()
                W_new = W[:, keep_indices].copy()
                
                U_new, V_new, W_new = alternating_minimize(U_new, V_new, W_new, max_iter=300)
                
                valid, err = verify_decomposition(U_new, V_new, W_new)
                if valid:
                    best_U, best_V, best_W = U_new, V_new, W_new
                    best_rank = new_R
                else:
                    # If not valid, stop trying higher reductions
                    break
        
        return best_U, best_V, best_W, best_rank
    
    def polish_decomposition(U, V, W, max_iter=1000, tol=1e-12):
        """Final high-precision polish pass with tighter tolerance."""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            # Alternating minimization with tight tolerance
            for r in range(U.shape[1]):
                Rec = np.einsum('ar,br,cr->abc', U, V, W)
                diff = T - Rec + np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
                
                num = np.einsum('abc,b,c->a', diff, V[:,r], W[:,r])
                den = np.dot(V[:,r], V[:,r]) * np.dot(W[:,r], W[:,r])
                
                if den > 1e-12:
                    U[:,r] = num / den
            
            for r in range(V.shape[1]):
                Rec = np.einsum('ar,br,cr->abc', U, V, W)
                diff = T - Rec + np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
                
                num = np.einsum('abc,a,c->b', diff, U[:,r], W[:,r])
                den = np.dot(U[:,r], U[:,r]) * np.dot(W[:,r], W[:,r])
                
                if den > 1e-12:
                    V[:,r] = num / den
            
            for r in range(W.shape[1]):
                Rec = np.einsum('ar,br,cr->abc', U, V, W)
                diff = T - Rec + np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
                
                num = np.einsum('abc,a,b->c', diff, U[:,r], V[:,r])
                den = np.dot(U[:,r], U[:,r]) * np.dot(V[:,r], V[:,r])
                
                if den > 1e-12:
                    W[:,r] = num / den
            
            # Check convergence
            Rec = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.max(np.abs(T - Rec))
            if err < tol:
                break
        
        return U, V, W
    
    # Strategy: Use Strassen on 2x2x2 blocks where possible
    terms = []
    
    def add_strassen(i_start, j_start, k_start):
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
    
    def add_schoolbook_term(i, j, k):
        """Add a single schoolbook term."""
        u = {(i, j): 1}
        v = {(j, k): 1}
        w = {(k, i): 1}
        terms.append((u, v, w))
    
    # Decomposition plan for 2x5x5:
    add_strassen(0, 0, 0)
    add_strassen(0, 0, 2)
    add_strassen(0, 2, 0)
    add_strassen(0, 2, 2)
    
    # j in {0,1,2,3}, k=4
    for j in range(4):
        for i in range(2):
            add_schoolbook_term(i, j, 4)
    
    # j=4, k in {0,1,2,3,4}
    for k in range(5):
        for i in range(2):
            add_schoolbook_term(i, 4, k)
    
    # Build the decomposition
    U, V, W = build_decomposition_from_terms(terms)
    R = U.shape[1]
    
    # Verify
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
        # Try to reduce rank further
        if time.time() < deadline - 25:
            U, V, W, R = try_reduce_rank(U, V, W, max_reduction=8)
        
        # Final polish
        if time.time() < deadline - 5:
            U, V, W = polish_decomposition(U, V, W, max_iter=500)
    
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
# EVOLVE-BLOCK-END
