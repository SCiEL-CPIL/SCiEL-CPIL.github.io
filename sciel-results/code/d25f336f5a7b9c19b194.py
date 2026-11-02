import time
import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for 2x5 * 5x5 matrix multiplication.
    
    Uses Strassen's algorithm on 2x2x2 sub-blocks to reduce rank from 50 to 46,
    then attempts further reduction via optimized alternating minimization.
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
        Optimized Alternating Minimization: fix two factors, solve for the third.
        Uses incremental residual updates to reduce computation.
        """
        U = U.copy()
        V = V.copy()
        W = W.copy()
        R = U.shape[1]
        
        # Precompute initial Rec
        Rec = np.einsum('ar,br,cr->abc', U, V, W)
        prev_err = np.max(np.abs(T - Rec))
        
        for it in range(max_iter):
            if time.time() > deadline - 10:
                break
            
            # Solve for U
            for r in range(R):
                # Rec_without_r = Rec - U[:,r] outer V[:,r] outer W[:,r]
                term_r = np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
                Rec_without_r = Rec - term_r
                
                # diff = T - Rec_without_r
                diff = T - Rec_without_r
                
                # num = (diff * V[:,r] * W[:,r]).sum(axis=(1,2))
                num = np.einsum('abc,b,c->a', diff, V[:,r], W[:,r])
                den = np.dot(V[:,r], V[:,r]) * np.dot(W[:,r], W[:,r])
                
                if den > 1e-12:
                    U_new = num / den
                    # Update Rec: Rec = Rec_without_r + U_new outer V[:,r] outer W[:,r]
                    term_new = np.einsum('a,b,c->abc', U_new, V[:,r], W[:,r])
                    Rec = Rec_without_r + term_new
                    U[:,r] = U_new
                else:
                    U[:,r] = 0
                    Rec = Rec_without_r

            # Solve for V
            for r in range(R):
                term_r = np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
                Rec_without_r = Rec - term_r
                
                diff = T - Rec_without_r
                
                num = np.einsum('abc,a,c->b', diff, U[:,r], W[:,r])
                den = np.dot(U[:,r], U[:,r]) * np.dot(W[:,r], W[:,r])
                
                if den > 1e-12:
                    V_new = num / den
                    term_new = np.einsum('a,b,c->abc', U[:,r], V_new, W[:,r])
                    Rec = Rec_without_r + term_new
                    V[:,r] = V_new
                else:
                    V[:,r] = 0
                    Rec = Rec_without_r

            # Solve for W
            for r in range(R):
                term_r = np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
                Rec_without_r = Rec - term_r
                
                diff = T - Rec_without_r
                
                num = np.einsum('abc,a,b->c', diff, U[:,r], V[:,r])
                den = np.dot(U[:,r], U[:,r]) * np.dot(V[:,r], V[:,r])
                
                if den > 1e-12:
                    W_new = num / den
                    term_new = np.einsum('a,b,c->abc', U[:,r], V[:,r], W_new)
                    Rec = Rec_without_r + term_new
                    W[:,r] = W_new
                else:
                    W[:,r] = 0
                    Rec = Rec_without_r
            
            # Check convergence
            err = np.max(np.abs(T - Rec))
            if abs(prev_err - err) < tol:
                break
            prev_err = err
        
        return U, V, W

    def try_reduce_rank(U, V, W, max_reduction=5):
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
            
            for attempt in range(2): # Reduced attempts to save time
                if time.time() > deadline - 15:
                    break
                
                if attempt == 0:
                    # Remove smallest
                    keep_indices = np.sort(sorted_indices[-new_R:])
                else:
                    # Remove random subset
                    all_indices = np.arange(R)
                    remove_count = reduction
                    remove_indices = np.random.choice(all_indices, size=remove_count, replace=False)
                    keep_indices = np.sort(np.setdiff1d(all_indices, remove_indices))
                
                U_new = U[:, keep_indices].copy()
                V_new = V[:, keep_indices].copy()
                W_new = W[:, keep_indices].copy()
                
                # Run alternating minimization
                U_new, V_new, W_new = alternating_minimize(U_new, V_new, W_new, max_iter=200)
                
                # Verify
                valid, err = verify_decomposition(U_new, V_new, W_new)
                if valid:
                    best_U, best_V, best_W = U_new, V_new, W_new
                    best_rank = new_R
                    found_valid = True
                    break
            
            if not found_valid:
                # If not valid, stop trying higher reductions for this strategy
                # We don't break the outer loop immediately because maybe a different reduction size works?
                # But usually if 1 fails, 2 is harder.
                # However, sometimes removing 1 specific term fails but removing 2 works? Unlikely.
                # Let's break to save time.
                break
        
        return best_U, best_V, best_W, best_rank
    
    def polish_decomposition(U, V, W, max_iter=200, tol=1e-12):
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
            U, V, W, R = try_reduce_rank(U, V, W, max_reduction=5)
        
        # Final polish
        if time.time() < deadline - 5:
            U, V, W = polish_decomposition(U, V, W, max_iter=200)
    
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
