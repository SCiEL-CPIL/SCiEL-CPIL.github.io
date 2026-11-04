"""Matrix multiplication 3x4x4 rank-minimal tensor decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145.0  # Safety margin before 150s kill
    
    # Build the matrix multiplication tensor
    # T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k; 0 otherwise
    # U shape: (n*m, rank), V shape: (m*p, rank), W shape: (p*n, rank)
    # Decomposition: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    
    # Schoolbook solution
    rank_school = n * m * p  # 48
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1.0
                V_school[j * p + k, col] = 1.0
                W_school[k * n + i, col] = 1.0
                col += 1
    
    best_result = None
    best_rank = rank_school
    
    # Precompute nonzero tensor entries
    # For each a = i*m+j, the nonzero (b,c) pairs are (j*p+k, k*n+i) for k in range(p)
    # For each b = j*p+k, the nonzero (a,c) pairs are (i*m+j, k*n+i) for i in range(n)
    # For each c = k*n+i, the nonzero (a,b) pairs are (i*m+j, j*p+k) for j in range(m)
    
    # Build lookup structures
    # For U update: for each a, list of (b, c) where T[a,b,c] = 1
    a_to_bc = {}
    for i in range(n):
        for j in range(m):
            a = i * m + j
            pairs = []
            for k in range(p):
                b = j * p + k
                c = k * n + i
                pairs.append((b, c))
            a_to_bc[a] = pairs
    
    # For V update: for each b, list of (a, c) where T[a,b,c] = 1
    b_to_ac = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                b = j * p + k
                a = i * m + j
                c = k * n + i
                if b not in b_to_ac:
                    b_to_ac[b] = []
                b_to_ac[b].append((a, c))
    
    # For W update: for each c, list of (a, b) where T[a,b,c] = 1
    c_to_ab = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = k * n + i
                a = i * m + j
                b = j * p + k
                if c not in c_to_ab:
                    c_to_ab[c] = []
                c_to_ab[c].append((a, b))
    
    def verify(U, V, W, tol=1e-8):
        """Check if U,V,W exactly decompose the tensor."""
        # Check nonzero entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    pred = float(np.dot(U[a, :], V[b, :] * W[c, :]))
                    if abs(pred - 1.0) > tol:
                        return False
        # Check some zero entries (sample)
        # For a = i*m+j, b = j'*p+k, c = k'*n+i' where j' != j or i' != i
        # These should be 0
        # Check all zero entries for correctness
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    # Check if this is a nonzero entry
                    is_nonzero = False
                    for i in range(n):
                        for j in range(m):
                            if i * m + j == a:
                                for k in range(p):
                                    if j * p + k == b and k * n + i == c:
                                        is_nonzero = True
                    if not is_nonzero:
                        pred = float(np.dot(U[a, :], V[b, :] * W[c, :]))
                        if abs(pred) > tol:
                            return False
        return True
    
    def verify_fast(U, V, W, tol=1e-8):
        """Faster verification using einsum."""
        # Compute full tensor prediction
        pred = np.einsum('ar,br,cr->abc', U, V, W)
        # Build target tensor
        T = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i * m + j, j * p + k, k * n + i] = 1.0
        return np.max(np.abs(pred - T)) < tol
    
    def als_optimize(target_rank, seed, max_iters=100):
        """Alternating least squares for tensor decomposition."""
        np.random.seed(seed)
        
        # Initialize with random values
        U = np.random.randn(n * m, target_rank) * 0.5
        V = np.random.randn(m * p, target_rank) * 0.5
        W = np.random.randn(p * n, target_rank) * 0.5
        
        # Normalize to avoid scale issues
        U = U / np.max(np.abs(U)) * 1.0
        V = V / np.max(np.abs(V)) * 1.0
        W = W / np.max(np.abs(W)) * 1.0
        
        prev_loss = float('inf')
        
        for it in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            
            # Update U: for each a, solve linear system
            # For fixed a, minimize sum_{b,c} (T[a,b,c] - sum_r U[a,r]*V[b,r]*W[c,r])^2
            # This is a least squares problem: y = X * u
            # where y[b,c] = T[a,b,c], X[b,c,r] = V[b,r]*W[c,r]
            # But we only need to consider all (b,c) pairs since V,W are dense
            # Actually, we can use the structure: for fixed a=i*m+j,
            # T[a,b,c] = 1 only when b=j*p+k, c=k*n+i
            # So y has p nonzero entries (all 1) and rest are 0
            
            for a in range(n * m):
                i, j = divmod(a, m)
                # Build design matrix for this a
                # For each (b,c) pair, the row of X is [V[b,0]*W[c,0], V[b,1]*W[c,1], ...]
                # We need to solve: minimize ||X*u - y||^2
                # X has shape (m*p*p, target_rank) -- too large
                # Instead, use the fact that only p entries of y are nonzero
                # And the gradient/Hessian can be computed efficiently
                
                # Alternative: use the normal equations
                # X^T X * u = X^T y
                # X^T X = sum_{b,c} outer(V[b,:]*W[c,:], V[b,:]*W[c,:])
                # = sum_{b,c} (V[b,:]*W[c,:]) (V[b,:]*W[c,:])^T
                # = (sum_c V[b,:] V[b,:]^T W[c,:] W[c,:]^T) ... this doesn't factor nicely
                
                # Let's just build the full X for this a
                # Number of (b,c) pairs is m*p * p*n = 4*4 * 4*3 = 48
                # That's manageable
                X = np.zeros((m * p * p * n, target_rank))
                y = np.zeros(m * p * p * n)
                idx = 0
                for b in range(m * p):
                    for c in range(p * n):
                        X[idx, :] = V[b, :] * W[c, :]
                        # Check if T[a,b,c] = 1
                        # a = i*m+j, b = j'*p+k, c = k'*n+i'
                        # T = 1 if j'==j and i'==i (and k is free, but b determines k via b=j'*p+k)
                        # Wait: a = i*m+j fixes i,j
                        # b = j'*p+k, c = k'*n+i'
                        # T[a,b,c] = 1 iff j'==j AND i'==i (k and k' are linked: k=k')
                        # Actually no: T[i*m+j, j'*p+k, k'*n+i'] = 1 iff j'==j and i'==i and k==k'
                        # Hmm, let me reconsider.
                        # C[i,k] = sum_j A[i,j] B[j,k]
                        # So the tensor T is defined as: T[i*m+j, j'*p+k, k'*n+i'] = 1 
                        #   if and only if j'==j and i'==i and k==k'
                        # Wait, that's not right either. Let me think again.
                        # The multiplication tensor maps A (n x m) and B (m x p) to C (n x p).
                        # C[i,k] = sum_j A[i,j] * B[j,k]
                        # So the trilinear form is: sum_{i,j,k} C[i,k] * A[i,j] * B[j,k] = sum_{i,j,k} A[i,j] * B[j,k] * C[i,k]
                        # The tensor T has T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
                        # And T[a,b,c] = 0 otherwise.
                        # So for fixed a = i*m+j:
                        #   T[a,b,c] = 1 when b = j*p+k and c = k*n+i for some k in range(p)
                        #   That is, for each k, b = j*p+k, c = k*n+i
                        #   There are exactly p such (b,c) pairs.
                        # For other (b,c), T[a,b,c] = 0.
                        
                        # So for fixed a, y has p entries equal to 1 and (m*p*p*n - p) entries equal to 0
                        # But we don't need all of them. The zero entries contribute to X^T X but not X^T y.
                        # For efficiency, let's use the full set but it's only 48 entries.
                        
                        # Check if T[a,b,c] = 1
                        # a = i*m+j
                        # b = j'*p+k, c = k'*n+i'
                        # T = 1 iff j' == j and i' == i and k == k'
                        jp, kp = divmod(b, p)  # b = jp*p + kp
                        kn, ip = divmod(c, n)  # c = kn*n + ip
                        if jp == j and ip == i and kp == kn:
                            y[idx] = 1.0
                        idx += 1
                
                # Solve least squares: X u = y
                # Use numpy's lstsq
                u_sol, residuals, rank_x, sv = np.linalg.lstsq(X, y, rcond=None)
                U[a, :] = u_sol
            
            # Update V: for each b, solve linear system
            for b in range(m * p):
                jp, kp = divmod(b, p)  # b = jp*p + kp, so j' = jp, k = kp
                # T[a,b,c] = 1 when a = i*m+jp and c = kp*n+i for some i in range(n)
                # So for fixed b, there are n nonzero entries
                X = np.zeros((n * m * p * n, target_rank))
                y = np.zeros(n * m * p * n)
                idx = 0
                for a in range(n * m):
                    for c in range(p * n):
                        X[idx, :] = U[a, :] * W[c, :]
                        # Check if T[a,b,c] = 1
                        # a = i*m+j, c = k'*n+i'
                        # T = 1 iff j == jp (from b) and i' == i and k == kp (from b) and k' == kp
                        # So: j == jp, i' == i, k' == kp
                        im, jm = divmod(a, m)
                        kn, ip = divmod(c, n)
                        if jm == jp and ip == im and kn == kp:
                            y[idx] = 1.0
                        idx += 1
                
                v_sol, residuals, rank_x, sv = np.linalg.lstsq(X, y, rcond=None)
                V[b, :] = v_sol
            
            # Update W: for each c, solve linear system
            for c in range(p * n):
                kn, ip = divmod(c, n)  # c = kn*n + ip, so k' = kn, i' = ip
                # T[a,b,c] = 1 when a = ip*m+j and b = j*p+kn for some j in range(m)
                # So for fixed c, there are m nonzero entries
                X = np.zeros((n * m * m * p, target_rank))
                y = np.zeros(n * m * m * p)
                idx = 0
                for a in range(n * m):
                    for b in range(m * p):
                        X[idx, :] = U[a, :] * V[b, :]
                        # Check if T[a,b,c] = 1
                        # a = i*m+j, b = j'*p+k
                        # T = 1 iff i == ip (from c) and j' == j and k == kn (from c)
                        im, jm = divmod(a, m)
                        jp, kp = divmod(b, p)
                        if im == ip and jp == jm and kp == kn:
                            y[idx] = 1.0
                        idx += 1
                
                w_sol, residuals, rank_x, sv = np.linalg.lstsq(X, y, rcond=None)
                W[c, :] = w_sol
            
            # Check convergence
            # Compute loss
            loss = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        pred = float(np.dot(U[a, :], V[b, :] * W[c, :]))
                        loss += (pred - 1.0) ** 2
            # Also add zero entries (sample)
            # For simplicity, just check nonzero entries for convergence
            
            if abs(prev_loss - loss) < 1e-10 * max(1.0, prev_loss):
                break
            prev_loss = loss
        
        return U, V, W
    
    # Try ranks from low to high
    # Known lower bound: for matmul(n,m,p), rank >= max(n*m, m*p, p*n) / something
    # Actually, the minimum rank is at least max(n, m, p) * min(n, m, p) in some cases
    # For 3x4x4: let's start from a reasonable lower bound
    # The schoolbook rank is 48. Let's try to go lower.
    
    # Start from rank 24 and go up, trying multiple seeds
    for target_rank in range(24, rank_school + 1):
        if time.time() > deadline - 10.0:
            break
        
        # Try multiple random initializations
        n_trials = max(1, min(10, int((deadline - time.time()) / 12.0)))
        
        for trial in range(n_trials):
            if time.time() > deadline - 5.0:
                break
            
            seed = 42 + target_rank * 1000 + trial * 7
            U, V, W = als_optimize(target_rank, seed, max_iters=80)
            
            # Verify
            if verify_fast(U, W, V, tol=1e-8):
                # Wait, verify_fast expects U, V, W in that order
                if verify_fast(U, V, W, tol=1e-8):
                    best_rank = target_rank
                    # Convert to lists
                    U_out = [[float(x) for x in row] for row in U]
                    V_out = [[float(x) for x in row] for row in V]
                    W_out = [[float(x) for x in row] for row in W]
                    best_result = {"U": U_out, "V": V_out, "W": W_out}
                    break
        
        if best_result is not None:
            break
    
    if best_result is not None:
        return best_result
    
    # Fallback to schoolbook
    U_out = [[float(x) for x in row] for row in U_school]
    V_out = [[float(x) for x in row] for row in V_school]
    W_out = [[float(x) for x in row] for row in W_school]
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
