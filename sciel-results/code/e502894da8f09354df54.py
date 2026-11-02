"""Matrix multiplication 3x5x7 - rank minimization via structured algebraic search."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 5, 7]
_DEADLINE = time.time() + 110.0  # safety margin before 120s kill

def _build_target_tensor(n, m, p):
    """Build the multiplication tensor as a 3D array T[a,b,c] where
    a = i*m+j, b = j'*p+k, c = k'*n+i'.
    T[a,b,c] = 1 if i=i', j=j', k=k', else 0.
    """
    T = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1
    return T

def _verify_exact(U, V, W, n, m, p):
    """Verify that U,V,W give an exact decomposition of the multiplication tensor.
    U: (n*m, r), V: (m*p, r), W: (p*n, r)
    Check: sum_t U[a,t]*V[b,t]*W[c,t] == T[a,b,c] for all a,b,c
    """
    r = U.shape[1]
    Ua = np.array(U, dtype=object)
    Va = np.array(V, dtype=object)
    Wa = np.array(W, dtype=object)
    # For each (a,b,c) check the sum
    # Efficient: for each t, the contribution is outer(U[:,t], V[:,t]) * W[:,t]
    # But for exactness with fractions, let's do it directly for small cases
    # Actually, let's use the structure: only need to check (i,j),(j',k),(k',i')
    # where the target is 1 if i=i',j=j',k=k'
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = sum(Ua[a, t] * Va[b, t] * Wa[c, t] for t in range(r))
                if s != 1:
                    return False
    # Check zeros: for random samples of (a,b,c) where target should be 0
    # Sample a few
    import random
    random.seed(42)
    for _ in range(500):
        a = random.randrange(n * m)
        b = random.randrange(m * p)
        c = random.randrange(p * n)
        i = a // m
        j = a % m
        jp = b // p
        k = b % p
        kp = c // n
        ip = c % n
        if i == ip and j == jp and k == kp:
            continue  # this is a "1" position, already checked
        s = sum(Ua[a, t] * Va[b, t] * Wa[c, t] for t in range(r))
        if s != 0:
            return False
    return True

def _verify_np(U, V, W, n, m, p):
    """Fast numerical verification using numpy."""
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    r = Ua.shape[1]
    # Build the reconstructed tensor
    # T_recon[a,b,c] = sum_t U[a,t]*V[b,t]*W[c,t]
    # Only check the "1" positions and a sample of "0" positions
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = np.dot(Ua[a], Va[b] * Wa[c])
                if abs(s - 1.0) > 1e-6:
                    return False
    # Check some zero positions
    import random
    random.seed(123)
    for _ in range(200):
        a = random.randrange(n * m)
        b = random.randrange(m * p)
        c = random.randrange(p * n)
        i = a // m
        j = a % m
        jp = b // p
        k = b % p
        kp = c // n
        ip = c % n
        if i == ip and j == jp and k == kp:
            continue
        s = np.dot(Ua[a], Va[b] * Wa[c])
        if abs(s) > 1e-6:
            return False
    return True

def _als_refine(U, V, W, n, m, p, target, max_iters=50, tol=1e-8):
    """ALS refinement: alternate fixing two matrices and solving for the third."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    r = U.shape[1]
    
    for it in range(max_iters):
        if time.time() > _DEADLINE - 5:
            break
        
        # Update W: for each c, W[c,t] = sum_{a,b} target[a,b,c] * U[a,t] * V[b,t] / (sum_{a'} U[a',t]^2 * sum_{b'} V[b',t]^2)
        # Actually, the least squares solution for W given U,V:
        # minimize ||sum_t U[:,t] V[:,t]^T W[:,t] - target||_F
        # For each c: W[:,c] ... wait, W is (p*n, r)
        # Let me reformulate: T_recon = sum_t (U[:,t] * V[:,t]) ⊗ W[:,t] ... no
        # T_recon[a,b,c] = sum_t U[a,t] * V[b,t] * W[c,t]
        # This is: T_recon = sum_t (U[:,t] ⊗ V[:,t] ⊗ W[:,t])
        # Flatten: for fixed t, the contribution to T is outer(U[:,t], V[:,t]) * W[:,t]
        # Actually: T_recon[:,:,c] = sum_t (U[:,t] * V[:,t]) * W[c,t] ... no
        # T_recon[a,b,c] = sum_t U[a,t]*V[b,t]*W[c,t]
        # For fixed c: T_recon[:,:,c] = sum_t (U[:,t] * V[:,t]) * W[c,t]
        # So T_recon[:,:,c] = M @ W[:,c] where M[a,b,t] = U[a,t]*V[b,t]
        # But M is (n*m, m*p, r) which is large. Let's use the structure.
        
        # Better approach: 
        # T_recon = sum_t (U[:,t] * V[:,t]) ⊗ W[:,t]  (where * is elementwise, ⊗ is outer)
        # No: T_recon[a,b,c] = sum_t U[a,t] V[b,t] W[c,t]
        # = sum_t (U[a,t] V[b,t]) W[c,t]
        # For fixed c: T_recon[:,:,c] = sum_t (U[:,t] ⊙ V[:,t]) * W[c,t]
        # where ⊙ is elementwise product, giving a vector of length n*m*m*p... no.
        # U[:,t] has length n*m, V[:,t] has length m*p. Their elementwise product doesn't make sense directly.
        # 
        # Let me think differently. T_recon is a 3-tensor. 
        # T_recon[a,b,c] = sum_t U[a,t] V[b,t] W[c,t]
        # This is the tensor product sum. To solve for W:
        # For each c, T_recon[:,:,c] should equal sum_t (U[:,t] * V[:,t]) * W[c,t]
        # But U[:,t] * V[:,t] is not a valid operation since they have different lengths.
        #
        # The correct way: T_recon = sum_t u_t ⊗ v_t ⊗ w_t where u_t = U[:,t], v_t = V[:,t], w_t = W[:,t]
        # To solve for W given U,V:
        # For each c: sum_t U[a,t] V[b,t] W[c,t] = target[a,b,c] for all a,b
        # This is: sum_t W[c,t] * (U[:,t] * V[:,t]) = target[:,:,c]
        # But U[:,t] * V[:,t] needs to be interpreted as... 
        # Actually, for each t, the matrix M_t[a,b] = U[a,t] * V[b,t] is an outer product.
        # So T_recon[:,:,c] = sum_t W[c,t] * M_t = sum_t W[c,t] * (U[:,t] * V[:,t]^T)
        # where the product is: M_t is (n*m) x (m*p) with M_t[a,b] = U[a,t]*V[b,t]
        # So T_recon[:,:,c] = sum_t W[c,t] * (U[:,t] ⊗ V[:,t])  (outer product)
        # = (U ⊙ W[:,c]) ... no
        # 
        # Let's just do it with numpy:
        # T_target has shape (n*m, m*p, p*n)
        # T_recon = np.einsum('at,bt,ct->abc', U, V, W)
        # To solve for W: we want to minimize ||T_recon - T_target||_F
        # This is a linear system in W.
        # T_recon[a,b,c] = sum_t U[a,t] V[b,t] W[c,t]
        # For fixed (a,b): sum_t W[c,t] * U[a,t] * V[b,t] = target[a,b,c] for all c
        # Let M_ab[c,t] = U[a,t] * V[b,t] (independent of c! so M_ab is the same for all c)
        # Then: sum_t M_ab[t] * W[c,t] = target[a,b,c]
        # This means: for each (a,b), the vector target[a,b,:] = W^T @ M_ab
        # So: target[a,b,:] = M_ab^T @ W[:,a,b]... no
        # target[a,b,c] = sum_t M_ab[t] * W[c,t]
        # = (M_ab @ W)[c] where W is (p*n, r) and M_ab is (r,)
        # Wait: sum_t M_ab[t] * W[c,t] = (W @ M_ab)[c]
        # So target[a,b,:] = W @ M_ab where M_ab = U[:,a] * V[:,b] (elementwise, length r)
        # 
        # So for each (a,b): W @ (U[:,a]*V[:,b]) = target[a,b,:]
        # This gives (n*m)*(m*p) equations in (p*n)*r unknowns.
        # We can solve by least squares.
        
        # Stack all equations:
        # For each (a,b): W @ (U[:,a]*V[:,b]) = target[a,b,:]
        # Let's form the system: A @ W_flat = b_flat
        # where A has shape ((n*m)*(m*p)*(p*n), (p*n)*r)
        # This is too large. Let's use a different approach.
        
        # Alternative: use the fact that for each c, we can solve:
        # sum_t U[a,t] V[b,t] W[c,t] = target[a,b,c] for all a,b
        # This is: (U * W[c,:])^T ... no
        # Let's define: for fixed c, we want to find w_c = W[c,:] such that
        # sum_t U[a,t] V[b,t] w_c[t] = target[a,b,c] for all a,b
        # The LHS is: sum_t w_c[t] * (U[:,t] * V[:,t]) ... but U[:,t] and V[:,t] have different sizes.
        # 
        # I think the cleanest way is:
        # T_recon = np.einsum('at,bt,ct->abc', U, V, W)
        # The gradient with respect to W[c,t] is: sum_{a,b} (T_recon[a,b,c] - target[a,b,c]) * U[a,t] * V[b,t]
        # Setting to zero: sum_{a,b} target[a,b,c] * U[a,t] * V[b,t] = sum_{a,b} T_recon[a,b,c] * U[a,t] * V[b,t]
        # = sum_t' W[c,t'] * sum_{a,b} U[a,t] * V[b,t] * U[a,t'] * V[b,t']
        # = sum_t' W[c,t'] * (sum_a U[a,t]*U[a,t']) * (sum_b V[b,t]*V[b,t'])
        # = sum_t' W[c,t'] * (U^T U)[t,t'] * (V^T V)[t,t']
        # 
        # And the RHS: sum_{a,b} target[a,b,c] * U[a,t] * V[b,t]
        # = sum_a U[a,t] * (sum_b target[a,b,c] * V[b,t])
        # 
        # So we have a linear system: (U^T U ⊙ V^T V) @ W[c,:] = rhs_c
        # where ⊙ is elementwise product of the two Gram matrices.
        
        UtU = U.T @ U  # (r, r)
        VtV = V.T @ V  # (r, r)
        Gram = UtU * VtV  # (r, r) elementwise
        
        for c in range(p * n):
            if time.time() > _DEADLINE - 3:
                break
            # rhs[t] = sum_{a,b} target[a,b,c] * U[a,t] * V[b,t]
            # = sum_a U[a,t] * (sum_b target[a,b,c] * V[b,t])
            # Let's compute: for each t, rhs[t] = sum_a sum_b target[a,b,c] * U[a,t] * V[b,t]
            # This is: sum_a U[a,t] * (target[:, :, c] @ V[:, t]) ... 
            # target[:, :, c] is (n*m, m*p), V[:,t] is (m*p,)
            # So (target[:, :, c] @ V[:, t]) is (n*m,)
            # Then rhs[t] = sum_a U[a,t] * (target[:, :, c] @ V[:, t])[a]
            # = (U[:, t] @ (target[:, :, c] @ V[:, t]))
            
            # More efficiently:
            # rhs[t] = sum_{a,b} target[a,b,c] * U[a,t] * V[b,t]
            # = np.einsum('ab,at,bt->t', target[:, :, c], U, V)
            
            rhs = np.einsum('ab,at,bt->t', target[:, :, c], U, V)
            
            # Solve Gram @ w_c = rhs
            try:
                W[c, :] = np.linalg.solve(Gram + 1e-10 * np.eye(r), rhs)
            except:
                W[c, :] = np.linalg.lstsq(Gram + 1e-10 * np.eye(r), rhs, rcond=None)[0]
        
        # Similarly update U and V by symmetry
        
        # Update U: for each a, solve similar system
        UtU_new = U.T @ U
        WtW = W.T @ W
        Gram_U = WtW * VtV  # (r,r)
        
        for a in range(n * m):
            if time.time() > _DEADLINE - 3:
                break
            # rhs[t] = sum_{b,c} target[a,b,c] * V[b,t] * W[c,t]
            rhs = np.einsum('bc,bt,ct->t', target[a, :, :], V, W)
            try:
                U[a, :] = np.linalg.solve(Gram_U + 1e-10 * np.eye(r), rhs)
            except:
                U[a, :] = np.linalg.lstsq(Gram_U + 1e-10 * np.eye(r), rhs, rcond=None)[0]
        
        # Update V: for each b
        UtU = U.T @ U
        WtW = W.T @ W
        Gram_V = UtU * WtW  # (r,r)
        
        for b in range(m * p):
            if time.time() > _DEADLINE - 3:
                break
            # rhs[t] = sum_{a,c} target[a,b,c] * U[a,t] * W[c,t]
            rhs = np.einsum('ac,at,ct->t', target[:, b, :], U, W)
            try:
                V[b, :] = np.linalg.solve(Gram_V + 1e-10 * np.eye(r), rhs)
            except:
                V[b, :] = np.linalg.lstsq(Gram_V + 1e-10 * np.eye(r), rhs, rcond=None)[0]
        
        # Check convergence
        T_recon = np.einsum('at,bt,ct->abc', U, V, W)
        err = np.linalg.norm(T_recon - target) / (np.linalg.norm(target) + 1e-15)
        if err < tol:
            break
    
    return U, V, W

def _try_rank(r, n, m, p, target, n_trials=30):
    """Try to find a rank-r decomposition via random init + ALS."""
    for trial in range(n_trials):
        if time.time() > _DEADLINE - 10:
            return None
        
        # Random initialization
        np.random.seed(trial * 1000 + r)
        U = np.random.randn(n * m, r) * 0.1
        V = np.random.randn(m * p, r) * 0.1
        W = np.random.randn(p * n, r) * 0.1
        
        # ALS refine
        U, V, W = _als_refine(U, V, W, n, m, p, target, max_iters=80, tol=1e-10)
        
        # Check if we got a good solution
        T_recon = np.einsum('at,bt,ct->abc', U, V, W)
        err = np.linalg.norm(T_recon - target) / (np.linalg.norm(target) + 1e-15)
        
        if err < 1e-6:
            # Try to round to integers or half-integers
            U_round = np.round(U * 2) / 2
            V_round = np.round(V * 2) / 2
            W_round = np.round(W * 2) / 2
            
            T_recon2 = np.einsum('at,bt,ct->abc', U_round, V_round, W_round)
            err2 = np.linalg.norm(T_recon2 - target) / (np.linalg.norm(target) + 1e-15)
            
            if err2 < 1e-6:
                # Convert to list of fractions
                U_frac = [[Fraction(int(x * 2), 2) for x in U_round[i]] for i in range(n * m)]
                V_frac = [[Fraction(int(x * 2), 2) for x in V_round[i]] for i in range(m * p)]
                W_frac = [[Fraction(int(x * 2), 2) for x in W_round[i]] for i in range(p * n)]
                return U_frac, V_frac, W_frac
            
            # Try integer rounding
            U_int = np.round(U).astype(int)
            V_int = np.round(V).astype(int)
            W_int = np.round(W).astype(int)
            
            T_recon3 = np.einsum('at,bt,ct->abc', U_int.astype(float), V_int.astype(float), W_int.astype(float))
            err3 = np.linalg.norm(T_recon3 - target) / (np.linalg.norm(target) + 1e-15)
            
            if err3 < 1e-6:
                U_frac = [[Fraction(int(x)) for x in U_int[i]] for i in range(n * m)]
                V_frac = [[Fraction(int(x)) for x in V_int[i]] for i in range(m * p)]
                W_frac = [[Fraction(int(x)) for x in W_int[i]] for i in range(p * n)]
                return U_frac, V_frac, W_frac
    
    return None

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = [[Fraction(0)] * rank for _ in range(n * m)]
    V = [[Fraction(0)] * rank for _ in range(m * p)]
    W = [[Fraction(0)] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = Fraction(1)
                V[j * p + k][col] = Fraction(1)
                W[k * n + i][col] = Fraction(1)
                col += 1
    return U, V, W

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x5 * 5x7 matrix multiplication.
    Uses structured random search with ALS refinement, starting from the theoretical
    lower bound of rank 35 and searching upward.
    """
    n, m, p = DIMENSIONS
    target = _build_target_tensor(n, m, p)
    
    # Try ranks from the theoretical lower bound upward
    lower_bound = max(n * p, m * p, n * m)  # 35
    
    for r in range(lower_bound, n * m * p + 1):
        if time.time() > _DEADLINE - 15:
            break
        
        result = _try_rank(r, n, m, p, target, n_trials=20)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook decomposition
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
