"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 4
    deadline = time.time() + 130  # safety margin before 150s kill

    def build_tensor():
        T = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1.0
        return T

    T = build_tensor()

    def verify(U, V, W, tol=1e-8):
        recon = np.einsum('at,bt,ct->abc', U, V, W)
        return np.max(np.abs(recon - T)) < tol

    def to_lists(U, V, W, r):
        U_list = [[float(U[i, t]) for t in range(r)] for i in range(n * m)]
        V_list = [[float(V[i, t]) for t in range(r)] for i in range(m * p)]
        W_list = [[float(W[i, t]) for t in range(r)] for i in range(p * n)]
        return {"U": U_list, "V": V_list, "W": W_list}

    def strassen_2x2_decomp():
        U = np.zeros((4, 7)); V = np.zeros((4, 7)); W = np.zeros((4, 7))
        U[0,0]=1.0; U[3,0]=1.0; V[0,0]=1.0; V[3,0]=1.0; W[0,0]=1.0; W[3,0]=1.0
        U[2,1]=1.0; U[3,1]=1.0; V[0,1]=1.0; W[1,1]=1.0; W[3,1]=-1.0
        U[0,2]=1.0; V[1,2]=1.0; V[3,2]=-1.0; W[2,2]=1.0; W[3,2]=1.0
        U[3,3]=1.0; V[2,3]=1.0; V[0,3]=-1.0; W[0,3]=1.0; W[1,3]=1.0
        U[0,4]=1.0; U[1,4]=1.0; V[3,4]=1.0; W[0,4]=-1.0; W[2,4]=1.0
        U[2,5]=1.0; U[0,5]=-1.0; V[0,5]=1.0; V[1,5]=1.0; W[3,5]=1.0
        U[1,6]=1.0; U[3,6]=-1.0; V[2,6]=1.0; V[3,6]=1.0; W[0,6]=1.0
        return U, V, W

    def build_strassen28():
        U_s, V_s, W_s = strassen_2x2_decomp()
        full_rank = 28
        U_full = np.zeros((n * m, full_rank))
        V_full = np.zeros((m * p, full_rank))
        W_full = np.zeros((p * n, full_rank))
        blocks = [
            (lambda li,lj: li*4+lj,          lambda lj,lk: lj*4+lk,          lambda lk,li: lk*2+li),
            (lambda li,lj: li*4+(lj+2),      lambda lj,lk: (lj+2)*4+lk,      lambda lk,li: lk*2+li),
            (lambda li,lj: li*4+lj,          lambda lj,lk: lj*4+(lk+2),      lambda lk,li: (lk+2)*2+li),
            (lambda li,lj: li*4+(lj+2),      lambda lj,lk: (lj+2)*4+(lk+2),  lambda lk,li: (lk+2)*2+li),
        ]
        for bi,(ga,gb,gc) in enumerate(blocks):
            for t in range(7):
                col = bi*7+t
                for li in range(2):
                    for lj in range(2):
                        U_full[ga(li,lj), col] = U_s[li*2+lj, t]
                for lj in range(2):
                    for lk in range(2):
                        V_full[gb(lj,lk), col] = V_s[lj*2+lk, t]
                for lk in range(2):
                    for li in range(2):
                        W_full[gc(lk,li), col] = W_s[lk*2+li, t]
        return U_full, V_full, W_full

    U28, V28, W28 = build_strassen28()
    
    # Verify rank-28 works
    if not verify(U28, V28, W28, tol=1e-8):
        # fallback to schoolbook
        U = np.zeros((n*m, 32)); V = np.zeros((m*p, 32)); W = np.zeros((p*n, 32))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col]=1.0; V[j*p+k, col]=1.0; W[k*n+i, col]=1.0; col+=1
        return to_lists(U, V, W, 32)

    # === Attempt rank-27 by finding and eliminating a linear dependency ===
    # The 28 rank-1 tensors u_t ⊗ v_t ⊗ w_t may have a linear relation.
    # Flatten each term: term_t = U[:,t] ⊗ V[:,t] ⊗ W[:,t] as a vector in R^(n*m * m*p * p*n)
    # If sum alpha_t * term_t = 0 with some alpha not all zero, we can remove one term.
    
    def find_dependency(U, V, W, r):
        """Find a linear dependency among the r rank-1 tensors.
        Returns (coeffs, removed_index) or None."""
        dim = (n*m) * (m*p) * (p*n)
        # Build matrix where each column is the flattened rank-1 tensor
        M = np.zeros((dim, r))
        for t in range(r):
            M[:, t] = np.einsum('a,b,c->abc', U[:,t], V[:,t], W[:,t]).flatten()
        
        # Find null space
        try:
            _, s, vh = np.linalg.svd(M, full_matrices=False)
            # The smallest singular value indicates near-dependency
            if s[-1] < 1e-10:
                # There's an exact dependency
                null_vec = vh[-1, :]
                # Find the index with largest |coeff| to remove
                idx = np.argmax(np.abs(null_vec))
                return null_vec, idx
        except np.linalg.LinAlgError:
            pass
        return None

    dep = find_dependency(U28, V28, W28, 28)
    if dep is not None:
        coeffs, rem_idx = dep
        # Remove the column at rem_idx
        keep = [i for i in range(28) if i != rem_idx]
        U27 = U28[:, keep]
        V27 = V28[:, keep]
        W27 = W28[:, keep]
        if verify(U27, V27, W27, tol=1e-8):
            return to_lists(U27, V27, W27, 27)

    # === Alternative: modify 2 columns to absorb a dependency ===
    # Strategy: fix 26 columns, solve for 2 columns to exactly match T
    # The residual after 26 columns must be a sum of 2 rank-1 tensors
    
    def try_rank27_by_solve():
        """Try to find rank-27 by fixing 26 Strassen columns and solving for 2."""
        r = 28
        # Try different pairs of columns to free
        for free1 in range(r):
            for free2 in range(free1+1, r):
                if time.time() > deadline:
                    return None
                fixed = [i for i in range(r) if i != free1 and i != free2]
                Uf = U28[:, fixed]
                Vf = V28[:, fixed]
                Wf = W28[:, fixed]
                
                # Residual to match
                recon_fixed = np.einsum('at,bt,ct->abc', Uf, Vf, Wf)
                residual = T - recon_fixed
                
                # We need residual = u1⊗v1⊗w1 + u2⊗v2⊗w2
                # This is a system of nonlinear equations. Use a simpler approach:
                # Assume we can express residual as sum of 2 rank-1 terms by
                # choosing v1, v2, w1, w2 first, then solving for u1, u2.
                
                # Simpler: use the fact that for the specific structure of this tensor,
                # the dependency is known. Let's try a direct algebraic approach.
                # 
                # Actually, let's use a different strategy: 
                # The 2x4x4 tensor has rank 27. The Strassen construction gives 28.
                # The extra rank comes from the fact that the 4 blocks share structure.
                # 
                # Let's try: use 3 full Strassen blocks (21 rank) + 1 modified block (6 rank) = 27
                # Or: use a different block decomposition entirely.
                
                # For now, try random initialization + gradient descent for the 2 free columns
                rng = np.random.RandomState(free1 * 100 + free2)
                U_free = rng.randn(n*m, 2) * 0.3
                V_free = rng.randn(m*p, 2) * 0.3
                W_free = rng.randn(p*n, 2) * 0.3
                
                base = recon_fixed
                
                for it in range(2000):
                    if time.time() > deadline:
                        break
                    recon = base + np.einsum('at,bt,ct->abc', U_free, V_free, W_free)
                    F = recon - T
                    gU = np.einsum('abc,bt,ct->at', F, V_free, W_free)
                    gV = np.einsum('abc,at,ct->bt', F, U_free, W_free)
                    gW = np.einsum('abc,at,bt->ct', F, U_free, V_free)
                    err = float(np.sum(F*F))
                    lr = 0.1 / (1.0 + err)
                    U_free -= lr * gU
                    V_free -= lr * gV
                    W_free -= lr * gW
                    
                    if err < 1e-14:
                        # Snap to half-integers
                        U_free = np.round(U_free * 2) / 2
                        V_free = np.round(V_free * 2) / 2
                        W_free = np.round(W_free * 2) / 2
                        # Build full matrices
                        U_full = np.zeros((n*m, r))
                        V_full = np.zeros((m*p, r))
                        W_full = np.zeros((p*n, r))
                        for fi, fcol in enumerate([free1, free2]):
                            U_full[:, fcol] = U_free[:, fi]
                            V_full[:, fcol] = V_free[:, fi]
                            W_full[:, fcol] = W_free[:, fi]
                        for fi, fcol in enumerate(fixed):
                            U_full[:, fcol] = U28[:, fcol]
                            V_full[:, fcol] = V28[:, fcol]
                            W_full[:, fcol] = W28[:, fcol]
                        if verify(U_full, V_full, W_full, tol=1e-8):
                            return (U_full, V_full, W_full)
                        break
        return None

    result = try_rank27_by_solve()
    if result is not None:
        Uf, Vf, Wf = result
        return to_lists(Uf, Vf, Wf, 28)

    # Fallback: verified rank-28 Strassen
    return to_lists(U28, V28, W28, 28)
# EVOLVE-BLOCK-END
