"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import itertools

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    
    def verify(U, V, W, n, m, p, tol=1e-6):
        """Verify that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k"""
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def verify_fast(U, V, W, n, m, p, tol=1e-6):
        """Fast vectorized verification using einsum"""
        r = U.shape[1]
        # Reshape U to (n, m, r), V to (m, p, r), W to (p, n, r)
        U_3d = U.reshape(n, m, r)
        V_3d = V.reshape(m, p, r)
        W_3d = W.reshape(p, n, r)
        # Compute tensor: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # Use einsum: 'ijr,jkr,kr->ijk' won't work directly due to index order
        # Let's do it step by step
        # First compute A[i,j,k] = sum_r U[i,j,r] * V[j,k,r]
        A = np.einsum('ijr,jkr->ijk', U_3d, V_3d)
        # Then T[i,j,k] = sum_r A[i,j,k] * W[k,i,r]... wait, W is indexed by [k*n+i, r]
        # W_3d[k,i,r] = W[k*n+i, r]
        # T[i,j,k] = sum_r A[i,j,k,r]? No, A already summed over r.
        # Actually: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Let's compute directly with broadcasting
        # U_rows: for each i,j -> row i*m+j
        # V_rows: for each j,k -> row j*p+k  
        # W_rows: for each k,i -> row k*n+i
        # T[i,j,k] = dot(U[i*m+j], V[j*p+k] * W[k*n+i])
        
        # Build index arrays
        i_idx = np.arange(n)[:, None, None]  # (n,1,1)
        j_idx = np.arange(m)[None, :, None]  # (1,m,1)
        k_idx = np.arange(p)[None, None, :]  # (1,1,p)
        
        # Row indices
        u_rows = i_idx * m + j_idx  # (n,m,1)
        v_rows = j_idx * p + k_idx  # (1,m,p)
        w_rows = k_idx * n + i_idx  # (n,1,p)
        
        # Broadcast to (n,m,p)
        u_rows = np.broadcast_to(u_rows, (n,m,p))
        v_rows = np.broadcast_to(v_rows, (n,m,p))
        w_rows = np.broadcast_to(w_rows, (n,m,p))
        
        # Gather
        U_sel = U[u_rows, :]  # (n,m,p,r)
        V_sel = V[v_rows, :]  # (n,m,p,r)
        W_sel = W[w_rows, :]  # (n,m,p,r)
        
        T = np.einsum('ijpr,ijpr,ijpr->ijp', U_sel, V_sel, W_sel)
        
        return np.allclose(T, 1.0, atol=tol)

    def als_step(U, V, W, n, m, p, r):
        """
        Perform one ALS iteration.
        """
        new_U = np.zeros_like(U)
        for i in range(n):
            for j in range(m):
                w_idx = np.arange(p) * n + i
                w_slice = W[w_idx, :]
                v_slice = V[j*p : (j+1)*p, :]
                A = v_slice * w_slice
                b = np.ones(p)
                try:
                    u_sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    new_U[i*m+j, :] = u_sol
                except np.linalg.LinAlgError:
                    new_U[i*m+j, :] = U[i*m+j, :]
        
        new_V = np.zeros_like(V)
        for j in range(m):
            for k in range(p):
                u_idx = np.arange(n) * m + j
                u_slice = U[u_idx, :]
                w_idx = k * n + np.arange(n)
                w_slice = W[w_idx, :]
                A = u_slice * w_slice
                b = np.ones(n)
                try:
                    v_sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    new_V[j*p+k, :] = v_sol
                except np.linalg.LinAlgError:
                    new_V[j*p+k, :] = V[j*p+k, :]
        
        new_W = np.zeros_like(W)
        for k in range(p):
            for i in range(n):
                u_idx = i * m + np.arange(m)
                u_slice = U[u_idx, :]
                v_idx = np.arange(m) * p + k
                v_slice = V[v_idx, :]
                A = u_slice * v_slice
                b = np.ones(m)
                try:
                    w_sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    new_W[k*n+i, :] = w_sol
                except np.linalg.LinAlgError:
                    new_W[k*n+i, :] = W[k*n+i, :]
        
        return new_U, new_V, new_W

    def try_rank_als(r, max_iters=30, n_restarts=5):
        if time.time() > deadline:
            return None
        
        best_err = float('inf')
        best_UVW = None
        
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            
            # Random initialization with different scales
            scale = 1.0 + restart * 0.5
            U = np.random.randn(n * m, r) * scale
            V = np.random.randn(m * p, r) * scale
            W = np.random.randn(p * n, r) * scale
            
            prev_err = float('inf')
            for it in range(max_iters):
                if time.time() > deadline:
                    break
                
                U, V, W = als_step(U, V, W, n, m, p, r)
                
                # Compute error
                if it % 5 == 0 or it == max_iters - 1:
                    if verify_fast(U, V, W, n, m, p, tol=1e-4):
                        # Try rounding
                        U_rounded = np.round(U * 2) / 2
                        V_rounded = np.round(V * 2) / 2
                        W_rounded = np.round(W * 2) / 2
                        if verify_fast(U_rounded, V_rounded, W_rounded, n, m, p, tol=1e-6):
                            return {'U': U_rounded.tolist(), 'V': V_rounded.tolist(), 'W': W_rounded.tolist()}
                        
                        # Also try without rounding if close enough
                        if verify_fast(U, V, W, n, m, p, tol=1e-8):
                            return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
            
            # Check final state
            if verify_fast(U, V, W, n, m, p, tol=1e-8):
                return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
        
        return None

    def try_greedy_merge():
        """Start from schoolbook and try to merge linearly dependent terms."""
        if time.time() > deadline:
            return None
        
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        
        # Try to find pairs of columns that can be merged
        # Two terms r1, r2 can be merged if their U, V, W vectors are linearly dependent
        # i.e., there exists alpha, beta not both zero such that alpha*U[:,r1] + beta*U[:,r2] = 0
        # and same for V and W
        
        merged = False
        new_U = U.copy()
        new_V = V.copy()
        new_W = W.copy()
        keep = [True] * rank
        
        for r1 in range(rank):
            if time.time() > deadline:
                break
            if not keep[r1]:
                continue
            for r2 in range(r1 + 1, rank):
                if time.time() > deadline:
                    break
                if not keep[r2]:
                    continue
                
                # Check if U[:,r1] and U[:,r2] are linearly dependent
                u1 = U[:, r1]
                u2 = U[:, r2]
                v1 = V[:, r1]
                v2 = V[:, r2]
                w1 = W[:, r1]
                w2 = W[:, r2]
                
                # Check rank of [u1, u2]
                try:
                    rank_u = np.linalg.matrix_rank(np.column_stack([u1, u2]))
                    rank_v = np.linalg.matrix_rank(np.column_stack([v1, v2]))
                    rank_w = np.linalg.matrix_rank(np.column_stack([w1, w2]))
                    
                    if rank_u <= 1 and rank_v <= 1 and rank_w <= 1:
                        # All three pairs are linearly dependent, can merge
                        # Find the relationship
                        # u2 = a * u1, v2 = b * v1, w2 = c * w1
                        if np.linalg.norm(u1) > 1e-10:
                            a = np.dot(u1, u2) / np.dot(u1, u1)
                        else:
                            continue
                        if np.linalg.norm(v1) > 1e-10:
                            b = np.dot(v1, v2) / np.dot(v1, v1)
                        else:
                            continue
                        if np.linalg.norm(w1) > 1e-10:
                            c = np.dot(w1, w2) / np.dot(w1, w1)
                        else:
                            continue
                        
                        # New term: U_new = u1 + u2, V_new = v1 + v2, W_new = w1 + w2
                        # But this doesn't preserve the product unless we scale
                        # Actually, we need to find alpha, beta such that:
                        # alpha*u1 + beta*u2 = 0, alpha*v1 + beta*v2 = 0, alpha*w1 + beta*w2 = 0
                        # This requires u1, v1, w1 to be proportional to u2, v2, w2
                        # with the same proportionality constant
                        if abs(a - b) < 1e-6 and abs(b - c) < 1e-6:
                            # Same proportionality, can merge
                            new_U[:, r1] = u1 + u2
                            new_V[:, r1] = v1 + v2
                            new_W[:, r1] = w1 + w2
                            keep[r2] = False
                            merged = True
                except:
                    pass
        
        if merged:
            # Compact
            keep_idx = [i for i in range(rank) if keep[i]]
            new_U = new_U[:, keep_idx]
            new_V = new_V[:, keep_idx]
            new_W = new_W[:, keep_idx]
            
            if verify_fast(new_U, new_V, new_W, n, m, p, tol=1e-6):
                return {'U': new_U.tolist(), 'V': new_V.tolist(), 'W': new_W.tolist()}
        
        return None

    best_result = schoolbook()
    best_rank = n * m * p
    
    lower_bound = max(n*m, m*p, p*n)
    
    # First try greedy merge
    res = try_greedy_merge()
    if res is not None:
        best_result = res
        best_rank = len(res['U'][0])
    
    # Try ALS from lower bound up
    for r in range(lower_bound, best_rank):
        if time.time() > deadline:
            break
        
        found = False
        n_attempts = 8 if r < 40 else 5
        for attempt in range(n_attempts):
            if time.time() > deadline:
                break
            
            res = try_rank_als(r, max_iters=25, n_restarts=1)
            if res is not None:
                best_result = res
                best_rank = r
                found = True
                break
        
        if found:
            break
    
    return best_result
# EVOLVE-BLOCK-END
