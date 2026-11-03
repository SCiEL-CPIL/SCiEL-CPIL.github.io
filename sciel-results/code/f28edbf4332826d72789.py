import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    deadline = time.time() + 140  # 140s safety margin
    
    # Strassen steps for 2x2x2:
    strassen = [
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}),   # M1
        ({2:1, 3:1}, {0:1}, {2:1, 3:-1}),        # M2
        ({0:1}, {1:1, 3:-1}, {1:1, 3:1}),        # M3
        ({3:1}, {2:1, 0:-1}, {0:1, 2:1}),        # M4
        ({0:1, 1:1}, {3:1}, {0:-1, 1:1}),        # M5
        ({2:1, 0:-1}, {0:1, 1:1}, {3:1}),        # M6
        ({1:1, 3:-1}, {2:1, 3:1}, {0:1}),        # M7
    ]
    
    def emit_block(a_cs, b_rs, b_cs, c_cs, off, U, V, W, rank):
        for s in range(7):
            col = off + s
            if col >= rank:
                continue
            uc, vc, wc = strassen[s]
            for li, c in uc.items():
                r, cc = li // 2, li % 2
                U[r * m + a_cs + cc][col] += c
            for li, c in vc.items():
                r, cc = li // 2, li % 2
                V[(b_rs + r) * p + b_cs + cc][col] += c
            for li, c in wc.items():
                r, cc = li // 2, li % 2
                W[(c_cs + cc) * n + r][col] += c
    
    def build_strassen_28():
        rank = 28
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        emit_block(0, 0, 0, 0, 0, U, V, W, rank)
        emit_block(0, 0, 2, 2, 7, U, V, W, rank)
        emit_block(2, 2, 0, 0, 14, U, V, W, rank)
        emit_block(2, 2, 2, 2, 21, U, V, W, rank)
        return U, V, W
    
    def verify_decomposition(U, V, W, n, m, p):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if s != 1:
                        return False
        return True
    
    def verify_decomposition_fast(U, V, W, n, m, p):
        """Vectorized verification using numpy."""
        rank = len(U[0])
        Ua = np.array(U, dtype=float)  # (n*m, rank)
        Va = np.array(V, dtype=float)  # (m*p, rank)
        Wa = np.array(W, dtype=float)  # (p*n, rank)
        
        # Reshape to (n, m, rank), (m, p, rank), (p, n, rank)
        Ua_r = Ua.reshape(n, m, rank)
        Va_r = Va.reshape(m, p, rank)
        Wa_r = Wa.reshape(p, n, rank)
        
        # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # Compute for all (i,j,k)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(Ua_r[i,j,:] * Va_r[j,k,:] * Wa_r[k,i,:])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def try_reduce_rank(U, V, W, n, m, p, max_iters=500):
        """Try to reduce rank by finding a linear dependency among rank-1 terms."""
        rank = len(U[0])
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        
        # Build tensor space representation
        # Each rank-1 term r: U[:,r] ⊗ V[:,r] ⊗ W[:,r]
        # Flatten into a vector of size n*m*p
        dim = n * m * p
        tensors = np.zeros((rank, dim))
        for r in range(rank):
            u = Ua[:, r]  # (n*m,)
            v = Va[:, r]  # (m*p,)
            w = Wa[:, r]  # (p*n,)
            # Reshape
            ur = u.reshape(n, m)
            vr = v.reshape(m, p)
            wr = w.reshape(p, n)
            # Tensor: T[i,j,k] = ur[i,j] * vr[j,k] * wr[k,i]
            t = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        t[i,j,k] = ur[i,j] * vr[j,k] * wr[k,i]
            tensors[r] = t.flatten()
        
        # Check if tensors are linearly independent
        # If rank < dim, there might be dependencies
        # Try to find if one tensor is in the span of others
        for target in range(rank):
            if time.time() > deadline - 30:
                break
            others = [t for t in range(rank) if t != target]
            if len(others) == 0:
                continue
            # Solve: tensors[target] = sum_{t in others} c_t * tensors[t]
            A = tensors[others, :]  # (len(others), dim)
            b = tensors[target]  # (dim,)
            try:
                # Use least squares
                coeffs, residuals, rankA, sv = np.linalg.lstsq(A, b, rcond=None)
                # Check if residual is small
                if residuals is not None and len(residuals) > 0:
                    res = residuals[0]
                else:
                    res = np.linalg.norm(A @ coeffs - b)
                if res < 1e-8:
                    # Found a dependency! Remove the target factor
                    new_rank = rank - 1
                    new_U = Ua[:, [t for t in range(rank) if t != target]]
                    new_V = Va[:, [t for t in range(rank) if t != target]]
                    new_W = Wa[:, [t for t in range(rank) if t != target]]
                    # Verify
                    if verify_decomposition_fast(new_U, new_V, new_W, n, m, p):
                        return (new_U.tolist(), new_V.tolist(), new_W.tolist(), new_rank)
            except Exception:
                continue
        return None
    
    U28, V28, W28 = build_strassen_28()
    
    # Verify the Strassen-28 decomposition
    if not verify_decomposition_fast(U28, V28, W28, n, m, p):
        # Fallback: return rank-32 trivial decomposition
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
        return {"U": U, "V": V, "W": W}
    
    # Try to reduce to rank 27
    result = try_reduce_rank(U28, V28, W28, n, m, p)
    if result is not None:
        U, V, W, r = result
        return {"U": U, "V": V, "W": W}
    
    # Return rank-28 solution
    return {"U": U28, "V": V28, "W": W28}
# EVOLVE-BLOCK-END
