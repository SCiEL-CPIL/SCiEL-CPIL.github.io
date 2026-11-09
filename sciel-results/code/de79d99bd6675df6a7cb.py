# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = 3, 3, 4
    
    # Precompute indices
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    # For a given a (i, j), the non-zero (b, c) pairs are:
    # k in 0..p-1
    # b = j*p + k
    # c = k*n + i
    
    # Store for each a, the list of (b, c)
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
            
    # Also need to check zero entries.
    # Zero entries are all (a, b, c) not in the support.
    # For check, we can iterate all a, b, c and skip if in support.
    # Or, we can just check the support for 1s and a sample of zeros.
    # But to be safe, check all zeros if time permits.
    
    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1
                    V[b, col] = 1
                    W[c, col] = 1
                    col += 1
        return U.tolist(), V.tolist(), W.tolist()

    def check(U, V, W, tol=1e-9):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Check non-zero entries
        for a in range(n * m):
            for b, c in a_to_bc[a]:
                val = np.dot(U_arr[a], V_arr[b] * W_arr[c])
                if abs(val - 1.0) > tol:
                    return False
        
        # Check zero entries
        # To be efficient, we can check a random subset or all.
        # Total entries: 9 * 12 * 12 = 1296.
        # Non-zero: 36.
        # Zero: 1260.
        # 1260 * R dot products.
        # If R=30, 37800 ops. Fast.
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    # Check if (a,b,c) is in support
                    is_nz = False
                    for bb, cc in a_to_bc[a]:
                        if bb == b and cc == c:
                            is_nz = True
                            break
                    if not is_nz:
                        val = np.dot(U_arr[a], V_arr[b] * W_arr[c])
                        if abs(val) > tol:
                            return False
        return True

    def round_to_half(arr):
        return np.round(arr * 2) / 2

    def als_decompose(rank, n_iters=100, n_restarts=10):
        for restart in range(n_restarts):
            if time.time() > deadline - 10:
                return None
                
            U = np.random.randn(n * m, rank) * 0.1
            V = np.random.randn(m * p, rank) * 0.1
            W = np.random.randn(p * n, rank) * 0.1
            
            for it in range(n_iters):
                if time.time() > deadline - 10:
                    return None
                    
                # Update U
                for a in range(n * m):
                    for r in range(rank):
                        num = 0.0
                        den = 0.0
                        for b, c in a_to_bc[a]:
                            vw = V[b, r] * W[c, r]
                            num += vw  # T[a,b,c] is 1
                            den += vw * vw
                        if den > 1e-12:
                            U[a, r] = num / den
                        else:
                            U[a, r] = 0.0

                # Update V
                for b in range(m * p):
                    for r in range(rank):
                        num = 0.0
                        den = 0.0
                        # Find a, c such that (a,b,c) is in support
                        # b = j*p + k => j = b//p, k = b%p
                        # c = k*n + i => i = (c - k*n) ? No, c depends on i.
                        # For fixed b (fixed j, k), a = i*m + j, c = k*n + i.
                        # i ranges 0..n-1.
                        j = b // p
                        k = b % p
                        for i in range(n):
                            a = i * m + j
                            c = k * n + i
                            uw = U[a, r] * W[c, r]
                            num += uw
                            den += uw * uw
                        if den > 1e-12:
                            V[b, r] = num / den
                        else:
                            V[b, r] = 0.0

                # Update W
                for c in range(p * n):
                    for r in range(rank):
                        num = 0.0
                        den = 0.0
                        # c = k*n + i => k = c//n, i = c%n
                        # a = i*m + j, b = j*p + k.
                        # j ranges 0..m-1.
                        k = c // n
                        i = c % n
                        for j in range(m):
                            a = i * m + j
                            b = j * p + k
                            uv = U[a, r] * V[b, r]
                            num += uv
                            den += uv * uv
                        if den > 1e-12:
                            W[c, r] = num / den
                        else:
                            W[c, r] = 0.0

            # Round and check
            U_r = round_to_half(U)
            V_r = round_to_half(V)
            W_r = round_to_half(W)
            
            if check(U_r.tolist(), V_r.tolist(), W_r.tolist()):
                return U_r.tolist(), V_r.tolist(), W_r.tolist()
                
        return None

    # Try ranks from low to high
    # 36 is schoolbook
    # Try 24, 25, ..., 35
    best_U, best_V, best_W = get_schoolbook()
    
    for r in range(24, 36):
        if time.time() > deadline - 20:
            break
        res = als_decompose(r, n_iters=50, n_restarts=5)
        if res:
            best_U, best_V, best_W = res
            break # Found a lower rank, return it
            
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
