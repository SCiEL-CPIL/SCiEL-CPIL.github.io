import time
import numpy as np
from scipy.linalg import null_space

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    # 150s wall-clock kill; keep generous safety margin
    DEADLINE = start_time + 135.0

    def now():
        return time.time()

    def remaining():
        return DEADLINE - now()

    U_dim = n * m  # 10
    V_dim = m * p  # 25
    W_dim = p * n  # 10

    # Build target tensor T[u,v,w] where u=i*m+j, v=j*p+k, w=k*n+i
    T = np.zeros((U_dim, V_dim, W_dim))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def contract(U, V, W):
        return np.einsum('ar,br,cr->abc', U, V, W, optimize=True)

    def is_exact(U, V, W, atol=1e-7):
        prod = contract(U, V, W)
        return np.allclose(prod, T, atol=atol)

    def rank_of(U, V, W):
        return U.shape[1]

    def to_output(U, V, W):
        return {
            "U": [[float(x) for x in row] for row in U],
            "V": [[float(x) for x in row] for row in V],
            "W": [[float(x) for x in row] for row in W],
        }

    def build_rank46():
        rank = 46
        U = np.zeros((U_dim, rank))
        V = np.zeros((V_dim, rank))
        W = np.zeros((W_dim, rank))

        def add_strassen(j_off, k_off, col_start):
            c = col_start
            # P1 = (A00 + A11)(B00 + B11)
            U[0 * m + j_off][c] += 1
            U[1 * m + (j_off + 1)][c] += 1
            V[j_off * p + k_off][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] += 1
            W[k_off * n + 0][c] += 1
            W[(k_off + 1) * n + 1][c] += 1
            c += 1
            # P2 = (A10 + A11) B00
            U[1 * m + j_off][c] += 1
            U[1 * m + (j_off + 1)][c] += 1
            V[j_off * p + k_off][c] += 1
            W[k_off * n + 1][c] += 1
            W[(k_off + 1) * n + 1][c] -= 1
            c += 1
            # P3 = A00 (B01 - B11)
            U[0 * m + j_off][c] += 1
            V[j_off * p + (k_off + 1)][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] -= 1
            W[(k_off + 1) * n + 0][c] += 1
            W[(k_off + 1) * n + 1][c] += 1
            c += 1
            # P4 = A11 (B10 - B00)
            U[1 * m + (j_off + 1)][c] += 1
            V[(j_off + 1) * p + k_off][c] += 1
            V[j_off * p + k_off][c] -= 1
            W[k_off * n + 0][c] += 1
            W[k_off * n + 1][c] += 1
            c += 1
            # P5 = (A00 + A01) B11
            U[0 * m + j_off][c] += 1
            U[0 * m + (j_off + 1)][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] += 1
            W[k_off * n + 0][c] -= 1
            W[(k_off + 1) * n + 0][c] += 1
            c += 1
            # P6 = (A10 - A00)(B00 + B01)
            U[1 * m + j_off][c] += 1
            U[0 * m + j_off][c] -= 1
            V[j_off * p + k_off][c] += 1
            V[j_off * p + (k_off + 1)][c] += 1
            W[(k_off + 1) * n + 1][c] += 1
            c += 1
            # P7 = (A01 - A11)(B10 + B11)
            U[0 * m + (j_off + 1)][c] += 1
            U[1 * m + (j_off + 1)][c] -= 1
            V[(j_off + 1) * p + k_off][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] += 1
            W[k_off * n + 0][c] += 1

        add_strassen(0, 0, 0)
        add_strassen(0, 2, 7)
        add_strassen(2, 0, 14)
        add_strassen(2, 2, 21)

        col = 28
        for i in range(2):
            for k in range(4):
                U[i * m + 4][col] = 1
                V[4 * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
        for i in range(2):
            for j in range(4):
                U[i * m + j][col] = 1
                V[j * p + 4][col] = 1
                W[4 * n + i][col] = 1
                col += 1
        for i in range(2):
            U[i * m + 4][col] = 1
            V[4 * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1
        return U, V, W

    # Verify baseline
    U_base, V_base, W_base = build_rank46()
    if not is_exact(U_base, V_base, W_base):
        rank = n * m * p
        U = np.zeros((U_dim, rank))
        V = np.zeros((V_dim, rank))
        W = np.zeros((W_dim, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return to_output(U, V, W)

    best_rank = 46
    best_result = (U_base, V_base, W_base)

    def snap(arr):
        return np.round(arr * 2) / 2

    # ------------------------------------------------------------------
    # Lattice-based search: exploit bilinear structure
    # For fixed U, the constraint sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c] is linear in V*W
    # More practically: for each candidate rank R, build the coefficient matrix where
    # each column of (U,V,W) defines a bilinear form. We search over structured
    # integer/half-integer patterns.
    # ------------------------------------------------------------------
    
    def try_rank_R(R, seed, max_iters=50):
        """Try to find a valid decomposition of rank R using structured search."""
        if R >= best_rank:
            return None
        rng = np.random.RandomState(seed)
        
        # Strategy: start from rank-46 baseline, try to merge columns
        # by finding linear dependencies among the rank-1 tensors
        # Each rank-1 term is u_r ⊗ v_r ⊗ w_r
        # If sum_{r in S} alpha_r * (u_r ⊗ v_r ⊗ w_r) = 0, we can reduce rank
        
        # Build the tensor for each column of baseline
        # Then look for null space vectors that allow column elimination
        
        # Alternative: direct random search with better initialization
        for it in range(max_iters):
            if now() > DEADLINE - 8:
                return None
            
            # Random initialization: pick R random combinations
            U = rng.randn(U_dim, R) * 0.5
            V = rng.randn(V_dim, R) * 0.5
            W = rng.randn(W_dim, R) * 0.5
            
            # Gradient descent
            mU = np.zeros_like(U); vU = np.zeros_like(U)
            mV = np.zeros_like(V); vV = np.zeros_like(V)
            mW = np.zeros_like(W); vW = np.zeros_like(W)
            lr = 0.03
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            
            for t in range(80):
                if now() > DEADLINE - 6:
                    break
                prod = contract(U, V, W)
                diff = prod - T
                loss = float(np.sum(diff * diff))
                if loss < 1e-10:
                    U, V, W = snap(U), snap(V), snap(W)
                    if is_exact(U, V, W):
                        return (U, V, W)
                    break
                gU = np.einsum('abc,br,cr->ar', diff, V, W, optimize=True)
                gV = np.einsum('abc,ar,cr->br', diff, U, W, optimize=True)
                gW = np.einsum('abc,ar,br->cr', diff, U, V, optimize=True)
                tt = t + 1
                mU = beta1*mU + (1-beta1)*gU; vU = beta2*vU + (1-beta2)*gU**2
                mV = beta1*mV + (1-beta1)*gV; vV = beta2*vV + (1-beta2)*gV**2
                mW = beta1*mW + (1-beta1)*gW; vW = beta2*vW + (1-beta2)*gW**2
                U -= lr*(mU/(1-beta1**tt))/(np.sqrt(vU/(1-beta2**tt))+eps)
                V -= lr*(mV/(1-beta1**tt))/(np.sqrt(vV/(1-beta2**tt))+eps)
                W -= lr*(mW/(1-beta1**tt))/(np.sqrt(vW/(1-beta2**tt))+eps)
                lr *= 0.99
                
            U2, V2, W2 = snap(U), snap(V), snap(W)
            if is_exact(U2, V2, W2):
                return (U2, V2, W2)
        return None

    # Try lattice-inspired structured search for lower ranks
    seed = 0
    for R in [45, 44, 43, 42, 41, 40]:
        if now() > DEADLINE - 15:
            break
        if R >= best_rank:
            continue
        
        alloc = min(20.0, max(3.0, remaining() * 0.3))
        max_iters = max(2, int(alloc / 0.5))
        
        found = None
        for s in range(6):
            if now() > DEADLINE - 10:
                break
            found = try_rank_R(R, seed + s, max_iters)
            if found is not None:
                break
        
        if found is not None:
            U_r, V_r, W_r = found
            if is_exact(U_r, V_r, W_r):
                best_rank = R
                best_result = (U_r, V_r, W_r)

    # Return best
    U_f, V_f, W_f = best_result
    return to_output(U_f, V_f, W_f)

# EVOLVE-BLOCK-END
