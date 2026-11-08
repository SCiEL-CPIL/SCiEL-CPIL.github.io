import numpy as np
import time

_START_TIME = time.time()
DEADLINE_SECONDS = 130

def _elapsed():
    return time.time() - _START_TIME

def _time_remaining():
    return DEADLINE_SECONDS - _elapsed()

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x4 * 4x7 matrix multiplication.
    
    Strategy:
    1. Start with a structured decomposition using Strassen 2x2x2 blocks where possible.
    2. Use schoolbook multiplication for the remaining partial blocks.
    3. Attempt to reduce rank by identifying and merging linearly dependent rank-1 components.
    """
    n, m, p = 2, 4, 7
    nA = n * m   # 8
    nB = m * p   # 28
    nW = p * n   # 14
    
    # Tensor T has shape (n, m, p) with T[i,j,k] = 1
    # Decomposition: T[i,j,k] = sum_l U[i*m+j, l] * V[j*p+k, l] * W[k*n+i, l]
    
    def build_strassen_block_decomposition():
        """
        Build decomposition using Strassen 2x2x2 for 2x2 * 2x2 blocks.
        m=4, p=7. We can tile the 4x7 matrix into 2x2 blocks plus a 2x1 remainder.
        Specifically, split j into {0,1,2,3} and k into {0,1,2,3,4,5,6}.
        
        Blocks:
        - j in {0,1}, k in {0,1} -> Strassen
        - j in {0,1}, k in {2,3} -> Strassen
        - j in {0,1}, k in {4,5} -> Strassen
        - j in {0,1}, k in {6}   -> Schoolbook 2x1
        - j in {2,3}, k in {0,1} -> Strassen
        - j in {2,3}, k in {2,3} -> Strassen
        - j in {2,3}, k in {4,5} -> Strassen
        - j in {2,3}, k in {6}   -> Schoolbook 2x1
        """
        rank_max = 50  # Upper bound: 3*7 + 2*2 = 21+4 = 25? Let's be safe.
        # Actually, each Strassen block uses 7 terms, each schoolbook 2x1 uses 2 terms.
        # 3 Strassen blocks per j-pair * 2 j-pairs = 6 Strassen blocks = 42 terms
        # 2 schoolbook blocks = 4 terms
        # Total = 46. Let's use 50 for safety.
        
        U = [[0] * rank_max for _ in range(nA)]
        V = [[0] * rank_max for _ in range(nB)]
        W = [[0] * rank_max for _ in range(nW)]
        col = 0

        def add_strassen(i0, j0, k0):
            """
            Add Strassen terms for A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2] -> C[i0:i0+2, k0:k0+2]
            """
            nonlocal col
            # A indices
            a_idx = i0 * m + j0
            b_idx = i0 * m + (j0 + 1)
            c_idx = (i0 + 1) * m + j0
            d_idx = (i0 + 1) * m + (j0 + 1)
            # B indices
            e_idx = j0 * p + k0
            f_idx = j0 * p + (k0 + 1)
            g_idx = (j0 + 1) * p + k0
            h_idx = (j0 + 1) * p + (k0 + 1)
            # C indices
            c00 = k0 * n + i0
            c01 = (k0 + 1) * n + i0
            c10 = k0 * n + (i0 + 1)
            c11 = (k0 + 1) * n + (i0 + 1)

            # M1 = (a+d)(e+h) -> C[0,0] += M1, C[1,1] += M1
            if col >= rank_max: return
            U[a_idx][col] += 1; U[d_idx][col] += 1
            V[e_idx][col] += 1; V[h_idx][col] += 1
            W[c00][col] += 1; W[c11][col] += 1
            col += 1

            # M2 = (c+d)e -> C[1,0] += M2, C[1,1] -= M2
            if col >= rank_max: return
            U[c_idx][col] += 1; U[d_idx][col] += 1
            V[e_idx][col] += 1
            W[c10][col] += 1; W[c11][col] -= 1
            col += 1

            # M3 = a(f-h) -> C[0,1] += M3, C[1,1] += M3
            if col >= rank_max: return
            U[a_idx][col] += 1
            V[f_idx][col] += 1; V[h_idx][col] -= 1
            W[c01][col] += 1; W[c11][col] += 1
            col += 1

            # M4 = d(g-e) -> C[0,0] += M4, C[1,0] += M4
            if col >= rank_max: return
            U[d_idx][col] += 1
            V[g_idx][col] += 1; V[e_idx][col] -= 1
            W[c00][col] += 1; W[c10][col] += 1
            col += 1

            # M5 = (a+b)h -> C[0,0] -= M5, C[0,1] += M5
            if col >= rank_max: return
            U[a_idx][col] += 1; U[b_idx][col] += 1
            V[h_idx][col] += 1
            W[c00][col] -= 1; W[c01][col] += 1
            col += 1

            # M6 = (c-a)(e+f) -> C[1,1] += M6
            if col >= rank_max: return
            U[c_idx][col] += 1; U[a_idx][col] -= 1
            V[e_idx][col] += 1; V[f_idx][col] += 1
            W[c11][col] += 1
            col += 1

            # M7 = (b-d)(g+h) -> C[0,0] += M7
            if col >= rank_max: return
            U[b_idx][col] += 1; U[d_idx][col] -= 1
            V[g_idx][col] += 1; V[h_idx][col] += 1
            W[c00][col] += 1
            col += 1

        def add_schoolbook_2x1(i0, j0, k0):
            """
            Add schoolbook terms for A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0] -> C[i0:i0+2, k0]
            """
            nonlocal col
            a_idx = i0 * m + j0
            b_idx = i0 * m + (j0 + 1)
            c_idx = (i0 + 1) * m + j0
            d_idx = (i0 + 1) * m + (j0 + 1)
            e_idx = j0 * p + k0
            g_idx = (j0 + 1) * p + k0
            c00 = k0 * n + i0
            c10 = k0 * n + (i0 + 1)

            # C[0,0] += a*e + b*g
            if col >= rank_max: return
            U[a_idx][col] += 1; V[e_idx][col] += 1; W[c00][col] += 1; col += 1
            if col >= rank_max: return
            U[b_idx][col] += 1; V[g_idx][col] += 1; W[c00][col] += 1; col += 1
            
            # C[1,0] += c*e + d*g
            if col >= rank_max: return
            U[c_idx][col] += 1; V[e_idx][col] += 1; W[c10][col] += 1; col += 1
            if col >= rank_max: return
            U[d_idx][col] += 1; V[g_idx][col] += 1; W[c10][col] += 1; col += 1

        # Apply blocks
        # j pairs: (0,1) and (2,3)
        # k pairs: (0,1), (2,3), (4,5) and single k=6
        for j_start in [0, 2]:
            for k_start in [0, 2, 4]:
                add_strassen(0, j_start, k_start)
            add_schoolbook_2x1(0, j_start, 6)

        # Trim to actual used columns
        U = [row[:col] for row in U]
        V = [row[:col] for row in V]
        W = [row[:col] for row in W]
        
        return U, V, W, col

    def verify(U, V, W):
        """Verify exact decomposition."""
        r = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    ai = i * m + j
                    bi = j * p + k
                    wi = k * n + i
                    for l in range(r):
                        val += U[ai][l] * V[bi][l] * W[wi][l]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def trim_trailing_zeros(U, V, W):
        """Remove columns that are all zeros."""
        r = len(U[0])
        keep = []
        for l in range(r):
            if (all(U[i][l] == 0 for i in range(nA)) and
                all(V[i][l] == 0 for i in range(nB)) and
                all(W[i][l] == 0 for i in range(nW))):
                continue
            keep.append(l)
        if not keep:
            return U, V, W, 0
        U2 = [[U[i][l] for l in keep] for i in range(nA)]
        V2 = [[V[i][l] for l in keep] for i in range(nB)]
        W2 = [[W[i][l] for l in keep] for i in range(nW)]
        return U2, V2, W2, len(keep)

    # Step 1: Build initial decomposition
    U, V, W, rank = build_strassen_block_decomposition()
    
    # Step 2: Verify
    if not verify(U, V, W):
        # Fallback to pure schoolbook if Strassen construction has a bug
        rank = n * m * p
        U = [[0] * rank for _ in range(nA)]
        V = [[0] * rank for _ in range(nB)]
        W = [[0] * rank for _ in range(nW)]
        col = 0
        for j in range(m):
            for i in range(n):
                for k in range(p):
                    ai = i * m + j
                    bi = j * p + k
                    wi = k * n + i
                    U[ai][col] = 1
                    V[bi][col] = 1
                    W[wi][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Step 3: Trim zero columns
    U, V, W, rank = trim_trailing_zeros(U, V, W)
    best_rank = rank
    best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
    
    # Step 3.5: Try to find a better tiling that reduces rank
    # Alternative: use 2x2x2 Strassen for all blocks where possible, and optimize the remainder
    # For 2x4 * 4x7, we can try different block decompositions
    
    # Try: split j into {0,1,2,3} and k into {0,1,2,3,4,5,6}
    # Current: 6 Strassen blocks (2x2x2) + 2 schoolbook 2x1 blocks = 6*7 + 2*2 = 46
    # Alternative: Can we do better?
    # What if we use a different blocking strategy?
    
    # Actually, let's try to see if we can merge some of the Strassen terms across blocks
    # by finding linear dependencies in the full tensor space.

    # Step 4: Attempt rank reduction via linear dependency detection
    # This is a heuristic: if two rank-1 tensors are linearly dependent, we can merge them.
    # We check for columns that are scalar multiples of each other in the full tensor unfolding.
    
    if _time_remaining() < 30:
        return {"U": best_U, "best_V": best_V, "best_W": best_W}

    # Convert to numpy for linear algebra
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    r = U_np.shape[1]
    
    # Unfold each rank-1 component into a vector of length n*m*p
    # t_l[i*m+j, j*p+k, k*n+i] is not a simple vector index.
    # We need to map (i,j,k) to a flat index. Let's use index = i*(m*p) + j*p + k
    # But the tensor is T[i,j,k] = sum_l U[i*m+j, l] * V[j*p+k, l] * W[k*n+i, l]
    # So for a fixed l, the contribution is a 3D array T_l[i,j,k] = U[i*m+j, l] * V[j*p+k, l] * W[k*n+i, l]
    
    # We can form a matrix M where each column l is the flattened T_l.
    # Then if columns are linearly dependent, we can reduce rank.
    
    def flatten_tensor_l(l):
        vec = np.zeros(n * m * p)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i * (m * p) + j * p + k
                    vec[idx] = U_np[i * m + j, l] * V_np[j * p + k, l] * W_np[k * n + i, l]
        return vec

    # Build the matrix of flattened tensors
    # This might be slow, but n*m*p = 56 and r is small (~46)
    T_matrix = np.zeros((n * m * p, r))
    for l in range(r):
        T_matrix[:, l] = flatten_tensor_l(l)
    
    # Find linear dependencies
    # We want to find a subset of columns that forms a basis.
    # Use QR with column pivoting or SVD.
    
    try:
        # SVD to find rank
        U_svd, s, Vt = np.linalg.svd(T_matrix, full_matrices=False)
        # Rank is the number of significant singular values
        tol = 1e-10
        effective_rank = np.sum(s > tol)
        
        if effective_rank < r:
            # We can potentially reduce rank to effective_rank
            # However, the decomposition must be exact with integer/half-integer entries.
            # Merging via SVD might produce irrational numbers.
            # So we only do this if we can round to half-integers and verify.
            
            # Get the basis vectors
            basis = U_svd[:, :effective_rank] * s[:effective_rank]
            
            # This is complex to implement correctly with integer constraints.
            # For now, we just return the current best.
            # A more advanced approach would try to find integer linear combinations.
            pass
    except:
        pass

    return {"U": best_U, "V": best_V, "W": best_W}
