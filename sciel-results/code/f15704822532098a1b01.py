"""Matrix multiplication 2x3x8: rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110.0  # safety margin before 120s limit
    n, m, p = DIMENSIONS

    # Build the matrix multiplication tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    # We want rank-r decomposition: T = sum_c U[:,c] V[:,c]^T ... actually
    # T[i,j,k] = sum_c U[i,c]*V[j,c]*W[k,c]
    # where U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)

    # The tensor is T_{i,j,k} = 1 if j = j (shared index), i.e.,
    # T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k
    # This is the standard matrix multiplication tensor.

    # Strategy: Use a systematic search for low-rank decompositions.
    # Start from the schoolbook rank n*m*p = 48 and try to reduce.
    # Known results: For 2x3x8, we can potentially do better than 48.
    # The cell finding mentions rank 44 (four 2x3x2 sub-tensors, each rank 11).

    # Let's build the tensor and try to find a rank decomposition.
    # T is a 3-tensor of size (n*m) x (m*p) x (p*n) = 6 x 24 x 16

    # Flatten to a 2D matrix for each mode to use tensor rank algorithms.
    # Actually, let's use a direct approach: enumerate possible rank-r decompositions.

    # For small tensors, we can use the "border rank" or "rank" computation.
    # But here we need an EXACT decomposition with integer/half-integer entries.

    # Approach: Use the known structure. The 2x3x8 multiplication can be
    # decomposed by splitting the 8 columns into four groups of 2.
    # Each 2x3x2 sub-multiplication can be done in rank 11 (as noted).
    # Total rank = 4 * 11 = 44.

    # Let's construct the rank-11 decomposition for 2x3x2.
    # For 2x3x2: A is 2x3, B is 3x2, C is 2x2.
    # Schoolbook rank = 2*3*2 = 12.
    # Can we do rank 11? Let's try.

    # Actually, let me think about this differently.
    # The tensor for 2x3x2 has 12 nonzero entries.
    # T[i*3+j, j*2+k, k*2+i] = 1 for i in {0,1}, j in {0,1,2}, k in {0,1}

    # Let's try to find a rank-11 decomposition for 2x3x2.
    # If we can, then for 2x3x8 we get rank 44.

    # For 2x3x2, the tensor is:
    # T[0,0,0]=1, T[0,1,1]=1, T[0,2,2]=1  (i=0, j=0,1,2, k=0,1)
    # Wait, let me re-index.
    # i in {0,1} (rows of A), j in {0,1,2} (cols of A / rows of B), k in {0,1} (cols of B)
    # U index: i*m+j = i*3+j, range 0..5
    # V index: j*p+k = j*2+k, range 0..5
    # W index: k*n+i = k*2+i, range 0..3

    # T[i*3+j, j*2+k, k*2+i] = 1
    # Nonzero entries:
    # (0,0,0), (0,1,1), (0,2,2)  -- i=0,j=0,k=0; i=0,j=1,k=1; i=0,j=2,k=2? No, k only goes to 1.
    # Let me redo: k in {0,1}
    # i=0,j=0,k=0: (0,0,0)
    # i=0,j=0,k=1: (0,1,1)
    # i=0,j=1,k=0: (1,0,2)
    # i=0,j=1,k=1: (1,1,3)
    # i=0,j=2,k=0: (2,0,0)
    # i=0,j=2,k=1: (2,1,1)
    # i=1,j=0,k=0: (3,0,2)
    # i=1,j=0,k=1: (3,1,3)
    # i=1,j=1,k=0: (4,0,0)
    # i=1,j=1,k=1: (4,1,1)
    # i=1,j=2,k=0: (5,0,2)
    # i=1,j=2,k=1: (5,1,3)

    # So 12 nonzero entries in a 6x6x4 tensor.

    # For rank 11, we need to find U (6x11), V (6x11), W (4x11) such that
    # T = sum_c U[:,c] outer V[:,c] outer W[:,c]

    # This is a hard problem. Let me try a different approach: use random initialization
    # and gradient descent to minimize the Frobenius norm of (T - sum_c ...)^2.

    # But we need EXACT integer/half-integer solutions. Let me try to construct them.

    # Alternative: use the fact that for 2x3x2, we can use Strassen-like ideas.
    # Actually, for 2x2x2, Strassen gives rank 7 < 8.
    # For 2x3x2, maybe we can get rank 11 < 12.

    # Let me try a computational approach: set up the equations and solve.
    # For rank r, we have r*6 + r*6 + r*4 = 16r unknowns.
    # We have 12 equations (one per nonzero entry) + the requirement that
    # all other entries are zero. Total: 6*6*4 = 144 entries, 12 are 1, 132 are 0.
    # So 144 equations, 16r unknowns. For r=11: 176 unknowns > 144 equations.
    # This is underdetermined, so solutions may exist.

    # Let me try to solve this numerically and then round to half-integers.

    def find_decomposition_2x3x2(max_rank=12, time_limit=10.0):
        """Find a rank-r decomposition of the 2x3x2 matrix multiplication tensor."""
        n2, m2, p2 = 2, 3, 2
        dim_u = n2 * m2  # 6
        dim_v = m2 * p2  # 6
        dim_w = p2 * n2  # 4

        # Build the target tensor
        T = np.zeros((dim_u, dim_v, dim_w))
        for i in range(n2):
            for j in range(m2):
                for k in range(p2):
                    T[i*m2+j, j*p2+k, k*n2+i] = 1.0

        # Try ranks from low to high
        for r in range(8, max_rank+1):
            if time.time() > time_limit:
                break
            # Random initialization
            np.random.seed(42 + r)
            U = np.random.randn(dim_u, r) * 0.5
            V = np.random.randn(dim_v, r) * 0.5
            W = np.random.randn(dim_w, r) * 0.5

            # Gradient descent to minimize ||T - sum_c U_c V_c W_c||_F^2
            lr = 0.01
            for it in range(5000):
                if time.time() > time_limit:
                    break
                # Compute current decomposition
                recon = np.zeros((dim_u, dim_v, dim_w))
                for c in range(r):
                    recon += np.einsum('i,j,k->ijk', U[:,c], V[:,c], W[:,c])
                err = T - recon
                # Gradients
                # dL/dU[i,c] = sum_{j,k} err[i,j,k] * V[j,c] * W[k,c]
                dU = np.einsum('ijk,jc,kc->ic', err, V, W)
                dV = np.einsum('ijk,ic,kc->jc', err, U, W)
                dW = np.einsum('ijk,ic,jc->kc', err, U, V)
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW

                if it % 500 == 0:
                    frob = np.linalg.norm(err)
                    if frob < 1e-8:
                        break

            recon = np.zeros((dim_u, dim_v, dim_w))
            for c in range(r):
                recon += np.einsum('i,j,k->ijk', U[:,c], V[:,c], W[:,c])
            err = T - recon
            frob = np.linalg.norm(err)

            if frob < 1e-6:
                # Try to round to half-integers
                U_round = np.round(2*U) / 2.0
                V_round = np.round(2*V) / 2.0
                W_round = np.round(2*W) / 2.0

                recon_r = np.zeros((dim_u, dim_v, dim_w))
                for c in range(r):
                    recon_r += np.einsum('i,j,k->ijk', U_round[:,c], V_round[:,c], W_round[:,c])
                err_r = T - recon_r
                if np.linalg.norm(err_r) < 1e-9:
                    return r, U_round, V_round, W_round

                # If rounding doesn't work, try scaling
                # Maybe the solution uses integers
                U_round = np.round(U)
                V_round = np.round(V)
                W_round = np.round(W)
                recon_r = np.zeros((dim_u, dim_v, dim_w))
                for c in range(r):
                    recon_r += np.einsum('i,j,k->ijk', U_round[:,c], V_round[:,c], W_round[:,c])
                err_r = T - recon_r
                if np.linalg.norm(err_r) < 1e-9:
                    return r, U_round, V_round, W_round

        return None

    # Try to find a good decomposition for 2x3x2
    result_232 = find_decomposition_2x3x2(max_rank=12, time_limit=30.0)

    if result_232 is not None:
        r_sub, U_sub, V_sub, W_sub = result_232
        # Build the full 2x3x8 decomposition by tiling
        # The 8 columns are split into 4 groups of 2
        # For group g (g=0,1,2,3), the sub-problem is 2x3x2
        # U index: i*3+j (same for all groups)
        # V index: j*8+k where k = g*2 + k_sub
        # W index: (g*2+k_sub)*2+i = k*2+i

        rank_total = 4 * r_sub
        U_full = np.zeros((6, rank_total))
        V_full = np.zeros((24, rank_total))
        W_full = np.zeros((16, rank_total))

        for g in range(4):
            col_offset = g * r_sub
            for c in range(r_sub):
                # U is the same for all groups
                U_full[:, col_offset + c] = U_sub[:, c]
                # V: j*8 + (g*2 + k_sub)
                for j in range(3):
                    for k_sub in range(2):
                        V_full[j*8 + g*2 + k_sub, col_offset + c] = V_sub[j*2 + k_sub, c]
                # W: (g*2 + k_sub)*2 + i
                for k_sub in range(2):
                    for i in range(2):
                        W_full[(g*2 + k_sub)*2 + i, col_offset + c] = W_sub[k_sub*2 + i, c]

        # Verify
        T_full = np.zeros((6, 24, 16))
        for i in range(2):
            for j in range(3):
                for k in range(8):
                    T_full[i*3+j, j*8+k, k*2+i] = 1.0

        recon_full = np.zeros((6, 24, 16))
        for c in range(rank_total):
            recon_full += np.einsum('i,j,k->ijk', U_full[:,c], V_full[:,c], W_full[:,c])
        err_full = T_full - recon_full
        if np.linalg.norm(err_full) < 1e-9:
            return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}

    # Fallback: schoolbook decomposition
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
