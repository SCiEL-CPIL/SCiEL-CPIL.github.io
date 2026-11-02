import numpy as np
import time

DIMENSIONS = [2, 2, 16]


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 125.0

    def build_strassen_56():
        """Build the known rank-56 block Strassen decomposition."""
        rank = 56
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))

        u_vecs = [
            [1, 0, 0, 1],
            [0, 0, 1, 1],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [-1, 0, 1, 0],
            [0, 1, 0, -1]
        ]
        v_vecs = [
            [1, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, -1],
            [-1, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1]
        ]
        w_vecs = [
            [1, 0, 0, 1],
            [0, 0, 1, -1],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [-1, 1, 0, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0]
        ]

        for r_blk in range(8):
            base_col = 7 * r_blk
            for s in range(7):
                col = base_col + s
                for li in range(4):
                    if u_vecs[s][li] != 0:
                        U[li, col] = u_vecs[s][li]
                v_idx = [2 * r_blk, 2 * r_blk + 1, 16 + 2 * r_blk, 16 + 2 * r_blk + 1]
                for li in range(4):
                    if v_vecs[s][li] != 0:
                        V[v_idx[li], col] = v_vecs[s][li]
                w_idx = [4 * r_blk, 4 * r_blk + 2, 4 * r_blk + 1, 4 * r_blk + 3]
                for li in range(4):
                    if w_vecs[s][li] != 0:
                        W[w_idx[li], col] = w_vecs[s][li]
        return U, V, W, rank

    def verify(U, V, W):
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        W = np.asarray(W, dtype=float)
        T_comp = np.einsum('ir,jr,kr->ijk', U, V, W)
        T_true = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_true[i * m + j, j * p + k, i * p + k] = 1.0
        return bool(np.allclose(T_comp, T_true, atol=1e-9))

    # Build base solution
    U_base, V_base, W_base, rank_base = build_strassen_56()

    if not verify(U_base, V_base, W_base):
        # Fallback: schoolbook rank 64
        rank_fb = 64
        U_fb = np.zeros((n * m, rank_fb))
        V_fb = np.zeros((m * p, rank_fb))
        W_fb = np.zeros((p * n, rank_fb))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U_fb[i * m + j, r] = 1.0
                    V_fb[j * p + k, r] = 1.0
                    W_fb[i * p + k, r] = 1.0
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    best_U, best_V, best_W = U_base, V_base, W_base
    best_rank = rank_base

    # Attempt to find lower-rank structured decomposition
    # The tensor is: C[i,k] = sum_j A[i,j] * B[j,k] for i in {0,1}, j in {0,1}, k in {0,...,15}
    # Key: A is the same 2x2 matrix applied to all 8 column pairs.
    
    # Try a unified decomposition: treat the problem as multiplying a 2x2 matrix
    # by a 2x16 matrix. Think of this as:
    # [A[0,0] A[0,1]]   [B[0,0] B[0,1] ... B[0,15]]   [C[0,0] C[0,1] ... C[0,15]]
    # [A[1,0] A[1,1]] * [B[1,0] B[1,1] ... B[1,15]] = [C[1,0] C[1,1] ... C[1,15]]
    
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
    
    # This is a system of 32 output elements, each a sum of 2 products.
    # Naive: 64 terms. Can we do better with shared structure?
    
    # The tensor has rank at most 4*32 = 128 (trivial), but structure reduces it.
    # Lower bound from tensor rank: The tensor is supported on a variety of rank-3 tensors.
    
    # Try: Use the fact that all 8 column-pairs use the same A multiplication.
    # For each column-pair r (columns 2r, 2r+1 of B), the sub-problem is:
    # [B0[2r] B0[2r+1]]     [A00 A01]   [B0[2r] B0[2r+1]]
    # [B1[2r] B1[2r+1]]  x  [A10 A11]   [B1[2r] B1[2r+1]]
    # but wait, C is 2x16 not 2x2 per block.
    
    # Actually C[0,2r] = A00*B0[2r] + A01*B1[2r]
    # C[0,2r+1] = A00*B0[2r+1] + A01*B1[2r+1]
    # C[1,2r] = A10*B0[2r] + A11*B1[2r]
    # C[1,2r+1] = A10*B0[2r+1] + A11*B1[2r+1]
    
    # Each of the 32 C elements is a linear combination of 2 A*B products.
    # Total: 32 outputs, each needs 2 multiplications = 64 total.
    
    # To reduce below 64, we need to share multiplication terms across different C outputs.
    # The only way: find combinations of A elements that can be multiplied with 
    # combinations of B elements once, then reused.
    
    # For 2x2 matrix multiplication, Strassen reduces 8 scalar mults to 7.
    # Here, the 32 C elements decompose into 8 groups of 4, each group being a 
    # 2x2 "effective" multiplication.
    
    # The block Strassen approach already exploits this: 8 * 7 = 56.
    # Can we do better by exploiting that all 8 blocks share the same A?
    
    # Idea: Precompute 7 Strassen-style combinations of A once, then for each block
    # only need linear combinations of B with those precomputed A values.
    # But Strassen requires 7 products, each of which is (linear combo of A) * (linear combo of B).
    # The A part is the same across all 8 blocks!
    
    # So: Let M1 = (A00+A11) * (B0+..), M2 = (A10+A11)*(B0+..), etc.
    # The A-linear-combos are shared: only 7 unique A combinations needed.
    # But each M_s = (A_combo_s) * (B_combo_s), and B_combo differs per block.
    # So we still need 7 products per block = 56 total.
    
    # Unless we can batch: M_s for block r = A_combo_s * B_combo_s(r)
    # This is 8 separate scalar products for each s, total 56. No savings.
    
    # The structure doesn't allow cross-block sharing of the multiplication itself.
    # The rank-56 appears to be the natural structured bound.
    
    # Attempt: Use randomized rounding / structured perturbation to find if a 
    # slightly different pattern gives rank < 56.
    # Given time constraints, verify and return best found.
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
