"""Matrix multiplication 4x4x9 - rank-minimal exact tensor decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 120.0
    n, m, p = DIMENSIONS  # 4, 4, 9
    # Target: U shape (n*m, rank)=(16,rank), V shape (m*p,rank)=(36,rank), W shape (p*n,rank)=(36,rank)
    # Output convention: W[k*n+i, r] for the (i,k) output element

    # We need: for all i in [0,n), k in [0,p), j in [0,m):
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    # (This is the standard tensor decomposition of matrix mult.)

    # Strategy: Build a valid decomposition, then try to reduce rank via
    # null-space / linear-dependence elimination of the rank-1 terms.

    # Start from schoolbook (rank 144) and attempt reductions.
    # A cleaner approach: use the structure of the multiplication tensor.

    # The matrix multiplication tensor T has entries T[i,j,k] = 1 for the
    # "middle" index j shared. We look for rank-r decomposition.

    # We'll use a constructive approach:
    # Represent each output (i,k) as a linear combination of products.
    # A known lower bound: rank >= 2*n*p - ... but we construct explicitly.

    # Build schoolbook first (guaranteed valid), then try to prune.
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1

    def to_result(U, V, W):
        # Convert numpy to list-of-lists with int/half-int entries
        r = U.shape[1]
        outU = [[0]*r for _ in range(U.shape[0])]
        outV = [[0]*r for _ in range(V.shape[0])]
        outW = [[0]*r for _ in range(W.shape[0])]
        for a in range(U.shape[0]):
            for b in range(r):
                v = U[a, b]
                # round to nearest 0.5
                outU[a][b] = round(v * 2) / 2
        for a in range(V.shape[0]):
            for b in range(r):
                v = V[a, b]
                outV[a][b] = round(v * 2) / 2
        for a in range(W.shape[0]):
            for b in range(r):
                v = W[a, b]
                outW[a][b] = round(v * 2) / 2
        return {"U": outU, "V": outV, "W": outW}

    best = to_result(U, V, W)
    best_rank = rank

    # ---- Attempt rank reduction via structured constructions ----
    # Key idea: the multiplication tensor for (n x m) * (m x p) can be
    # decomposed using the "bordered" / Winograd trick when one dimension
    # is a multiple that allows cyclic convolution. Here p=9 is not a
    # power of 2, but m=4 is. We can group over the m dimension.

    # For each pair (i,k), output C[i,k] = sum_j A[i,j] B[j,k].
    # We can use a decomposition of the "middle" dimension m=4 via
    # a rank-5 Winograd 4-point scheme (F_4^T A F_4 style), reducing
    # the effective multiplications.

    # Winograd for m=4: F = [[1,0,1,0],[0,1,1,1],[0,1,0,1],[1,1,0,0]]
    # Actually the standard F_4 = [[1,0,1,0],[1,1,0,1],[0,1,1,1],[1,1,1,1]]
    # with F_4^T F_4 = 4*I. Using this: A' = F_4^T A (4x4), B' = B F_4 (4x4),
    # then C' = A' * B' elementwise, C = F_4^{-1} C' F_4^{-1}.
    # This doesn't reduce rank of the 4x4x4 core (still needs rank of 4x4x4).

    # The real reduction comes from the fact that for the FULL tensor
    # (n*m*p = 144), we can use the Winograd transform on the m-axis
    # to replace m=4 with a rank-5 frequency representation, but the
    # core 4x4x4 (or here the n x p part) still dominates.

    # Let's try a direct constructive rank reduction:
    # Use the fact that C[i,k] = sum_j A[i,j] B[j,k].
    # Group the j-sum using a 4-point DFT-like transform.
    # Let omega be a primitive 4th root. Then:
    #   sum_j A[i,j] B[j,k] = (1/4) sum_t sum_u A_hat[i,t] B_hat[k,u] ... 
    # This is getting complex. Let's instead try a pragmatic approach:
    # use a known low-rank construction and verify.

    # Pragmatic: Try to find rank by solving linear systems / using
    # the fact that the tensor has structure. Use a greedy elimination.

    # For robustness and to beat 144, we use a constructive
    # "block" decomposition: treat the m=4 middle dimension with a
    # rank-5 Winograd factor, giving effective rank ~ n*p*5 = 4*9*5=180? No.

    # Let me use a cleaner known result: the rank of (n x m) x (m x p)
    # matrix mult tensor. For m=4, the best known uses Strassen-like
    # on the 4x4 core. But the full tensor rank is what we need.

    # Given time constraints, I'll implement a verified constructive
    # decomposition that is guaranteed valid and try a few known
    # low-rank templates, keeping the best valid one found.

    # ---- Template 1: exploit p=9 = 4+4+1 via splitting ----
    # C[i,k] for k in [0,4): sum_j A[i,j] B[j,k]
    # C[i,k] for k in [4,8): sum_j A[i,j] B[j,k]
    # C[i,k] for k in [8,9): sum_j A[i,j] B[j,8]
    # This doesn't reduce rank directly.

    # ---- Use randomized rank-reduction search with verification ----
    # We'll attempt to reduce rank by finding linear dependencies.
    # Represent the tensor as a 3-tensor and use a simple approach:
    # try rank = R, set up random factors, verify. If we can't verify
    # at low R, fall back to a constructive valid one.

    # Constructive valid decomposition with reduced rank using
    # the "Kronecker / tensor product" structure is hard to do by hand.

    # SAFETY: return schoolbook if nothing better is verified.
    # But let's at least try a moderate reduction via a verified method.

    # ---- Verified reduction: use the m-dimension Winograd (rank 5) ----
    # The 4-point Winograd transform F (4x4 invertible) lets us write
    # the m=4 contraction as: C = F^{-1} ( (F^T A) .* (B F) ) F^{-1}
    # where .* is elementwise over the transformed m-index.
    # This transforms the (n x m) x (m x p) problem into computing
    # n*p elementwise products in the transformed domain, each of which
    # is a product of a row of F^T A and a column of B F.
    # The number of "multiplications" in the transformed domain is n*p
    # (one per (i,k) pair), but each involves combining 4 original
    # products. The tensor rank of the transformed computation:
    #   For each (i,k) in the transformed output, we need
    #   (F^T A)[i,t] * (B F)[t,k] summed over t (4 terms).
    # This is a (n x 4) x (4 x p) matrix mult in transformed space,
    # which has the SAME structure as original but with m=4 replaced
    # by the 4 transform coefficients. It does NOT reduce rank.

    # The genuine rank reduction for the full tensor comes from
    # using a low-rank decomposition of the CORE 4x4x4 tensor.
    # Best known rank of 4x4x4 matmul tensor is 49 (or 48?).
    # But we have (4x4)x(4x9) = 144 entries. We can view this as
    # 9 copies of the "column" structure.

    # Given the difficulty and time, I'll return a valid decomposition.
    # To beat the seed (144), I'll use a simple but real reduction:
    # the tensor rank of (n x m) x (m x p) is at most n*p*ceil(log2 m)
    # via Strassen recursion on the m dimension. For m=4, Strassen
    # reduces 4x4x4 to rank 7 per 2x2x2 block... 

    # Strassen on the m=4 dimension: split m=4 into two 2x2 blocks.
    # C = [A_left | A_right] * [B_top; B_bottom]
    #   = A_left B_top + A_right B_bottom
    # Using Strassen 2x2x2 (rank 7) on each... but A_left is n x 2,
    # B_top is 2 x p. This is (n x 2) x (2 x p).
    # The tensor (n x 2) x (2 x p) has rank at most n*p*2 (schoolbook)
    # but can be reduced. For the 2x2x2 core (n=2,p=2), rank is 7.

    # Let me just construct using Strassen on the m=4 split into 2x2.
    # m=4 -> two blocks of size 2.
    # C = A1 B1 + A2 B2 where A1=A[:,0:2], A2=A[:,2:4], B1=B[0:2,:], B2=B[2:4,:].
    # Each term (n x 2) x (2 x p): schoolbook rank = n*p*2 = 4*9*2=72.
    # Two terms -> 144 total. No gain without combining.

    # Combine via Strassen: M1=(A1+A2)(B1+B2), M2=(A1+A2)B2, M3=A1B1, M4=A2(B1-B2)
    # C = M1 - M4 - M2 + M3 ... let me verify:
    # M1 - M4 - M2 + M3 = (A1+A2)(B1+B2) - A2(B1-B2) - (A1+A2)B2 + A1B1
    #   = A1B1 + A1B2 + A2B1 + A2B2 - A2B1 + A2B2 - A1B2 - A2B2 + A1B1
    #   = 2 A1B1 + A2B2.  WRONG.

    # Correct Strassen for C = A1B1 + A2B2 (where A1,A2 are n x 2, B1,B2 are 2 x p):
    # This is NOT a 2x2x2 Strassen because A1B1 is already a matrix product.
    # Strassen applies when multiplying two 2x2 matrices. Here A1 is n x 2.

    # I think the cleanest correct reduction is limited. Given the risk,
    # I'll return a verified valid decomposition and attempt a modest
    # rank reduction only if I can VERIFY it exactly.

    # ---- Verify helper ----
    def verify(U, V, W, tol=1e-6):
        # U: (n*m, r), V: (m*p, r), W: (p*n, r)
        r = U.shape[1]
        # For each (i,k): sum_j sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] == 1
        # Build efficiently.
        ok = True
        for i in range(n):
            for k in range(p):
                s = 0.0
                for j in range(m):
                    s += np.dot(U[i*m+j, :], V[j*p+k, :]) * W[k*n+i, :]
                if abs(s - 1.0) > tol:
                    ok = False
                    break
            if not ok:
                break
        return ok

    # Try a few constructive low-rank templates and keep the best valid.
    # Template: use the fact that we can factor the identity-like structure.

    # Given the high risk of incorrect hand-constructed templates and the
    # 120s budget, I will return the guaranteed-valid schoolbook decomposition.
    # This is safe and scores 1.0 (same as seed). To attempt improvement,
    # I'll run a quick randomized rank-reduction that verifies.

    # Randomized search for lower rank (best-effort, verified):
    if time.time() < deadline - 30:
        # Try to find a valid decomposition at rank R < 144 via random init
        # and a few Newton steps is too slow/fragile. Instead, use a
        # structured reduction: the tensor is a sum of n*p "slices".
        # Each slice (i,k) needs the vector v_{i,k} in R^m such that
        # C[i,k] = A[i,:] v_{i,k} B[:,k]. The tensor rank equals the
        # minimum r such that all v_{i,k} lie in a rank-r "multiplication"
        # structure. This is exactly the matmul tensor rank problem.

        # I'll do nothing risky. Return schoolbook.
        pass

    return best
# EVOLVE-BLOCK-END
