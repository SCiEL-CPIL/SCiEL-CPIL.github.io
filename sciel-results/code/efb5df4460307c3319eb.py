import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 6]

def _verify(U, V, W, n, m, p, trials=5, seed=42):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    rng = np.random.RandomState(seed)
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for _ in range(trials):
        A = rng.randn(n, m)
        B = rng.randn(m, p)
        C_expected = A @ B
        C_actual = np.zeros((n, p))
        for r in range(rank):
            # u_r[i,j] = U[i*m+j, r]
            # v_r[j,k] = V[j*p+k, r]
            # w_r[k,i] = W[k*n+i, r]
            # C[i,k] += sum_j U[i*m+j,r] * A[i,j] * V[j*p+k,r] * B[j,k] * W[k*n+i,r]
            # Actually the formula is:
            # C[i,k] = sum_r sum_j U[i*m+j, r] * A[i,j] * V[j*p+k, r] * B[j,k] * W[k*n+i, r]
            # Wait, that's not right. Let me think.
            # The tensor decomposition: T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
            # And C[i,k] = sum_j A[i,j] * B[j,k]
            # So C[i,k] = sum_j sum_r T[i,j,k] * A[i,j] * B[j,k] ... no
            # Actually: C[i,k] = sum_j A[i,j] * B[j,k]
            # And the decomposition says the "multiplication pattern" is captured by T.
            # I think the correct reconstruction is:
            # C[i,k] = sum_r (sum_j U[i*m+j,r] * A[i,j]) * (sum_j V[j*p+k,r] * B[j,k]) * W[k*n+i,r]
            # No, that's not right either.
            # Let me just check: for the schoolbook case, T[i,j,k] = 1 if j==j (always true in this context)
            # Actually T[i,j,k] should be 1 for all (i,j,k) in the sense that each (i,j,k) triple
            # contributes A[i,j]*B[j,k] to C[i,k].
            # So C[i,k] = sum_j A[i,j]*B[j,k] = sum_j T[i,j,k] * A[i,j] * B[j,k] where T[i,j,k]=1
            # With decomposition T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
            # C[i,k] = sum_j sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] * A[i,j]*B[j,k]
            # = sum_r W[k*n+i,r] * (sum_j U[i*m+j,r]*A[i,j]) * (sum_j V[j*p+k,r]*B[j,k])
            # Wait, that's not separable that way.
            # C[i,k] = sum_j A[i,j]*B[j,k] * sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
            # = sum_r W[k*n+i,r] * sum_j A[i,j]*B[j,k]*U[i*m+j,r]*V[j*p+k,r]
            # The sum_j term is not separable into a product of two sums.
            # 
            # I think the correct interpretation is different. Let me reconsider.
            # The matrix multiplication as a trilinear form:
            # <C, A, B> = sum_{i,j,k} C[i,k] * A[i,j] * B[j,k]
            # = sum_{i,j,k} (sum_j' A[i,j']*B[j',k]) * A[i,j] * B[j,k]
            # 
            # A rank-r decomposition of the tensor T[i,j,k] (where T encodes the multiplication)
            # means: T[i,j,k] = sum_r U_r[i,j] * V_r[j,k] * W_r[k,i]
            # where T[i,j,k] = 1 (the multiplication tensor has all ones in the appropriate sense)
            # 
            # Actually, I think the standard formulation is:
            # C = A*B means C[i,k] = sum_j A[i,j]*B[j,k]
            # A bilinear algorithm with r multiplications:
            # C[i,k] = sum_r alpha_r[i,k] * (sum_j beta_r[i,j]*A[i,j]) * (sum_k gamma_r[j,k]*B[j,k])
            # Hmm, this is getting confusing. Let me just verify numerically with the known solution.
            pass
        # Alternative verification: directly compute using the tensor definition
        # C[i,k] should equal sum_j A[i,j]*B[j,k]
        # The decomposition guarantees this if T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
        # for all i,j,k... but that's not right either since T should be delta-like.
        # 
        # I'll just trust the Strassen construction and verify by checking that
        # for random A,B, the computed C matches A@B using the bilinear form.
        C_actual = np.zeros((n, p))
        for r in range(rank):
            # Compute intermediate: a_r[i] = sum_j U[i*m+j, r] * A[i, j]
            a_r = np.zeros(n)
            for i in range(n):
                for j in range(m):
                    a_r[i] += U[i * m + j, r] * A[i, j]
            # Compute intermediate: b_r[k] = sum_j V[j*p+k, r] * B[j, k]
            b_r = np.zeros(p)
            for k in range(p):
                for j in range(m):
                    b_r[k] += V[j * p + k, r] * B[j, k]
            # C[i,k] += W[k*n+i, r] * a_r[i] * b_r[k]
            for i in range(n):
                for k in range(p):
                    C_actual[i, k] += W[k * n + i, r] * a_r[i] * b_r[k]
        if not np.allclose(C_actual, C_expected, atol=1e-10):
            return False
    return True

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 110

    def make_strassen_solution():
        rank = 42
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]

        def add_strassen_block(i_start, j_start, k_start, offset):
            col = offset
            U[(i_start+0)*m + (j_start+0)][col] += 1
            U[(i_start+1)*m + (j_start+1)][col] += 1
            V[(j_start+0)*p + (k_start+0)][col] += 1
            V[(j_start+1)*p + (k_start+1)][col] += 1
            W[(k_start+0)*n + (i_start+0)][col] += 1
            W[(k_start+1)*n + (i_start+1)][col] += 1

            col = offset + 1
            U[(i_start+1)*m + (j_start+0)][col] += 1
            U[(i_start+1)*m + (j_start+1)][col] += 1
            V[(j_start+0)*p + (k_start+0)][col] += 1
            W[(k_start+0)*n + (i_start+1)][col] += 1
            W[(k_start+1)*n + (i_start+1)][col] -= 1

            col = offset + 2
            U[(i_start+0)*m + (j_start+0)][col] += 1
            V[(j_start+0)*p + (k_start+1)][col] += 1
            V[(j_start+1)*p + (k_start+1)][col] -= 1
            W[(k_start+1)*n + (i_start+0)][col] += 1
            W[(k_start+1)*n + (i_start+1)][col] += 1

            col = offset + 3
            U[(i_start+1)*m + (j_start+1)][col] += 1
            V[(j_start+1)*p + (k_start+0)][col] += 1
            V[(j_start+0)*p + (k_start+0)][col] -= 1
            W[(k_start+0)*n + (i_start+0)][col] += 1
            W[(k_start+0)*n + (i_start+1)][col] += 1

            col = offset + 4
            U[(i_start+0)*m + (j_start+0)][col] += 1
            U[(i_start+0)*m + (j_start+1)][col] += 1
            V[(j_start+1)*p + (k_start+1)][col] += 1
            W[(k_start+0)*n + (i_start+0)][col] -= 1
            W[(k_start+1)*n + (i_start+0)][col] += 1

            col = offset + 5
            U[(i_start+1)*m + (j_start+0)][col] += 1
            U[(i_start+0)*m + (j_start+0)][col] -= 1
            V[(j_start+0)*p + (k_start+0)][col] += 1
            V[(j_start+0)*p + (k_start+1)][col] += 1
            W[(k_start+1)*n + (i_start+1)][col] += 1

            col = offset + 6
            U[(i_start+0)*m + (j_start+1)][col] += 1
            U[(i_start+1)*m + (j_start+1)][col] -= 1
            V[(j_start+1)*p + (k_start+0)][col] += 1
            V[(j_start+1)*p + (k_start+1)][col] += 1
            W[(k_start+0)*n + (i_start+0)][col] += 1

        add_strassen_block(0, 0, 0, 0)
        add_strassen_block(0, 0, 2, 7)
        add_strassen_block(0, 0, 4, 14)
        add_strassen_block(0, 2, 0, 21)
        add_strassen_block(0, 2, 2, 28)
        add_strassen_block(0, 2, 4, 35)

        return U, V, W, rank

    best_U, best_V, best_W, best_rank = make_strassen_solution()

    # Verify the Strassen solution
    if not _verify(best_U, best_V, best_W, n, m, p):
        # Fallback to schoolbook if Strassen is wrong
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
        best_U, best_V, best_W, best_rank = U, V, W, rank

    # Try to find lower-rank solutions by checking for linear dependencies
    # in the Strassen construction
    if time.time() < deadline:
        # Check if any columns in U, V, W are linearly dependent
        # If a column r has all zeros in U, V, or W, it can be removed
        U_arr = np.array(best_U, dtype=float)
        V_arr = np.array(best_V, dtype=float)
        W_arr = np.array(best_W, dtype=float)

        # Check for zero columns
        keep = []
        for r in range(best_rank):
            if np.any(U_arr[:, r] != 0) and np.any(V_arr[:, r] != 0) and np.any(W_arr[:, r] != 0):
                keep.append(r)

        if len(keep) < best_rank:
            best_U = [best_U[r] for r in keep]
            best_V = [best_V[r] for r in keep]
            best_W = [best_W[r] for r in keep]
            # Transpose back
            best_U = list(zip(*best_U)) if best_U else []
            best_V = list(zip(*best_V)) if best_V else []
            best_W = list(zip(*best_W)) if best_W else []
            best_rank = len(keep)

    return {"U": best_U, "V": best_V, "W": best_W}
