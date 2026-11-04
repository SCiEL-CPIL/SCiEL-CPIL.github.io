import time
import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Leave margin before 150s limit

    # First, try to find a lower-rank decomposition
    best_rank = 42
    best_U, best_V, best_W = None, None, None

    # Try ranks from 41 down to 36 (lower bound is ~24)
    for rank in range(41, 35, -1):
        if time.time() > deadline - 30:
            break
        # Try a few random initializations
        max_attempts = 50
        for attempt in range(max_attempts):
            if time.time() > deadline - 30:
                break
            # Initialize with random half-integers in [-2, 2]
            U = np.random.randint(-4, 5, size=(n*m, rank)) / 2.0
            V = np.random.randint(-4, 5, size=(m*p, rank)) / 2.0
            W = np.random.randint(-4, 5, size=(p*n, rank)) / 2.0

            # Check if this is a valid decomposition
            # The tensor T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
            # should equal 1 if j is in the "middle" index range for the
            # matrix multiplication tensor, else 0
            # Actually for matrix multiplication 2x4 * 4x6:
            # T[i,j,k] = 1 if the "path" i->j->k is valid
            # i in 0..1, j in 0..3, k in 0..5
            # The tensor has T[i,j,k] = 1 always (it's the structure tensor)
            # Wait, let me reconsider. The matrix multiplication tensor for
            # (n x m) * (m x p) has entries T[i, j, k] where i in [0,n), j in [0,m), k in [0,p)
            # and T[i,j,k] = 1. The decomposition is:
            # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            
            # Let's verify by computing the tensor and checking
            U_mat = U
            V_mat = V
            W_mat = W
            
            # Compute sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # This is a 3D tensor contraction
            # For efficiency, compute for each r: outer product of U[:,r], V[:,r], W[:,r]
            # and sum over r
            
            # Actually, let's just check a few random entries
            valid = True
            for _ in range(100):
                i = np.random.randint(n)
                j = np.random.randint(m)
                k = np.random.randint(p)
                # Compute T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                val = np.sum(U_mat[i*m+j, :] * V_mat[j*p+k, :] * W_mat[k*n+i, :])
                if abs(val - 1) > 1e-6:
                    valid = False
                    break
            
            if valid:
                best_rank = rank
                best_U = U
                best_V = V
                best_W = W
                break
        if best_rank < 42:
            break

    if best_rank < 42 and best_U is not None:
        # Convert to list of lists with half-integer entries
        U_list = [[int(round(x * 2)) / 2 for x in row] for row in best_U]
        V_list = [[int(round(x * 2)) / 2 for x in row] for row in best_V]
        W_list = [[int(round(x * 2)) / 2 for x in row] for row in best_W]
        return {"U": U_list, "V": V_list, "W": W_list}

    # Fallback: Strassen-based rank 42 decomposition
    rank = 42
    U = [[0] * rank for _ in range(n * m)]  # 8 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 12 x 42

    def add_strassen_block(i_start, j_start, k_start, offset):
        # Adds a 2x2x2 Strassen block
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

    # Block 1: A[0:2, 0:2] * B[0:2, 0:2]
    add_strassen_block(0, 0, 0, 0)
    # Block 2: A[0:2, 0:2] * B[0:2, 2:4]
    add_strassen_block(0, 0, 2, 7)
    # Block 3: A[0:2, 0:2] * B[0:2, 4:6]
    add_strassen_block(0, 0, 4, 14)
    # Block 4: A[0:2, 2:4] * B[2:4, 0:2]
    add_strassen_block(0, 2, 0, 21)
    # Block 5: A[0:2, 2:4] * B[2:4, 2:4]
    add_strassen_block(0, 2, 2, 28)
    # Block 6: A[0:2, 2:4] * B[2:4, 4:6]
    add_strassen_block(0, 2, 4, 35)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
