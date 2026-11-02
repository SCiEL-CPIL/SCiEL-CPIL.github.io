"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 4
    nm = n * m
    mp = m * p
    pn = p * n

    # Schoolbook decomposition as fallback
    rank_school = n * m * p
    U_school = np.zeros((nm, rank_school))
    V_school = np.zeros((mp, rank_school))
    W_school = np.zeros((pn, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1

    def to_list(mat):
        return [[int(round(v * 1e6)) / 1e6 if abs(v - round(v * 1e6)) < 1e-9 else v for v in row] for row in mat]

    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school

    # Try to find lower rank decompositions using algebraic structure
    # The tensor T_{i,j,k} = delta_{i,k} for (i,j,k) in [0,n)x[0,m)x[0,p)
    # We want sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = delta_{i,k}

    # Strategy: Use the fact that for fixed k, the matrix M_k[i,j] = delta_{i,k}
    # This is a rank-1 matrix (column vector e_k) for each k.
    # Total tensor can be viewed as sum over k of e_k * e_k^T in the (i,j) plane
    # but with the V factor depending on j*p+k.

    # Let's try rank 22-24 using a structured approach
    # For 3x3x3, optimal rank is 23. For 3x3x4, we expect rank around 24-26.
    # Try to construct explicitly.

    # Approach: Use the 3x3x3 rank-23 decomposition as base, extend to 3x3x4
    # Actually, let's try a direct algebraic construction for lower rank.

    # Known: rank(M_{3,3,3}) = 23
    # For M_{3,3,4}, we can think of it as 4 slices of 3x3, each being e_k * e_k^T
    # But the V matrix couples j and k.

    # Let's try a gradient-based search for rank 24, 25, 26
    start_time = time.time()
    deadline = 150 - 10  # safety margin

    def verify(U, V, W, rank):
        """Check if U, V, W give exact decomposition."""
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    expected = 1.0 if i == k else 0.0
                    err += (val - expected) ** 2
        return err

    # Try structured rank-24 construction
    # Idea: For each pair (i,k), we need contribution only when i=k.
    # Let's use rank = 4*3 + 12 = 24? Or 3*3 + 3*4 = 21? No.
    # Let's try: for each k, use 3 terms for the i dimension (but only one non-zero per k)
    # That gives 3*3 = 9 terms. But we also need the j dimension to be handled.
    
    # Actually, let's use the following construction:
    # For each k in 0..2, we have a 3x3 identity-like slice.
    # For k=3, we have another slice.
    # Total: 4 slices * 3 (for the rank-1 within each slice) = 12 terms? 
    # But V couples j and k, so we can't just have independent slices.

    # Let me think differently. The tensor is:
    # T[i,j,k] = delta_{i,k}
    # We need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = delta_{i,k}

    # For fixed k, sum over i,j: U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] should form delta_{i,k} in (i,j)
    # For fixed k, let u_c[i,j] = U[i*m+j,c], v_c[j] = V[j*p+k,c], w_c[i] = W[k*n+i,c]
    # Then sum_c u_c[i,j] * v_c[j] * w_c[i] = delta_{i,k}

    # This is like a tensor train / matrix product state for the 3D tensor.

    # Let's try a simple construction with rank = 24:
    # Use 6 terms per k (4 k's * 6 = 24)
    # Or 3 terms per (i,k) pair where i=k: 3*3 = 9, plus extra terms for coupling?

    # Let's just do a numerical search for rank 22-26 with random initialization and gradient descent.
    rng = np.random.default_rng(42)

    for target_rank in [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]:
        if time.time() > deadline:
            break
        if target_rank >= best_rank:
            continue

        found = False
        for attempt in range(50):
            if time.time() > deadline:
                break

            U = rng.normal(size=(nm, target_rank)) * 0.1
            V = rng.normal(size=(mp, target_rank)) * 0.1
            W = rng.normal(size=(pn, target_rank)) * 0.1

            # Gradient descent
            lr = 0.01
            for it in range(2000):
                if time.time() > deadline:
                    break

                # Compute gradients
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)

                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            expected = 1.0 if i == k else 0.0
                            val = 0.0
                            for c in range(target_rank):
                                val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                            diff = val - expected

                            for c in range(target_rank):
                                dU[i*m+j, c] += diff * V[j*p+k, c] * W[k*n+i, c]
                                dV[j*p+k, c] += diff * U[i*m+j, c] * W[k*n+i, c]
                                dW[k*n+i, c] += diff * U[i*m+j, c] * V[j*p+k, c]

                U -= lr * dU
                V -= lr * dV
                W -= lr * dW

                if it % 500 == 0:
                    err = verify(U, V, W, target_rank)
                    if err < 1e-12:
                        found = True
                        break

            if found:
                best_rank = target_rank
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                break

        if time.time() > deadline:
            break

    # If we found a better rank, use it; otherwise use schoolbook
    if best_rank < rank_school:
        U_out = to_list(best_U)
        V_out = to_list(best_V)
        W_out = to_list(best_W)
        return {"U": U_out, "V": V_out, "W": W_out}
    else:
        U_out = to_list(U_school)
        V_out = to_list(V_school)
        W_out = to_list(W_school)
        return {"U": U_out, "V": V_out, "W": W_out}

# EVOLVE-BLOCK-END
