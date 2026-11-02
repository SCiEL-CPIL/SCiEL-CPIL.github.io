import numpy as np
import time

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0  # 10s safety margin

    # Target tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    # Schoolbook decomposition (Rank 48)
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1

    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school

    # Try to find a lower rank decomposition using structured search
    # Strategy: Use a "transposed Strassen" block structure
    # For M_{3,4,4}, partition the 4x4 middle into 2x2 blocks
    # This gives us a structured subspace to search in

    # The key idea: exploit the rectangular structure
    # A is 3x4, B is 4x4, C is 3x4
    # We can write B as [[B11, B12], [B21, B22]] where each Bij is 2x2
    # Then C = A * B can be computed using block operations

    # Let's try a systematic search with alternating least squares
    # We'll use the fact that the tensor has a specific structure

    def tensor_to_matrix(T_tensor, U, V):
        """Compute the matrix representation of sum_k U[:,k] outer V[:,k] outer W[:,k]"""
        # This is the bilinear map: for each output index c,
        # C[c] = sum_k W[c,k] * (U^T V)[c,k]
        # Actually, we need to compute the full 3D tensor
        pass

    def compute_tensor(U, V, W):
        """Compute the 3D tensor from the decomposition"""
        n, m, p = DIMENSIONS
        T_computed = np.zeros((n*m, m*p, p*n))
        for k in range(U.shape[1]):
            T_computed += np.einsum('a,b,c->abc', U[:, k], V[:, k], W[:, k])
        return T_computed

    def tensor_error(U, V, W, T_target):
        """Compute the Frobenius norm of the error"""
        T_computed = compute_tensor(U, V, W)
        return np.linalg.norm(T_computed - T_target)

    # Try ranks from 20 up to 47
    # Use a structured initialization based on block decomposition
    for R in range(20, 48):
        if time.time() > deadline - 20:
            break

        # Try multiple random restarts
        found = False
        for restart in range(5):
            if time.time() > deadline - 10:
                break

            # Structured initialization: use the schoolbook pattern but with random mixing
            U = np.zeros((n*m, R))
            V = np.zeros((m*p, R))
            W = np.zeros((p*n, R))

            # Initialize with random values in {-1, 0, 1}
            U = np.random.choice([-1, 0, 1], size=(n*m, R))
            V = np.random.choice([-1, 0, 1], size=(m*p, R))
            W = np.random.choice([-1, 0, 1], size=(p*n, R))

            # Alternating least squares optimization
            # Fix U and V, optimize W
            # Fix U and W, optimize V
            # Fix V and W, optimize U

            for iteration in range(50):
                if time.time() > deadline - 5:
                    break

                # Optimize W: for each c, W[c,:] should minimize
                # || T_target[:, :, c] - sum_k U[:,k] outer V[:,k] * W[c,k] ||
                # This is a linear least squares problem
                # Let M_c = T_target[:, :, c] (shape n*m x m*p)
                # Let A_c be the matrix where A_c[a, b, k] = U[a,k] * V[b,k]
                # We want to find W[c,:] such that sum_k W[c,k] * U[:,k] outer V[:,k] approximates M_c
                # This is: for each (a,b), M_c[a,b] = sum_k U[a,k] * V[b,k] * W[c,k]
                # So we can solve: (U^T * V) @ W[c,:] = M_c.ravel() ? No, that's not right.

                # Actually, the constraint is:
                # T_computed[a,b,c] = sum_k U[a,k] * V[b,k] * W[c,k]
                # For fixed a,b, this is linear in W[c,:].
                # But we want to optimize all W[c,:] simultaneously.

                # Better approach: use gradient descent
                # Compute gradient with respect to W
                # dT/dW[c,k] = sum_{a,b} (T_target[a,b,c] - T_computed[a,b,c]) * U[a,k] * V[b,k]

                T_computed = compute_tensor(U, V, W)
                error = T_computed - T_target

                # Gradient for W: dW[c,k] = sum_{a,b} error[a,b,c] * U[a,k] * V[b,k]
                # This can be computed as: W_grad = np.einsum('abc,ak,bk->ck', error, U, V)
                W_grad = np.einsum('abc,ak,bk->ck', error, U, V)

                # Gradient step
                lr = 0.01
                W -= lr * W_grad

                # Similarly for U and V
                U_grad = np.einsum('abc,bk,ck->ak', error, V, W)
                V_grad = np.einsum('abc,ak,ck->bk', error, U, W)

                U -= lr * U_grad
                V -= lr * V_grad

            # Check if we found an exact decomposition
            T_computed = compute_tensor(U, V, W)
            max_error = np.max(np.abs(T_computed - T_target))

            if max_error < 1e-6:
                # We found an exact decomposition!
                best_U, best_V, best_W = U, V, W
                best_rank = R
                found = True
                break

        if found:
            break

    # Return the best found
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
