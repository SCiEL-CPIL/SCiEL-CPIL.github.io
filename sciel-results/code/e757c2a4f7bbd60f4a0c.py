import numpy as np
import time
import random

DIMENSIONS = [2, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 20s margin before 150s kill

    n, m, p = 2, 3, 4
    # U: n*m = 6 x rank
    # V: m*p = 12 x rank
    # W: p*n = 8 x rank

    # Start with the known rank-22 block decomposition
    rank = 22
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def add_term(u_indices, v_indices, w_indices, col):
        for idx, val in u_indices:
            U[idx][col] += val
        for idx, val in v_indices:
            V[idx][col] += val
        for idx, val in w_indices:
            W[idx][col] += val

    # --- Block 1: A[0:2, 0:2] * B[0:2, 0:2] ---
    add_term([(0, 1), (4, 1)], [(0, 1), (5, 1)], [(0, 1), (3, 1)], 0)
    add_term([(3, 1), (4, 1)], [(0, 1)], [(1, 1), (3, -1)], 1)
    add_term([(0, 1)], [(1, 1), (5, -1)], [(2, 1), (3, 1)], 2)
    add_term([(4, 1)], [(4, 1), (0, -1)], [(0, 1), (1, 1)], 3)
    add_term([(0, 1), (1, 1)], [(5, 1)], [(0, -1), (2, 1)], 4)
    add_term([(3, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)], 5)
    add_term([(1, 1), (4, -1)], [(4, 1), (5, 1)], [(0, 1)], 6)

    # --- Block 2: A[0:2, 0:2] * B[0:2, 2:4] ---
    add_term([(0, 1), (4, 1)], [(2, 1), (7, 1)], [(4, 1), (7, 1)], 7)
    add_term([(3, 1), (4, 1)], [(2, 1)], [(5, 1), (7, -1)], 8)
    add_term([(0, 1)], [(3, 1), (7, -1)], [(6, 1), (7, 1)], 9)
    add_term([(4, 1)], [(6, 1), (2, -1)], [(4, 1), (5, 1)], 10)
    add_term([(0, 1), (1, 1)], [(7, 1)], [(4, -1), (6, 1)], 11)
    add_term([(3, 1), (0, -1)], [(2, 1), (3, 1)], [(7, 1)], 12)
    add_term([(1, 1), (4, -1)], [(6, 1), (7, 1)], [(4, 1)], 13)

    # --- Block 3: A[0:2, 2:3] * B[2:3, 0:4] ---
    add_term([(2, 1)], [(8, 1)], [(0, 1)], 14)
    add_term([(2, 1)], [(9, 1)], [(2, 1)], 15)
    add_term([(2, 1)], [(10, 1)], [(4, 1)], 16)
    add_term([(2, 1)], [(11, 1)], [(6, 1)], 17)
    add_term([(5, 1)], [(8, 1)], [(1, 1)], 18)
    add_term([(5, 1)], [(9, 1)], [(3, 1)], 19)
    add_term([(5, 1)], [(10, 1)], [(5, 1)], 20)
    add_term([(5, 1)], [(11, 1)], [(7, 1)], 21)

    def verify_decomposition(U_mat, V_mat, W_mat):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        # The tensor T[i,j,k] = 1 if i is row of A, j is col of B, k is (row,col) of C
        # such that C[k] = sum_j A[i,j] * B[j,k]
        # In our indexing:
        # U has shape (n*m, rank): U[a_idx, r]
        # V has shape (m*p, rank): V[b_idx, r]  
        # W has shape (p*n, rank): W[c_idx, r]
        # The decomposition says: for each output C[i,k],
        #   C[i,k] = sum_j A[i,j] * B[j,k]
        # The tensor entry T[(i,j), (j,k), (i,k)] should be 1
        # And we need: sum_r U[a_idx, r] * V[b_idx, r] * W[c_idx, r] = T[a_idx, b_idx, c_idx]
        # where a_idx = i*m+j, b_idx = j*p+k, c_idx = k*n+i
        
        # Build the target tensor
        # T is a 3-tensor with dimensions (n*m, m*p, p*n)
        # T[a_idx, b_idx, c_idx] = 1 if a_idx = i*m+j, b_idx = j*p+k, c_idx = k*n+i
        # Note: a_idx and b_idx must share the same j (the inner dimension)
        
        # For each r, we compute the outer product U[:,r] ⊗ V[:,r] ⊗ W[:,r]
        # and sum over r. This should equal T.
        
        # Let's compute the sum and check against T
        # T[a_idx, b_idx, c_idx] = 1 iff:
        #   i = a_idx // m, j = a_idx % m
        #   j2 = b_idx // p, k = b_idx % p
        #   i2 = c_idx % n, k2 = c_idx // n
        #   and j == j2 and i == i2 and k == k2
        
        U_np = np.array(U_mat, dtype=float)
        V_np = np.array(V_mat, dtype=float)
        W_np = np.array(W_mat, dtype=float)
        
        rank_val = U_np.shape[1]
        
        # Compute the sum of outer products
        # This is expensive for large rank, but rank is small here
        result = np.zeros((n*m, m*p, p*n))
        for r in range(rank_val):
            result += np.einsum('a,b,c->abc', U_np[:, r], V_np[:, r], W_np[:, r])
        
        # Build target tensor
        target = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    c_idx = k * n + i
                    target[a_idx, b_idx, c_idx] = 1.0
        
        # Check if result matches target
        return np.allclose(result, target, atol=1e-10)

    # Verify the initial decomposition
    assert verify_decomposition(U, V, W), "Initial decomposition is invalid!"

    # Try to reduce rank by merging columns or finding linear dependencies
    # Strategy: try to express one column as a linear combination of others
    # and eliminate it if possible
    
    best_rank = rank
    best_U, best_V, best_W = U, V, W

    # Try to reduce rank by checking if any column is redundant
    # For a given set of columns, if one column can be written as a linear combination
    # of the others in all three matrices simultaneously, we can eliminate it.
    
    def try_reduce_rank(U_mat, V_mat, W_mat):
        """Try to reduce rank by eliminating redundant columns."""
        U_np = np.array(U_mat, dtype=float)
        V_np = np.array(V_mat, dtype=float)
        W_np = np.array(W_mat, dtype=float)
        r = U_np.shape[1]
        
        # For each column c, check if it's in the span of the others
        # This is a hard problem in general, but we can try simple cases
        # where the column is a linear combination with small integer coefficients
        
        for c in range(r):
            # Try to see if column c can be expressed as a combination of others
            # with coefficients in {-1, 0, 1}
            others = [i for i in range(r) if i != c]
            if len(others) < 1:
                continue
            
            # Try all subsets of others with coefficients -1, 0, 1
            # This is exponential, so limit to small sets
            # Try pairs first
            for i in others:
                for j in others:
                    if i == j:
                        continue
                    for ci in [-1, 0, 1]:
                        for cj in [-1, 0, 1]:
                            if ci == 0 and cj == 0:
                                continue
                            # Check if U[:,c] = ci*U[:,i] + cj*U[:,j]
                            # and similarly for V and W
                            if np.allclose(U_np[:, c], ci * U_np[:, i] + cj * U_np[:, j], atol=1e-10) and \
                               np.allclose(V_np[:, c], ci * V_np[:, i] + cj * V_np[:, j], atol=1e-10) and \
                               np.allclose(W_np[:, c], ci * W_np[:, i] + cj * W_np[:, j], atol=1e-10):
                                # Column c is redundant
                                new_U = [row[:c] + row[c+1:] for row in U_mat]
                                new_V = [row[:c] + row[c+1:] for row in V_mat]
                                new_W = [row[:c] + row[c+1:] for row in W_mat]
                                return new_U, new_V, new_W
            # Try triples
            for i in others:
                for j in others:
                    if i == j:
                        continue
                    for k in others:
                        if i == k or j == k:
                            continue
                        for ci in [-1, 0, 1]:
                            for cj in [-1, 0, 1]:
                                for ck in [-1, 0, 1]:
                                    if ci == 0 and cj == 0 and ck == 0:
                                        continue
                                    if np.allclose(U_np[:, c], ci * U_np[:, i] + cj * U_np[:, j] + ck * U_np[:, k], atol=1e-10) and \
                                       np.allclose(V_np[:, c], ci * V_np[:, i] + cj * V_np[:, j] + ck * V_np[:, k], atol=1e-10) and \
                                       np.allclose(W_np[:, c], ci * W_np[:, i] + cj * W_np[:, j] + ck * W_np[:, k], atol=1e-10):
                                        new_U = [row[:c] + row[c+1:] for row in U_mat]
                                        new_V = [row[:c] + row[c+1:] for row in V_mat]
                                        new_W = [row[:c] + row[c+1:] for row in W_mat]
                                        return new_U, new_V, new_W
        
        return U_mat, V_mat, W_mat

    # Attempt rank reduction
    current_U, current_V, current_W = U, V, W
    for _ in range(10):  # Try up to 10 reductions
        new_U, new_V, new_W = try_reduce_rank(current_U, current_V, current_W)
        if len(new_U[0]) < len(current_U[0]):
            current_U, current_V, current_W = new_U, new_V, new_W
            if len(current_U[0]) < best_rank:
                best_rank = len(current_U[0])
                best_U, best_V, best_W = current_U, current_V, current_W
        else:
            break

    if time.time() > deadline:
        return {"U": best_U, "V": best_V, "W": best_W}

    # Try a different approach: use random linear combinations to find
    # a lower-rank decomposition via tensor rank minimization
    
    # The tensor T has dimensions (6, 12, 8). We want to find the minimum rank
    # such that T = sum_r u_r ⊗ v_r ⊗ w_r
    
    # Use the fact that we can permute and scale the factors
    # Try to find a decomposition with rank < 22 by solving a system of equations
    
    # For rank r, we need to find U (6xr), V (12xr), W (8xr) such that
    # for all (a,b,c): sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
    
    # This is a system of 6*12*8 = 576 equations in 6*r + 12*r + 8*r = 26*r unknowns
    # But it's nonlinear (bilinear in pairs of matrices)
    
    # Try rank 21
    for target_rank in range(best_rank - 1, 15, -1):
        if time.time() > deadline:
            break
        
        # Initialize with random values
        U_try = np.zeros((n*m, target_rank))
        V_try = np.zeros((m*p, target_rank))
        W_try = np.zeros((p*n, target_rank))
        
        # Use the existing decomposition as a starting point
        # Take the first target_rank columns
        U_try[:, :target_rank] = np.array(best_U)[:, :target_rank]
        V_try[:, :target_rank] = np.array(best_V)[:, :target_rank]
        W_try[:, :target_rank] = np.array(best_W)[:, :target_rank]
        
        # Try to optimize using gradient descent on the nonlinear objective
        # f = sum_{a,b,c} (sum_r U[a,r]*V[b,r]*W[c,r] - T[a,b,c])^2
        
        target_tensor = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    c_idx = k * n + i
                    target_tensor[a_idx, b_idx, c_idx] = 1.0
        
        # Gradient descent
        lr = 0.01
        for step in range(200):
            if time.time() > deadline:
                break
            
            # Compute current tensor
            current_tensor = np.zeros((n*m, m*p, p*n))
            for r in range(target_rank):
                current_tensor += np.einsum('a,b,c->abc', U_try[:, r], V_try[:, r], W_try[:, r])
            
            error = current_tensor - target_tensor
            loss = np.sum(error ** 2)
            
            if loss < 1e-10:
                break
            
            # Compute gradients
            dU = np.zeros_like(U_try)
            dV = np.zeros_like(V_try)
            dW = np.zeros_like(W_try)
            
            for r in range(target_rank):
                dU[:, r] = np.einsum('abc,bc->a', error, V_try[:, r] * W_try[:, r])
                dV[:, r] = np.einsum('abc,ac->b', error, U_try[:, r] * W_try[:, r])
                dW[:, r] = np.einsum('abc,ab->c', error, U_try[:, r] * V_try[:, r])
            
            U_try -= lr * dU
            V_try -= lr * dV
            W_try -= lr * dW
        
        if loss < 1e-10:
            # Convert to list format
            U_list = [[U_try[i, j] for j in range(target_rank)] for i in range(n*m)]
            V_list = [[V_try[i, j] for j in range(target_rank)] for i in range(m*p)]
            W_list = [[W_try[i, j] for j in range(target_rank)] for i in range(p*n)]
            
            if verify_decomposition(U_list, V_list, W_list):
                best_rank = target_rank
                best_U, best_V, best_W = U_list, V_list, W_list
                break

    return {"U": best_U, "V": best_V, "W": best_W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
