import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Try to find a rank < 120 decomposition
    # We try rank 119
    # This is a very hard problem, so we do a limited random search
    
    best_result = None
    best_rank = rank_school
    
    # Function to check if a decomposition is valid
    def check_decomposition(U, V, W, r):
        # U: (n*m, r), V: (m*p, r), W: (p*n, r)
        # Target: T[i,j,k] = 1
        # Sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
        # We can vectorize this check
        # Reshape U to (n, m, r), V to (m, p, r), W to (p, n, r)
        U_reshaped = np.array(U).reshape(n, m, r)
        V_reshaped = np.array(V).reshape(m, p, r)
        W_reshaped = np.array(W).reshape(p, n, r)
        
        # Compute the tensor product sum
        # T[i, j, k] = sum_c U[i, j, c] V[j, k, c] W[k, i, c]
        # This is a contraction.
        # We can use np.einsum
        # 'ijc,jkc,kic->ijk'
        T = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
        # Check if T is all ones
        return np.allclose(T, np.ones((n, m, p)), atol=1e-9)

    # Function to generate a random decomposition
    def random_decomposition(r):
        # Entries should be integers or half-integers
        # Let's pick from {-1, -0.5, 0, 0.5, 1}
        vals = np.array([-1, -0.5, 0, 0.5, 1])
        U = np.random.choice(vals, size=(n * m, r))
        V = np.random.choice(vals, size=(m * p, r))
        W = np.random.choice(vals, size=(p * n, r))
        return U, V, W

    # Try a few random restarts for rank 119
    # Given the complexity, this is unlikely to succeed quickly, but we try
    for rank in [119]:
        if time.time() > deadline:
            break
        for attempt in range(100): # Limit attempts
            if time.time() > deadline:
                break
            U, V, W = random_decomposition(rank)
            if check_decomposition(U, V, W, rank):
                best_result = {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
                best_rank = rank
                break
    
    # If no better rank found, use schoolbook
    if best_result is None:
        rank = rank_school
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        best_result = {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
        
    return best_result
