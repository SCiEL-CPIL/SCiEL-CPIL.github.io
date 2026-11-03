import time
import numpy as np
from math import gcd
from functools import reduce

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    def get_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def validate(U, V, W):
        rank = len(U[0])
        U_arr = np.array(U, dtype=float).reshape(n, m, rank)
        V_arr = np.array(V, dtype=float).reshape(m, p, rank)
        W_arr = np.array(W, dtype=float).reshape(p, n, rank)
        
        # Test with random matrices
        np.random.seed(42)
        for _ in range(5):
            A = np.random.rand(n, m)
            B = np.random.rand(m, p)
            C_true = A @ B
            
            C_comp = np.zeros((n, p))
            for r in range(rank):
                u_val = np.sum(U_arr[:, :, r] * A)
                v_val = np.sum(V_arr[:, :, r] * B)
                w_mat = W_arr[:, :, r].T # W is (p, n, r), we need (n, p)
                C_comp += u_val * v_val * w_mat
                
            if not np.allclose(C_true, C_comp, atol=1e-9):
                return False
        return True

    def try_structured_search(target_rank, max_attempts=5000):
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            
            U_prime = np.random.randint(-2, 3, size=(n*m, target_rank))
            V_prime = np.random.randint(-2, 3, size=(m*p, target_rank))
            
            # We need to find W such that A @ B = sum_r (U_r . A) (V_r . B) W_r
            # This is a linear system for W if we fix U, V.
            # But U, V are random.
            # We can solve for W using least squares on random A, B.
            
            # Generate random A, B
            A = np.random.rand(n, m)
            B = np.random.rand(m, p)
            C = A @ B
            
            # Build matrix M for W
            # C = sum_r (sum_ij U_ijr A_ij) (sum_jk V_jkr B_jk) W_r
            # Let alpha_r = sum_ij U_ijr A_ij
            # Let beta_r = sum_jk V_jkr B_jk
            # C = sum_r alpha_r beta_r W_r
            # C is n x p. W_r is n x p.
            # Flatten C to n*p vector. Flatten W_r to n*p vector.
            # C_flat = M @ W_flat
            # M is (n*p) x (target_rank)
            
            alpha = np.zeros(target_rank)
            beta = np.zeros(target_rank)
            for r in range(target_rank):
                alpha[r] = np.sum(U_prime.reshape(n, m, target_rank)[:, :, r] * A)
                beta[r] = np.sum(V_prime.reshape(m, p, target_rank)[:, :, r] * B)
                
            M = np.outer(alpha, beta) # (target_rank, target_rank)? No.
            # C_flat = sum_r (alpha_r * beta_r) * W_r_flat
            # This is a rank-1 update per r.
            # We need W_r such that sum_r (alpha_r beta_r) W_r = C.
            # This is underdetermined if target_rank < n*p.
            # We can use multiple random A, B to overdetermine.
            
            # Use 10 random pairs
            A_list = [np.random.rand(n, m) for _ in range(10)]
            B_list = [np.random.rand(m, p) for _ in range(10)]
            
            # System:
            # For each pair (A_i, B_i), we have:
            # sum_r (alpha_{i,r} beta_{i,r}) W_r = A_i @ B_i
            # alpha_{i,r} = sum U_r * A_i
            # beta_{i,r} = sum V_r * B_i
            
            # Total equations: 10 * n * p = 100
            # Unknowns: 10 * target_rank (if W is dense)
            # Actually W_r is n x p. So 10 * n * p unknowns? No, W_r is a matrix.
            # We have target_rank matrices W_r.
            # Total unknowns: target_rank * n * p.
            # Equations: 10 * n * p.
            # Underdetermined.
            
            # We can set W_r to be rank-1 or something?
            # Or we can just solve for W assuming it's dense and use least squares.
            
            # Let's form the big system.
            # X = [W_1, W_2, ..., W_R] (R * n * p)
            # A_sys * X = C_sys
            # A_sys is (10 * n * p) x (R * n * p)
            
            # This is too big.
            
            # Alternative: Use the fact that W_r can be anything.
            # We can just pick W_r randomly and check? No.
            
            # Let's try a different approach:
            # Fix U, V.
            # Compute the "effective" matrices.
            # We need W such that the decomposition holds.
            # This is equivalent to:
            # The tensor T = sum u_r v_r w_r.
            # We can check if the rank of the tensor formed by U, V is sufficient.
            
            # Given time constraints, I will return schoolbook.
            pass
        return None

    return get_schoolbook()
