import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = 2, 3, 4
    
    # Target tensor T of shape (n*m, m*p, p*n)
    # T[alpha, beta, gamma] = 1 if alpha=i*m+j, beta=j*p+k, gamma=k*n+i
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                T[alpha, beta, gamma] = 1.0

    def check_decomposition(U, V, W):
        """Check if U, V, W exactly decompose T"""
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.linalg.norm(T - T_approx)
        return err < 1e-9

    def try_rank_als(R, max_iters=50, seed=42):
        """Try to find rank-R decomposition using ALS"""
        if time.time() > deadline - 15:
            return None
            
        np.random.seed(seed)
        U = np.random.randn(n * m, R)
        V = np.random.randn(m * p, R)
        W = np.random.randn(p * n, R)
        
        for it in range(max_iters):
            if time.time() > deadline - 15:
                return None
                
            # Update U
            for r in range(R):
                num = np.einsum('abc,b,c->a', T, V[:, r], W[:, r])
                den = np.dot(V[:, r], V[:, r]) * np.dot(W[:, r], W[:, r])
                if den > 1e-10:
                    U[:, r] = num / den
                else:
                    U[:, r] = 0
                    
            # Update V
            for r in range(R):
                num = np.einsum('abc,a,c->b', T, U[:, r], W[:, r])
                den = np.dot(U[:, r], U[:, r]) * np.dot(W[:, r], W[:, r])
                if den > 1e-10:
                    V[:, r] = num / den
                else:
                    V[:, r] = 0
                    
            # Update W
            for r in range(R):
                num = np.einsum('abc,a,b->c', T, U[:, r], V[:, r])
                den = np.dot(U[:, r], U[:, r]) * np.dot(V[:, r], V[:, r])
                if den > 1e-10:
                    W[:, r] = num / den
                else:
                    W[:, r] = 0
        
        # Check error
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.linalg.norm(T - T_approx)
        
        if err < 1e-6:
            # Try to round to half-integers
            U_int = np.round(U * 2) / 2
            V_int = np.round(V * 2) / 2
            W_int = np.round(W * 2) / 2
            
            if check_decomposition(U_int, V_int, W_int):
                return U_int, V_int, W_int
            else:
                # Try rounding to integers
                U_int = np.round(U)
                V_int = np.round(V)
                W_int = np.round(W)
                if check_decomposition(U_int, V_int, W_int):
                    return U_int, V_int, W_int
        return None

    def try_rank_linear_algebra(R):
        """Try to find rank-R decomposition using linear algebra approach"""
        if time.time() > deadline - 15:
            return None
            
        # For matrix multiplication tensor, we can use the fact that
        # the tensor has a special structure. Let's try a direct approach.
        
        # The tensor T[alpha, beta, gamma] = 1 if alpha=i*m+j, beta=j*p+k, gamma=k*n+i
        # We can think of this as: for each (i,j,k), we need a term
        # U[alpha,r] * V[beta,r] * W[gamma,r] = 1 for the specific (alpha,beta,gamma)
        
        # Strategy: use the known Strassen-like decomposition for 2x2 blocks
        # and extend to 2x3 and 3x4
        
        # For 2x2 * 2x2, rank 7 is known (Strassen)
        # For 2x3 * 3x4, we need to handle the extra dimensions
        
        # Let's try a systematic approach: enumerate all (i,j,k) triples
        # and try to find a minimal set of terms
        
        # Actually, let's use a different approach: 
        # The tensor can be written as a sum of rank-1 tensors.
        # We can try to find a basis for the space of rank-1 tensors that span T.
        
        # For now, let's fall back to the manual construction if this doesn't work quickly
        
        return None

    # Try Rank 21 with multiple random seeds
    for seed in [42, 123, 456, 789, 1000]:
        if time.time() > deadline - 20:
            break
        result = try_rank_als(21, max_iters=80, seed=seed)
        if result:
            U, V, W = result
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}

    # Fallback to Rank 22 (Current Program Logic)
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

    # Block 1: A[0:2, 0:2] * B[0:2, 0:2]
    add_term([(0, 1), (4, 1)], [(0, 1), (5, 1)], [(0, 1), (3, 1)], 0)
    add_term([(3, 1), (4, 1)], [(0, 1)], [(1, 1), (3, -1)], 1)
    add_term([(0, 1)], [(1, 1), (5, -1)], [(2, 1), (3, 1)], 2)
    add_term([(4, 1)], [(4, 1), (0, -1)], [(0, 1), (1, 1)], 3)
    add_term([(0, 1), (1, 1)], [(5, 1)], [(0, -1), (2, 1)], 4)
    add_term([(3, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)], 5)
    add_term([(1, 1), (4, -1)], [(4, 1), (5, 1)], [(0, 1)], 6)

    # Block 2: A[0:2, 0:2] * B[0:2, 2:4]
    add_term([(0, 1), (4, 1)], [(2, 1), (7, 1)], [(4, 1), (7, 1)], 7)
    add_term([(3, 1), (4, 1)], [(2, 1)], [(5, 1), (7, -1)], 8)
    add_term([(0, 1)], [(3, 1), (7, -1)], [(6, 1), (7, 1)], 9)
    add_term([(4, 1)], [(6, 1), (2, -1)], [(4, 1), (5, 1)], 10)
    add_term([(0, 1), (1, 1)], [(7, 1)], [(4, -1), (6, 1)], 11)
    add_term([(3, 1), (0, -1)], [(2, 1), (3, 1)], [(7, 1)], 12)
    add_term([(1, 1), (4, -1)], [(6, 1), (7, 1)], [(4, 1)], 13)

    # Block 3: A[0:2, 2:3] * B[2:3, 0:4]
    add_term([(2, 1)], [(8, 1)], [(0, 1)], 14)
    add_term([(2, 1)], [(9, 1)], [(2, 1)], 15)
    add_term([(2, 1)], [(10, 1)], [(4, 1)], 16)
    add_term([(2, 1)], [(11, 1)], [(6, 1)], 17)
    add_term([(5, 1)], [(8, 1)], [(1, 1)], 18)
    add_term([(5, 1)], [(9, 1)], [(3, 1)], 19)
    add_term([(5, 1)], [(10, 1)], [(5, 1)], 20)
    add_term([(5, 1)], [(11, 1)], [(7, 1)], 21)

    for mat in [U, V, W]:
        for row in mat:
            for idx, val in enumerate(row):
                row[idx] = int(round(val))

    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
