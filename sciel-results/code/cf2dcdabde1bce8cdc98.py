<<<<<<< SEARCH
import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # Split B (4x6) into B1 (4x4) and B2 (4x2)
    # C = A * B = A * B1 + A * B2
    # C1 = A * B1 (4x4), C2 = A * B2 (4x2)
    
    # Rank for 4x4x4 using Laderman's algorithm is 49
    # Rank for 4x4x2 using Schoolbook is 32
    rank_laderman = 49
    rank_schoolbook = 32
    total_rank = rank_laderman + rank_schoolbook
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # --- Part 1: 4x4x4 (Laderman) ---
    # Indices for B1: cols 0-3
    # U, V, W for Laderman
    # Note: I need the actual Laderman matrices. 
    # Since I cannot guarantee the exact 49 coefficients from memory, 
    # I will use a fallback: if I were to use schoolbook for 4x4x4, rank is 64.
    # 64 + 32 = 96. No improvement.
    # I will stick to schoolbook for the entire 4x4x6 to ensure correctness, 
    # as I cannot reliably generate Laderman's specific factors here.
    
    # Reverting to Schoolbook for safety
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
=======
import numpy as np
import time
import random

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # Leave 5s margin before 150s kill
    
    # Build the multiplication tensor T[i,j,k] = 1 if this is a valid term
    # The tensor has shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    
    # We want to find rank-r decomposition:
    # T = sum_{t=1}^{r} U[:,t] * V[:,t] * W[:,t]
    # where U is (n*m x r), V is (m*p x r), W is (p*n x r)
    
    # Start with schoolbook rank
    schoolbook_rank = n * m * p
    best_result = None
    best_rank = schoolbook_rank
    
    # Generate schoolbook solution as fallback
    U_school = [[0] * schoolbook_rank for _ in range(n * m)]
    V_school = [[0] * schoolbook_rank for _ in range(m * p)]
    W_school = [[0] * schoolbook_rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    
    # Try to find lower rank decompositions
    # Use iterative refinement with random initialization
    
    def verify_decomposition(U, V, W, tol=1e-10):
        """Verify that U, V, W correctly decompose the multiplication tensor"""
        r = len(U[0])
        # Check all (i,j,k) triples
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0
                    actual = 0.0
                    for t in range(r):
                        actual += U[i*m+j][t] * V[j*p+k][t] * W[k*n+i][t]
                    if abs(actual - expected) > tol:
                        return False
        # Check that non-tensor entries are zero
        # For any (a, b, c) where the corresponding i,j,k don't satisfy the pattern
        # a = i*m+j, b = j'*p+k, c = k'*n+i'
        # The term contributes only if j == j' and k == k' and i == i'
        # So for random (a, b, c), check if sum_t U[a,t]*V[b,t]*W[c,t] = 0
        # unless (a,b,c) corresponds to a valid (i,j,k)
        
        # Build set of valid indices
        valid = set()
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    valid.add((i*m+j, j*p+k, k*n+i))
        
        # Sample some invalid indices to verify they give 0
        for _ in range(100):
            a = random.randint(0, n*m-1)
            b = random.randint(0, m*p-1)
            c = random.randint(0, p*n-1)
            if (a, b, c) not in valid:
                actual = 0.0
                for t in range(r):
                    actual += U[a][t] * V[b][t] * W[c][t]
                if abs(actual) > tol:
                    return False
        return True
    
    def try_rank(r, max_iters=500):
        """Try to find a rank-r decomposition"""
        nonlocal best_result, best_rank
        if r >= best_rank:
            return False
        
        U = np.zeros((n*m, r))
        V = np.zeros((m*p, r))
        W = np.zeros((p*n, r))
        
        # Initialize with random values
        U[:] = np.random.randn(n*m, r) * 0.1
        V[:] = np.random.randn(m*p, r) * 0.1
        W[:] = np.random.randn(p*n, r) * 0.1
        
        # Iterative refinement: minimize ||T - sum_t U[:,t] V[:,t] W[:,t]||^2
        # Use gradient descent on the factor matrices
        
        lr = 0.01
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Compute residual for each tensor entry
            # For efficiency, compute the full tensor and compare
            
            # Build the tensor from current factors
            # T[a,b,c] = sum_t U[a,t] * V[b,t] * W[c,t]
            
            # For each factor, compute gradient
            # dL/dU[a,t] = sum_{b,c: (a,b,c) in tensor} T[a,b,c] * V[b,t] * W[c,t]
            #             - sum_{b,c} T_hat[a,b,c] * V[b,t] * W[c,t]
            
            # This is expensive, so use a simplified approach:
            # For each valid (i,j,k), the target is 1
            # For invalid (a,b,c), the target is 0
            
            # Compute gradients
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            # For valid entries
            for i in range(n):
                a = i * m
                for j in range(m):
                    b = j * p
                    for k in range(p):
                        c = k * n + i
                        # Target: 1
                        for t in range(r):
                            vu = V[b, t] * W[c, t]
                            vw = U[a + j, t] * V[b, t]
                            wu = U[a + j, t] * W[c, t]
                            # T_hat contribution
                            th = U[a+j, t] * V[b, t] * W[c, t]
                            # Gradient of (target - th)^2
                            # d/dU = -2*(1-th)*V*W
                            # d/dV = -2*(1-th)*U*W
                            # d/dW = -2*(1-th)*U*V
                            grad_U[a+j, t] += -2 * (1 - th) * vu
                            grad_V[b, t] += -2 * (1 - th) * wu
                            grad_W[c, t] += -2 * (1 - th) * vw
            
            # For some invalid entries (sampled)
            for _ in range(50):
                a = random.randint(0, n*m-1)
                b = random.randint(0, m*p-1)
                c = random.randint(0, p*n-1)
                # Check if valid
                ia = a // m
                ja = a % m
                ib = b // p
                jb = b % p
                ic = c % n
                kc = c // n
                if not (ja == ib and jb == kc and ia == ic):
                    # Invalid entry, target is 0
                    th = 0.0
                    for t in range(r):
                        th += U[a, t] * V[b, t] * W[c, t]
                    for t in range(r):
                        vu = V[b, t] * W[c, t]
                        vw = U[a, t] * V[b, t]
                        wu = U[a, t] * W[c, t]
                        grad_U[a, t] += -2 * (0 - th) * vu
                        grad_V[b, t] += -2 * (0 - th) * wu
                        grad_W[c, t] += -2 * (0 - th) * vw
            
            # Update
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Check convergence
            if it % 50 == 0:
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            th = 0.0
                            for t in range(r):
                                th += U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
                            err += (1 - th) ** 2
                if err < 1e-12:
                    # Check invalid entries
                    valid = True
                    for _ in range(100):
                        a = random.randint(0, n*m-1)
                        b = random.randint(0, m*p-1)
                        c = random.randint(0, p*n-1)
                        ia = a // m
                        ja = a % m
                        ib = b // p
                        jb = b % p
                        ic = c % n
                        kc = c // n
                        if not (ja == ib and jb == kc and ia == ic):
                            th = 0.0
                            for t in range(r):
                                th += U[a, t] * V[b, t] * W[c, t]
                            if abs(th) > 1e-10:
                                valid = False
                                break
                    if valid:
                        # Convert to list format
                        U_list = [[int(round(U[i, t] * 2)) / 2 for t in range(r)] for i in range(n*m)]
                        V_list = [[int(round(V[i, t] * 2)) / 2 for t in range(r)] for i in range(m*p)]
                        W_list = [[int(round(W[i, t] * 2)) / 2 for t in range(r)] for i in range(p*n)]
                        if verify_decomposition(U_list, V_list, W_list):
                            best_result = {"U": U_list, "V": V_list, "W": W_list}
                            best_rank = r
                            return True
        return False
    
    # Try ranks from low to high
    for r in range(20, schoolbook_rank):
        if time.time() > deadline - 5:
            break
        if try_rank(r):
            break
    
    return best_result
>>>>>>> REPLACE
