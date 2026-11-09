import numpy as np
import random
import time

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Schoolbook fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Check if a decomposition is valid
    def is_valid(U, V, W, rank):
        # Convert to numpy for faster checking
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Check integer/half-integer
        # The problem says "integer/half-integer entries"
        # We assume the search produces integers, so this is fine.
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    s = 0
                    for r in range(rank):
                        s += U_np[u_idx, r] * V_np[v_idx, r] * W_np[w_idx, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    # Search for a solution of a given rank
    def search(rank, time_limit=5.0):
        start_time = time.time()
        U = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(nm)]
        V = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(mp)]
        W = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(pn)]
        
        # Convert to numpy for faster calculation in loop
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Precompute indices
        indices = [(i, j, k, i*m+j, j*p+k, k*n+i) for i in range(n) for j in range(m) for k in range(p)]
        
        while time.time() - start_time < time_limit:
            # Calculate errors
            errors = []
            for i, j, k, ui, vi, wi in indices:
                s = np.sum(U_np[ui, :] * V_np[vi, :] * W_np[wi, :])
                errors.append((s - 1.0, i, j, k, ui, vi, wi))
            
            # Check if solved
            if all(abs(e[0]) < 1e-9 for e in errors):
                # Convert back to lists of integers
                U_list = [[int(x) for x in row] for row in U_np]
                V_list = [[int(x) for x in row] for row in V_np]
                W_list = [[int(x) for x in row] for row in W_np]
                return U_list, V_list, W_list
            
            # Pick the worst error
            errors.sort(key=lambda x: abs(x[0]), reverse=True)
            _, i, j, k, ui, vi, wi = errors[0]
            
            # Try to fix by changing one variable
            # Randomly pick which matrix to change
            choice = random.choice([0, 1, 2])
            r = random.randint(0, rank - 1)
            
            if choice == 0:
                # Change U
                old_val = U_np[ui, r]
                U_np[ui, r] = old_val + random.choice([-1, 1])
            elif choice == 1:
                # Change V
                old_val = V_np[vi, r]
                V_np[vi, r] = old_val + random.choice([-1, 1])
            else:
                # Change W
                old_val = W_np[wi, r]
                W_np[wi, r] = old_val + random.choice([-1, 1])
                
            # Optional: Occasional random restart or larger jump
            if random.random() < 0.01:
                U_np[ui, r] = random.choice([-1, 0, 1])
                V_np[vi, r] = random.choice([-1, 0, 1])
                W_np[wi, r] = random.choice([-1, 0, 1])

        return None

    # Try to find a better decomposition
    best_result = None
    for rank in [25, 26, 27]:
        result = search(rank, time_limit=10.0)
        if result:
            U, V, W = result
            # Double check
            if is_valid(U, V, W, rank):
                best_result = {"U": U, "V": V, "W": W}
                break # Found a valid one, stop searching for higher ranks
        
    if best_result:
        return best_result
    else:
        return schoolbook()
