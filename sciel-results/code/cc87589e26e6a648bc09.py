import time
import numpy as np
import random

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    # Schoolbook fallback
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
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U_arr[i*m+j]
                    v_val = V_arr[j*p+k]
                    w_val = W_arr[k*n+i]
                    s = np.dot(u_val, v_val * w_val)
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    def try_structured_search(target_rank, max_attempts=50000):
        """
        Search for U, V such that for each (i,k), there exists r with C[i,k,r] in {+-1, +-2}.
        Then construct W.
        """
        # We need to assign each (i,k) to an r.
        # There are n*p = 10 pairs.
        # We have target_rank r's.
        # We can try to find U, V randomly and then solve the assignment.
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            
            # Generate random U, V with small integers
            # Entries in {-1, 0, 1}
            U = np.random.randint(-1, 2, size=(n*m, target_rank))
            V = np.random.randint(-1, 2, size=(m*p, target_rank))
            
            # Calculate C[i, k, r] = sum_j U[i*m+j, r] * V[j*p+k, r]
            # C has shape (n, p, target_rank)
            C = np.zeros((n, p, target_rank), dtype=int)
            for i in range(n):
                for k in range(p):
                    for r in range(target_rank):
                        s = 0
                        for j in range(m):
                            s += U[i*m+j, r] * V[j*p+k, r]
                        C[i, k, r] = s
            
            # Check if we can assign each (i,k) to a unique r?
            # Actually, multiple (i,k) can use the same r, but W[k,i,r] would need to satisfy
            # sum_r C[i,k,r] W[k,i,r] = 1.
            # If we use only one r for each (i,k), then W[k,i,r] = 1/C[i,k,r].
            # We need C[i,k,r] to be +-1 or +-2.
            
            # Let's try to find a matching where each (i,k) is assigned to an r
            # such that C[i,k,r] is in {1, -1, 2, -2}.
            # It's not a strict matching, r can be reused.
            # But if r is reused for (i1,k1) and (i2,k2), W[k1,i1,r] and W[k2,i2,r] are independent.
            # So we just need: for each (i,k), there exists r such that C[i,k,r] in {1, -1, 2, -2}.
            
            valid = True
            W = np.zeros((p*n, target_rank), dtype=float)
            
            for i in range(n):
                for k in range(p):
                    found_r = -1
                    found_val = 0
                    for r in range(target_rank):
                        c = C[i, k, r]
                        if c in (1, -1, 2, -2):
                            found_r = r
                            found_val = c
                            break
                    if found_r == -1:
                        valid = False
                        break
                    # Set W[k*n+i, found_r] = 1/found_val
                    W[k*n+i, found_r] = 1.0 / found_val
                if not valid:
                    break
            
            if valid:
                # Check if W entries are integer or half-integer
                # 1/1=1, 1/-1=-1, 1/2=0.5, 1/-2=-0.5
                # These are all valid.
                # Now we must verify the full tensor decomposition
                # Because we assumed only one r contributes for each (i,k).
                # But the tensor equation is sum_j sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1.
                # If we set W[k,i,r] = 1/C[i,k,r] for the chosen r, and 0 otherwise,
                # then for a specific (i,k), the sum over r becomes C[i,k,r_chosen] * (1/C[i,k,r_chosen]) = 1.
                # And the sum over j is already included in C.
                # So the condition is satisfied.
                
                # Convert to lists
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                
                # Final validation to be safe
                if validate(U_list, V_list, W_list):
                    return {"U": U_list, "V": V_list, "W": W_list}
        
        return None

    # Try ranks from lower bound upwards
    # Lower bound is 15.
    # Try 15, 16, ...
    for rank in range(15, 30):
        if time.time() > deadline:
            break
        result = try_structured_search(rank, max_attempts=20000)
        if result is not None:
            return result

    # Fallback to schoolbook
    return get_schoolbook()
