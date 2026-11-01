"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 2, 2, 5
    nm, mp, pn = n*m, m*p, p*n  # 4, 10, 10
    
    # Build support set
    support = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                support.append((a, b, c))
    support_set = set(support)
    
    # Schoolbook decomposition (rank = n*m*p = 20)
    rank_school = n * m * p
    U_school = np.zeros((nm, rank_school))
    V_school = np.zeros((mp, rank_school))
    W_school = np.zeros((pn, rank_school))
    for idx, (a, b, c) in enumerate(support):
        U_school[a, idx] = 1.0
        V_school[b, idx] = 1.0
        W_school[c, idx] = 1.0
    
    def to_list(arr):
        return arr.tolist()
    
    def compute_error(U, V, W):
        max_err = 0.0
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    val = np.dot(U[a, :], V[b, :] * W[c, :])
                    if (a, b, c) in support_set:
                        err = abs(val - 1.0)
                    else:
                        err = abs(val)
                    if err > max_err:
                        max_err = err
        return max_err
    
    def try_rank(r, num_attempts, time_budget):
        """Try to find a rank-r decomposition."""
        deadline = time.time() + time_budget
        for attempt in range(num_attempts):
            if time.time() > deadline:
                return None
            np.random.seed(12345 + r * 1000 + attempt)
            U = np.random.randn(nm, r)
            V = np.random.randn(mp, r)
            W = np.random.randn(pn, r)
            
            # Alternating minimization
            for iteration in range(500):
                if time.time() > deadline:
                    return None
                
                # Update U
                for a in range(nm):
                    rows = []
                    targets = []
                    for (a2, b, c) in support:
                        if a2 == a:
                            rows.append(V[b, :] * W[c, :])
                            targets.append(1.0)
                    for b in range(mp):
                        for c in range(pn):
                            if (a, b, c) not in support_set:
                                rows.append(V[b, :] * W[c, :])
                                targets.append(0.0)
                    if rows:
                        A_mat = np.array(rows)
                        b_vec = np.array(targets)
                        sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                        U[a, :] = sol
                
                # Update V
                for b in range(mp):
                    rows = []
                    targets = []
                    for (a, b2, c) in support:
                        if b2 == b:
                            rows.append(U[a, :] * W[c, :])
                            targets.append(1.0)
                    for a in range(nm):
                        for c in range(pn):
                            if (a, b, c) not in support_set:
                                rows.append(U[a, :] * W[c, :])
                                targets.append(0.0)
                    if rows:
                        A_mat = np.array(rows)
                        b_vec = np.array(targets)
                        sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                        V[b, :] = sol
                
                # Update W
                for c in range(pn):
                    rows = []
                    targets = []
                    for (a, b, c2) in support:
                        if c2 == c:
                            rows.append(U[a, :] * V[b, :])
                            targets.append(1.0)
                    for a in range(nm):
                        for b in range(mp):
                            if (a, b, c) not in support_set:
                                rows.append(U[a, :] * V[b, :])
                                targets.append(0.0)
                    if rows:
                        A_mat = np.array(rows)
                        b_vec = np.array(targets)
                        sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                        W[c, :] = sol
                
                # Check convergence every 10 iterations
                if iteration % 10 == 0:
                    err = compute_error(U, V, W)
                    if err < 1e-10:
                        return U, V, W
        
        return None
    
    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school
    
    deadline = time.time() + 110
    
    # Try ranks from 10 upward
    for r in range(10, rank_school):
        if time.time() > deadline - 15:
            break
        remaining = deadline - 15 - time.time()
        if remaining < 5:
            break
        
        # Allocate time: more attempts for lower ranks
        time_per_rank = min(remaining * 0.4, 30)
        result = try_rank(r, 20, time_per_rank)
        if result is not None:
            best_rank = r
            best_U, best_V, best_W = result
            break
    
    if best_rank < rank_school:
        return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
    else:
        return {"U": to_list(U_school), "V": to_list(V_school), "W": to_list(W_school)}
# EVOLVE-BLOCK-END
