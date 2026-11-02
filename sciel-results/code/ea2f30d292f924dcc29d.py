<<<<<<< SEARCH
# EVOLVE-BLOCK-START
def construct():
    # The seed program provided is already the schoolbook decomposition.
    # To satisfy the "SEARCH" directive without risking timeout on a hard problem,
    # we explicitly check for any known small-dimension optimizations.
    # For 2x4x7, no simple hardcoded improvement is available in standard libraries.
    # We return the schoolbook decomposition which is rank 56.
    
    # Time budget check is trivial here as construction is O(nmp)
    # If we had a search algorithm, we would put it here.
    # Since we don't, we return the best known valid solution.
    
    n, m, p = DIMENSIONS
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
# EVOLVE-BLOCK-END
=======
# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    nmp = n * m * p
    
    import time
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin for 150s limit
    
    def schoolbook():
        rank = nmp
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_result = schoolbook()
    best_rank = nmp
    
    # Try to find lower rank decompositions
    lower_bound = max(nm, mp, pn)
    
    for rank in range(lower_bound, nmp):
        if time.time() > deadline - 15:
            break
        
        found = False
        max_attempts = 10
        for attempt in range(max_attempts):
            if time.time() > deadline - 3:
                break
            
            np.random.seed(attempt * 7919 + rank)
            U = np.random.randn(nm, rank)
            V = np.random.randn(mp, rank)
            W = np.random.randn(pn, rank)
            
            # Alternating least squares
            converged = False
            for it in range(100):
                if time.time() > deadline - 1:
                    break
                
                # Update U rows
                for i in range(n):
                    for j in range(m):
                        A = np.stack([V[j*p+k, :] * W[k*n+i, :] for k in range(p)], axis=0)
                        try:
                            U[i*m+j, :] = np.linalg.lstsq(A, np.ones(p), rcond=None)[0]
                        except:
                            pass
                
                # Update V rows
                for j in range(m):
                    for k in range(p):
                        A = np.stack([U[i*m+j, :] * W[k*n+i, :] for i in range(n)], axis=0)
                        try:
                            V[j*p+k, :] = np.linalg.lstsq(A, np.ones(n), rcond=None)[0]
                        except:
                            pass
                
                # Update W rows
                for k in range(p):
                    for i in range(n):
                        A = np.stack([U[i*m+j, :] * V[j*p+k, :] for j in range(m)], axis=0)
                        try:
                            W[k*n+i, :] = np.linalg.lstsq(A, np.ones(m), rcond=None)[0]
                        except:
                            pass
                
                # Check convergence
                max_err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                            err = abs(val - 1.0)
                            if err > max_err:
                                max_err = err
                
                if max_err < 1e-5:
                    converged = True
                    break
            
            if converged:
                found = True
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = rank
                break
        
        if found:
            break
    
    return best_result
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
