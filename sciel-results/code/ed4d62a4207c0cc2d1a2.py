"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    # Build the list of non-zero (a,b,c) triples
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                triples.append((a, b, c))
    
    # Try to find a low-rank decomposition
    # For each rank, try multiple random initializations
    
    for rank in range(12, 25):
        if time.time() > deadline:
            break
        
        for trial in range(50):
            if time.time() > deadline:
                break
            
            np.random.seed(trial * 1000 + rank)
            
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            
            # Alternating least squares, only on the 24 constraints
            for it in range(100):
                if time.time() > deadline:
                    break
                
                # Update U: for each a, solve min sum_r U[a,r] * (V[:,r]*W[:,r])[b,c] = T[a,b,c]
                # For each a, the constraint is: for all (a,b,c) in triples with this a:
                #   sum_r U[a,r] * V[b,r] * W[c,r] = 1
                # This is a linear system in U[a,:]
                
                U_new = np.zeros_like(U)
                for a in range(n*m):
                    # Find all (b,c) such that (a,b,c) is in triples
                    constraints = []
                    for (ta, tb, tc) in triples:
                        if ta == a:
                            constraints.append((tb, tc))
                    
                    if len(constraints) == 0:
                        U_new[a, :] = 0
                        continue
                    
                    # Build matrix A where A[i, r] = V[b_i, r] * W[c_i, r]
                    A = np.zeros((len(constraints), rank))
                    for idx, (b, c) in enumerate(constraints):
                        A[idx, :] = V[b, :] * W[c, :]
                    
                    # Solve A @ U[a,:] = ones
                    try:
                        U_new[a, :] = np.linalg.lstsq(A, np.ones(len(constraints)), rcond=None)[0]
                    except:
                        U_new[a, :] = 0
                
                U = U_new
                
                # Update V similarly
                V_new = np.zeros_like(V)
                for b in range(m*p):
                    constraints = []
                    for (ta, tb, tc) in triples:
                        if tb == b:
                            constraints.append((ta, tc))
                    
                    if len(constraints) == 0:
                        V_new[b, :] = 0
                        continue
                    
                    A = np.zeros((len(constraints), rank))
                    for idx, (a, c) in enumerate(constraints):
                        A[idx, :] = U[a, :] * W[c, :]
                    
                    try:
                        V_new[b, :] = np.linalg.lstsq(A, np.ones(len(constraints)), rcond=None)[0]
                    except:
                        V_new[b, :] = 0
                
                V = V_new
                
                # Update W similarly
                W_new = np.zeros_like(W)
                for c in range(p*n):
                    constraints = []
                    for (ta, tb, tc) in triples:
                        if tc == c:
                            constraints.append((ta, tb))
                    
                    if len(constraints) == 0:
                        W_new[c, :] = 0
                        continue
                    
                    A = np.zeros((len(constraints), rank))
                    for idx, (a, b) in enumerate(constraints):
                        A[idx, :] = U[a, :] * V[b, :]
                    
                    try:
                        W_new[c, :] = np.linalg.lstsq(A, np.ones(len(constraints)), rcond=None)[0]
                    except:
                        W_new[c, :] = 0
                
                W = W_new
            
            if time.time() > deadline:
                break
            
            # Check if we have an exact solution
            # Check all 24 constraints
            ok = True
            for (a, b, c) in triples:
                val = sum(U[a,r] * V[b,r] * W[c,r] for r in range(rank))
                if abs(val - 1.0) > 1e-8:
                    ok = False
                    break
            
            if ok:
                # Try to round to integers or half-integers
                for denom in [1, 2]:
                    scale = float(denom)
                    U_r = np.round(U * scale) / scale
                    V_r = np.round(V * scale) / scale
                    W_r = np.round(W * scale) / scale
                    
                    ok2 = True
                    for (a, b, c) in triples:
                        val = sum(U_r[a,r] * V_r[b,r] * W_r[c,r] for r in range(rank))
                        if abs(val - 1.0) > 1e-9:
                            ok2 = False
                            break
                    
                    if ok2:
                        # Also need to check that non-triple entries are 0
                        # But the tensor is defined only on triples... 
                        # Actually, the decomposition must satisfy:
                        # sum_r U[a,r]*V[b,r]*W[c,r] = 1 if (a,b,c) in triples, 0 otherwise
                        # Let's verify a few non-triple entries
                        all_ok = True
                        for a in range(n*m):
                            for b in range(m*p):
                                for c in range(p*n):
                                    is_triple = (a, b, c) in triples
                                    val = sum(U_r[a,r] * V_r[b,r] * W_r[c,r] for r in range(rank))
                                    expected = 1.0 if is_triple else 0.0
                                    if abs(val - expected) > 1e-9:
                                        all_ok = False
                                        break
                                if not all_ok:
                                    break
                            if not all_ok:
                                break
                        
                        if all_ok:
                            U_list = [[float(U_r[a, r]) for r in range(rank)] for a in range(n*m)]
                            V_list = [[float(V_r[b, r]) for r in range(rank)] for b in range(m*p)]
                            W_list = [[float(W_r[c, r]) for r in range(rank)] for c in range(p*n)]
                            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook
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
# EVOLVE-BLOCK-END
