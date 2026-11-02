import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def construct():
    start_time = time.time()
    deadline = 110  # safety margin before 120s limit
    n, m, p = DIMENSIONS  # 3, 4, 8
    
    def verify_decomposition(U, V, W, rank):
        """Verify that the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(s - 1.0) > 1e-8:
                        return False
        return True
    
    def schoolbook_decomposition():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return (U, V, W), rank
    
    # Start with schoolbook as fallback
    best_result, best_rank = schoolbook_decomposition()
    
    # Strategy: Use tensor product structure
    # M_{3,4,8} = M_{3,4,4} ⊗ M_{1,1,2}
    # M_{1,1,2}: rank 2
    # If M_{3,4,4} has rank R, total rank is 2R
    
    # M_{1,1,2} decomposition:
    # A is 1x1, B is 1x2, C is 1x2
    # T[0,0,0]=1, T[0,1,1]=1
    # U_112 = [[1, 1]]  shape (1*1, 2) = (1,2)
    # V_112 = [[1, 0], [0, 1]]  shape (1*2, 2) = (2,2)
    # W_112 = [[1, 0], [0, 1]]  shape (2*1, 2) = (2,2)
    
    U_112 = np.array([[1, 1]], dtype=float)
    V_112 = np.array([[1, 0], [0, 1]], dtype=float)
    W_112 = np.array([[1, 0], [0, 1]], dtype=float)
    
    # Now find best rank for M_{3,4,4}
    # M_{3,4,4}: 3x4 * 4x4 -> 3x4, schoolbook rank = 48
    # Try to find lower rank decompositions
    
    n2, m2, p2 = 3, 4, 4
    
    def verify_344(U, V, W, rank):
        for i in range(n2):
            for j in range(m2):
                for k in range(p2):
                    s = np.sum(U[i*m2+j, :] * V[j*p2+k, :] * W[k*n2+i, :])
                    if abs(s - 1.0) > 1e-8:
                        return False
        return True
    
    def schoolbook_344():
        rank = n2 * m2 * p2
        U = np.zeros((n2*m2, rank))
        V = np.zeros((m2*p2, rank))
        W = np.zeros((p2*n2, rank))
        col = 0
        for i in range(n2):
            for j in range(m2):
                for k in range(p2):
                    U[i*m2+j, col] = 1
                    V[j*p2+k, col] = 1
                    W[k*n2+i, col] = 1
                    col += 1
        return (U, V, W), rank
    
    best_344, best_344_rank = schoolbook_344()
    
    # Try to find better rank for M_{3,4,4}
    for target_rank in range(47, 6, -1):
        if time.time() - start_time > deadline - 20:
            break
        
        found = False
        for attempt in range(30):
            if time.time() - start_time > deadline - 20:
                break
            
            np.random.seed(attempt * 1000 + target_rank + 42)
            U = np.random.randint(-2, 3, size=(n2*m2, target_rank)).astype(float)
            V = np.random.randint(-2, 3, size=(m2*p2, target_rank)).astype(float)
            
            # Compute M[i,k,c] = sum_j U[i*m2+j,c] * V[j*p2+k,c]
            M = np.zeros((n2, p2, target_rank))
            for i in range(n2):
                for k in range(p2):
                    for c in range(target_rank):
                        M[i, k, c] = sum(U[i*m2+j, c] * V[j*p2+k, c] for j in range(m2))
            
            # Solve for W
            W = np.zeros((p2*n2, target_rank))
            valid = True
            for i in range(n2):
                for k in range(p2):
                    mv = M[i, k, :]
                    norm_sq = np.dot(mv, mv)
                    if norm_sq < 1e-10:
                        valid = False
                        break
                    W[k*n2+i, :] = mv / norm_sq
            if not valid:
                continue
            
            # Round to half-integers
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            
            if verify_344(U_r, V_r, W_r, target_rank):
                best_344 = (U_r, V_r, W_r)
                best_344_rank = target_rank
                found = True
                break
        
        if found:
            break
    
    # Combine M_{3,4,4} with M_{1,1,2} via tensor product
    # M_{3,4,8} = M_{3,4,4} ⊗ M_{1,1,2}
    # U_total[i*4+j, c1*2+c2] = U_344[i*4+j, c1] * U_112[0, c2]
    # V_total[j*8+k, c1*2+c2] = V_344[j*4+k1, c1] * V_112[k1, c2] where k = k1*2+k2... 
    # Actually need to be more careful with the indexing.
    
    # For M_{a,b,c} ⊗ M_{d,e,f} = M_{ad, be, cf}:
    # U_total[i1*d+i2, c1*f+c2] = U1[i1, c1] * U2[i2, c2]
    # V_total[j1*e+j2, c1*f+c2] = V1[j1, c1] * V2[j2, c2]  
    # W_total[k1*f+k2, c1*f+c2]... wait, W has shape (c*a, rank)
    # W_total[k1*a1+k2, c1*f+c2] = W1[k1, c1] * W2[k2, c2]
    
    # M_{3,4,4} ⊗ M_{1,1,2} = M_{3*1, 4*1, 4*2} = M_{3,4,8} ✓
    
    U344, V344, W344 = best_344
    rank_total = best_344_rank * 2
    
    U = np.zeros((n*m, rank_total))
    V = np.zeros((m*p, rank_total))
    W = np.zeros((p*n, rank_total))
    
    for i1 in range(3):
        for i2 in range(1):  # M_{1,1,2} has n=1
            for j1 in range(4):
                for j2 in range(1):  # M_{1,1,2} has m=1
                    for k1 in range(4):
                        for k2 in range(2):  # M_{1,1,2} has p=2
                            for c1 in range(best_344_rank):
                                for c2 in range(2):
                                    c = c1 * 2 + c2
                                    # U index: (i1*1+i2, c1*2+c2) -> i = i1, c = c1*2+c2
                                    # But U for M_{3,4,8} has shape (3*4, rank) = (12, rank)
                                    # U[i*m+j, c] where i is row of A (0..2), j is col of A (0..3)
                                    # From tensor product: U_total[i1*1+i2, j1*1+j2, c1*2+c2]
                                    # = U_344[i1, j1, c1] * U_112[i2, c2]
                                    u_val = U344[i1*4+j1, c1] * U_112[i2, c2]
                                    if u_val != 0:
                                        U[(i1*1+i2)*4 + (j1*1+j2), c] = u_val
                                    
                                    # V index: V[j*p+k, c]
                                    # V_total[j1*1+j2, k1*2+k2, c1*2+c2]
                                    # = V_344[j1, k1, c1] * V_112[j2, c2]
                                    v_val = V344[j1*4+k1, c1] * V_112[j2, c2]
                                    if v_val != 0:
                                        V[(j1*1+j2)*8 + (k1*2+k2), c] = v_val
                                    
                                    # W index: W[k*n+i, c]
                                    # W_total[k1*1+k2, i1*1+i2, c1*2+c2]
                                    # = W_344[k1, i1, c1] * W_112[k2, c2]
                                    w_val = W344[k1*3+i1, c1] * W_112[k2, c2]
                                    if w_val != 0:
                                        W[(k1*1+k2)*3 + (i1*1+i2), c] = w_val
    
    # Verify the combined decomposition
    if verify_decomposition(U, V, W, rank_total):
        if rank_total < best_rank:
            best_rank = rank_total
            best_result = (U, V, W)
    
    # Also try direct randomized search for M_{3,4,8}
    for target_rank in [best_rank - 1, best_rank - 2, 40, 36, 32, 28, 24, 20, 16, 14, 12, 10]:
        if time.time() - start_time > deadline:
            break
        if target_rank >= best_rank:
            continue
        
        found = False
        for attempt in range(20):
            if time.time() - start_time > deadline:
                break
            
            np.random.seed(attempt * 7919 + target_rank + 13)
            U = np.random.randint(-2, 3, size=(n*m, target_rank)).astype(float)
            V = np.random.randint(-2, 3, size=(m*p, target_rank)).astype(float)
            
            M = np.zeros((n, p, target_rank))
            for i in range(n):
                for k in range(p):
                    for c in range(target_rank):
                        M[i, k, c] = sum(U[i*m+j, c] * V[j*p+k, c] for j in range(m))
            
            W = np.zeros((p*n, target_rank))
            valid = True
            for i in range(n):
                for k in range(p):
                    mv = M[i, k, :]
                    norm_sq = np.dot(mv, mv)
                    if norm_sq < 1e-10:
                        valid = False
                        break
                    W[k*n+i, :] = mv / norm_sq
            if not valid:
                continue
            
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            
            if verify_decomposition(U_r, V_r, W_r, target_rank):
                best_rank = target_rank
                best_result = (U_r, V_r, W_r)
                found = True
                break
        
        if found:
            break
    
    U, V, W = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
