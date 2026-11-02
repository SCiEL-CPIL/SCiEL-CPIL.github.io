"""Matrix multiplication 4x4x8 - rank-minimal tensor decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    def verify_decomp(U, V, W, n, m, p, tol=1e-9):
        """Verify tensor decomposition: T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1"""
        r = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def strassen_2x2x2():
        """Return rank-7 decomposition for 2x2x2 matrix multiplication."""
        U = np.array([
            [1, 0, 1, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, -1, 1],
            [0, 1, 0, 0, 0, 1, 0],
            [1, 1, 0, 1, 0, 0, -1],
        ], dtype=float)
        V = np.array([
            [1, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 1],
            [1, 0, -1, -1, 1, 0, 1],
        ], dtype=float)
        W = np.array([
            [1, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 1, 0, 0],
        ], dtype=float)
        
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    s = sum(U[i*2+j, c] * V[j*2+k, c] * W[k*2+i, c] for c in range(7))
                    if abs(s - 1) > 1e-9:
                        return None
        return U, V, W

    def kron_decomp(U1, V1, W1, U2, V2, W2, a1, b1, c1, a2, b2, c2):
        """
        Kronecker product of two tensor decompositions.
        T1: a1 x b1 x c1 with rank r1
        T2: a2 x b2 x c2 with rank r2
        Result: (a1*a2) x (b1*b2) x (c1*c2) with rank r1*r2
        """
        r1 = U1.shape[1]
        r2 = U2.shape[1]
        r = r1 * r2
        
        U = np.zeros((a1*a2*b1*b2, r))
        V = np.zeros((b1*b2*c1*c2, r))
        W = np.zeros((c1*c2*a1*a2, r))
        
        for c1_idx in range(r1):
            for c2_idx in range(r2):
                c = c1_idx * r2 + c2_idx
                for i1 in range(a1):
                    for j1 in range(b1):
                        for i2 in range(a2):
                            for j2 in range(b2):
                                U[(i1*a2+i2)*b1*b2 + j1*b2+j2, c] = U1[i1*b1+j1, c1_idx] * U2[i2*b2+j2, c2_idx]
                for j1 in range(b1):
                    for k1 in range(c1):
                        for j2 in range(b2):
                            for k2 in range(c2):
                                V[(j1*b2+j2)*c1*c2 + k1*c2+k2, c] = V1[j1*c1+k1, c1_idx] * V2[j2*c2+k2, c2_idx]
                for k1 in range(c1):
                    for i1 in range(a1):
                        for k2 in range(c2):
                            for i2 in range(a2):
                                W[(k1*c2+k2)*a1*a2 + i1*a2+i2, c] = W1[k1*a1+i1, c1_idx] * W2[k2*a2+i2, c2_idx]
        
        return U, V, W

    def extend_4x4x4_to_4x4x8(U444, V444, W444):
        """
        Extend a 4x4x4 decomposition to 4x4x8.
        For k=0..3 use original, for k=4..7 use shifted version.
        """
        r = U444.shape[1]
        U = np.zeros((16, 2*r))
        V = np.zeros((32, 2*r))
        W = np.zeros((32, 2*r))
        
        for c in range(r):
            U[:16, c] = U444[:, c]
            for j in range(4):
                for k in range(4):
                    V[j*8+k, c] = V444[j*4+k, c]
            for k in range(4):
                for i in range(4):
                    W[k*4+i, c] = W444[k*4+i, c]
        
        for c in range(r):
            U[:16, r+c] = U444[:, c]
            for j in range(4):
                for k in range(4):
                    V[j*8+(k+4), r+c] = V444[j*4+k, c]
            for k in range(4):
                for i in range(4):
                    W[(k+4)*4+i, r+c] = W444[k*4+i, c]
        
        return U, V, W

    def try_rank_r(n, m, p, r, max_iter=1000, seed=42, deadline=145.0):
        """
        Try to find a rank-r decomposition using iterative refinement.
        Uses ALS-style approach with better initialization.
        """
        if time.time() > deadline - 5:
            return None
        
        rng = np.random.RandomState(seed)
        
        # Initialize with random factors, scaled appropriately
        U = rng.randn(n*m, r) * 0.3
        V = rng.randn(m*p, r) * 0.3
        W = rng.randn(p*n, r) * 0.3
        
        # Normalize columns to avoid divergence
        for c in range(r):
            norm = max(np.linalg.norm(U[:, c]), np.linalg.norm(V[:, c]), np.linalg.norm(W[:, c]), 1e-10)
            U[:, c] /= norm
            V[:, c] /= norm
            W[:, c] /= norm
        
        best_err = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        # Precompute index structures for speed
        U_indices = [(i, j) for i in range(n) for j in range(m)]
        V_indices = [(j, k) for j in range(m) for k in range(p)]
        W_indices = [(k, i) for k in range(p) for i in range(n)]
        
        for it in range(max_iter):
            if time.time() > deadline - 10:
                break
            
            # Update U: for each (i,j), solve least squares
            for i, j in U_indices:
                A = np.zeros((p, r))
                for k in range(p):
                    A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                try:
                    U[i*m+j, :] = np.linalg.lstsq(A.T, np.ones(p), rcond=None)[0]
                except:
                    pass
            
            # Update V: for each (j,k), solve least squares
            for j, k in V_indices:
                A = np.zeros((n, r))
                for i in range(n):
                    A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                try:
                    V[j*p+k, :] = np.linalg.lstsq(A.T, np.ones(n), rcond=None)[0]
                except:
                    pass
            
            # Update W: for each (k,i), solve least squares
            for k, i in W_indices:
                A = np.zeros((m, r))
                for j in range(m):
                    A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                try:
                    W[k*n+i, :] = np.linalg.lstsq(A.T, np.ones(m), rcond=None)[0]
                except:
                    pass
            
            # Compute error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                        err += (s - 1.0)**2
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if best_err < 1e-8:
                break
        
        # Verify the best found
        if best_err < 1e-6:
            U_list = [[float(x) for x in best_U[i]] for i in range(n*m)]
            V_list = [[float(x) for x in best_V[i]] for i in range(m*p)]
            W_list = [[float(x) for x in best_W[i]] for i in range(p*n)]
            if verify_decomp(U_list, V_list, W_list, n, m, p, tol=1e-6):
                return U_list, V_list, W_list
        
        return None

    def try_rank_r_with_polish(n, m, p, r, max_iter=500, seed=42, deadline=145.0):
        """
        Try rank-r decomposition with a final high-precision polish pass.
        """
        result = try_rank_r(n, m, p, r, max_iter=max_iter, seed=seed, deadline=deadline)
        if result is not None:
            # Polish: do a few more iterations with tighter tolerance
            U, V, W = result
            U = np.array(U, dtype=float)
            V = np.array(V, dtype=float)
            W = np.array(W, dtype=float)
            
            # Final polish iterations
            for polish_it in range(20):
                if time.time() > deadline - 5:
                    break
                    
                # Recompute with current factors and refine
                for i in range(n):
                    for j in range(m):
                        A = np.zeros((p, r))
                        for k in range(p):
                            A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                        try:
                            U[i*m+j, :] = np.linalg.lstsq(A.T, np.ones(p), rcond=None)[0]
                        except:
                            pass
                
                for j in range(m):
                    for k in range(p):
                        A = np.zeros((n, r))
                        for i in range(n):
                            A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                        try:
                            V[j*p+k, :] = np.linalg.lstsq(A.T, np.ones(n), rcond=None)[0]
                        except:
                            pass
                
                for k in range(p):
                    for i in range(n):
                        A = np.zeros((m, r))
                        for j in range(m):
                            A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                        try:
                            W[k*n+i, :] = np.linalg.lstsq(A.T, np.ones(m), rcond=None)[0]
                        except:
                            pass
            
            U_list = [[float(x) for x in U[i]] for i in range(n*m)]
            V_list = [[float(x) for x in V[i]] for i in range(m*p)]
            W_list = [[float(x) for x in W[i]] for i in range(p*n)]
            
            if verify_decomp(U_list, V_list, W_list, n, m, p, tol=1e-9):
                return U_list, V_list, W_list
        
        return None

    # Start with schoolbook baseline (rank 128)
    best_rank = n * m * p  # 128
    best_decomp = None
    
    # Schoolbook
    U_sb = np.zeros((n*m, best_rank))
    V_sb = np.zeros((m*p, best_rank))
    W_sb = np.zeros((p*n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    best_decomp = (U_sb.tolist(), V_sb.tolist(), W_sb.tolist())
    
    # Try Strassen Kronecker: 2x2x2 ⊗ 2x2x2 = 4x4x4 rank 49, then extend to 4x4x8 rank 98
    s2 = strassen_2x2x2()
    if s2 is not None:
        U444, V444, W444 = kron_decomp(s2[0], s2[1], s2[2], s2[0], s2[1], s2[2], 2, 2, 2, 2, 2, 2)
        
        # Verify 4x4x4
        valid_444 = True
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    s = sum(U444[i*4+j, c] * V444[j*4+k, c] * W444[k*4+i, c] for c in range(49))
                    if abs(s - 1) > 1e-9:
                        valid_444 = False
                        break
                if not valid_444:
                    break
            if not valid_444:
                break
        
        if valid_444:
            U448, V448, W448 = extend_4x4x4_to_4x4x8(U444, V444, W444)
            
            valid_448 = True
            for i in range(4):
                for j in range(4):
                    for k in range(8):
                        s = sum(U448[i*4+j, c] * V448[j*8+k, c] * W448[k*4+i, c] for c in range(98))
                        if abs(s - 1) > 1e-9:
                            valid_448 = False
                            break
                    if not valid_448:
                        break
                if not valid_448:
                    break
            
            if valid_448:
                best_rank = 98
                best_decomp = (U448.tolist(), V448.tolist(), W448.tolist())
    
    # Try to find better decompositions using ALS with various ranks
    # Focus on ranks just below the current best
    candidate_ranks = [97, 96, 95, 94, 93, 92, 91, 90, 88, 85, 80, 75, 70, 65, 60]
    
    for r in candidate_ranks:
        if r >= best_rank:
            continue
        if time.time() > deadline - 20:
            break
        
        found = False
        # Try multiple seeds for better chance of convergence
        for seed in range(3):
            if time.time() > deadline - 15:
                break
            
            # Use polish version for better precision
            result = try_rank_r_with_polish(n, m, p, r, max_iter=400, seed=seed, deadline=deadline)
            if result is not None:
                U_list, V_list, W_list = result
                best_rank = r
                best_decomp = (U_list, V_list, W_list)
                found = True
                break
        
        if found:
            # Found a better rank, continue trying lower
            continue
    
    # If we have time, try some more ranks with different seeds
    if best_rank > 60 and time.time() < deadline - 15:
        for r in [55, 50, 45, 40]:
            if r >= best_rank:
                continue
            if time.time() > deadline - 10:
                break
            result = try_rank_r(n, m, p, r, max_iter=300, seed=123, deadline=deadline)
            if result is not None:
                U_list, V_list, W_list = result
                best_rank = r
                best_decomp = (U_list, V_list, W_list)
    
    U, V, W = best_decomp
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    result = construct()
    r = len(result["U"][0])
    print(f"Rank: {r}")
    
    n, m, p = DIMENSIONS
    U, V, W = result["U"], result["V"], result["W"]
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(r):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1) > 1e-9:
                    valid = False
                    print(f"Failed at ({i},{j},{k}): sum={s}")
                    break
            if not valid:
                break
        if not valid:
            break
    print(f"Valid: {valid}")
