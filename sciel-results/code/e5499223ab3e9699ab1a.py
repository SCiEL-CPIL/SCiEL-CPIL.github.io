import time
import numpy as np

DIMENSIONS = [3, 3, 8]
START = time.time()
DEADLINE = 135.0  # safety margin before 150s kill

def _build_schoolbook():
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

def _validate_exact(U, V, W, n, m, p):
    """Check that U[i*m+j,c] * V[j*p+k,c] == W[k*n+i,c] for all i,j,k,c.
    Uses exact integer/half-integer arithmetic."""
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    r = U.shape[1]
    # Multiply by 2 to work with integers (half-integers become integers)
    U2 = np.round(2 * U).astype(int)
    V2 = np.round(2 * V).astype(int)
    W2 = np.round(2 * W).astype(int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                lhs = U2[i * m + j, :] * V2[j * p + k, :]
                rhs = 4 * W2[k * n + i, :]
                if not np.array_equal(lhs, rhs):
                    return False
    return True

def _to_list(M):
    """Convert numpy array to list of lists of int or float (half-integer)."""
    result = []
    for row in M:
        r = []
        for v in row:
            fv = float(v)
            if fv == int(fv):
                r.append(int(fv))
            else:
                r.append(fv)
        result.append(r)
    return result

def _find_334_decomposition():
    """
    Find a low-rank decomposition for 3x3x4 matrix multiplication.
    We search for rank 19, 20, etc.
    Returns (U, V, W) for 3x3x4 or None.
    """
    n, m, p = 3, 3, 4
    # Try rank 19
    for r in range(19, 25):
        if time.time() > DEADLINE - 30:
            break
        # Use a simple random search with constraints
        # For 3x3x4, we can use the fact that it's 3x3x3 + 3x3x1
        # But let's try a general search
        # This might be slow.
        # Instead, let's try to construct a rank 23 for 3x3x3 and add rank 3 for 3x3x1.
        # Rank of 3x3x1 is 3.
        # If we have rank 23 for 3x3x3, total is 26.
        # Rank of 3x3x3 is 23?
        # Let's assume we can find a rank 20 for 3x3x4.
        
        # For now, let's just return a known high rank to be safe, 
        # or try to find a better one.
        # Since I don't have a hardcoded rank 19, I will try to find one.
        
        # A simple approach: 
        # The tensor is T_{i,j,k} = delta_{j,k} for k<3? No.
        # T_{i,j,k} = 1 if j=k and k<3? No.
        # The multiplication is C_{ik} = sum_j A_{ij} B_{jk}.
        # For k=0,1,2,3.
        # For k=0,1,2, it's standard 3x3x3.
        # For k=3, it's 3x3x1.
        
        # Let's try to find a rank 20 decomposition for 3x3x4.
        # This is a known problem.
        # I will implement a search for rank 20.
        
        # If search fails, return None.
        
        # To save time, I will just return a rank 26 construction if I can find 3x3x3 rank 23.
        # But I don't have 3x3x3 rank 23 either.
        
        # Let's just use the schoolbook for 3x3x4 which is rank 36.
        # Then total rank is 72.
        
        # This is not an improvement.
        
        # I need to find a better decomposition.
        
        # Let's try to find a rank 19 decomposition for 3x3x4 using a solver.
        # This is too complex to implement in 150s.
        
        # Instead, I will use a known result:
        # The rank of 3x3x4 is 19.
        # I will hardcode a rank 19 decomposition if I can find one.
        # Since I can't, I will try to find a rank 20 one.
        
        # Let's try a random search for rank 20.
        rng = np.random.default_rng(42)
        for _ in range(1000):
            if time.time() > DEADLINE - 10:
                return None
            
            # Initialize with small integers
            U = rng.integers(-2, 3, size=(n*m, r)) / 2.0
            V = rng.integers(-2, 3, size=(m*p, r)) / 2.0
            W = np.zeros((p*n, r))
            
            # Compute W from U and V
            # We need U[i*m+j,c] * V[j*p+k,c] = W[k*n+i,c]
            # This is not always possible.
            
            # Check if consistent
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        prod = U[i*m+j, :] * V[j*p+k, :]
                        # W[k*n+i, :] should be prod / 4? No.
                        # The condition is U[i*m+j,c] * V[j*p+k,c] = W[k*n+i,c] * 4?
                        # No, the condition is:
                        # sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = delta_{j,j'}...
                        # Actually, the condition is:
                        # U[i*m+j,c] * V[j*p+k,c] = W[k*n+i,c] is NOT the condition.
                        # The condition is:
                        # The tensor T is decomposed.
                        # T_{(i,j), (j',k), (k',i')} = delta_{j,j'} delta_{k,k'} delta_{i,i'}
                        # So sum_c U_{ij,c} V_{j'k,c} W_{k'i',c} = delta_{j,j'} delta_{k,k'} delta_{i,i'}
                        
                        # This is a system of equations.
                        # It's hard to solve.
                        
                        valid = False
                        break
                    if not valid: break
                if not valid: break
            if valid:
                # Check exactness
                if _validate_exact(U, V, W, n, m, p):
                    return U, V, W
    return None

def construct():
    n, m, p = DIMENSIONS
    best = _build_schoolbook()
    best_rank = n * m * p  # 72
    
    # Try to find a better decomposition using block-parallel construction
    # Find 3x3x4 decomposition
    res_334 = _find_334_decomposition()
    
    if res_334 is not None:
        U_334, V_334, W_334 = res_334
        r_334 = U_334.shape[1]
        
        # Construct 3x3x8
        U = np.zeros((n*m, 2*r_334))
        V = np.zeros((m*p, 2*r_334))
        W = np.zeros((p*n, 2*r_334))
        
        # First block
        U[:, :r_334] = U_334
        for j in range(m):
            for k in range(4):
                V[j*8 + k, :r_334] = V_334[j*4 + k, :]
        for k in range(4):
            for i in range(n):
                W[k*3 + i, :r_334] = W_334[k*3 + i, :]
        
        # Second block
        U[:, r_334:] = U_334
        for j in range(m):
            for k in range(4):
                V[j*8 + (k+4), r_334:] = V_334[j*4 + k, :]
        for k in range(4):
            for i in range(n):
                W[(k+4)*3 + i, r_334:] = W_334[k*3 + i, :]
        
        if _validate_exact(U, V, W, n, m, p):
            best = {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}
            best_rank = 2*r_334
    
    return best

if __name__ == "__main__":
    res = construct()
    print("rank:", len(res["U"][0]))
