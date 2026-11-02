"""Matrix multiplication 4x4 by 4x8 - rank-minimal decomposition via structured search."""
import time
import numpy as np
import random
from itertools import product

DIMENSIONS = [4, 4, 8]
n, m, p = DIMENSIONS
N, M, P = n, m, p

def verify(U, V, W):
    """Check if the decomposition is exact for the all-ones tensor."""
    n, m, p = DIMENSIONS
    r = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > 1e-9:
                    return False, (i,j,k), s
    return True, None, None

def verify_fast(U, V, W):
    """Vectorized verification using numpy."""
    n, m, p = DIMENSIONS
    r = U.shape[1]
    U_arr = np.array(U)  # (nm, r)
    V_arr = np.array(V)  # (mp, r)
    W_arr = np.array(W)  # (pn, r)
    
    # Reshape to 3D
    U3 = U_arr.reshape(n, m, r)
    V3 = V_arr.reshape(m, p, r)
    W3 = W_arr.reshape(p, n, r)
    
    # Compute sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    # Result shape: (n, m, p)
    # For each c: U3[:,:,c] @ V3[:,:,c] gives (n,p), then multiply by W3[:,:,c]
    # Actually: result[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    
    # Use einsum
    result = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    
    if not np.allclose(result, 1.0, atol=1e-9):
        return False
    return True

def make_schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def try_construct_rank(r, deadline):
    """Try to find a valid rank-r decomposition using structured random search."""
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Try different structured initializations
    for attempt in range(200):
        if time.time() > deadline:
            return None
        
        # Strategy: Use random orthogonal-like structures
        # Initialize with random values from a small set
        if attempt % 4 == 0:
            # Binary initialization
            U = np.random.randint(0, 2, size=(nm, r))
            V = np.random.randint(0, 2, size=(mp, r))
            W = np.random.randint(0, 2, size=(pn, r))
        elif attempt % 4 == 1:
            # Small integer values
            U = np.random.choice([-2, -1, 0, 1, 2], size=(nm, r))
            V = np.random.choice([-2, -1, 0, 1, 2], size=(mp, r))
            W = np.random.choice([-2, -1, 0, 1, 2], size=(pn, r))
        elif attempt % 4 == 2:
            # Half-integer values
            U = np.random.choice([-1.5, -1, -0.5, 0, 0.5, 1, 1.5], size=(nm, r))
            V = np.random.choice([-1.5, -1, -0.5, 0, 0.5, 1, 1.5], size=(mp, r))
            W = np.random.choice([-1.5, -1, -0.5, 0, 0.5, 1, 1.5], size=(pn, r))
        else:
            # Sparse random
            U = np.zeros((nm, r))
            V = np.zeros((mp, r))
            W = np.zeros((pn, r))
            for c in range(r):
                # Each column has a few non-zero entries
                nz_u = np.random.choice(nm, size=min(4, nm), replace=False)
                nz_v = np.random.choice(mp, size=min(4, mp), replace=False)
                nz_w = np.random.choice(pn, size=min(4, pn), replace=False)
                U[nz_u, c] = np.random.choice([-1, 1], size=len(nz_u))
                V[nz_v, c] = np.random.choice([-1, 1], size=len(nz_v))
                W[nz_w, c] = np.random.choice([-1, 1], size=len(nz_w))
        
        # Check if valid
        U3 = U.reshape(n, m, r)
        V3 = V.reshape(m, p, r)
        W3 = W.reshape(p, n, r)
        result = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        
        if np.allclose(result, 1.0, atol=1e-9):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def try_greedy_construct(r, deadline):
    """Try a greedy approach to build the decomposition column by column."""
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # We need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
    # Think of it as: we have r "terms" each contributing a rank-1 3-tensor
    # We need the sum to be all-ones
    
    # Start with the residual being all-ones
    residual = np.ones((n, m, p))
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    for c in range(r):
        if time.time() > deadline:
            break
        
        # Find the best rank-1 3-tensor to subtract from residual
        # A rank-1 3-tensor is: a[i,j] * b[j,k] * d[k,i]
        # We want to maximize alignment with residual
        
        # Try random candidates
        best_score = -np.inf
        best_a, best_b, best_d = None, None, None
        
        for trial in range(50):
            # Random a, b, d with small values
            a = np.random.choice([-1, 0, 1], size=(n, m))
            b = np.random.choice([-1, 0, 1], size=(m, p))
            d = np.random.choice([-1, 0, 1], size=(p, n))
            
            # Compute the 3-tensor
            tensor = np.einsum('ij,jk,ki->ijk', a, b, d)
            
            # Score: how well does this reduce the residual?
            # We want tensor to be close to residual in the sense that residual - tensor is smaller
            score = np.sum(tensor * residual) - 0.5 * np.sum(tensor**2)
            
            if score > best_score:
                best_score = score
                best_a, best_b, best_d = a.copy(), b.copy(), d.copy()
        
        if best_a is not None and best_score > 0:
            U_cols.append(best_a.flatten().tolist())
            V_cols.append(best_b.flatten().tolist())
            W_cols.append(best_d.flatten().tolist())
            
            # Update residual
            tensor = np.einsum('ij,jk,ki->ijk', best_a, best_b, best_d)
            residual = residual - tensor
            
            # Check if we're done
            if np.allclose(residual, 0, atol=1e-9):
                break
    
    if len(U_cols) == 0:
        return None
    
    # Build final matrices
    r_actual = len(U_cols)
    U = np.zeros((nm, r_actual))
    V = np.zeros((mp, r_actual))
    W = np.zeros((pn, r_actual))
    
    for c in range(r_actual):
        U[:, c] = U_cols[c]
        V[:, c] = V_cols[c]
        W[:, c] = W_cols[c]
    
    # Verify
    U3 = U.reshape(n, m, r_actual)
    V3 = V.reshape(m, p, r_actual)
    W3 = W.reshape(p, n, r_actual)
    result = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    
    if np.allclose(result, 1.0, atol=1e-9):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin
    
    best = make_schoolbook()
    best_rank = n * m * p  # 128
    
    # Try structured constructions for lower ranks
    # The theoretical lower bound is max(n*p, m*p, n*m) = 32
    # Try ranks from 32 upward
    
    for r in range(32, best_rank):
        if time.time() > deadline:
            break
        
        # Try random structured search
        result = try_construct_rank(r, deadline)
        if result is not None:
            best = result
            best_rank = r
            break
        
        # Try greedy approach
        result = try_greedy_construct(r, deadline)
        if result is not None:
            best = result
            best_rank = r
            break
    
    return best

if __name__ == "__main__":
    result = construct()
    r = len(result["U"][0])
    print(f"Rank: {r}")
    valid, err, val = verify(result["U"], result["V"], result["W"])
    print(f"Valid: {valid}")
