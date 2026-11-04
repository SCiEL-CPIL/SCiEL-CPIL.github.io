import time
import numpy as np

START = time.time()
DEADLINE = START + 130.0

def _strassen_2x2x2():
    """Strassen 2x2x2 decomposition, rank 7."""
    U = np.zeros((4, 7), dtype=np.int16)
    V = np.zeros((4, 7), dtype=np.int16)
    W = np.zeros((4, 7), dtype=np.int16)
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
    U[2,1]=1; U[3,1]=1; V[0,1]=1; W[1,1]=1; W[3,1]=-1
    U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[2,2]=1; W[3,2]=1
    U[3,3]=1; V[0,3]=-1; V[2,3]=1; W[0,3]=1; W[1,3]=1
    U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[2,4]=1
    U[0,5]=-1; U[2,5]=1; V[0,5]=1; V[1,5]=1; W[3,5]=1
    U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1
    return U, V, W

def _build_4x4x4_tensors():
    """Build 4x4x4 via Kronecker product of two 2x2x2 Strassen."""
    U2, V2, W2 = _strassen_2x2x2()
    rank = 49
    U4 = np.zeros((16, rank), dtype=np.int16)
    V4 = np.zeros((16, rank), dtype=np.int16)
    W4 = np.zeros((16, rank), dtype=np.int16)
    for c0 in range(7):
        for c1 in range(7):
            c = c0*7 + c1
            for i in range(4):
                for j in range(4):
                    U4[i*4+j, c] = U2[(i//2)*2+(j//2), c0] * U2[(i%2)*2+(j%2), c1]
            for j in range(4):
                for k in range(4):
                    V4[j*4+k, c] = V2[(j//2)*2+(k//2), c0] * V2[(j%2)*2+(k%2), c1]
            for k in range(4):
                for i in range(4):
                    W4[k*4+i, c] = W2[(k//2)*2+(i//2), c0] * W2[(k%2)*2+(i//2), c1]
    return U4, V4, W4

def _build_rank98():
    """Standard rank-98 decomposition: two independent 4x4x4 blocks."""
    U4, V4, W4 = _build_4x4x4_tensors()
    total_rank = 98
    U = np.zeros((16, total_rank), dtype=np.int16)
    V = np.zeros((32, total_rank), dtype=np.int16)
    W = np.zeros((32, total_rank), dtype=np.int16)
    
    for block in range(2):
        col_start = block * 49
        col_end = (block + 1) * 49
        U[:, col_start:col_end] = U4
        for j in range(4):
            for k_local in range(4):
                k_global = block * 4 + k_local
                V[j*8 + k_global, col_start:col_end] = V4[j*4 + k_local, :]
        for k_local in range(4):
            k_global = block * 4 + k_local
            for i in range(4):
                W[k_global*4 + i, col_start:col_end] = W4[k_local*4 + i, :]
    
    # Verify
    n, m, p = 4, 4, 8
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for r in range(total_rank):
                    total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(total - 1.0) > 1e-9:
                    return None
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _verify(U, V, W, n, m, p):
    """Verify decomposition."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U_arr[i*m+j, t] * V_arr[j*p+k, t] * W_arr[k*n+i, t] for t in range(rank))
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _try_reduce_rank(U0, V0, W0, target_rank, n, m, p, max_iter=200):
    """Try to reduce rank from len(U0[0]) to target_rank using iterative refinement."""
    r = target_rank
    U = np.zeros((n*m, r), dtype=float)
    V = np.zeros((m*p, r), dtype=float)
    W = np.zeros((p*n, r), dtype=float)
    
    # Initialize: take first r columns of the rank-98 decomposition
    U[:, :] = np.array(U0, dtype=float)[:, :r]
    V[:, :] = np.array(V0, dtype=float)[:, :r]
    W[:, :] = np.array(W0, dtype=float)[:, :r]
    
    for iteration in range(max_iter):
        if time.time() > DEADLINE - 10:
            break
        
        # Update U
        for i in range(n):
            for j in range(m):
                A = np.array([V[j*p+k, :] * W[k*n+i, :] for k in range(p)])
                b = np.ones(p)
                try:
                    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    U[i*m+j, :] = x
                except:
                    pass
        
        # Update V
        for j in range(m):
            for k in range(p):
                A = np.array([U[i*m+j, :] * W[k*n+i, :] for i in range(n)])
                b = np.ones(n)
                try:
                    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    V[j*p+k, :] = x
                except:
                    pass
        
        # Update W
        for k in range(p):
            for i in range(n):
                A = np.array([U[i*m+j, :] * V[j*p+k, :] for j in range(m)])
                b = np.ones(m)
                try:
                    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    W[k*n+i, :] = x
                except:
                    pass
        
        # Check convergence
        if iteration % 20 == 0:
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = sum(U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] for t in range(r))
                        max_err = max(max_err, abs(val - 1.0))
            if max_err < 1e-8:
                # Convert to list format
                return U.tolist(), V.tolist(), W.tolist()
    
    return None

def _schoolbook(n=4, m=4, p=8):
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

def _try_hybrid_structure():
    """
    Try a hybrid approach: use Strassen for the 4x4x4 part but with a different 
    decomposition that might allow rank reduction.
    
    Key insight: The 4x4x8 tensor can be seen as 4x4x4 ⊗ 4x4x2 or similar.
    Let's try to find a decomposition where the two 4x4x4 blocks share some structure.
    
    Alternative: Use a mixed approach where we treat the 8 columns as two groups of 4,
    but allow cross-terms between them.
    """
    n, m, p = 4, 4, 8
    U4, V4, W4 = _build_4x4x4_tensors()
    
    # Start with the standard rank-98 and try to find linear dependencies
    # that allow rank reduction
    
    # Build the full tensor T[i,j,k] = 1 for all i,j,k
    # and check if we can express it with fewer terms
    
    # Try a different approach: use the fact that the problem is 4x4x8
    # which can be decomposed as (4x4x4) + (4x4x4) but maybe with shared factors
    
    # Let's try to find if there's a rank-90 or so decomposition by 
    # exploiting the specific structure
    
    # For now, return the standard rank-98 if no better found
    return None

def construct():
    n, m, p = 4, 4, 8
    
    # Get the rank-98 decomposition
    rank98_result = None
    if time.time() < DEADLINE - 30:
        try:
            rank98_result = _build_rank98()
        except Exception:
            pass
    
    if rank98_result is None:
        return _schoolbook()
    
    best_result = rank98_result
    best_rank = 98
    
    # Try to reduce rank
    U98 = rank98_result['U']
    V98 = rank98_result['V']
    W98 = rank98_result['W']
    
    # Try ranks from 97 down to 85
    for target_rank in range(97, 84, -1):
        if time.time() > DEADLINE - 15:
            break
        
        result = _try_reduce_rank(U98, V98, W98, target_rank, n, m, p, max_iter=100)
        if result is not None:
            U_new, V_new, W_new = result
            if _verify(U_new, V_new, W_new, n, m, p):
                best_result = {"U": U_new, "V": V_new, "W": W_new}
                best_rank = target_rank
                break  # Found lower rank, stop
    
    return best_result

if __name__ == "__main__":
    res = construct()
    rank = len(res['U'][0])
    print(f"Rank: {rank}")
    n, m, p = 4, 4, 8
    U, V, W = res['U'], res['V'], res['W']
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    print(f"Valid: {valid}")
