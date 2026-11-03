"""Matrix multiplication tensor decomposition for 3x4 x 4x8."""
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = 145.0  # Safety margin before 150s kill

def construct():
    start = time.time()
    
    n, m, p = 3, 4, 8
    
    # Build the target tensor T
    # T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some valid i,j,k
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # First, build the rank-88 hybrid decomposition as baseline
    rank_88 = 88
    U_88 = np.zeros((n*m, rank_88))
    V_88 = np.zeros((m*p, rank_88))
    W_88 = np.zeros((p*n, rank_88))
    
    col = 0
    # Part 1: i=0, schoolbook for 1x4x8 (32 terms)
    for j in range(4):
        for k in range(8):
            U_88[0 * m + j, col] = 1
            V_88[j * p + k, col] = 1
            W_88[k * n + 0, col] = 1
            col += 1
    
    # Part 2: i=1,2 using Strassen-like for 2x4x8
    # Split j into two blocks of 2, k into four blocks of 2
    # For each (j_block, k_block), use Strassen on 2x2x2
    # Strassen matrices for 2x2x2:
    # U rows indexed by (i,j) in 2x2, V rows by (j,k) in 2x2, W rows by (k,i) in 2x2
    
    # Standard Strassen for 2x2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A10)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # U matrix: rows are (i,j) pairs, columns are the 7 Strassen terms
    # For local indices: i in {0,1}, j in {0,1}
    # M1: A00, A11 -> U[0,0]=1, U[1,1]=1
    # M2: A10, A11 -> U[1,0]=1, U[1,1]=1
    # M3: A00 -> U[0,0]=1
    # M4: A11 -> U[1,1]=1
    # M5: A00, A10 -> U[0,0]=1, U[1,0]=1
    # M6: A10, A00 -> U[1,0]=1, U[0,0]=-1
    # M7: A01, A11 -> U[0,1]=1, U[1,1]=-1
    
    strassen_U = np.array([
        [1, 1, 1, 0, 1, -1, 1],   # (0,0)
        [1, 1, 0, 1, 0, 1, 0],    # (0,1)
        [0, 0, 0, 1, 1, 1, -1],   # (1,0)
        [0, 0, 1, 1, 0, 0, 1],    # (1,1)
    ])
    
    strassen_V = np.array([
        [1, 1, 0, -1, 0, 1, 0],   # (0,0)
        [0, 0, -1, 1, 0, 0, 1],   # (0,1)
        [1, 0, 1, 0, 1, 1, 0],    # (1,0)
        [1, 0, 0, -1, 1, 0, 1],   # (1,1)
    ])
    
    strassen_W = np.array([
        [1, 0, -1, 1, 0, 0, 0],   # (0,0)
        [0, 0, 0, 0, 1, 0, 1],    # (0,1)
        [1, 0, 0, 0, 0, -1, 0],   # (1,0)
        [0, 0, 1, -1, 0, 1, 0],   # (1,1)
    ])
    
    # Verify Strassen: C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M4
    # C10 = M2 + M5
    # C11 = M1 - M2 + M3 + M6
    
    # Now apply Strassen for each (j_block, k_block) pair
    for j_block in range(2):
        j_start = j_block * 2
        for k_block in range(4):
            k_start = k_block * 2
            for t in range(7):
                # U contributions: local (i,j) -> global (1+i, j_start+j)
                for li in range(2):
                    for lj in range(2):
                        val = strassen_U[li * 2 + lj, t]
                        if val != 0:
                            gi = 1 + li
                            gj = j_start + lj
                            U_88[gi * m + gj, col] += val
                # V contributions: local (j,k) -> global (j_start+j, k_start+k)
                for lj in range(2):
                    for lk in range(2):
                        val = strassen_V[lj * 2 + lk, t]
                        if val != 0:
                            gj = j_start + lj
                            gk = k_start + lk
                            V_88[gj * p + gk, col] += val
                # W contributions: local (k,i) -> global (k_start+k, 1+i)
                for lk in range(2):
                    for li in range(2):
                        val = strassen_W[lk * 2 + li, t]
                        if val != 0:
                            gk = k_start + lk
                            gi = 1 + li
                            W_88[gk * n + gi, col] += val
                col += 1
    
    # Verify the rank-88 decomposition
    def verify_decomposition(U, V, W, T, tol=1e-10):
        """Verify that sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r] = T"""
        U_arr = np.array(U) if not isinstance(U, np.ndarray) else U
        V_arr = np.array(V) if not isinstance(V, np.ndarray) else V
        W_arr = np.array(W) if not isinstance(W, np.ndarray) else W
        T_arr = np.array(T) if not isinstance(T, np.ndarray) else T
        
        # Compute the tensor from the decomposition
        result = np.zeros_like(T_arr)
        for r in range(U_arr.shape[1]):
            result += np.einsum('a,b,c->abc', U_arr[:, r], V_arr[:, r], W_arr[:, r])
        
        max_err = np.max(np.abs(result - T_arr))
        return max_err < tol, max_err
    
    valid_88, err_88 = verify_decomposition(U_88, V_88, W_88, T)
    
    if not valid_88:
        # Fallback to schoolbook if Strassen construction is wrong
        rank_school = n * m * p
        U_school = np.zeros((n*m, rank_school))
        V_school = np.zeros((m*p, rank_school))
        W_school = np.zeros((p*n, rank_school))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i * m + j, col] = 1
                    V_school[j * p + k, col] = 1
                    W_school[k * n + i, col] = 1
                    col += 1
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    best_U, best_V, best_W = U_88.copy(), V_88.copy(), W_88.copy()
    best_rank = 88
    
    # Try to find a lower rank using numerical optimization
    # Strategy: try rank r < 88 and see if we can find a valid decomposition
    # Use a greedy/iterative approach
    
    # Try a few candidate ranks
    candidate_ranks = [80, 76, 72, 68, 64, 60, 56]
    
    for target_rank in candidate_ranks:
        if time.time() - start > _DEADLINE - 20:
            break
        
        # Try to find a rank-target_rank decomposition
        # Initialize with random values
        np.random.seed(42)
        
        # Try multiple random initializations
        for attempt in range(3):
            if time.time() - start > _DEADLINE - 15:
                break
            
            U_rand = np.random.randn(n*m, target_rank)
            V_rand = np.random.randn(m*p, target_rank)
            W_rand = np.random.randn(p*n, target_rank)
            
            # Use gradient descent to minimize ||T - sum_r U[:,r]⊗V[:,r]⊗W[:,r]||^2
            # This is non-convex, so we use a simple iterative approach
            
            lr = 0.01
            for iteration in range(200):
                if time.time() - start > _DEADLINE - 10:
                    break
                
                # Compute current tensor
                current_T = np.zeros_like(T)
                for r in range(target_rank):
                    current_T += np.einsum('a,b,c->abc', U_rand[:, r], V_rand[:, r], W_rand[:, r])
                
                # Compute gradient
                error = current_T - T
                err_norm = np.linalg.norm(error)
                
                if err_norm < 1e-10:
                    break
                
                # Update U
                for r in range(target_rank):
                    grad_U = np.einsum('abc,b->a', error, V_rand[:, r] * W_rand[:, r])
                    U_rand[:, r] -= lr * grad_U
                    grad_V = np.einsum('abc,a->c', error, U_rand[:, r] * W_rand[:, r])
                    V_rand[:, r] -= lr * grad_V
                    grad_W = np.einsum('abc,a->b', error, U_rand[:, r] * V_rand[:, r])
                    W_rand[:, r] -= lr * grad_W
                
                # Check convergence
                if err_norm < 1e-8:
                    break
            
            # Verify if we found a valid decomposition
            valid, err = verify_decomposition(U_rand, V_rand, W_rand, T, tol=1e-8)
            if valid:
                best_U, best_V, best_W = U_rand, V_rand, W_rand
                best_rank = target_rank
                break
    
    # Try to round to integers/half-integers if close
    def round_to_half(x):
        return np.round(2 * x) / 2.0
    
    U_rounded = round_to_half(best_U)
    V_rounded = round_to_half(best_V)
    W_rounded = round_to_half(best_W)
    
    valid_rounded, err_rounded = verify_decomposition(U_rounded, V_rounded, W_rounded, T, tol=1e-8)
    if valid_rounded:
        best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
    
    # Convert to lists for return
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
