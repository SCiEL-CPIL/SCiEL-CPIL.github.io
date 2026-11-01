"""Matrix multiplication 2,2,11 - rank-minimal decomposition via backtracking search over sparse patterns."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 2, 11]

def _build_target_tensor():
    """Build the multiplication tensor T of shape (n*m, m*p, p*n)."""
    n, m, p = 2, 2, 11
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _verify(U, V, W, T=None):
    """Verify that the decomposition exactly reproduces the multiplication tensor."""
    if T is None:
        T = _build_target_tensor()
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(T, T_decomp, atol=1e-9)

def _build_schoolbook():
    """Build the schoolbook rank-44 decomposition."""
    n, m, p = 2, 2, 11
    rank = n * m * p  # 44
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return U, V, W

def _build_strassen_correct():
    """
    Build a verified Strassen-based decomposition for 2x2 * 2x11.
    Uses 5 Strassen blocks (rank 7 each) for column pairs + 1 schoolbook column (rank 2).
    Total rank: 37.
    """
    n, m, p = 2, 2, 11
    num_pairs = p // 2  # 5
    remainder = p % 2   # 1
    
    rank = num_pairs * 7 + remainder * 2
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    # Correct Strassen formulas for C = A*B:
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[ae+bg, af+bh],[ce+dg, cf+dh]]
    # p1 = (a+d)(e+h), p2 = (c+d)e, p3 = a(g-h), p4 = d(f-e)
    # p5 = (a+b)h, p6 = (c-a)(e+f), p7 = (b-d)(g+h)
    # C11 = p1 + p4 - p5 + p7, C12 = p3 + p5
    # C21 = p2 + p4, C22 = p1 - p2 + p3 + p6
    
    # U: coefficients for A = [a, b, c, d] (indices 0,1,2,3)
    # V: coefficients for B = [e, f, g, h] (indices 0,1,2,3)
    # W: coefficients for C = [C11, C12, C21, C22] (indices 0,1,2,3)
    
    strassen_U = [
        [1, 0, 0, 1],   # p1: a+d
        [0, 0, 1, 1],   # p2: c+d
        [1, 0, 0, 0],   # p3: a
        [0, 0, 0, 1],   # p4: d
        [1, 1, 0, 0],   # p5: a+b
        [1, 0, -1, 0],  # p6: c-a
        [0, 1, 0, -1],  # p7: b-d
    ]
    
    strassen_V = [
        [1, 0, 0, 1],   # p1: e+h
        [1, 0, 0, 0],   # p2: e
        [0, 0, 1, -1],  # p3: g-h
        [0, 1, 0, -1],  # p4: f-e
        [0, 0, 0, 1],   # p5: h
        [1, 1, 0, 0],   # p6: e+f
        [0, 0, 1, 1],   # p7: g+h
    ]
    
    strassen_W = [
        [1, 0, 0, 1, -1, 0, 1],   # C11
        [0, 0, 1, 0, 1, 0, 0],    # C12
        [0, 1, 0, 1, 0, 0, 0],    # C21
        [1, -1, 1, 0, 0, 1, 0],   # C22
    ]
    
    col = 0
    for b in range(num_pairs):
        for t in range(7):
            # U: same for all blocks
            U[:, col] = strassen_U[t]
            
            # V: block b uses B columns 2b and 2b+1
            # B indices: j*p + k where j in {0,1}, k in {2b, 2b+1}
            for j in range(2):
                for kp in range(2):
                    local_idx = j * 2 + kp  # 0=e,1=f,2=g,3=h
                    global_idx = j * p + (2 * b + kp)
                    V[global_idx, col] = strassen_V[t][local_idx]
            
            # W: block b uses C columns 2b and 2b+1
            # C indices: k*n + i where k in {2b, 2b+1}, i in {0,1}
            for i in range(2):
                for kp in range(2):
                    local_idx = i * 2 + kp  # 0=C11,1=C12,2=C21,3=C22
                    global_idx = (2 * b + kp) * n + i
                    W[global_idx, col] = strassen_W[local_idx][t]
            
            col += 1
    
    # Handle remainder column (column 10) with schoolbook
    if remainder == 1:
        # C[i,10] = A[i,0]*B[0,10] + A[i,1]*B[1,10]
        # Term 1: A[i,0]*B[0,10]
        U[0, col] = 1  # A[0,0]
        U[2, col] = 1  # A[1,0]
        V[0 * p + 10, col] = 1  # B[0,10]
        W[10 * n + 0, col] = 1  # C[0,10]
        W[10 * n + 1, col] = 1  # C[1,10]
        col += 1
        
        # Term 2: A[i,1]*B[1,10]
        U[1, col] = 1  # A[0,1]
        U[3, col] = 1  # A[1,1]
        V[1 * p + 10, col] = 1  # B[1,10]
        W[10 * n + 0, col] = 1  # C[0,10]
        W[10 * n + 1, col] = 1  # C[1,10]
        col += 1
    
    return U, V, W

def _backtracking_search(T, max_rank=30, deadline=140):
    """
    Backtracking search for a rank-minimal decomposition.
    We model each rank-1 term as having sparse structure.
    
    For the 2x2 * 2x11 case, each term r has:
    - U[:,r] in R^4 (coefficients for A[0,0], A[0,1], A[1,0], A[1,1])
    - V[:,r] in R^22 (coefficients for B[j,k])
    - W[:,r] in R^22 (coefficients for C[k,i])
    
    The constraint is: for all (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    
    We search over small integer/half-integer values.
    """
    start_time = time.time()
    n, m, p = 2, 2, 11
    I, J, K = T.shape  # 4, 22, 22
    
    # Candidate values for each entry
    vals = [-1, -0.5, 0, 0.5, 1]
    
    # We'll build terms one at a time
    # For efficiency, we track the residual tensor
    residual = T.copy()
    
    # Precompute which (a,b,c) indices are non-zero in T
    # T[a,b,c] = 1 iff a=i*m+j, b=j*p+k, c=k*n+i for some valid i,j,k
    # i in {0,1}, j in {0,1}, k in {0,...,10}
    # So the non-zero entries are:
    nz_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                nz_indices.append((a, b, c))
    
    best_result = None
    best_rank = max_rank + 1
    
    # Try ranks from low to high
    for target_rank in range(20, max_rank + 1):
        if time.time() - start_time > deadline:
            break
        
        # Use a greedy + backtracking hybrid
        # Start with a partial solution and try to complete it
        U_partial = []
        V_partial = []
        W_partial = []
        
        def _try_complete(residual, U_list, V_list, W_list, depth, max_depth):
            nonlocal best_result, best_rank, start_time
            
            if time.time() - start_time > deadline:
                return False
            
            if depth == max_depth:
                # Check if residual is zero
                if np.allclose(residual, 0, atol=1e-9):
                    U_arr = np.array(U_list).T
                    V_arr = np.array(V_list).T
                    W_arr = np.array(W_list).T
                    best_result = (U_arr, V_arr, W_arr, max_depth)
                    best_rank = max_depth
                    return True
                return False
            
            # If residual has very small norm, check if we can finish
            res_norm = np.linalg.norm(residual)
            if res_norm < 1e-9:
                # Pad with zeros
                while len(U_list) < max_depth:
                    U_list.append(np.zeros(I))
                    V_list.append(np.zeros(J))
                    W_list.append(np.zeros(K))
                U_arr = np.array(U_list).T
                V_arr = np.array(V_list).T
                W_arr = np.array(W_list).T
                best_result = (U_arr, V_arr, W_arr, max_depth)
                best_rank = max_depth
                return True
            
            # Greedy: find the best rank-1 approximation to residual
            # For each candidate (u, v, w), compute the residual after subtracting u*v*w
            # To keep it tractable, we limit the search space
            
            # Strategy: try each combination of sparse patterns
            # U has 4 entries, V has 22, W has 22
            # We'll try patterns where U is sparse (1-2 non-zero entries)
            
            # Enumerate sparse U patterns
            u_patterns = []
            for nz_count in range(1, 3):
                for indices in product(range(I), repeat=nz_count):
                    if len(set(indices)) != nz_count:
                        continue
                    for vals_combo in product(vals, repeat=nz_count):
                        u = np.zeros(I)
                        for idx, val in zip(indices, vals_combo):
                            u[idx] = val
                        if np.any(u != 0):
                            u_patterns.append(u)
            
            # For each U pattern, try to find V and W
            found = False
            for u in u_patterns:
                if time.time() - start_time > deadline:
                    return False
                
                # The residual equation: sum_r u_r * v_r * w_r = residual
                # For a single term: u * outer(v, w) should approximate residual
                # residual[a,b,c] ≈ u[a] * v[b] * w[c]
                
                # For each a where u[a] != 0:
                # residual[a,b,c] / u[a] should be v[b]*w[c]
                
                # Find the best (v, w) such that for all a with u[a]!=0:
                # v[b]*w[c] ≈ residual[a,b,c]/u[a]
                
                # This is a rank-1 approximation problem
                # Use SVD on the matrix M[a] = residual[a,:,:] / u[a]
                
                # Combine all a's: we want v*w to match residual[a]/u[a] for all a
                # Weighted: minimize sum_a ||M_a - v*w||^2 where M_a[b,c] = residual[a,b,c]/u[a]
                
                # Simple approach: average the matrices
                M_avg = np.zeros((J, K))
                count = 0
                for a in range(I):
                    if abs(u[a]) > 1e-9:
                        M_avg += residual[a, :, :] / u[a]
                        count += 1
                
                if count == 0:
                    continue
                
                M_avg /= count
                
                # SVD for rank-1 approximation
                try:
                    U_svd, s, Vt_svd = np.linalg.svd(M_avg, full_matrices=False)
                    if s[0] > 1e-12:
                        v = U_svd[:, 0] * np.sqrt(s[0])
                        w = Vt_svd[0, :] * np.sqrt(s[0])
                    else:
                        continue
                except:
                    continue
                
                # Round v and w to nearest half-integers
                v_rounded = np.round(v * 2) / 2
                w_rounded = np.round(w * 2) / 2
                
                # Check if this term helps
                term = np.einsum('a,b,c->abc', u, v_rounded, w_rounded)
                new_residual = residual - term
                
                # Only accept if it reduces the norm
                if np.linalg.norm(new_residual) < res_norm * 0.99:
                    U_list.append(u.copy())
                    V_list.append(v_rounded.copy())
                    W_list.append(w_rounded.copy())
                    
                    if _try_complete(new_residual, U_list, V_list, W_list, depth + 1, max_depth):
                        return True
                    
                    U_list.pop()
                    V_list.pop()
                    W_list.pop()
            
            return found
        
        # Try to find a decomposition of the given rank
        if _try_complete(T.copy(), [], [], [], 0, target_rank):
            if best_result is not None:
                U_arr, V_arr, W_arr, r = best_result
                if _verify(U_arr.tolist(), V_arr.tolist(), W_arr.tolist(), T):
                    return U_arr, V_arr, W_arr
    
    return None

def construct():
    start_time = time.time()
    DEADLINE = 140  # seconds, leave margin before 150s limit
    
    T = _build_target_tensor()
    
    # First, try the corrected Strassen-based rank-37 construction
    U, V, W = _build_strassen_correct()
    
    if _verify(U.tolist(), V.tolist(), W.tolist(), T):
        # Now try to reduce rank further using backtracking search
        if time.time() - start_time < DEADLINE - 20:
            result = _backtracking_search(T, max_rank=36, deadline=DEADLINE)
            if result is not None:
                U_bt, V_bt, W_bt = result
                if _verify(U_bt.tolist(), V_bt.tolist(), W_bt.tolist(), T):
                    return {'U': U_bt.tolist(), 'V': V_bt.tolist(), 'W': W_bt.tolist()}
        
        # Return rank 37 if no improvement found
        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    
    # Fall back to schoolbook rank 44
    U, V, W = _build_schoolbook()
    if _verify(U.tolist(), V.tolist(), W.tolist(), T):
        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    
    # Should not reach here
    raise RuntimeError("No valid decomposition found")
