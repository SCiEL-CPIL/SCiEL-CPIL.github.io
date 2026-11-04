"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m  # 6
    dim_v = m * p  # 27
    dim_w = p * n  # 18
    
    start_time = time.time()
    deadline = start_time + 145  # leave 5s margin
    
    # Schoolbook decomposition as fallback
    schoolbook_rank = n * m * p  # 54
    U_sb = np.zeros((dim_u, schoolbook_rank))
    V_sb = np.zeros((dim_v, schoolbook_rank))
    W_sb = np.zeros((dim_w, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = schoolbook_rank
    
    # Precompute triple indices
    nonzero_triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                nonzero_triples.append((i*m+j, j*p+k, k*n+i))
    
    zero_triples = []
    for a in range(dim_u):
        for b in range(dim_v):
            for c in range(dim_w):
                i = a // m
                j = a % m
                j2 = b // p
                k = b % p
                k2 = c // n
                i2 = c % n
                if not (j == j2 and k == k2 and i == i2):
                    zero_triples.append((a, b, c))
    
    # Precompute index mappings
    triples_by_a = {a: [] for a in range(dim_u)}
    triples_by_b = {b: [] for b in range(dim_v)}
    triples_by_c = {c: [] for c in range(dim_w)}
    
    for (a, b, c) in nonzero_triples:
        triples_by_a[a].append((b, c, 1.0))
        triples_by_b[b].append((a, c, 1.0))
        triples_by_c[c].append((a, b, 1.0))
    
    for (a, b, c) in zero_triples:
        triples_by_a[a].append((b, c, 0.0))
        triples_by_b[b].append((a, c, 0.0))
        triples_by_c[c].append((a, b, 0.0))
    
    def verify_decomposition(U, V, W, tol=1e-10):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i*m+j
                    b = j*p+k
                    c = k*n+i
                    val = np.sum(U[a,:] * V[b,:] * W[c,:])
                    if abs(val - 1) > tol:
                        return False
        for a in range(dim_u):
            for b in range(dim_v):
                for c in range(dim_w):
                    i = a // m
                    j = a % m
                    j2 = b // p
                    k = b % p
                    k2 = c // n
                    i2 = c % n
                    if not (j == j2 and k == k2 and i == i2):
                        val = np.sum(U[a,:] * V[b,:] * W[c,:])
                        if abs(val) > tol:
                            return False
        return True
    
    def compute_loss_fast(U, V, W):
        """Compute total squared error efficiently."""
        loss = 0.0
        for (a, b, c) in nonzero_triples:
            val = np.sum(U[a,:] * V[b,:] * W[c,:])
            loss += (val - 1)**2
        for (a, b, c) in zero_triples:
            val = np.sum(U[a,:] * V[b,:] * W[c,:])
            loss += val**2
        return loss
    
    def alternating_minimization_fast(U, V, W, max_iters=500, tol=1e-14):
        """Fast alternating least squares exploiting tensor structure."""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        rank = U.shape[1]
        
        prev_loss = np.inf
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            # Solve for U: for each a, solve linear system
            for a in range(dim_u):
                entries = triples_by_a[a]
                num_eq = len(entries)
                if num_eq == 0:
                    continue
                A_mat = np.zeros((num_eq, rank))
                b_vec = np.zeros(num_eq)
                for idx, (b_t, c_t, target) in enumerate(entries):
                    for d in range(rank):
                        A_mat[idx, d] = V[b_t, d] * W[c_t, d]
                    b_vec[idx] = target
                if num_eq >= rank:
                    try:
                        U[a, :], _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    except:
                        pass
            
            # Solve for V: for each b, solve linear system
            for b in range(dim_v):
                entries = triples_by_b[b]
                num_eq = len(entries)
                if num_eq == 0:
                    continue
                A_mat = np.zeros((num_eq, rank))
                b_vec = np.zeros(num_eq)
                for idx, (a_t, c_t, target) in enumerate(entries):
                    for d in range(rank):
                        A_mat[idx, d] = U[a_t, d] * W[c_t, d]
                    b_vec[idx] = target
                if num_eq >= rank:
                    try:
                        V[b, :], _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    except:
                        pass
            
            # Solve for W: for each c, solve linear system
            for c in range(dim_w):
                entries = triples_by_c[c]
                num_eq = len(entries)
                if num_eq == 0:
                    continue
                A_mat = np.zeros((num_eq, rank))
                b_vec = np.zeros(num_eq)
                for idx, (a_t, b_t, target) in enumerate(entries):
                    for d in range(rank):
                        A_mat[idx, d] = U[a_t, d] * V[b_t, d]
                    b_vec[idx] = target
                if num_eq >= rank:
                    try:
                        W[c, :], _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    except:
                        pass
            
            # Compute loss
            loss = compute_loss_fast(U, V, W)
            
            if abs(prev_loss - loss) < tol:
                break
            prev_loss = loss
        
        return U, V, W, prev_loss
    
    # Try different ranks from low to high
    for rank in range(25, schoolbook_rank + 1):
        if time.time() > deadline - 5:
            break
        
        found = False
        for trial in range(50):
            if time.time() > deadline - 5:
                break
            
            # Initialize with random values
            rng = np.random.RandomState(trial * 1000 + rank)
            U = rng.randn(dim_u, rank) * 0.1
            V = rng.randn(dim_v, rank) * 0.1
            W = rng.randn(dim_w, rank) * 0.1
            
            # Run alternating minimization
            U, V, W, loss = alternating_minimization_fast(U, V, W, max_iters=300, tol=1e-14)
            
            # Check if the decomposition is valid
            if loss < 1e-10:
                # Verify fully
                if verify_decomposition(U, V, W, tol=1e-10):
                    # Round to half-integers if possible
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    
                    if verify_decomposition(U_r, V_r, W_r, tol=1e-10):
                        best_result = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                        best_rank = rank
                        found = True
                    else:
                        # Use the unrounded version
                        best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                        best_rank = rank
                        found = True
                    break
        
        if found:
            # Found a valid decomposition at this rank, try lower ranks
            continue
    
    return best_result
# EVOLVE-BLOCK-END
