import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = a.shape[0]
    deadline = time.time() + 115.0  # Safety margin before 120s limit
    
    def compute_cost(p):
        return np.sum(a * b[np.ix_(p, p)])
    
    # --- Initialization: Spectral Embedding + Rank Matching ---
    # Center the matrices
    a_centered = a - a.mean()
    b_centered = b - b.mean()
    
    # Get eigenvectors
    try:
        eigvals_a, eigvecs_a = np.linalg.eigh(a_centered)
        eigvals_b, eigvecs_b = np.linalg.eigh(b_centered)
    except np.linalg.LinAlgError:
        eigvals_a, eigvecs_a = np.linalg.eig(a_centered)
        eigvals_b, eigvecs_b = np.linalg.eig(b_centered)
        idx_a = np.argsort(-np.abs(eigvals_a))
        idx_b = np.argsort(-np.abs(eigvals_b))
        eigvecs_a = eigvecs_a[:, idx_a]
        eigvecs_b = eigvecs_b[:, idx_b]
    else:
        idx_a = np.argsort(-np.abs(eigvals_a))
        idx_b = np.argsort(-np.abs(eigvals_b))
        eigvecs_a = eigvecs_a[:, idx_a]
        eigvecs_b = eigvecs_b[:, idx_b]
        
    # Use spectral info combined with rank matching
    a_sums = a.sum(axis=1)
    b_sums = b.sum(axis=1)
    
    # Sort facilities by flow strength (descending)
    heavy_facilities = np.argsort(-a_sums, kind='stable')
    # Sort locations by distance centrality (ascending)
    central_locations = np.argsort(b_sums, kind='stable')
    
    p = np.empty(n, dtype=np.int64)
    p[heavy_facilities] = central_locations
    
    current_cost = compute_cost(p)
    best_cost = current_cost
    best_p = p.copy()
    
    # Precompute for fast delta calculation
    # We'll maintain the permutation p and use vectorized operations for delta
    
    def get_delta(i, j, p, a, b):
        x = p[i]
        y = p[j]
        
        # Term 1: (A_ii - A_jj)(B_yy - B_xx)
        term1 = (a[i, i] - a[j, j]) * (b[y, y] - b[x, x])
        
        # Term 2: (A_ij - A_ji)(B_yx - B_xy)
        term2 = (a[i, j] - a[j, i]) * (b[y, x] - b[x, y])
        
        # Term 3: Sum_{k != i, j} (A_ik - A_jk)(B_y, p[k] - B_x, p[k])
        diff_a = a[i, :] - a[j, :]
        diff_b = b[y, p] - b[x, p]
        
        total_dot = np.dot(diff_a, diff_b)
        
        # Subtract k=i term: (a[i, i] - a[j, i]) * (b[y, x] - b[x, x])
        term_ik = (a[i, i] - a[j, i]) * (b[y, x] - b[x, x])
        
        # Subtract k=j term: (a[i, j] - a[j, j]) * (b[y, y] - b[x, y])
        term_jk = (a[i, j] - a[j, j]) * (b[y, y] - b[x, y])
        
        term3 = total_dot - term_ik - term_jk
        
        return term1 + term2 + term3
    
    # Linearized assignment step using scipy's Hungarian algorithm
    def linearized_assignment(p, a, b, n):
        # Build the linearized cost matrix
        # C[i, j] = sum_k a[i, k] * b[j, p[k]]
        # This is a matrix product: a @ b[:, p]
        # But we need C[i, j] for facility i to location j
        # C[i, j] = sum_k a[i, k] * b[j, p[k]]
        # Let B_p = b[:, p] (columns of b permuted by p)
        # Then C = a @ B_p.T? No.
        # C[i, j] = sum_k a[i, k] * b[j, p[k]]
        # Let's define M where M[k, l] = a[i, k] * b[l, p[k]]? No, that's not right.
        # Actually, C[i, j] = sum_k a[i, k] * b[j, p[k]]
        # This can be written as: C = a @ (b[:, p]).T? Let's check.
        # (a @ B_p.T)[i, j] = sum_k a[i, k] * B_p.T[k, j] = sum_k a[i, k] * B_p[j, k] = sum_k a[i, k] * b[j, p[k]]
        # Yes! So C = a @ b[:, p].T
        
        B_p = b[:, p]  # shape (n, n), column k is b[:, p[k]]
        C = a @ B_p.T  # shape (n, n), C[i, j] = sum_k a[i, k] * b[j, p[k]]
        
        # Solve the linear assignment problem
        row_ind, col_ind = linear_sum_assignment(C)
        return col_ind
    
    # Main loop: alternate between linearized assignment and 2-opt local search
    while time.time() < deadline:
        # Step 1: Linearized assignment
        if time.time() < deadline - 5:
            p_new = linearized_assignment(p, a, b, n)
            cost_new = compute_cost(p_new)
            
            if cost_new < current_cost:
                p = p_new
                current_cost = cost_new
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
        
        # Step 2: 2-opt local search
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            for j in range(i + 1, n):
                if time.time() > deadline:
                    break
                
                delta = get_delta(i, j, p, a, b)
                
                if delta < 0:
                    p[i], p[j] = p[j], p[i]
                    current_cost += delta
                    improved = True
                    
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p.copy()
                    
                    break
        
        if not improved:
            # Local Minimum Reached. Perturb.
            if time.time() < deadline - 10:
                # Random Reversal Perturbation
                len_seg = np.random.randint(3, 15)
                if len_seg < n:
                    start = np.random.randint(0, n - len_seg)
                    end = start + len_seg
                    p[start:end+1] = p[start:end+1][::-1]
                    current_cost = compute_cost(p)
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p.copy()
                else:
                    for _ in range(10):
                        i = np.random.randint(n)
                        j = np.random.randint(n)
                        if i != j:
                            p[i], p[j] = p[j], p[i]
                    current_cost = compute_cost(p)
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p.copy()

    return best_p.tolist()
