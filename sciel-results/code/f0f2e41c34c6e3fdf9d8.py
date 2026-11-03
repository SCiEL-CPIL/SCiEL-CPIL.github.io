"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    # Convert to float64 for faster arithmetic if needed, but keep int64 for cost
    # Actually, let's keep original types but ensure operations are fast.
    # a and b are int64.
    
    def compute_cost_fast(p):
        """Compute cost using vectorized matrix multiplication."""
        # cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
        # = trace(A @ B_p) where B_p[i,j] = b[p[i], p[j]]
        # B_p = b[np.ix_(p, p)]
        # cost = np.sum(a * b[np.ix_(p, p)])
        # This is O(n^2) memory and time.
        # Alternative:
        # cost = np.einsum('ij,ij->', a, b[np.ix_(p, p)])
        
        # Even faster for large n?
        # Let's try:
        # bp = b[np.ix_(p, p)]
        # return int(np.sum(a * bp))
        
        # Another way:
        # cost = np.sum(a * b[np.ix_(p, p)])
        
        # Let's use the loop version if n is small, but n=150 is small enough for vectorization.
        # However, creating b[np.ix_(p,p)] is a copy.
        
        # Let's try:
        # cost = 0
        # for i in range(n):
        #     cost += np.dot(a[i,:], b[p[i],:])
        # This is n dot products of length n.
        
        # Let's use the vectorized one first.
        bp = b[np.ix_(p, p)]
        return int(np.sum(a * bp))

    def compute_swap_delta(i, j, p):
        """Compute cost delta when swapping positions of facilities i and j."""
        li = p[i]
        lj = p[j]
        
        if li == lj:
            return 0
        
        delta = 0
        
        # Diagonal terms
        delta += a[i,i] * (b[lj,lj] - b[li,li])
        delta += a[j,j] * (b[li,li] - b[lj,lj])
        
        # Cross terms between i and j
        delta += a[i,j] * (b[lj,li] - b[li,lj])
        delta += a[j,i] * (b[li,lj] - b[lj,li])
        
        # Terms with other facilities k
        # For k not in {i,j}:
        # a[i,k]*b[li,p[k]] -> a[i,k]*b[lj,p[k]]  : delta += a[i,k]*(b[lj,p[k]] - b[li,p[k]])
        # a[k,i]*b[p[k],li] -> a[k,i]*b[p[k],lj]  : delta += a[k,i]*(b[p[k],lj] - b[p[k],li])
        # a[j,k]*b[lj,p[k]] -> a[j,k]*b[li,p[k]]  : delta += a[j,k]*(b[li,p[k]] - b[lj,p[k]])
        # a[k,j]*b[p[k],lj] -> a[k,j]*b[p[k],li]  : delta += a[k,j]*(b[p[k],li] - b[p[k],lj])
        
        # Vectorized computation for all k:
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        mask[j] = False
        ks = np.where(mask)[0]
        
        if len(ks) > 0:
            pk = p[ks]
            
            # a[i,k] * (b[lj,pk] - b[li,pk])
            delta += np.sum(a[i, ks] * (b[lj, pk] - b[li, pk]))
            # a[k,i] * (b[pk,lj] - b[pk,li])
            delta += np.sum(a[ks, i] * (b[pk, lj] - b[pk, li]))
            # a[j,k] * (b[li,pk] - b[lj,pk])
            delta += np.sum(a[j, ks] * (b[li, pk] - b[lj, pk]))
            # a[k,j] * (b[pk,li] - b[pk,lj])
            delta += np.sum(a[ks, j] * (b[pk, li] - b[pk, lj]))
        
        return delta

    def local_search(p, best_cost):
        """Perform 2-opt local search."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(n):
                if time.time() > deadline:
                    break
                for j in range(i+1, n):
                    if time.time() > deadline:
                        break
                    delta = compute_swap_delta(i, j, p)
                    if delta < 0:
                        p[i], p[j] = p[j], p[i]
                        best_cost += delta
                        improved = True
                        if best_cost < best_cost:
                            best_p = p.copy()
                            best_cost = compute_cost_fast(p)  # Recalculate to avoid drift
        return p, best_cost

    # Initial heuristic: rank matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    p = p.astype(np.int64)
    
    best_p = p.copy()
    best_cost = compute_cost_fast(p)
    
    # Local search from initial solution
    best_p, best_cost = local_search(best_p, best_cost)
    
    # Spectral initialization:
    # We want to find a permutation p that minimizes sum_{i,j} a[i,j] b[p[i], p[j]].
    # This is equivalent to maximizing sum_{i,j} a[i,j] (-b[p[i], p[j]]) if we flip sign.
    # Let's use a spectral method based on the idea of embedding.
    
    # One common spectral heuristic for QAP is to use the eigenvectors of the matrices.
    # Let's try:
    # 1. Compute the Laplacian or some matrix derived from b.
    # 2. Compute the eigenvectors.
    # 3. Use the signs of the eigenvectors to assign facilities to locations.
    
    # However, a simpler spectral approach is to use the correlation between a and b.
    # Let's try:
    # Compute the matrix C = a @ b.T
    # Then find the permutation that maximizes sum_{i,j} C[i,j] * x[i,j] where x is the assignment matrix.
    
    # Actually, let's try a different spectral approach:
    # Use the top eigenvector of the matrix M = a @ b.T + b @ a.T
    # Then assign facilities to locations based on the signs of the eigenvector components.
    
    # But this is heuristic and may not work well.
    
    # Let's try a simpler approach:
    # Use the eigenvectors of the distance matrix b.
    # Compute the top k eigenvectors of b.
    # Then assign facilities to locations based on the projections onto these eigenvectors.
    
    # Let's try:
    # 1. Compute the eigenvectors of b.
    # 2. Use the top eigenvector to assign facilities to locations.
    
    # However, this is not directly applicable.
    
    # Let's try a different spectral method:
    # Use the matrix A = a @ b.T
    # Then find the permutation that maximizes sum_{i,j} A[i,j] * x[i,j]
    
    # This is a linear assignment problem, which can be solved using the Hungarian algorithm.
    
    # Let's try:
    # 1. Compute A = a @ b.T
    # 2. Solve the linear assignment problem to find the permutation that maximizes sum_{i,j} A[i,j] * x[i,j]
    
    # This is a valid heuristic for QAP.
    
    if time.time() < deadline - 10:
        # Compute A = a @ b.T
        A = a @ b.T
        
        # Solve the linear assignment problem
        # We want to maximize sum_{i,j} A[i,j] * x[i,j]
        # This is equivalent to minimizing sum_{i,j} -A[i,j] * x[i,j]
        
        from scipy.optimize import linear_sum_assignment
        
        row_ind, col_ind = linear_sum_assignment(-A)
        
        p_spectral = np.empty(n, dtype=np.int64)
        p_spectral[row_ind] = col_ind
        
        cost_spectral = compute_cost_fast(p_spectral)
        
        if cost_spectral < best_cost:
            best_p = p_spectral.copy()
            best_cost = cost_spectral
        
        # Local search from spectral solution
        p_spectral, cost_spectral = local_search(p_spectral, cost_spectral)
        
        if cost_spectral < best_cost:
            best_p = p_spectral.copy()
            best_cost = cost_spectral
    
    # Try another spectral approach:
    # Use the eigenvectors of the matrix M = a + a.T
    # Then assign facilities to locations based on the signs of the eigenvector components.
    
    if time.time() < deadline - 10:
        # Compute the eigenvectors of M = a + a.T
        M = a + a.T
        eigvals, eigvecs = np.linalg.eigh(M)
        
        # Use the top eigenvector
        top_eigvec = eigvecs[:, -1]
        
        # Assign facilities to locations based on the signs of the eigenvector components
        # This is heuristic and may not work well.
        
        # Let's try a different approach:
        # Use the eigenvectors of the distance matrix b.
        
        eigvals_b, eigvecs_b = np.linalg.eigh(b)
        
        # Use the top eigenvector of b
        top_eigvec_b = eigvecs_b[:, -1]
        
        # Assign facilities to locations based on the correlation between a and b
        # This is heuristic and may not work well.
        
        # Let's try a simpler approach:
        # Use the eigenvectors of the matrix C = a @ b.T
        # Then assign facilities to locations based on the signs of the eigenvector components.
        
        C = a @ b.T
        eigvals_c, eigvecs_c = np.linalg.eigh(C)
        
        # Use the top eigenvector of C
        top_eigvec_c = eigvecs_c[:, -1]
        
        # Assign facilities to locations based on the signs of the eigenvector components
        # This is heuristic and may not work well.
        
        # Let's try a different approach:
        # Use the eigenvectors of the matrix D = a @ b.T + b @ a.T
        # Then assign facilities to locations based on the signs of the eigenvector components.
        
        D = a @ b.T + b @ a.T
        eigvals_d, eigvecs_d = np.linalg.eigh(D)
        
        # Use the top eigenvector of D
        top_eigvec_d = eigvecs_d[:, -1]
        
        # Assign facilities to locations based on the signs of the eigenvector components
        # This is heuristic and may not work well.
        
        # Let's try a simpler approach:
        # Use the eigenvectors of the matrix E = a @ b.T - b @ a.T
        # Then assign facilities to locations based on the signs of the eigenvector components.
        
        E = a @ b.T - b @ a.T
        eigvals_e, eigvecs_e = np.linalg.eigh(E)
        
        # Use the top eigenvector of E
        top_eigvec_e = eigvecs_e[:, -1]
        
        # Assign facilities to locations based on the signs of the eigenvector components
        # This is heuristic and may not work well.
        
        # Let's try a different approach:
        # Use the eigenvectors of the matrix F = a @ b.T + a.T @ b
        # Then assign facilities to locations based on the signs of the eigenvector components.
        
        F = a @ b.T + a.T @ b
        eigvals_f, eigvecs_f = np.linalg.eigh(F)
        
        # Use the top eigenvector of F
        top_eigvec_f = eigvecs_f[:, -1]
        
        # Assign facilities to locations based on the signs of the eigenvector components
        # This is heuristic and may not work well.
        
        # Let's try a simpler approach:
        # Use the eigenvectors of the matrix G = a @ b.T - a.T @ b
        # Then assign facilities to locations based on the signs of the eigenvector components.
        
        G = a @ b.T - a.T @ b
        eigvals_g, eigvecs_g = np.linalg.eigh(G)
        
        # Use the top eigenvector of G
        top_eigvec_g = eigvecs_g[:, -1]
        
        # Assign facilities to locations based on the signs of the eigenvector components
        # This is heuristic and may not work well.
    
    # Try random restarts with local search
    if time.time() < deadline - 5:
        rng = np.random.RandomState(42)
        for restart in range(10):
            if time.time() > deadline - 5:
                break
            
            # Random initial permutation
            p_random = rng.permutation(n)
            
            cost_random = compute_cost_fast(p_random)
            
            if cost_random < best_cost:
                best_p = p_random.copy()
                best_cost = cost_random
            
            # Local search from random solution
            p_random, cost_random = local_search(p_random, cost_random)
            
            if cost_random < best_cost:
                best_p = p_random.copy()
                best_cost = cost_random
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
