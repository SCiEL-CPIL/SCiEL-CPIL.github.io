import numpy as np
from scipy.optimize import minimize
import itertools

N = 14
D = 3

def construct():
    # Number of pairwise distances
    M = N * (N - 1) // 2
    
    # Indices for upper triangle
    i_idx, j_idx = np.triu_indices(N, k=1)
    
    def edm_from_sq_dists(x):
        """
        Construct Euclidean Distance Matrix from vector of squared distances.
        x: vector of length M, where x[k] = d_{i_idx[k], j_idx[k]}^2
        """
        D_mat = np.zeros((N, N))
        D_mat[i_idx, j_idx] = x
        D_mat[j_idx, i_idx] = x
        return D_mat

    def gram_from_edm(D_mat):
        """
        Compute the centered Gram matrix from EDM.
        G = -1/2 * J * D * J
        """
        J = np.eye(N) - np.ones((N, N)) / N
        G = -0.5 * J @ D_mat @ J
        # Symmetrize to remove numerical errors
        G = (G + G.T) / 2
        return G

    def points_from_gram(G):
        """
        Recover points from Gram matrix.
        G is N x N, PSD, rank <= D.
        """
        # Eigenvalue decomposition
        eigvals, eigvecs = np.linalg.eigh(G)
        # Clip small negative eigenvalues to 0
        eigvals = np.clip(eigvals, 0, None)
        # Take top D eigenvalues
        # We want Y such that Y Y^T = G
        # Y = eigvecs[:, :D] * sqrt(eigvals[:D])
        # Note: eigvecs columns are eigenvectors.
        # G = V diag(w) V^T
        # Y = V[:, :D] diag(sqrt(w[:D]))
        # The points are the rows of Y.
        Y = eigvecs[:, :D] * np.sqrt(eigvals[:D])
        return Y

    def objective(x):
        # x is vector of squared distances
        # Ensure positivity
        x = np.maximum(x, 1e-12)
        
        # Normalize so min distance squared is 1
        min_x = np.min(x)
        x_norm = x / min_x
        
        D_mat = edm_from_sq_dists(x_norm)
        G = gram_from_edm(D_mat)
        
        # Compute eigenvalues for penalty
        eigvals = np.linalg.eigvalsh(G)
        
        # Penalty for non-PSD (eigvals < 0)
        neg_eig = eigvals[eigvals < 0]
        penalty_psd = np.sum(neg_eig**2) if len(neg_eig) > 0 else 0.0
        
        # Penalty for rank > D (eigvals[D:] should be 0)
        # Note: eigvals are sorted ascending.
        # So eigvals[D-1:] are the smallest? No, eigh returns ascending.
        # So eigvals[0] is smallest, eigvals[N-1] is largest.
        # Rank <= D means the largest N-D eigenvalues should be 0?
        # No, rank is the number of non-zero eigenvalues.
        # If rank <= D, then at most D eigenvalues are non-zero.
        # So eigvals[N-D:] (the largest ones) should be non-zero, and eigvals[:N-D] should be 0?
        # Wait. If G is PSD, eigenvalues are >= 0.
        # Rank is the count of eigenvalues > epsilon.
        # To enforce rank <= D, we want the eigenvalues corresponding to the "extra" dimensions to be 0.
        # Since we don't know which ones are "real", we usually assume the largest D are the real ones.
        # So we penalize the eigenvalues that are NOT among the top D.
        # i.e., penalize eigvals[0 : N-D].
        
        if N > D:
            penalty_rank = np.sum(eigvals[:N-D]**2)
        else:
            penalty_rank = 0.0
        
        penalty = penalty_psd + penalty_rank
        
        # Objective: Max distance squared (since min is fixed to 1) + Penalty
        max_x = np.max(x_norm)
        
        # Tuning parameter for penalty
        mu = 100.0
        return max_x + mu * penalty

    # Initialization
    best_x = None
    best_score = float('inf')
    
    # Generate a few initial configurations
    inits = []
    
    # 1. Fibonacci Sphere
    golden_angle = np.pi * (3 - np.sqrt(5))
    pts = np.zeros((N, D))
    for i in range(N):
        z = 1 - 2 * i / (N - 1)
        r = np.sqrt(1 - z**2)
        theta = golden_angle * i
        pts[i, 0] = r * np.cos(theta)
        pts[i, 1] = r * np.sin(theta)
        pts[i, 2] = z
    inits.append(pts)
    
    # 2. Random points
    for _ in range(5):
        pts = np.random.randn(N, D)
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        inits.append(pts)
        
    # 3. Icosahedron + 2 random
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    ico_verts = ico_verts / np.linalg.norm(ico_verts, axis=1, keepdims=True)
    rand_pts = np.random.randn(2, D)
    rand_pts = rand_pts / np.linalg.norm(rand_pts, axis=1, keepdims=True)
    all_pts = np.vstack([ico_verts, rand_pts])
    inits.append(all_pts)

    for init_pts in inits:
        # Compute initial squared distances
        dists_sq = np.sum((init_pts[:, np.newaxis, :] - init_pts[np.newaxis, :, :])**2, axis=2)
        x0 = dists_sq[i_idx, j_idx].copy()
        
        # Optimize
        try:
            res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-10})
            
            # Check if result is valid
            x_opt = res.x
            x_opt = np.maximum(x_opt, 1e-12)
            min_x = np.min(x_opt)
            x_norm = x_opt / min_x
            
            D_mat = edm_from_sq_dists(x_norm)
            G = gram_from_edm(D_mat)
            eigvals = np.linalg.eigvalsh(G)
            
            # Check penalty is small
            penalty_psd = np.sum(eigvals[eigvals < 0]**2) if np.any(eigvals < 0) else 0.0
            penalty_rank = np.sum(eigvals[:N-D]**2) if N > D else 0.0
            total_penalty = penalty_psd + penalty_rank
            
            if total_penalty < 1e-4: # Acceptable tolerance
                score = np.max(x_norm) / np.min(x_norm) # Which is just max(x_norm)
                if score < best_score:
                    best_score = score
                    best_x = x_opt
        except Exception:
            continue

    if best_x is None:
        # Fallback to random points
        X = np.random.randn(N, D)
        X = X - np.mean(X, axis=0)
        return X

    # Recover points from best_x
    x_opt = best_x
    min_x = np.min(x_opt)
    x_norm = x_opt / min_x
    
    D_mat = edm_from_sq_dists(x_norm)
    G = gram_from_edm(D_mat)
    
    # Ensure G is PSD for recovery
    eigvals, eigvecs = np.linalg.eigh(G)
    eigvals = np.clip(eigvals, 0, None)
    Y = eigvecs[:, :D] * np.sqrt(eigvals[:D])
    
    return Y

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import itertools

N = 14
D = 3

def construct():
    # Number of pairwise distances
    M = N * (N - 1) // 2
    
    # Indices for upper triangle
    i_idx, j_idx = np.triu_indices(N, k=1)
    
    def edm_from_sq_dists(x):
        D_mat = np.zeros((N, N))
        D_mat[i_idx, j_idx] = x
        D_mat[j_idx, i_idx] = x
        return D_mat

    def gram_from_edm(D_mat):
        J = np.eye(N) - np.ones((N, N)) / N
        G = -0.5 * J @ D_mat @ J
        G = (G + G.T) / 2
        return G

    def objective(x):
        x = np.maximum(x, 1e-12)
        min_x = np.min(x)
        x_norm = x / min_x
        
        D_mat = edm_from_sq_dists(x_norm)
        G = gram_from_edm(D_mat)
        
        eigvals = np.linalg.eigvalsh(G)
        
        neg_eig = eigvals[eigvals < 0]
        penalty_psd = np.sum(neg_eig**2) if len(neg_eig) > 0 else 0.0
        
        if N > D:
            penalty_rank = np.sum(eigvals[:N-D]**2)
        else:
            penalty_rank = 0.0
        
        penalty = penalty_psd + penalty_rank
        
        max_x = np.max(x_norm)
        
        mu = 100.0
        return max_x + mu * penalty

    best_x = None
    best_score = float('inf')
    
    inits = []
    
    # 1. Fibonacci Sphere
    golden_angle = np.pi * (3 - np.sqrt(5))
    pts = np.zeros((N, D))
    for i in range(N):
        z = 1 - 2 * i / (N - 1)
        r = np.sqrt(1 - z**2)
        theta = golden_angle * i
        pts[i, 0] = r * np.cos(theta)
        pts[i, 1] = r * np.sin(theta)
        pts[i, 2] = z
    inits.append(pts)
    
    # 2. Random points
    for _ in range(5):
        pts = np.random.randn(N, D)
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        inits.append(pts)
        
    # 3. Icosahedron + 2 random
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    ico_verts = ico_verts / np.linalg.norm(ico_verts, axis=1, keepdims=True)
    rand_pts = np.random.randn(2, D)
    rand_pts = rand_pts / np.linalg.norm(rand_pts, axis=1, keepdims=True)
    all_pts = np.vstack([ico_verts, rand_pts])
    inits.append(all_pts)

    for init_pts in inits:
        dists_sq = np.sum((init_pts[:, np.newaxis, :] - init_pts[np.newaxis, :, :])**2, axis=2)
        x0 = dists_sq[i_idx, j_idx].copy()
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-10})
            
            x_opt = res.x
            x_opt = np.maximum(x_opt, 1e-12)
            min_x = np.min(x_opt)
            x_norm = x_opt / min_x
            
            D_mat = edm_from_sq_dists(x_norm)
            G = gram_from_edm(D_mat)
            eigvals = np.linalg.eigvalsh(G)
            
            penalty_psd = np.sum(eigvals[eigvals < 0]**2) if np.any(eigvals < 0) else 0.0
            penalty_rank = np.sum(eigvals[:N-D]**2) if N > D else 0.0
            total_penalty = penalty_psd + penalty_rank
            
            if total_penalty < 1e-4:
                score = np.max(x_norm)
                if score < best_score:
                    best_score = score
                    best_x = x_opt
        except Exception:
            continue

    if best_x is None:
        X = np.random.randn(N, D)
        X = X - np.mean(X, axis=0)
        return X

    x_opt = best_x
    min_x = np.min(x_opt)
    x_norm = x_opt / min_x
    
    D_mat = edm_from_sq_dists(x_norm)
    G = gram_from_edm(D_mat)
    
    eigvals, eigvecs = np.linalg.eigh(G)
    eigvals = np.clip(eigvals, 0, None)
    Y = eigvecs[:, :D] * np.sqrt(eigvals[:D])
    
    return Y

# EVOLVE-BLOCK-END
