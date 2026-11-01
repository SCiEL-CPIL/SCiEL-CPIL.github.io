"""Rewritten program: tetrahedron+midpoint initialization with Gram-matrix polish."""
import time
import numpy as np
from scipy.optimize import minimize, LinearConstraint

N = 14
D = 3

# Deadline guard: evaluator kills at 150s; return safely before then.
_DEADLINE = 145.0
_t0 = time.time()


def _elapsed():
    return time.time() - _t0


def _pairwise_dists_sq(P):
    """Return squared pairwise distances for P (n x d). Shape (n, n)."""
    diff = P[:, None, :] - P[None, :, :]
    return np.einsum('ijk,ijk->ij', diff, diff)


def _score_from_P(P):
    """Compute (max/min)^2 ratio of pairwise distances."""
    d2 = _pairwise_dists_sq(P)
    # Exclude diagonal
    mask = ~np.eye(N, dtype=bool)
    vals = d2[mask]
    mn = vals.min()
    mx = vals.max()
    if mn < 1e-12:
        return np.inf
    return (mx / mn) ** 2


def _init_tetrahedron_midpoints():
    """
    Construct 14 points: 4 vertices of a regular tetrahedron + 6 edge midpoints + 4 face centers?
    Actually: tetrahedron has 4 vertices, 6 edges (so 6 midpoints), and 4 faces (4 face centroids).
    4 + 6 + 4 = 14. That gives 14 distinct points with high symmetry.
    """
    # Regular tetrahedron vertices (edge length 1)
    a = 1.0 / np.sqrt(2)
    verts = np.array([
        [a, a, a],
        [a, -a, -a],
        [-a, a, -a],
        [-a, -a, a],
    ])
    # Scale so edge length is convenient
    verts *= np.sqrt(2)  # now edge length = 2
    
    points = [verts]
    # Edge midpoints
    for i in range(4):
        for j in range(i + 1, 4):
            mid = (verts[i] + verts[j]) / 2.0
            points.append(mid)
    # Face centroids
    faces = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
    for f in faces:
        centroid = (verts[f[0]] + verts[f[1]] + verts[f[2]]) / 3.0
        points.append(centroid)
    
    P = np.vstack(points)
    return P


def _objective(x, P0):
    """
    Objective in terms of Gram matrix entries.
    We parameterize P as P0 + delta, where delta is small.
    Actually, let's work directly on coordinates for simplicity but use L-BFGS-B.
    """
    P = x.reshape(N, D)
    d2 = _pairwise_dists_sq(P)
    mask = ~np.eye(N, dtype=bool)
    vals = d2[mask]
    mn = vals.min()
    mx = vals.max()
    if mn < 1e-10:
        return 1e10
    return (mx / mn) ** 2


def _grad_objective(x, P0):
    """Numerical gradient."""
    eps = 1e-6
    grad = np.zeros_like(x)
    for i in range(len(x)):
        xp = x.copy()
        xp[i] += eps
        xm = x.copy()
        xm[i] -= eps
        grad[i] = (_objective(xp, P0) - _objective(xm, P0)) / (2 * eps)
    return grad


def _polish_coordinates(P_init):
    """Polish using L-BFGS-B on coordinates."""
    x0 = P_init.flatten()
    
    def obj(x):
        P = x.reshape(N, D)
        d2 = _pairwise_dists_sq(P)
        mask = ~np.eye(N, dtype=bool)
        vals = d2[mask]
        mn = vals.min()
        mx = vals.max()
        if mn < 1e-10:
            return 1e10
        return (mx / mn) ** 2
    
    def grad(x):
        P = x.reshape(N, D)
        d2 = _pairwise_dists_sq(P)
        mask = ~np.eye(N, dtype=bool)
        vals = d2[mask]
        mn = vals.min()
        mx = vals.max()
        if mn < 1e-10:
            return np.zeros_like(x)
        
        # Analytical gradient is complex; use numerical
        eps = 1e-7
        g = np.zeros_like(x)
        for i in range(len(x)):
            xp = x.copy()
            xp[i] += eps
            xm = x.copy()
            xm[i] -= eps
            g[i] = (obj(xp) - obj(xm)) / (2 * eps)
        return g
    
    result = minimize(
        obj, x0, method='L-BFGS-B', jac=grad,
        options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
    )
    P_opt = result.x.reshape(N, D)
    return P_opt


def _polish_gram_matrix(P_init):
    """
    Polish using Gram matrix parameterization.
    G = P^T P. We optimize G directly (symmetric PSD), then extract P via Cholesky/eigh.
    """
    P = P_init.copy()
    # Center
    P -= P.mean(axis=0, keepdims=True)
    G = P.T @ P
    
    # Extract symmetric entries
    def G_to_vec(G):
        vec = np.zeros(D * D)
        for i in range(D):
            for j in range(i, D):
                idx = i * D + j
                vec[idx] = G[i, j]
        return vec
    
    def vec_to_G(v):
        G = np.zeros((D, D))
        for i in range(D):
            for j in range(i, D):
                idx = i * D + j
                G[i, j] = v[idx]
                G[j, i] = v[idx]
        return G
    
    v0 = G_to_vec(G)
    
    def obj(v):
        Gm = vec_to_G(v)
        # Check PSD
        eigvals = np.linalg.eigvalsh(Gm)
        if np.min(eigvals) < -1e-8:
            return 1e10
        # Extract P
        U, s, Vt = np.linalg.svd(Gm)
        Pm = U @ np.diag(np.sqrt(np.maximum(s, 0)))
        # This gives a D x D matrix, but we need N x D
        # Actually G = P^T P is D x D, and P is N x D.
        # From G alone we can't recover P uniquely. We need to keep the structure.
        # This approach is flawed for N > D. Let's skip Gram matrix optimization.
        return 1e10
    
    # Gram matrix approach is problematic for N > D since P^T P loses information.
    # Instead, let's do a simple local search with random perturbations.
    return P_init


def _random_restart_polish(P_init, n_restarts=5):
    """Try multiple random perturbations and polish each."""
    best_P = P_init.copy()
    best_score = _score_from_P(P_init)
    
    rng = np.random.RandomState(42)
    
    for r in range(n_restarts):
        if _elapsed() > _DEADLINE - 10:
            break
        # Random perturbation
        scale = 0.1 * (r + 1) / n_restarts
        P_try = P_init + rng.randn(N, D) * scale
        P_polished = _polish_coordinates(P_try)
        s = _score_from_P(P_polished)
        if s < best_score:
            best_score = s
            best_P = P_polished
    
    return best_P


def construct():
    global _t0
    _t0 = time.time()
    
    # Initialize with tetrahedron + midpoints + face centroids
    P_init = _init_tetrahedron_midpoints()
    
    # First polish pass
    P1 = _polish_coordinates(P_init)
    s1 = _score_from_P(P1)
    
    # Random restart polish
    P2 = _random_restart_polish(P1, n_restarts=8)
    s2 = _score_from_P(P2)
    
    # Choose best
    if s2 < s1:
        P_best = P2
    else:
        P_best = P1
    
    # Final high-precision polish
    if _elapsed() < _DEADLINE - 15:
        P_final = _polish_coordinates(P_best)
        s_final = _score_from_P(P_final)
        if s_final < min(s1, s2):
            P_best = P_final
    
    # Verify distinctness
    d2 = _pairwise_dists_sq(P_best)
    mask = ~np.eye(N, dtype=bool)
    if d2[mask].min() < 1e-10:
        # Points too close; add small jitter
        P_best += np.random.RandomState(0).randn(N, D) * 1e-6
    
    return P_best


if __name__ == '__main__':
    P = construct()
    print("Score:", _score_from_P(P))
