"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _cut_size(adj, labels):
    """Compute the number of edges crossing the partition."""
    # labels are 0/1. An edge (v,u) is in the cut if labels[v] != labels[u].
    # cut = 0.5 * sum_{v,u} adj[v,u] * (labels[v] != labels[u])
    # Since adj is symmetric and we count each edge twice in the full sum, divide by 2.
    # (labels[v] != labels[u]) is 1 if different, 0 if same.
    # We can compute: for each v, count neighbors u where labels[u] != labels[v].
    # total = sum_v (adj[v] @ (1 - labels))  ... wait, if labels[v]=0, we want neighbors with label 1.
    # if labels[v]=1, we want neighbors with label 0.
    # So for v, contribution is adj[v] @ (1 - labels) if labels[v]==0, else adj[v] @ labels.
    # Alternatively: cut = 0.5 * sum_{v,u} adj[v,u] * |labels[v] - labels[u]|
    # Since labels are 0/1, |labels[v]-labels[u]| is 1 if different.
    # Let's use: cut = 0.5 * (labels @ adj @ (1-labels) + (1-labels) @ adj @ labels)
    # Actually simpler:
    # cut = 0.5 * sum_{v} sum_{u} adj[v,u] * (labels[v] - labels[u])^2
    # = 0.5 * (labels^T A labels - 2 labels^T A (1-labels) + (1-labels)^T A (1-labels)) ... no.
    # Let's just do:
    # diff = labels[:, None] != labels[None, :]
    # cut = 0.5 * (adj * diff).sum()
    # This is O(n^2) memory and time. For n=100, it's fine.
    # But we can do it in O(n^2) time without extra large memory if we loop?
    # Or use matrix multiplication:
    # cut = 0.5 * (labels @ (adj @ (1-labels)) + (1-labels) @ (adj @ labels))
    # Let's verify:
    # labels @ (adj @ (1-labels)) = sum_v labels[v] * sum_u adj[v,u] * (1-labels[u])
    # = sum_{v,u} labels[v] * adj[v,u] * (1-labels[u])
    # This counts edges (v,u) where labels[v]=1 and labels[u]=0.
    # Similarly the other term counts edges where labels[v]=0 and labels[u]=1.
    # Sum is exactly the number of cut edges (counting each once? No, each cut edge is counted once in one term and not the other?
    # If v is 1 and u is 0, first term has 1*1*1=1. Second term has 0*1*0=0. Sum=1.
    # If v is 0 and u is 1, first term has 0*1*0=0. Second term has 1*1*1=1. Sum=1.
    # So sum is exactly the cut size. No factor of 0.5 needed?
    # Wait, adj is symmetric. The sum over v,u covers all ordered pairs.
    # The expression sum_{v,u} labels[v] adj[v,u] (1-labels[u]) counts each directed edge from 1 to 0.
    # Since the graph is undirected, for every edge (v,u) with v in 1, u in 0, there is also (u,v) with u in 0, v in 1.
    # The first term counts (v,u) where v=1, u=0.
    # The second term counts (v,u) where v=0, u=1.
    # So the total sum is 2 * (number of cut edges).
    # So cut = 0.5 * (labels @ (adj @ (1-labels)) + (1-labels) @ (adj @ labels))
    # Let's use this vectorized approach.
    l = labels.astype(float)
    o = 1.0 - l
    return 0.5 * (l @ (adj @ o) + o @ (adj @ l))


def _greedy_flip(adj, labels, max_iter=100):
    """Greedy 1-flip local improvement."""
    n = len(labels)
    adj = np.asarray(adj, dtype=float)
    labels = labels.copy()
    
    # Precompute degrees? No, we need dynamic counts.
    # For each vertex v, let s(v) = sum_{u ~ v} (2*labels[u] - 1)
    # If we flip v, the change in cut size is:
    # new_cut - old_cut = sum_{u ~ v} (labels[u] != new_labels[v]) - (labels[u] != labels[v])
    # Let x_v = 2*labels[v] - 1.
    # The gain from flipping v is proportional to -x_v * sum_{u ~ v} x_u.
    # Specifically, if x_v * sum_{u~v} x_u < 0, flipping improves the cut.
    
    x = 2.0 * labels - 1.0
    for _ in range(max_iter):
        improved = False
        # Compute sum of neighbor x values
        # s = adj @ x
        s = adj @ x
        # For each v, gain = -x_v * s_v
        gains = -x * s
        # Find the vertex with the largest positive gain
        max_gain = gains.max()
        if max_gain <= 1e-12:
            break
        v = gains.argmax()
        labels[v] = 1 - labels[v]
        x[v] = -x[v]
        improved = True
        if not improved:
            break
    return labels


def _two_swap(adj, labels, max_iter=50):
    """Local improvement using 2-swap moves."""
    n = len(labels)
    adj = np.asarray(adj, dtype=float)
    labels = labels.copy()
    
    for _ in range(max_iter):
        improved = False
        best_gain = 0
        best_pair = None
        # Try all pairs? O(n^4) is too slow for n=100 (100^4 = 100M).
        # We can restrict to promising pairs or use a heuristic.
        # For now, let's stick to 1-flip which is O(n^2) per sweep.
        # 2-swap is harder to vectorize efficiently without loops.
        # Let's skip 2-swap for now and rely on multiple restarts.
        pass
    return labels


def _qaoa_like(adj, n_layers=3, n_angles=100):
    """
    A QAOA-inspired variational approach.
    We simulate the expectation value of the cost Hamiltonian for given angles.
    H_C = -sum_{(i,j) in E} Z_i Z_j  (Max Cut)
    H_M = -sum_i X_i
    State: |psi> = e^{-i beta_m X} e^{-i gamma_m Z} ... e^{-i beta_1 X} e^{-i gamma_1 Z} |+>^n
    We want to maximize <psi| H_C |psi>.
    
    This is hard to simulate exactly for n=100.
    Instead, we use a classical surrogate:
    Let x_i be the "probability" or "amplitude" of vertex i being in state 1.
    We can use a mean-field approximation or a simple iterative update.
    
    Alternatively, we can use the fact that QAOA on Max Cut with 1 layer is related to
    optimizing a function of the angles.
    
    Let's try a simpler variational approach:
    Maximize f(theta) = sum_{(i,j) in E} cos(theta_i - theta_j) where theta_i are angles.
    This is equivalent to Max Cut if we map theta_i to {0, pi}.
    We can optimize theta_i using gradient descent or coordinate descent.
    """
    n = len(adj)
    adj = np.asarray(adj, dtype=float)
    
    # Initialize angles randomly
    theta = np.random.uniform(0, 2 * np.pi, n)
    
    # Coordinate descent
    for _ in range(50):
        for i in range(n):
            # Gradient of f w.r.t theta_i
            # f = sum_{j} adj[i,j] cos(theta_i - theta_j)
            # df/dtheta_i = -sum_{j} adj[i,j] sin(theta_i - theta_j)
            # We want to maximize f, so we update theta_i -= lr * df/dtheta_i
            # But we can also just search for the best theta_i in a small neighborhood.
            
            # Let's do a local search for theta_i
            # The optimal theta_i satisfies sum_{j} adj[i,j] sin(theta_i - theta_j) = 0
            # This is like a "phase matching" problem.
            
            # Simple update: theta_i = atan2(sum_j adj[i,j] sin(theta_j), sum_j adj[i,j] cos(theta_j))
            # This minimizes the "energy" of the Ising model with continuous spins.
            sin_sum = (adj[i] * np.sin(theta)).sum()
            cos_sum = (adj[i] * np.cos(theta)).sum()
            theta[i] = np.arctan2(sin_sum, cos_sum)
            
    # Map angles to binary labels
    # If theta_i is close to 0, label 0. If close to pi, label 1.
    # Or simply: label = 1 if theta_i > pi/2 else 0?
    # Better: label = 1 if cos(theta_i) < 0 else 0?
    # Let's use: label = 1 if theta_i in (pi/2, 3pi/2) else 0.
    labels = (theta > np.pi / 2).astype(int)
    # Adjust for range 0 to 2pi
    labels = ((theta % (2*np.pi)) > np.pi/2).astype(int)
    
    return labels


def partition(adjacency):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n = len(adjacency)
    adj = np.asarray(adjacency, dtype=float)
    
    best_labels = None
    best_cut = -1
    
    # Multiple restarts with different initialization strategies
    n_restarts = 50
    
    for r in range(n_restarts):
        if time.time() > deadline:
            break
            
        # Choose initialization
        if r % 3 == 0:
            # Random
            labels = np.random.randint(0, 2, n)
        elif r % 3 == 1:
            # Spectral initialization (Laplacian Fiedler vector)
            # D - A
            deg = adj.sum(axis=1)
            L = np.diag(deg) - adj
            # Compute eigenvector corresponding to smallest non-zero eigenvalue
            # For n=100, full eigendecomposition is fast.
            try:
                w, v = np.linalg.eigh(L)
                # Sort by eigenvalue
                idx = np.argsort(w)
                # The first eigenvector is usually the constant vector.
                # The second is the Fiedler vector.
                fiedler = v[:, idx[1]]
                labels = (fiedler > 0).astype(int)
            except:
                labels = np.random.randint(0, 2, n)
        else:
            # QAOA-like initialization
            labels = _qaoa_like(adj)
        
        # Greedy improvement
        labels = _greedy_flip(adj, labels, max_iter=200)
        
        # Evaluate
        cut = _cut_size(adj, labels)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
            
    # Final polish: more aggressive greedy
    if best_labels is not None and time.time() < deadline - 5:
        best_labels = _greedy_flip(adj, best_labels, max_iter=1000)
        best_cut = _cut_size(adj, best_labels)
        
    if best_labels is None:
        # Fallback
        best_labels = np.arange(n) % 2
        
    return best_labels
# EVOLVE-BLOCK-END
