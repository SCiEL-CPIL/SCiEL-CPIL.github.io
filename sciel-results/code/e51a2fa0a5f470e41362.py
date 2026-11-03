"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _qap_cost(a, b, p):
    """Compute exact QAP cost for permutation p."""
    bp = b[np.ix_(p, p)]
    return int(np.sum(a * bp))


def _swap_delta(a, b, p, i, j):
    """
    Compute the change in QAP cost when swapping facilities at positions i and j
    in the permutation p (i.e., swapping p[i] and p[j]).
    
    p[i] -> p[j], p[j] -> p[i]
    Let u = p[i], v = p[j].
    
    The cost is sum_{k,l} a[k,l] * b[p[k], p[l]].
    Only terms involving p[i] or p[j] change.
    
    Changed terms:
    - a[i,i]*b[u,u] -> a[i,i]*b[v,v]
    - a[j,j]*b[v,v] -> a[j,j]*b[u,u]
    - a[i,j]*b[u,v] -> a[i,j]*b[v,u]
    - a[j,i]*b[v,u] -> a[j,i]*b[u,v]
    - For k != i,j: a[i,k]*b[u,p[k]] -> a[i,k]*b[v,p[k]]
    - For k != i,j: a[j,k]*b[v,p[k]] -> a[j,k]*b[u,p[k]]
    - For k != i,j: a[k,i]*b[p[k],u] -> a[k,i]*b[p[k],v]
    - For k != i,j: a[k,j]*b[p[k],v] -> a[k,j]*b[p[k],u]
    
    Delta = new - old
    """
    u = p[i]
    v = p[j]
    
    # Terms that swap between i and j
    # a[i,i]*b[u,u] + a[j,j]*b[v,v] + a[i,j]*b[u,v] + a[j,i]*b[v,u]
    old_ii = a[i, i] * b[u, u] + a[j, j] * b[v, v] + a[i, j] * b[u, v] + a[j, i] * b[v, u]
    # a[i,i]*b[v,v] + a[j,j]*b[u,u] + a[i,j]*b[v,u] + a[j,i]*b[u,v]
    new_ii = a[i, i] * b[v, v] + a[j, j] * b[u, u] + a[i, j] * b[v, u] + a[j, i] * b[u, v]
    
    # Terms with other positions k
    # For k != i,j:
    # old: a[i,k]*b[u,p[k]] + a[j,k]*b[v,p[k]] + a[k,i]*b[p[k],u] + a[k,j]*b[p[k],v]
    # new: a[i,k]*b[v,p[k]] + a[j,k]*b[u,p[k]] + a[k,i]*b[p[k],v] + a[k,j]*b[p[k],u]
    
    # Vectorized computation for all k != i, j
    mask = np.ones(len(p), dtype=bool)
    mask[i] = False
    mask[j] = False
    k_indices = np.where(mask)[0]
    
    if len(k_indices) > 0:
        pk = p[k_indices]
        
        # a[i,k] for all k in k_indices
        a_i_k = a[i, k_indices]
        a_j_k = a[j, k_indices]
        a_k_i = a[k_indices, i]
        a_k_j = a[k_indices, j]
        
        # b[u,pk], b[v,pk]
        b_u_pk = b[u, pk]
        b_v_pk = b[v, pk]
        b_pk_u = b[pk, u]
        b_pk_v = b[pk, v]
        
        old_cross = np.sum(a_i_k * b_u_pk + a_j_k * b_v_pk + a_k_i * b_pk_u + a_k_j * b_pk_v)
        new_cross = np.sum(a_i_k * b_v_pk + a_j_k * b_u_pk + a_k_i * b_pk_v + a_k_j * b_pk_u)
    else:
        old_cross = 0
        new_cross = 0
    
    delta = (new_ii + new_cross) - (old_ii + old_cross)
    return int(delta)


def _two_opt_search(a, b, p, max_iterations=100000, time_limit=None, start_time=None):
    """
    Perform 2-opt local search on permutation p.
    Returns (best_p, best_cost).
    """
    if time_limit is None:
        time_limit = 100
    if start_time is None:
        start_time = time.time()
    
    n = len(p)
    p = p.copy()
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    improved = True
    iteration = 0
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        
        # Check time limit
        if time.time() - start_time > time_limit - 2:
            break
        
        # Try all pairs (i, j) with i < j
        # To speed up, we can sample or use a heuristic order
        # For now, try all pairs but break early if time is running out
        
        for i in range(n):
            if time.time() - start_time > time_limit - 2:
                break
            
            for j in range(i + 1, n):
                if time.time() - start_time > time_limit - 1:
                    break
                
                delta = _swap_delta(a, b, p, i, j)
                if delta < 0:
                    # Accept the swap
                    p[i], p[j] = p[j], p[i]
                    current_cost += delta
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p.copy()
                    improved = True
    
    return best_p, best_cost


def _continuous_embedding_init(a, b, time_budget=30, dim=64, lr=0.01, n_iters=2000, seed=42):
    """
    Initialize permutation using continuous vector assignment.
    
    Model: Each facility i is represented as a vector x_i in R^d.
    The objective is to minimize sum_{i,j} a[i,j] * ||x_i - x_j||^2
    subject to the constraint that the pairwise distances match those implied by b.
    
    We use a smooth surrogate: minimize sum_{i,j} a[i,j] * (||x_i - x_j||^2 - d_ij)^2
    where d_ij is a target distance derived from b.
    
    After optimization, extract a permutation by matching facilities to locations
    based on their embedded positions.
    """
    start_time = time.time()
    n = len(a)
    rng = np.random.default_rng(seed)
    
    # Compute target distances from b matrix
    # Use b[i,j] as a proxy for desired "closeness" - larger b means closer
    # Target distance: d_ij = 1 / (b[i,j] + eps) for i != j, 0 for i == j
    eps = 1e-6
    b_safe = b + eps
    target_dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                target_dist[i, j] = 1.0 / b_safe[i, j]
    
    # Normalize target distances to [0, 1] range
    max_td = target_dist.max()
    if max_td > 0:
        target_dist = target_dist / max_td
    
    # Initialize positions randomly
    X = rng.standard_normal((n, dim)) * 0.1
    
    # Precompute a for efficiency
    # Gradient of sum_{i,j} a[i,j] * (||x_i - x_j||^2 - d_ij)^2 w.r.t. x_i
    # = 4 * sum_j a[i,j] * (||x_i - x_j||^2 - d_ij) * (x_i - x_j)
    
    # Use Adam-like optimizer for better convergence
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    t = 0
    
    for it in range(n_iters):
        if time.time() - start_time > time_budget - 2:
            break
        
        t += 1
        grad = np.zeros_like(X)
        
        # Compute gradient
        for i in range(n):
            # grad_i = 4 * sum_j a[i,j] * (||x_i - x_j||^2 - d_ij) * (x_i - x_j)
            diff = X[i] - X  # shape (n, dim)
            dist_sq = np.sum(diff**2, axis=1)  # shape (n,)
            residual = dist_sq - target_dist[i]  # shape (n,)
            # Weighted sum
            weighted = a[i] * residual  # shape (n,)
            grad[i] = 4.0 * np.sum(weighted[:, np.newaxis] * diff, axis=0)
        
        # Adam update
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        X -= lr * m_hat / (np.sqrt(v_hat) + epsilon)
    
    # Extract permutation from embedding
    # Match facilities to locations using their embedded positions
    # We want to find a permutation p such that the embedded distances match b
    
    # Strategy: Use the embedding to define a similarity matrix, then solve
    # an assignment problem
    
    # Compute pairwise distances in embedding space
    X_norm = X - X.mean(axis=0)
    dist_embed = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_embed[i, j] = np.sum((X[i] - X[j])**2)
    
    # We want to match facility i to location p[i] such that the embedding
    # distances are consistent with b
    # Use a cost matrix: cost[i, j] = sum_k a[i,k] * |dist_embed[i,k] - dist_embed[j,k]|^2
    
    # Simpler approach: Use the embedding to rank facilities and match to locations
    # based on their "centrality" in the embedding
    
    # Compute centrality of each facility in embedding
    embed_centrality = np.sum(dist_embed, axis=1)
    
    # Compute centrality of each location from b
    b_centrality = b.sum(axis=1)
    
    # Match by rank
    facility_rank = np.argsort(embed_centrality)
    location_rank = np.argsort(b_centrality)
    
    p = np.empty(n, dtype=np.int64)
    for i, loc in enumerate(location_rank):
        p[loc] = facility_rank[i]
    
    return p


def _rank_matching_init(a, b):
    """
    Simple rank-matching initialization based on row sums.
    """
    n = len(a)
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_row_sums, kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    return p


def _hybrid_search(a, b, time_budget=110):
    """
    Perform hybrid search: continuous embedding init + rank matching + 2-opt polishing.
    """
    start_time = time.time()
    n = len(a)
    
    best_p = None
    best_cost = float('inf')
    
    # 1. Continuous embedding initialization
    embed_time = min(25, time_budget // 4)
    if embed_time > 5:
        p_embed = _continuous_embedding_init(a, b, time_budget=embed_time, dim=64, lr=0.01, n_iters=2000, seed=42)
        cost = _qap_cost(a, b, p_embed)
        if cost < best_cost:
            best_cost = cost
            best_p = p_embed.copy()
    
    # 2. Rank matching initialization
    p_rank = _rank_matching_init(a, b)
    cost = _qap_cost(a, b, p_rank)
    if cost < best_cost:
        best_cost = cost
        best_p = p_rank.copy()
    
    # 3. Polish the best solution with 2-opt
    if best_p is not None:
        remaining_time = time_budget - (time.time() - start_time)
        if remaining_time > 5:
            p_opt, cost_opt = _two_opt_search(a, b, best_p, time_limit=remaining_time - 2, start_time=start_time)
            if cost_opt < best_cost:
                best_cost = cost_opt
                best_p = p_opt.copy()
    
    # 4. Try additional random restarts if time allows
    rng = np.random.default_rng(123)
    while time.time() - start_time < time_budget - 10:
        remaining_time = time_budget - (time.time() - start_time)
        if remaining_time < 5:
            break
        
        p_rand = rng.permutation(n)
        per_restart_time = min(remaining_time - 2, 10)
        if per_restart_time < 2:
            break
            
        p_opt, cost_opt = _two_opt_search(a, b, p_rand, time_limit=per_restart_time, start_time=time.time())
        
        if cost_opt < best_cost:
            best_cost = cost_opt
            best_p = p_opt.copy()
    
    return best_p, best_cost


def solve(a, b):
    """
    Solve the QAP instance tai256c.
    Returns a permutation p of 0..n-1 as a list of integers.
    """
    start_time = time.time()
    n = len(a)
    
    # Time budget: leave 10s margin for safety
    time_budget = 110
    
    best_p, best_cost = _hybrid_search(a, b, time_budget=time_budget)
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
