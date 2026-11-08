"""
Strategy: Multi-Armed Bandit (UCB) over a portfolio of scoring heuristics.
Instead of trying to improve a single score vector, we evaluate multiple
distinct heuristics (Degree, PageRank, Eigenvector, Random, etc.) and use
UCB to decide which heuristic to re-evaluate or refine if time permits, 
though here we primarily use it to select the final best heuristic after 
sampling them. The "arms" are the heuristics themselves.
"""

import numpy as np
import time
import random

def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    adj = np.asarray(adjacency, dtype=np.bool_)
    n = adj.shape[0]
    
    if n == 0:
        return np.array([])
    
    # Precompute neighbor lists for fast greedy evaluation
    neighbor_indices = [np.nonzero(adj[i])[0] for i in range(n)]
    
    def greedy_size_from_order(order):
        """Compute greedy MIS size given a priority order."""
        # Use a set for O(1) lookup of selected vertices
        # But for n=200, a boolean array is faster
        selected = np.zeros(n, dtype=bool)
        count = 0
        for v in order:
            if not selected[v]:
                count += 1
                selected[v] = True
                if len(neighbor_indices[v]) > 0:
                    selected[neighbor_indices[v]] = True
        return count
    
    def order_from_scores(s):
        """Get vertex order by descending score."""
        return np.argsort(-s)
    
    # Define a portfolio of heuristics
    # Each heuristic returns a score vector
    heuristics = []
    
    # 1. Degree (Low degree first)
    def h_degree(adj, n):
        degrees = adj.sum(axis=1)
        return -degrees.astype(float)
    heuristics.append(h_degree)
    
    # 2. PageRank (Inverse of "centrality")
    def h_pagerank(adj, n):
        # Simple power iteration for PageRank
        # Low PageRank = less central = better for MIS
        d = adj.sum(axis=1)
        d = np.where(d == 0, 1, d)
        pr = np.ones(n) / n
        for _ in range(50):
            new_pr = 0.85 * (adj.T @ (pr / d)) + 0.15 / n
            pr = new_pr
        return -pr
    heuristics.append(h_pagerank)
    
    # 3. Eigenvector Centrality (Inverse)
    def h_eigen(adj, n):
        try:
            # Use numpy's power iteration or eig
            # For small n, eig is fine
            vals, vecs = np.linalg.eig(adj)
            # Principal eigenvector
            idx = np.argmax(vals.real)
            vec = vecs[:, idx].real
            vec = np.abs(vec)
            return -vec
        except:
            return -adj.sum(axis=1).astype(float)
    heuristics.append(h_eigen)
    
    # 4. Random
    def h_random(adj, n):
        return np.random.rand(n)
    heuristics.append(h_random)
    
    # 5. Local Density (Inverse of 2-hop neighborhood size)
    def h_local_density(adj, n):
        # Approximate 2-hop density
        scores = np.zeros(n)
        for i in range(n):
            # Count neighbors of neighbors
            # adj[i] is boolean mask
            # adj @ adj[i] gives count of 2-hop paths
            # But this is expensive for n=200? 200*200*200 = 8M ops, fine.
            pass
        # Vectorized:
        # 2-hop counts: adj @ (adj.astype(float)) 
        # Actually, we want the size of the union of neighbors of neighbors?
        # Simpler: just use (adj @ adj.sum(axis=1)) / (adj.sum(axis=1)+1e-9)
        # This is the average degree of neighbors.
        deg = adj.sum(axis=1)
        avg_deg_neighbors = np.where(deg > 0, (adj @ deg) / deg, 0)
        return -avg_deg_neighbors
    heuristics.append(h_local_density)
    
    # 6. Max Cut Heuristic (Random signs, then score by "conflict")
    # This is complex, so we'll stick to simpler structural ones.
    
    # 7. Weighted Degree (Degree * Average Neighbor Degree)
    def h_weighted_degree(adj, n):
        deg = adj.sum(axis=1)
        # Sum of degrees of neighbors
        sum_deg_neighbors = adj @ deg
        # Score: negative of (deg * sum_deg_neighbors)
        # Vertices with low degree and low neighbor degree are good
        return -(deg * sum_deg_neighbors).astype(float)
    heuristics.append(h_weighted_degree)
    
    # 8. Coreness (K-core approximation)
    def h_coreness(adj, n):
        # Approximate k-core by iterative removal of low degree
        # We'll just use a few steps of peeling
        core = np.ones(n, dtype=float)
        remaining = np.ones(n, dtype=bool)
        degs = adj.sum(axis=1).copy()
        for _ in range(10):
            if not remaining.any():
                break
            min_deg = degs[remaining].min()
            to_remove = (degs == min_deg) & remaining
            if not to_remove.any():
                break
            # Remove these nodes
            core[to_remove] = min_deg
            remaining[to_remove] = False
            # Update degrees of neighbors
            for v in np.nonzero(to_remove)[0]:
                for u in neighbor_indices[v]:
                    if remaining[u]:
                        degs[u] -= 1
        return -core
    heuristics.append(h_coreness)
    
    num_heuristics = len(heuristics)
    
    # UCB State
    # counts[i] = number of times heuristic i has been evaluated
    # values[i] = sum of greedy sizes obtained
    counts = np.zeros(num_heuristics)
    values = np.zeros(num_heuristics)
    
    best_score = -1
    best_scores_vector = None
    best_order = None
    
    # Time management
    # We will cycle through heuristics, evaluating them.
    # Since heuristics are deterministic (except random), 
    # we don't gain much from re-evaluating the same one.
    # However, we can add "noise" to random heuristics or 
    # use the time to run local search on the best candidate found.
    
    # Phase 1: Evaluate all heuristics at least once
    for i in range(num_heuristics):
        if time.time() > deadline:
            break
        scores = heuristics[i](adj, n)
        order = order_from_scores(scores)
        size = greedy_size_from_order(order)
        counts[i] += 1
        values[i] += size
        
        if size > best_score:
            best_score = size
            best_scores_vector = scores.copy()
            best_order = order.copy()
            
    # Phase 2: Use remaining time to refine the best heuristic via local search
    # or to try more random restarts if the best heuristic is random.
    # Since the task is to return a static score vector, we can't return "random".
    # We must return the specific scores that led to the best size.
    
    # If we have time, try to improve the best_order found so far
    # using a simple 2-opt or 1-swap local search on the order.
    # This is effectively what the seed program did, but now we start
    # from the best of 7 heuristics, not just degree.
    
    if best_order is not None and time.time() < deadline:
        current_order = best_order.copy()
        current_size = best_score
        improved = True
        max_passes = 100
        passes = 0
        
        while improved and passes < max_passes:
            if time.time() > deadline:
                break
            improved = False
            passes += 1
            # Try swapping adjacent elements
            for i in range(n - 1):
                if time.time() > deadline:
                    break
                
                v1 = current_order[i]
                v2 = current_order[i + 1]
                
                # Swap
                current_order[i] = v2
                current_order[i + 1] = v1
                
                new_size = greedy_size_from_order(current_order)
                
                if new_size > current_size:
                    current_size = new_size
                    improved = True
                    # Update best
                    if current_size > best_score:
                        best_score = current_size
                        # We need to map this order back to scores
                        # We'll construct rank-based scores
                        new_scores = np.zeros(n)
                        for rank, v in enumerate(current_order):
                            new_scores[v] = n - rank
                        best_scores_vector = new_scores
                        best_order = current_order.copy()
                else:
                    # Revert
                    current_order[i] = v1
                    current_order[i + 1] = v2
                    
        # If we improved the order, we already updated best_scores_vector.
        # If not, best_scores_vector is from the initial heuristic.
        
    # If no heuristic was evaluated (timeout immediately), return degree
    if best_scores_vector is None:
        return -adj.sum(axis=1).astype(float)
        
    return best_scores_vector.astype(float)
