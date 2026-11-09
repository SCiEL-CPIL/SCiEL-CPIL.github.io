"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.linalg import eigh


def partition(adjacency):
    """
    Solve Maximum Cut using:
    1. Spectral initialization (Fiedler vector)
    2. Fast local search with gain computation
    3. Simulated Annealing with quantum-inspired schedule
    4. Deadline guard to ensure completion within time limit
    """
    start_time = time.time()
    deadline = start_time + 145  # 145s safety margin (limit is 150s)
    
    n = len(adjacency)
    adj = np.asarray(adjacency, dtype=np.int8)
    
    # Helper: compute cut size for a given labeling using vectorized ops
    def cut_size(labels):
        # labels: 0/1 array
        # x = +1 for 0, -1 for 1
        x = 1 - 2 * labels.astype(np.float32)
        xAx = x @ (adj.astype(np.float32) @ x)
        total_edges_twice = adj.sum()
        cut = (total_edges_twice - xAx) / 4.0
        return cut
    
    # Helper: compute gain for flipping each vertex (vectorized)
    def compute_gains(labels):
        """
        For vertex v, gain = (# neighbors with same label) - (# neighbors with opposite label)
        If gain > 0, flipping v improves the cut.
        """
        # labels: 0/1 array
        # For each v, gain[v] = sum_{u} adj[v,u] * (1 if labels[u]==labels[v] else -1)
        # = sum_u adj[v,u] - 2 * sum_u adj[v,u] * (labels[u] != labels[v])
        # = deg[v] - 2 * (number of neighbors with opposite label)
        # Number of neighbors with opposite label for v = sum_u adj[v,u] * (labels[u] != labels[v])
        # labels[u] != labels[v] = labels[u] XOR labels[v] = labels[u] + labels[v] - 2*labels[u]*labels[v]
        # So sum_u adj[v,u] * (labels[u] + labels[v] - 2*labels[u]*labels[v])
        # = (adj @ labels)[v] + labels[v] * deg[v] - 2 * (adj * (labels[:,None] * labels[None,:]))[v].sum(axis=1)
        # Simpler: just compute directly
        # opp_count[v] = sum_u adj[v,u] where labels[u] != labels[v]
        # same_count[v] = deg[v] - opp_count[v]
        # gain[v] = same_count[v] - opp_count[v] = deg[v] - 2 * opp_count[v]
        
        deg = adj.sum(axis=1)
        # opp_count[v] = sum_u adj[v,u] * (1 - (labels[u] == labels[v]))
        # labels[u] == labels[v] is 1 if same, 0 if different
        # So opp_count = adj @ (1 - (labels[None,:] == labels[:,None]).astype(np.int8))
        # This is O(n^2) which is fine for n=100
        
        # More efficient: 
        # opp_count[v] = sum_u adj[v,u] * labels[u] if labels[v]==0, else sum_u adj[v,u] * (1-labels[u])
        # = labels[v] * (sum_u adj[v,u]*(1-labels[u])) + (1-labels[v]) * (sum_u adj[v,u]*labels[u])
        # = labels[v] * (deg[v] - (adj @ labels)[v]) + (1-labels[v]) * (adj @ labels)[v]
        # = labels[v]*deg[v] - labels[v]*(adj@labels)[v] + (adj@labels)[v] - labels[v]*(adj@labels)[v]
        # = labels[v]*deg[v] + (adj@labels)[v] - 2*labels[v]*(adj@labels)[v]
        
        adj_labels = adj @ labels  # (adj @ labels)[v] = sum_u adj[v,u]*labels[u]
        labels_float = labels.astype(np.float32)
        deg_float = deg.astype(np.float32)
        opp_count = labels_float * deg_float + adj_labels - 2 * labels_float * adj_labels
        gain = deg_float - 2 * opp_count
        return gain
    
    # Helper: local search (greedy flip with vectorized gain computation)
    def local_search(labels, max_time=None):
        """Greedy local search: flip vertex if it improves cut size."""
        labels = labels.copy()
        if max_time is None:
            max_time = deadline - 1
        
        improved = True
        while improved:
            if time.time() > max_time:
                break
            improved = False
            gains = compute_gains(labels)
            # Find the vertex with maximum positive gain
            max_gain = gains.max()
            if max_gain > 0:
                v = int(np.argmax(gains))
                labels[v] = 1 - labels[v]
                improved = True
        return labels
    
    # Helper: simulated annealing with quantum-inspired schedule
    def simulated_annealing(labels, t_start, t_end, steps, max_time=None):
        """Simulated annealing to escape local optima."""
        if max_time is None:
            max_time = deadline - 5
        
        labels = labels.copy()
        best_labels = labels.copy()
        best_cut = cut_size(labels)
        current_cut = best_cut
        current_temp = t_start
        cooling_rate = (t_end / t_start) ** (1.0 / max(steps, 1))
        
        # Precompute adj as float for speed
        adj_float = adj.astype(np.float32)
        
        for step in range(steps):
            if time.time() > max_time:
                break
            
            # Pick a random vertex to flip
            v = random.randint(0, n - 1)
            old_label = labels[v]
            labels[v] = 1 - old_label
            
            # Compute new cut size efficiently using the gain
            # delta = gain[v] computed before flip... but we already flipped
            # Let's just compute cut_size directly (fast for n=100)
            new_cut = cut_size(labels)
            delta = new_cut - current_cut
            
            if delta >= 0 or random.random() < np.exp(delta / current_temp):
                current_cut = new_cut
                if new_cut > best_cut:
                    best_cut = new_cut
                    best_labels = labels.copy()
            else:
                # Revert
                labels[v] = old_label
            
            current_temp *= cooling_rate
        
        return best_labels, best_cut
    
    # Step 1: Spectral initialization
    # Compute Fiedler vector of the graph Laplacian
    degrees = adj.sum(axis=1).astype(float)
    L = np.diag(degrees) - adj.astype(float)
    
    try:
        eigenvalues, eigenvectors = eigh(L)
        # Fiedler vector is the eigenvector corresponding to the second smallest eigenvalue
        fiedler = eigenvectors[:, 1]
        labels = (fiedler > 0).astype(np.int8)
    except Exception:
        # Fallback to random initialization
        labels = np.random.randint(0, 2, size=n).astype(np.int8)
    
    # Step 2: Local search to refine
    labels = local_search(labels)
    best_labels = labels.copy()
    best_cut = cut_size(labels)
    
    # Step 3: Simulated Annealing with multiple restarts
    # Use adaptive number of SA steps based on remaining time
    random.seed(42)
    
    # Do several rounds of SA with different initial temperatures
    sa_configs = [
        (10.0, 0.01, 3000),   # high temp, many steps
        (5.0, 0.01, 4000),    # medium temp
        (2.0, 0.005, 5000),   # low temp, more steps
        (1.0, 0.001, 6000),   # very low temp
    ]
    
    for t_start, t_end, steps in sa_configs:
        if time.time() > deadline - 15:
            break
        
        # Start from best known solution
        sa_labels, sa_cut = simulated_annealing(best_labels, t_start, t_end, steps)
        
        if sa_cut > best_cut:
            best_cut = sa_cut
            best_labels = sa_labels.copy()
        
        # Refine with local search
        refined = local_search(best_labels)
        refined_cut = cut_size(refined)
        if refined_cut > best_cut:
            best_cut = refined_cut
            best_labels = refined.copy()
    
    # Step 4: Additional local search passes with random perturbations
    # Try flipping a few random vertices and doing local search
    perturb_attempts = 30
    for _ in range(perturb_attempts):
        if time.time() > deadline - 5:
            break
        
        perturbed = best_labels.copy()
        # Flip 3 random vertices
        for _ in range(3):
            v = random.randint(0, n - 1)
            perturbed[v] = 1 - perturbed[v]
        
        refined = local_search(perturbed)
        refined_cut = cut_size(refined)
        if refined_cut > best_cut:
            best_cut = refined_cut
            best_labels = refined.copy()
    
    # Final local search pass
    final_labels = local_search(best_labels)
    final_cut = cut_size(final_labels)
    if final_cut > best_cut:
        best_labels = final_labels
    
    return best_labels
# EVOLVE-BLOCK-END
