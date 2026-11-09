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
    2. Simulated Annealing with adaptive cooling
    3. Deadline guard to ensure completion within time limit
    """
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin (limit is 150s)
    
    n = len(adjacency)
    adjacency = np.asarray(adjacency, dtype=np.int8)
    
    # Helper: compute cut size for a given labeling
    def cut_size(labels):
        # labels: 0/1 array
        # Cut size = number of edges (i,j) where labels[i] != labels[j]
        # For undirected graph, each edge counted once in adjacency
        # Cut = sum over i<j where labels differ
        # Efficient: use matrix multiplication
        # labels_as_float: +1 for label 0, -1 for label 1 (or vice versa)
        x = 1 - 2 * labels  # +1 for 0, -1 for 1
        # Cut size = (1/4) * sum_{i,j} A[i,j] * (1 - x[i]*x[j]) / 2 ... 
        # Actually: number of crossing edges = (1/4) * sum_{i != j} A[i,j] (1 - x_i x_j) / 2
        # Let's use: cut = (1/2) * sum_{i,j} A[i,j] * (x_i != x_j)
        # = (1/2) * sum_{i,j} A[i,j] * (1 - x_i * x_j) / 2
        # = (1/4) * (sum_{i,j} A[i,j] - sum_{i,j} A[i,j] * x_i * x_j)
        # sum_{i,j} A[i,j] = 2 * |E| (since undirected, each edge counted twice)
        # sum_{i,j} A[i,j] x_i x_j = x^T A x
        total_edges_twice = adjacency.sum()
        x_float = x.astype(float)
        xAx = x_float @ adjacency.astype(float) @ x_float
        cut = (total_edges_twice - xAx) / 4.0
        return cut
    
    # Helper: local search (greedy flip)
    def local_search(labels, max_iters=1000):
        """Greedy local search: flip vertex if it improves cut size."""
        labels = labels.copy()
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            for v in range(n):
                # Compute gain from flipping v
                # Current contribution: edges from v to same-side neighbors are NOT in cut
                # After flip: those edges become in cut, and edges to other-side become not in cut
                same_side = (labels == labels[v])
                other_side = (labels != labels[v])
                same_count = adjacency[v][same_side].sum()
                other_count = adjacency[v][other_side].sum()
                # Flipping v: gain = same_count - other_count
                # (same_count edges move into cut, other_count move out of cut)
                gain = same_count - other_count
                if gain > 0:
                    labels[v] = 1 - labels[v]
                    improved = True
        return labels
    
    # Helper: simulated annealing step
    def simulated_annealing(labels, t_start, t_end, steps):
        """Simulated annealing to escape local optima."""
        labels = labels.copy()
        best_labels = labels.copy()
        best_cut = cut_size(labels)
        current_cut = best_cut
        current_temp = t_start
        cooling_rate = (t_end / t_start) ** (1.0 / max(steps, 1))
        
        for step in range(steps):
            # Pick a random vertex to flip
            v = random.randint(0, n - 1)
            old_label = labels[v]
            labels[v] = 1 - old_label
            
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
            
            # Check deadline periodically
            if step % 100 == 0 and time.time() > deadline - 5:
                break
        
        return best_labels, best_cut
    
    # Step 1: Spectral initialization
    # Compute Fiedler vector of the graph Laplacian
    degrees = adjacency.sum(axis=1).astype(float)
    L = np.diag(degrees) - adjacency.astype(float)
    
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
        (10.0, 0.01, 2000),   # high temp, many steps
        (5.0, 0.01, 3000),    # medium temp
        (2.0, 0.005, 4000),   # low temp, more steps
        (1.0, 0.001, 5000),   # very low temp
    ]
    
    for t_start, t_end, steps in sa_configs:
        if time.time() > deadline - 10:
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
    perturb_attempts = 20
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
