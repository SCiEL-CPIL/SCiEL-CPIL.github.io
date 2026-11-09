"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def partition(adjacency):
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=np.int8)
    
    adjacency = np.asarray(adjacency, dtype=np.int8)
    
    # Helper to calculate cut size
    def get_cut_size(labels):
        diff = (labels[:, None] != labels[None, :])
        return np.sum(adjacency * diff) // 2
        
    def refine(labels):
        """Greedy local search: flip vertex if it improves cut."""
        current_score = get_cut_size(labels)
        improved = True
        while improved:
            improved = False
            # Shuffle order to avoid bias? No, standard scan is fine for greedy.
            for v in range(n):
                same_mask = labels == labels[v]
                opp_mask = labels != labels[v]
                
                same_count = adjacency[v, same_mask].sum()
                opp_count = adjacency[v, opp_mask].sum()
                
                delta = same_count - opp_count
                
                if delta > 0:
                    labels[v] = 1 - labels[v]
                    current_score += delta
                    improved = True
        return labels, current_score

    best_labels = np.zeros(n, dtype=np.int8)
    best_score = -1
    
    # Precompute Laplacian and Fiedler vector once? 
    # The graph is fixed, so the spectral structure is fixed.
    # We can compute it once outside the loop.
    degrees = adjacency.sum(axis=1)
    D = np.diag(degrees)
    L = D - adjacency
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    fiedler_vector = eigenvectors[:, 1]
    base_labels = np.sign(fiedler_vector).astype(np.int8)
    base_labels[base_labels == 0] = 0
    
    num_restarts = 50 # More restarts to compensate for greedy getting stuck
    
    for r in range(num_restarts):
        # Start from spectral solution
        labels = base_labels.copy()
        
        # Perturb: flip a random subset of vertices
        # Flip probability 0.1
        flip_mask = np.random.rand(n) < 0.1
        labels[flip_mask] = 1 - labels[flip_mask]
        
        # Refine
        refined_labels, score = refine(labels.copy())
        
        if score > best_score:
            best_score = score
            best_labels = refined_labels.copy()

    return best_labels
# EVOLVE-BLOCK-END
