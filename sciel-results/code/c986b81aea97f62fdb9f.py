"""Rewritten program for Maximum Cut using Simulated Annealing."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def partition(adjacency):
    """
    Solve Maximum Cut using Simulated Annealing.
    
    Args:
        adjacency: bool matrix (n x n) representing the graph adjacency
        
    Returns:
        0/1 label per vertex (numpy array of length n)
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=np.int8)
    
    # Convert to efficient representation
    adj = np.asarray(adjacency, dtype=np.int8)
    
    # Precompute degree for fast delta computation
    degrees = adj.sum(axis=1)
    
    def cut_size(labels):
        """Compute the cut size for a given labeling."""
        # Cut size = sum of edges where endpoints have different labels
        # Using: cut = 0.5 * sum_i sum_j A[i,j] * (1 - labels[i]*labels[j]) / 2
        # More efficiently: for each edge (i,j), it's cut if labels[i] != labels[j]
        # We can compute: total_edges - sum over same-side edges
        # Or directly: sum of A[i,j] where labels differ
        # Fast vectorized: 
        # For each node i, count neighbors on opposite side
        # cut = 0.5 * sum_i (degree[i] - same_side_count[i])
        # same_side_count[i] = sum_j A[i,j] * (labels[i] == labels[j])
        
        # Convert labels to +1/-1 for easier computation
        # If we use labels in {0,1}, edge is cut if labels[i] != labels[j]
        # cut = sum_{i<j, A[i,j]=1} [labels[i] != labels[j]]
        
        # Efficient: cut = 0.5 * (sum_i degree[i] - sum_i sum_j A[i,j] * [labels[i]==labels[j]])
        # sum_i sum_j A[i,j] * [labels[i]==labels[j]] = sum_i sum_j A[i,j] * (2*labels[i]*labels[j] - 1) if labels in {-1,1}
        # Let's use labels in {-1, 1} internally
        
        # Actually, let's just compute directly with the 0/1 labels
        # For node i, same-side neighbors: sum_j A[i,j] where labels[j] == labels[i]
        # This is: for label 0 nodes: sum of A[i, j] where labels[j]=0
        # For label 1 nodes: sum of A[i, j] where labels[j]=1
        
        # Vectorized approach:
        # Let L be the label vector (0/1)
        # For each i, same_side[i] = sum_j A[i,j] * (L[j] == L[i])
        # = sum_j A[i,j] * (1 - |L[j] - L[i]|)
        # = sum_j A[i,j] - sum_j A[i,j] * |L[j] - L[i]|
        
        # sum_j A[i,j] * |L[j] - L[i]| is the number of neighbors on the opposite side
        # So cut_size = sum_i (neighbors on opposite side) / 2
        
        # Let's compute: for each i, opp[i] = sum_j A[i,j] * (1 - (L[j] == L[i]))
        # = sum_j A[i,j] - sum_j A[i,j] * (L[j] == L[i])
        
        # Using matrix operations:
        # Let L0 = L (0/1), L1 = 1-L
        # For nodes with label 0: opp = degree - A[:, L==0] @ ones... 
        # Actually simpler: 
        # Create a matrix M where M[i,j] = 1 if L[i] != L[j] else 0
        # M = 1 - (L[:,None] == L[None,:])
        # cut = sum(A * M) / 2
        
        # This is O(n^2) but n=100 so it's fast
        M = (labels[:, None] != labels[None, :]).astype(np.int8)
        return int(np.sum(adj * M) // 2)
    
    def delta_cut(v, labels):
        """
        Compute the change in cut size if we flip vertex v.
        Returns the change (new_cut - old_cut).
        """
        # If we flip v, edges from v to same-side become cut, and edges from v to opposite-side become non-cut
        # Let same = number of neighbors of v on the same side
        # Let opp = number of neighbors of v on the opposite side
        # After flip: same become opp, opp become same
        # Change in cut: edges from v that were non-cut (same-side) become cut: +same
        # Edges from v that were cut (opp-side) become non-cut: -opp
        # But each edge is counted once per endpoint... wait, cut size counts each edge once.
        # When we flip v, only edges incident to v change status.
        # For each edge (v, u):
        #   If u was on same side as v: edge was non-cut, becomes cut: +1
        #   If u was on opposite side from v: edge was cut, becomes non-cut: -1
        # So delta = same - opp where same = neighbors of v on same side, opp = neighbors on opposite side
        # Note: same + opp = degree[v]
        
        lv = labels[v]
        # same_side neighbors: labels[u] == lv
        # opp_side neighbors: labels[u] != lv
        
        # Count using the adjacency row
        row = adj[v]
        same = int(np.sum(row[labels == lv]))
        opp = int(np.sum(row[labels != lv]))
        
        return same - opp
    
    # Spectral initialization using Fiedler vector
    try:
        from scipy.linalg import eigh
        # Build Laplacian L = D - A
        L = np.diag(degrees.astype(float)) - adj.astype(float)
        # Get eigenvalues and eigenvectors
        eigenvalues, eigenvectors = eigh(L)
        # Fiedler vector is the eigenvector corresponding to the second smallest eigenvalue
        # For connected graphs, eigenvalues[0] = 0, eigenvalues[1] is the Fiedler value
        fiedler = eigenvectors[:, 1]
        # Split by sign: nodes with positive fiedler component get label 0, negative get label 1
        labels = (fiedler > 0).astype(np.int8)
    except Exception:
        # Fallback: random initialization
        rng = np.random.default_rng(42)
        labels = rng.integers(0, 2, size=n)
    
    # Ensure we have a valid starting point
    current_cut = cut_size(labels)
    best_labels = labels.copy()
    best_cut = current_cut
    
    # Simulated Annealing parameters
    T_initial = 10.0
    T_final = 0.1
    alpha = 0.995  # Cooling rate
    max_iterations = 50000
    
    rng = np.random.default_rng(12345)
    
    T = T_initial
    iteration = 0
    
    while T > T_final and iteration < max_iterations:
        # Check deadline
        if time.time() > deadline:
            break
        
        # Pick a random vertex to flip
        v = rng.integers(0, n)
        
        # Compute delta
        delta = delta_cut(v, labels)
        
        # Acceptance criterion
        if delta >= 0 or rng.random() < np.exp(delta / T):
            # Accept the move
            labels[v] = 1 - labels[v]
            current_cut += delta
            
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = labels.copy()
        
        # Cool down
        T *= alpha
        iteration += 1
    
    # Final local improvement (greedy hill climbing)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for v in range(n):
            if time.time() > deadline:
                break
            delta = delta_cut(v, best_labels)
            if delta > 0:
                best_labels[v] = 1 - best_labels[v]
                best_cut += delta
                improved = True
    
    return best_labels
# EVOLVE-BLOCK-END
