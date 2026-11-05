"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def partition(adjacency):
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    # Convert to numpy array for fast operations
    A = np.array(adjacency, dtype=np.int8)
    
    # Deadline guard: The evaluator kills after 150s. 
    # We aim to finish well before that to be safe, say 145s.
    start_time = time.time()
    deadline = 145.0
    
    best_labels = None
    best_cut = -1
    
    # Helper to calculate cut size
    def calc_cut_size(labels):
        # labels are 0/1. 
        # Cut size = sum_{i,j} A[i][j] * (labels[i] != labels[j]) / 2 
        # Since A is symmetric and unweighted, and we count edges once.
        # Actually, if we sum A[i][j] * (labels[i] != labels[j]), we count each edge twice.
        # So cut_size = 0.5 * np.sum(A * (labels[:, None] != labels[None, :]))
        # For n=100, this is 10,000 ops. Very fast.
        mask = labels[:, None] != labels[None, :]
        return 0.5 * np.sum(A * mask)
    
    # Spectral Initialization:
    # The Fiedler vector (eigenvector corresponding to the second smallest eigenvalue 
    # of the Laplacian) provides a good initial partition for Max Cut.
    # L = D - A
    degrees = A.sum(axis=1)
    L = np.diag(degrees) - A
    L = sp.csr_matrix(L)
    
    # Find the Fiedler vector. 
    # We want the eigenvector for the smallest non-zero eigenvalue.
    # scipy.sparse.linalg.eigsh can find k smallest eigenvalues.
    # For n=100, this is very fast.
    try:
        # k=2 to get the two smallest. The first is usually 0 (with vector 1).
        # The second is the Fiedler vector.
        vals, vecs = spla.eigsh(L, k=2, which='SM')
        # Sort by eigenvalue just in case, though eigsh with k=2 usually returns sorted
        idx = np.argsort(vals)
        fiedler_vec = vecs[:, idx[1]]
    except Exception:
        # Fallback to random if spectral fails
        fiedler_vec = np.random.randn(n)
    
    # Normalize Fiedler vector to 0/1 labels
    # Threshold at the mean or median
    median_val = np.median(fiedler_vec)
    spectral_labels = (fiedler_vec > median_val).astype(np.int8)
    
    # Ensure both labels are present (if all 0s or all 1s, flip one)
    if np.all(spectral_labels == 0):
        spectral_labels[0] = 1
    elif np.all(spectral_labels == 1):
        spectral_labels[0] = 0
        
    # List of initializations to try
    inits = [spectral_labels.copy()]
    
    # Add some random perturbations of the spectral solution
    # And some pure random starts
    num_inits = 100 # We will generate up to this many, but constrained by time
    
    # We will use a multi-start approach.
    # Each start:
    # 1. Initialize labels (spectral, perturbed spectral, or random)
    # 2. Run a local search (steepest ascent) until no improvement
    
    def local_search(labels):
        # Steepest ascent local search
        # Flip the vertex that gives the maximum increase in cut size.
        # Delta for flipping v: 
        # gain = (neighbors in other set) - (neighbors in same set)
        # We can compute this for all v at once?
        # Let s_v = 1 if label is 1, -1 if label is 0? No, stick to 0/1.
        # Let's use the spin formulation for easier vectorization.
        # x_v = 1 if label 1, -1 if label 0.
        # Cut = 0.25 * sum_{i,j} A[i][j] * (1 - x_i x_j) = const - 0.25 * x^T A x
        # Maximizing Cut is equivalent to minimizing x^T A x.
        # If we flip x_v to -x_v, the change in x^T A x is:
        # Delta (x^T A x) = -4 * x_v * sum_{j != v} A[v][j] x_j
        # So we want to minimize this change, i.e., make it as negative as possible.
        # Equivalent to maximizing x_v * (A @ x)[v]
        
        x = labels.copy()
        x[x == 0] = -1
        x[x == 1] = 1
        
        # Current energy (to be minimized)
        # We don't strictly need the value, just the best labels.
        
        improved = True
        while improved:
            # Check time
            if time.time() > deadline:
                break
            
            # Calculate gain for each vertex
            # Ax = A @ x
            Ax = A @ x
            # Gain for flipping v is proportional to x_v * Ax[v]
            # If x_v * Ax[v] > 0, flipping reduces x^T A x (increases cut)
            gains = x * Ax
            
            # Find the vertex with the maximum positive gain
            max_gain_idx = np.argmax(gains)
            max_gain = gains[max_gain_idx]
            
            if max_gain > 0:
                # Flip the vertex
                x[max_gain_idx] = -x[max_gain_idx]
                improved = True
            else:
                improved = False
        
        # Convert back to 0/1
        labels_out = (x > 0).astype(np.int8)
        return labels_out

    # Multi-Start Loop
    # We will try the spectral init first.
    current_labels = inits[0]
    current_labels = local_search(current_labels)
    current_cut = calc_cut_size(current_labels)
    if current_cut > best_cut:
        best_cut = current_cut
        best_labels = current_labels.copy()
        
    # Now try random and perturbed starts
    # We can generate many starts quickly.
    # Local search is O(n^2 * iterations). For n=100, iterations are usually small.
    # We have ~145s. 
    # Let's estimate time per local search.
    # If it takes ~1ms, we can do 100,000. If 10ms, 10,000.
    
    start_loop_time = time.time()
    
    for i in range(num_inits):
        if time.time() > deadline - 5: # Safety margin
            break
            
        # Decide initialization type
        if i % 3 == 0:
            # Random
            labels = np.random.randint(0, 2, size=n).astype(np.int8)
        elif i % 3 == 1:
            # Perturbed Spectral
            labels = spectral_labels.copy()
            # Flip a few random vertices
            num_flips = np.random.randint(1, 5)
            flip_idx = np.random.choice(n, num_flips, replace=False)
            labels[flip_idx] = 1 - labels[flip_idx]
        else:
            # Random
            labels = np.random.randint(0, 2, size=n).astype(np.int8)
            
        # Run local search
        labels = local_search(labels)
        
        # Evaluate
        cut = calc_cut_size(labels)
        
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
            
    return best_labels
# EVOLVE-BLOCK-END
