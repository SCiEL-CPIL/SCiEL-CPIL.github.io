"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial.distance import cdist

def partition(adjacency):
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    A = np.array(adjacency, dtype=float)
    # Ensure symmetry
    A = (A + A.T) / 2.0
    
    start_time = time.time()
    deadline = 145.0
    
    # 1. Compute degrees for the SDP diagonal constraint
    # In Max Cut SDP, we usually fix x_ii = 1.
    # The objective is 1/4 * sum_{i,j} A_{ij} (1 - x_i^T x_j)
    # To use standard SDP solvers (like SCS via scipy.optimize), 
    # we can solve the dual or use a specific formulation.
    # However, scipy.optimize.minimize doesn't handle SDPs natively.
    # We need a different strategy for "Vectorized SDP" if scipy.optimize isn't enough.
    # Alternative: Use the fact that for n=100, we can run multiple instances of 
    # a fast heuristic that mimics the SDP outcome or use a simpler convex relaxation.
    
    # Actually, implementing a full SDP solver in Python stdlib+numpy+scipy without 
    # a dedicated CVXPY-like library is hard. 
    # Let's pivot to a different "Structural" change that is implementable:
    # **Parallel Multi-Start Local Search with Nesterov Acceleration / Momentum**
    # OR
    # **Goemans-Williamson Rounding on a Randomized Spectral Initialization**
    
    # Let's try: Spectral Partitioning (Fiedler vector) + Rounding + Local Search Polish.
    # Spectral methods are global in nature (eigenvalue problem) unlike local SA.
    
    # 1. Compute Laplacian L = D - A
    D = np.diag(A.sum(axis=1))
    L = D - A
    
    # 2. Compute Fiedler vector (2nd smallest eigenvalue vector)
    # For n=100, dense eigendecomposition is fast.
    try:
        # We want the eigenvector corresponding to the smallest non-zero eigenvalue.
        # Since L is symmetric, we can use eigh.
        vals, vecs = np.linalg.eigh(L)
        # Sort eigenvalues
        idx = np.argsort(vals)
        # The first eigenvalue is 0 (constant vector). The second is the Fiedler value.
        fiedler_vec = vecs[:, idx[1]]
    except:
        # Fallback if eig fails
        fiedler_vec = np.random.randn(n)
        fiedler_vec -= np.mean(fiedler_vec)
        
    # 3. Initial partition based on Fiedler vector
    # Split by sign
    initial_labels = (fiedler_vec > 0).astype(int)
    
    # 4. Define a fast local search (Flip)
    def calc_cut(labels):
        # labels: 0/1 array
        # Cut = sum(A * (labels[:, None] != labels[None, :])) / 2
        # Using matrix multiplication for speed:
        # Let x be +1/-1 vector.
        # Cut = (n^2 - sum(x_i x_j A_ij)) / 4 ? 
        # Simpler: 
        mask = labels[:, None] != labels[None, :]
        return np.sum(A * mask) / 2.0
        
    def flip_gain(v, labels):
        # Gain from flipping v
        # neighbors_same = sum(A[v][labels == labels[v]])
        # neighbors_other = sum(A[v][labels != labels[v]])
        # delta = neighbors_other - neighbors_same
        # Note: A[v] includes self-loop? Usually no.
        same_mask = (labels == labels[v])
        # Exclude self from sum? A[v,v] is usually 0.
        sum_same = np.dot(A[v], same_mask)
        total_deg = A[v].sum()
        sum_other = total_deg - sum_same
        return sum_other - sum_same

    best_labels = initial_labels.copy()
    best_cut = calc_cut(best_labels)
    
    # 5. Multi-start with Spectral + Random perturbations
    # We will run multiple "instances" of local search starting from slightly different spectral seeds
    # to escape local minima.
    
    num_restarts = 20
    labels_list = []
    
    # Generate diverse initializations
    # 1. Pure Fiedler
    labels_list.append(initial_labels.copy())
    
    # 2. Fiedler + Noise
    for i in range(num_restarts - 1):
        noise = np.random.randn(n) * 0.1
        perturbed = fiedler_vec + noise
        labels = (perturbed > 0).astype(int)
        labels_list.append(labels)
        
    # 3. Random initializations
    for i in range(10):
        labels = np.random.randint(0, 2, size=n)
        labels_list.append(labels)

    # Process each restart
    for init_labels in labels_list:
        if time.time() > deadline - 10: # Safety margin
            break
            
        labels = init_labels.copy()
        current_cut = calc_cut(labels)
        
        # Local Search: Greedy Flip until convergence
        # To speed up, we can do multiple sweeps or just random flips until no improvement.
        # Given n=100, a full sweep is O(n^2). 
        # We will do a "Best Improvement" or "First Improvement" local search.
        # First improvement is faster per step.
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Shuffle vertex order to avoid bias
            order = np.random.permutation(n)
            for v in order:
                gain = flip_gain(v, labels)
                if gain > 0:
                    labels[v] = 1 - labels[v]
                    current_cut += gain
                    improved = True
                    # Update best if needed
                    if current_cut > best_cut:
                        best_cut = current_cut
                        best_labels = labels.copy()
                    # Early break if we've done a lot of work in this restart?
                    # No, let it converge. It's fast for n=100.
        
        if current_cut > best_cut:
            best_cut = current_cut
            best_labels = labels.copy()

    # 6. Final Polish: Simulated Annealing on the best found so far
    # Use remaining time for SA to potentially jump out of the local optimum found by LS.
    labels = best_labels.copy()
    current_cut = best_cut
    T = 5.0
    alpha = 0.95
    
    while time.time() < deadline - 2:
        v = np.random.randint(0, n)
        gain = flip_gain(v, labels)
        
        if gain > 0:
            accept = True
        else:
            if T > 0.1:
                p = np.exp(gain / T)
                accept = np.random.rand() < p
            else:
                accept = False
        
        if accept:
            labels[v] = 1 - labels[v]
            current_cut += gain
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = labels.copy()
        
        T *= alpha

    return best_labels
# EVOLVE-BLOCK-END
