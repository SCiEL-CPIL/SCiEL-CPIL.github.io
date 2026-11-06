import numpy as np
import time
from scipy.linalg import eigh

def partition(adjacency):
    start_time = time.time()
    DEADLINE = 145.0  # Leave 5s safety margin for 150s limit
    
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    # Convert to numpy float64 for numerical stability in SDP
    adj = np.array(adjacency, dtype=np.float64)
    # Ensure symmetry
    adj = (adj + adj.T) / 2.0
    # Remove self-loops if any
    np.fill_diagonal(adj, 0)
    
    degrees = adj.sum(axis=1)
    total_edges = 0.5 * adj.sum()
    
    # Helper: compute cut size
    def cut_size(labels):
        s = 2.0 * labels.astype(np.float64) - 1.0
        val = s @ (adj @ s)
        return 0.5 * total_edges - 0.25 * val

    def best_improvement_local_search(labels, max_passes=100):
        """
        Deterministic Best-Improvement Local Search.
        Repeatedly finds the vertex with the maximum positive gain and flips it.
        """
        labels = labels.copy()
        for _ in range(max_passes):
            if time.time() > DEADLINE - 10:
                break
            
            s = 2.0 * labels.astype(np.float64) - 1.0
            # Compute gain for flipping each vertex
            # Gain[v] = 2 * (sum_{u in N(v), s[u] != s[v]} 1) - degree[v]
            # Alternatively:
            # Let A_s = adj * s (elementwise)
            # sum_{u} adj[v,u] s[v] s[u] = s[v] * (adj @ s)[v]
            # This is the contribution of v to s^T A s.
            # If we flip v, s[v] becomes -s[v].
            # New contribution: -s[v] * (adj @ s)[v]
            # Change in s^T A s = -s[v]*(adj@s)[v] - s[v]*(adj@s)[v] = -2*s[v]*(adj@s)[v]
            # We want to maximize cut, which is proportional to -s^T A s.
            # So we want to minimize s^T A s.
            # We flip if change in s^T A s is negative, i.e., -2*s[v]*(adj@s)[v] < 0.
            # Which means s[v]*(adj@s)[v] > 0.
            # Gain in cut size = -0.25 * (change in s^T A s) = 0.5 * s[v]*(adj@s)[v].
            # Wait, let's re-verify.
            # Cut = 0.5 * E - 0.25 * (s^T A s)
            # Let Q = s^T A s.
            # Flip v: s'_v = -s_v.
            # Q' = Q + delta_Q.
            # delta_Q = s'^T A s' - s^T A s
            # = (-s_v) A_v s + s_v A_v (-s_v) ... no, let's use matrix update.
            # Q = sum_{i,j} s_i A_{ij} s_j.
            # Q' = sum_{i,j != v} s_i A_{ij} s_j + sum_{j != v} (-s_v) A_{vj} s_j + sum_{i != v} s_i A_{iv} (-s_v) + (-s_v) A_{vv} (-s_v)
            # Since A_{vv}=0:
            # Q' = Q - 2 * s_v * sum_{j != v} A_{vj} s_j = Q - 2 * s_v * (A s)_v
            # Change in Cut = -0.25 * delta_Q = -0.25 * (-2 * s_v * (A s)_v) = 0.5 * s_v * (A s)_v.
            # So gain = 0.5 * s * (adj @ s).
            
            as_val = adj @ s
            gains = 0.5 * s * as_val
            
            # Find best positive gain
            max_gain = np.max(gains)
            if max_gain <= 1e-9:
                break
            
            # Flip all vertices with max gain? Or just one?
            # Best improvement usually flips one. But flipping all with same max gain is also valid if unique?
            # To be safe, flip the one with max gain. If multiple, flip all?
            # Flipping multiple simultaneously can be risky if they interact.
            # Let's flip the single best vertex.
            idx = np.argmax(gains)
            labels[idx] = 1 - labels[idx]
            
        return labels

    def sdpx_relaxation(adj, n):
        """
        Solve SDP relaxation:
        Maximize 0.5 * sum_{i,j} A_{ij} Y_{ij}
        Subject to:
        Y_{ii} = 1
        Y is PSD
        Y is symmetric
        
        This is equivalent to:
        Maximize 0.5 * <A, Y>
        
        We use projected gradient ascent on the space of symmetric matrices with diagonal 1.
        Actually, standard trick:
        Maximize <A/2, Y>
        s.t. Y_{ii}=1, Y >= 0
        
        Let Y = v v^T approximately.
        The dual problem is Minimize 0.5 * Tr(D) subject to D - A/2 >= 0, D_ii >= 1? No.
        
        Let's stick to primal.
        We want to find Y.
        Initialize Y with random rank-1 matrices or spectral init.
        
        Actually, a simpler approach for the "vector" version (which is what local search works on) is to use the SDP solution to get a good initial vector.
        However, solving full SDP is hard.
        
        Alternative: Use the Fiedler vector of the Laplacian as the primary init, but refine it using a few steps of gradient ascent on the SDP objective projected onto the sphere?
        
        Let's try a different angle: **Power Iteration on the modified Laplacian**.
        The spectral gap gives a good cut.
        
        Let's implement a fast "SDP-like" initialization using the top 2 eigenvectors of the Laplacian, combined with random rotations.
        """
        pass

    # --- Phase 1: Spectral Initialization ---
    # Compute Fiedler vector (2nd smallest eigenvector of Laplacian L = D - A)
    D = np.diag(degrees)
    L = D - adj
    
    try:
        # Compute smallest eigenvalues
        # For n=100, full eigh is fast
        vals, vecs = eigh(L)
        # Fiedler vector is the eigenvector corresponding to the 2nd smallest eigenvalue
        # Note: eigh returns in ascending order
        fiedler = vecs[:, 1]
        # Ensure consistent sign
        if np.sum(fiedler) < 0:
            fiedler = -fiedler
            
        # Initialize labels from Fiedler vector
        # Threshold at 0
        init_labels = (fiedler > 0).astype(int)
        
        # Handle degenerate cases (all 0 or all 1)
        if np.all(init_labels == 0) or np.all(init_labels == 1):
            # Use median
            med = np.median(fiedler)
            init_labels = (fiedler > med).astype(int)
            if np.all(init_labels == 0) or np.all(init_labels == 1):
                init_labels = np.arange(n) % 2
                
    except Exception:
        init_labels = np.arange(n) % 2

    # --- Phase 2: Multi-Start with Random Rotations of Spectral Init ---
    # The SDP solution is often a mixture of spectral vectors.
    # We can simulate this by rotating the Fiedler vector in the subspace spanned by the first few eigenvectors.
    
    best_labels = init_labels
    best_cut = cut_size(init_labels)
    
    # Get first few eigenvectors
    try:
        k = min(5, n)
        _, vecs_k = eigh(L, subset_by_index=[1, k])
        # vecs_k is n x (k-1)
        
        # Create random rotations in this subspace
        rng = np.random.default_rng(42)
        
        # Add the original Fiedler vector (index 0 in vecs_k)
        # Actually, let's just use random linear combinations of the first k eigenvectors
        for i in range(10):
            if time.time() > DEADLINE - 20:
                break
                
            # Random coefficients
            coeffs = rng.normal(size=k-1)
            v = vecs_k @ coeffs
            
            # Project out the mean to avoid trivial solutions?
            # The SDP constraint is Y_ii = 1. The vector v should be normalized?
            # For random rounding, we just need the direction.
            
            # Threshold at 0
            labels = (v > 0).astype(int)
            
            # Ensure both classes non-empty
            if np.all(labels == 0) or np.all(labels == 1):
                med = np.median(v)
                labels = (v > med).astype(int)
                if np.all(labels == 0) or np.all(labels == 1):
                    continue
                    
            # Polish with BILS
            improved = best_improvement_local_search(labels, max_passes=50)
            cut = cut_size(improved)
            
            if cut > best_cut:
                best_cut = cut
                best_labels = improved.copy()
                
    except Exception:
        pass

    # Also polish the original Fiedler init
    improved_fiedler = best_improvement_local_search(init_labels, max_passes=100)
    cut_fiedler = cut_size(improved_fiedler)
    if cut_fiedler > best_cut:
        best_cut = cut_fiedler
        best_labels = improved_fiedler.copy()

    # --- Phase 3: Goemans-Williamson Random Rounding from SDP-like vectors ---
    # If we had the full SDP matrix Y, we would sample x_i = 1 with prob (1 + y_i)/2? No.
    # Standard GW: Sample random hyperplane, assign based on sign of dot product.
    # Here, our "Y" is approximated by the outer product of the spectral vector?
    # Or we can use the random vectors generated above.
    
    # Let's do a few more random restarts with local search to ensure we don't miss a better basin.
    # Since n=100, BILS is very fast.
    
    rng2 = np.random.default_rng(123)
    for i in range(20):
        if time.time() > DEADLINE - 10:
            break
            
        # Random init
        labels = rng2.integers(0, 2, size=n)
        
        # Polish
        improved = best_improvement_local_search(labels, max_passes=50)
        cut = cut_size(improved)
        
        if cut > best_cut:
            best_cut = cut
            best_labels = improved.copy()

    # --- Phase 4: Final Polish ---
    # One last deep BILS on the best found
    final_labels = best_improvement_local_search(best_labels, max_passes=200)
    final_cut = cut_size(final_labels)
    
    if final_cut > best_cut:
        best_labels = final_labels
        best_cut = final_cut

    return best_labels.astype(int)

# Dummy entry point to match signature if needed, but partition is the main one
if __name__ == "__main__":
    # Test with a small graph
    n = 10
    adj = np.zeros((n, n))
    for i in range(n-1):
        adj[i, i+1] = 1
        adj[i+1, i] = 1
    # Add a cycle
    adj[0, n-1] = 1
    adj[n-1, 0] = 1
    
    res = partition(adj)
    print(res)
    print(cut_size(res) if 'cut_size' in locals() else "N/A")
