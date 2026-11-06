"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _cut_size(adj, labels):
    """Compute the number of edges crossing the partition."""
    # A cut edge is one where endpoints have different labels.
    # Using matrix multiplication: (labels @ adj @ (1 - labels)) gives sum over edges
    # Actually, let's use a vectorized approach.
    # For each edge (i,j) where adj[i][j] is True, it's cut if labels[i] != labels[j].
    # We can compute: for each row i, sum adj[i] where labels differ.
    # Vectorized: 
    # cut = 0
    # For each i: cut += adj[i, labels != labels[i]].sum()
    # This is O(n^2) per evaluation which is fine for n=100.
    
    # More efficient vectorized approach:
    # The number of cut edges = 0.5 * sum_i sum_j adj[i][j] * (labels[i] != labels[j])
    # But since adj is symmetric and unweighted, and labels are 0/1:
    # labels[i] != labels[j] is equivalent to 1 - labels[i]*labels[j] - (1-labels[i])*(1-labels[j])
    # = 1 - 2*labels[i]*labels[j] + labels[i] + labels[j] - 1 ... let me think differently.
    
    # Simpler: convert labels to +1/-1
    # Let s_i = 2*labels[i] - 1, so s_i is +1 or -1.
    # An edge (i,j) is cut if s_i != s_j, i.e., s_i * s_j = -1.
    # Number of cut edges = 0.5 * sum_{i != j} adj[i][j] * (1 - s_i * s_j) / 2 ... 
    # Actually: indicator that edge is cut = (1 - s_i * s_j) / 2
    # So cut size = 0.5 * sum_{i,j} adj[i][j] * (1 - s_i * s_j) / 2 ... no.
    # sum over all pairs (i,j) of adj[i][j] * indicator(cut)
    # = sum_{i,j} adj[i][j] * (1 - s_i s_j) / 2
    # = (1/2) * [sum_{i,j} adj[i][j] - sum_{i,j} adj[i][j] s_i s_j]
    # = (1/2) * [2 * E - s^T A s] where E is total edges, s is vector of s_i
    # Since graph is undirected, sum_{i,j} adj[i][j] = 2*E (each edge counted twice)
    # So cut = (1/2) * [2E - s^T A s] = E - 0.5 * s^T A s
    
    # Let me just use the direct approach for clarity and correctness:
    n = len(labels)
    # labels is 0/1 array
    # For each vertex i, count neighbors with different label
    # cut = sum_i (number of neighbors of i with label != labels[i]) / 2
    # because each cut edge is counted twice (once from each endpoint)
    
    # Vectorized: 
    # diff = labels[:, None] != labels[None, :]  # n x n boolean
    # cut = (adj & diff).sum() // 2
    # This creates an n x n array which is fine for n=100.
    
    diff = (labels[:, None] != labels[None, :])
    cut = (adj & diff).sum() // 2
    return cut


def _greedy_improve(adj, labels, max_passes=10):
    """Greedy local search: flip vertices that improve the cut."""
    n = len(labels)
    labels = labels.copy()
    current_cut = _cut_size(adj, labels)
    
    for _ in range(max_passes):
        improved = False
        for v in range(n):
            # If we flip v, how does the cut change?
            # Currently, edges from v to same-label neighbors are NOT cut.
            # Edges from v to different-label neighbors ARE cut.
            # After flip: edges to same-label become cut, edges to diff-label become not cut.
            # Delta = (same_count - diff_count)
            # Where same_count = number of neighbors with same label as v
            #       diff_count = number of neighbors with different label as v
            
            same_mask = (labels == labels[v])
            diff_mask = (labels != labels[v])
            same_count = adj[v, same_mask].sum() - adj[v, v]  # exclude self if present
            # Actually adj[v,v] should be 0 for simple graphs, but let's be safe
            diff_count = adj[v, diff_mask].sum()
            
            # If we flip, new cut contribution from v's edges:
            # Previously cut edges from v: diff_count
            # Previously non-cut edges from v: same_count
            # After flip: cut edges from v: same_count, non-cut: diff_count
            # Net change in total cut (counting each edge once): 
            # The edges incident to v change: 
            # Old contribution to cut from these edges: diff_count (but each edge counted once globally)
            # Wait, _cut_size counts each cut edge once (divides by 2).
            # Let me think in terms of the full sum before dividing by 2.
            # Let S = sum_{i,j} adj[i][j] * [labels[i] != labels[j]]
            # cut = S / 2
            # When we flip vertex v:
            # For each neighbor u: 
            #   if labels[u] == labels[v], then [labels[u] != labels[v]] was 0, becomes 1
            #   if labels[u] != labels[v], then [labels[u] != labels[v]] was 1, becomes 0
            # Also, the diagonal term adj[v][v] * [labels[v] != labels[v]] = 0 always.
            # So delta_S = sum_{u: labels[u]==labels[v]} adj[v][u] - sum_{u: labels[u]!=labels[v]} adj[v][u]
            # = same_count - diff_count
            
            delta = same_count - diff_count
            if delta > 0:
                labels[v] = 1 - labels[v]
                current_cut += delta // 2  # approximate, will recompute at end
                improved = True
        
        if not improved:
            break
    
    # Recompute exact cut size
    final_cut = _cut_size(adj, labels)
    return labels, final_cut


def _random_restart(adj, n, rng):
    """Generate a random initial partition."""
    labels = rng.integers(0, 2, size=n)
    return labels


def partition(adjacency):
    """
    Solve Maximum Cut using multi-start greedy local search.
    
    Strategy:
    1. Convert adjacency to numpy boolean matrix.
    2. Run multiple random restarts with greedy local search.
    3. Keep the best solution found within the time budget.
    """
    start_time = time.time()
    DEADLINE = 115.0  # seconds, leave 5s safety margin
    
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    # Convert to numpy bool matrix for efficient operations
    adj = np.array(adjacency, dtype=bool)
    
    # Ensure symmetry (in case input isn't perfectly symmetric)
    adj = adj | adj.T
    
    best_labels = None
    best_cut = -1
    
    rng = np.random.default_rng(42)
    
    # First, try the alternating start (deterministic)
    labels_alt = np.arange(n) % 2
    labels_alt, cut_alt = _greedy_improve(adj, labels_alt, max_passes=20)
    if cut_alt > best_cut:
        best_cut = cut_alt
        best_labels = labels_alt.copy()
    
    # Then, do random restarts
    restart_count = 0
    while time.time() < DEADLINE:
        # Generate random initial labels
        labels = rng.integers(0, 2, size=n)
        
        # Run greedy improvement
        labels, cut = _greedy_improve(adj, labels, max_passes=20)
        
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
        
        restart_count += 1
        
        # Check time
        if time.time() >= DEADLINE:
            break
    
    if best_labels is None:
        # Fallback: return alternating labels
        return np.arange(n) % 2
    
    return best_labels
# EVOLVE-BLOCK-END
