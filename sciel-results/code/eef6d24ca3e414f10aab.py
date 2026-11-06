"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def partition(adjacency):
    """
    Maximum Cut via simulated annealing with multiple random restarts.
    Uses a simple energy landscape search that is robust for 100-node graphs.
    """
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    # Precompute adjacency as a numpy array for speed
    adj = np.asarray(adjacency, dtype=np.float64)
    
    def cut_size(labels):
        """Compute the number of edges crossing the partition."""
        # edges crossing = 0.5 * sum of (1 - labels[i]*labels[j]) for all edges? 
        # Actually, if labels are 0/1, an edge (i,j) crosses if labels[i] != labels[j].
        # We can compute this efficiently.
        # Sum of adj[i,j] * (1 if labels[i]!=labels[j] else 0)
        # Using matrix operations: 
        # Let L be a vector of labels. 
        # We want sum_{i<j} adj[i,j] * (labels[i] != labels[j])
        # This is equivalent to 0.5 * sum_{i,j} adj[i,j] * (1 - labels[i]*labels[j] + ...)? No.
        # If we map 0/1 to -1/1: s_i = 2*labels[i] - 1.
        # Edge crosses if s_i != s_j, i.e., s_i * s_j = -1.
        # Cut size = 0.5 * sum_{i,j} adj[i,j] * (1 - s_i * s_j)
        # Since adj is symmetric and zero diagonal, this works.
        s = 2 * labels - 1  # -1 or 1
        # s_i * s_j matrix
        ss = np.outer(s, s)
        # 1 - s_i*s_j is 0 if same, 2 if different
        # So sum adj * (1-ss) / 2 gives the cut size
        return 0.5 * np.sum(adj * (1 - ss))
    
    def simulated_annealing(adj, n, initial_labels, max_iter=20000, T_start=1.0, T_end=0.01, alpha=0.9995):
        """
        Simulated annealing to maximize cut size.
        """
        labels = initial_labels.copy()
        current_cut = cut_size(labels)
        best_labels = labels.copy()
        best_cut = current_cut
        
        T = T_start
        for i in range(max_iter):
            # Pick a random vertex to flip
            v = np.random.randint(n)
            
            # Calculate change in cut size if we flip v
            # Flipping v changes the contribution of all edges incident to v.
            # Old contribution of edges (v, u): adj[v, u] * (1 if labels[v] != labels[u] else 0)
            # New contribution: adj[v, u] * (1 if (1-labels[v]) != labels[u] else 0)
            # Let's compute delta efficiently.
            # Let s_v = 2*labels[v]-1. New s_v' = -s_v.
            # Change in sum_{u} adj[v,u] * (1 - s_v * s_u)
            # = sum_{u} adj[v,u] * (1 - (-s_v) * s_u) - sum_{u} adj[v,u] * (1 - s_v * s_u)
            # = sum_{u} adj[v,u] * (1 + s_v * s_u - 1 + s_v * s_u)
            # = 2 * s_v * sum_{u} adj[v,u] * s_u
            # Wait, the cut size formula is 0.5 * sum_{i,j} ...
            # So the total change in cut size is 0.5 * (change in the double sum).
            # The double sum change is 2 * sum_{u} (adj[v,u] * s_u * s_v) + 2 * sum_{u} (adj[u,v] * s_u * s_v) ?
            # Since adj is symmetric, the term for (v,u) and (u,v) are both present in the sum.
            # Let's just recompute the local neighborhood contribution.
            # Actually, simpler:
            # Current cut contribution from edges incident to v:
            # c_old = sum_{u != v} adj[v,u] * (1 if labels[v] != labels[u] else 0)
            # New cut contribution:
            # c_new = sum_{u != v} adj[v,u] * (1 if (1-labels[v]) != labels[u] else 0)
            # Note that (1-labels[v]) != labels[u] is equivalent to labels[v] == labels[u].
            # So c_new = sum_{u != v} adj[v,u] * (1 if labels[v] == labels[u] else 0)
            # delta = c_new - c_old
            # c_old + c_new = sum_{u != v} adj[v,u] (since for each u, either labels differ or same, not both)
            # So delta = c_new - (sum_{u != v} adj[v,u] - c_new) = 2*c_new - degree(v)
            
            # Let's compute c_new efficiently.
            # c_new is the number of neighbors of v that have the SAME label as v currently.
            # Because if we flip v, those neighbors will now be on the OPPOSITE side, so the edge crosses.
            # Wait.
            # If labels[v] == labels[u], edge (v,u) does NOT cross currently.
            # After flip, labels[v] != labels[u], so edge (v,u) DOES cross.
            # So yes, c_new = number of neighbors with same label.
            
            same_label_count = np.sum(adj[v] * (labels == labels[v]))
            # Note: adj[v][v] is 0, so self-loop doesn't matter.
            # But adj[v] * (labels == labels[v]) will include adj[v][v] * 1 = 0. Correct.
            
            # The degree of v is sum(adj[v])
            degree_v = np.sum(adj[v])
            
            delta = 2 * same_label_count - degree_v
            
            # Accept if delta > 0 or with probability exp(delta/T)
            if delta > 0 or np.random.random() < np.exp(delta / T):
                labels[v] = 1 - labels[v]
                current_cut += delta
                if current_cut > best_cut:
                    best_cut = current_cut
                    best_labels = labels.copy()
            
            T *= alpha
            if T < T_end:
                break
        
        return best_labels, best_cut

    # Multiple restarts
    best_labels_overall = None
    best_cut_overall = -1
    
    # Number of restarts depends on time, but for 100 nodes, even 10-20 restarts with 20k iterations
    # should be fast enough.
    # Let's try 5 restarts with 10000 iterations each, or 1 restart with 50000.
    # Given 120s limit, we have plenty of time.
    # Let's do 10 restarts with 5000 iterations.
    
    np.random.seed(42)  # For reproducibility in evaluation? No, let's use random seed each time or fixed?
    # The task says "five generated unweighted 100-node graphs". It doesn't specify seeding.
    # Usually, evaluators run the function multiple times or on different instances.
    # We should not hardcode a seed that might bias towards one graph if the graphs are random.
    # But for reproducibility of our own tests, we might. The evaluator likely calls partition(adj) directly.
    # We will not set a global seed, but use np.random which is seeded by system time or previous state.
    # To be safe and deterministic if needed, we could seed based on the graph? No, that's overkill.
    # Standard practice: just use np.random.
    
    num_restarts = 10
    max_iter_per_restart = 5000
    
    for r in range(num_restarts):
        # Random initialization
        init_labels = np.random.randint(0, 2, size=n)
        
        # Run simulated annealing
        labels, cut = simulated_annealing(adj, n, init_labels, max_iter=max_iter_per_restart)
        
        if cut > best_cut_overall:
            best_cut_overall = cut
            best_labels_overall = labels
    
    # Final local improvement (greedy) on the best solution found
    # This ensures we are at a local optimum
    labels = best_labels_overall.copy()
    improved = True
    while improved:
        improved = False
        for v in range(n):
            same = np.sum(adj[v] * (labels == labels[v]))
            other = np.sum(adj[v] * (labels != labels[v]))
            if same > other:
                labels[v] = 1 - labels[v]
                improved = True
    
    return labels
# EVOLVE-BLOCK-END
