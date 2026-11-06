"""Maximum Cut via Simulated Annealing."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def partition(adjacency):
    """
    Assign 0/1 labels to vertices to maximize the number of cut edges.
    Uses Simulated Annealing with a time-adaptive schedule to escape local optima.
    """
    adj = np.asarray(adjacency, dtype=np.int8)
    n = len(adj)
    
    if n <= 1:
        return np.zeros(n, dtype=np.int8)
    
    # Precompute degrees for O(1) gain calculation
    # If vertex v is in group g, its contribution to cut is sum(adj[v, ~g])
    # gain(v) = sum(adj[v, ~labels]) - sum(adj[v, labels])
    #          = (2 * sum(adj[v, ~labels])) - degree[v]
    # If we flip v, new contribution is degree[v] - old_contribution
    # Change in cut size = new - old = degree[v] - (2*sum_adj_~l - deg[v]) 
    #                    = 2*deg[v] - 2*sum_adj_~l = 2 * (deg[v] - sum_adj_~l)
    #                    = 2 * sum_adj_l
    # So, flipping a vertex changes the cut size by 2 * (number of neighbors in same group)
    # Wait, let's re-verify.
    # Cut = sum_{u<v} (label[u] != label[v])
    # If we flip v, edges (v, u) change status.
    # If label[u] == label[v] (before), edge was NOT in cut. After flip, label[u] != label[v], edge IS in cut. Gain +1.
    # If label[u] != label[v] (before), edge WAS in cut. After flip, label[u] == label[v], edge NOT in cut. Gain -1.
    # Total gain = (# neighbors in same group) - (# neighbors in other group)
    # Let k_same = sum(adj[v, labels == labels[v]])
    # Let k_other = degree[v] - k_same
    # Gain = k_same - k_other = 2*k_same - degree[v]
    
    degrees = adj.sum(axis=1)
    
    # Deadline: 150s kill, use 145s for safety, leave 5s for overhead/return
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Initial state: random
    best_labels = None
    best_cut = -1
    current_labels = np.random.randint(0, 2, size=n)
    
    # Compute initial cut
    # Cut = 0.5 * sum_v (degree[v] - 2 * sum(adj[v, labels==labels[v]])) ? No.
    # Easier: Cut = 0.5 * sum_{u,v} (1 - label[u]*label[v] if we map to +/-1)
    # Let's just compute directly or update incrementally.
    # Direct computation for initial state:
    # cut = 0
    # for u in range(n):
    #   for v in range(u+1, n):
    #     if adj[u,v] and labels[u] != labels[v]: cut += 1
    # Vectorized:
    # Map labels to +/-1: s = 2*labels - 1
    # Cut = 0.25 * sum_{u,v} (1 - s[u]*s[v]) * adj[u,v] ?
    # (1 - s_u s_v) is 0 if same, 2 if diff.
    # Sum over all u,v counts each edge twice.
    # So Cut = 0.25 * sum_{u,v} (1 - s_u s_v) adj[u,v]
    #        = 0.25 * (sum_{u,v} adj[u,v] - sum_{u,v} s_u s_v adj[u,v])
    #        = 0.25 * (2*E - s^T A s)
    #        = 0.5 * E - 0.25 * s^T A s
    
    def compute_cut(labels):
        s = 2 * labels.astype(np.int8) - 1
        # s^T A s
        # A is symmetric.
        # s^T A s = sum_i s_i (A s)_i
        # As = adj @ s
        # val = s @ (adj @ s)
        return 0.5 * (degrees.sum() / 2) - 0.25 * (s @ (adj @ s))
    
    # Wait, degrees.sum() / 2 is E.
    # Let's stick to a simpler incremental update to be safe and fast.
    
    # State:
    # labels: int8 array of 0/1
    # cut_size: int
    
    current_labels = np.random.randint(0, 2, size=n)
    current_cut = int(compute_cut(current_labels))
    
    best_labels = current_labels.copy()
    best_cut = current_cut
    
    # Simulated Annealing Parameters
    # We need to estimate the number of steps we can take.
    # Let's run until deadline.
    # Cooling schedule: T = T0 * (T_end / T0)^(iter / max_iter) is hard without max_iter.
    # Use T = T0 * exp(-alpha * time_elapsed)
    # Or simply T = T0 * (1 - time_fraction)^k
    
    T0 = 10.0  # Initial temperature
    T_min = 1e-6
    
    # We will run steps in a loop, checking time periodically.
    # To make it faster, we can do a batch of steps then check time.
    
    while time.time() < deadline:
        # Adjust temperature based on elapsed time
        elapsed = time.time() - start_time
        # Assume we want to reach T_min around 140s
        # T = T0 * exp(-lambda * elapsed)
        # T_min = T0 * exp(-lambda * 140)
        # lambda = ln(T0/T_min) / 140
        # Let's use a simpler linear decay in log-space or just a factor.
        # For simplicity, let's use a fixed number of steps per "phase" and decay T.
        
        # Actually, a robust way is to track steps and decay T per step, but we don't know max steps.
        # Let's use time-based decay.
        # T = T0 * (T_min/T0)^(elapsed / 140.0)
        T = T0 * (T_min / T0) ** (elapsed / 140.0)
        
        # Pick a random vertex
        v = np.random.randint(0, n)
        
        # Calculate gain of flipping v
        # gain = 2 * k_same - degree[v]
        # k_same = sum(adj[v, labels == labels[v]])
        # We can compute this quickly.
        # labels == labels[v] is a boolean mask.
        # sum(adj[v, mask])
        
        # Optimization: 
        # If we maintain an array `same_count` for each vertex, we can update it incrementally.
        # But for now, direct computation is O(n) per step.
        # n=100, so O(100) per step. 
        # 145s / 100 ops is very fast. We can do millions of steps.
        
        k_same = adj[v][current_labels == current_labels[v]].sum()
        gain = 2 * k_same - degrees[v]
        
        # Acceptance
        if gain > 0:
            accept = True
        else:
            # Metropolis criterion
            # delta = -gain (since gain is negative)
            # accept if exp(gain/T) > random
            if T > T_min:
                r = np.random.random()
                # gain is negative
                if r < np.exp(gain / T):
                    accept = True
                else:
                    accept = False
            else:
                accept = False
        
        if accept:
            current_labels[v] = 1 - current_labels[v]
            current_cut += gain
            
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = current_labels.copy()
        
        # Check time more frequently to avoid overshooting
        if time.time() > deadline - 1.0:
            break

    return best_labels
# EVOLVE-BLOCK-END
