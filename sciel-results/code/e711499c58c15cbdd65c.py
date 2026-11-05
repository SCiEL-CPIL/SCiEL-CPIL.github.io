"""Program for Maximum Independent Set via greedy simulation."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(adjacency):
    """
    Assign priority scores to vertices by simulating a greedy Maximum Independent Set
    construction. Vertices are picked in an order that minimizes the number of
    blocked neighbors. Scores are assigned to ensure the greedy decoder replicates
    this picking order.
    
    Higher score = selected earlier in greedy decoder.
    """
    n = adjacency.shape[0]
    if n == 0:
        return np.array([], dtype=float)
    
    # Work with integer adjacency for efficiency
    adj = adjacency.astype(np.int32)
    
    # remaining: vertices not yet processed (neither picked nor blocked)
    remaining = np.ones(n, dtype=bool)
    # active_neighbors: number of neighbors in the 'remaining' set
    active_neighbors = adj.sum(axis=1).astype(np.int32)
    
    scores = np.zeros(n, dtype=float)
    # We use a large offset to separate picked and blocked scores
    # Picked scores will be positive and decreasing.
    # Blocked scores will be negative and decreasing (in magnitude) or just very low.
    
    step = 0
    large_const = n + 1
    
    while remaining.any():
        # Find vertices that are remaining and have 0 active neighbors
        eligible = remaining & (active_neighbors == 0)
        
        if eligible.any():
            # Pick all eligible vertices. They form an independent set.
            picked = eligible.copy()
            # Identify neighbors of picked vertices that are still remaining
            # These will be blocked
            # We need to be careful: if a vertex is picked, it's not blocked.
            # Its neighbors are blocked.
            
            # Get indices of picked
            picked_indices = np.where(picked)[0]
            
            # Find neighbors of picked vertices
            # adj[picked_indices, :] gives rows for picked vertices
            # Sum over picked vertices to get count of picked neighbors for each vertex
            # But we just need a boolean mask of "is neighbor of any picked"
            # We can use matrix multiplication or logical OR
            # Since n is small (200), we can do:
            # neighbor_mask = np.any(adj[picked_indices, :], axis=0)
            # But we only care about 'remaining' vertices.
            
            # Update remaining:
            # 1. Remove picked vertices
            # 2. Remove neighbors of picked vertices
            
            # Create a mask of all vertices to remove
            to_remove = np.zeros(n, dtype=bool)
            to_remove[picked] = True
            
            # Add neighbors of picked
            # adj is bool/int. 
            # neighbors_of_picked = adj[picked_indices, :].any(axis=0)
            # This creates a bool array of length n
            neighbors_mask = np.zeros(n, dtype=bool)
            if len(picked_indices) > 0:
                # Efficiently compute which vertices have at least one neighbor in picked_indices
                # Using matrix multiplication with ones is fast
                # Or just loop if small? n=200 is small.
                # Let's use numpy broadcasting or reduction
                # adj[picked_indices, :] is (k, n). 
                # We want column-wise any.
                neighbors_mask = np.any(adj[picked_indices, :] != 0, axis=0)
            
            to_remove |= neighbors_mask
            
            # Assign scores
            # Picked vertices get high score for this step
            # Blocked vertices (neighbors) get low score
            # We want picked > blocked in this step.
            # And picked in step i > picked in step i+1.
            
            # Score for picked: (n - step) * large_const - index
            # Score for blocked: -(n - step) * large_const - index  (Very low)
            
            # Actually, simpler:
            # Picked score base: n - step
            # Blocked score base: - (n - step)
            
            picked_score_base = (n - step) * large_const
            blocked_score_base = - (n - step) * large_const
            
            scores[picked] = picked_score_base - np.arange(n, dtype=float)[picked]
            
            # Blocked vertices are those in 'to_remove' that are NOT picked
            blocked = to_remove & ~picked
            if blocked.any():
                scores[blocked] = blocked_score_base - np.arange(n, dtype=float)[blocked]
            
            # Update remaining
            remaining[to_remove] = False
            
            # Update active_neighbors
            # We need to decrement active_neighbors for all vertices
            # that have neighbors removed.
            # The vertices removed are 'to_remove'.
            # For any vertex u, active_neighbors[u] decreases by the number of its neighbors in 'to_remove'.
            # This is the column sum of adj[:, to_remove]
            # Or row sum of adj[to_remove, :] for the removed vertices?
            # We need to update the counts for the *remaining* vertices.
            # The count of active neighbors for vertex u is sum(adj[u, remaining]).
            # If we remove a set R, the new count is sum(adj[u, remaining \ R]).
            # Change is - sum(adj[u, R]).
            
            # sum(adj[u, R]) for all u is the matrix product adj @ R_vector?
            # R_vector is 1 for removed, 0 otherwise.
            # adj (n x n) @ R_vector (n,) -> (n,)
            # This gives for each u, the number of neighbors in R.
            
            remove_vector = to_remove.astype(np.int32)
            neighbor_counts_removed = adj.dot(remove_vector)
            
            # Only update active_neighbors for vertices that are still remaining?
            # Or just update all, and rely on 'remaining' mask later?
            # active_neighbors is used to check `active_neighbors == 0`.
            # If a vertex is not remaining, it's ignored by `eligible = remaining & ...`.
            # So we can just update all.
            active_neighbors -= neighbor_counts_removed
            
        else:
            # No eligible vertices (0 active neighbors).
            # Pick the vertex with the minimum active_neighbors.
            # Tie-break by index.
            remaining_indices = np.where(remaining)[0]
            if len(remaining_indices) == 0:
                break
            
            min_neighbors = active_neighbors[remaining_indices].min()
            candidates = remaining_indices[active_neighbors[remaining_indices] == min_neighbors]
            chosen = candidates[0]
            
            picked = np.zeros(n, dtype=bool)
            picked[chosen] = True
            picked_indices = [chosen]
            
            # Find neighbors
            neighbors_mask = adj[chosen, :] != 0
            
            to_remove = np.zeros(n, dtype=bool)
            to_remove[picked] = True
            to_remove |= neighbors_mask
            
            # Assign scores
            picked_score_base = (n - step) * large_const
            blocked_score_base = - (n - step) * large_const
            
            scores[picked] = picked_score_base - np.arange(n, dtype=float)[picked]
            
            blocked = to_remove & ~picked
            if blocked.any():
                scores[blocked] = blocked_score_base - np.arange(n, dtype=float)[blocked]
            
            # Update remaining
            remaining[to_remove] = False
            
            # Update active_neighbors
            remove_vector = to_remove.astype(np.int32)
            neighbor_counts_removed = adj.dot(remove_vector)
            active_neighbors -= neighbor_counts_removed
            
        step += 1
    
    return scores
# EVOLVE-BLOCK-END
