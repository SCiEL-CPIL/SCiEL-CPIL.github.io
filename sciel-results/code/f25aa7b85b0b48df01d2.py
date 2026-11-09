import numpy as np
import time

def priority(adjacency):
    """
    Generates vertex scores to guide a greedy decoder to find a large independent set.
    The strategy is to simulate a 'minimum degree' greedy heuristic on the input graph
    and assign higher scores to vertices that would be picked earlier by this heuristic.
    """
    start_time = time.time()
    deadline = 150.0  # Safety margin for the 150s limit, though this should be fast.
    
    # Ensure adjacency is a boolean numpy array
    adj = np.asarray(adjacency, dtype=bool)
    n = adj.shape[0]
    
    # scores will be assigned. Higher is better.
    scores = np.zeros(n, dtype=float)
    
    # available: boolean mask of vertices still in the "game"
    available = np.ones(n, dtype=bool)
    
    # We will assign scores in descending order of "pick order".
    # Step 1 picked gets highest score, Step 2 gets next, etc.
    # Vertices removed as neighbors get low scores.
    
    current_max_score = float(n)
    step = 0
    
    while np.any(available):
        if time.time() > deadline - 1.0:
            # Break if close to deadline, return current scores
            break
            
        # Calculate degrees of available vertices within the available subgraph
        # adj[available] is the submatrix of rows for available vertices
        # We need column sums of that submatrix, restricted to available columns
        # deg[i] = sum(adj[i, j] for j in available)
        
        # Efficient calculation:
        # sub_adj = adj[np.ix_(available, available)]
        # degs = sub_adj.sum(axis=1)
        
        # But we need degrees for all available vertices, indexed by original index.
        # Let's get the indices of available vertices
        avail_indices = np.where(available)[0]
        
        if len(avail_indices) == 0:
            break
            
        # Compute degrees for these vertices
        # adj[avail_indices, :] gives rows for available vertices
        # We only care about columns that are available
        # So we sum over the columns in avail_indices
        degs = adj[avail_indices, :][:, avail_indices].sum(axis=1)
        
        # Find the index in avail_indices with the minimum degree
        min_idx_local = np.argmin(degs)
        # In case of ties, argmin returns the first one. 
        # To be deterministic and potentially better, we can just take the first min.
        
        v = avail_indices[min_idx_local]
        
        # Assign score to v
        # We want earlier picks to have higher scores.
        # Let's use a scheme where score decreases as step increases.
        scores[v] = current_max_score
        current_max_score -= 1
        step += 1
        
        # Remove v and its neighbors from available
        # Neighbors of v are where adj[v] is True
        # We only remove those that are currently available
        neighbors = np.where(adj[v] & available)[0]
        
        available[v] = False
        available[neighbors] = False
        
        # Assign low scores to removed neighbors?
        # They are not picked, so they should have low priority.
        # The decoder will skip them if they are neighbors of a picked vertex.
        # Since v is picked, its neighbors are skipped.
        # We should ensure their scores are lower than v's score.
        # Since we are assigning scores in descending order of picks,
        # and these neighbors are not picks, they should get very low scores.
        # Let's assign them a score based on when they were removed, or just 0.
        # Actually, if they are not picked, they don't contribute to the set size directly,
        # but their low score ensures they are picked late (or never if blocked).
        # The decoder skips neighbors of *picked* vertices.
        # So if v is picked, neighbors are skipped regardless of their score.
        # But if a neighbor u was picked *before* v, then v would be skipped.
        # Our simulation ensures v is picked only if it was available (not neighbor of previous picks).
        # So the order of picks in simulation matches the decoder's picks.
        # The neighbors removed in this step were not picked, so they should have low scores.
        # Let's assign them a score of 0 or negative to ensure they are at the bottom.
        scores[neighbors] = 0.0 
        # Wait, if we assign 0, and later we pick another vertex, it gets a positive score.
        # This is fine. The decoder picks descending.
        # Picked vertices get n, n-1, ...
        # Unpicked vertices get 0.
        # This ensures picked vertices are always prioritized over unpicked ones.
        
    return scores
