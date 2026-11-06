def update_affected(e_idx, colours):
    affected_k4s = edge_affected_k4_indices[e_idx]
    affected_edges = edge_affected_k4_edges[e_idx] # (91, 6)
    
    # Get colors for these edges
    c = colours[affected_edges] # (91, 6)
    
    # Calculate pair counts
    matches = (c[:, :, None] == c[:, None, :])
    counts = np.sum(matches[:, mask], axis=1)
    
    # Update global pair_counts
    pair_counts[affected_k4s] = counts
