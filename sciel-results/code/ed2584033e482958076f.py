def construct():
    # ... (code as above) ...
    # Final sanity check
    norms = np.linalg.norm(best_pts, axis=1)
    if np.any(norms > 1.0):
        # Push back
        best_pts = best_pts * (1.0 / norms)[:, None]
        
    return best_pts
