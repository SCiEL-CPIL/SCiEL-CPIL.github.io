import numpy as np
import time

def priority(adjacency):
    start_time = time.time()
    
    # Ensure adjacency is a numpy array for efficient operations
    if not isinstance(adjacency, np.ndarray):
        adjacency = np.array(adjacency)
    
    n = adjacency.shape[0]
    if n == 0:
        return np.array([], dtype=float)
    
    # 1. Compute Degrees
    deg = np.sum(adjacency, axis=1).astype(float)
    
    # 2. Compute Graph Laplacian L = D - A
    # D is diagonal matrix of degrees
    D = np.diag(deg)
    L = D - adjacency.astype(float)
    
    # 3. Compute Eigenvalues and Eigenvectors
    # We are interested in the Fiedler vector, which corresponds to the
    # second smallest eigenvalue of the Laplacian.
    # For an undirected graph, the smallest eigenvalue is 0 with eigenvector 1.
    # The second smallest gives information about the graph's connectivity structure.
    # Vertices with large absolute values in the Fiedler vector are often "peripheral"
    # or on the "edges" of the main connected component's structure.
    # Picking these first is less likely to disconnect the graph or isolate
    # high-value clusters, potentially leading to a larger independent set.
    
    try:
        # Use numpy's eigh for symmetric matrices (L is symmetric)
        # This is faster and more stable than eig for symmetric cases.
        evals, evecs = np.linalg.eigh(L)
        
        # The eigenvalues are sorted in ascending order by eigh.
        # evals[0] should be ~0.
        # evecs[:, 1] is the Fiedler vector.
        fiedler_vec = evecs[:, 1]
        
        # The magnitude of the Fiedler vector indicates how "far" a vertex is from the
        # "center" of the graph's bipartition or main component.
        # High magnitude -> peripheral.
        # We want to pick peripheral vertices first? Or central?
        
        # Let's think:
        # If we pick a central vertex (low magnitude in Fiedler vector, often near 0 if the graph is well-connected),
        # its neighbors are spread out.
        # If we pick a peripheral vertex (high magnitude), its neighbors are likely also peripheral or in a "lobe".
        
        # The hypothesis from the prompt: "identifies vertices at the periphery... safer to include... without fragmenting".
        # So, high |Fiedler| -> High Priority.
        
        # However, there is a known heuristic for MIS:
        # "Low degree" is the primary driver.
        # Spectral methods often suggest that vertices with high "eigenvector centrality" (largest eigenvalue) are central.
        # Fiedler vector is about connectivity/bottlenecks.
        
        # Let's combine Fiedler magnitude with Degree.
        # If we just use Fiedler magnitude, we might pick isolated nodes (which is good) but also nodes in sparse parts.
        # In ER graphs, degree is the strongest predictor.
        # Let's try: Score = - (w1 * deg_norm + w2 * fiedler_mag_norm)
        # Or maybe: Score = - deg_norm + w * fiedler_mag_norm?
        
        # Actually, let's look at the "periphery" idea.
        # In a connected graph, the Fiedler vector changes sign. Vertices near 0 are "bridges".
        # Vertices with large |value| are in the "lobes".
        # Picking from lobes might be safer because removing a bridge (low |value|) disconnects the graph,
        # which might not be bad for MIS (components can be solved independently),
        # BUT the greedy algorithm doesn't solve components independently, it just skips neighbors.
        # If we pick a bridge, we remove it and its neighbors. The remaining graph is disconnected.
        # The greedy algorithm continues on the whole set of unpicked nodes.
        # Disconnection doesn't hurt the greedy algorithm's correctness, but it might affect the *quality* if the
        # "best" nodes for the next step are in different components and we just pick globally by score.
        
        # Let's try a different spectral approach:
        # Use the eigenvector corresponding to the LARGEST eigenvalue (Principal Eigenvector).
        # This is the "eigenvector centrality".
        # High centrality -> Central.
        # Low centrality -> Peripheral.
        # For MIS, we generally want to pick low-degree, low-centrality nodes.
        
        # Let's stick to the prompt's specific instruction: "Fiedler vector... magnitude as priority score".
        # It says: "identifies vertices at the periphery... safer to include".
        # So: High |Fiedler| = High Priority.
        
        fiedler_mag = np.abs(fiedler_vec)
        
        # Normalize Fiedler magnitude
        f_min, f_max = fiedler_mag.min(), fiedler_mag.max()
        if f_max - f_min < 1e-12:
            f_norm = np.zeros_like(fiedler_mag)
        else:
            f_norm = (fiedler_mag - f_min) / (f_max - f_min)
            
        # Normalize Degree
        d_min, d_max = deg.min(), deg.max()
        if d_max - d_min < 1e-12:
            d_norm = np.zeros_like(deg)
        else:
            d_norm = (deg - d_min) / (d_max - d_min)
            
        # Combine
        # The prompt implies Fiedler magnitude is the key new signal.
        # Let's try a weighted sum.
        # If we pick high Fiedler mag (periphery) AND low degree, that's ideal.
        # So we want to maximize Fiedler Mag and minimize Degree.
        # Score = w_f * f_norm - w_d * d_norm
        # Or Score = f_norm - d_norm
        
        # Let's check the time budget.
        elapsed = time.time() - start_time
        if elapsed > 100.0:
            # Fallback to simple degree if spectral is slow (unlikely for N=200, but safe)
            return -deg
            
        # Heuristic:
        # Peripheral nodes (high Fiedler) are "safer".
        # Low degree nodes are "safer".
        # Let's try: score = 1.0 * f_norm - 1.0 * d_norm
        # This prioritizes peripheral, low-degree nodes.
        
        score = 1.0 * f_norm - 1.0 * d_norm
        
    except Exception:
        # If spectral decomposition fails or is too slow, fall back to degree
        score = -deg
        
    return score
