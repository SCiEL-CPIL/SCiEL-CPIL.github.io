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
    
    # 2. Construct Laplacian Matrix L = D - A
    # D is the diagonal matrix of degrees
    # A is the adjacency matrix
    # For efficiency, we can use scipy.sparse or just dense numpy since N=200 is small
    # L = diag(deg) - adjacency
    # We need the Fiedler vector, which is the eigenvector corresponding to the second smallest eigenvalue.
    
    # Time guard: if we've used too much time, fall back to simpler scoring
    elapsed = time.time() - start_time
    if elapsed > 100.0:
        # Simple fallback: -degree
        return -deg
    
    try:
        # Build Laplacian
        L = np.diag(deg) - adjacency
        
        # Compute eigenvalues and eigenvectors
        # For symmetric matrix, use eigh
        # We only need the second smallest eigenvalue's eigenvector
        # np.linalg.eigh returns eigenvalues in ascending order
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        
        # The Fiedler vector is the eigenvector corresponding to eigenvalues[1]
        # Note: For a connected graph, eigenvalues[0] is 0. If the graph is disconnected,
        # there will be multiple zero eigenvalues. The "Fiedler" vector is typically the
        # first non-zero eigenvalue's eigenvector, or the second one if we assume connected.
        # In ER graphs with N=200, they are almost certainly connected.
        # Let's pick the eigenvector for the second smallest eigenvalue (index 1).
        fiedler_vec = eigenvectors[:, 1]
        
        # The magnitude of the Fiedler vector indicates how "peripheral" a node is.
        # Nodes with large |fiedler_vec[i]| are at the extremes of the graph's main cut.
        # Nodes with small |fiedler_vec[i]| are near the "center" or boundary of the cut.
        
        # Strategy:
        # We want to pick vertices that are "safe" to include.
        # In the context of MIS, picking a vertex removes its neighbors.
        # If we pick a "central" vertex (low |fiedler|), we might block many other central vertices?
        # Actually, the Fiedler vector identifies the bipartition that minimizes the cut.
        # Vertices with high magnitude are strongly associated with one side of the cut.
        # Vertices with low magnitude are "mixed" or central.
        
        # Let's test the hypothesis: Peripheral vertices (high |fiedler|) are better candidates?
        # Or Central vertices (low |fiedler|)?
        
        # Consider a path graph. Fiedler vector is a sine wave.
        # Endpoints have max magnitude. Center has min magnitude (near 0).
        # For MIS on a path, we want to pick alternating nodes.
        # Endpoints are "safe" in the sense that they have fewer neighbors (degree 1).
        # Center nodes have degree 2.
        # So high magnitude correlates with low degree here.
        
        # Consider a star graph. Center has degree N-1, leaves have degree 1.
        # Fiedler vector for star: Center is one value, leaves are another.
        # Actually, for a star, the Fiedler vector has the center at one extreme and all leaves at the other?
        # No, the Fiedler vector for a star graph has the center at 0? 
        # Let's check: L = D - A.
        # Star: Center 0, leaves 1..N-1.
        # Eigenvalues: 0, 1 (multiplicity N-2), N.
        # Eigenvector for 1: Center is 0, leaves sum to 0.
        # So leaves have non-zero values, center is 0.
        # High magnitude -> Leaves. Low magnitude -> Center.
        # We want to pick Leaves (high magnitude) because they have low degree and blocking them doesn't hurt much.
        # We want to avoid picking Center (low magnitude) because it blocks everyone.
        
        # So, score should favor high |fiedler_vec|.
        # However, we also need to account for degree.
        # A pure spectral score might ignore local degree nuances.
        # Let's combine: Score = w1 * |fiedler| - w2 * degree_normalized
        
        # Normalize fiedler magnitude
        fiedler_mag = np.abs(fiedler_vec)
        f_min, f_max = fiedler_mag.min(), fiedler_mag.max()
        if f_max - f_min < 1e-12:
            fiedler_norm = np.zeros_like(fiedler_mag)
        else:
            fiedler_norm = (fiedler_mag - f_min) / (f_max - f_min)
        
        # Normalize degree
        d_min, d_max = deg.min(), deg.max()
        if d_max - d_min < 1e-12:
            deg_norm = np.zeros_like(deg)
        else:
            deg_norm = (deg - d_min) / (d_max - d_min)
        
        # Combine
        # We want high fiedler_norm (peripheral) and low deg_norm (low degree).
        # Score = fiedler_norm - deg_norm
        # The greedy decoder picks in descending score order.
        # So higher score = picked earlier.
        
        score = 1.0 * fiedler_norm - 0.5 * deg_norm
        
        # Check time again
        elapsed = time.time() - start_time
        if elapsed > 110.0:
            # If we are running out of time, return the current best guess
            pass
            # Actually, if we are here, we already computed the score.
            # The time guard at the start handles the case where we haven't started.
            # If we are in the middle, we should just return.
            
        return score
        
    except Exception as e:
        # Fallback if eigenvalue computation fails or takes too long (though eigh is fast for N=200)
        return -deg
