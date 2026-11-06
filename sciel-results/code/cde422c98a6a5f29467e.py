"""Priority function using refined Fiedler vector for Maximum Independent Set."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def priority(adjacency):
    """
    Score vertices using the Fiedler vector of the graph Laplacian.
    Vertices with large |Fiedler component| are at the periphery of the
    graph's main connected structure and are safer to pick for an
    independent set.
    
    Improvements:
    1. Use dense linear algebra for robustness and speed on N=200.
    2. Increased weight on degree penalty to avoid picking high-degree hubs.
    3. Safety check for disconnected graphs (fall back to degree-based).
    """
    start_time = time.time()
    DEADLINE = 115.0  # safety margin before 120s limit

    A = np.asarray(adjacency, dtype=np.float64)
    n = A.shape[0]

    if n == 0:
        return np.array([], dtype=np.float64)
    
    if n == 1:
        return np.array([0.0], dtype=np.float64)

    # Ensure symmetry
    A = (A + A.T) / 2.0

    # Compute degree vector
    deg = A.sum(axis=1)

    # Build Laplacian L = D - A
    L = np.diag(deg) - A

    try:
        # Dense eigendecomposition is fast and robust for N=200
        vals, vecs = np.linalg.eigh(L)
        
        # Check for disconnected graph
        # If the second smallest eigenvalue is very close to 0, the graph is disconnected
        # In that case, the Fiedler vector is not well-defined as a global periphery indicator
        if vals[1] < 1e-5:
            # Fallback: Degree-based score (lower degree is better)
            # Normalize degree to [0, 1] for consistent scaling
            max_deg = deg.max()
            if max_deg > 0:
                deg_norm = deg / max_deg
            else:
                deg_norm = deg
            return -deg_norm.astype(np.float64)

        # Fiedler vector is the eigenvector corresponding to the second smallest eigenvalue
        fiedler = vecs[:, 1]
        
        # Use magnitude
        base_scores = np.abs(fiedler)
        
        # Normalize Fiedler scores to [0, 1] range for better combination with degree
        fiedler_min = base_scores.min()
        fiedler_max = base_scores.max()
        if fiedler_max > fiedler_min:
            scores = (base_scores - fiedler_min) / (fiedler_max - fiedler_min)
        else:
            scores = np.zeros(n, dtype=np.float64)

        # Adaptive spectral weighting: use eigenvalue gap to modulate peripherality signal
        # When eigenvalues are closely spaced, the Fiedler vector is less distinct
        # Use the ratio of gap to total spread to weight the peripherality component
        if len(vals) >= 2:
            eigenvalue_spread = vals[-1] - vals[0]
            if eigenvalue_spread > 1e-10:
                # Gap between first two eigenvalues relative to total spread
                gap_ratio = (vals[1] - vals[0]) / eigenvalue_spread
                # Clamp to [0.3, 1.0] to maintain some peripherality signal
                adaptive_weight = np.clip(gap_ratio * 3.0, 0.3, 1.0)
            else:
                adaptive_weight = 1.0
        else:
            adaptive_weight = 1.0
        
        # Apply adaptive weighting to Fiedler scores
        weighted_scores = adaptive_weight * scores

        # Damped fixed-point iteration for stability
        # Iterate: new_score = (1-alpha)*old_score + alpha*base_score
        # This smooths out noise in the Fiedler vector
        alpha = 0.5  # Damping factor
        for iteration in range(5):
            if time.time() > DEADLINE:
                break
            # Combine with a slight smoothing toward the mean
            mean_score = weighted_scores.mean()
            # Damped update: move 50% toward base score, 50% keep current
            weighted_scores = (1 - alpha) * weighted_scores + alpha * weighted_scores
            # Add small regularization to prevent extreme values
            weighted_scores = weighted_scores + 0.01 * (mean_score - weighted_scores)
        
        # Re-normalize after iteration
        ws_min = weighted_scores.min()
        ws_max = weighted_scores.max()
        if ws_max > ws_min:
            weighted_scores = (weighted_scores - ws_min) / (ws_max - ws_min)
        else:
            weighted_scores = np.zeros(n, dtype=np.float64)

        # Degree penalty: We want to avoid high-degree vertices.
        # Normalize degree to [0, 1]
        max_deg = deg.max()
        if max_deg > 0:
            deg_norm = deg / max_deg
        else:
            deg_norm = deg

        # Local neighborhood refinement: penalize vertices adjacent to high-scoring neighbors
        # This makes the score more competitive-aware
        if time.time() < DEADLINE:
            # Compute neighbor score sum for each vertex
            # A @ weighted_scores gives sum of neighbor scores
            neighbor_score_sum = A @ weighted_scores
            neighbor_count = deg
            # Average neighbor score (handle zero-degree vertices)
            avg_neighbor_score = np.where(neighbor_count > 0, 
                                          neighbor_score_sum / np.maximum(neighbor_count, 1), 
                                          0.0)
            
            # Penalize vertices whose neighbors have high scores (more contested)
            # Scale the penalty by the adaptive weight
            neighborhood_penalty = 0.3 * adaptive_weight * avg_neighbor_score
            
            # Combine: High weighted Fiedler, Low Degree, Low neighborhood contention
            final_scores = weighted_scores - 0.4 * deg_norm - neighborhood_penalty
        else:
            # Fallback if time is running out
            final_scores = weighted_scores - 0.4 * deg_norm

        return final_scores.astype(np.float64)

    except Exception:
        # Ultimate fallback: degree-based
        return -deg.astype(np.float64)
# EVOLVE-BLOCK-END
