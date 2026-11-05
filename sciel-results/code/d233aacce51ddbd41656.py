"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(adjacency):
    """
    Assign a score to each vertex for a greedy maximum independent set decoder.
    
    Strategy: Use spectral partitioning to identify a large independent set candidate.
    The principal eigenvector of the adjacency matrix provides a 2-coloring heuristic.
    Vertices with similar signs belong to the same "community" or partition.
    In sparse graphs, one of these partitions is often a large independent set (or close to one).
    We prefer vertices from the larger partition, and within that, those with low degree.
    
    Steps:
    1. Compute the principal eigenvector of the adjacency matrix.
    2. Use the sign of the eigenvector components to partition vertices.
    3. Identify the larger partition (by size or by sum of degrees? Size is better for MIS count).
    4. Assign high scores to vertices in the larger partition.
    5. Within the larger partition, prefer low degree vertices.
    6. Assign low scores to vertices in the smaller partition.
    
    This leverages global structure rather than just local degree.
    """
    if not isinstance(adjacency, np.ndarray):
        adjacency = np.array(adjacency)
    
    # Ensure float for eigen computation
    A = adjacency.astype(float)
    
    # Compute principal eigenvector
    # For sparse matrices, use sparse eigen solver if available, but N=200 is small.
    # np.linalg.eigh is efficient for symmetric matrices.
    try:
        # eigh returns eigenvalues in ascending order
        eigenvalues, eigenvectors = np.linalg.eigh(A)
        # Principal eigenvector is the last column (largest eigenvalue)
        # For ER graphs, the largest eigenvalue is positive, so the vector can be signed.
        # We can shift the sign so that the sum of components is positive, or just use absolute values?
        # Actually, the sign is arbitrary. We need a consistent partition.
        # Let's take the eigenvector corresponding to the largest eigenvalue.
        v = eigenvectors[:, -1]
        
        # The sign of v is arbitrary. We can fix it by ensuring the component with the largest magnitude is positive.
        max_idx = np.argmax(np.abs(v))
        if v[max_idx] < 0:
            v = -v
            
        # Partition: positive vs negative
        # Vertices with v[i] > 0 are in group 1, v[i] < 0 in group 2.
        # Zeros? Rare in continuous eigenvectors, but handle just in case.
        # If v[i] is close to 0, it's on the boundary. Let's treat positive (including 0) as group 1?
        # Better: use the sign.
        pos_mask = v > 0
        neg_mask = v < 0
        
        # Count sizes
        n_pos = np.sum(pos_mask)
        n_neg = np.sum(neg_mask)
        
        # We want to pick the larger group as the "primary" independent set candidate.
        # If they are equal, it doesn't matter much, but let's pick the one with lower total degree?
        # Actually, for MIS, we just want a large set. The larger group is the better candidate.
        
        # Degrees
        deg = A.sum(axis=1)
        
        # Initialize scores
        # We want high score for the primary group, low for the secondary.
        # Within primary, lower degree -> higher score.
        # Within secondary, lower degree -> higher score (but still lower than primary).
        
        # Let's define scores relative to the groups.
        # Score = BaseScore(Group) - Alpha * Degree
        
        # What is BaseScore?
        # Let's say Primary Base = 1000, Secondary Base = 0.
        # Alpha should be such that degree differences within a group matter,
        # but group difference dominates.
        # Max degree is N-1 = 199.
        # So Alpha * 199 should be less than the gap between bases.
        # If Base gap is 1000, and Alpha = 1, max degree penalty is 199.
        # So a low-degree vertex in Primary (e.g., deg 0, score 1000) > high-degree in Secondary (e.g., deg 199, score -199).
        # This ensures we pick from Primary first.
        
        if n_pos >= n_neg:
            primary_mask = pos_mask
            secondary_mask = neg_mask
        else:
            primary_mask = neg_mask
            secondary_mask = pos_mask
        
        # Handle vertices with v == 0 (rare)
        # Assign them to the primary group if we want to maximize size?
        # Or just leave them out?
        # Let's assign zeros to the group with fewer vertices to balance?
        # Or just treat them as secondary?
        # For simplicity, let's treat v > 0 as one group, v <= 0 as the other.
        # Wait, if we flip v, the groups swap.
        # Let's just use the masks computed.
        
        # Create score array
        scores = np.zeros(len(v))
        
        # Primary group: High base, penalized by degree
        # Secondary group: Low base, penalized by degree
        
        # Let's use a dynamic base to ensure separation.
        # Max possible degree penalty: max_deg * alpha
        # Let alpha = 1.0
        # Let base_gap = max_deg + 1
        max_deg = np.max(deg)
        base_gap = max_deg + 1.0
        
        # Primary score: base_gap - deg
        # Secondary score: -deg
        # This ensures all Primary scores > all Secondary scores.
        
        scores[primary_mask] = base_gap - deg[primary_mask]
        scores[secondary_mask] = -deg[secondary_mask]
        
        # What about vertices not in either mask (v == 0)?
        zero_mask = ~(primary_mask | secondary_mask)
        if np.any(zero_mask):
            # Assign to primary if it makes the primary set larger?
            # Or just assign to secondary?
            # Let's assign to primary to maximize the candidate pool for the high-score group.
            scores[zero_mask] = base_gap - deg[zero_mask]
            
        return scores
        
    except Exception:
        # Fallback to simple degree heuristic if eigen computation fails
        deg = adjacency.sum(axis=1).astype(float)
        return -deg

# EVOLVE-BLOCK-END
