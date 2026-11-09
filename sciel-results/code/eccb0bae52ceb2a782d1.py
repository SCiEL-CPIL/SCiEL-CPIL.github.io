import numpy as np
import time

def priority(adjacency):
    """
    Calculates priority scores for vertices to maximize the size of the independent set
    selected by a greedy decoder (descending score).
    
    Strategy: Spectral Partitioning Heuristic.
    1. Compute the graph Laplacian L = D - A.
    2. Find the second-smallest eigenvalue (Fiedler value) and its corresponding eigenvector (Fiedler vector).
    3. The Fiedler vector partitions the graph into two sets (positive and negative components).
    4. Vertices with a strong signal in the Fiedler vector (far from zero) are considered "boundary" vertices
       or vertices with high structural significance in the partition.
    5. We prioritize vertices that are "core" to one side of the partition. 
       Specifically, we want to select vertices that are less likely to be neighbors of each other.
       A common heuristic for MIS is to pick low-degree vertices first. 
       However, spectral methods can identify communities. 
       
       Let's try a different spectral approach:
       The greedy algorithm picks vertices in descending order of score.
       If we score vertices by their "centrality" in a way that separates neighbors, we might do better.
       
       Alternative: Use the Fiedler vector to assign scores.
       Vertices with high absolute Fiedler vector values are "important" for the cut.
       Vertices with low absolute values are "inside" the clusters.
       For MIS, we want to pick vertices that are far apart.
       If we pick all vertices from one side of the Fiedler cut, we might get a large independent set?
       No, because neighbors within that side would conflict.
       
       Let's look at the "Meta-heuristic" inspiration: "simulating the greedy decoder locally".
       The current program is a static heuristic.
       
       Let's try a hybrid:
       1. Compute Fiedler vector.
       2. Score = -degree + alpha * fiedler_value?
       
       Actually, the directive suggests: "Use a spectral gap heuristic ... prioritize vertices in the smaller partition to minimize the number of edges crossing the independent set boundary."
       
       Let's implement:
       1. Compute L.
       2. Compute Fiedler vector (eigenvector of 2nd smallest eigenvalue).
       3. Partition V into V+ (fiedler > 0) and V- (fiedler <= 0).
       4. Calculate the number of edges crossing the cut.
       5. The "better" partition is the one that allows a larger independent set.
          Roughly, the independent set size is bounded by n - (min vertex cover).
          But for a greedy decoder, we just need a good ordering.
       
       Hypothesis: Vertices in the smaller partition might be "more isolated" or "less connected" to the rest, 
       or vice versa.
       
       Let's try scoring based on the Fiedler vector magnitude and degree.
       High |fiedler| means the vertex is a "bridge" or "boundary".
       Low |fiedler| means the vertex is "deep" in a cluster.
       
       If we pick "deep" vertices first, we might pick many neighbors.
       If we pick "boundary" vertices first, we might block many potential picks.
       
       Let's try: Score = -degree - beta * fiedler_value^2.
       This penalizes high degree and high spectral significance.
       
       Or, simpler:
       The Fiedler vector approximates a potential.
       Let's try to use the sign of the Fiedler vector to group vertices.
       We want to pick vertices such that we pick many from one group and few from the other?
       
       Actually, let's look at the "Meta-heuristic" again.
       "simulating the greedy decoder locally on the input graph".
       This implies we can run the greedy algorithm on the graph to test different scoring functions and pick the best one?
       But the function `priority` only returns scores. The decoder is external.
       So we can't "pick the best scoring function" based on the result of the decoder, because we don't see the result.
       We just return scores.
       
       However, we *can* simulate the greedy decoder *inside* the `priority` function to evaluate candidate scoring strategies and return the scores from the best strategy?
       Yes! The `priority` function receives the adjacency matrix. We can:
       1. Generate a set of candidate scoring vectors (e.g., based on degree, spectral, etc.).
       2. For each candidate, simulate the greedy MIS algorithm.
       3. Keep the candidate that yields the largest independent set size.
       4. Return the scores of the best candidate.
       
       This is a "Meta-heuristic" approach. It uses the specific graph structure to adapt the scoring.
       
       Candidates to test:
       1. Pure Degree (ascending) -> Score = -degree
       2. Pure Spectral (Fiedler) -> Score = -|fiedler| or similar
       3. Combined -> Score = -degree - alpha * |fiedler|
       
       Since n=200, simulating greedy MIS is O(n^2) or O(n*E) which is very fast (200^2 = 40,000 ops).
       We can test many candidates.
       
       Let's implement this.
    """
    start_time = time.time()
    deadline = start_time + 140.0
    
    A = np.array(adjacency, dtype=bool)
    n = A.shape[0]
    
    if n == 0:
        return np.array([])
        
    degrees = A.sum(axis=1).astype(int)
    
    # Helper function to simulate greedy MIS given scores
    def simulate_greedy(scores, A, n):
        # Sort vertices by descending score
        order = np.argsort(-scores)
        selected = np.zeros(n, dtype=bool)
        count = 0
        for v in order:
            if not selected[v]:
                # Check if any neighbor is selected
                if not selected[A[v]].any():
                    selected[v] = True
                    count += 1
        return count

    # Candidate 1: Degree based (lower degree is better)
    scores_deg = -degrees.astype(float)
    
    # Candidate 2: Spectral based
    # Compute Laplacian
    D = np.diag(degrees)
    L = D - A.astype(float)
    
    # Get eigenvalues and eigenvectors
    # n=200 is small, full eigendecomposition is fine
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
        # Second smallest eigenvalue is index 1 (index 0 is 0 for connected graphs)
        # If graph is disconnected, there are multiple 0 eigenvalues.
        # We want the smallest positive eigenvalue's eigenvector.
        # Usually index 1 is the Fiedler vector for connected graphs.
        fiedler_vec = eigvecs[:, 1]
        
        # Score based on Fiedler vector
        # Vertices with high absolute value are "boundary".
        # We want to avoid boundary? Or pick them?
        # Let's try: Score = -|fiedler| (prefer deep vertices)
        scores_spec1 = -np.abs(fiedler_vec)
        
        # Or Score = fiedler_vec (prefer one side)
        scores_spec2 = fiedler_vec
        
        # Or Score = -fiedler_vec^2
        scores_spec3 = -fiedler_vec**2
        
    except Exception:
        fiedler_vec = np.zeros(n)
        scores_spec1 = np.zeros(n)
        scores_spec2 = np.zeros(n)
        scores_spec3 = np.zeros(n)
        
    # Candidate 3: Combined
    # Normalize fiedler vector to [0, 1] roughly
    if np.max(np.abs(fiedler_vec)) > 0:
        fiedler_norm = np.abs(fiedler_vec) / np.max(np.abs(fiedler_vec))
    else:
        fiedler_norm = np.zeros(n)
        
    # We need to normalize degree too
    max_deg = np.max(degrees) if np.max(degrees) > 0 else 1
    deg_norm = degrees.astype(float) / max_deg
    
    # Try different weights
    candidates = []
    candidates.append(("deg", scores_deg))
    candidates.append(("spec1", scores_spec1))
    candidates.append(("spec2", scores_spec2))
    candidates.append(("spec3", scores_spec3))
    
    for alpha in [0.1, 0.5, 1.0, 2.0]:
        # Score = -deg_norm - alpha * fiedler_norm
        candidates.append((f"comb_{alpha}", -deg_norm - alpha * fiedler_norm))
        
        # Score = -deg_norm + alpha * fiedler_vec (signed)
        candidates.append((f"comb_signed_{alpha}", -deg_norm + alpha * fiedler_vec))
        
    # Evaluate all candidates
    best_score = -1
    best_scores = scores_deg # Fallback
    
    for name, scores in candidates:
        if time.time() > deadline:
            break
        size = simulate_greedy(scores, A, n)
        if size > best_score:
            best_score = size
            best_scores = scores
            
    # If we didn't evaluate all, ensure we return something valid
    # If we ran out of time, we return the best found so far
    
    return best_scores
