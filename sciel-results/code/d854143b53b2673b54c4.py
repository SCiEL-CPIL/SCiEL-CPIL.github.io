import numpy as np
import time

def priority(adjacency):
    start_time = time.time()
    DEADLINE = 115.0

    A = np.asarray(adjacency, dtype=float)
    N = A.shape[0]

    if N == 0:
        return np.array([])

    # Ensure symmetric (just in case)
    A = (A + A.T) / 2.0
    # Zero diagonal
    np.fill_diagonal(A, 0.0)
    A_bool = A > 0.5

    D = A.sum(axis=1)
    D_safe = D + 1e-9

    # Features
    SND = A @ D
    MeanND = SND / D_safe

    # Triangles
    A_sq = A @ A
    Tri = np.diag(A_sq) / 2.0

    # Clustering
    Clus = np.zeros(N)
    mask = D > 1
    Clus[mask] = 2.0 * Tri[mask] / (D[mask] * (D[mask] - 1.0))

    # Open pairs: C(D,2) - Tri
    OpenPairs = np.maximum(0, 0.5 * D * (D - 1.0) - Tri)

    # New feature: weighted degree of neighbors (how many edges from i's neighborhood go outward)
    OutwardExp = np.zeros(N)
    for i in range(N):
        nbrs = np.where(A_bool[i])[0]
        if len(nbrs) > 0:
            OutwardExp[i] = np.sum(D[nbrs] - 1.0)
    OutwardExp = OutwardExp / D_safe

    # Another feature: minimum degree among neighbors
    MinNbrDeg = np.zeros(N)
    for i in range(N):
        nbrs = np.where(A_bool[i])[0]
        if len(nbrs) > 0:
            MinNbrDeg[i] = D[nbrs].min()

    # Feature: number of degree-1 neighbors (leaves attached)
    LeafNbr = np.zeros(N)
    for i in range(N):
        nbrs = np.where(A_bool[i])[0]
        if len(nbrs) > 0:
            LeafNbr[i] = np.sum(D[nbrs] == 1)

    # Feature: 2-hop distance count (spread)
    # For each vertex, count nodes at distance exactly 2
    A2 = A @ A
    # Nodes at distance 2 from i: those j where A2[i,j] > 0 and A[i,j] == 0 and j != i
    # We can compute this as: for each i, sum over j of (A2[i,j] > 0) * (A[i,j] == 0) * (j != i)
    # But this is expensive. Let's use a simpler proxy:
    # Number of 2-hop neighbors that are not direct neighbors
    # = sum_j A2[i,j] - sum_j A[i,j]*A2[i,j] - A2[i,i]
    # Actually, let's just use the total 2-hop reach minus direct neighbors
    TwoHopReach = np.zeros(N)
    for i in range(N):
        if D[i] > 0:
            # A2[i] gives counts of 2-step paths
            # Direct neighbors contribute A2[i, nbrs] which includes paths through i itself
            # Simplification: use sum of A2[i] minus contributions from direct neighbors
            # For now, use a simpler feature: variance of neighbor degrees
            nbrs = np.where(A_bool[i])[0]
            if len(nbrs) > 1:
                TwoHopReach[i] = np.var(D[nbrs])
            elif len(nbrs) == 1:
                TwoHopReach[i] = 0.0

    # Normalize
    def norm(x):
        min_val = x.min()
        max_val = x.max()
        if max_val == min_val:
            return np.zeros_like(x)
        return (x - min_val) / (max_val - min_val)

    D_n = norm(D)
    MeanND_n = norm(MeanND)
    Tri_n = norm(Tri)
    Clus_n = norm(Clus)
    OpenPairs_n = norm(OpenPairs)
    OutwardExp_n = norm(OutwardExp)
    MinNbrDeg_n = norm(MinNbrDeg)
    LeafNbr_n = norm(LeafNbr)
    TwoHopReach_n = norm(TwoHopReach)

    # Fast vectorized evaluation
    adj_rows = A_bool

    def evaluate_fast(scores):
        """Vectorized greedy independent set evaluation."""
        indices = np.argsort(-scores, kind='stable')
        is_picked = np.zeros(N, dtype=bool)
        count = 0
        for v in indices:
            if not is_picked[v]:
                count += 1
                is_picked[v] = True
                is_picked[adj_rows[v]] = True
        return count

    best_size = -1
    best_scores = None

    # Baseline: Low degree
    base_score = -D_n
    base_size = evaluate_fast(base_score)
    best_size = base_size
    best_scores = base_score.copy()

    # Search for better weights
    rng = np.random.default_rng(42)

    # Try a few deterministic combinations first
    candidates = [
        (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),  # -D
        (1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0),  # -D + Open
        (1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),  # -D + MeanND
        (1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),  # -D - Clus
        (1.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0),  # Mix
        (0.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),  # MeanND + -D
        (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),  # -D + OutwardExp
        (1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0), # -D + Open + OutwardExp
        (1.0, 0.5, 0.5, 0.5, 0.0, 0.5, 0.0, 0.0, 0.0), # Mix + OutwardExp
        (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0), # -D + MinNbrDeg
        (1.0, 0.5, 0.5, 0.5, 0.0, 0.5, 0.5, 0.0, 0.0), # All mixed
        (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0), # -D + LeafNbr
        (1.0, 0.5, 0.5, 0.5, 0.0, 0.5, 0.5, 1.0, 0.5), # All with LeafNbr
        (1.0, 0.5, 0.5, 0.5, 0.0, 0.5, 0.5, 1.0, 1.0), # All
    ]

    for w in candidates:
        scores = (-w[0] * D_n + w[1] * MeanND_n - w[2] * Clus_n 
                  + w[3] * OpenPairs_n - w[4] * Tri_n 
                  + w[5] * OutwardExp_n + w[6] * MinNbrDeg_n
                  + w[7] * LeafNbr_n + w[8] * TwoHopReach_n)
        size = evaluate_fast(scores)
        if size > best_size:
            best_size = size
            best_scores = scores.copy()

    # Random search with more diverse sampling
    max_trials = 8000
    trial_count = 0

    while trial_count < max_trials:
        if time.time() > DEADLINE:
            break

        # Sample weights with broader ranges
        w1 = rng.uniform(0.3, 2.5)      # degree weight
        w2 = rng.uniform(0.0, 2.0)      # mean neighbor degree
        w3 = rng.uniform(0.0, 1.5)      # clustering
        w4 = rng.uniform(0.0, 2.0)      # open pairs
        w5 = rng.uniform(0.0, 0.8)      # triangles
        w6 = rng.uniform(0.0, 1.5)      # outward expansion
        w7 = rng.uniform(0.0, 1.0)      # min neighbor degree
        w8 = rng.uniform(0.0, 1.0)      # leaf neighbors
        w9 = rng.uniform(0.0, 1.0)      # two-hop reach
        
        scores = (-w1 * D_n + w2 * MeanND_n - w3 * Clus_n 
                  + w4 * OpenPairs_n - w5 * Tri_n 
                  + w6 * OutwardExp_n + w7 * MinNbrDeg_n
                  + w8 * LeafNbr_n + w9 * TwoHopReach_n)
        
        size = evaluate_fast(scores)
        
        if size > best_size:
            best_size = size
            best_scores = scores.copy()
            
        trial_count += 1

    # Local refinement around best found weights
    if best_scores is not None and time.time() < DEADLINE - 5.0:
        # Try small perturbations around the best score vector
        for _ in range(200):
            if time.time() > DEADLINE:
                break
            perturbed = best_scores + rng.normal(0, 0.1, size=N)
            size = evaluate_fast(perturbed)
            if size > best_size:
                best_size = size
                best_scores = perturbed.copy()

    return best_scores
