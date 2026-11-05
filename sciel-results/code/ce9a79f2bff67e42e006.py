"""Combined program: Rich feature engineering with weight-space search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(adjacency):
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin below 120s limit

    A = np.asarray(adjacency, dtype=float)
    N = A.shape[0]

    if N == 0:
        return np.array([])

    # Ensure symmetric and zero diagonal
    A = (A + A.T) / 2.0
    np.fill_diagonal(A, 0.0)
    A_bool = A > 0.5

    # --- Feature Extraction ---
    
    # 1. Degree
    D = A.sum(axis=1)
    D_safe = D + 1e-9

    # 2. Mean Neighbor Degree
    SND = A @ D
    MeanND = SND / D_safe

    # 3. Triangles (number of triangles each vertex is in)
    A_sq = A @ A
    Tri = np.diag(A_sq) / 2.0

    # 4. Clustering Coefficient
    Clus = np.zeros(N)
    mask = D > 1
    Clus[mask] = 2.0 * Tri[mask] / (D[mask] * (D[mask] - 1.0))

    # 5. Open Pairs (pairs of neighbors that are NOT connected)
    # C(D, 2) - Tri
    OpenPairs = np.maximum(0, 0.5 * D * (D - 1.0) - Tri)

    # 6. Outward Expansion (how "spread out" the neighborhood is)
    # Sum of (degree of neighbor - 1) for all neighbors, normalized by degree
    # This measures edges leaving the ego-network.
    OutwardExp = np.zeros(N)
    # Vectorized approach: 
    # For each i, sum over neighbors j of (D[j] - 1)
    # This is equivalent to (A @ (D - 1))[i]
    # But we must be careful: A[i,j] is 1 if connected.
    # Sum_{j in N(i)} (D[j] - 1) = (A @ (D - 1))[i]
    # Since A has 0 diagonal, this sums over neighbors only.
    OutwardExp_raw = A @ (D - 1.0)
    OutwardExp = OutwardExp_raw / D_safe

    # 7. Min Neighbor Degree
    # This is harder to vectorize efficiently without loops for N=200, but loops are fine.
    # Alternatively, use a quantile of neighbor degrees. Let's stick to Min as in Insp 1.
    MinNbrDeg = np.zeros(N)
    for i in range(N):
        nbrs = np.where(A_bool[i])[0]
        if len(nbrs) > 0:
            MinNbrDeg[i] = D[nbrs].min()
        else:
            MinNbrDeg[i] = 0.0

    # Normalize features to [0, 1]
    def norm(x):
        min_val = np.min(x)
        max_val = np.max(x)
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

    # --- Evaluation Function ---
    # Precompute adjacency rows for fast masking
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
                # Mark neighbors as picked (blocked)
                is_picked[adj_rows[v]] = True
        return count

    best_size = -1
    best_scores = None

    # --- Search Phase 1: Deterministic Candidates ---
    # Weights: (w_deg, w_meanND, w_clus, w_open, w_tri, w_outward, w_minND)
    # Score = -w_deg*D + w_meanND*MeanND - w_clus*Clus + w_open*Open - w_tri*Tri + w_outward*Outward + w_minND*MinND
    
    candidates = [
        (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),  # Baseline: -Degree
        (1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),  # -Deg + Open
        (1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0),  # -Deg + MeanND
        (1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0),  # -Deg - Clus
        (1.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0),  # Mix
        (0.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0),  # MeanND heavy
        (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0),  # -Deg + Outward
        (1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0), # -Deg + Open + Outward
        (1.0, 0.5, 0.5, 0.5, 0.0, 0.5, 0.0), # Mix + Outward
        (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0), # -Deg + MinND
        (1.0, 0.5, 0.5, 0.5, 0.0, 0.5, 0.5), # All mixed
        (1.5, 0.5, 0.5, 1.0, 0.0, 0.5, 0.0), # Stronger Degree weight
    ]
    
    for w in candidates:
        scores = (-w[0] * D_n + w[1] * MeanND_n - w[2] * Clus_n 
                  + w[3] * OpenPairs_n - w[4] * Tri_n 
                  + w[5] * OutwardExp_n + w[6] * MinNbrDeg_n)
        size = evaluate_fast(scores)
        if size > best_size:
            best_size = size
            best_scores = scores.copy()

    # --- Search Phase 2: Random Search ---
    rng = np.random.default_rng(42)
    max_trials = 10000  # Upper bound, will be cut by time
    
    # Ranges based on Inspiration 1's successful ranges
    ranges = {
        'w1': (0.3, 2.5),   # Degree
        'w2': (0.0, 2.0),   # MeanND
        'w3': (0.0, 1.5),   # Clustering
        'w4': (0.0, 2.0),   # Open Pairs
        'w5': (0.0, 0.8),   # Triangles
        'w6': (0.0, 1.5),   # Outward Exp
        'w7': (0.0, 1.0),   # Min Nbr Deg
    }

    trial_count = 0
    while trial_count < max_trials:
        if time.time() > DEADLINE - 5.0: # Reserve 5s for final check/refinement
            break
            
        w1 = rng.uniform(*ranges['w1'])
        w2 = rng.uniform(*ranges['w2'])
        w3 = rng.uniform(*ranges['w3'])
        w4 = rng.uniform(*ranges['w4'])
        w5 = rng.uniform(*ranges['w5'])
        w6 = rng.uniform(*ranges['w6'])
        w7 = rng.uniform(*ranges['w7'])
        
        scores = (-w1 * D_n + w2 * MeanND_n - w3 * Clus_n 
                  + w4 * OpenPairs_n - w5 * Tri_n 
                  + w6 * OutwardExp_n + w7 * MinNbrDeg_n)
        
        size = evaluate_fast(scores)
        
        if size > best_size:
            best_size = size
            best_scores = scores.copy()
            
        trial_count += 1

    # --- Search Phase 3: Local Refinement ---
    # Perturb the best found score vector to fine-tune ordering
    if best_scores is not None and time.time() < DEADLINE - 1.0:
        for _ in range(500):
            if time.time() > DEADLINE - 0.5:
                break
            # Small Gaussian noise
            perturbed = best_scores + rng.normal(0, 0.05, size=N)
            size = evaluate_fast(perturbed)
            if size > best_size:
                best_size = size
                best_scores = perturbed.copy()

    if best_scores is None:
        # Fallback to simple degree if something went wrong
        return -D.astype(float)

    return best_scores
# EVOLVE-BLOCK-END
