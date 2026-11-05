import numpy as np
import time
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def priority(adjacency):
    start_time = time.time()
    DEADLINE = 148.0  # 2s safety margin before 150s kill

    A = np.asarray(adjacency, dtype=float)
    N = A.shape[0]

    if N == 0:
        return np.array([])

    # Ensure symmetric and no self-loops
    A = (A + A.T) / 2.0
    np.fill_diagonal(A, 0.0)
    A_bool = A > 0.5

    # Precompute neighbor lists for fast greedy evaluation
    nbrs = [np.where(A_bool[i])[0] for i in range(N)]
    
    # --- Feature Computation ---
    
    # 1. Degree
    D = A.sum(axis=1)
    
    # 2. Sum of Neighbor Degrees (SND)
    SND = A @ D
    
    # 3. Clustering Coefficient
    Tri = np.diag(A @ A) / 2.0
    Clus = np.zeros(N)
    mask = D > 1
    Clus[mask] = 2.0 * Tri[mask] / (D[mask] * (D[mask] - 1.0))
    
    # 4. Core Number (k-core)
    # Using a simple peeling algorithm, efficient for N=200
    core = np.zeros(N, dtype=int)
    deg = D.copy().astype(int)
    remaining = set(range(N))
    while remaining:
        min_deg = min(deg[i] for i in remaining)
        to_remove = [i for i in remaining if deg[i] == min_deg]
        for i in to_remove:
            core[i] = min_deg
            remaining.remove(i)
            for j in nbrs[i]:
                if j in remaining:
                    deg[j] -= 1
                    
    # 5. Fiedler Vector (2nd smallest eigenvector of Laplacian)
    # L = D - A
    L = np.diag(D) - A
    try:
        # For N=200, dense eigendecomposition is fine and fast (< 1s)
        eigvals, eigvecs = np.linalg.eigh(L)
        # Sort by eigenvalue
        idx = np.argsort(eigvals)
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        # Fiedler vector is the 2nd column (index 1)
        fiedler = eigvecs[:, 1]
        # Use absolute value, normalized
        fiedler_abs = np.abs(fiedler)
        fiedler_max = fiedler_abs.max()
        if fiedler_max > 0:
            fiedler_abs = fiedler_abs / fiedler_max
        else:
            fiedler_abs = np.zeros(N)
    except Exception:
        fiedler_abs = np.zeros(N)

    # 6. Distance-2 Non-Neighbors (Space)
    A_sq = A @ A
    mask_non_nbr = ~A_bool
    np.fill_diagonal(mask_non_nbr, False)
    Dist2NonNbr = (A_sq * mask_non_nbr).sum(axis=1)

    # Normalize features
    def norm(x):
        min_val = x.min()
        max_val = x.max()
        if max_val == min_val:
            return np.zeros_like(x)
        return (x - min_val) / (max_val - min_val)

    D_n = norm(D)
    SND_n = norm(SND)
    Clus_n = norm(Clus)
    Core_n = norm(core.astype(float))
    Fiedler_n = fiedler_abs # Already 0-1
    Dist2_n = norm(Dist2NonNbr)

    # --- Greedy Evaluation Function ---
    def evaluate(scores):
        indices = np.argsort(-scores, kind='stable')
        blocked = np.zeros(N, dtype=bool)
        count = 0
        for v in indices:
            if not blocked[v]:
                count += 1
                blocked[v] = True
                if len(nbrs[v]) > 0:
                    blocked[nbrs[v]] = True
        return count

    # --- Initial Score Construction ---
    # Hypothesis: Low degree is good. High Fiedler (peripheral) is good. 
    # High SND (neighbors are hubs) might be bad if we want to avoid hubs, 
    # OR good if we want to "kill" hubs with low cost.
    # Let's try: Minimize Degree, Maximize Fiedler, Maximize Dist2 (space).
    
    # We will search for weights w = [w_D, w_SND, w_Clus, w_Core, w_Fiedler, w_Dist2]
    # Score = -w_D*D_n - w_SND*SND_n - w_Clus*Clus_n - w_Core*Core_n + w_Fiedler*Fiedler_n + w_Dist2*Dist2_n
    
    # Initial guess
    weights = np.array([-1.0, -0.5, -0.1, -0.1, 1.0, 0.5])
    
    def calc_scores(w):
        return (w[0]*D_n + w[1]*SND_n + w[2]*Clus_n + w[3]*Core_n + 
                w[4]*Fiedler_n + w[5]*Dist2_n)

    current_scores = calc_scores(weights)
    current_size = evaluate(current_scores)
    best_scores = current_scores.copy()
    best_size = current_size

    # --- Local Search ---
    # We perform a hill-climbing search on the weight vector or directly on the scores.
    # Direct score perturbation is more flexible.
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Coarse search on weights
    # Try a few promising weight combinations
    candidate_weights = [
        [-1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [-1.0, -0.5, 0.0, 0.0, 0.5, 0.5],
        [-0.8, -0.3, -0.2, 0.0, 1.0, 0.5],
        [-1.0, 0.0, -0.5, 0.0, 0.5, 0.5],
        [-1.5, 0.0, 0.0, -0.5, 1.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0, 0.0, 1.5, 0.5],
    ]
    
    for w in candidate_weights:
        s = calc_scores(np.array(w))
        sz = evaluate(s)
        if sz > best_size:
            best_size = sz
            best_scores = s.copy()
            current_scores = s.copy()
            current_size = sz
            if time.time() > DEADLINE - 10.0:
                break

    # Phase 2: Fine-tuning via score perturbation
    # Focus on vertices that are "borderline" in the greedy selection.
    
    max_iter = 50000
    step_size = 0.1
    
    for it in range(max_iter):
        if time.time() > DEADLINE - 5.0:
            break
        
        # Identify the set of selected vertices
        indices = np.argsort(-current_scores, kind='stable')
        blocked = np.zeros(N, dtype=bool)
        selected = []
        for v in indices:
            if not blocked[v]:
                selected.append(v)
                blocked[v] = True
                if len(nbrs[v]) > 0:
                    blocked[nbrs[v]] = True
        
        # Vertices that are not selected but are not blocked by selected ones? 
        # Actually, in greedy, "blocked" means "cannot be selected".
        # Candidates for improvement:
        # 1. Selected vertices that might be "blocking" many others. (Lower their score)
        # 2. Unselected vertices that are "just behind" selected ones. (Raise their score)
        
        # Heuristic: Pick a random vertex to perturb.
        # Bias towards vertices with high impact.
        # Impact of selected v: number of neighbors it blocks that are NOT selected.
        # Impact of unselected u: how close it is to being selected.
        
        # Simple approach: Random vertex, random direction.
        v = rng.integers(0, N)
        
        # Determine direction
        # If v is selected, try lowering score (to see if it drops out and allows more)
        # If v is not selected, try raising score (to see if it gets picked)
        is_selected = v in selected
        # Note: 'in' is O(N). For N=200, it's fine. But we can use a boolean array.
        sel_mask = np.zeros(N, dtype=bool)
        for sv in selected:
            sel_mask[sv] = True
            
        if sel_mask[v]:
            # Lower score
            delta = -rng.uniform(0.01, step_size)
        else:
            # Raise score
            delta = rng.uniform(0.01, step_size)
            
        new_scores = current_scores.copy()
        new_scores[v] += delta
        
        new_size = evaluate(new_scores)
        
        if new_size > current_size:
            current_scores = new_scores
            current_size = new_size
            if current_size > best_size:
                best_size = current_size
                best_scores = new_scores.copy()
        elif new_size == current_size and rng.random() < 0.1:
            # Accept sideways moves to escape local minima
            current_scores = new_scores
        else:
            # Revert
            pass
            
    return best_scores
