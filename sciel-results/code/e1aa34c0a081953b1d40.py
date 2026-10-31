import numpy as np
import time
from itertools import combinations, product

D = 11

def generate_candidates():
    """Generate candidate integer vectors with norms squared 2 through 5."""
    cands = []
    # Norm squared = 2: two ±1's
    for combo in combinations(range(D), 2):
        for s1 in (1, -1):
            for s2 in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[combo[0]] = s1
                v[combo[1]] = s2
                cands.append(v)
    # Norm squared = 3: three ±1's
    for combo in combinations(range(D), 3):
        for signs in product((1, -1), repeat=3):
            v = np.zeros(D, dtype=np.int64)
            for i, s in enumerate(signs):
                v[combo[i]] = s
            cands.append(v)
    # Norm squared = 4: four ±1's OR one ±2 and one ±0 (no, just four ±1's)
    # Actually norm^2=4 can be: four 1's, or one 2 and one 0... but 2^2=4 so one coordinate is ±2
    # Let's include both types
    # Type A: four ±1's
    for combo in combinations(range(D), 4):
        for signs in product((1, -1), repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for i, s in enumerate(signs):
                v[combo[i]] = s
            cands.append(v)
    # Type B: one coordinate ±2, rest 0 (norm^2 = 4)
    for i in range(D):
        for s in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = 2 * s
            cands.append(v)
    # Norm squared = 5: five ±1's OR one ±2 and one ±1 (4+1=5)
    # Type A: five ±1's
    for combo in combinations(range(D), 5):
        for signs in product((1, -1), repeat=5):
            v = np.zeros(D, dtype=np.int64)
            for i, s in enumerate(signs):
                v[combo[i]] = s
            cands.append(v)
    # Type B: one ±2 and one ±1
    for i in range(D):
        for j in range(D):
            if i == j:
                continue
            for s2 in (1, -1):
                for s1 in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = 2 * s2
                    v[j] = 1 * s1
                    cands.append(v)
    
    cands = np.array(cands, dtype=np.int64)
    return cands

def construct():
    start_time = time.time()
    deadline = 145  # Safety margin before 150s kill
    
    cands = generate_candidates()
    n = len(cands)
    
    # Precompute dot products
    G = cands @ cands.T
    norms_sq = np.diag(G).copy()
    
    # For a set S, the constraint is: for all i != j in S, ||v_i - v_j|| >= max_{k in S} ||v_k||
    # ||v_i - v_j||^2 = ||v_i||^2 + ||v_j||^2 - 2 v_i.v_j
    # Let R = max_{k in S} ||v_k||
    # Need: ||v_i||^2 + ||v_j||^2 - 2 v_i.v_j >= R^2
    # Since ||v_i||^2 <= R^2 and ||v_j||^2 <= R^2, we have:
    # R^2 + R^2 - 2 v_i.v_j >= R^2 => R^2 >= 2 v_i.v_j => v_i.v_j <= R^2/2
    # But this is only sufficient, not necessary. The actual constraint is tighter when norms differ.
    
    # For simulated annealing, we need to evaluate the objective for a given set.
    # Objective: size of the set, subject to the constraint.
    # Let's define a validity check for a set of indices.
    
    def is_valid(indices):
        """Check if the set of candidate indices forms a valid kissing configuration."""
        if len(indices) == 0:
            return True
        idx_arr = np.array(indices)
        R2 = np.max(norms_sq[idx_arr])
        # For each pair, check ||v_i - v_j||^2 >= R2
        # ||v_i - v_j||^2 = norms_sq[i] + norms_sq[j] - 2*G[i,j]
        # We need this >= R2 for all pairs i != j
        n_sel = len(idx_arr)
        if n_sel < 2:
            return True
        # Build submatrix
        sub_G = G[np.ix_(idx_arr, idx_arr)]
        sub_norms = norms_sq[idx_arr]
        # Compute pairwise distances squared
        # dist2[i,j] = sub_norms[i] + sub_norms[j] - 2*sub_G[i,j]
        dist2 = sub_norms[:, None] + sub_norms[None, :] - 2 * sub_G
        # Check all off-diagonal entries >= R2
        np.fill_diagonal(dist2, np.inf)
        return np.all(dist2 >= R2)
    
    # Simulated annealing
    # State: a set of selected indices (as a boolean mask)
    # Move: add a random candidate, remove a random candidate, or swap
    
    # Initialize with a greedy solution
    active = np.ones(n, dtype=bool)
    selected = []
    
    # Greedy initialization
    order = np.random.permutation(n)
    for idx in order:
        if not active[idx]:
            continue
        # Check if adding idx keeps the set valid
        # For efficiency, check against all selected
        if selected:
            sel_arr = np.array(selected)
            # New R2 would be max(current_max, norms_sq[idx])
            current_R2 = np.max(norms_sq[sel_arr])
            new_R2 = max(current_R2, norms_sq[idx])
            # Check all pairs involving idx
            # For each j in selected: norms_sq[idx] + norms_sq[j] - 2*G[idx,j] >= new_R2
            ok = True
            for j in selected:
                d2 = norms_sq[idx] + norms_sq[j] - 2 * G[idx, j]
                if d2 < new_R2:
                    ok = False
                    break
            if not ok:
                continue
        selected.append(idx)
        active[idx] = False
    
    best_set = list(selected)
    best_size = len(best_set)
    
    # Simulated annealing
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    rng = np.random.default_rng(42)
    
    while time.time() < deadline - 5:
        # Generate a move
        move_type = rng.random()
        
        if move_type < 0.4:
            # Add a random candidate
            candidates_to_add = np.where(active)[0]
            if len(candidates_to_add) == 0:
                continue
            idx = rng.choice(candidates_to_add)
            new_selected = selected + [idx]
            if is_valid(new_selected):
                new_size = len(new_selected)
                delta_obj = new_size - len(selected)
                if delta_obj > 0 or rng.random() < np.exp(delta_obj / T):
                    selected = new_selected
                    active[idx] = False
                    if len(selected) > best_size:
                        best_size = len(selected)
                        best_set = list(selected)
            else:
                # Invalid move, reject
                pass
        elif move_type < 0.8:
            # Remove a random candidate
            if len(selected) == 0:
                continue
            idx = rng.choice(selected)
            selected.remove(idx)
            active[idx] = True
            # Always accept removal (we might add better ones later)
            # But track best
        else:
            # Swap: remove one, add another
            if len(selected) == 0:
                continue
            candidates_to_add = np.where(active)[0]
            if len(candidates_to_add) == 0:
                continue
            idx_remove = rng.choice(selected)
            idx_add = rng.choice(candidates_to_add)
            new_selected = [x for x in selected if x != idx_remove] + [idx_add]
            if is_valid(new_selected):
                new_size = len(new_selected)
                delta_obj = new_size - len(selected)
                if delta_obj > 0 or rng.random() < np.exp(delta_obj / T):
                    selected = new_selected
                    active[idx_remove] = True
                    active[idx_add] = False
                    if len(selected) > best_size:
                        best_size = len(selected)
                        best_set = list(selected)
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    if best_size == 0:
        return np.array([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)
    
    return cands[best_set]
