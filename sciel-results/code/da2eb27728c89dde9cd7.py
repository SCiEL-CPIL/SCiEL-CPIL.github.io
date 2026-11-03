"""Combined: Support-set CSP with forward checking + sign assignment optimization for kissing number d=11."""
import math
import numpy as np
import itertools
import time


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    D = 11

    # Generate all 4-subsets of coordinates
    all_4subsets = list(itertools.combinations(range(D), 4))
    num_supports = len(all_4subsets)  # 330
    
    # Create mapping from 4-subset to index
    subset_to_idx = {s: i for i, s in enumerate(all_4subsets)}
    
    # Precompute intersection sizes between all pairs of supports
    # Two supports are compatible if they intersect in at most 1 coordinate
    # (disjoint or share exactly one coordinate)
    intersect_size = np.zeros((num_supports, num_supports), dtype=np.uint8)
    for i in range(num_supports):
        set_i = set(all_4subsets[i])
        for j in range(i + 1, num_supports):
            set_j = set(all_4subsets[j])
            inter = len(set_i & set_j)
            intersect_size[i, j] = inter
            intersect_size[j, i] = inter
    
    # Conflict matrix: 1 if supports intersect in 2 or more coordinates
    support_conflict = (intersect_size >= 2).astype(np.uint8)
    np.fill_diagonal(support_conflict, 0)
    
    # For each support, precompute the list of compatible supports
    compatible_supports = [np.where(support_conflict[i] == 0)[0] for i in range(num_supports)]
    # Remove self from compatible list
    for i in range(num_supports):
        compatible_supports[i] = compatible_supports[i][compatible_supports[i] != i]
    
    # Sign assignment: for a given set of supports, we need to assign signs
    # such that for any two selected vectors, their dot product is <= 1
    # (since max_norm_sq = 4, we need ||v_i - v_j||^2 >= 4, i.e., dot <= (4+4-4)/2 = 2)
    # Wait: ||v_i - v_j||^2 = ||v_i||^2 + ||v_j||^2 - 2*v_i.v_j >= 4
    # If both have norm_sq = 4: 4 + 4 - 2*dot >= 4 => dot <= 2
    # But we want to be safe, so let's use dot <= 1 for a stronger constraint
    
    # Actually, let me reconsider. The constraint is:
    # min pairwise distance >= max norm
    # If max norm = 2 (norm_sq = 4), then min distance >= 2, so min distance_sq >= 4
    # For two weight-4 vectors: ||v_i - v_j||^2 = 4 + 4 - 2*v_i.v_j = 8 - 2*v_i.v_j >= 4
    # So v_i.v_j <= 2
    
    # For weight-4 vectors, the dot product depends on:
    # - Number of shared coordinates with same sign: a
    # - Number of shared coordinates with opposite sign: b
    # - dot = a - b
    # - a + b = intersection size
    # So dot = 2a - intersection_size
    # We need dot <= 2, so 2a - intersection_size <= 2
    # If intersection_size = 0: dot = 0 <= 2 (always OK)
    # If intersection_size = 1: dot = 2a - 1, where a in {0,1}, so dot in {-1, 1} <= 2 (always OK)
    # If intersection_size = 2: dot = 2a - 2, where a in {0,1,2}, so dot in {-2, 0, 2} <= 2 (OK if a<=2, always true)
    # If intersection_size = 3: dot = 2a - 3, where a in {0,1,2,3}, so dot in {-3,-1,1,3}. Need dot<=2, so a<=2 (not a=3)
    # If intersection_size = 4: dot = 2a - 4, where a in {0,1,2,3,4}, so dot in {-4,-2,0,2,4}. Need dot<=2, so a<=3 (not a=4)
    
    # So the constraint is:
    # - Supports with intersection 0 or 1: always compatible (any signs)
    # - Supports with intersection 2: always compatible (any signs, max dot = 2)
    # - Supports with intersection 3: need at most 2 coordinates with same sign (i.e., at least 1 opposite)
    # - Supports with intersection 4: need at most 3 coordinates with same sign (i.e., at least 1 opposite)
    
    # For the CSP, let's first select a maximal set of supports that are pairwise compatible
    # (intersection <= 2), then optimize signs for the remaining constraints.
    
    # Actually, let's be more aggressive: allow intersection up to 3, but handle sign constraints.
    # This is more complex. Let's start with intersection <= 2 for simplicity.
    
    # Greedy selection of supports with forward checking
    def greedy_select_supports(seed=42):
        rng = np.random.default_rng(seed)
        remaining = np.ones(num_supports, dtype=bool)
        selected = []
        
        while remaining.any():
            if time.time() > deadline - 2.0:
                break
            
            # Compute degree for remaining supports
            deg = np.zeros(num_supports, dtype=np.int32)
            for i in np.where(remaining)[0]:
                deg[i] = np.sum(support_conflict[i] & remaining)
            
            # Pick support with minimum degree
            min_deg = deg[remaining].min()
            candidates = np.where(remaining & (deg == min_deg))[0]
            if len(candidates) == 0:
                break
            idx = rng.choice(candidates)
            selected.append(idx)
            
            # Remove incompatible supports
            incompatible = np.where(support_conflict[idx])[0]
            remaining[incompatible] = False
            remaining[idx] = False
        
        return selected
    
    # Sign assignment for a given set of supports
    def assign_signs(support_indices, max_iters=10000):
        """Assign signs to minimize maximum dot product violation."""
        k = len(support_indices)
        if k == 0:
            return []
        
        # For each support, create a vector of signs (4 values, each +1 or -1)
        # Represent as integer: bit 0-3 for the 4 positions, 0=+1, 1=-1
        # Total 16 possible sign assignments per support
        
        # Precompute for each pair of supports with intersection >= 3,
        # which sign assignments are compatible
        # This is expensive, so let's use a simpler approach:
        # Start with random signs, then do local search to reduce violations
        
        # Random initial assignment
        signs = np.random.randint(0, 2, size=(k, 4))
        
        # For each support, create the actual vector
        def make_vector(support_idx, sign_bits):
            support = all_4subsets[support_idx]
            v = np.zeros(D, dtype=np.int32)
            for pos, bit in zip(support, sign_bits):
                v[pos] = 1 if bit == 0 else -1
            return v
        
        # Compute all vectors
        vectors = [make_vector(support_indices[i], signs[i]) for i in range(k)]
        
        # Compute dot products
        def compute_violations():
            max_viol = 0
            for i in range(k):
                for j in range(i + 1, k):
                    dot = np.dot(vectors[i], vectors[j])
                    if dot > 2:
                        max_viol = max(max_viol, dot - 2)
            return max_viol
        
        current_viol = compute_violations()
        
        # Local search: flip individual sign bits to reduce violations
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline - 1.0:
            improved = False
            iters += 1
            
            # Try flipping each sign bit
            for i in range(k):
                for bit_pos in range(4):
                    old_bit = signs[i, bit_pos]
                    signs[i, bit_pos] = 1 - old_bit
                    vectors[i] = make_vector(support_indices[i], signs[i])
                    
                    new_viol = compute_violations()
                    if new_viol < current_viol:
                        current_viol = new_viol
                        improved = True
                    else:
                        # Revert
                        signs[i, bit_pos] = old_bit
                        vectors[i] = make_vector(support_indices[i], signs[i])
                    
                    if time.time() > deadline - 1.0:
                        return vectors
        
        return vectors
    
    # Better approach: use the full vector set from the current program
    # and apply the support-set insight to improve the greedy initialization
    
    # Generate all candidates: weight-4 (5280), weight-2 (220), weight-1 (22)
    candidates_list = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates_list.append(v)
    for positions in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates_list.append(v)
    for pos in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[pos] = s
            candidates_list.append(v)
    
    candidates = np.array(candidates_list, dtype=np.int8)
    n = len(candidates)  # 5522
    
    # Precompute norms squared
    norms_sq = np.sum(candidates.astype(np.int32) ** 2, axis=1)
    max_norm_sq = int(norms_sq.max())  # 4
    
    # Precompute dot product matrix
    dot = candidates.astype(np.int32) @ candidates.T.astype(np.int32)
    
    # Constraint: ||v_i - v_j||^2 >= max_norm_sq
    threshold = (norms_sq[:, np.newaxis] + norms_sq[np.newaxis, :] - max_norm_sq) / 2.0
    compat = (dot <= threshold).astype(np.uint8)
    np.fill_diagonal(compat, 1)
    
    # conflict[i,j] = 1 if i and j CANNOT both be selected
    conflict = (1 - compat).astype(np.uint8)
    np.fill_diagonal(conflict, 0)
    
    # Fast Greedy Initialization using support-set insight
    def greedy_init_enhanced():
        # First, select a good set of 4-subset supports
        support_sel = greedy_select_supports(seed=42)
        
        # For each selected support, add one vector with that support
        # Choose the sign assignment that minimizes conflicts with already selected
        selected = []
        selected_set = set()
        
        for sup_idx in support_sel:
            support = all_4subsets[sup_idx]
            # Find all candidates with this support
            mask = np.zeros(n, dtype=bool)
            for pos in support:
                pass  # Need to check if candidate has 1s exactly at these positions
            
            # Simpler: just pick a random candidate with this support
            # Find candidates with weight 4 and support matching
            weight_4 = np.where(norms_sq == 4)[0]
            candidates_with_support = []
            for c_idx in weight_4:
                v = candidates[c_idx]
                if all(v[pos] != 0 for pos in support) and all(v[pos] == 0 for pos in range(D) if pos not in support):
                    candidates_with_support.append(c_idx)
            
            if not candidates_with_support:
                continue
            
            # Pick the one with minimum conflict with already selected
            best_c = None
            best_conf = float('inf')
            for c in candidates_with_support:
                conf_count = np.sum(conflict[c] & np.isin(np.arange(n), list(selected_set)))
                if conf_count < best_conf:
                    best_conf = conf_count
                    best_c = c
            
            if best_c is not None:
                selected.append(best_c)
                selected_set.add(best_c)
        
        # Add remaining candidates greedily
        remaining = np.ones(n, dtype=bool)
        for s in selected:
            remaining[s] = False
            remaining[conflict[s] > 0] = False
        
        while remaining.any():
            if time.time() > deadline - 5.0:
                break
            
            deg = (conflict * remaining[np.newaxis, :]).sum(axis=1)
            deg[~remaining] = n + 1
            min_deg = deg.min()
            if min_deg >= n:
                break
            mask = (deg == min_deg) & remaining
            picks = np.where(mask)[0]
            if len(picks) == 0:
                break
            new_idx = picks[0]
            selected.append(new_idx)
            remaining = remaining & (conflict[new_idx] == 0)
            remaining[new_idx] = False
        
        return selected
    
    init_sel = greedy_init_enhanced()
    x = np.zeros(n, dtype=np.uint8)
    for i in init_sel:
        x[i] = 1
    
    current_count = len(init_sel)
    current_violations = 0  # Greedy ensures validity
    
    # Track best valid solution
    best_x = x.copy()
    best_count = current_count
    
    # Simulated Annealing with swap moves
    Lambda = 100.0
    T_start = 50.0
    T_end = 0.1
    max_steps = 150000
    
    for step in range(max_steps):
        if time.time() > deadline - 5.0:
            break
            
        alpha = (T_end / T_start) ** (1.0 / max_steps)
        T = T_start * (alpha ** step)
        
        move_type = np.random.random()
        old_count = current_count
        old_viol = current_violations
        old_E = Lambda * old_viol - old_count
        
        new_count = old_count
        new_viol = old_viol
        changes = []
        
        if move_type < 0.4:
            unselected = np.where(x == 0)[0]
            if len(unselected) == 0: continue
            i = unselected[np.random.randint(len(unselected))]
            delta_v = int(np.dot(conflict[i], x))
            new_viol = old_viol + delta_v
            new_count = old_count + 1
            changes.append((i, 0))
            
        elif move_type < 0.8:
            selected = np.where(x == 1)[0]
            if len(selected) == 0: continue
            i = selected[np.random.randint(len(selected))]
            delta_v = -int(np.dot(conflict[i], x))
            new_viol = old_viol + delta_v
            new_count = old_count - 1
            changes.append((i, 1))
            
        else:
            selected = np.where(x == 1)[0]
            unselected = np.where(x == 0)[0]
            if len(selected) == 0 or len(unselected) == 0: continue
            i_remove = selected[np.random.randint(len(selected))]
            i_add = unselected[np.random.randint(len(unselected))]
            
            viol_remove = int(np.dot(conflict[i_remove], x))
            viol_add_with_old = int(np.dot(conflict[i_add], x))
            delta_v = -viol_remove + (viol_add_with_old - int(conflict[i_add, i_remove]))
            new_viol = old_viol + delta_v
            new_count = old_count
            changes.append((i_remove, 1))
            changes.append((i_add, 0))
        
        new_E = Lambda * new_viol - new_count
        delta_E = new_E - old_E
        
        accepted = False
        if delta_E <= 0:
            accepted = True
        else:
            prob = math.exp(-delta_E / T)
            if np.random.random() < prob:
                accepted = True
        
        if accepted:
            for idx, old_val in changes:
                x[idx] = 1 - old_val
            
            current_violations = new_viol
            current_count = new_count
            
            if new_viol == 0 and new_count > best_count:
                best_count = new_count
                best_x = x.copy()
        
        # Periodic local search
        if step % 5000 == 0 and step > 0:
            # Try to add more vectors
            unselected = np.where(x == 0)[0]
            for i in unselected:
                if np.dot(conflict[i], x) == 0:
                    x[i] = 1
                    current_count += 1
                    if current_violations == 0 and current_count > best_count:
                        best_count = current_count
                        best_x = x.copy()
    
    if best_count == 0:
        v = np.zeros(D, dtype=np.int64)
        v[0] = 1
        v[1] = 1
        return v.reshape(1, -1)
        
    result = candidates[best_x.astype(bool)]
    return result.astype(np.int64)
