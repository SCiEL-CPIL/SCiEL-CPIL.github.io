import numpy as np
from itertools import combinations
import time
import random

def kissing_number11() -> np.ndarray:
    d = 11
    deadline = time.time() + 590  # safety margin
    
    all_4sets = list(combinations(range(d), 4))
    n_sets = len(all_4sets)  # 330
    
    # Precompute bitmask for each 4-set
    masks = []
    for s in all_4sets:
        m = 0
        for idx in s:
            m |= (1 << idx)
        masks.append(m)
    
    # Total candidates: each 4-set has 16 sign patterns
    # Candidate index = set_idx * 16 + sign_idx
    # sign_idx encodes 4 bits for the 4 signs
    
    total_candidates = n_sets * 16
    
    # Precompute for each candidate: which mask it belongs to
    cand_mask = [masks[i // 16] for i in range(total_candidates)]
    
    # Precompute popcount of intersection for each pair of 4-sets
    # Two candidates conflict if their dot product > 2
    # For weight-4 vectors with signs:
    # If supports are disjoint: dot = 0
    # If share 1 index: dot = ±1
    # If share 2 indices: dot = ±1 ±1 = -2, 0, or 2
    # If share 3 indices: dot = ±1 ±1 ±1 = -3, -1, 1, 3 → can be > 2
    # If share 4 indices (same support): dot = ±1±1±1±1 = -4,-2,0,2,4
    
    # So conflict (dot > 2) happens when:
    # - Same support (4 shared) and dot = 4 (same signs)
    # - 3 shared and dot = 3 (same signs on 3 shared, 4th different)
    # - 2 shared: max dot = 2, so no conflict
    # - 1 or 0 shared: no conflict
    
    # So we need: no two candidates with same support and same signs
    # And no two candidates with 3 shared indices and matching signs on those 3
    
    # Let me precompute the conflict structure more carefully.
    # For candidate a (support sa, signs sa) and candidate b (support sb, signs sb):
    # Let shared = sa ∩ sb, |shared| = k
    # dot = sum over i in shared of sa[i] * sb[i]
    # Conflict if dot > 2
    
    # For k=4 (same support): dot ranges from -4 to 4. Conflict if dot=4 (all same signs)
    # For k=3: dot ranges from -3 to 3. Conflict if dot=3 (all 3 shared same signs)
    # For k=2: dot ranges from -2 to 2. No conflict (max=2, not >2)
    # For k<=1: no conflict
    
    # So conflicts only for k>=3.
    
    # Strategy: Represent each candidate by (support_tuple, sign_tuple)
    # For fast checking, precompute for each pair of supports with |intersection|>=3
    
    # Let me build a list of (support_idx_a, support_idx_b) pairs where |intersection| >= 3
    conflict_support_pairs = []
    for i in range(n_sets):
        for j in range(i+1, n_sets):
            inter = masks[i] & masks[j]
            # Count bits
            cnt = bin(inter).count('1')
            if cnt >= 3:
                conflict_support_pairs.append((i, j, cnt))
    
    # Now for each candidate, determine which other candidates it conflicts with
    # This is expensive. Let me use a different approach:
    # During greedy, for each new candidate, check against all selected.
    
    # To speed up: for each selected candidate, store its support_idx and sign pattern.
    # When checking a new candidate (support_idx s, signs sig):
    #   For each selected (support_idx t, signs sig2):
    #     if s == t: check if signs match exactly → conflict
    #     elif |support_s ∩ support_t| == 3: check if signs match on the 3 shared → conflict
    #     else: no conflict
    
    # Precompute for each pair of supports with intersection >= 3: which indices are shared
    support_inter = {}  # (i,j) -> tuple of shared indices
    for i, j, cnt in conflict_support_pairs:
        inter = masks[i] & masks[j]
        shared = tuple(idx for idx in range(d) if (inter >> idx) & 1)
        support_inter[(i,j)] = shared
        support_inter[(j,i)] = shared
    
    # Sign pattern: encode as 4-bit integer, bit k corresponds to the k-th element of the support
    # For support (a,b,c,d): sign bit 0 = sign at a, bit 1 = sign at b, etc.
    # sign_val = 0 or 1 (0 for +1, 1 for -1)
    # Actual sign = 1 if bit=0, -1 if bit=1
    
    def get_sign(sign_val, pos_in_support):
        return 1 if ((sign_val >> pos_in_support) & 1) == 0 else -1
    
    def dot_product(support_a, signs_a, support_b, signs_b):
        # Compute dot product
        sa = set(support_a)
        sb = set(support_b)
        shared = sa & sb
        if not shared:
            return 0
        # Map position in support to index
        pos_a = {idx: i for i, idx in enumerate(support_a)}
        pos_b = {idx: i for i, idx in enumerate(support_b)}
        dot = 0
        for idx in shared:
            sa_sign = get_sign(signs_a, pos_a[idx])
            sb_sign = get_sign(signs_b, pos_b[idx])
            dot += sa_sign * sb_sign
        return dot
    
    # For speed, precompute sign arrays
    all_signs = list(range(16))
    
    # Greedy with random ordering
    best_points = []
    rng = random.Random(42)
    
    def greedy_pass(order, time_limit):
        selected = []  # list of (support_idx, sign_val, point_array)
        support_to_candidates = {}  # support_idx -> list of selected sign_vals
        
        for cand_idx in order:
            if time.time() > time_limit:
                break
            s_idx = cand_idx // 16
            sign_val = cand_idx % 16
            support = all_4sets[s_idx]
            
            valid = True
            for (sel_s_idx, sel_sign, _) in selected:
                if sel_s_idx == s_idx:
                    # Same support: conflict only if same signs
                    if sel_sign == sign_val:
                        valid = False
                        break
                else:
                    key = (s_idx, sel_s_idx)
                    if key in support_inter:
                        shared = support_inter[key]
                        # Check if signs match on all shared indices
                        pos_new = {idx: i for i, idx in enumerate(support)}
                        pos_sel = {idx: i for i, idx in enumerate(all_4sets[sel_s_idx])}
                        match_count = 0
                        for idx in shared:
                            new_sign = get_sign(sign_val, pos_new[idx])
                            sel_sign_bit = get_sign(sel_sign, pos_sel[idx])
                            if new_sign == sel_sign_bit:
                                match_count += 1
                        if match_count == len(shared):
                            valid = False
                            break
            if valid:
                # Create point
                v = np.zeros(d, dtype=np.int64)
                for i, idx in enumerate(support):
                    v[idx] = get_sign(sign_val, i)
                selected.append((s_idx, sign_val, v))
        
        return selected
    
    # Generate all candidate indices
    all_cands = list(range(total_candidates))
    
    num_tries = 200
    best_count = 0
    
    for trial in range(num_tries):
        if time.time() > deadline - 30:
            break
        rng.shuffle(all_cands)
        selected = greedy_pass(all_cands, deadline - 30)
        if len(selected) > best_count:
            best_count = len(selected)
            best_points = [pt for (_, _, pt) in selected]
    
    # Local search: try to add more
    if best_points:
        selected_set = set()
        for (s_idx, sign_val, pt) in [(i//16, i%16, best_points[i//16*16 + i%16]) for i in range(len(best_points))]:
            selected_set.add((s_idx, sign_val))
        
        # Actually, let me redo this more carefully
        # Reconstruct selected from best_points
        selected_info = []
        for pt in best_points:
            non_zero = [i for i in range(d) if pt[i] != 0]
            if len(non_zero) == 4:
                support = tuple(non_zero)
                s_idx = all_4sets.index(support)
                sign_val = 0
                for i, idx in enumerate(support):
                    if pt[idx] == -1:
                        sign_val |= (1 << i)
                selected_info.append((s_idx, sign_val))
        
        selected_set = set(selected_info)
        
        # Try to add any candidate not in selected_set
        for cand_idx in all_cands:
            if time.time() > deadline - 20:
                break
            s_idx = cand_idx // 16
            sign_val = cand_idx % 16
            if (s_idx, sign_val) in selected_set:
                continue
            
            valid = True
            for (sel_s_idx, sel_sign) in selected_info:
                if sel_s_idx == s_idx:
                    if sel_sign == sign_val:
                        valid = False
                        break
                else:
                    key = (s_idx, sel_s_idx)
                    if key in support_inter:
                        shared = support_inter[key]
                        support = all_4sets[s_idx]
                        sel_support = all_4sets[sel_s_idx]
                        pos_new = {idx: i for i, idx in enumerate(support)}
                        pos_sel = {idx: i for i, idx in enumerate(sel_support)}
                        match_count = 0
                        for idx in shared:
                            new_sign = get_sign(sign_val, pos_new[idx])
                            sel_sign_bit = get_sign(sel_sign, pos_sel[idx])
                            if new_sign == sel_sign_bit:
                                match_count += 1
                        if match_count == len(shared):
                            valid = False
                            break
            if valid:
                v = np.zeros(d, dtype=np.int64)
                support = all_4sets[s_idx]
                for i, idx in enumerate(support):
                    v[idx] = get_sign(sign_val, i)
                best_points.append(v)
                selected_info.append((s_idx, sign_val))
                selected_set.add((s_idx, sign_val))
    
    if not best_points:
        # Fallback: return a minimal valid set
        v1 = np.zeros(d, dtype=np.int64)
        v1[0] = 1; v1[1] = 1; v1[2] = 1; v1[3] = 1
        v2 = np.zeros(d, dtype=np.int64)
        v2[4] = 1; v2[5] = 1; v2[6] = 1; v2[7] = 1
        best_points = [v1, v2]
    
    points_array = np.array(best_points, dtype=np.int64)
    return np.around(points_array).astype(np.int64)
