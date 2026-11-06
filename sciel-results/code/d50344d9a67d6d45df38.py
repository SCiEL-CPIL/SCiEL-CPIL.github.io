import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 140  # Safety margin
    
    # 1. Construct Paley Tournament on p=67
    p = 67
    squares = set()
    for x in range(1, p):
        squares.add((x * x) % p)
    
    # orientation[i][j] for i < j
    # 0: i -> j, 1: j -> i
    # We need to generate the full orientation list for n=67 first?
    # Or just generate the adjacency structure.
    
    # Let's create an adjacency matrix for Paley 67
    # adj[i][j] = 1 if i -> j, 0 if j -> i (for i != j)
    adj_67 = np.zeros((p, p), dtype=np.int8)
    for i in range(p):
        for j in range(i + 1, p):
            if (j - i) % p in squares:
                # i -> j
                adj_67[i][j] = 1
                adj_67[j][i] = 0
            else:
                # j -> i
                adj_67[i][j] = 0
                adj_67[j][i] = 1
                
    best_n = 67
    # Generate orientation list for n=67
    orientation_67 = []
    for i in range(p):
        for j in range(i + 1, p):
            if adj_67[i][j] == 1:
                orientation_67.append(0) # i -> j
            else:
                orientation_67.append(1) # j -> i
                
    best_orientation = orientation_67
    
    # 2. Try to find a valid tournament on n=66 by deleting one vertex
    target_n = 66
    m_target = target_n * (target_n - 1) // 2
    
    def get_idx(i, j, n):
        return i * (n - 1) - i * (i + 1) // 2 + (j - i - 1)
        
    min_bad_count = float('inf')
    best_66_orient = None
    best_66_bad_count = float('inf')
    
    # Precompute 4-sets for n=66?
    # There are C(66, 4) = 737,886 sets.
    # We can generate them on the fly or precompute indices.
    # Precomputing indices for all 4-sets and their 6 edges is memory heavy but fast.
    # 737,886 * 6 ints = 4.4M ints. ~35MB. Acceptable.
    
    quad_sets = []
    for a in range(target_n):
        for b in range(a + 1, target_n):
            for c in range(b + 1, target_n):
                for d in range(c + 1, target_n):
                    quad_sets.append((a, b, c, d))
                    
    num_quads = len(quad_sets)
    
    # Precompute edge indices for each quad
    # This is independent of the specific tournament, only depends on n
    edge_indices_per_quad = np.zeros((num_quads, 6), dtype=np.int32)
    for idx, (a, b, c, d) in enumerate(quad_sets):
        edge_indices_per_quad[idx, 0] = get_idx(a, b, target_n)
        edge_indices_per_quad[idx, 1] = get_idx(a, c, target_n)
        edge_indices_per_quad[idx, 2] = get_idx(a, d, target_n)
        edge_indices_per_quad[idx, 3] = get_idx(b, c, target_n)
        edge_indices_per_quad[idx, 4] = get_idx(b, d, target_n)
        edge_indices_per_quad[idx, 5] = get_idx(c, d, target_n)
        
    # Precompute which quads are affected by which edge
    edge_to_quads = [[] for _ in range(m_target)]
    for q in range(num_quads):
        for k in range(6):
            e_idx = edge_indices_per_quad[q, k]
            edge_to_quads[e_idx].append(q)
            
    def count_bad_sets(orient):
        """orient is a list or array of length m_target"""
        bad = 0
        for q_idx in range(num_quads):
            e_ab = orient[edge_indices_per_quad[q_idx, 0]]
            e_ac = orient[edge_indices_per_quad[q_idx, 1]]
            e_ad = orient[edge_indices_per_quad[q_idx, 2]]
            e_bc = orient[edge_indices_per_quad[q_idx, 3]]
            e_bd = orient[edge_indices_per_quad[q_idx, 4]]
            e_cd = orient[edge_indices_per_quad[q_idx, 5]]
            
            # a dominates: e_ab=0, e_ac=0, e_ad=0
            if e_ab == 0 and e_ac == 0 and e_ad == 0:
                continue
            # b dominates: e_ab=1, e_bc=0, e_bd=0
            if e_ab == 1 and e_bc == 0 and e_bd == 0:
                continue
            # c dominates: e_ac=1, e_bc=1, e_cd=0
            if e_ac == 1 and e_bc == 1 and e_cd == 0:
                continue
            # d dominates: e_ad=1, e_bd=1, e_cd=1
            if e_ad == 1 and e_bd == 1 and e_cd == 1:
                continue
                
            bad += 1
        return bad

    # Try deleting each vertex from Paley 67
    for v_del in range(p):
        if time.time() - start_time > deadline - 30:
            break
            
        # Map old indices to new indices
        # New indices are 0..65
        # Old indices are 0..66, excluding v_del
        mapping = {}
        new_idx = 0
        for i in range(p):
            if i == v_del:
                continue
            mapping[i] = new_idx
            new_idx += 1
            
        # Build orientation for the 66-vertex tournament
        # We need to fill an array of size m_target
        # orient_66[get_idx(new_i, new_j)]
        orient_66 = np.zeros(m_target, dtype=np.int8)
        
        # Iterate over all pairs in the new tournament
        # It's easier to iterate over old pairs and map them
        # Or iterate over new pairs and look up old values
        for i_new in range(target_n):
            for j_new in range(i_new + 1, target_n):
                # Find old i, j
                # We can reverse the mapping
                # Let's build a reverse map
                pass
        
        # Better: build reverse map
        rev_mapping = [0] * p
        new_idx = 0
        for i in range(p):
            if i == v_del:
                continue
            rev_mapping[new_idx] = i
            new_idx += 1
            
        for i_new in range(target_n):
            old_i = rev_mapping[i_new]
            for j_new in range(i_new + 1, target_n):
                old_j = rev_mapping[j_new]
                # Get direction from adj_67
                # adj_67[old_i][old_j] is 1 if old_i -> old_j
                if adj_67[old_i][old_j] == 1:
                    orient_66[get_idx(i_new, j_new, target_n)] = 0 # i_new -> j_new
                else:
                    orient_66[get_idx(i_new, j_new, target_n)] = 1 # j_new -> i_new
                    
        bad_count = count_bad_sets(orient_66)
        
        if bad_count < best_66_bad_count:
            best_66_bad_count = bad_count
            best_66_orient = orient_66.copy()
            
        if bad_count == 0:
            # Found a valid 66-tournament
            return {'n': 66, 'orientation': orient_66.tolist()}
            
    # 3. Local Search on the best 66-candidate
    if best_66_orient is not None and time.time() - start_time < deadline - 10:
        orient = best_66_orient.copy()
        current_bad = best_66_bad_count
        
        T = 5.0
        T_min = 0.01
        cooling_rate = 0.999
        last_improve = time.time()
        
        while time.time() - start_time < deadline - 5:
            # Pick random edge
            e_idx = random.randint(0, m_target - 1)
            
            # Calculate delta
            delta = 0
            for q_idx in edge_to_quads[e_idx]:
                # Check old good
                e_ab = orient[edge_indices_per_quad[q_idx, 0]]
                e_ac = orient[edge_indices_per_quad[q_idx, 1]]
                e_ad = orient[edge_indices_per_quad[q_idx, 2]]
                e_bc = orient[edge_indices_per_quad[q_idx, 3]]
                e_bd = orient[edge_indices_per_quad[q_idx, 4]]
                e_cd = orient[edge_indices_per_quad[q_idx, 5]]
                
                old_good = False
                if e_ab == 0 and e_ac == 0 and e_ad == 0: old_good = True
                elif e_ab == 1 and e_bc == 0 and e_bd == 0: old_good = True
                elif e_ac == 1 and e_bc == 1 and e_cd == 0: old_good = True
                elif e_ad == 1 and e_bd == 1 and e_cd == 1: old_good = True
                
                # Flip
                old_val = orient[e_idx]
                orient[e_idx] = 1 - old_val
                
                # Check new good
                e_ab = orient[edge_indices_per_quad[q_idx, 0]]
                e_ac = orient[edge_indices_per_quad[q_idx, 1]]
                e_ad = orient[edge_indices_per_quad[q_idx, 2]]
                e_bc = orient[edge_indices_per_quad[q_idx, 3]]
                e_bd = orient[edge_indices_per_quad[q_idx, 4]]
                e_cd = orient[edge_indices_per_quad[q_idx, 5]]
                
                new_good = False
                if e_ab == 0 and e_ac == 0 and e_ad == 0: new_good = True
                elif e_ab == 1 and e_bc == 0 and e_bd == 0: new_good = True
                elif e_ac == 1 and e_bc == 1 and e_cd == 0: new_good = True
                elif e_ad == 1 and e_bd == 1 and e_cd == 1: new_good = True
                
                # Revert
                orient[e_idx] = old_val
                
                if old_good and not new_good:
                    delta += 1
                elif not old_good and new_good:
                    delta -= 1
                    
            # Acceptance
            if delta < 0 or random.random() < np.exp(-delta / T):
                orient[e_idx] = 1 - orient[e_idx]
                current_bad += delta
                if current_bad < best_66_bad_count:
                    best_66_bad_count = current_bad
                    if current_bad == 0:
                        return {'n': 66, 'orientation': orient.tolist()}
                    last_improve = time.time()
                    
            T *= cooling_rate
            if T < T_min:
                T = T_min
                
            # If no improvement for a long time, restart T?
            # For now, just cool down.
            
    return {'n': best_n, 'orientation': best_orientation}
