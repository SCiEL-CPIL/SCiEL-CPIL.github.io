import time
import random
import math
import numpy as np

def construct():
    d = 7
    n = 1 << d  # 128 vertices
    deadline = time.time() + 118  # Safety margin

    # 1. Precompute edges
    all_edges = []
    edge_idx = {}
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(all_edges)
                all_edges.append((x, y))
                edge_idx[(x, y)] = idx
    m = len(all_edges)  # 448

    # 2. Precompute faces and incidence
    face_edges = []
    edge_faces = [[] for _ in range(m)]
    
    for i in range(d):
        for j in range(i + 1, d):
            for x in range(n):
                if (x >> i) & 1 == 0 and (x >> j) & 1 == 0:
                    a = x
                    b = x ^ (1 << i)
                    c = x ^ (1 << j)
                    d2 = x ^ (1 << i) ^ (1 << j)
                    
                    e1 = edge_idx[(min(a, b), max(a, b))]
                    e2 = edge_idx[(min(b, d2), max(b, d2))]
                    e3 = edge_idx[(min(d2, c), max(d2, c))]
                    e4 = edge_idx[(min(c, a), max(c, a))]
                    
                    fid = len(face_edges)
                    face_edges.append([e1, e2, e3, e4])
                    for ei in (e1, e2, e3, e4):
                        edge_faces[ei].append(fid)
                        
    n_faces = len(face_edges)  # 672
    
    # Build numpy incidence matrix: face_edge_inc[f, e] = 1 if edge e in face f
    face_edge_inc = np.zeros((n_faces, m), dtype=np.int8)
    for f in range(n_faces):
        for e in face_edges[f]:
            face_edge_inc[f, e] = 1
    
    # edge_face_inc: for each edge, list of faces (as numpy array)
    edge_face_list = [np.array(edge_faces[e], dtype=np.int32) for e in range(m)]
    
    # 3. Greedy hitting set with randomization
    def greedy_init(seed=None):
        if seed is not None:
            random.seed(seed)
        
        R = np.zeros(m, dtype=bool)
        face_count = np.zeros(n_faces, dtype=np.int8)
        uncovered_mask = np.ones(n_faces, dtype=bool)
        num_uncovered = n_faces
        
        while num_uncovered > 0:
            # Find edge covering most uncovered faces
            best_e = -1
            best_gain = -1
            for e in range(m):
                if R[e]:
                    continue
                # Count uncovered faces this edge covers
                faces = edge_face_list[e]
                gain = np.sum(uncovered_mask[faces])
                if gain > best_gain:
                    best_gain = gain
                    best_e = e
                elif gain == best_gain and gain > 0:
                    # Random tie-breaking for diversity
                    if random.random() < 0.3:
                        best_e = e
            
            if best_e == -1 or best_gain == 0:
                break
                
            R[best_e] = True
            faces = edge_face_list[best_e]
            newly_covered = np.sum(uncovered_mask[faces])
            face_count[faces] += 1
            uncovered_mask[faces] = False
            num_uncovered -= newly_covered
            
        return R, face_count, num_uncovered

    # 4. SA optimization on R (minimize |R| subject to all faces covered)
    LAMBDA = 50.0  # Heavy penalty for uncovered faces

    def get_cost(size, unc):
        return size + LAMBDA * unc

    def sa_optimize(R_init, face_count_init, num_uncovered_init, time_budget):
        R = R_init.copy()
        face_count = face_count_init.copy()
        current_size = int(np.sum(R))
        current_uncovered = num_uncovered_init
        current_cost = get_cost(current_size, current_uncovered)
        
        best_size = current_size
        best_R = R.copy()
        best_uncovered = current_uncovered
        
        T = 3.0
        T_min = 0.01
        cooling = 0.9995
        
        end_time = time.time() + time_budget
        rng = random.Random()
        
        while time.time() < end_time:
            if T < T_min:
                break
            
            # Choose move type
            move_type = rng.random()
            
            if move_type < 0.6:
                # Single flip
                e = rng.randint(0, m - 1)
                faces = edge_face_list[e]
                
                if R[e]:
                    # Try removing e
                    # Faces that become uncovered: those with count==1
                    inc_unc = int(np.sum(face_count[faces] == 1))
                    new_size = current_size - 1
                    new_uncovered = current_uncovered + inc_unc
                    delta_cost = (new_size - current_size) + LAMBDA * (new_uncovered - current_uncovered)
                    
                    # Apply
                    R[e] = False
                    face_count[faces] -= 1
                    current_size = new_size
                    current_uncovered = new_uncovered
                    current_cost += delta_cost
                else:
                    # Try adding e
                    dec_unc = int(np.sum(face_count[faces] == 0))
                    new_size = current_size + 1
                    new_uncovered = current_uncovered - dec_unc
                    delta_cost = (new_size - current_size) + LAMBDA * (new_uncovered - current_uncovered)
                    
                    R[e] = True
                    face_count[faces] += 1
                    current_size = new_size
                    current_uncovered = new_uncovered
                    current_cost += delta_cost
                
                # Accept/reject
                if delta_cost < 0 or rng.random() < math.exp(-delta_cost / T):
                    pass  # accept
                else:
                    # Revert
                    if not R[e]:  # we added it, so remove
                        R[e] = False
                        face_count[faces] -= 1
                        current_size -= 1
                        current_uncovered += dec_unc
                        current_cost -= delta_cost
                    else:  # we removed it, so add back
                        R[e] = True
                        face_count[faces] += 1
                        current_size += 1
                        current_uncovered -= inc_unc
                        current_cost -= delta_cost
            
            else:
                # Swap: remove one from R, add one not in R
                # Find candidates
                in_R = np.where(R)[0]
                not_in_R = np.where(~R)[0]
                
                if len(in_R) == 0 or len(not_in_R) == 0:
                    T *= cooling
                    continue
                    
                e_rem = in_R[rng.randint(0, len(in_R) - 1)]
                e_add = not_in_R[rng.randint(0, len(not_in_R) - 1)]
                
                faces_rem = edge_face_list[e_rem]
                faces_add = edge_face_list[e_add]
                
                # Compute delta correctly
                # Get all affected faces
                all_affected = np.unique(np.concatenate([faces_rem, faces_add]))
                old_counts = face_count[all_affected].copy()
                
                # New counts
                new_counts = old_counts.copy()
                for f in faces_rem:
                    new_counts[f] -= 1
                for f in faces_add:
                    new_counts[f] += 1
                
                inc_unc = int(np.sum((old_counts > 0) & (new_counts == 0)))
                dec_unc = int(np.sum((old_counts == 0) & (new_counts > 0)))
                
                new_uncovered = current_uncovered - dec_unc + inc_unc
                delta_cost = LAMBDA * (new_uncovered - current_uncovered)  # size unchanged
                
                # Apply
                R[e_rem] = False
                R[e_add] = True
                face_count[faces_rem] -= 1
                face_count[faces_add] += 1
                current_uncovered = new_uncovered
                current_cost += delta_cost
                
                if delta_cost < 0 or rng.random() < math.exp(-delta_cost / T):
                    pass  # accept
                else:
                    # Revert
                    R[e_rem] = True
                    R[e_add] = False
                    face_count[faces_rem] += 1
                    face_count[faces_add] -= 1
                    current_uncovered = current_uncovered + dec_unc - inc_unc
                    current_cost -= delta_cost
            
            # Track best valid solution
            if current_uncovered == 0 and current_size < best_size:
                best_size = current_size
                best_R = R.copy()
                best_uncovered = 0
            
            T *= cooling
        
        return best_R, best_size, best_uncovered

    # 5. Polish: try to remove edges from R one by one
    def polish(R):
        R = R.copy()
        face_count = np.zeros(n_faces, dtype=np.int8)
        for e in range(m):
            if R[e]:
                face_count[edge_face_list[e]] += 1
        
        improved = True
        while improved:
            improved = False
            # Try removing each edge in R
            for e in range(m):
                if not R[e]:
                    continue
                faces = edge_face_list[e]
                if np.all(face_count[faces] > 1):
                    # Can remove
                    R[e] = False
                    face_count[faces] -= 1
                    improved = True
                    break  # Restart scan after change
        
        return R

    # 6. Main search loop with multiple restarts
    best_overall_size = m  # Start with worst case
    best_overall_R = np.ones(m, dtype=bool)
    
    # First, get a good initial solution
    best_init_size = m
    best_init_R = None
    best_init_fc = None
    best_init_unc = n_faces
    
    for attempt in range(20):
        R_g, fc_g, unc_g = greedy_init(seed=attempt * 12345)
        sz = int(np.sum(R_g))
        if unc_g == 0 and sz < best_init_size:
            best_init_size = sz
            best_init_R = R_g
            best_init_fc = fc_g
            best_init_unc = 0
    
    if best_init_R is None:
        # Fallback: use first greedy result
        best_init_R, best_init_fc, best_init_unc = greedy_init(seed=0)
        best_init_size = int(np.sum(best_init_R))
    
    # Polish the best initial solution
    polished_R = polish(best_init_R)
    polished_size = int(np.sum(polished_R))
    if polished_size < best_init_size:
        best_init_R = polished_R
        best_init_size = polished_size
        # Recompute face counts
        best_init_fc = np.zeros(n_faces, dtype=np.int8)
        for e in range(m):
            if best_init_R[e]:
                best_init_fc[edge_face_list[e]] += 1
        best_init_unc = int(np.sum(best_init_fc == 0))
    
    if best_init_unc == 0 and best_init_size < best_overall_size:
        best_overall_size = best_init_size
        best_overall_R = best_init_R.copy()
    
    # Now run SA with multiple restarts
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        # Split time among SA runs
        n_sa_runs = min(5, max(1, int(remaining_time / 20)))
        time_per_run = remaining_time / n_sa_runs - 2  # 2s margin
        
        for run in range(n_sa_runs):
            if time.time() >= deadline - 3:
                break
            
            # Create perturbed initial solution
            R_start = best_init_R.copy()
            # Randomly flip some edges
            n_flips = random.randint(5, 20)
            for _ in range(n_flips):
                e = random.randint(0, m - 1)
                R_start[e] = not R_start[e]
            
            # Recompute face counts
            fc_start = np.zeros(n_faces, dtype=np.int8)
            for e in range(m):
                if R_start[e]:
                    fc_start[edge_face_list[e]] += 1
            unc_start = int(np.sum(fc_start == 0))
            sz_start = int(np.sum(R_start))
            
            R_sa, sz_sa, unc_sa = sa_optimize(R_start, fc_start, unc_start, time_per_run)
            
            # Polish result
            R_polished = polish(R_sa)
            sz_polished = int(np.sum(R_polished))
            
            if unc_sa == 0 and sz_sa < best_overall_size:
                best_overall_size = sz_sa
                best_overall_R = R_sa.copy()
            
            if sz_polished < best_overall_size:
                # Check validity
                fc_check = np.zeros(n_faces, dtype=np.int8)
                for e in range(m):
                    if R_polished[e]:
                        fc_check[edge_face_list[e]] += 1
                if np.all(fc_check > 0):
                    best_overall_size = sz_polished
                    best_overall_R = R_polished.copy()
    
    # Final polish
    best_overall_R = polish(best_overall_R)
    best_overall_size = int(np.sum(best_overall_R))
    
    # Verify validity
    fc_final = np.zeros(n_faces, dtype=np.int8)
    for e in range(m):
        if best_overall_R[e]:
            fc_final[edge_face_list[e]] += 1
    
    if not np.all(fc_final > 0):
        # Invalid - use greedy fallback
        best_overall_R, _, _ = greedy_init(seed=0)
        best_overall_size = int(np.sum(best_overall_R))
    
    # 7. Construct output
    out = []
    for e in range(m):
        if not best_overall_R[e]:
            x, y = all_edges[e]
            out.append([x, y])
            
    return out

if __name__ == "__main__":
    t0 = time.time()
    edges = construct()
    t1 = time.time()
    print(f"Time: {t1-t0:.2f}s, Edges: {len(edges)}")
    
    # Validate
    d = 7
    n = 1 << d
    selected_set = set()
    for u, v in edges:
        selected_set.add((min(u, v), max(u, v)))
        
    valid = True
    for i in range(d):
        for j in range(i + 1, d):
            for x in range(n):
                if (x >> i) & 1 == 0 and (x >> j) & 1 == 0:
                    a = x
                    b = x ^ (1 << i)
                    c = x ^ (1 << j)
                    d2 = x ^ (1 << i) ^ (1 << j)
                    
                    e1 = (min(a, b), max(a, b))
                    e2 = (min(b, d2), max(b, d2))
                    e3 = (min(d2, c), max(d2, c))
                    e4 = (min(c, a), max(c, a))
                    
                    cnt = 0
                    if e1 in selected_set: cnt += 1
                    if e2 in selected_set: cnt += 1
                    if e3 in selected_set: cnt += 1
                    if e4 in selected_set: cnt += 1
                    
                    if cnt == 4:
                        print(f"Invalid face found at {x}, dims {i},{j}")
                        valid = False
                        break
            if not valid: break
        if not valid: break
    if valid:
        print("Valid!")
