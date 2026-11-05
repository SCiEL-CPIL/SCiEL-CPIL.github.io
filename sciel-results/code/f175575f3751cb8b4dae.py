import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit (or 150s if evaluator is lenient, but prompt says 120s finish)
    
    d = 7
    n_vertices = 1 << d
    n_edges = d * (1 << (d - 1)) # 448
    
    # 1. Precompute edges and face incidences
    # Edge (u, v) where u < v and u^v is power of 2
    edges = []
    edge_to_idx = {}
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                idx = len(edges)
                edges.append((u, v))
                edge_to_idx[(u, v)] = idx
                
    # Faces: defined by 2 bits i, j and fixed other bits.
    # Total faces: C(7, 2) * 2^(7-2) = 21 * 32 = 672
    face_edges = []
    edge_faces = [[] for _ in range(n_edges)]
    
    for i in range(d):
        for j in range(i + 1, d):
            mask_i = 1 << i
            mask_j = 1 << j
            # Iterate over all vertices x where bit i is 0 and bit j is 0
            for x in range(n_vertices):
                if (x & mask_i) == 0 and (x & mask_j) == 0:
                    v1 = x
                    v2 = x | mask_i
                    v3 = x | mask_j
                    v4 = x | mask_i | mask_j
                    
                    e1 = (min(v1, v2), max(v1, v2))
                    e2 = (min(v1, v3), max(v1, v3))
                    e3 = (min(v2, v4), max(v2, v4))
                    e4 = (min(v3, v4), max(v3, v4))
                    
                    idxs = [edge_to_idx[e1], edge_to_idx[e2], edge_to_idx[e3], edge_to_idx[e4]]
                    face_idx = len(face_edges)
                    face_edges.append(idxs)
                    for idx in idxs:
                        edge_faces[idx].append(face_idx)
                        
    n_faces = len(face_edges)
    
    # State:
    # present: bool array of length n_edges
    # face_counts: int array of length n_faces
    
    def get_score(present):
        return np.sum(present)
        
    def is_valid_addition(present, face_counts, idx):
        if present[idx]:
            return False
        for f in edge_faces[idx]:
            if face_counts[f] == 3:
                return False
        return True
        
    def add_edge(present, face_counts, idx):
        present[idx] = True
        for f in edge_faces[idx]:
            face_counts[f] += 1
            
    def remove_edge(present, face_counts, idx):
        present[idx] = False
        for f in edge_faces[idx]:
            face_counts[f] -= 1
            
    def greedy_init(rng):
        present = np.zeros(n_edges, dtype=bool)
        face_counts = np.zeros(n_faces, dtype=np.int32)
        order = list(range(n_edges))
        rng.shuffle(order)
        for idx in order:
            if is_valid_addition(present, face_counts, idx):
                add_edge(present, face_counts, idx)
        return present, face_counts

    def local_search(present, face_counts, rng, time_limit):
        current_score = np.sum(present)
        no_improve_count = 0
        max_no_improve = 5000 # Reset if stuck too long? Or just break.
        
        while time.time() < time_limit:
            improved = False
            
            # 1. Try adding edges (Greedy fill)
            # We can iterate through all absent edges and try to add them.
            # To be faster, we can just pick random absent edges or iterate sequentially.
            # Sequential is deterministic, random is better for diversity.
            # Let's try a mix: iterate through all, if addable, add.
            # This is O(N_edges * 6) per pass. 448 * 6 is small.
            
            added_any = False
            for idx in range(n_edges):
                if not present[idx] and is_valid_addition(present, face_counts, idx):
                    add_edge(present, face_counts, idx)
                    current_score += 1
                    added_any = True
            if added_any:
                improved = True
                no_improve_count = 0
                continue # Restart scan as adding one might open up others? 
                         # Actually, adding edges only closes faces, so it never opens up others.
                         # So we can just do one pass of adds.
            
            # 2. Try 1-for-1 swaps
            # Remove e1, Add e2. Score stays same, but might allow future adds?
            # Or, if we are stuck, a swap might change the configuration to allow a 2-for-1 later.
            # But primarily, we want to increase score.
            # So, try 1-for-2 swaps directly.
            
            # 3. Try 1-for-2 swaps
            # Pick an edge e_remove to remove.
            # Check if we can add 2 edges from the set of edges that were blocked by e_remove.
            # An edge e is blocked if it shares a face with 3 other present edges.
            # When we remove e_remove, the faces incident to e_remove have their count reduced by 1.
            # This might unblock edges that were in those faces.
            
            # Strategy:
            # For each present edge e_remove:
            #   Temporarily remove it.
            #   Find all absent edges that are now addable.
            #   If we can add >= 2, do it.
            #   If we can add 1, do it only if it leads to a better state? (Score -1+1=0, neutral).
            #   Usually we want score increase. So look for >= 2.
            
            # To speed up, we only consider e_remove that are "critical" or random.
            # Let's pick random present edges.
            
            present_indices = np.where(present)[0]
            if len(present_indices) == 0:
                break
                
            # Pick a random edge to remove
            e_rem_idx = rng.choice(present_indices)
            
            # Save state
            old_present = present[e_rem_idx]
            remove_edge(present, face_counts, e_rem_idx)
            
            # Find addable edges
            addable = []
            # We only need to check edges that share a face with e_rem_idx, 
            # because other edges' addability didn't change (their face counts didn't change).
            # Wait, removing e_rem_idx decreases counts in its faces.
            # An edge e is addable if all its faces have count < 3.
            # Before removal, e was NOT addable, so it had at least one face with count 3.
            # After removal, that face count drops to 2.
            # So e might become addable.
            # Which edges to check? Those that share a face with e_rem_idx.
            
            candidates = set()
            for f in edge_faces[e_rem_idx]:
                for e_idx in face_edges[f]:
                    if e_idx != e_rem_idx and not present[e_idx]:
                        candidates.add(e_idx)
                        
            # Check candidates
            valid_adds = []
            for e_idx in candidates:
                if is_valid_addition(present, face_counts, e_idx):
                    valid_adds.append(e_idx)
            
            # Sort valid_adds? We want to maximize future potential. 
            # Greedy: just take the first 2? Or the ones that block the least?
            # For now, take random 2 or all if possible.
            if len(valid_adds) >= 2:
                # Add 2 edges
                # To maximize score, we add 2.
                # Which 2? Maybe the ones that are "least blocking"?
                # Let's just take the first 2 found.
                add_edge(present, face_counts, valid_adds[0])
                add_edge(present, face_counts, valid_adds[1])
                current_score += 1 # -1 + 2 = +1
                improved = True
                no_improve_count = 0
            else:
                # Restore state
                add_edge(present, face_counts, e_rem_idx)
                no_improve_count += 1
                if no_improve_count > 10000:
                    break
                    
            if not improved and no_improve_count > 2000:
                # Try a 1-for-1 swap to shake up?
                # Or just break and perturb.
                pass
                
        return present, face_counts, current_score

    best_score = 0
    best_present = None
    best_face_counts = None
    
    # ILS Loop
    rng = random.Random(42)
    
    # Initial Greedy
    present, face_counts = greedy_init(rng)
    score = np.sum(present)
    if score > best_score:
        best_score = score
        best_present = present.copy()
        best_face_counts = face_counts.copy()
        
    # Local Search
    present, face_counts, score = local_search(present, face_counts, rng, deadline - 10)
    if score > best_score:
        best_score = score
        best_present = present.copy()
        best_face_counts = face_counts.copy()
        
    # Iterate
    while time.time() < deadline - 5:
        # Perturb
        current_present = best_present.copy()
        current_face_counts = best_face_counts.copy()
        current_score = best_score
        
        # Remove k random edges
        k = rng.randint(5, 15)
        present_indices = np.where(current_present)[0]
        if len(present_indices) > k:
            remove_indices = rng.sample(list(present_indices), k)
            for idx in remove_indices:
                remove_edge(current_present, current_face_counts, idx)
            current_score -= k
            
        # Local Search
        current_present, current_face_counts, current_score = local_search(
            current_present, current_face_counts, rng, min(deadline - 2, time.time() + 5)
        )
        
        if current_score > best_score:
            best_score = current_score
            best_present = current_present.copy()
            best_face_counts = current_face_counts.copy()
            
    # Construct output
    out = []
    for idx, is_present in enumerate(best_present):
        if is_present:
            u, v = edges[idx]
            out.append([u, v])
            
    return out
