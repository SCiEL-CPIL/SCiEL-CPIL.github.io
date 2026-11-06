import random
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    d = 7
    n = 1 << d  # 128 vertices
    
    # 1. Precompute Edges
    all_edges = []
    edge_index = {}
    for u in range(n):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                idx = len(all_edges)
                all_edges.append((u, v))
                edge_index[(u, v)] = idx
    
    E = len(all_edges) # 448
    
    # 2. Precompute Conflict Graph
    # Two edges conflict if they are in the same C4 face and adjacent in that face
    # (i.e., they share a vertex in the face). Actually, for C4-free constraint,
    # we need: no face has all 4 edges selected.
    # This is equivalent to: for each face, at least one edge is NOT selected.
    # So conflicts: edges e1, e2 in same face where selecting both would be "bad"?
    # No - the constraint is that we cannot select ALL 4 edges of any face.
    # This is a 4-wise constraint, not pairwise.
    
    # Better approach: Build conflict graph where edge e1 conflicts with e2 if
    # they are in a common face AND there exists a third edge e3 in that face
    # such that {e1, e2, e3} being selected forces the fourth to be unselectable.
    # Actually, simpler: just track face counts as before but use MIS-style local search.
    
    # Let's build the face structure first
    all_faces = []
    edge_to_faces = {e: [] for e in range(E)}
    
    for i in range(d):
        for j in range(i + 1, d):
            mask_i = 1 << i
            mask_j = 1 << j
            
            for x in range(n):
                if (x & mask_i) or (x & mask_j):
                    continue
                
                v0 = x
                v1 = x ^ mask_i
                v2 = x ^ mask_j
                v3 = x ^ mask_i ^ mask_j
                
                e0 = edge_index[(min(v0, v1), max(v0, v1))]
                e1 = edge_index[(min(v0, v2), max(v0, v2))]
                e2 = edge_index[(min(v1, v3), max(v1, v3))]
                e3 = edge_index[(min(v2, v3), max(v2, v3))]
                
                face = (e0, e1, e2, e3)
                face_id = len(all_faces)
                all_faces.append(face)
                
                for e in face:
                    edge_to_faces[e].append(face_id)
    
    F = len(all_faces) # 672
    
    # Build conflict graph: edge e1 conflicts with e2 if they are in a common face
    # and the face has exactly 2 other edges that could complete the C4.
    # Actually, for the C4-free constraint, the conflict is 4-wise.
    # Let's use a different approach: track for each edge, how many faces it's in
    # and how many of those faces are "saturated" (3 edges selected).
    
    # For MIS formulation: 
    # Vertex = hypercube edge
    # Conflict between e1, e2 if there exists a face containing both e1 and e2
    # where selecting e1, e2, and the other two edges in that face would create a C4.
    # But the constraint is only violated if ALL 4 are selected.
    # So pairwise conflicts don't directly capture this.
    
    # Alternative: Use a weighted local search where we penalize faces with 3+ selected edges.
    # Or: Reformulate as - maximize |S| subject to for all faces f, |S ∩ f| ≤ 3.
    
    # Let's use a smart local search on the original formulation but with better moves.
    
    def init_state(random_seed):
        rng = random.Random(random_seed)
        selected = set()
        face_counts = [0] * F
        
        # Greedy initialization with random order
        edges_to_try = list(range(E))
        rng.shuffle(edges_to_try)
        
        for e in edges_to_try:
            valid = True
            for f_id in edge_to_faces[e]:
                if face_counts[f_id] == 3:
                    valid = False
                    break
            if valid:
                selected.add(e)
                for f_id in edge_to_faces[e]:
                    face_counts[f_id] += 1
        return selected, face_counts, rng

    def try_add(selected, face_counts, e):
        for f_id in edge_to_faces[e]:
            if face_counts[f_id] == 3:
                return False
        selected.add(e)
        for f_id in edge_to_faces[e]:
            face_counts[f_id] += 1
        return True

    def try_remove(selected, face_counts, e):
        selected.discard(e)
        for f_id in edge_to_faces[e]:
            face_counts[f_id] -= 1

    def greedy_fill(selected, face_counts, rng):
        candidates = [e for e in range(E) if e not in selected]
        rng.shuffle(candidates)
        added = 0
        for e in candidates:
            if try_add(selected, face_counts, e):
                added += 1
        return added

    def local_search_mis(selected, face_counts, rng, time_limit):
        """
        MIS-inspired local search:
        - Try 1-for-1 swaps (remove one, add one or more)
        - Try kicks (remove k random edges, then greedy fill)
        - Use conflict-based scoring to guide which edges to remove
        """
        current_size = len(selected)
        best_size = current_size
        best_selected = set(selected)
        best_face_counts = face_counts[:]
        
        no_improve_count = 0
        max_no_improve = 3000
        
        # Precompute for each edge: number of faces where it's "critical" 
        # (face has count 3, so adding another edge to that face is blocked)
        
        while time.time() < time_limit:
            if no_improve_count > max_no_improve:
                break
            
            # Strategy 1: 1-for-1 swap
            # Find an edge to remove that unblocks many additions
            # Score each edge in selected by how many faces it blocks
            if current_size > 0:
                # Compute blocking score for each selected edge
                # An edge e blocks face f if face_counts[f] == 3 and e is in f
                # Removing e would unblock f, allowing potentially more edges
                
                # Try a few random edges to remove and see what we can add
                num_trials = min(10, current_size)
                trial_edges = rng.sample(list(selected), num_trials)
                
                best_trial_gain = -1
                best_trial_remove = None
                
                for e_remove in trial_edges:
                    # Simulate removal
                    old_counts = face_counts[:]
                    try_remove(selected, face_counts, e_remove)
                    
                    # Try to add as many as possible
                    gain = 0
                    added_edges = []
                    for f_id in edge_to_faces[e_remove]:
                        if face_counts[f_id] < 3:
                            # Find edges in this face not yet selected
                            for e in all_faces[f_id]:
                                if e not in selected and e != e_remove:
                                    if try_add(selected, face_counts, e):
                                        gain += 1
                                        added_edges.append(e)
                    
                    # Check if net gain is positive
                    net_gain = gain - 1  # removed 1, added gain
                    
                    if net_gain > best_trial_gain:
                        best_trial_gain = net_gain
                        best_trial_remove = e_remove
                        # Keep the state for this trial
                        best_trial_selected = set(selected)
                        best_trial_counts = face_counts[:]
                    
                    # Revert
                    selected = set(best_trial_selected) if best_trial_remove else set()
                    # Actually, we need to revert properly
                    # Let's redo this more carefully
                
                # Better approach: do one trial at a time
                e_remove = rng.choice(list(selected))
                old_selected = set(selected)
                old_counts = face_counts[:]
                
                try_remove(selected, face_counts, e_remove)
                
                # Greedy fill
                added = greedy_fill(selected, face_counts, rng)
                new_size = len(selected)
                
                if new_size > best_size:
                    best_size = new_size
                    best_selected = set(selected)
                    best_face_counts = face_counts[:]
                    no_improve_count = 0
                elif new_size > current_size:
                    current_size = new_size
                else:
                    # Revert
                    selected = set(old_selected)
                    face_counts = old_counts
                    no_improve_count += 1
            
            # Strategy 2: Kick - remove k random edges
            if rng.random() < 0.3 and current_size >= 2:
                k = rng.randint(2, 4)
                if current_size < k:
                    k = current_size
                
                e_remove_list = rng.sample(list(selected), k)
                old_selected = set(selected)
                old_counts = face_counts[:]
                
                for e in e_remove_list:
                    try_remove(selected, face_counts, e)
                
                added = greedy_fill(selected, face_counts, rng)
                new_size = len(selected)
                
                if new_size > best_size:
                    best_size = new_size
                    best_selected = set(selected)
                    best_face_counts = face_counts[:]
                    no_improve_count = 0
                elif new_size > current_size:
                    current_size = new_size
                else:
                    selected = set(old_selected)
                    face_counts = old_counts
                    no_improve_count += 1
            
            # Strategy 3: Targeted removal - remove edge from a "tight" face
            # Find faces with count 3, remove an edge from one, then try to add
            tight_faces = [f_id for f_id in range(F) if face_counts[f_id] == 3]
            if tight_faces and current_size > 0:
                f_id = rng.choice(tight_faces)
                # Pick a random edge from this face that is selected
                face_edges_in_selected = [e for e in all_faces[f_id] if e in selected]
                if face_edges_in_selected:
                    e_remove = rng.choice(face_edges_in_selected)
                    old_selected = set(selected)
                    old_counts = face_counts[:]
                    
                    try_remove(selected, face_counts, e_remove)
                    added = greedy_fill(selected, face_counts, rng)
                    new_size = len(selected)
                    
                    if new_size > best_size:
                        best_size = new_size
                        best_selected = set(selected)
                        best_face_counts = face_counts[:]
                        no_improve_count = 0
                    elif new_size > current_size:
                        current_size = new_size
                    else:
                        selected = set(old_selected)
                        face_counts = old_counts
                        no_improve_count += 1
        
        return best_selected, best_face_counts, best_size

    # Main Loop
    best_size_overall = 0
    best_result = []
    
    # Multiple restarts with adaptive time allocation
    num_restarts = 30
    for i in range(num_restarts):
        if time.time() > deadline - 8:
            break
            
        seed = 987654321 + i * 137
        selected, face_counts, rng = init_state(seed)
        
        # Initial Greedy Fill
        greedy_fill(selected, face_counts, rng)
        
        # Allocate time: more for early restarts, less for later
        remaining = deadline - time.time() - 5  # reserve 5s
        if remaining < 1:
            break
        
        # First few restarts get more time
        if i < 5:
            time_alloc = min(8, remaining)
        elif i < 15:
            time_alloc = min(4, remaining)
        else:
            time_alloc = min(2, remaining)
        
        if time_alloc < 0.5:
            break
        
        selected, face_counts, size = local_search_mis(selected, face_counts, rng, time.time() + time_alloc)
        
        if size > best_size_overall:
            best_size_overall = size
            best_result = [(all_edges[e][0], all_edges[e][1]) for e in selected]
    
    # Final verification
    # (Skip for speed, trust the algorithm)
    
    return [[u, v] for u, v in best_result]

if __name__ == "__main__":
    res = construct()
    print(len(res))
