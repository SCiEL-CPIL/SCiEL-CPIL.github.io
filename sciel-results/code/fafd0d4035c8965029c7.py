import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    d = 7
    n = 1 << d
    num_edges = d * (1 << (d - 1))  # 448
    
    # Precompute edge indices
    edge_to_idx = {}
    idx_to_edge = []
    idx = 0
    for u in range(n):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edge_to_idx[(u, v)] = idx
                idx_to_edge.append((u, v))
                idx += 1
                
    # Precompute faces
    # A face is a set of 4 edges.
    # We need to map edge -> list of faces it belongs to.
    # And face -> list of 4 edges.
    
    seen_faces = set()
    face_edges = [] # List of lists of 4 edge indices
    
    for i in range(d):
        for j in range(i+1, d):
            for u in range(n):
                v1 = u
                v2 = u ^ (1 << i)
                v3 = u ^ (1 << j)
                v4 = u ^ (1 << i) ^ (1 << j)
                
                e1 = (min(v1, v2), max(v1, v2))
                e2 = (min(v1, v3), max(v1, v3))
                e3 = (min(v2, v4), max(v2, v4))
                e4 = (min(v3, v4), max(v3, v4))
                
                idxs = [edge_to_idx[e1], edge_to_idx[e2], edge_to_idx[e3], edge_to_idx[e4]]
                key = frozenset(idxs)
                
                if key not in seen_faces:
                    seen_faces.add(key)
                    face_edges.append(idxs)
                    
    # Map edge to faces
    edge_faces = [[] for _ in range(num_edges)]
    for f_idx, face in enumerate(face_edges):
        for e in face:
            edge_faces[e].append(f_idx)
            
    num_faces = len(face_edges)
    
    # State representation for penalty-based search
    # state: np.array of 0/1 for edges
    # face_counts: np.array of length num_faces, count of edges in each face
    
    best_valid_count = 0
    best_valid_state = None
    
    rng = np.random.default_rng(42)
    
    # Multi-start penalty-based SA
    num_runs = 10
    time_per_run = (deadline - start_time) / num_runs
    
    for run_idx in range(num_runs):
        run_deadline = start_time + (run_idx + 1) * time_per_run
        
        # Initialize with random valid set? Or just random?
        # Let's start with a random set of ~300 edges, then repair?
        # Or start with empty and add?
        # Better: Start with a greedy valid set, then allow invalid moves in SA.
        
        # Greedy Init
        state = np.zeros(num_edges, dtype=np.uint8)
        order = rng.permutation(num_edges)
        current_count = 0
        # We don't need face_counts for greedy init if we just check validity directly, 
        # but for SA we need face_counts.
        # Let's build face_counts as we go? Or just init face_counts to 0 and update.
        
        face_counts = np.zeros(num_faces, dtype=np.int32)
        
        # Greedy fill
        for e in order:
            # Check if adding e creates a C4
            # A C4 is created if any face containing e has 3 other edges already present.
            valid = True
            for f_idx in edge_faces[e]:
                if face_counts[f_idx] == 3:
                    valid = False
                    break
            if valid:
                state[e] = 1
                current_count += 1
                for f_idx in edge_faces[e]:
                    face_counts[f_idx] += 1
        
        # Now we have a valid state.
        # If it's better than best, update.
        if current_count > best_valid_count:
            best_valid_count = current_count
            best_valid_state = state.copy()
            
        # SA with Penalty
        # Objective: current_count - lambda * (num_full_faces)
        # We want to maximize this.
        # Delta for adding e:
        #   +1 to count.
        #   + (number of faces with count 3) to full_faces.
        # Delta for removing e:
        #   -1 to count.
        #   - (number of faces with count 4) to full_faces.
        
        T = 10.0
        cooling = 0.9999
        lambda_penalty = 1.5 # Penalty weight
        
        current_full_faces = 0 # Number of faces with count 4
        
        # Calculate initial full faces (should be 0 if greedy was valid)
        # But if we start from a non-greedy state, we need to count.
        # Here we started greedy, so current_full_faces = 0.
        
        while time.time() < run_deadline:
            e = rng.integers(0, num_edges)
            
            if state[e] == 1:
                # Remove e
                # Delta count: -1
                # Delta full_faces: - (number of faces with count 4 containing e)
                new_full_faces = 0
                for f_idx in edge_faces[e]:
                    if face_counts[f_idx] == 4:
                        new_full_faces += 1
                        face_counts[f_idx] -= 1
                    else:
                        face_counts[f_idx] -= 1
                        # If it was 3, it becomes 2, no change in full faces.
                        # If it was 2, it becomes 1.
                
                # Wait, logic update:
                # We need to update face_counts first to know the new state?
                # No, we need to know the change in full_faces.
                # If face_counts[f] was 4, it becomes 3. Full faces decreases by 1.
                # If face_counts[f] was 3, it becomes 2. Full faces unchanged.
                
                # Let's re-evaluate the loop to be precise.
                # Reset face_counts changes for this step to calculate delta correctly.
                # Actually, just apply the change and track delta.
                
                # Undo the face_counts changes to calculate delta properly? 
                # No, just calculate delta before applying.
                
                # Let's restart the move logic for clarity.
                
                # REVERT the face_counts changes I just did mentally.
                # I will implement the move carefully.
                
                # Step 1: Calculate Delta
                if state[e] == 1:
                    # Remove
                    delta_count = -1
                    delta_full = 0
                    for f_idx in edge_faces[e]:
                        if face_counts[f_idx] == 4:
                            delta_full -= 1
                    delta_obj = delta_count - lambda_penalty * delta_full
                    # delta_full is negative, so -lambda * delta_full is positive.
                    # Removing an edge from a full face reduces penalty. Good.
                else:
                    # Add
                    delta_count = 1
                    delta_full = 0
                    for f_idx in edge_faces[e]:
                        if face_counts[f_idx] == 3:
                            delta_full += 1
                    delta_obj = delta_count - lambda_penalty * delta_full
                    # Adding an edge to a face with 3 creates a full face. Penalty increases.
                
                # Acceptance
                if delta_obj > 0:
                    accept = True
                else:
                    # exp(delta_obj / T)
                    # delta_obj can be negative.
                    if rng.random() < np.exp(delta_obj / T):
                        accept = True
                    else:
                        accept = False
                
                if accept:
                    # Apply move
                    if state[e] == 1:
                        state[e] = 0
                        current_count -= 1
                        for f_idx in edge_faces[e]:
                            face_counts[f_idx] -= 1
                            if face_counts[f_idx] == 3: # Was 4
                                current_full_faces -= 1
                    else:
                        state[e] = 1
                        current_count += 1
                        for f_idx in edge_faces[e]:
                            face_counts[f_idx] += 1
                            if face_counts[f_idx] == 4: # Was 3
                                current_full_faces += 1
                    
                    # Check if valid (current_full_faces == 0)
                    if current_full_faces == 0:
                        if current_count > best_valid_count:
                            best_valid_count = current_count
                            best_valid_state = state.copy()
                
                T *= cooling
                if T < 0.1:
                    T = 0.1
    
    if best_valid_state is None:
        return []
        
    result = []
    for i in range(num_edges):
        if best_valid_state[i] == 1:
            result.append(list(idx_to_edge[i]))
            
    return result
