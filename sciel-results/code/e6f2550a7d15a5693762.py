import random
import time

def construct():
    d = 7
    n = 2 ** d
    total_edges = n * d // 2
    
    # Generate all edges
    all_edges = []
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                all_edges.append((x, y))
                
    # Precompute faces (C4s)
    faces = []
    for i in range(d):
        for j in range(i + 1, d):
            mask_ij = (1 << i) | (1 << j)
            for k in range(n):
                if (k & mask_ij) == 0:
                    v0 = k
                    v1 = k ^ (1 << i)
                    v2 = k ^ (1 << j)
                    v3 = k ^ (1 << i) ^ (1 << j)
                    
                    e1 = (min(v0, v1), max(v0, v1))
                    e2 = (min(v0, v2), max(v0, v2))
                    e3 = (min(v1, v3), max(v1, v3))
                    e4 = (min(v2, v3), max(v2, v3))
                    
                    faces.append([e1, e2, e3, e4])
                    
    edge_to_faces = {e: [] for e in all_edges}
    for idx, face in enumerate(faces):
        for e in face:
            edge_to_faces[e].append(idx)

    def is_valid_addition(e, current_set):
        for f_idx in edge_to_faces[e]:
            face = faces[f_idx]
            if all(x in current_set for x in face if x != e):
                return False
        return True

    def repair(current_set):
        """Remove edges until no C4s remain. Returns the set of violated faces initially."""
        violated_faces = set()
        for idx, face in enumerate(faces):
            if all(e in current_set for e in face):
                violated_faces.add(idx)
                
        while violated_faces:
            vf_idx = random.choice(list(violated_faces))
            face = faces[vf_idx]
            e_to_remove = random.choice(face)
            current_set.remove(e_to_remove)
            
            for f_idx in edge_to_faces[e_to_remove]:
                if f_idx in violated_faces:
                    if not all(e in current_set for e in faces[f_idx]):
                        violated_faces.remove(f_idx)
                        
    start_time = time.time()
    max_time = 115.0
    
    best_size = 0
    best_set = set()
    
    # Initialize with all edges
    current_set = set(all_edges)
    
    while time.time() - start_time < max_time:
        # 1. Repair: Make it valid
        repair(current_set)
        
        # 2. Improve: Greedy Add
        missing_edges = [e for e in all_edges if e not in current_set]
        random.shuffle(missing_edges)
        for e in missing_edges:
            if time.time() - start_time > max_time:
                break
            if is_valid_addition(e, current_set):
                current_set.add(e)
                
        # 3. Update Best
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = set(current_set)
            
        # 4. Kick: Perturb
        if len(current_set) > 20:
            kick_size = random.randint(5, 15)
            edges_to_kick = random.sample(list(current_set), kick_size)
            for e in edges_to_kick:
                current_set.remove(e)
            # The set is now likely invalid or suboptimal, repair will fix it in next loop

    return [list(e) for e in best_set]
