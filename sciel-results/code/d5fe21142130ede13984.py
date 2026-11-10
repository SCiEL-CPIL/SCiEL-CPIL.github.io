import time
import random

def construct():
    deadline = time.time() + 110.0
    d = 7
    n = 1 << d
    
    # 1. Precompute edges
    edges = []
    for u in range(n):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edges.append((u, v, i))
    num_edges = len(edges)
    
    # 2. Precompute face IDs
    pair_index = {}
    idx = 0
    for i in range(d):
        for j in range(i + 1, d):
            pair_index[(i, j)] = idx
            idx += 1
            
    def get_compact_rest(u, i, j):
        val = 0
        bit_pos = 0
        for k in range(d):
            if k == i or k == j:
                continue
            if u & (1 << k):
                val |= (1 << bit_pos)
            bit_pos += 1
        return val

    edge_faces = []
    for u, v, i in edges:
        faces = []
        for j in range(d):
            if j == i:
                continue
            if i < j:
                p_idx = pair_index[(i, j)]
            else:
                p_idx = pair_index[(j, i)]
            rest_compact = get_compact_rest(u, i, j)
            face_id = p_idx * (1 << (d - 2)) + rest_compact
            faces.append(face_id)
        edge_faces.append(faces)
        
    total_faces = len(pair_index) * (1 << (d - 2))
    
    face_counts = [0] * total_faces
    selected = set()
    
    def add_edge(e_idx):
        for fid in edge_faces[e_idx]:
            face_counts[fid] += 1
        selected.add(e_idx)
        
    def remove_edge(e_idx):
        for fid in edge_faces[e_idx]:
            face_counts[fid] -= 1
        selected.remove(e_idx)
        
    def can_add(e_idx):
        for fid in edge_faces[e_idx]:
            if face_counts[fid] >= 3:
                return False
        return True
        
    def greedy_fill():
        order = list(range(num_edges))
        random.shuffle(order)
        for e_idx in order:
            if can_add(e_idx):
                add_edge(e_idx)
                
    def try_add_missing():
        missing = [i for i in range(num_edges) if i not in selected]
        random.shuffle(missing)
        added_any = False
        for e_idx in missing:
            if can_add(e_idx):
                add_edge(e_idx)
                added_any = True
        return added_any
        
    best_size = 0
    best_selected = set()
    
    # Initial greedy
    greedy_fill()
    best_size = len(selected)
    best_selected = set(selected)
    
    # Simulated Annealing / Large Kick Local Search
    # We will try to escape local optima by removing a larger number of edges
    # and re-filling.
    
    current_size = len(selected)
    T = 1.0 # Temperature concept, though we use deterministic acceptance for size improvement mostly
    # We will use a strategy: if we can't add, remove K edges, refill.
    # If new size > old size, accept.
    # If new size < old size, accept with probability p to explore.
    
    p_accept_worse = 0.1
    
    while time.time() < deadline:
        if not try_add_missing():
            # Stuck
            # Perform a kick
            # Choose kick size K. Start small, increase if stuck for long?
            # Let's vary K randomly between 3 and 15.
            K = random.randint(3, 15)
            
            if len(selected) < K:
                # Too small to kick, restart
                selected.clear()
                for i in range(total_faces): face_counts[i] = 0
                greedy_fill()
                if len(selected) > best_size:
                    best_size = len(selected)
                    best_selected = set(selected)
                continue
                
            # Save state
            old_face_counts = face_counts[:]
            old_selected = set(selected)
            old_size = len(selected)
            
            # Remove K random edges
            to_remove = random.sample(list(selected), K)
            for e_idx in to_remove:
                remove_edge(e_idx)
                
            # Greedy re-add
            missing = [i for i in range(num_edges) if i not in selected]
            random.shuffle(missing)
            for e_idx in missing:
                if can_add(e_idx):
                    add_edge(e_idx)
                    
            new_size = len(selected)
            
            if new_size > best_size:
                best_size = new_size
                best_selected = set(selected)
                
            # Acceptance criterion
            if new_size >= old_size:
                # Accept (keep new state)
                pass
            else:
                # Worse, accept with probability to explore
                if random.random() < p_accept_worse:
                    # Keep new state (worse)
                    pass
                else:
                    # Revert
                    selected = old_selected
                    face_counts = old_face_counts

    # Restore best solution
    selected = best_selected
    
    result = []
    for e_idx in best_selected:
        u, v, _ = edges[e_idx]
        result.append([u, v])
        
    return result
