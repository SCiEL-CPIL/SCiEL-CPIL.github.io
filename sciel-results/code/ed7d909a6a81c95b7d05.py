import time
import random
import numpy as np

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
    
    # Use numpy arrays for faster face count updates
    face_counts = np.zeros(total_faces, dtype=np.int8)
    selected = np.zeros(num_edges, dtype=bool)
    
    def add_edge(e_idx):
        for fid in edge_faces[e_idx]:
            face_counts[fid] += 1
        selected[e_idx] = True
        
    def remove_edge(e_idx):
        for fid in edge_faces[e_idx]:
            face_counts[fid] -= 1
        selected[e_idx] = False
        
    def can_add(e_idx):
        for fid in edge_faces[e_idx]:
            if face_counts[fid] >= 3:
                return False
        return True

    def get_face_pressure(e_idx):
        """Return the minimum slack (3 - count) among faces of this edge."""
        min_slack = 3
        for fid in edge_faces[e_idx]:
            slack = 3 - face_counts[fid]
            if slack < min_slack:
                min_slack = slack
        return min_slack

    def greedy_fill():
        """Add all addable edges, prioritizing those with most slack."""
        added_any = False
        # Build list of addable edges with their pressure
        addable = []
        for e_idx in range(num_edges):
            if not selected[e_idx] and can_add(e_idx):
                addable.append((get_face_pressure(e_idx), e_idx))
        # Sort by pressure (higher pressure = more room)
        addable.sort(key=lambda x: -x[0])
        for _, e_idx in addable:
            if not selected[e_idx] and can_add(e_idx):
                add_edge(e_idx)
                added_any = True
        return added_any

    def algebraic_init():
        """
        Structured initialization using parity-based rules.
        """
        selected[:] = False
        face_counts[:] = 0
        
        # For each dimension, use a different linear form
        for e_idx in range(num_edges):
            u, v, i = edges[e_idx]
            if can_add(e_idx):
                u_mask = u & ~(1 << i)
                
                # Multiple parity checks
                parity1 = bin(u_mask).count('1') % 2
                even_bits = sum((u_mask >> k) & 1 for k in range(d) if k % 2 == 0 and k != i)
                parity2 = even_bits % 2
                odd_bits = sum((u_mask >> k) & 1 for k in range(d) if k % 2 == 1 and k != i)
                parity3 = odd_bits % 2
                
                even_count = (1 if parity1 == 0 else 0) + (1 if parity2 == 0 else 0) + (1 if parity3 == 0 else 0)
                if even_count >= 2:
                    add_edge(e_idx)
        
        # Second pass with different criterion
        for e_idx in range(num_edges):
            if not selected[e_idx] and can_add(e_idx):
                u, v, i = edges[e_idx]
                u_mask = u & ~(1 << i)
                # Include if the bit at position (i+1)%d is 0
                if (u_mask >> ((i + 1) % d)) & 1 == 0:
                    add_edge(e_idx)
    
    def two_flip_move():
        """
        Systematic 2-flip: remove 2 selected edges, try to add 2 different missing edges.
        Returns True if the move was accepted (net gain >= 0).
        """
        sel_indices = np.where(selected)[0]
        if len(sel_indices) < 2:
            return False
            
        # Pick 2 random edges to remove
        e1, e2 = random.sample(list(sel_indices), 2)
        
        # Save state
        old_face_counts = face_counts.copy()
        old_selected = selected.copy()
        
        # Remove both
        remove_edge(e1)
        remove_edge(e2)
        
        # Find missing edges that can now be added
        missing = np.where(~selected)[0]
        random.shuffle(missing)
        
        added_count = 0
        for e_idx in missing:
            if added_count >= 2:
                break
            if can_add(e_idx):
                add_edge(e_idx)
                added_count += 1
        
        if added_count >= 2:
            return True
        else:
            # Revert
            selected[:] = old_selected
            face_counts[:] = old_face_counts
            return False

    def multi_flip(k):
        """
        Remove k random edges and greedily re-add.
        """
        sel_indices = np.where(selected)[0]
        if len(sel_indices) < k:
            return False
            
        old_face_counts = face_counts.copy()
        old_selected = selected.copy()
        old_size = len(sel_indices)
        
        to_remove = random.sample(list(sel_indices), k)
        for e_idx in to_remove:
            remove_edge(e_idx)
            
        # Greedy re-add
        missing = np.where(~selected)[0]
        random.shuffle(missing)
        for e_idx in missing:
            if can_add(e_idx):
                add_edge(e_idx)
                
        new_size = int(np.sum(selected))
        
        if new_size >= old_size:
            return True
        else:
            # Revert with some probability to allow exploration
            if random.random() > 0.1:
                selected[:] = old_selected
                face_counts[:] = old_face_counts
            return False

    def targeted_two_flip():
        """
        Targeted 2-flip: identify edges whose removal would free up the most faces,
        then try to add edges that benefit from those freed faces.
        """
        sel_indices = np.where(selected)[0]
        if len(sel_indices) < 2:
            return False
            
        # Score each selected edge by how many faces it blocks
        edge_scores = []
        for e_idx in sel_indices:
            # Count how many faces this edge contributes to that are at count 3
            blocking = 0
            for fid in edge_faces[e_idx]:
                if face_counts[fid] >= 3:
                    blocking += 1
            edge_scores.append((blocking, e_idx))
        
        # Sort by blocking (most blocking first)
        edge_scores.sort(key=lambda x: -x[0])
        
        # Try removing the top 2 most-blocking edges
        if len(edge_scores) < 2:
            return False
        
        e1 = edge_scores[0][1]
        e2 = edge_scores[1][1]
        
        old_face_counts = face_counts.copy()
        old_selected = selected.copy()
        
        remove_edge(e1)
        remove_edge(e2)
        
        # Try to add missing edges
        missing = np.where(~selected)[0]
        random.shuffle(missing)
        
        added_count = 0
        for e_idx in missing:
            if added_count >= 2:
                break
            if can_add(e_idx):
                add_edge(e_idx)
                added_count += 1
        
        if added_count >= 2:
            return True
        else:
            selected[:] = old_selected
            face_counts[:] = old_face_counts
            return False

    best_size = 0
    best_selected = None
    
    # Start with algebraic construction
    algebraic_init()
    best_size = int(np.sum(selected))
    best_selected = selected.copy()
    
    # Try to improve with greedy fill
    greedy_fill()
    if int(np.sum(selected)) > best_size:
        best_size = int(np.sum(selected))
        best_selected = selected.copy()
    
    # Local search with adaptive strategies
    stuck_count = 0
    iter_count = 0
    
    while time.time() < deadline:
        iter_count += 1
        
        if not greedy_fill():
            stuck_count += 1
        else:
            stuck_count = 0
            if int(np.sum(selected)) > best_size:
                best_size = int(np.sum(selected))
                best_selected = selected.copy()
        
        # Adaptive move selection based on stuckness
        if stuck_count > 0:
            r = random.random()
            
            if r < 0.35:
                # Random 2-flip
                two_flip_move()
            elif r < 0.65:
                # Targeted 2-flip
                targeted_two_flip()
            else:
                # Multi-flip with adaptive size
                kick_size = min(3 + stuck_count // 4, 8)
                multi_flip(kick_size)
            
            if int(np.sum(selected)) > best_size:
                best_size = int(np.sum(selected))
                best_selected = selected.copy()
        
        # Periodic restart to escape deep local optima
        if stuck_count > 40 or (iter_count % 200 == 0 and random.random() < 0.05):
            algebraic_init()
            greedy_fill()
            if int(np.sum(selected)) > best_size:
                best_size = int(np.sum(selected))
                best_selected = selected.copy()
            stuck_count = 0

    # Final polish: try to add any remaining edges
    greedy_fill()
    if int(np.sum(selected)) > best_size:
        best_size = int(np.sum(selected))
        best_selected = selected.copy()

    # Restore best solution
    result = []
    for e_idx in range(num_edges):
        if best_selected[e_idx]:
            u, v, _ = edges[e_idx]
            result.append([u, v])
        
    return result
