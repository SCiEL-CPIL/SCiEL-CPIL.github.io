"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 145s to leave 5s safety margin
    
    d = 7
    n = 2 ** d
    num_edges = d * 2 ** (d - 1)  # 448
    
    # Generate all edges and map them to indices
    all_edges = []
    edge_index = {}
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(all_edges)
                all_edges.append((x, y))
                edge_index[(x, y)] = idx
    
    # For each edge, find which dimension it's along
    edge_dim = []
    for (x, y) in all_edges:
        diff = x ^ y
        edge_dim.append(diff.bit_length() - 1)
    
    # Precompute faces
    # A face is (i, j, base) where i < j, base has bits i,j = 0
    # For a given (i, j, base), the 4 vertices are:
    #   base, base^(1<<i), base^(1<<j), base^(1<<(i)|1<<(j))
    # The 4 edges are:
    #   (base, base^(1<<i)) - dim i
    #   (base^(1<<j), base^(1<<(i)|1<<(j))) - dim i
    #   (base, base^(1<<j)) - dim j
    #   (base^(1<<i), base^(1<<(i)|1<<(j))) - dim j
    
    faces = []  # list of (i, j, base)
    face_to_edges = []  # list of 4 edge indices per face
    edge_to_faces = [[] for _ in range(num_edges)]
    
    for i in range(d):
        for j in range(i + 1, d):
            mask_i = 1 << i
            mask_j = 1 << j
            mask_ij = mask_i | mask_j
            # base has bits i and j = 0
            for base in range(n):
                if (base & mask_i) == 0 and (base & mask_j) == 0:
                    v0 = base
                    v1 = base ^ mask_i
                    v2 = base ^ mask_j
                    v3 = base ^ mask_ij
                    
                    e0 = edge_index.get((min(v0, v1), max(v0, v1)))
                    e1 = edge_index.get((min(v1, v3), max(v1, v3)))
                    e2 = edge_index.get((min(v0, v2), max(v0, v2)))
                    e3 = edge_index.get((min(v2, v3), max(v2, v3)))
                    
                    if e0 is not None and e1 is not None and e2 is not None and e3 is not None:
                        face_idx = len(faces)
                        faces.append((i, j, base))
                        face_edges = [e0, e1, e2, e3]
                        face_to_edges.append(face_edges)
                        for e in face_edges:
                            edge_to_faces[e].append(face_idx)
    
    num_faces = len(faces)
    assert num_faces == 672, f"Expected 672 faces, got {num_faces}"
    
    # Convert to numpy arrays for speed
    face_edges_arr = np.array(face_to_edges, dtype=np.int32)  # shape (672, 4)
    
    # Precompute edge_to_faces as numpy arrays for vectorized validity check
    edge_to_faces_np = []
    for e in range(num_edges):
        edge_to_faces_np.append(np.array(edge_to_faces[e], dtype=np.int32))
    
    def is_valid_add(chosen, face_counts, edge_idx):
        """Check if edge can be added using vectorized max"""
        faces = edge_to_faces_np[edge_idx]
        if len(faces) == 0:
            return True
        return np.max(face_counts[faces]) < 3
    
    def add_edge(chosen, face_counts, edge_idx):
        chosen[edge_idx] = True
        faces = edge_to_faces_np[edge_idx]
        face_counts[faces] += 1
    
    def remove_edge(chosen, face_counts, edge_idx):
        chosen[edge_idx] = False
        faces = edge_to_faces_np[edge_idx]
        face_counts[faces] -= 1
    
    # Phase 1: Greedy construction with multiple orderings
    chosen = np.zeros(num_edges, dtype=bool)
    face_counts = np.zeros(num_faces, dtype=np.int32)
    
    # Try multiple greedy orderings
    best_count = 0
    best_chosen = None
    best_face_counts = None
    
    random.seed(42)
    
    # Greedy 1: natural order
    for e in range(num_edges):
        if is_valid_add(chosen, face_counts, e):
            add_edge(chosen, face_counts, e)
    
    count = int(chosen.sum())
    if count > best_count:
        best_count = count
        best_chosen = chosen.copy()
        best_face_counts = face_counts.copy()
    
    # Greedy 2: random orders
    for restart in range(80):
        if time.time() > deadline - 60:
            break
        chosen = np.zeros(num_edges, dtype=bool)
        face_counts = np.zeros(num_faces, dtype=np.int32)
        order = list(range(num_edges))
        random.shuffle(order)
        for e in order:
            if is_valid_add(chosen, face_counts, e):
                add_edge(chosen, face_counts, e)
        count = int(chosen.sum())
        if count > best_count:
            best_count = count
            best_chosen = chosen.copy()
            best_face_counts = face_counts.copy()
    
    # Phase 2: Local search - remove one edge, greedily repair
    chosen = best_chosen.copy()
    face_counts = best_face_counts.copy()
    current_count = int(chosen.sum())
    
    def greedy_repair(chosen, face_counts):
        """Try to add as many edges as possible"""
        changed = True
        while changed:
            changed = False
            for e in range(num_edges):
                if not chosen[e] and is_valid_add(chosen, face_counts, e):
                    add_edge(chosen, face_counts, e)
                    changed = True
        return int(chosen.sum())
    
    # Local search iterations
    max_ls_rounds = 500
    for rnd in range(max_ls_rounds):
        if time.time() > deadline - 15:
            break
        
        improved = False
        selected = [e for e in range(num_edges) if chosen[e]]
        random.shuffle(selected)
        
        # Try removing up to 5 random edges
        for e_remove in selected[:5]:
            c2 = chosen.copy()
            fc2 = face_counts.copy()
            remove_edge(c2, fc2, e_remove)
            
            # Greedy repair
            new_count = greedy_repair(c2, fc2)
            
            if new_count > current_count:
                chosen = c2
                face_counts = fc2
                current_count = new_count
                if current_count > best_count:
                    best_count = current_count
                    best_chosen = chosen.copy()
                    best_face_counts = face_counts.copy()
                improved = True
                break
        
        if not improved and rnd > 50:
            # Try a few more rounds before giving up
            if rnd > 200:
                break
    
    # Phase 3: More aggressive local search - remove 2 edges at a time
    chosen = best_chosen.copy()
    face_counts = best_face_counts.copy()
    current_count = int(chosen.sum())
    
    for rnd in range(300):
        if time.time() > deadline - 10:
            break
        
        improved = False
        selected = [e for e in range(num_edges) if chosen[e]]
        random.shuffle(selected)
        
        # Try removing pairs of edges
        for i in range(min(10, len(selected))):
            for j in range(i + 1, min(10, len(selected))):
                e1, e2 = selected[i], selected[j]
                c2 = chosen.copy()
                fc2 = face_counts.copy()
                remove_edge(c2, fc2, e1)
                remove_edge(c2, fc2, e2)
                
                new_count = greedy_repair(c2, fc2)
                
                if new_count > current_count:
                    chosen = c2
                    face_counts = fc2
                    current_count = new_count
                    if current_count > best_count:
                        best_count = current_count
                        best_chosen = chosen.copy()
                        best_face_counts = face_counts.copy()
                    improved = True
                    break
            if improved:
                break
        
        if not improved and rnd > 100:
            break
    
    # Final verification
    def is_valid(chosen_set):
        for face_idx in range(num_faces):
            edges_in_face = face_to_edges[face_idx]
            count_in_face = sum(1 for e in edges_in_face if chosen_set[e])
            if count_in_face == 4:
                return False
        return True
    
    # Use the best solution found
    final_chosen = best_chosen
    final_count = best_count
    
    # Convert to output format
    out = []
    for idx in range(num_edges):
        if final_chosen[idx]:
            out.append(list(all_edges[idx]))
    
    return out
# EVOLVE-BLOCK-END
