import random
import time
import numpy as np

def construct():
    d = 7
    n_vertices = 1 << d  # 128
    n_edges = n_vertices * d  # 896 directed, 448 undirected
    n_faces = n_vertices * (d * (d - 1) // 2)  # 128 * 21 = 2688
    
    # Time management
    start_time = time.time()
    time_limit = 110.0  # Leave 10s buffer for 120s limit
    
    # Precompute edges (undirected, sorted)
    edges = []
    edge_index = {}
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                idx = len(edges)
                edges.append((u, v))
                edge_index[(u, v)] = idx
    
    E = len(edges)  # 448
    
    # Precompute for each edge, the faces it belongs to and the opposite edges
    # A face is defined by (vertex, dim1, dim2) where dim1 < dim2
    # For an edge (u, v) differing in bit i, it's in faces where the other dimension j != i
    # Face (u, j, i) if j < i, or (u, i, j) if i < j
    # The opposite edges in the face are determined
    
    # Let's precompute: for each edge, list of (face_id, other_two_edges)
    # face_id = u * 21 + j*6 + k (for j<k, dimensions)
    # Actually let's use: face identified by (base_vertex, dim_a, dim_b) where dim_a < dim_b
    # base_vertex is the vertex with both bits 0 in dims dim_a and dim_b
    
    # For each face, store its 4 edges
    face_edges = {}  # face_id -> list of 4 edge indices
    edge_faces = [[] for _ in range(E)]  # edge_index -> list of (face_id, opposite_edge_idx_in_face)
    
    # Build face structure
    for u in range(n_vertices):
        for a in range(d):
            for b in range(a + 1, d):
                # Face vertices: u, u^(1<<a), u^(1<<b), u^(1<<a)^(1<<b)
                # But we need the "base" vertex where bits a and b are 0
                # Actually, any of the 4 vertices can be the base. Let's use the one
                # where bits a and b are 0.
                base = u & ~((1 << a) | (1 << b))
                # Check if base is actually a vertex in this face
                # The face is uniquely determined by base, a, b
                
                v1 = base | (1 << a)
                v2 = base | (1 << b)
                v3 = base | (1 << a) | (1 << b)
                
                e1 = (min(base, v1), max(base, v1))
                e2 = (min(base, v2), max(base, v2))
                e3 = (min(v1, v3), max(v1, v3))
                e4 = (min(v2, v3), max(v2, v3))
                
                e1_idx = edge_index[e1]
                e2_idx = edge_index[e2]
                e3_idx = edge_index[e3]
                e4_idx = edge_index[e4]
                
                face_id = base * 21 + a * 6 + b  # unique identifier
                
                if face_id not in face_edges:
                    face_edges[face_id] = [e1_idx, e2_idx, e3_idx, e4_idx]
                    # Record for each edge which faces it's in
                    # In this face, opposite pairs: e1<->e3, e2<->e4
                    edge_faces[e1_idx].append((face_id, e3_idx))
                    edge_faces[e2_idx].append((face_id, e4_idx))
                    edge_faces[e3_idx].append((face_id, e1_idx))
                    edge_faces[e4_idx].append((face_id, e2_idx))
    
    # Convert to numpy arrays for speed
    face_ids = list(face_edges.keys())
    n_faces_actual = len(face_ids)
    
    # face_edges_list: face_id -> array of 4 edge indices
    face_edges_arr = {}
    for fid in face_ids:
        face_edges_arr[fid] = np.array(face_edges[fid], dtype=np.int32)
    
    # For each edge, store list of face_ids and corresponding opposite edge indices
    edge_face_list = []
    for e in range(E):
        fl = []
        for fid, opp in edge_faces[e]:
            fl.append((fid, opp))
        edge_face_list.append(fl)
    
    # Simulated Annealing
    rng = random.Random(42)
    
    # Initialize with greedy
    present = np.zeros(E, dtype=bool)
    face_count = np.zeros(n_faces_actual, dtype=np.int32)
    
    # Map face_id to index for faster lookup
    face_id_to_idx = {fid: i for i, fid in enumerate(face_ids)}
    
    def add_edge(e):
        """Add edge e, update face counts. Returns True if no conflict created."""
        if present[e]:
            return True
        # Check if adding e creates any C4
        for fid, opp in edge_face_list[e]:
            if face_count[face_id_to_idx[fid]] >= 3:
                return False
        present[e] = True
        for fid, opp in edge_face_list[e]:
            face_count[face_id_to_idx[fid]] += 1
        return True
    
    def remove_edge(e):
        """Remove edge e, update face counts."""
        if not present[e]:
            return
        present[e] = False
        for fid, opp in edge_face_list[e]:
            face_count[face_id_to_idx[fid]] -= 1
    
    # Greedy initialization
    edge_order = list(range(E))
    rng.shuffle(edge_order)
    for e in edge_order:
        add_edge(e)
    
    best_count = int(np.sum(present))
    best_present = present.copy()
    best_count_so_far = best_count
    
    # SA parameters
    T_start = 10.0
    T_end = 0.1
    cool_rate = 0.9995
    
    T = T_start
    step = 0
    max_steps = 5000000  # Safety cap
    
    while T > T_end and step < max_steps:
        if time.time() - start_time > time_limit:
            break
        
        # Choose a move
        if step % 10000 == 0:
            # Occasionally try to add a missing edge
            missing = [e for e in range(E) if not present[e]]
            if missing:
                e = rng.choice(missing)
                if add_edge(e):
                    current_count = best_count + 1
                    if current_count > best_count_so_far:
                        best_count_so_far = current_count
                        best_present = present.copy()
                    best_count = current_count
                else:
                    # Try swap
                    # Find conflicting faces
                    for fid, opp in edge_face_list[e]:
                        if face_count[face_id_to_idx[fid]] >= 3:
                            # This face has 3 edges already
                            # Find which edge to remove
                            f_edges = face_edges_arr[fid]
                            candidates = [fe for fe in f_edges if present[fe] and fe != e]
                            if candidates:
                                e_out = rng.choice(candidates)
                                remove_edge(e_out)
                                add_edge(e)
                                # Count unchanged
                            break
            step += 1
            continue
        
        # Random move
        e = rng.randint(0, E - 1)
        
        if present[e]:
            # Try to remove
            old_count = int(np.sum(present))
            remove_edge(e)
            new_count = old_count - 1
            delta = new_count - old_count  # -1
            accept_prob = 1.0 if delta >= 0 else np.exp(delta / T)
            if rng.random() < accept_prob:
                best_count = new_count
            else:
                # Revert
                add_edge(e)
        else:
            # Try to add
            if add_edge(e):
                best_count = int(np.sum(present))
                if best_count > best_count_so_far:
                    best_count_so_far = best_count
                    best_present = present.copy()
            else:
                # Try swap: find a conflicting face
                for fid, opp in edge_face_list[e]:
                    if face_count[face_id_to_idx[fid]] >= 3:
                        f_edges = face_edges_arr[fid]
                        candidates = [fe for fe in f_edges if present[fe]]
                        if candidates:
                            e_out = rng.choice(candidates)
                            remove_edge(e_out)
                            add_edge(e)
                        break
        
        T *= cool_rate
        step += 1
    
    # Final polish: try to add any missing edges
    for e in range(E):
        if not present[e]:
            if add_edge(e):
                best_count = int(np.sum(present))
                if best_count > best_count_so_far:
                    best_count_so_far = best_count
                    best_present = present.copy()
    
    # Use best found
    if best_count_so_far > int(np.sum(present)):
        present = best_present
    
    # Build output
    out = []
    for i, (u, v) in enumerate(edges):
        if present[i]:
            out.append([u, v])
    
    return out
