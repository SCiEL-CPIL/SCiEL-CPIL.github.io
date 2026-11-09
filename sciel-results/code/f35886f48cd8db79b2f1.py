    # Generate all edges
    edges = []
    edge_id = {}
    for x in range(n_vertices):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                e = (x, y)
                edge_id[e] = len(edges)
                edges.append(e)
    n_edges = len(edges)

    # Generate all unique 2-faces
    faces = []
    face_id = {}
    for v in range(n_vertices):
        for i in range(d):
            for j in range(i + 1, d):
                v_i = v ^ (1 << i)
                v_j = v ^ (1 << j)
                v_ij = v ^ (1 << i) ^ (1 << j)
                e1 = (min(v, v_i), max(v, v_i))
                e2 = (min(v, v_j), max(v, v_j))
                e3 = (min(v_i, v_ij), max(v_i, v_ij))
                e4 = (min(v_j, v_ij), max(v_j, v_ij))
                f_key = tuple(sorted([e1, e2, e3, e4]))
                if f_key not in face_id:
                    face_id[f_key] = len(faces)
                    faces.append(f_key)
    n_faces = len(faces)

    # Map edge index to list of face indices
    edge_to_faces = [[] for _ in range(n_edges)]
    for f_idx, f in enumerate(faces):
        for e in f:
            e_idx = edge_id[e]
            edge_to_faces[e_idx].append(f_idx)
