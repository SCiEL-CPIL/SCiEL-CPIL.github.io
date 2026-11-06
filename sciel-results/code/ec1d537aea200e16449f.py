import time
import random

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    d = 7
    n_vertices = 1 << d
    n_edges = d * (1 << (d - 1))  # 448
    
    # Edge representation: (u, v) with u < v
    # We map each edge to an index 0..447
    edge_to_idx = {}
    idx_to_edge = []
    idx = 0
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edge_to_idx[(u, v)] = idx
                idx_to_edge.append((u, v))
                idx += 1
    
    # Precompute faces
    # A face is defined by two distinct dimensions i, j and a vertex u that does not have bits i or j set?
    # Actually, a face is a set of 4 vertices: u, u^i, u^j, u^i^j where i,j are the dimensions.
    # To avoid duplicates, we can iterate over all pairs of dimensions (i,j) with i<j,
    # and for each, iterate over all vertices u that are "canonical" for that face.
    # A vertex u is canonical for face (i,j) if u has 0 in bits i and j.
    
    faces = []  # List of lists of edge indices
    edge_to_faces = [[] for _ in range(n_edges)]
    
    for i in range(d):
        for j in range(i + 1, d):
            # Iterate over all u such that u has 0 in bits i and j
            # We can iterate over all u in 0..127, and check if (u & (1<<i)) == 0 and (u & (1<<j)) == 0
            for u in range(n_vertices):
                if (u & (1 << i)) == 0 and (u & (1 << j)) == 0:
                    # The 4 vertices are:
                    v0 = u
                    v1 = u ^ (1 << i)
                    v2 = u ^ (1 << j)
                    v3 = u ^ (1 << i) ^ (1 << j)
                    
                    # The 4 edges are:
                    e0 = (min(v0, v1), max(v0, v1))
                    e1 = (min(v2, v3), max(v2, v3))
                    e2 = (min(v0, v2), max(v0, v2))
                    e3 = (min(v1, v3), max(v1, v3))
                    
                    face_edges = [edge_to_idx[e0], edge_to_idx[e1], edge_to_idx[e2], edge_to_idx[e3]]
                    faces.append(face_edges)
                    
                    for e in face_edges:
                        edge_to_faces[e].append(len(faces) - 1)
    
    n_faces = len(faces)  # 672
    
    def greedy_hit_set(order=None):
        if order is None:
            order = list(range(n_edges))
        
        R = set()
        face_covered = [False] * n_faces
        
        for e in order:
            # Check if e covers any uncovered face
            add = False
            for f in edge_to_faces[e]:
                if not face_covered[f]:
                    add = True
                    break
            if add:
                R.add(e)
                for f in edge_to_faces[e]:
                    face_covered[f] = True
        return R
    
    def local_search(R, face_covered):
        # face_covered is a list of bools, but we need counts for removal
        # Let's recompute counts
        face_count = [0] * n_faces
        for e in R:
            for f in edge_to_faces[e]:
                face_count[f] += 1
        
        improved = True
        while improved:
            improved = False
            # Try to remove each edge
            # To avoid order bias, we can shuffle the list of edges in R
            edges_in_R = list(R)
            random.shuffle(edges_in_R)
            
            for e in edges_in_R:
                # Check if e can be removed
                can_remove = True
                for f in edge_to_faces[e]:
                    if face_count[f] == 1:
                        can_remove = False
                        break
                
                if can_remove:
                    R.remove(e)
                    for f in edge_to_faces[e]:
                        face_count[f] -= 1
                    improved = True
        
        return R, face_count
    
    best_R = None
    best_size = n_edges  # Max possible size of R is n_edges, but we want min
    
    # Multiple restarts
    while time.time() < deadline:
        # Random order for greedy
        order = list(range(n_edges))
        random.shuffle(order)
        
        R = greedy_hit_set(order)
        
        # Compute face_covered for initial R
        face_covered = [False] * n_faces
        for e in R:
            for f in edge_to_faces[e]:
                face_covered[f] = True
        
        R, face_count = local_search(R, face_covered)
        
        if len(R) < best_size:
            best_size = len(R)
            best_R = set(R)
            
            # If we reach the theoretical lower bound or a known good bound, we can stop early
            # Theoretical lower bound: Each edge covers 32 faces. 672 faces / 32 = 21. But overlaps exist.
            # Known best: 448 - 304 = 144 edges to remove.
            if best_size <= 144:
                break
    
    # Convert best_R to edge list
    edges_out = []
    for i, (u, v) in enumerate(idx_to_edge):
        if i not in best_R:
            edges_out.append([u, v])
    
    return edges_out
