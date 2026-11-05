import random
import time
import sys

def construct():
    """
    Construct a C4-free edge set of Q7 maximizing the number of edges.
    Uses multi-start greedy initialization + targeted local search (1-remove, add-neighbors).
    """
    d = 7
    n = 1 << d  # 128 vertices
    all_edges = []
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                all_edges.append((x, y))
    m = len(all_edges)  # 448
    
    # Build face structure
    edge_index = {}
    for idx, (u, v) in enumerate(all_edges):
        edge_index[(u, v)] = idx
        edge_index[(v, u)] = idx
    
    faces = []  # list of tuples of 4 edge indices
    face_of_edge = [[] for _ in range(m)]  # for each edge, list of face indices
    
    for i in range(d):
        for j in range(i + 1, d):
            bi, bj = 1 << i, 1 << j
            for x in range(n):
                if (x & bi) == 0 and (x & bj) == 0:
                    v1, v2, v3, v4 = x, x | bi, x | bj, x | bi | bj
                    e1 = edge_index[(min(v1, v2), max(v1, v2))]
                    e2 = edge_index[(min(v2, v4), max(v2, v4))]
                    e3 = edge_index[(min(v4, v3), max(v4, v3))]
                    e4 = edge_index[(min(v3, v1), max(v3, v1))]
                    face_idx = len(faces)
                    faces.append((e1, e2, e3, e4))
                    for e in (e1, e2, e3, e4):
                        face_of_edge[e].append(face_idx)
    
    # Precompute neighbors: edges that share a face with e
    neighbors = [set() for _ in range(m)]
    for face in faces:
        for e1 in face:
            for e2 in face:
                if e1 != e2:
                    neighbors[e1].add(e2)
    neighbors = [list(s) for s in neighbors]

    # Multi-start Greedy initialization
    best_initial_count = 0
    best_initial_selected = None
    best_initial_face_count = None
    
    for seed in range(20):
        random.seed(seed)
        order = list(range(m))
        random.shuffle(order)
        
        sel = [False] * m
        fc = [0] * len(faces)
        cnt = 0
        for e in order:
            can_add = True
            for fi in face_of_edge[e]:
                if fc[fi] == 3:
                    can_add = False
                    break
            if can_add:
                sel[e] = True
                cnt += 1
                for fi in face_of_edge[e]:
                    fc[fi] += 1
        
        if cnt > best_initial_count:
            best_initial_count = cnt
            best_initial_selected = sel
            best_initial_face_count = fc
            
    selected = best_initial_selected
    face_count = best_initial_face_count
    current_count = best_initial_count
    best_count = current_count
    best_selected = selected[:]
    
    deadline = time.time() + 110  # safety margin

    # Local search: Targeted 1-remove, add-neighbors
    # We try to remove one edge and add any neighbors that become available.
    # If we add more than 1 edge, we gain. If we add 1, we accept (random walk).
    # If we add 0, we revert.
    
    while time.time() < deadline:
        improved = False
        sel_edges = [i for i in range(m) if selected[i]]
        random.shuffle(sel_edges)
        
        for e in sel_edges:
            if time.time() > deadline:
                break
            
            # Try removing e
            selected[e] = False
            for fi in face_of_edge[e]:
                face_count[fi] -= 1
            
            added_edges = []
            
            # Try to add neighbors of e
            # Shuffling neighbors helps explore different configurations
            nbs = neighbors[e][:]
            random.shuffle(nbs)
            
            for u in nbs:
                if selected[u]:
                    continue
                can_add = True
                for fi in face_of_edge[u]:
                    if face_count[fi] >= 3:
                        can_add = False
                        break
                if can_add:
                    selected[u] = True
                    added_edges.append(u)
                    for fi in face_of_edge[u]:
                        face_count[fi] += 1
            
            # Evaluate move
            # We removed 1 edge (e). We added len(added_edges) edges.
            # Net change = len(added_edges) - 1
            if len(added_edges) > 1:
                # Net gain. Accept.
                current_count += len(added_edges) - 1
                improved = True
                if current_count > best_count:
                    best_count = current_count
                    best_selected = selected[:]
            elif len(added_edges) == 1:
                # Net neutral. Accept to move to different local optimum.
                # current_count remains unchanged.
                pass
            else:
                # Net loss. Revert.
                for u in added_edges:
                    selected[u] = False
                    for fi in face_of_edge[u]:
                        face_count[fi] -= 1
                selected[e] = True
                for fi in face_of_edge[e]:
                    face_count[fi] += 1
    
    result = []
    for i in range(m):
        if best_selected[i]:
            result.append([all_edges[i][0], all_edges[i][1]])
    return result
