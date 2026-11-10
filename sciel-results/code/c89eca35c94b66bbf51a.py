import random
import time

def construct():
    d = 7
    n = 1 << d
    m = d * (1 << (d - 1))  # 448 edges
    
    # Build edges
    edges = []
    edge_index = {}
    for u in range(n):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                idx = len(edges)
                edges.append((u, v))
                edge_index[(u, v)] = idx
                edge_index[(v, u)] = idx
                
    # Build faces
    faces = []
    face_for_edge = [[] for _ in range(m)]
    
    for i in range(d):
        for j in range(i+1, d):
            for k in range(1 << (d-2)):
                u = 0
                bit_pos = 0
                for b in range(d):
                    if b == i or b == j:
                        continue
                    if k & (1 << bit_pos):
                        u |= (1 << b)
                    bit_pos += 1
                
                v1 = u ^ (1 << i)
                v2 = u ^ (1 << j)
                v3 = u ^ (1 << i) ^ (1 << j)
                
                e1 = edge_index[(min(u, v1), max(u, v1))]
                e2 = edge_index[(min(v1, v3), max(v1, v3))]
                e3 = edge_index[(min(v3, v2), max(v3, v2))]
                e4 = edge_index[(min(v2, u), max(v2, u))]
                
                f_idx = len(faces)
                faces.append([e1, e2, e3, e4])
                
                face_for_edge[e1].append(f_idx)
                face_for_edge[e2].append(f_idx)
                face_for_edge[e3].append(f_idx)
                face_for_edge[e4].append(f_idx)

    num_faces = len(faces)
    
    selected = [False] * m
    face_counts = [0] * num_faces
    
    def add_edge(idx):
        if selected[idx]:
            return False
        for f in face_for_edge[idx]:
            if face_counts[f] >= 3:
                return False
        selected[idx] = True
        for f in face_for_edge[idx]:
            face_counts[f] += 1
        return True

    def remove_edge(idx):
        if not selected[idx]:
            return False
        selected[idx] = False
        for f in face_for_edge[idx]:
            face_counts[f] -= 1
        return True

    def get_count():
        return sum(selected)

    def greedy_init():
        for i in range(m):
            selected[i] = False
        for i in range(num_faces):
            face_counts[i] = 0
        order = list(range(m))
        random.shuffle(order)
        for idx in order:
            add_edge(idx)

    def local_search_add_only(max_iters=20000):
        added = 0
        for _ in range(max_iters):
            idx = random.randint(0, m-1)
            if not selected[idx]:
                if add_edge(idx):
                    added += 1
        return added

    start_time = time.time()
    deadline = start_time + 140
    
    best_count = 0
    best_solution = []
    
    greedy_init()
    local_search_add_only(50000)
    best_count = get_count()
    best_solution = [list(edges[i]) for i in range(m) if selected[i]]
    
    while time.time() < deadline:
        # Check if we can add more
        # If local search didn't add anything, we are stuck.
        # We do a kick.
        
        # Try a short local search first to see if we are stuck
        added = local_search_add_only(1000)
        
        if added == 0:
            # Stuck, perform kick
            curr = get_count()
            # Remove 10-20%
            num_to_remove = max(1, int(curr * 0.15))
            for _ in range(num_to_remove):
                sel_edges = [i for i in range(m) if selected[i]]
                if not sel_edges:
                    break
                idx = random.choice(sel_edges)
                remove_edge(idx)
            
            # Greedy refill
            order = list(range(m))
            random.shuffle(order)
            for idx in order:
                add_edge(idx)
                
            # Long local search
            local_search_add_only(20000)
            
            curr = get_count()
            if curr > best_count:
                best_count = curr
                best_solution = [list(edges[i]) for i in range(m) if selected[i]]
        else:
            # We are still improving, keep going
            local_search_add_only(10000)
            curr = get_count()
            if curr > best_count:
                best_count = curr
                best_solution = [list(edges[i]) for i in range(m) if selected[i]]

    return best_solution
