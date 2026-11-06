import time
import random
import math

def construct():
    d = 7
    n = 1 << d  # 128 vertices
    deadline = time.time() + 110  # Safety margin

    # Precompute edges
    all_edges = []
    edge_idx = {}
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(all_edges)
                all_edges.append((x, y))
                edge_idx[(x, y)] = idx
    m = len(all_edges)  # 448

    # Precompute faces
    face_edges = []
    edge_faces = [[] for _ in range(m)]
    
    for i in range(d):
        for j in range(i + 1, d):
            for x in range(n):
                if (x >> i) & 1 == 0 and (x >> j) & 1 == 0:
                    a = x
                    b = x ^ (1 << i)
                    c = x ^ (1 << j)
                    d2 = x ^ (1 << i) ^ (1 << j)
                    
                    e1 = edge_idx[(min(a, b), max(a, b))]
                    e2 = edge_idx[(min(b, d2), max(b, d2))]
                    e3 = edge_idx[(min(d2, c), max(d2, c))]
                    e4 = edge_idx[(min(c, a), max(c, a))]
                    
                    fid = len(face_edges)
                    face_edges.append([e1, e2, e3, e4])
                    for ei in (e1, e2, e3, e4):
                        edge_faces[ei].append(fid)
                        
    n_faces = len(face_edges)  # 672

    # Greedy Hitting Set Initialization
    def greedy_init():
        R = [False] * m
        face_count = [0] * n_faces
        uncovered = [True] * n_faces
        num_uncovered = n_faces
        
        while num_uncovered > 0:
            best_e = -1
            best_gain = -1
            for e in range(m):
                if R[e]:
                    continue
                gain = 0
                for f in edge_faces[e]:
                    if uncovered[f]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_e = e
            
            if best_e == -1:
                break
                
            R[best_e] = True
            for f in edge_faces[best_e]:
                if uncovered[f]:
                    uncovered[f] = False
                    num_uncovered -= 1
                face_count[f] += 1
                
        return R, face_count

    R, face_count = greedy_init()
    current_size = sum(R)
    best_size = current_size
    best_R = R[:]
    
    def get_uncovered_count(fc):
        cnt = 0
        for c in fc:
            if c == 0:
                cnt += 1
        return cnt

    current_uncovered = get_uncovered_count(face_count)
    
    LAMBDA = 10.0
    
    def get_cost(size, unc):
        return size + LAMBDA * unc

    current_cost = get_cost(current_size, current_uncovered)
    
    T = 1.0
    T_min = 0.001
    cooling = 0.9995
    
    while time.time() < deadline:
        if T < T_min:
            T = T_min
            break
            
        e = random.randint(0, m - 1)
        
        new_size = current_size
        new_uncovered = current_uncovered
        delta_cost = 0.0
        
        if R[e]:
            safe = True
            inc_unc = 0
            for f in edge_faces[e]:
                if face_count[f] == 1:
                    safe = False
                    inc_unc += 1
            
            new_size = current_size - 1
            new_uncovered = current_uncovered + (0 if safe else inc_unc)
            new_cost = get_cost(new_size, new_uncovered)
            delta_cost = new_cost - current_cost
            
            R[e] = False
            for f in edge_faces[e]:
                face_count[f] -= 1
            current_size = new_size
            current_uncovered = new_uncovered
        else:
            dec_unc = 0
            for f in edge_faces[e]:
                if face_count[f] == 0:
                    dec_unc += 1
            new_uncovered = current_uncovered - dec_unc
            new_size = current_size + 1
            new_cost = get_cost(new_size, new_uncovered)
            delta_cost = new_cost - current_cost
            
            R[e] = True
            for f in edge_faces[e]:
                face_count[f] += 1
            current_size = new_size
            current_uncovered = new_uncovered
            
        accept = False
        if delta_cost < 0:
            accept = True
        else:
            if random.random() < math.exp(-delta_cost / T):
                accept = True
        
        if not accept:
            # Revert
            if R[e]:
                # We just added
                R[e] = False
                for f in edge_faces[e]:
                    face_count[f] -= 1
                current_size -= 1
                current_uncovered += dec_unc
            else:
                # We just removed
                R[e] = True
                for f in edge_faces[e]:
                    face_count[f] += 1
                current_size += 1
                current_uncovered -= inc_unc
                
        if current_uncovered == 0 and current_size < best_size:
            best_size = current_size
            best_R = R[:]
            
        T *= cooling

    # Check validity of best_R
    fc_best = [0] * n_faces
    for e in range(m):
        if best_R[e]:
            for f in edge_faces[e]:
                fc_best[f] += 1
                
    is_valid = all(c > 0 for c in fc_best)
    
    if not is_valid:
        R_g, fc_g = greedy_init()
        best_R = R_g
        best_size = sum(R_g)

    # Final Greedy Reduction Pass
    # Try to remove more edges from best_R while maintaining validity
    # We iterate and try to remove any edge that is "redundant" (face_count > 1 for all its faces)
    changed = True
    while changed:
        changed = False
        # Compute face counts for best_R
        fc = [0] * n_faces
        for e in range(m):
            if best_R[e]:
                for f in edge_faces[e]:
                    fc[f] += 1
        
        for e in range(m):
            if best_R[e]:
                # Check if removing e keeps all faces covered
                redundant = True
                for f in edge_faces[e]:
                    if fc[f] <= 1:
                        redundant = False
                        break
                if redundant:
                    best_R[e] = False
                    changed = True
                    # Update fc
                    for f in edge_faces[e]:
                        fc[f] -= 1
                    break  # Restart scan with new fc

    out = []
    for e in range(m):
        if not best_R[e]:
            x, y = all_edges[e]
            out.append([x, y])
            
    return out

if __name__ == "__main__":
    t0 = time.time()
    edges = construct()
    t1 = time.time()
    print(f"Time: {t1-t0:.2f}s, Edges: {len(edges)}")
    
    d = 7
    n = 1 << d
    all_edges_set = set()
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                all_edges_set.add((x, y))
                
    selected_set = set()
    for u, v in edges:
        selected_set.add((min(u, v), max(u, v)))
        
    valid = True
    for i in range(d):
        for j in range(i + 1, d):
            for x in range(n):
                if (x >> i) & 1 == 0 and (x >> j) & 1 == 0:
                    a = x
                    b = x ^ (1 << i)
                    c = x ^ (1 << j)
                    d2 = x ^ (1 << i) ^ (1 << j)
                    
                    e1 = (min(a, b), max(a, b))
                    e2 = (min(b, d2), max(b, d2))
                    e3 = (min(d2, c), max(d2, c))
                    e4 = (min(c, a), max(c, a))
                    
                    cnt = 0
                    if e1 in selected_set: cnt += 1
                    if e2 in selected_set: cnt += 1
                    if e3 in selected_set: cnt += 1
                    if e4 in selected_set: cnt += 1
                    
                    if cnt == 4:
                        print(f"Invalid face found at {x}, dims {i},{j}")
                        valid = False
                        break
            if not valid:
                break
        if not valid:
            break
    
    if valid:
        print("Valid!")
    else:
        print("Invalid!")
