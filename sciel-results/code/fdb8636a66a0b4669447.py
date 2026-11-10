import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580
    
    p = 7
    q = p * p
    k = 3
    
    def mul_f49(p1, p2):
        a1, b1 = p1
        a2, b2 = p2
        return ((a1*a2 + k*b1*b2) % p, (a1*b2 + a2*b1) % p)
        
    all_points = [(xa, xb, ya, yb) for xa in range(p) for xb in range(p) for ya in range(p) for yb in range(p)]
    n_points = len(all_points)
    
    point_to_lines = [[] for _ in range(n_points)]
    line_to_points = []
    
    for xa in range(p):
        for xb in range(p):
            pts = [i for i, (x_a, x_b, y_a, y_b) in enumerate(all_points) if x_a == xa and x_b == xb]
            line_idx = len(line_to_points)
            line_to_points.append(pts)
            for pt in pts:
                point_to_lines[pt].append(line_idx)
                
    for ma in range(p):
        for mb in range(p):
            if ma == 0 and mb == 0:
                continue
            lines_dict = {}
            for i, (xa, xb, ya, yb) in enumerate(all_points):
                mx = mul_f49((ma, mb), (xa, xb))
                c = ((ya - mx[0]) % p, (yb - mx[1]) % p)
                if c not in lines_dict:
                    lines_dict[c] = []
                lines_dict[c].append(i)
            for c, pts in lines_dict.items():
                line_idx = len(line_to_points)
                line_to_points.append(pts)
                for pt in pts:
                    point_to_lines[pt].append(line_idx)
                    
    n_lines = len(line_to_points)
    
    best_C_size = 0
    best_in_C = [False] * n_points
    
    max_restarts = 1000
    for restart in range(max_restarts):
        if time.time() > deadline - 5:
            break
            
        in_C = [False] * n_points
        W = [set() for _ in range(n_points)]
        line_sum = [0] * n_lines
        
        indices = list(range(n_points))
        random.shuffle(indices)
        
        for i in indices:
            if in_C[i]:
                continue
                
            candidate_lines = []
            for L in point_to_lines[i]:
                if line_sum[L] == 0:
                    candidate_lines.append(L)
                    
            if not candidate_lines:
                continue
                
            safe = True
            for L in point_to_lines[i]:
                if line_sum[L] == 1:
                    for j in line_to_points[L]:
                        if in_C[j]:
                            if L in W[j]:
                                if len(W[j]) == 1:
                                    safe = False
                                    break
                    if not safe:
                        break
                        
            if safe:
                in_C[i] = True
                W[i] = set(candidate_lines)
                
                for L in point_to_lines[i]:
                    old_sum = line_sum[L]
                    line_sum[L] += 1
                    if old_sum == 1:
                        for j in line_to_points[L]:
                            if in_C[j] and j != i:
                                if L in W[j]:
                                    W[j].remove(L)

        c_size = sum(in_C)
        if c_size > best_C_size:
            best_C_size = c_size
            best_in_C = in_C.copy()
            
    pts = [all_points[i] for i in range(n_points) if not best_in_C[i]]
    result = np.array([[[pt[0], pt[1]], [pt[2], pt[3]]] for pt in pts], dtype=np.int64)
    return result
