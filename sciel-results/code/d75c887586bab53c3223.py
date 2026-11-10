# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from math import gcd

def _normalize_line(A, B, C):
    if A == 0 and B == 0 and C == 0:
        return (0, 0, 0)
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g > 0:
        A //= g
        B //= g
        C //= g
    if A < 0:
        A, B, C = -A, -B, -C
    elif A == 0 and B < 0:
        A, B, C = -A, -B, -C
    return (A, B, C)

def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = start_time + 585.0
    
    N = 54
    M = 27
    
    # 1. Initial Configuration: 6x9 Grid
    k_x = 6
    k_y = 9
    xs = np.arange(k_x)
    ys = np.arange(k_y)
    px, py = np.meshgrid(xs, ys, indexing='ij')
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    # Lines construction
    lines_list = []
    # Vertical: x = i
    for i in range(k_x):
        lines_list.append([1, 0, -i])
    # Horizontal: y = j
    for j in range(k_y):
        lines_list.append([0, 1, -j])
    # Diagonals: x - y = c
    candidates_c = []
    for c in range(-8, 6):
        y_min = max(0, -c)
        y_max = min(8, 5 - c)
        if y_max >= y_min:
            count = y_max - y_min + 1
            candidates_c.append((count, c))
    candidates_c.sort(key=lambda x: x[0], reverse=True)
    for _, c in candidates_c[:12]:
        lines_list.append([1, -1, -c])
    
    lines = np.array(lines_list, dtype=np.int64)
    
    # Build incidence matrix M: M[i,j] = True if point i is on line j
    x_col = points[:, 0][:, np.newaxis]  # (N,1)
    y_col = points[:, 1][:, np.newaxis]  # (N,1)
    A_row = lines[:, 0][np.newaxis, :]   # (1,M)
    B_row = lines[:, 1][np.newaxis, :]   # (1,M)
    C_row = lines[:, 2][np.newaxis, :]   # (1,M)
    Mmat = (x_col * A_row + y_col * B_row + C_row) == 0  # (N,M)
    
    I = int(np.sum(Mmat))
    best_I = I
    best_points = points.copy()
    best_lines = lines.copy()
    best_Mmat = Mmat.copy()
    
    # Point set for O(1) duplicate checking
    point_set = set(map(tuple, points.tolist()))
    # Line norms for duplicate checking
    norm_lines = [_normalize_line(int(lines[i,0]), int(lines[i,1]), int(lines[i,2])) for i in range(M)]
    norm_set = set(norm_lines)
    
    # SA Parameters
    T = 5.0
    T_min = 0.01
    cooling_rate = 0.99995
    
    iteration = 0
    while time.time() < deadline:
        iteration += 1
        if random.random() < 0.45:
            # Move a Point
            i = random.randint(0, N - 1)
            old_px, old_py = int(points[i, 0]), int(points[i, 1])
            old_row_sum = int(np.sum(Mmat[i]))
            
            best_delta = -1000
            best_new_px = None
            best_new_py = None
            best_new_row = None
            
            # Try neighbors in a spiral pattern for better coverage
            for dx in range(-4, 5):
                for dy in range(-4, 5):
                    if dx == 0 and dy == 0:
                        continue
                    new_px = old_px + dx
                    new_py = old_py + dy
                    if (new_px, new_py) in point_set:
                        continue
                    
                    # Compute new row: which lines pass through (new_px, new_py)
                    new_row = (lines[:, 0] * new_px + lines[:, 1] * new_py + lines[:, 2]) == 0
                    new_row_sum = int(np.sum(new_row))
                    delta = new_row_sum - old_row_sum
                    
                    if delta > best_delta:
                        best_delta = delta
                        best_new_px = new_px
                        best_new_py = new_py
                        best_new_row = new_row
            
            if best_new_px is None:
                continue
            
            # Acceptance
            if best_delta > 0 or random.random() < np.exp(best_delta / T):
                # Update
                point_set.discard((old_px, old_py))
                point_set.add((best_new_px, best_new_py))
                points[i, 0] = best_new_px
                points[i, 1] = best_new_py
                Mmat[i] = best_new_row
                I += best_delta
                
                if I > best_I:
                    best_I = I
                    best_points = points.copy()
                    best_lines = lines.copy()
                    best_Mmat = Mmat.copy()
        else:
            # Move a Line (Retarget through two points)
            j = random.randint(0, M - 1)
            old_col_sum = int(np.sum(Mmat[:, j]))
            old_norm = norm_lines[j]
            
            best_delta = -1000
            best_new_A = None
            best_new_B = None
            best_new_C = None
            best_new_norm = None
            best_new_col = None
            
            # Try pairs of points
            # Sample 40 pairs, prioritizing pairs that are already on many common lines
            for _ in range(40):
                a = random.randint(0, N - 1)
                b = random.randint(0, N - 1)
                if a == b:
                    continue
                
                x_a, y_a = int(points[a, 0]), int(points[a, 1])
                x_b, y_b = int(points[b, 0]), int(points[b, 1])
                
                A = y_b - y_a
                B = x_a - x_b
                C = -(A * x_a + B * y_a)
                
                if A == 0 and B == 0:
                    continue
                
                norm = _normalize_line(A, B, C)
                
                # Check duplicate
                if norm in norm_set and norm != old_norm:
                    continue
                
                # Compute new column
                new_col = (A * points[:, 0] + B * points[:, 1] + C) == 0
                new_col_sum = int(np.sum(new_col))
                delta = new_col_sum - old_col_sum
                
                if delta > best_delta:
                    best_delta = delta
                    best_new_A = A
                    best_new_B = B
                    best_new_C = C
                    best_new_norm = norm
                    best_new_col = new_col
            
            if best_new_A is None:
                continue
            
            # Acceptance
            if best_delta > 0 or random.random() < np.exp(best_delta / T):
                # Update
                norm_set.discard(old_norm)
                norm_set.add(best_new_norm)
                norm_lines[j] = best_new_norm
                lines[j, 0] = best_new_A
                lines[j, 1] = best_new_B
                lines[j, 2] = best_new_C
                Mmat[:, j] = best_new_col
                I += best_delta
                
                if I > best_I:
                    best_I = I
                    best_points = points.copy()
                    best_lines = lines.copy()
                    best_Mmat = Mmat.copy()
        
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
    
    # Final polish: try to improve each point and line with exhaustive local search
    # This is a deterministic improvement pass
    improved = True
    polish_deadline = start_time + 595.0
    while improved and time.time() < polish_deadline:
        improved = False
        # Polish points
        for i in range(N):
            old_px, old_py = int(best_points[i, 0]), int(best_points[i, 1])
            old_row_sum = int(np.sum(best_Mmat[i]))
            best_local_delta = 0
            best_local_px, best_local_py = old_px, old_py
            best_local_row = best_Mmat[i].copy()
            for dx in range(-5, 6):
                for dy in range(-5, 6):
                    if dx == 0 and dy == 0:
                        continue
                    new_px = old_px + dx
                    new_py = old_py + dy
                    # Check against all other best points
                    is_dup = False
                    for j in range(N):
                        if j == i:
                            continue
                        if int(best_points[j, 0]) == new_px and int(best_points[j, 1]) == new_py:
                            is_dup = True
                            break
                    if is_dup:
                        continue
                    new_row = (best_lines[:, 0] * new_px + best_lines[:, 1] * new_py + best_lines[:, 2]) == 0
                    new_row_sum = int(np.sum(new_row))
                    delta = new_row_sum - old_row_sum
                    if delta > best_local_delta:
                        best_local_delta = delta
                        best_local_px = new_px
                        best_local_py = new_py
                        best_local_row = new_row
            if best_local_delta > 0:
                best_points[i, 0] = best_local_px
                best_points[i, 1] = best_local_py
                best_Mmat[i] = best_local_row
                best_I += best_local_delta
                improved = True
        
        # Polish lines
        for j in range(M):
            old_A, old_B, old_C = int(best_lines[j, 0]), int(best_lines[j, 1]), int(best_lines[j, 2])
            old_col_sum = int(np.sum(best_Mmat[:, j]))
            old_norm = _normalize_line(old_A, old_B, old_C)
            best_local_delta = 0
            best_local_A, best_local_B, best_local_C = old_A, old_B, old_C
            best_local_norm = old_norm
            best_local_col = best_Mmat[:, j].copy()
            for a in range(N):
                for b in range(a+1, N):
                    x_a, y_a = int(best_points[a, 0]), int(best_points[a, 1])
                    x_b, y_b = int(best_points[b, 0]), int(best_points[b, 1])
                    A = y_b - y_a
                    B = x_a - x_b
                    C = -(A * x_a + B * y_a)
                    if A == 0 and B == 0:
                        continue
                    norm = _normalize_line(A, B, C)
                    # Check duplicate against all other lines
                    is_dup = False
                    for k in range(M):
                        if k == j:
                            continue
                        other_norm = _normalize_line(int(best_lines[k,0]), int(best_lines[k,1]), int(best_lines[k,2]))
                        if other_norm == norm:
                            is_dup = True
                            break
                    if is_dup:
                        continue
                    new_col = (A * best_points[:, 0] + B * best_points[:, 1] + C) == 0
                    new_col_sum = int(np.sum(new_col))
                    delta = new_col_sum - old_col_sum
                    if delta > best_local_delta:
                        best_local_delta = delta
                        best_local_A = A
                        best_local_B = B
                        best_local_C = C
                        best_local_norm = norm
                        best_local_col = new_col
            if best_local_delta > 0:
                best_lines[j, 0] = best_local_A
                best_lines[j, 1] = best_local_B
                best_lines[j, 2] = best_local_C
                best_Mmat[:, j] = best_local_col
                best_I += best_local_delta
                improved = True
    
    return best_points, best_lines

# EVOLVE-BLOCK-END
