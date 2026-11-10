<<<<<<< SEARCH
def improve_configuration(lines, candidate_points, time_limit, seed=42):
    """
    Joint optimization of points and lines using simulated annealing.
    """
    random.seed(seed)
    current_lines = lines.copy()
    current_points, _ = get_best_points_for_lines(current_lines, candidate_points)
    current_I = count_incidences(current_points, current_lines)
    
    best_lines = current_lines.copy()
    best_points = current_points.copy()
    best_I = current_I
    
    start_time = time.time()
    T = 10.0
    cooling = 0.9995
    min_T = 0.5
    
    no_improve = 0
    n_cand_pts = len(candidate_points)
    
    while time.time() - start_time < time_limit:
        if no_improve > 400:
            T = max(T, 10.0)
            no_improve = 0
            
        move_type = random.random()
        
        if move_type < 0.35:
            # Replace a line with a line through a pair of current points
            idx = random.randint(0, 26)
            old_line = current_lines[idx].copy()
            
            p1_idx, p2_idx = random.sample(range(54), 2)
            p1 = current_points[p1_idx]
            p2 = current_points[p2_idx]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            A = dy
            B = -dx
            C = dx * p1[1] - dy * p1[0]
            
            new_line = normalize_line(int(A), int(B), int(C))
            if new_line is None:
                continue
            
            new_line_tup = tuple(new_line)
            if any(tuple(l) == new_line_tup for l in current_lines):
                continue
                
            current_lines[idx] = new_line
            new_points, _ = get_best_points_for_lines(current_lines, candidate_points)
            new_I = count_incidences(new_points, current_lines)
            
            delta_I = new_I - current_I
            
            if delta_I > 0 or (T > 0 and random.random() < np.exp(delta_I / T)):
                current_I = new_I
                current_points = new_points
                if current_I > best_I:
                    best_I = current_I
                    best_lines = current_lines.copy()
                    best_points = current_points.copy()
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                current_lines[idx] = old_line
                no_improve += 1
                
        elif move_type < 0.60:
            # Replace a line with a random line from small coefficients
            idx = random.randint(0, 26)
            old_line = current_lines[idx].copy()
            
            A = random.randint(-7, 7)
            B = random.randint(-7, 7)
            C = random.randint(-30, 30)
            
            new_line = normalize_line(A, B, C)
            if new_line is None:
                continue
            
            new_line_tup = tuple(new_line)
            if any(tuple(l) == new_line_tup for l in current_lines):
                continue
                
            current_lines[idx] = new_line
            new_points, _ = get_best_points_for_lines(current_lines, candidate_points)
            new_I = count_incidences(new_points, current_lines)
            
            delta_I = new_I - current_I
            
            if delta_I > 0 or (T > 0 and random.random() < np.exp(delta_I / T)):
                current_I = new_I
                current_points = new_points
                if current_I > best_I:
                    best_I = current_I
                    best_lines = current_lines.copy()
                    best_points = current_points.copy()
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                current_lines[idx] = old_line
                no_improve += 1
                
        elif move_type < 0.85:
            # Swap a point
            pt_idx = random.randint(0, 53)
            old_pt = current_points[pt_idx].copy()
            
            cand_idx = random.randint(0, n_cand_pts-1)
            new_pt = candidate_points[cand_idx].copy()
            
            if any(np.array_equal(current_points[i], new_pt) for i in range(54)):
                continue
            
            current_points[pt_idx] = new_pt
            new_I = count_incidences(current_points, current_lines)
            
            delta_I = new_I - current_I
            
            if delta_I > 0 or (T > 0 and random.random() < np.exp(delta_I / T)):
                current_I = new_I
                if current_I > best_I:
                    best_I = current_I
                    best_lines = current_lines.copy()
                    best_points = current_points.copy()
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                current_points[pt_idx] = old_pt
                no_improve += 1
                
        else:
            # Rebuild lines greedily
            new_lines = greedy_select_lines_from_points(current_points, 27)
            
            unique_lines = []
            for l in new_lines:
                nl = normalize_line(int(l[0]), int(l[1]), int(l[2]))
                if nl and tuple(nl) not in [tuple(u) for u in unique_lines]:
                    unique_lines.append(nl)
            
            if len(unique_lines) >= 27:
                new_lines = np.array(unique_lines[:27], dtype=np.int64)
                new_points, _ = get_best_points_for_lines(new_lines, candidate_points)
                new_I = count_incidences(new_points, new_lines)
                
                delta_I = new_I - current_I
                
                if delta_I > 0 or (T > 0 and random.random() < np.exp(delta_I / (T * 0.5))):
                    current_I = new_I
                    current_lines = new_lines
                    current_points = new_points
                    if current_I > best_I:
                        best_I = current_I
                        best_lines = current_lines.copy()
                        best_points = current_points.copy()
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    no_improve += 1
                
        T *= cooling
        T = max(T, min_T)
        
    return best_lines, best_points, best_I
=======
def improve_configuration(lines, candidate_points, time_limit, seed=42):
    """
    Joint optimization of points and lines using simulated annealing with
    a smooth surrogate objective (soft-max over incidence counts) and
    an auxiliary variable t for epigraph-style smoothing.
    """
    random.seed(seed)
    current_lines = lines.copy()
    current_points, _ = get_best_points_for_lines(current_lines, candidate_points)
    current_I = count_incidences(current_points, current_lines)
    
    best_lines = current_lines.copy()
    best_points = current_points.copy()
    best_I = current_I
    
    start_time = time.time()
    T = 10.0
    cooling = 0.9995
    min_T = 0.5
    
    no_improve = 0
    n_cand_pts = len(candidate_points)
    
    # Auxiliary variable t for smooth surrogate: soft-max temperature
    t = 1.0
    t_cooling = 0.999
    t_min = 0.1
    
    def smooth_score(points, lines):
        """
        Compute a smooth surrogate for the incidence count using log-sum-exp.
        For each line, compute how many points lie on it; use soft-max to get
        a differentiable-like objective.
        """
        P = points.astype(np.int64)
        L = lines.astype(np.int64)
        
        px = P[:, 0].reshape(-1, 1)
        py = P[:, 1].reshape(-1, 1)
        
        A = L[:, 0].reshape(1, -1)
        B = L[:, 1].reshape(1, -1)
        C = L[:, 2].reshape(1, -1)
        
        vals = A * px + B * py + C
        # For each line, count incidences
        line_counts = np.sum(vals == 0, axis=0)
        
        # Soft-max surrogate: sum of exp(count_i / t)
        scaled = line_counts.astype(np.float64) / t
        max_scaled = np.max(scaled)
        log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
        return float(log_sum_exp)
    
    def exact_score(points, lines):
        return count_incidences(points, lines)
    
    while time.time() - start_time < time_limit:
        if no_improve > 400:
            T = max(T, 10.0)
            t = max(t, 1.0)
            no_improve = 0
            
        move_type = random.random()
        
        if move_type < 0.35:
            # Replace a line with a line through a pair of current points
            idx = random.randint(0, 26)
            old_line = current_lines[idx].copy()
            
            p1_idx, p2_idx = random.sample(range(54), 2)
            p1 = current_points[p1_idx]
            p2 = current_points[p2_idx]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            A = dy
            B = -dx
            C = dx * p1[1] - dy * p1[0]
            
            new_line = normalize_line(int(A), int(B), int(C))
            if new_line is None:
                continue
            
            new_line_tup = tuple(new_line)
            if any(tuple(l) == new_line_tup for l in current_lines):
                continue
                
            current_lines[idx] = new_line
            new_points, _ = get_best_points_for_lines(current_lines, candidate_points)
            new_I = count_incidences(new_points, current_lines)
            
            # Use smooth surrogate for acceptance decision when T is high
            if T > 5.0:
                smooth_curr = smooth_score(current_points, current_lines)
                smooth_new = smooth_score(new_points, current_lines)
                delta_smooth = smooth_new - smooth_curr
                accept = delta_I > 0 or (T > 0 and random.random() < np.exp(delta_smooth / T))
            else:
                accept = delta_I > 0 or (T > 0 and random.random() < np.exp(delta_I / T))
            
            if accept:
                current_I = new_I
                current_points = new_points
                if current_I > best_I:
                    best_I = current_I
                    best_lines = current_lines.copy()
                    best_points = current_points.copy()
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                current_lines[idx] = old_line
                no_improve += 1
                
        elif move_type < 0.60:
            # Replace a line with a random line from small coefficients
            idx = random.randint(0, 26)
            old_line = current_lines[idx].copy()
            
            A = random.randint(-7, 7)
            B = random.randint(-7, 7)
            C = random.randint(-30, 30)
            
            new_line = normalize_line(A, B, C)
            if new_line is None:
                continue
            
            new_line_tup = tuple(new_line)
            if any(tuple(l) == new_line_tup for l in current_lines):
                continue
                
            current_lines[idx] = new_line
            new_points, _ = get_best_points_for_lines(current_lines, candidate_points)
            new_I = count_incidences(new_points, current_lines)
            
            if T > 5.0:
                smooth_curr = smooth_score(current_points, current_lines)
                smooth_new = smooth_score(new_points, current_lines)
                delta_smooth = smooth_new - smooth_curr
                accept = delta_I > 0 or (T > 0 and random.random() < np.exp(delta_smooth / T))
            else:
                accept = delta_I > 0 or (T > 0 and random.random() < np.exp(delta_I / T))
            
            if accept:
                current_I = new_I
                current_points = new_points
                if current_I > best_I:
                    best_I = current_I
                    best_lines = current_lines.copy()
                    best_points = current_points.copy()
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                current_lines[idx] = old_line
                no_improve += 1
                
        elif move_type < 0.85:
            # Swap a point
            pt_idx = random.randint(0, 53)
            old_pt = current_points[pt_idx].copy()
            
            cand_idx = random.randint(0, n_cand_pts-1)
            new_pt = candidate_points[cand_idx].copy()
            
            if any(np.array_equal(current_points[i], new_pt) for i in range(54)):
                continue
            
            current_points[pt_idx] = new_pt
            new_I = count_incidences(current_points, current_lines)
            
            if T > 5.0:
                smooth_curr = smooth_score(current_points, current_lines)
                smooth_new = smooth_score(current_points, current_lines)
                # For point swap, the lines are the same, so use exact delta
                accept = delta_I > 0 or (T > 0 and random.random() < np.exp(delta_I / T))
            else:
                accept = delta_I > 0 or (T > 0 and random.random() < np.exp(delta_I / T))
            
            if accept:
                current_I = new_I
                if current_I > best_I:
                    best_I = current_I
                    best_lines = current_lines.copy()
                    best_points = current_points.copy()
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                current_points[pt_idx] = old_pt
                no_improve += 1
                
        else:
            # Rebuild lines greedily
            new_lines = greedy_select_lines_from_points(current_points, 27)
            
            unique_lines = []
            for l in new_lines:
                nl = normalize_line(int(l[0]), int(l[1]), int(l[2]))
                if nl and tuple(nl) not in [tuple(u) for u in unique_lines]:
                    unique_lines.append(nl)
            
            if len(unique_lines) >= 27:
                new_lines = np.array(unique_lines[:27], dtype=np.int64)
                new_points, _ = get_best_points_for_lines(new_lines, candidate_points)
                new_I = count_incidences(new_points, new_lines)
                
                delta_I = new_I - current_I
                
                if T > 5.0:
                    smooth_curr = smooth_score(current_points, current_lines)
                    smooth_new = smooth_score(new_points, new_lines)
                    delta_smooth = smooth_new - smooth_curr
                    accept = delta_I > 0 or (T > 0 and random.random() < np.exp(delta_smooth / (T * 0.5)))
                else:
                    accept = delta_I > 0 or (T > 0 and random.random() < np.exp(delta_I / (T * 0.5)))
                
                if accept:
                    current_I = new_I
                    current_lines = new_lines
                    current_points = new_points
                    if current_I > best_I:
                        best_I = current_I
                        best_lines = current_lines.copy()
                        best_points = current_points.copy()
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    no_improve += 1
                
        T *= cooling
        T = max(T, min_T)
        t *= t_cooling
        t = max(t, t_min)
        
    return best_lines, best_points, best_I
>>>>>>> REPLACE

