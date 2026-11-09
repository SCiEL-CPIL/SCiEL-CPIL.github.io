<<<<<<< SEARCH
    # ---- Local search refinement -------------------------------------------
    current_points = list(best_points)
    current_score = best_score
    current_target = set(best_target)
    
    # Get the top target area for focused optimization
    if best_target:
        target_area = max(best_target, key=lambda a: 1)  # Pick one target
    else:
        target_area = 2  # Default small area

    random.seed(123)
    step_size = 3
    no_improve = 0
    iter_count = 0
    target_focus = target_area  # Focus on maximizing this specific area count
    
    # Track best target-area count seen
    best_target_count = score_target_area(current_points, target_focus)
    current_target_count = best_target_count
    temperature = 50.0  # Simulated annealing temperature

    while elapsed() < DEADLINE:
        iter_count += 1
        
        # Periodically cool down
        if iter_count % 200 == 0:
            temperature *= 0.97

        idx = random.randint(0, N - 1)
        old = current_points[idx]
        ox, oy = old

        r = random.random()
        if r < 0.03:
            nx = random.randint(-30, 30); ny = random.randint(-30, 30)
        elif r < 0.15:
            nx = ox + random.randint(-8, 8); ny = oy + random.randint(-8, 8)
        else:
            nx = ox + random.randint(-step_size, step_size)
            ny = oy + random.randint(-step_size, step_size)

        if nx < -30 or nx > 30 or ny < -30 or ny > 30:
            continue
        new_pt = (nx, ny)
        if new_pt == old:
            continue

        # collision check
        valid = True
        for j in range(N):
            if j != idx and current_points[j] == new_pt:
                valid = False
                break
        if not valid:
            continue

        current_points[idx] = new_pt
        
        # Fast targeted scoring first
        new_target_count = score_target_area(current_points, target_focus)
        
        # Acceptance: use simulated annealing on target-area count
        delta_tc = new_target_count - current_target_count
        if delta_tc > 0:
            # Always accept improvement in target count
            accept = True
        elif delta_tc == 0:
            # Tie: occasionally accept to explore
            accept = random.random() < 0.3
        else:
            # Deterioration: accept with probability exp(delta/temperature)
            if temperature > 1.0:
                import math
                accept = random.random() < math.exp(delta_tc / temperature)
            else:
                accept = False

        if accept:
            current_target_count = new_target_count
            if new_target_count > best_target_count:
                best_target_count = new_target_count
                no_improve = 0
                # Full evaluation to check if overall score improved
                new_score, new_counts = area_counts_vec(current_points)
                if new_score > current_score:
                    current_score = new_score
                    if new_score > best_score:
                        best_score = new_score
                        best_points = list(current_points)
                        current_target = top_areas(new_counts, 5)
                        if current_target:
                            target_focus = max(current_target, key=lambda a: 1)
                            best_target_count = score_target_area(current_points, target_focus)
                            current_target_count = best_target_count
                elif new_score == current_score:
                    # Same overall score but better target count - update tracking
                    pass
                else:
                    # Score dropped - revert
                    current_points[idx] = old
                    current_target_count = score_target_area(current_points, target_focus)
                    no_improve += 1
            else:
                no_improve += 1
        else:
            current_points[idx] = old
            no_improve += 1

        # Restart from best if stuck
        if no_improve > 500:
            current_points = list(best_points)
            current_score = best_score
            no_improve = 0
            step_size = random.randint(4, 10)
            temperature = max(temperature, 30.0)
            # Occasionally shift target focus
            if best_target and random.random() < 0.4:
                other_targets = [a for a in best_target if a != target_focus]
                if other_targets:
                    target_focus = random.choice(other_targets)
                    best_target_count = score_target_area(current_points, target_focus)
                    current_target_count = best_target_count

        if random.random() < 0.01:
            step_size = random.randint(1, 8)

    # Final full evaluation
    final_score, final_counts = area_counts_vec(best_points)
    if final_score > best_score:
        best_score = final_score

    return best_points
=======
    # ---- Local search refinement -------------------------------------------
    current_points = list(best_points)
    current_score = best_score
    current_target = set(best_target)
    
    # Get the top target areas for focused optimization
    if best_target:
        # Sort target areas by their count in descending order
        target_list = sorted(best_target, key=lambda a: -1)  # all equal weight initially
        # Use the top few as candidate targets
        candidate_targets = list(best_target)[:5]
    else:
        candidate_targets = [2, 4, 6, 8, 10]

    random.seed(123)
    step_size = 3
    no_improve = 0
    iter_count = 0
    target_idx = 0
    target_focus = candidate_targets[0] if candidate_targets else 2
    
    # Track best target-area count seen
    best_target_count = score_target_area(current_points, target_focus)
    current_target_count = best_target_count
    temperature = 50.0  # Simulated annealing temperature

    # Precompute pair indices for delta computation
    # For each point i, store which pairs involve it
    point_pairs = defaultdict(list)  # point index -> list of (pair_idx, other_point_idx)
    for pi in range(_M):
        ii = _ii[pi]; jj = _jj[pi]
        point_pairs[ii].append((pi, jj))
        point_pairs[jj].append((pi, ii))

    def delta_target_count(pts, idx, old_pt, new_pt, target):
        """Compute change in count of triples with doubled area == target when moving point idx."""
        P = np.asarray(pts, dtype=np.int64)
        old_p = np.array(old_pt, dtype=np.int64)
        new_p = np.array(new_pt, dtype=np.int64)
        
        delta = 0
        pairs = point_pairs[idx]
        
        for (pi, other_idx) in pairs:
            other = P[other_idx]
            # Old cross product: (other - old_pt) x (P[j] - old_pt) for each j
            # Actually, the cross product for triangle (idx, other_idx, j) is:
            # (P[other_idx] - P[idx]) x (P[j] - P[idx])
            # When we move P[idx] from old_pt to new_pt:
            # Old: (other - old) x (P[j] - old)
            # New: (other - new) x (P[j] - new)
            
            # For each j != idx, other_idx:
            for j in range(N):
                if j == idx or j == other_idx:
                    continue
                p_j = P[j]
                
                # Old doubled area
                old_vec1 = other - old_p
                old_vec2 = p_j - old_p
                old_cross = old_vec1[0] * old_vec2[1] - old_vec1[1] * old_vec2[0]
                old_abs = abs(old_cross)
                
                # New doubled area
                new_vec1 = other - new_p
                new_vec2 = p_j - new_p
                new_cross = new_vec1[0] * new_vec2[1] - new_vec1[1] * new_vec2[0]
                new_abs = abs(new_cross)
                
                if old_abs == target and new_abs != target:
                    delta -= 1
                elif old_abs != target and new_abs == target:
                    delta += 1
        
        return delta

    while elapsed() < DEADLINE:
        iter_count += 1
        
        # Periodically cool down
        if iter_count % 200 == 0:
            temperature *= 0.97

        idx = random.randint(0, N - 1)
        old = current_points[idx]
        ox, oy = old

        r = random.random()
        if r < 0.03:
            nx = random.randint(-30, 30); ny = random.randint(-30, 30)
        elif r < 0.15:
            nx = ox + random.randint(-8, 8); ny = oy + random.randint(-8, 8)
        else:
            nx = ox + random.randint(-step_size, step_size)
            ny = oy + random.randint(-step_size, step_size)

        if nx < -30 or nx > 30 or ny < -30 or ny > 30:
            continue
        new_pt = (nx, ny)
        if new_pt == old:
            continue

        # collision check
        valid = True
        for j in range(N):
            if j != idx and current_points[j] == new_pt:
                valid = False
                break
        if not valid:
            continue

        # Fast delta computation for target area
        delta_tc = delta_target_count(current_points, idx, old, new_pt, target_focus)
        
        # Acceptance: use simulated annealing on target-area count
        if delta_tc > 0:
            # Always accept improvement in target count
            accept = True
        elif delta_tc == 0:
            # Tie: occasionally accept to explore
            accept = random.random() < 0.3
        else:
            # Deterioration: accept with probability exp(delta/temperature)
            if temperature > 1.0:
                accept = random.random() < math.exp(delta_tc / temperature)
            else:
                accept = False

        if accept:
            current_points[idx] = new_pt
            current_target_count += delta_tc
            if current_target_count > best_target_count:
                best_target_count = current_target_count
                no_improve = 0
                # Full evaluation to check if overall score improved
                new_score, new_counts = area_counts_vec(current_points)
                if new_score > current_score:
                    current_score = new_score
                    if new_score > best_score:
                        best_score = new_score
                        best_points = list(current_points)
                        current_target = top_areas(new_counts, 5)
                        if current_target:
                            # Update candidate targets
                            candidate_targets = list(current_target)[:5]
                            target_focus = candidate_targets[target_idx % len(candidate_targets)]
                            best_target_count = score_target_area(current_points, target_focus)
                            current_target_count = best_target_count
                elif new_score == current_score:
                    # Same overall score but better target count - update tracking
                    pass
                else:
                    # Score dropped - revert
                    current_points[idx] = old
                    current_target_count -= delta_tc
                    no_improve += 1
            else:
                no_improve += 1
        else:
            no_improve += 1

        # Restart from best if stuck
        if no_improve > 500:
            current_points = list(best_points)
            current_score = best_score
            no_improve = 0
            step_size = random.randint(4, 10)
            temperature = max(temperature, 30.0)
            # Shift target focus to another candidate
            if len(candidate_targets) > 1:
                target_idx = (target_idx + 1) % len(candidate_targets)
                target_focus = candidate_targets[target_idx]
                best_target_count = score_target_area(current_points, target_focus)
                current_target_count = best_target_count

        if random.random() < 0.01:
            step_size = random.randint(1, 8)

    # Final full evaluation
    final_score, final_counts = area_counts_vec(best_points)
    if final_score > best_score:
        best_score = final_score

    return best_points
>>>>>>> REPLACE
