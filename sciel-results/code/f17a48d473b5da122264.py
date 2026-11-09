<<<<<<< SEARCH
def fast_local_search(classes, max_time=None):
    """Fast local search with precomputed positions and class-removal moves."""
    if max_time is None:
        max_time = time.time() + 10
    
    n = len(classes)
    if n == 0:
        return [], BASE
    
    # Precompute position indices for each class
    positions_list = []
    for i in range(n):
        m, r = classes[i]
        positions_list.append(np.arange(r, BASE, m))
    
    # Build coverage count array
    cov_count = np.zeros(BASE, dtype=np.int16)
    for i in range(n):
        cov_count[positions_list[i]] += 1
    
    current_score = int(np.sum(cov_count == 0))
    best_score = current_score
    best_classes = [list(c) for c in classes]
    
    # Track which classes are active
    active = [True] * n
    
    improved = True
    while improved and time.time() < max_time:
        improved = False
        
        for i in range(n):
            if not active[i]:
                continue
            m, r = classes[i]
            pos = positions_list[i]
            
            # Try residue changes
            best_r = r
            best_new_score = current_score
            
            for r_try in range(m):
                if r_try == r:
                    continue
                new_pos = np.arange(r_try, BASE, m)
                
                # Compute delta in uncovered count
                # Positions that were only covered by this class at old residue
                only_old = (cov_count[pos] == 1)
                # Positions that will be newly covered by this class at new residue
                newly_new = (cov_count[new_pos] == 0)
                
                # But we need to be careful about overlap
                # Old positions losing coverage: those where cov_count==1 and not in new_pos
                # New positions gaining coverage: those where cov_count==0 and in new_pos
                
                # More precise: 
                # After change, cov_count at pos decreases by 1, at new_pos increases by 1
                # But overlap positions (in both pos and new_pos) net zero change
                
                overlap = np.intersect1d(pos, new_pos)
                
                # Positions in pos but not new_pos: lose 1 coverage
                only_in_old = np.setdiff1d(pos, new_pos)
                # Positions in new_pos but not pos: gain 1 coverage  
                only_in_new = np.setdiff1d(new_pos, pos)
                
                # Uncovered increase: positions in only_in_old where cov_count was 1
                uncovered_increase = int(np.sum(cov_count[only_in_old] == 1))
                # Uncovered decrease: positions in only_in_new where cov_count was 0
                uncovered_decrease = int(np.sum(cov_count[only_in_new] == 0))
                
                new_score = current_score + uncovered_increase - uncovered_decrease
                
                if new_score < best_new_score:
                    best_new_score = new_score
                    best_r = r_try
            
            if best_r != r:
                # Apply the best residue change
                old_pos = positions_list[i]
                new_pos = np.arange(best_r, BASE, m)
                overlap = np.intersect1d(old_pos, new_pos)
                only_in_old = np.setdiff1d(old_pos, new_pos)
                only_in_new = np.setdiff1d(new_pos, old_pos)
                
                cov_count[only_in_old] -= 1
                cov_count[only_in_new] += 1
                
                current_score = best_new_score
                classes[i][1] = best_r
                positions_list[i] = new_pos
                improved = True
                
                if current_score < best_score:
                    best_score = current_score
                    best_classes = [list(c) for c in classes if active[_]] if False else [list(c) for c in classes]
                    # Only keep active classes
                    best_classes = [list(classes[j]) for j in range(n) if active[j]]
            
            # Try removing this class
            if m > 1:
                pos = positions_list[i]
                # Uncovered increase if we remove: positions where cov_count == 1
                uncovered_increase = int(np.sum(cov_count[pos] == 1))
                removal_score = current_score + uncovered_increase
                
                if removal_score < best_score:
                    # Remove the class
                    cov_count[pos] -= 1
                    current_score = removal_score
                    active[i] = False
                    best_score = current_score
                    best_classes = [list(classes[j]) for j in range(n) if active[j]]
                    improved = True
    
    return best_classes, best_score
=======
def fast_local_search(classes, max_time=None):
    """Fast local search with precomputed positions and class-removal moves."""
    if max_time is None:
        max_time = time.time() + 10
    
    n = len(classes)
    if n == 0:
        return [], BASE
    
    # Precompute position indices for each class
    positions_list = []
    for i in range(n):
        m, r = classes[i]
        positions_list.append(np.arange(r, BASE, m))
    
    # Build coverage count array
    cov_count = np.zeros(BASE, dtype=np.int16)
    for i in range(n):
        cov_count[positions_list[i]] += 1
    
    current_score = int(np.sum(cov_count == 0))
    best_score = current_score
    best_classes = [list(c) for c in classes]
    
    # Track which classes are active
    active = [True] * n
    
    def get_active_classes():
        return [list(classes[j]) for j in range(n) if active[j]]
    
    improved = True
    while improved and time.time() < max_time:
        improved = False
        
        for i in range(n):
            if not active[i]:
                continue
            m, r = classes[i]
            pos = positions_list[i]
            
            # Try residue changes
            best_r = r
            best_new_score = current_score
            
            for r_try in range(m):
                if r_try == r:
                    continue
                new_pos = np.arange(r_try, BASE, m)
                
                # Positions in old but not new: lose 1 coverage
                only_in_old = np.setdiff1d(pos, new_pos)
                # Positions in new but not old: gain 1 coverage  
                only_in_new = np.setdiff1d(new_pos, pos)
                
                # Uncovered increase: positions in only_in_old where cov_count was 1
                uncovered_increase = int(np.sum(cov_count[only_in_old] == 1))
                # Uncovered decrease: positions in only_in_new where cov_count was 0
                uncovered_decrease = int(np.sum(cov_count[only_in_new] == 0))
                
                new_score = current_score + uncovered_increase - uncovered_decrease
                
                if new_score < best_new_score:
                    best_new_score = new_score
                    best_r = r_try
            
            if best_r != r:
                # Apply the best residue change
                old_pos = positions_list[i]
                new_pos = np.arange(best_r, BASE, m)
                only_in_old = np.setdiff1d(old_pos, new_pos)
                only_in_new = np.setdiff1d(new_pos, old_pos)
                
                cov_count[only_in_old] -= 1
                cov_count[only_in_new] += 1
                
                current_score = best_new_score
                classes[i][1] = best_r
                positions_list[i] = new_pos
                improved = True
                
                if current_score < best_score:
                    best_score = current_score
                    best_classes = get_active_classes()
            
            # Try removing this class
            if m > 1:
                pos = positions_list[i]
                # Uncovered increase if we remove: positions where cov_count == 1
                uncovered_increase = int(np.sum(cov_count[pos] == 1))
                removal_score = current_score + uncovered_increase
                
                if removal_score < best_score:
                    # Remove the class
                    cov_count[pos] -= 1
                    current_score = removal_score
                    active[i] = False
                    best_score = current_score
                    best_classes = get_active_classes()
                    improved = True
    
    return best_classes, best_score
>>>>>>> REPLACE

