<<<<<<< SEARCH
def _first_improvement_ls(perm, pt, deadline, max_blocks=3):
    """
    First-improvement local search with insertion moves.
    Returns best permutation found.
    """
    n = len(perm)
    if n <= 1:
        return perm
    
    current_perm = list(perm)
    current_mk = _makespan(current_perm, pt)
    best_perm = current_perm[:]
    best_mk = current_mk
    
    while time.time() < deadline:
        improved = False
        # Try block sizes 1, 2, 3
        for b_size in range(1, min(max_blocks, n) + 1):
            if time.time() >= deadline:
                break
            for start in range(n - b_size + 1):
                if time.time() >= deadline:
                    break
                block = current_perm[start:start+b_size]
                remaining = current_perm[:start] + current_perm[start+b_size:]
                
                for pos in range(len(remaining) + 1):
                    if pos == start or pos == start + b_size:
                        continue
                    
                    new_perm = remaining[:pos] + block + remaining[pos:]
                    mk = _makespan(new_perm, pt)
                    
                    if mk < current_mk:
                        current_perm = new_perm
                        current_mk = mk
                        if mk < best_mk:
                            best_mk = mk
                            best_perm = current_perm[:]
                        improved = True
                        break  # First improvement: restart scan
                
                if improved:
                    break
            
            if improved:
                break
        
        if not improved:
            # No improvement found - try a random perturbation to escape local optimum
            if time.time() < deadline - 2.0 and n >= 2:
                # Perturb with a random insertion move
                i = random.randint(0, n-1)
                j = random.randint(0, n-1)
                if i != j:
                    perturbed = current_perm[:]
                    job = perturbed.pop(i)
                    perturbed.insert(j, job)
                    p_mk = _makespan(perturbed, pt)
                    # Accept perturbation (helps escape local optima)
                    current_perm = perturbed
                    current_mk = p_mk
            else:
                break
    
    return best_perm
=======
def _fast_local_search(perm, pt, deadline):
    """
    Fast first-improvement local search with single-job insertion moves.
    Optimized for speed by minimizing overhead.
    """
    n = len(perm)
    if n <= 1:
        return perm
    
    current_perm = list(perm)
    current_mk = _makespan(current_perm, pt)
    best_perm = current_perm[:]
    best_mk = current_mk
    
    while time.time() < deadline:
        improved = False
        # Try moving each job to every other position
        for i in range(n):
            if time.time() >= deadline:
                break
            job = current_perm[i]
            # Remove job
            remaining = current_perm[:i] + current_perm[i+1:]
            
            for j in range(len(remaining) + 1):
                if time.time() >= deadline:
                    break
                # Skip original position
                if j == i or j == i - 1:
                    continue
                
                new_perm = remaining[:j] + [job] + remaining[j:]
                mk = _makespan(new_perm, pt)
                
                if mk < current_mk:
                    current_perm = new_perm
                    current_mk = mk
                    if mk < best_mk:
                        best_mk = mk
                        best_perm = current_perm[:]
                    improved = True
                    break  # Restart scan after first improvement
        
        if not improved:
            # No improvement found in a full pass. 
            # Accept a random perturbation to escape local optimum if time permits.
            if time.time() < deadline - 1.0 and n >= 2:
                i = random.randint(0, n-1)
                j = random.randint(0, n-1)
                if i != j:
                    perturbed = current_perm[:]
                    job = perturbed.pop(i)
                    perturbed.insert(j, job)
                    p_mk = _makespan(perturbed, pt)
                    current_perm = perturbed
                    current_mk = p_mk
            else:
                break
    
    return best_perm
>>>>>>> REPLACE

