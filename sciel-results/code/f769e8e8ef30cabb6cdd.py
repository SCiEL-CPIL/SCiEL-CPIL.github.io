import math
import time
import random

def _is_compatible(m1, r1, m2, r2):
    d = math.gcd(m1, m2)
    return (r1 % d) != (r2 % d)

def _precompute_compatibility():
    """Precompute for each modulus m and each residue r, which other moduli
    have compatible residues. Returns a dict: (m, r) -> set of (m2, r2) compatible."""
    # Actually, let's precompute for each pair of moduli (m1, m2), the set of
    # incompatible residue pairs.
    # For efficiency, store: for each m1, m2, the set of r1 values that are
    # incompatible with each r2 value.
    # Better: for each (m1, m2), store a mapping from r1 to set of incompatible r2.
    
    moduli = list(range(20, 201))
    compat = {}
    for m1 in moduli:
        compat[m1] = {}
        for m2 in moduli:
            if m1 == m2:
                continue
            d = math.gcd(m1, m2)
            # For each r1 in [0, m1), find which r2 in [0, m2) are incompatible
            # Incompatible if r1 % d == r2 % d
            incompat_map = {}
            for r1 in range(m1):
                base = r1 % d
                # r2 is incompatible if r2 % d == base
                # Number of such r2: m2 // d
                incompat_map[r1] = base
            compat[m1][m2] = incompat_map
    return compat

def _find_valid_residues_fast(m, selected, compat):
    """Find valid residues for modulus m given selected classes, using precomputed compat."""
    valid = list(range(m))
    for (m_i, r_i) in selected:
        d = math.gcd(m, m_i)
        base_i = r_i % d
        # Remove residues r where r % d == base_i
        valid = [r for r in valid if r % d != base_i]
    return valid

def _greedy_construct_sorted(compat, random_seed=None):
    """Greedy construction processing moduli in order of 1/m (largest first)."""
    if random_seed is not None:
        random.seed(random_seed)
    
    selected = []
    # Process moduli in order of decreasing 1/m (i.e., increasing m)
    for m in range(20, 201):
        valid_residues = _find_valid_residues_fast(m, selected, compat)
        if valid_residues:
            # Choose residue that maximizes future flexibility
            # Heuristic: choose residue that leaves most options for remaining moduli
            best_r = None
            best_score = -1
            for r in valid_residues:
                # Score: number of remaining moduli that still have valid residues
                # This is expensive, so use a simpler heuristic
                # Choose residue that is "central" or has some other property
                # For now, just pick randomly or the first one
                best_r = r
                break
            selected.append((m, best_r))
    return selected

def _greedy_construct_with_lookahead(compat, random_seed=None, lookahead_depth=3):
    """Greedy with limited lookahead to choose better residues."""
    if random_seed is not None:
        random.seed(random_seed)
    
    selected = []
    remaining_moduli = list(range(20, 201))
    
    while remaining_moduli:
        # Choose the next modulus to add: the one with fewest valid residues (MRV heuristic)
        # Or process in order of 1/m
        best_m = None
        best_valids = None
        best_count = float('inf')
        
        for m in remaining_moduli:
            valids = _find_valid_residues_fast(m, selected, compat)
            if not valids:
                # Can't add this modulus, skip it for now
                continue
            if len(valids) < best_count:
                best_count = len(valids)
                best_m = m
                best_valids = valids
        
        if best_m is None:
            break
        
        # Choose residue for best_m
        # Heuristic: choose residue that maximizes the minimum number of valid residues
        # for the next few moduli
        best_r = best_valids[0]
        best_lookahead_score = -1
        
        for r in best_valids:
            temp_selected = selected + [(best_m, r)]
            
            # Look ahead: check how many remaining moduli still have valid residues
            valid_count = 0
            min_valids = float('inf')
            for m2 in remaining_moduli:
                if m2 == best_m:
                    continue
                valids2 = _find_valid_residues_fast(m2, temp_selected, compat)
                if valids2:
                    valid_count += 1
                    min_valids = min(min_valids, len(valids2))
            
            # Score: prefer choices that leave more options
            score = valid_count * 1000 + (min_valids if min_valids != float('inf') else 0)
            if score > best_lookahead_score:
                best_lookahead_score = score
                best_r = r
        
        selected.append((best_m, best_r))
        remaining_moduli.remove(best_m)
    
    return selected

def _local_improve_v2(classes, compat, time_limit=10):
    """Local improvement: try to add unselected moduli or swap residues."""
    start_time = time.time()
    selected = list(classes)
    selected_moduli = {m for m, r in selected}
    unselected = [m for m in range(20, 201) if m not in selected_moduli]
    unselected.sort(key=lambda m: 1.0/m, reverse=True)  # Prioritize small moduli
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for m_new in list(unselected):
            if (time.time() - start_time) > time_limit:
                break
            
            valids_new = _find_valid_residues_fast(m_new, selected, compat)
            if valids_new:
                # Choose best residue for new modulus
                r_new = valids_new[0]
                selected.append((m_new, r_new))
                selected_moduli.add(m_new)
                unselected.remove(m_new)
                improved = True
                continue
            
            # Try to swap a residue to make room for m_new
            found_change = False
            for i in range(len(selected)):
                if (time.time() - start_time) > time_limit:
                    break
                m_i, r_i = selected[i]
                others = [s for j, s in enumerate(selected) if j != i]
                valid_for_i = _find_valid_residues_fast(m_i, others, compat)
                
                if not valid_for_i:
                    continue
                
                for r_i_new in valid_for_i:
                    if r_i_new == r_i:
                        continue
                    temp_selected = list(selected)
                    temp_selected[i] = (m_i, r_i_new)
                    
                    if _find_valid_residues_fast(m_new, temp_selected, compat):
                        selected[i] = (m_i, r_i_new)
                        valids = _find_valid_residues_fast(m_new, selected, compat)
                        r_new = valids[0]
                        selected.append((m_new, r_new))
                        selected_moduli.add(m_new)
                        unselected.remove(m_new)
                        improved = True
                        found_change = True
                        break
                if found_change:
                    break
    
    return selected

def _score(classes):
    return sum(1.0 / m for m, r in classes)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 40s margin for safety
    
    # Precompute compatibility
    compat = _precompute_compatibility()
    
    best_score = 0.0
    best_result = []
    
    # Strategy 1: Greedy with lookahead
    classes1 = _greedy_construct_with_lookahead(compat, random_seed=42, lookahead_depth=3)
    score1 = _score(classes1)
    if score1 > best_score:
        best_score = score1
        best_result = classes1
    
    # Strategy 2: Multiple random restarts with local improvement
    num_restarts = 30
    for i in range(num_restarts):
        if time.time() > deadline:
            break
        
        seed = i * 7919 + 104729
        classes = _greedy_construct_sorted(compat, random_seed=seed)
        
        remaining = deadline - time.time()
        if remaining < 1:
            break
        
        time_for_improve = min(2.0, remaining * 0.4)
        improved_classes = _local_improve_v2(classes, compat, time_for_improve)
        
        score = _score(improved_classes)
        if score > best_score:
            best_score = score
            best_result = improved_classes
    
    # Fallback: deterministic construction
    if not best_result:
        assigned = []
        for m in range(20, 201):
            valids = _find_valid_residues_fast(m, assigned, compat)
            if valids:
                assigned.append((m, valids[0]))
        best_result = [[m, r] for (m, r) in assigned]
    
    return best_result
