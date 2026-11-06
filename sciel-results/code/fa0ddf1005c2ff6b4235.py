import math
import time
import random

def construct():
    deadline = time.time() + 110  # Safety margin
    
    # Precompute gcds for all pairs of moduli in [20, 200]
    moduli = list(range(20, 201))
    n_mods = len(moduli)
    
    # gcd_matrix[i][j] = gcd(moduli[i], moduli[j])
    gcd_matrix = [[0] * n_mods for _ in range(n_mods)]
    for i in range(n_mods):
        for j in range(i, n_mods):
            g = math.gcd(moduli[i], moduli[j])
            gcd_matrix[i][j] = g
            gcd_matrix[j][i] = g
            
    def is_valid_residue(r, m_idx, selected_indices, selected_residues):
        """
        Check if residue r for modulus index m_idx is disjoint from all selected residues.
        selected_indices: list of indices of selected moduli
        selected_residues: list of residues for selected moduli
        """
        m = moduli[m_idx]
        for k, sel_idx in enumerate(selected_indices):
            if sel_idx == m_idx:
                continue
            g = gcd_matrix[m_idx][sel_idx]
            if r % g == selected_residues[k] % g:
                return False
        return True

    def get_score(selected_indices):
        return sum(1.0 / moduli[i] for i in selected_indices)

    def greedy_init():
        """
        Greedy construction: add moduli in order of increasing m (decreasing 1/m)
        if a valid residue exists.
        """
        selected_indices = []
        selected_residues = []
        # Sort moduli by 1/m descending (i.e., m ascending)
        # We iterate through all moduli, but prioritize smaller m
        for m_idx in range(n_mods):
            m = moduli[m_idx]
            found = False
            # Try residues 0 to m-1
            for r in range(m):
                if is_valid_residue(r, m_idx, selected_indices, selected_residues):
                    selected_indices.append(m_idx)
                    selected_residues.append(r)
                    found = True
                    break
            # If not found, skip
        return selected_indices, selected_residues

    def local_search(selected_indices, selected_residues, time_limit):
        """
        Simulated Annealing / Local Search to improve the set.
        """
        current_score = get_score(selected_indices)
        best_indices = list(selected_indices)
        best_residues = list(selected_residues)
        best_score = current_score
        
        start_time = time.time()
        temp = 0.5
        cooling = 0.9999
        
        # Precompute valid residues for each modulus given current selection?
        # That's expensive. We check on the fly.
        
        while time.time() < time_limit:
            # Generate a move
            move_type = random.random()
            
            if move_type < 0.4:
                # Try to Add a new modulus
                unselected = [i for i in range(n_mods) if i not in selected_indices]
                if not unselected:
                    continue
                m_idx = random.choice(unselected)
                m = moduli[m_idx]
                # Find a valid residue
                valid_rs = []
                for r in range(m):
                    if is_valid_residue(r, m_idx, selected_indices, selected_residues):
                        valid_rs.append(r)
                if valid_rs:
                    r = random.choice(valid_rs)
                    # Apply move
                    new_indices = selected_indices + [m_idx]
                    new_residues = selected_residues + [r]
                    new_score = get_score(new_indices)
                    delta = new_score - current_score
                    if delta > 0 or random.random() < math.exp(delta / temp):
                        selected_indices = new_indices
                        selected_residues = new_residues
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_indices = list(selected_indices)
                            best_residues = list(selected_residues)
                            
            elif move_type < 0.7:
                # Try to Remove a modulus
                if not selected_indices:
                    continue
                rem_idx = random.choice(selected_indices)
                # Find its position in the list
                pos = selected_indices.index(rem_idx)
                new_indices = selected_indices[:pos] + selected_indices[pos+1:]
                new_residues = selected_residues[:pos] + selected_residues[pos+1:]
                new_score = get_score(new_indices)
                delta = new_score - current_score
                # Removal always decreases score, so only accept if it allows better future moves?
                # In SA, we accept with probability exp(delta/T). Since delta < 0, this is rare unless T is high.
                if random.random() < math.exp(delta / temp):
                    selected_indices = new_indices
                    selected_residues = new_residues
                    current_score = new_score
                    
            else:
                # Try to Change Residue of an existing modulus
                if not selected_indices:
                    continue
                chg_idx = random.choice(selected_indices)
                pos = selected_indices.index(chg_idx)
                m = moduli[chg_idx]
                old_r = selected_residues[pos]
                
                # Find a different valid residue
                # We can just pick a random r and check validity
                # Optimization: pick from valid set if possible, but computing valid set is O(K*m)
                # Let's just try a few random residues
                found_new_r = False
                for _ in range(10):
                    new_r = random.randint(0, m - 1)
                    if new_r == old_r:
                        continue
                    # Check if new_r is valid with respect to all OTHERS
                    valid = True
                    for k, sel_idx in enumerate(selected_indices):
                        if k == pos:
                            continue
                        g = gcd_matrix[chg_idx][sel_idx]
                        if new_r % g == selected_residues[k] % g:
                            valid = False
                            break
                    if valid:
                        # Apply change
                        new_residues = selected_residues[:]
                        new_residues[pos] = new_r
                        # Score doesn't change, but structure does
                        # Accept with some probability to explore, or always?
                        # Always accept if valid, as it doesn't hurt score and might unblock others.
                        # But wait, does it unblock others?
                        # Changing r might make it *less* disjoint from others? No, we checked validity.
                        # Does it help add new moduli? Yes, it changes the constraints on unselected moduli.
                        selected_residues = new_residues
                        found_new_r = True
                        break
                        
        return best_indices, best_residues, best_score

    # Run multiple restarts
    best_global_indices, best_global_residues = greedy_init()
    best_global_score = get_score(best_global_indices)
    
    start_time = time.time()
    num_restarts = 0
    while time.time() < deadline - 5: # Leave 5s for finalization
        # Initialize with greedy, then perturb slightly? Or just greedy + SA
        init_indices, init_residues = greedy_init()
        
        # Run SA for a short duration
        sa_time = min(10.0, deadline - 5 - time.time())
        if sa_time <= 0:
            break
            
        res_indices, res_residues, res_score = local_search(init_indices, init_residues, sa_time)
        
        if res_score > best_global_score:
            best_global_score = res_score
            best_global_indices = res_indices
            best_global_residues = res_residues
            
        num_restarts += 1
        
    # Final result format
    return [[moduli[i], r] for i, r in zip(best_global_indices, best_global_residues)]
