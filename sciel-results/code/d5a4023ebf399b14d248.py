import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    min_m, max_m = 20, 200
    all_moduli = list(range(min_m, max_m + 1))
    n_moduli = len(all_moduli)
    mod_index = {m: i for i, m in enumerate(all_moduli)}
    
    # Precompute gcds
    gcd_table = [[0] * n_moduli for _ in range(n_moduli)]
    for i in range(n_moduli):
        for j in range(i, n_moduli):
            g = math.gcd(all_moduli[i], all_moduli[j])
            gcd_table[i][j] = g
            gcd_table[j][i] = g
            
    # State: list of [m, r]
    # We represent the set as a dictionary m -> r for O(1) lookup
    # But for iteration, a list is fine if we keep it sorted by m
    
    def is_valid(m_new, r_new, state_dict, skip_m=None):
        """Check if (m_new, r_new) is valid against all (m, r) in state_dict, optionally skipping skip_m."""
        idx_new = mod_index[m_new]
        for m, r in state_dict.items():
            if m == skip_m:
                continue
            idx_old = mod_index[m]
            g = gcd_table[idx_new][idx_old]
            if r_new % g == r % g:
                return False
        return True

    def find_valid_residue(m_new, state_dict, skip_m=None):
        """Find any valid residue for m_new. Returns r or None."""
        idx_new = mod_index[m_new]
        for r in range(m_new):
            if is_valid(m_new, r, state_dict, skip_m):
                return r
        return None

    # 1. Initial Greedy Construction
    # Sort moduli ascending to prioritize high 1/m
    state_dict = {}
    for m in sorted(all_moduli):
        r = find_valid_residue(m, state_dict)
        if r is not None:
            state_dict[m] = r
            
    current_score = sum(1.0 / m for m in state_dict)
    
    def calculate_score(state_dict):
        return sum(1.0 / m for m in state_dict)
        
    def local_search(state_dict, max_iters=200):
        """
        Local search with kick-out moves.
        Try to add small m by kicking out larger m.
        """
        current_state = dict(state_dict)
        current_score = calculate_score(current_state)
        improved = True
        iter_count = 0
        
        while improved and iter_count < max_iters:
            iter_count += 1
            improved = False
            # Try to add moduli in ascending order of m
            for m_new in sorted(all_moduli):
                if m_new in current_state:
                    continue
                if time.time() > deadline:
                    return current_state, current_score
                
                # Check if m_new can be added directly
                r_new = find_valid_residue(m_new, current_state)
                if r_new is not None:
                    current_state[m_new] = r_new
                    current_score += 1.0 / m_new
                    improved = True
                    # Restart loop as state changed significantly
                    break 
                
                # Check if we can kick out a larger modulus to make room
                # We want to maximize sum, so we prefer kicking out large m if m_new is small
                # But we need to ensure m_new has a valid residue once m_old is removed
                # Iterate through current moduli, sorted descending by m (to kick out "worst" ones first? 
                # Actually, we want to swap m_old with m_new if m_new < m_old.
                # If m_new > m_old, swapping decreases score, so don't do it.
                # So only consider kicks where m_new < m_old.
                
                # Optimization: Only kick out if m_new is smaller than m_old
                # Sort current moduli descending to try kicking out largest first?
                # Or just iterate and check condition m_new < m_old
                for m_old in sorted(current_state.keys(), reverse=True):
                    if m_new >= m_old:
                        # No point in swapping if new is larger (lower weight)
                        # We are maximizing sum, so 1/m_new > 1/m_old implies m_new < m_old
                        continue
                    
                    # Try to find r_new for m_new assuming m_old is removed
                    r_new = find_valid_residue(m_new, current_state, skip_m=m_old)
                    if r_new is not None:
                        # Check if the rest of the set is still valid (it is, by construction of find_valid_residue with skip)
                        # We just need to make sure m_new doesn't conflict with others, which is handled.
                        # Do the swap
                        del current_state[m_old]
                        current_state[m_new] = r_new
                        current_score += (1.0 / m_new) - (1.0 / m_old)
                        improved = True
                        break # Restart main loop for this iteration
                if improved:
                    break # Restart for loop to re-evaluate with new state
        
        return current_state, current_score

    # Run local search
    state_dict, current_score = local_search(state_dict)
    
    # Try multiple random restarts or different initial greedies if time permits
    # Since the greedy is deterministic, let's try a few perturbation restarts
    import random
    rng = random.Random(42)
    
    best_state = dict(state_dict)
    best_score = current_score
    
    for _ in range(10): # 10 restarts
        if time.time() > deadline - 10:
            break
        
        # Perturb initial state: Randomly shuffle the order of greedy insertion?
        # Or start with a random subset?
        # Let's try a greedy with random tie-breaking for residues
        # Actually, the "kick-out" search is quite strong. 
        # Let's try restarting from a state where we removed a few random elements
        if best_state:
            start_state = dict(best_state)
            # Remove 2 random elements to create a hole
            keys = list(start_state.keys())
            if len(keys) > 2:
                to_remove = rng.sample(keys, 2)
                for k in to_remove:
                    del start_state[k]
            
            # Run local search on this perturbed state
            new_state, new_score = local_search(start_state, max_iters=100)
            if new_score > best_score:
                best_score = new_score
                best_state = new_state
                
    # Final polish: try to add any remaining small moduli
    for m in sorted(all_moduli):
        if time.time() > deadline:
            break
        if m in best_state:
            continue
        r = find_valid_residue(m, best_state)
        if r is not None:
            best_state[m] = r
            best_score += 1.0 / m

    # Convert to required format: [[m, r], ...]
    result = [[m, r] for m, r in best_state.items()]
    return result
