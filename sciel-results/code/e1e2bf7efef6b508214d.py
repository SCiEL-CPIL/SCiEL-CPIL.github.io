import math
import time

def construct():
    deadline = time.time() + 115  # Safety margin
    
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    def is_compatible(m1, r1, m2, r2):
        g = gcd(m1, m2)
        return (r1 % g) != (r2 % g)

    def find_valid_residue(m, selected):
        # Returns a list of all valid residues for m given selected
        valid_residues = []
        for r in range(m):
            if all(is_compatible(m, r, sm, sr) for sm, sr in selected):
                valid_residues.append(r)
        return valid_residues

    def get_score(selected):
        return sum(1.0 / m for m, r in selected)

    # Initial Greedy Construction
    # We try to pick residues that are "central" or random to avoid bias, 
    # but for now, just pick the first valid one.
    selected = []
    for m in range(20, 201):
        if time.time() > deadline - 10:
            break
        valid_rs = find_valid_residue(m, selected)
        if valid_rs:
            # Heuristic: pick the middle residue? Or just the first?
            # Let's pick the first one for now.
            selected.append((m, valid_rs[0]))
    
    best_selected = [list(x) for x in selected]
    best_score = get_score(best_selected)
    
    # Local Search
    while time.time() < deadline - 5:
        improved = False
        
        # 1. Try to add any missing modulus
        for m in range(20, 201):
            if any(sm == m for sm, _ in selected):
                continue
            
            valid_rs = find_valid_residue(m, selected)
            if valid_rs:
                # Try adding with the first valid residue
                new_selected = selected + [(m, valid_rs[0])]
                new_score = get_score(new_selected)
                if new_score > best_score:
                    best_score = new_score
                    best_selected = [list(x) for x in new_selected]
                    selected = new_selected
                    improved = True
                    break # Restart loop to update selected state

        if improved:
            continue

        # 2. Try to swap a selected modulus with an unselected one
        selected_moduli = {m for m, _ in selected}
        for m_old in selected_moduli:
            if time.time() > deadline - 5:
                break
            # Find the index of m_old
            idx = next(i for i, (m, r) in enumerate(selected) if m == m_old)
            r_old = selected[idx][1]
            
            # Try swapping with any unselected m_new
            for m_new in range(20, 201):
                if m_new in selected_moduli:
                    continue
                
                # Remove m_old
                temp_selected = [x for i, x in enumerate(selected) if i != idx]
                
                # Try to find a valid residue for m_new
                valid_rs = find_valid_residue(m_new, temp_selected)
                if valid_rs:
                    # We can potentially add m_new. 
                    # But we removed m_old. Net change is +1/m_new - 1/m_old.
                    # If 1/m_new > 1/m_old, it's likely an improvement.
                    # However, the residue choice for m_new might affect future additions, 
                    # but for this move, we just check immediate gain.
                    new_selected = temp_selected + [(m_new, valid_rs[0])]
                    new_score = get_score(new_selected)
                    if new_score > best_score:
                        best_score = new_score
                        best_selected = [list(x) for x in new_selected]
                        selected = new_selected
                        improved = True
                        break
            
            if improved:
                break
        
        if not improved:
            # Try to change residues of existing moduli to enable additions?
            # This is harder. Let's try a simple "kick" move:
            # Randomly change residue of one selected modulus to a different valid one,
            # then try to add any missing modulus.
            
            # Pick a random selected modulus
            if not selected:
                break
                
            idx = 0 # Just use the first one for simplicity in this basic version
            m, r = selected[idx]
            
            valid_rs = find_valid_residue(m, [x for i, x in enumerate(selected) if i != idx])
            if len(valid_rs) > 1:
                # Change to a different valid residue
                new_r = valid_rs[1] if valid_rs[0] == r else valid_rs[0]
                selected[idx] = (m, new_r)
                
                # Now try to add any missing modulus
                for m_new in range(20, 201):
                    if any(sm == m_new for sm, _ in selected):
                        continue
                    valid_rs_new = find_valid_residue(m_new, selected)
                    if valid_rs_new:
                        new_selected = selected + [(m_new, valid_rs_new[0])]
                        new_score = get_score(new_selected)
                        if new_score > best_score:
                            best_score = new_score
                            best_selected = [list(x) for x in new_selected]
                            selected = new_selected
                            improved = True
                            break

        if not improved and time.time() > deadline - 10:
            break
        # If no improvement in a full pass, we might be stuck. 
        # A more advanced version would do simulated annealing or restarts.
        # For now, if no improvement, break to save time? 
        # Or continue trying different residue assignments?
        # Let's add a simple counter to break if stuck for too long.
        # Actually, the "kick" move above is weak. 
        # Let's just break if no improvement found in Add/Swap.
        # But wait, the kick move is inside the loop. 
        # If improved is False after the kick attempt, we should break to avoid infinite loop.
        if not improved:
             # One more try: random restart of local search?
             # No, just break.
             break

    return [[m, r] for m, r in best_selected]
