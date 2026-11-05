import time
import math
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    moduli = list(range(20, 201))
    n_moduli = len(moduli)
    
    best_selected = []
    best_score = 0.0

    def is_compatible(m_new, r_new, selected_list):
        for m_old, r_old in selected_list:
            g = math.gcd(m_new, m_old)
            if (r_new - r_old) % g == 0:
                return False
        return True

    def get_valid_residues(m, selected_list):
        valid = []
        for r in range(m):
            if is_compatible(m, r, selected_list):
                valid.append(r)
        return valid

    def run_random_greedy():
        selected = []
        for m in moduli:
            valid_residues = get_valid_residues(m, selected)
            if valid_residues:
                r = random.choice(valid_residues)
                selected.append([m, r])
        
        score = sum(Fraction(1, m) for m, r in selected)
        return selected, float(score)

    def run_beam_search(beam_width=50):
        """
        Beam search over moduli in increasing order.
        At each step, keep the top beam_width partial solutions by score.
        For each partial solution, try adding the next modulus with various residues.
        """
        # beam: list of (score, selected_list)
        beam = [(Fraction(0, 1), [])]
        
        for m in moduli:
            new_beam = []
            for score, selected in beam:
                valid_residues = get_valid_residues(m, selected)
                if valid_residues:
                    # Try a subset of valid residues (or all if few)
                    if len(valid_residues) <= 10:
                        candidates = valid_residues
                    else:
                        # Sample up to 10 residues
                        candidates = random.sample(valid_residues, min(10, len(valid_residues)))
                    
                    for r in candidates:
                        new_score = score + Fraction(1, m)
                        new_selected = selected + [[m, r]]
                        new_beam.append((new_score, new_selected))
            
            if not new_beam:
                break
            
            # Keep top beam_width by score
            new_beam.sort(key=lambda x: x[0], reverse=True)
            beam = new_beam[:beam_width]
            
            if time.time() > deadline - 30.0:
                break
        
        # Return the best from the beam
        if beam:
            best = max(beam, key=lambda x: x[0])
            return best[1], float(best[0])
        return [], 0.0

    def local_polish(selected, max_iter=20000):
        """
        Final local polish: try to improve by swapping moduli or changing residues.
        Uses a stricter acceptance criterion (only accept improvements).
        """
        current = [[m, r] for m, r in selected]
        current_score = sum(Fraction(1, m) for m, r in current)
        best = [list(item) for item in current]
        best_score = current_score
        
        selected_moduli_set = set(m for m, r in current)
        
        for it in range(max_iter):
            if time.time() > deadline - 2.0:
                break
            
            improved = False
            move_type = random.random()
            
            if move_type < 0.4 and len(current) > 0:
                # Change residue
                idx = random.randint(0, len(current) - 1)
                m_old, r_old = current[idx]
                others = [current[j] for j in range(len(current)) if j != idx]
                valid_residues = get_valid_residues(m_old, others)
                
                if valid_residues:
                    new_r = random.choice(valid_residues)
                    if new_r != r_old:
                        new_state = [list(item) for item in current]
                        new_state[idx][1] = new_r
                        # Score doesn't change, but this might enable future improvements
                        # For now, skip since no score change
                        pass
                        
            elif move_type < 0.7 and len(current) > 0:
                # Remove a modulus
                idx = random.randint(0, len(current) - 1)
                m_old, r_old = current[idx]
                new_state = [list(item) for item in current]
                del new_state[idx]
                new_score = current_score - Fraction(1, m_old)
                
                # Only accept if it enables adding a better modulus later
                # For now, skip removals in polish
                
            else:
                # Add an unselected modulus
                unselected = [m for m in moduli if m not in selected_moduli_set]
                if unselected:
                    # Try the smallest few unselected moduli
                    unselected.sort()
                    for m_new in unselected[:5]:
                        valid_residues = get_valid_residues(m_new, current)
                        if valid_residues:
                            r_new = random.choice(valid_residues)
                            new_state = [list(item) for item in current]
                            new_state.append([m_new, r_new])
                            new_score = current_score + Fraction(1, m_new)
                            
                            if new_score > current_score:
                                current = new_state
                                current_score = new_score
                                selected_moduli_set = set(m for m, r in current)
                                improved = True
                                break
                            
                            # Also try swapping with a larger selected modulus
                            if len(current) > 0:
                                # Find a selected modulus larger than m_new
                                for j in range(len(current)):
                                    m_sel, r_sel = current[j]
                                    if m_sel > m_new:
                                        others = [current[k] for k in range(len(current)) if k != j]
                                        valid_residues_swap = get_valid_residues(m_new, others)
                                        if valid_residues_swap:
                                            r_new_swap = random.choice(valid_residues_swap)
                                            new_state_swap = [list(item) for item in current]
                                            new_state_swap[j] = [m_new, r_new_swap]
                                            new_score_swap = current_score - Fraction(1, m_sel) + Fraction(1, m_new)
                                            
                                            if new_score_swap > current_score:
                                                current = new_state_swap
                                                current_score = new_score_swap
                                                selected_moduli_set = set(m for m, r in current)
                                                improved = True
                                                break
                                if improved:
                                    break
            
            if current_score > best_score:
                best_score = current_score
                best = [list(item) for item in current]
            
            if not improved and it % 1000 == 0:
                # Periodically try a fresh add
                unselected = [m for m in moduli if m not in selected_moduli_set]
                if unselected:
                    unselected.sort()
                    for m_new in unselected[:3]:
                        valid_residues = get_valid_residues(m_new, current)
                        if valid_residues:
                            r_new = random.choice(valid_residues)
                            new_state = [list(item) for item in current]
                            new_state.append([m_new, r_new])
                            new_score = current_score + Fraction(1, m_new)
                            if new_score > current_score:
                                current = new_state
                                current_score = new_score
                                selected_moduli_set = set(m for m, r in current)
                                break
        
        return best, float(best_score)

    def simulated_annealing(initial_selected, max_iter=50000):
        """
        SA over the set of selected moduli and their residues.
        State: list of [m, r] pairs
        Moves: 
          1. Change residue of a random selected modulus
          2. Swap a selected modulus with an unselected one (if compatible)
          3. Add an unselected modulus (if compatible)
          4. Remove a selected modulus
        """
        current = [[m, r] for m, r in initial_selected]
        current_score = sum(Fraction(1, m) for m, r in current)
        
        sa_best = list(current)
        sa_best_score = current_score
        
        # Build lookup for current selection
        selected_moduli_set = set(m for m, r in current)
        
        # Temperature schedule
        T_start = 0.05
        T_end = 0.001
        cooling = (T_end / T_start) ** (1.0 / max_iter)
        T = T_start
        
        for it in range(max_iter):
            T *= cooling
            if time.time() > deadline - 2.0:
                break
            
            # Choose a move type
            move_type = random.random()
            
            new_state = None
            new_score = None
            
            if move_type < 0.5 and len(current) > 0:
                # Move 1: Change residue of a random selected modulus
                idx = random.randint(0, len(current) - 1)
                m_old, r_old = current[idx]
                
                # Find valid residues for this modulus given all OTHER selected
                others = [current[j] for j in range(len(current)) if j != idx]
                valid_residues = get_valid_residues(m_old, others)
                
                if valid_residues:
                    new_r = random.choice(valid_residues)
                    new_state = [list(item) for item in current]
                    new_state[idx][1] = new_r
                    new_score = current_score
                    
            elif move_type < 0.7 and len(current) > 0:
                # Move 2: Remove a random selected modulus
                idx = random.randint(0, len(current) - 1)
                m_old, r_old = current[idx]
                new_state = [list(item) for item in current]
                del new_state[idx]
                new_score = current_score - Fraction(1, m_old)
                
            elif move_type < 0.9:
                # Move 3: Add a random unselected modulus
                unselected = [m for m in moduli if m not in selected_moduli_set]
                if unselected:
                    m_new = random.choice(unselected)
                    valid_residues = get_valid_residues(m_new, current)
                    if valid_residues:
                        r_new = random.choice(valid_residues)
                        new_state = [list(item) for item in current]
                        new_state.append([m_new, r_new])
                        new_score = current_score + Fraction(1, m_new)
                        
            else:
                # Move 4: Swap a selected modulus with an unselected one
                if len(current) > 0:
                    unselected = [m for m in moduli if m not in selected_moduli_set]
                    if unselected:
                        idx = random.randint(0, len(current) - 1)
                        m_old, r_old = current[idx]
                        m_new = random.choice(unselected)
                        
                        # Check if m_new is compatible with all others except m_old
                        others = [current[j] for j in range(len(current)) if j != idx]
                        valid_residues = get_valid_residues(m_new, others)
                        
                        if valid_residues:
                            r_new = random.choice(valid_residues)
                            new_state = [list(item) for item in current]
                            new_state[idx] = [m_new, r_new]
                            new_score = current_score - Fraction(1, m_old) + Fraction(1, m_new)
            
            if new_state is not None and new_score is not None:
                delta = new_score - current_score
                if delta > 0 or random.random() < math.exp(delta / T):
                    current = new_state
                    current_score = new_score
                    selected_moduli_set = set(m for m, r in current)
                    
                    if current_score > sa_best_score:
                        sa_best_score = current_score
                        sa_best = [list(item) for item in current]
        
        return sa_best, float(sa_best_score)

    # Phase 1: Run random greedy and beam search to get good starting points
    greedy_restarts = 0
    while time.time() < deadline - 40.0 and greedy_restarts < 30:
        selected, score = run_random_greedy()
        if score > best_score:
            best_score = score
            best_selected = selected
        greedy_restarts += 1
    
    # Phase 2: Run beam search
    if time.time() < deadline - 30.0:
        beam_selected, beam_score = run_beam_search(beam_width=50)
        if beam_score > best_score:
            best_score = beam_score
            best_selected = beam_selected
    
    # Phase 3: Apply Simulated Annealing to the best solution found
    if best_selected and time.time() < deadline - 10.0:
        sa_selected, sa_score = simulated_annealing(best_selected, max_iter=80000)
        if sa_score > best_score:
            best_score = sa_score
            best_selected = sa_selected
    
    # Phase 4: If time remains, do more SA restarts from random greedy starts
    while time.time() < deadline - 5.0:
        # Get a new random greedy start
        selected, score = run_random_greedy()
        
        # Apply SA to this start
        sa_selected, sa_score = simulated_annealing(selected, max_iter=30000)
        
        if sa_score > best_score:
            best_score = sa_score
            best_selected = sa_selected
    
    # Phase 5: Final local polish
    if best_selected and time.time() < deadline - 1.0:
        polished_selected, polished_score = local_polish(best_selected, max_iter=20000)
        if polished_score > best_score:
            best_score = polished_score
            best_selected = polished_selected

    return best_selected
