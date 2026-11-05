"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # All candidate moduli in [20, 200]
    all_moduli = list(range(20, 201))
    num_moduli = len(all_moduli)
    
    # Precompute gcds for speed
    gcd_table = {}
    for i, m1 in enumerate(all_moduli):
        for j, m2 in enumerate(all_moduli):
            if i != j:
                gcd_table[(m1, m2)] = math.gcd(m1, m2)
    
    def is_compatible(m1, r1, m2, r2):
        """Check if (m1, r1) is disjoint from (m2, r2)."""
        g = gcd_table.get((m1, m2))
        if g is None:
            g = math.gcd(m1, m2)
        return (r1 % g) != (r2 % g)
    
    def is_compatible_with_all(m_new, r_new, selected, exclude_index=None):
        """Check if (m_new, r_new) is disjoint from all in selected, except at exclude_index."""
        for i, (m_old, r_old) in enumerate(selected):
            if i == exclude_index:
                continue
            if not is_compatible(m_new, r_new, m_old, r_old):
                return False
        return True

    def get_valid_residues(m_new, selected, exclude_index=None):
        """Get all valid residues for m_new given selected classes."""
        valid = []
        for r in range(m_new):
            if is_compatible_with_all(m_new, r, selected, exclude_index):
                valid.append(r)
        return valid

    def score_state(selected):
        """Calculate the score of the state."""
        s = 0.0
        for m, r in selected:
            s += 1.0 / m
        return s

    def greedy_build(moduli_order, randomize_residues=False):
        """
        Greedily build a set of disjoint residue classes.
        """
        selected = []
        for m in moduli_order:
            valid = get_valid_residues(m, selected)
            if not valid:
                continue
            if randomize_residues:
                r = random.choice(valid)
            else:
                r = valid[0]
            selected.append((m, r))
        return selected

    def simulated_annealing(initial_state, time_budget):
        """
        Run Simulated Annealing starting from initial_state.
        """
        state = initial_state[:]
        current_score = score_state(state)
        best_state = state[:]
        best_score = current_score
        
        # Temperature schedule
        T = 0.5
        T_min = 0.001
        cooling_rate = 0.995
        
        end_time = time.time() + time_budget
        
        while time.time() < end_time and T > T_min:
            # Generate a neighbor
            move_type = random.random()
            neighbor = state[:]
            neighbor_score = current_score
            
            if move_type < 0.3:
                # Add a random class
                # Pick a modulus not in state
                current_mods = {m for m, r in state}
                available_mods = [m for m in all_moduli if m not in current_mods]
                if available_mods:
                    m_new = random.choice(available_mods)
                    valid_res = get_valid_residues(m_new, state)
                    if valid_res:
                        r_new = random.choice(valid_res)
                        neighbor.append((m_new, r_new))
                        neighbor_score += 1.0 / m_new
                    else:
                        # If no valid residue, just skip this move (stay same)
                        continue
            elif move_type < 0.6:
                # Remove a random class
                if state:
                    idx = random.randint(0, len(state) - 1)
                    m_rem, r_rem = state[idx]
                    neighbor.pop(idx)
                    neighbor_score -= 1.0 / m_rem
                else:
                    continue
            elif move_type < 0.9:
                # Perturb residue of a random class
                if state:
                    idx = random.randint(0, len(state) - 1)
                    m_cur, r_cur = state[idx]
                    # Try a random new residue
                    r_new = random.randint(0, m_cur - 1)
                    if r_new == r_cur:
                        continue
                    # Check if r_new is compatible with all other classes
                    if is_compatible_with_all(m_cur, r_new, state, exclude_index=idx):
                        # Update neighbor
                        neighbor[idx] = (m_cur, r_new)
                        # Score doesn't change because modulus is same
                    else:
                        continue
                else:
                    continue
            else:
                # Swap: Remove one, add another (different modulus)
                if state:
                    idx_rem = random.randint(0, len(state) - 1)
                    m_rem, r_rem = state[idx_rem]
                    neighbor.pop(idx_rem)
                    neighbor_score -= 1.0 / m_rem
                    
                    current_mods = {m for m, r in neighbor}
                    available_mods = [m for m in all_moduli if m not in current_mods]
                    if available_mods:
                        m_new = random.choice(available_mods)
                        valid_res = get_valid_residues(m_new, neighbor)
                        if valid_res:
                            r_new = random.choice(valid_res)
                            neighbor.append((m_new, r_new))
                            neighbor_score += 1.0 / m_new
                        else:
                            # Revert
                            neighbor.insert(idx_rem, (m_rem, r_rem))
                            neighbor_score += 1.0 / m_rem
                            continue
                    else:
                        # Revert
                        neighbor.insert(idx_rem, (m_rem, r_rem))
                        neighbor_score += 1.0 / m_rem
                        continue
            
            # Acceptance criterion
            delta = neighbor_score - current_score
            if delta > 0 or random.random() < math.exp(delta / T):
                state = neighbor
                current_score = neighbor_score
                if current_score > best_score:
                    best_score = current_score
                    best_state = state[:]
            
            T *= cooling_rate
        
        return best_state, best_score

    # 1. Generate multiple initial states using greedy
    initial_states = []
    for _ in range(20):
        if time.time() > deadline:
            break
        moduli_order = all_moduli[:]
        random.shuffle(moduli_order)
        # Try different strategies: some deterministic, some random
        if random.random() < 0.5:
            state = greedy_build(moduli_order, randomize_residues=True)
        else:
            state = greedy_build(moduli_order, randomize_residues=False)
        initial_states.append(state)
    
    # Also add a fully random valid state construction
    # Start empty, add random valid classes until stuck
    state = []
    for _ in range(1000): # Max attempts
        available_mods = [m for m in all_moduli if m not in {mm for mm, rr in state}]
        if not available_mods:
            break
        m_new = random.choice(available_mods)
        valid_res = get_valid_residues(m_new, state)
        if valid_res:
            r_new = random.choice(valid_res)
            state.append((m_new, r_new))
    initial_states.append(state)

    # 2. Run SA on each initial state
    best_global_state = []
    best_global_score = 0.0
    
    # Distribute time among SA runs
    num_sa_runs = len(initial_states)
    # Estimate time per run
    # We have ~115s. Let's give each run 5s if we have 20 runs? 
    # But SA converges quickly. Let's do 5 seconds per run, max 10 runs.
    time_per_run = 5.0
    
    for i, init_state in enumerate(initial_states):
        if time.time() > deadline:
            break
        if not init_state:
            continue
            
        sa_state, sa_score = simulated_annealing(init_state, time_per_run)
        
        if sa_score > best_global_score:
            best_global_score = sa_score
            best_global_state = sa_state[:]
            
    # 3. Final Polish: A quick local search on the best state found
    # Try to add any missing moduli that fit
    # This is a "completion" step
    final_state = best_global_state[:]
    current_score = best_global_score
    
    # Sort final state by modulus for consistency? Not required, but nice.
    # final_state.sort(key=lambda x: x[0])
    
    # One last pass to see if we can add any more
    changed = True
    while changed:
        changed = False
        current_mods = {m for m, r in final_state}
        for m in all_moduli:
            if m in current_mods:
                continue
            valid_res = get_valid_residues(m, final_state)
            if valid_res:
                # Pick the one that maximizes... well, all give 1/m. 
                # Just pick the first valid one.
                r = valid_res[0]
                final_state.append((m, r))
                current_score += 1.0 / m
                changed = True
                current_mods.add(m)
        
        if time.time() > deadline:
            break

    # Verify validity of the final result
    def verify(state):
        for i in range(len(state)):
            for j in range(i+1, len(state)):
                m1, r1 = state[i]
                m2, r2 = state[j]
                if not is_compatible(m1, r1, m2, r2):
                    return False
        return True

    if not verify(final_state):
        # Fallback to best_global_state if it was valid
        if verify(best_global_state):
            final_state = best_global_state
        else:
            # Fallback to a simple greedy result if all else fails
            final_state = greedy_build(all_moduli, randomize_residues=False)

    # Convert to required format: [[m, r], ...]
    result = [[m, r] for m, r in final_state]
    return result
# EVOLVE-BLOCK-END
