import time
from math import gcd
from fractions import Fraction
import random

# EVOLVE-BLOCK-START

def _is_compatible(m, r, chosen):
    """Check if (m, r) is compatible with all chosen (m_i, r_i)."""
    for (m_i, r_i) in chosen:
        g = gcd(m, m_i)
        if r % g == r_i % g:
            return False
    return True

def _greedy_select(moduli_order):
    """Greedy: for each modulus in order, pick first valid residue."""
    chosen = []
    for m in moduli_order:
        for r in range(m):
            if _is_compatible(m, r, chosen):
                chosen.append((m, r))
                break
    return chosen

def _compute_sum(chosen):
    return sum(Fraction(1, m) for (m, r) in chosen)

def _hill_climb(initial_set, time_limit=100):
    """
    Hill climbing with moves:
    1. Add a modulus (with valid residue)
    2. Remove a modulus
    3. Change residue of a modulus (to enable future additions)
    4. Swap: remove one, add another
    """
    start_time = time.time()
    current_set = list(initial_set)
    current_sum = _compute_sum(current_set)
    best_set = list(current_set)
    best_sum = current_sum
    
    chosen_moduli = set(m for (m, r) in current_set)
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # Move 1: Add a modulus
        for m in range(20, 201):
            if m in chosen_moduli:
                continue
            for r in range(m):
                if _is_compatible(m, r, current_set):
                    new_set = current_set + [(m, r)]
                    new_sum = current_sum + Fraction(1, m)
                    if new_sum > best_sum:
                        current_set = new_set
                        current_sum = new_sum
                        best_set = list(current_set)
                        best_sum = new_sum
                        chosen_moduli.add(m)
                        improved = True
                        break
            if improved:
                break
        
        if improved:
            continue
        
        # Move 2: Try removing a modulus and see if we can add two others
        for i, (m_i, r_i) in enumerate(current_set):
            temp_set = [x for j, x in enumerate(current_set) if j != i]
            temp_sum = current_sum - Fraction(1, m_i)
            temp_moduli = chosen_moduli - {m_i}
            
            # Try to add moduli to temp_set
            added_any = False
            for m in range(20, 201):
                if m in temp_moduli:
                    continue
                for r in range(m):
                    if _is_compatible(m, r, temp_set):
                        new_set = temp_set + [(m, r)]
                        new_sum = temp_sum + Fraction(1, m)
                        if new_sum > best_sum:
                            current_set = new_set
                            current_sum = new_sum
                            best_set = list(current_set)
                            best_sum = new_sum
                            chosen_moduli = set(m for (m, r) in current_set)
                            added_any = True
                            improved = True
                            break
                if improved:
                    break
            
            if improved:
                break
        
        if not improved:
            break
    
    return best_set, best_sum

def construct():
    start_time = time.time()
    deadline = start_time + 135
    
    # Initial greedy
    moduli_order = list(range(20, 201))
    initial_set = _greedy_select(moduli_order)
    
    # Hill climb
    best_set, best_sum = _hill_climb(initial_set, time_limit=120)
    
    # Try more initializations if time permits
    if time.time() < deadline - 15:
        random.seed(123)
        for seed in range(20):
            if time.time() > deadline - 5:
                break
            moduli_order = list(range(20, 201))
            random.shuffle(moduli_order)
            initial_set = _greedy_select(moduli_order)
            set2, sum2 = _hill_climb(initial_set, time_limit=5)
            if sum2 > best_sum:
                best_set, best_sum = set2, sum2
    
    result = [[m, r] for (m, r) in best_set]
    return result

# EVOLVE-BLOCK-END
