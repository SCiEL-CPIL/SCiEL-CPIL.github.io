"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    moduli = list(range(20, 201))
    
    def is_compatible(m_new, r_new, selected_list):
        """Check if (m_new, r_new) is disjoint from all in selected_list."""
        for m_old, r_old in selected_list:
            g = math.gcd(m_new, m_old)
            if (r_new - r_old) % g == 0:
                return False
        return True
    
    def get_valid_residues(m, selected_list):
        """Get all valid residues for modulus m given selected_list."""
        valid = []
        for r in range(m):
            if is_compatible(m, r, selected_list):
                valid.append(r)
        return valid
    
    def greedy_construct(order, residue_strategy='first'):
        """Construct using given order and residue strategy."""
        selected = []
        for m in order:
            valid_r = get_valid_residues(m, selected)
            if not valid_r:
                continue
            if residue_strategy == 'first':
                r = valid_r[0]
            elif residue_strategy == 'last':
                r = valid_r[-1]
            elif residue_strategy == 'random':
                r = random.choice(valid_r)
            elif residue_strategy == 'middle':
                r = valid_r[len(valid_r)//2]
            else:
                r = valid_r[0]
            selected.append([m, r])
        return selected
    
    def score_of(selected):
        return sum(Fraction(1, m) for m, r in selected)
    
    best_selected = None
    best_score = Fraction(0)
    
    # Try different strategies and orders
    strategies = ['first', 'last', 'middle', 'random']
    orders = [
        moduli,                          # increasing
        sorted(moduli, key=lambda m: -m),  # decreasing
        sorted(moduli, key=lambda m: m % 7),  # some other order
    ]
    
    for order in orders:
        for strat in strategies:
            if time.time() > deadline - 10.0:
                break
            result = greedy_construct(order, strat)
            s = score_of(result)
            if s > best_score:
                best_score = s
                best_selected = result
    
    # Try random restarts with random residue choices
    if time.time() < deadline - 20.0:
        for _ in range(1000):
            if time.time() > deadline - 10.0:
                break
            result = greedy_construct(moduli, 'random')
            s = score_of(result)
            if s > best_score:
                best_score = s
                best_selected = result
    
    if best_selected is None:
        best_selected = greedy_construct(moduli, 'first')
    
    return best_selected
# EVOLVE-BLOCK-END
