"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd, log
from fractions import Fraction

def _score_fraction(chosen):
    """Compute exact rational score sum 1/(512-a) for a list of chosen integers."""
    if not chosen:
        return Fraction(0, 1)
    s = Fraction(0, 1)
    for a in chosen:
        s += Fraction(1, 512 - a)
    return s

def _score_float(chosen):
    """Compute float score for quick comparison."""
    return sum(1.0 / (512.0 - a) for a in chosen)

def _is_coprime_to_all(a, chosen):
    """Check if a is coprime to all elements in chosen."""
    for b in chosen:
        if gcd(a, b) != 1:
            return False
    return True

def _greedy_construct(deadline):
    """Greedy construction: pick largest a coprime to all previously chosen."""
    chosen = []
    for a in range(511, 0, -1):
        if time.time() > deadline:
            break
        if _is_coprime_to_all(a, chosen):
            chosen.append(a)
    return chosen

def _hill_climb(chosen, deadline, max_iterations=500000):
    """Hill-climbing: try swaps and additions to improve score."""
    best_score = _score_float(chosen)
    best_chosen = list(chosen)
    
    # Precompute values for quick score updates
    def _try_add(a, current):
        """Try adding a to current set."""
        if a in current:
            return None
        if not _is_coprime_to_all(a, current):
            return None
        return current + [a]
    
    def _try_swap(a_out, a_in, current):
        """Try removing a_out and adding a_in."""
        if a_in in current:
            return None
        new_set = [x for x in current if x != a_out]
        if not _is_coprime_to_all(a_in, new_set):
            return None
        return new_set + [a_in]
    
    iteration = 0
    while time.time() < deadline and iteration < max_iterations:
        iteration += 1
        improved = False
        
        # Try adding elements
        for a in range(511, 0, -1):
            if a in best_chosen:
                continue
            candidate = _try_add(a, best_chosen)
            if candidate is not None:
                new_score = _score_float(candidate)
                if new_score > best_score + 1e-15:
                    best_score = new_score
                    best_chosen = candidate
                    improved = True
                    break
        
        if improved:
            continue
        
        # Try swaps: remove one, add one
        for i, a_out in enumerate(best_chosen):
            for a_in in range(511, 0, -1):
                if a_in == a_out or a_in in best_chosen:
                    continue
                candidate = _try_swap(a_out, a_in, best_chosen)
                if candidate is not None:
                    new_score = _score_float(candidate)
                    if new_score > best_score + 1e-15:
                        best_score = new_score
                        best_chosen = candidate
                        improved = True
                        break
            if improved:
                break
        
        if not improved:
            # Try 2-for-1 swaps: remove two, add two
            found = False
            for i in range(len(best_chosen)):
                for j in range(i + 1, len(best_chosen)):
                    a_out1 = best_chosen[i]
                    a_out2 = best_chosen[j]
                    base = [x for x in best_chosen if x != a_out1 and x != a_out2]
                    for a_in1 in range(511, 0, -1):
                        if a_in1 in base or a_in1 == a_out1 or a_in1 == a_out2:
                            continue
                        if not _is_coprime_to_all(a_in1, base):
                            continue
                        for a_in2 in range(511, 0, -1):
                            if a_in2 in base or a_in2 == a_out1 or a_in2 == a_out2 or a_in2 == a_in1:
                                continue
                            if not _is_coprime_to_all(a_in2, base + [a_in1]):
                                continue
                            candidate = base + [a_in1, a_in2]
                            new_score = _score_float(candidate)
                            if new_score > best_score + 1e-15:
                                best_score = new_score
                                best_chosen = candidate
                                found = True
                                break
                        if found:
                            break
                    if found:
                        break
                if found:
                    break
            
            if not found:
                break
    
    return best_chosen

def construct():
    start_time = time.time()
    # Use 140s as deadline to stay well within 150s limit
    deadline = start_time + 140.0
    
    # Phase 1: Greedy construction
    chosen = _greedy_construct(deadline)
    
    if time.time() > deadline:
        return chosen
    
    # Phase 2: Hill climbing
    chosen = _hill_climb(chosen, deadline)
    
    return chosen
# EVOLVE-BLOCK-END
