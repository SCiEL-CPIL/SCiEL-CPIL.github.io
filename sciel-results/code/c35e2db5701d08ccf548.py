import time
import math
from fractions import Fraction

def _compatible(assign, m, r, ignore=None):
    """Check if (m, r) is compatible with all in assign (except ignore)."""
    for m2, r2 in assign.items():
        if m2 == ignore:
            continue
        g = math.gcd(m, m2)
        if r % g == r2 % g:
            return False
    return True

def _try_place(assign, m, deadline=None):
    """Try to place modulus m with first compatible residue."""
    for r in range(m):
        if _compatible(assign, m, r):
            assign[m] = r
            return True
    return False

def _score(assign):
    """Compute sum of 1/m for all moduli in assign."""
    return sum(1.0 / m for m in assign)

def _greedy_construct(deadline):
    """Greedy: add moduli from smallest to largest."""
    assign = {}
    for m in range(20, 201):
        if time.time() > deadline:
            break
        _try_place(assign, m)
    return assign

def _local_search(assign, deadline):
    """Add-remove-swap local search to improve the solution."""
    best_assign = dict(assign)
    best_score = _score(best_assign)
    
    while time.time() < deadline:
        improved = False
        
        # Try adding a new modulus
        for m in range(20, 201):
            if m in best_assign:
                continue
            if _try_place(dict(best_assign), m):
                new_assign = dict(best_assign)
                _try_place(new_assign, m)
                new_score = _score(new_assign)
                if new_score > best_score:
                    best_assign = new_assign
                    best_score = new_score
                    improved = True
                    break
            if time.time() > deadline:
                break
        
        if improved:
            continue
        
        # Try swap: remove one modulus, add another
        for m_remove in list(best_assign.keys()):
            if time.time() > deadline:
                break
            for m_add in range(20, 201):
                if m_add in best_assign or m_add == m_remove:
                    continue
                # Try removing m_remove and adding m_add
                test_assign = dict(best_assign)
                del test_assign[m_remove]
                if _try_place(test_assign, m_add):
                    new_score = _score(test_assign)
                    if new_score > best_score:
                        best_assign = test_assign
                        best_score = new_score
                        improved = True
                        break
            if improved or time.time() > deadline:
                break
        
        if not improved:
            break
    
    return best_assign

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Phase 1: Greedy construction
    assign = _greedy_construct(deadline)
    
    # Phase 2: Local search improvement
    if time.time() < deadline:
        assign = _local_search(assign, deadline)
    
    # Convert to required format
    result = [[m, r] for m, r in sorted(assign.items())]
    return result
