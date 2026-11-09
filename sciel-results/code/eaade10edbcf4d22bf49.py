"""Rewritten program: greedy placement of moduli 20..200 with backtracking polish."""
import time
import math
from fractions import Fraction

# EVOLVE-BLOCK-START
def _compat(assign, m, r, ignore=None):
    """Check if residue r for modulus m is compatible with all placed moduli (except ignore)."""
    for m2, r2 in assign.items():
        if m2 == ignore:
            continue
        g = math.gcd(m, m2)
        if r % g == r2 % g:
            return False
    return True

def _try_place(assign, m):
    """Try to place modulus m with the first compatible residue. Returns True on success."""
    for r in range(m):
        if _compat(assign, m, r):
            assign[m] = r
            return True
    return False

def _score(assign):
    """Compute the exact rational score and return as float."""
    total = Fraction(0, 1)
    for m in assign:
        total += Fraction(1, m)
    return float(total)

def _greedy_place():
    """Greedy: iterate moduli 20..200 in order, place each if possible."""
    assign = {}
    for m in range(20, 201):
        _try_place(assign, m)
    return assign

def _greedy_place_sorted():
    """Greedy: iterate moduli in order of 1/m (i.e., ascending m) but try to maximize inclusion.
    Actually same as above since we go 20..200 ascending. Let's try a different order:
    Try to include as many small moduli as possible by going ascending."""
    assign = {}
    for m in range(20, 201):
        _try_place(assign, m)
    return assign

def _backtrack_polish(assign, time_limit=10.0):
    """Try to improve the assignment by removing and re-adding moduli in different orders.
    Use a simple local search: try swapping residues or adding missing moduli."""
    start_time = time.time()
    best_assign = dict(assign)
    best_score = _score(best_assign)
    
    # Try to add any missing moduli
    missing = [m for m in range(20, 201) if m not in assign]
    
    # Try different orders of adding missing moduli
    # First, try adding them in ascending order
    for m in missing:
        if time.time() - start_time > time_limit:
            break
        if _try_place(best_assign, m):
            s = _score(best_assign)
            if s > best_score:
                best_score = s
                assign = dict(best_assign)
    
    # Now try to improve by re-placing some moduli with different residues
    # For each modulus, try all valid residues and see if it allows adding more
    for m in sorted(assign.keys()):
        if time.time() - start_time > time_limit:
            break
        current_r = assign[m]
        # Try removing m and re-adding with different residues
        for r in range(m):
            if r == current_r:
                continue
            if _compat(assign, m, r, ignore=m):
                # Try this residue
                old_r = assign[m]
                assign[m] = r
                # Try to add missing moduli
                added = False
                for m2 in missing:
                    if time.time() - start_time > time_limit:
                        break
                    if _try_place(assign, m2):
                        added = True
                s = _score(assign)
                if s > best_score:
                    best_score = s
                    best_assign = dict(assign)
                else:
                    # Revert
                    assign[m] = old_r
                    # Remove any that were added
                    for m2 in missing:
                        if m2 in assign and m2 not in best_assign:
                            del assign[m2]
    
    return best_assign

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin
    
    # Greedy placement
    assign = _greedy_place()
    current_score = _score(assign)
    
    # Try to polish with backtracking if we have time
    elapsed = time.time() - start_time
    if elapsed < 30:
        # Try a more sophisticated search
        # Use a simple simulated annealing-like approach: try random re-placements
        import random
        random.seed(42)
        best_assign = dict(assign)
        best_score = current_score
        
        for iteration in range(5000):
            if time.time() > deadline - 5:
                break
            
            # Pick a random modulus and try a different residue
            if not best_assign:
                break
            m = random.choice(list(best_assign.keys()))
            old_r = best_assign[m]
            
            # Find all valid residues for m
            valid_rs = [r for r in range(m) if _compat(best_assign, m, r, ignore=m)]
            if len(valid_rs) <= 1:
                continue
            
            new_r = random.choice(valid_rs)
            if new_r == old_r:
                continue
            
            best_assign[m] = new_r
            s = _score(best_assign)
            
            if s > best_score:
                best_score = s
                assign = dict(best_assign)
            else:
                best_assign[m] = old_r
        
        assign = best_assign
        current_score = best_score
    
    # Final polish: try to add any remaining missing moduli
    missing = [m for m in range(20, 201) if m not in assign]
    for m in missing:
        if time.time() > deadline - 2:
            break
        _try_place(assign, m)
    
    # Convert to required format
    classes = [[m, r] for m, r in sorted(assign.items())]
    return classes
# EVOLVE-BLOCK-END
