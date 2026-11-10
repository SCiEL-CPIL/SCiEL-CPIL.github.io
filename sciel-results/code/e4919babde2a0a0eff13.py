import time
from math import gcd

def _conflict(m1, r1, m2, r2):
    g = gcd(m1, m2)
    return (r1 - r2) % g == 0

def _compatible(m, r, selected):
    for (sm, sr) in selected:
        if _conflict(m, r, sm, sr):
            return False
    return True

def _find_valid_residue(m, selected):
    """Find a residue r for m that is compatible with selected."""
    # Try all residues
    for r in range(m):
        if _compatible(m, r, selected):
            return r
    return None

def _score(pairs):
    return sum(1.0 / m for (m, r) in pairs)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    # We want to include as many small m as possible.
    # Ideally all m in [20, 200].
    
    # Greedy approach: Add m in increasing order.
    # If a conflict arises, we might need to backtrack or swap.
    # Given the density, simple greedy often works.
    
    selected = []
    used_moduli = set()
    
    # Try to include all
    for m in range(20, 201):
        if time.time() > deadline:
            break
        r = _find_valid_residue(m, selected)
        if r is not None:
            selected.append((m, r))
            used_moduli.add(m)
        else:
            # If no residue found, we skip this m for now.
            # We might try to fix this later by reassigning residues of existing m's.
            pass

    best = selected[:]
    best_val = _score(best)
    
    # Local Search: Try to improve by swapping or reassigning
    # Since we want max sum, and 1/m is decreasing, 
    # the most valuable improvement is adding a small m that was skipped,
    # or replacing a large m with a smaller one.
    
    # However, if we included all, we are at the theoretical max.
    # Let's check if we included all.
    if len(selected) == 181:
        return [[m, r] for (m, r) in best]

    # If not all included, try to add skipped ones or swap.
    # This is complex. A simpler heuristic:
    # Randomized restarts with different residue selection orders.
    
    best_result = best
    best_score = best_val
    
    for _ in range(10): # 10 restarts
        if time.time() > deadline - 10:
            break
            
        # Randomize the order of residue selection slightly
        # Or just try a different greedy strategy: 
        # Pick residues that are "spread out".
        
        sel = []
        used = set()
        
        # Process m in increasing order
        for m in range(20, 201):
            # Try to find a residue that minimizes conflicts with future m's?
            # Hard to know future. 
            # Just pick the first valid one.
            r = _find_valid_residue(m, sel)
            if r is not None:
                sel.append((m, r))
                used.add(m)
        
        val = _score(sel)
        if val > best_score:
            best_score = val
            best_result = sel[:]

    return [[m, r] for (m, r) in best_result]
