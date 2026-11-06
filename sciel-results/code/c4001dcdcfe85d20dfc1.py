import math
import time
import random
from math import gcd

_START = time.time()
_DEADLINE = 145.0  # Safety margin under 150s

def _time_left():
    return _DEADLINE - (time.time() - _START)

def _pair_ok(m1, r1, m2, r2):
    g = gcd(m1, m2)
    return (r1 % g) != (r2 % g)

def _is_valid(classes):
    for i in range(len(classes)):
        m1, r1 = classes[i]
        for j in range(i + 1, len(classes)):
            m2, r2 = classes[j]
            if not _pair_ok(m1, r1, m2, r2):
                return False
    return True

def _score(classes):
    return sum(1.0 / m for m, _ in classes)

def _find_residue(mod_to_res, m):
    """
    Find a residue r for modulus m that is disjoint from all existing moduli in mod_to_res.
    Returns r if found, else None.
    """
    for r in range(m):
        ok = True
        for mi, ri in mod_to_res.items():
            g = gcd(m, mi)
            if (r % g) == (ri % g):
                ok = False
                break
        if ok:
            return r
    return None

def _find_repair_residue(mod_to_res, m_i, r_new_target, m_new):
    """
    Try to find a new residue r_i' for modulus m_i such that:
    1. It is compatible with the target residue r_new_target for modulus m_new.
    2. It is compatible with all other moduli in mod_to_res (excluding m_i).
    
    Returns r_i' if found, else None.
    """
    g_target = gcd(m_i, m_new)
    # Condition 1: r_i' % g_target != r_new_target % g_target
    forbidden_target_mod = r_new_target % g_target
    
    # Pre-compute constraints from other moduli
    other_mods = [(mi, ri) for mi, ri in mod_to_res.items() if mi != m_i]
    
    for r_i in range(m_i):
        # Check condition 1
        if r_i % g_target == forbidden_target_mod:
            continue
        
        # Check condition 2
        ok = True
        for mi, ri in other_mods:
            g = gcd(m_i, mi)
            if (r_i % g) == (ri % g):
                ok = False
                break
        
        if ok:
            return r_i
    return None

def _local_search(mod_to_res, rng):
    """
    Iteratively try to add missing moduli.
    Returns the updated mod_to_res.
    """
    t0 = time.time()
    used = set(mod_to_res.keys())
    
    # List of moduli to try adding, sorted by 1/m (descending) i.e. m ascending
    # But we should prioritize those that are "easy" to add or unblock.
    # Simple strategy: iterate through all missing moduli repeatedly.
    
    improved = True
    while improved and _time_left() > 5:
        improved = False
        # Shuffle to avoid bias? Or keep order?
        # Keeping order 20..200 is good because small m has high weight.
        missing = [m for m in range(20, 201) if m not in used]
        
        for m in missing:
            if _time_left() < 5:
                break
            
            # 1. Try direct addition
            r = _find_residue(mod_to_res, m)
            if r is not None:
                mod_to_res[m] = r
                used.add(m)
                improved = True
                # Since we added one, the set of missing changed, 
                # but continuing the loop is fine as we check 'if m not in used' 
                # effectively by re-evaluating or just breaking to restart?
                # Restarting is safer to pick up new opportunities from smaller m?
                # Actually, adding a larger m doesn't block smaller m's from being added 
                # if they were already missing. But it might block other large m's.
                # We can just continue.
                continue

            # 2. Try repair: change one existing modulus to make room for m
            # We need to find a candidate r for m that is blocked by only a few moduli.
            # Or just pick a random r for m and see who blocks it.
            
            # To be efficient, let's find the "most compatible" r for m.
            # Score each r by number of conflicts.
            best_r = None
            min_conflicts = float('inf')
            best_blockers = None
            
            for r in range(m):
                blockers = []
                for mi, ri in mod_to_res.items():
                    g = gcd(m, mi)
                    if (r % g) == (ri % g):
                        blockers.append(mi)
                
                if len(blockers) < min_conflicts:
                    min_conflicts = len(blockers)
                    best_r = r
                    best_blockers = blockers
                    if min_conflicts == 0:
                        # This shouldn't happen if direct addition failed, but safety
                        break
            
            if best_r is None:
                continue
                
            # If min_conflicts is 1, we can try to repair that single blocker.
            if min_conflicts == 1:
                m_blocker = best_blockers[0]
                r_blocker_old = mod_to_res[m_blocker]
                
                # Try to find a new residue for m_blocker
                r_blocker_new = _find_repair_residue(mod_to_res, m_blocker, best_r, m)
                if r_blocker_new is not None:
                    mod_to_res[m_blocker] = r_blocker_new
                    mod_to_res[m] = best_r
                    used.add(m)
                    improved = True
                    # Restart loop to see if more can be added
                    break 
            # If min_conflicts > 1, we could try 2-variable repair, but it's more complex.
            # For now, skip or try random swaps?
            # Let's try a simple random swap if stuck.
            elif min_conflicts == 2 and _time_left() > 10:
                # Try to swap two blockers?
                # This is getting complex. Let's stick to 1-blocker repair for now.
                # Maybe try random perturbation of existing residues?
                pass

    return mod_to_res

def construct():
    rng = random.Random(42)
    best = None
    best_score = -1.0
    
    def consider(mod_to_res):
        nonlocal best, best_score
        if not mod_to_res:
            return
        # Convert to list for validation and scoring
        classes = [[m, r] for m, r in mod_to_res.items()]
        if _is_valid(classes):
            s = _score(classes)
            if s > best_score:
                best_score = s
                best = classes

    # Strategy 1: Greedy + Local Search
    mod_to_res = {}
    # Initial greedy fill
    for m in range(20, 201):
        r = _find_residue(mod_to_res, m)
        if r is not None:
            mod_to_res[m] = r
            
    consider(mod_to_res)
    
    # Local search to improve
    mod_to_res = _local_search(mod_to_res, rng)
    consider(mod_to_res)
    
    # Strategy 2: Multiple restarts with different random seeds or orders?
    # The greedy order 20->200 is very strong.
    # Let's try a few random restarts of the local search phase with slight perturbations.
    
    # Time budget check
    if _time_left() > 10:
        # Try a few variations
        for _ in range(5):
            if _time_left() < 5:
                break
            # Start with the same greedy base
            mtr = dict(mod_to_res)
            # Randomly remove a few pairs to allow re-filling?
            # Or just run local search again? It might converge to same state.
            # Let's try shuffling the order of missing moduli in local search?
            # The current _local_search uses fixed order.
            
            # Alternative: Randomized Greedy
            mtr2 = {}
            mods = list(range(20, 201))
            rng.shuffle(mods)
            for m in mods:
                r = _find_residue(mtr2, m)
                if r is not None:
                    mtr2[m] = r
            consider(mtr2)
            mtr2 = _local_search(mtr2, rng)
            consider(mtr2)

    # Fallback
    if best is None:
        best = [[20, 0]]
        
    return best
