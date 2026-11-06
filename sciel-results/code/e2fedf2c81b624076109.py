import math
import random
import time

MAX_MOD = 128

def _disjoint(r1, m1, r2, m2):
    g = math.gcd(m1, m2)
    return (r2 - r1) % g != 0

def _valid(pairs):
    n = len(pairs)
    for i in range(n):
        for j in range(i + 1, n):
            if not _disjoint(pairs[i][0], pairs[i][1], pairs[j][0], pairs[j][1]):
                return False
    return True

def _try_assign(moduli, time_limit):
    """
    Try to find residues for the given moduli.
    Returns list of [r, m] or None if failed.
    """
    n = len(moduli)
    # Sort moduli to put "constrained" ones first? 
    # High moduli have more residues, so maybe put them last?
    # Or put those with many divisors first?
    # Let's just sort by value for determinism in restarts, but randomize residue order.
    mods = sorted(moduli)
    
    for _ in range(100): # 100 restarts
        if time.time() > time_limit:
            break
        pairs = []
        success = True
        # Randomize the order of moduli slightly? No, keep fixed order for greedy, but randomize residue trial order.
        for m in mods:
            found = False
            residues = list(range(m))
            random.shuffle(residues)
            for r in residues:
                ok = True
                for (r2, m2) in pairs:
                    if not _disjoint(r, m, r2, m2):
                        ok = False
                        break
                if ok:
                    pairs.append([r, m])
                    found = True
                    break
            if not found:
                success = False
                break
        if success:
            return pairs
    return None

def construct():
    t0 = time.time()
    deadline = t0 + 55.0  # 5s margin
    
    # Start with all even moduli
    all_mods = list(range(2, MAX_MOD + 1))
    even_mods = [m for m in all_mods if m % 2 == 0]
    
    best = []
    
    # Try to find a solution for the full set of even moduli
    # If it fails, try removing one at a time
    current_mods = even_mods[:]
    
    # We can also try other sets if even fails completely, but even is likely to work.
    # To be safe, we can try a greedy build from scratch as well.
    
    # Strategy:
    # 1. Try to assign residues to `current_mods`.
    # 2. If success, return.
    # 3. If fail, remove the last modulus (or random one) and retry.
    
    # Actually, the greedy assignment might fail due to bad ordering.
    # Let's try multiple random orderings of the moduli in the greedy step.
    
    best_pairs = None
    
    # Try with the full set of even moduli
    # We'll do a loop that tries to solve the current set.
    # If it fails, we shrink the set.
    
    # To maximize the chance, we can also try to add moduli back if we removed them?
    # No, just shrink.
    
    # Initial set: all evens
    mods_to_try = even_mods[:]
    
    # We will try to find a valid assignment for mods_to_try.
    # We use a time-boxed search.
    
    while mods_to_try and time.time() < deadline:
        # Try to find an assignment for mods_to_try
        # Use multiple random restarts inside _try_assign
        res = _try_assign(mods_to_try, deadline)
        if res:
            best_pairs = res
            break
        else:
            # Failed to find assignment for this set size
            # Remove one modulus
            # Which one? The one that is most "constraining"?
            # Or just the last one?
            # Let's remove the largest one to keep small moduli which are more restrictive?
            # Actually, small moduli have fewer residues, so they are harder to satisfy?
            # No, small moduli have fewer choices for r, so they are more likely to fail?
            # If m=2, r can be 0 or 1.
            # If m=128, r can be 0..127.
            # So small moduli are more restrictive.
            # If we fail, it's likely due to small moduli conflicting.
            # So maybe we should remove a large modulus?
            # Or remove a random one?
            # Let's remove the largest one first.
            mods_to_try.pop()
            
    if best_pairs is None:
        # Fallback: greedy build from scratch
        # Try to build a set greedily
        best = []
        for m in all_mods:
            # Try to add m to best
            found = False
            for r in range(m):
                ok = True
                for (r2, m2) in best:
                    if not _disjoint(r, m, r2, m2):
                        ok = False
                        break
                if ok:
                    best.append([r, m])
                    found = True
                    break
            if time.time() > deadline:
                break
    else:
        best = best_pairs

    return best
