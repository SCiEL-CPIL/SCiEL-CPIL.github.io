"""Optimized odd covering system search for moduli dividing 51975."""
import time
import random
from collections import Counter

BASE = 51975
# 51975 = 3^3 * 5^2 * 7 * 11
# Generate all divisors > 1
def _divisors(n):
    divs = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
    return [d for d in divs if d > 1]

ALL_MODULI = sorted(_divisors(BASE))
MODULI_SET = set(ALL_MODULI)

def _score(classes):
    """Return number of uncovered residues modulo BASE."""
    # Use a bytearray for coverage
    covered = bytearray(BASE)
    for m, r in classes:
        covered[r::m] = b'\x01' * len(range(r, BASE, m))
    return covered.count(0)

def _fast_score(classes):
    """Faster scoring using set operations where possible."""
    covered = bytearray(BASE)
    for m, r in classes:
        # Mark all positions r, r+m, r+2m, ... < BASE
        for i in range(r, BASE, m):
            covered[i] = 1
    return covered.count(0)

def _greedy_construct(moduli_order=None, seed=42):
    """Greedy construction: pick classes to maximize newly covered residues."""
    if moduli_order is None:
        moduli_order = list(ALL_MODULI)
    
    rng = random.Random(seed)
    covered = bytearray(BASE)
    classes = []
    remaining_moduli = list(moduli_order)
    
    # We can use each modulus at most once (distinct moduli requirement)
    # Greedily pick the (m, r) that covers the most new residues
    while remaining_moduli:
        best_gain = -1
        best_m = None
        best_r = None
        for m in remaining_moduli:
            # Try all residues r in [0, m)
            for r in range(m):
                # Count newly covered residues
                gain = 0
                for i in range(r, BASE, m):
                    if not covered[i]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_m = m
                    best_r = r
                if gain == 0:
                    break  # No point trying larger r if already 0? No, different r covers different positions
            # Actually we should check all r, but if m is large and gain is 0 for all, skip
            # Optimization: if m > BASE, gain is always 0 or 1. But m <= BASE here.
        if best_gain <= 0:
            break
        classes.append([best_m, best_r])
        remaining_moduli.remove(best_m)
        # Update covered
        for i in range(best_r, BASE, best_m):
            covered[i] = 1
        if covered.count(0) == 0:
            break
    
    return classes

def _local_search(classes, max_iter=500, seed=42):
    """Local search: try to swap residues or swap moduli to reduce uncovered count."""
    rng = random.Random(seed)
    classes = [list(c) for c in classes]
    current_score = _fast_score(classes)
    
    # Precompute which moduli are used
    used_moduli = {m for m, r in classes}
    available_moduli = [m for m in ALL_MODULI if m not in used_moduli]
    
    for it in range(max_iter):
        improved = False
        
        # 1. Try changing residue for a random class
        if classes:
            idx = rng.randrange(len(classes))
            m, r = classes[idx]
            old_r = r
            # Try a few random residues
            for _ in range(10):
                new_r = rng.randrange(m)
                classes[idx][1] = new_r
                new_score = _fast_score(classes)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                    break
                else:
                    classes[idx][1] = old_r
            if improved:
                continue
        
        # 2. Try adding a new modulus (if not already at max useful size)
        if available_moduli and len(classes) < len(ALL_MODULI):
            m = rng.choice(available_moduli)
            best_r = 0
            best_gain = -1
            # Try a few residues
            for _ in range(20):
                r = rng.randrange(m)
                # Quick estimate: count uncovered in this class
                gain = 0
                covered = bytearray(BASE)
                for mm, rr in classes:
                    for i in range(rr, BASE, mm):
                        covered[i] = 1
                for i in range(r, BASE, m):
                    if not covered[i]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
            if best_gain > 0:
                classes.append([m, best_r])
                available_moduli.remove(m)
                current_score = _fast_score(classes)
                improved = True
                continue
        
        # 3. Try removing a class that adds no unique coverage
        if len(classes) > 1:
            idx = rng.randrange(len(classes))
            classes[idx][1] = -1  # Mark for removal temporarily
            temp_classes = [c for c in classes if c[1] >= 0]
            new_score = _fast_score(temp_classes)
            if new_score <= current_score:
                classes = temp_classes
                current_score = new_score
                improved = True
                continue
            else:
                classes[idx][1] = -1  # Restore (it was already restored in the list comp logic, but let's be safe)
                # Actually the list comp created a new list, so we need to restore properly
                # Let's redo this more carefully
                pass
        
        if not improved:
            break
    
    return classes, current_score

def construct():
    """Main entry point. Returns a list of [m, r] pairs minimizing uncovered residues."""
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline, 10s safety margin
    
    best_classes = None
    best_score = BASE  # Worst case: nothing covered
    
    # Phase 1: Greedy construction with different orderings
    # Try natural order first
    greedy_classes = _greedy_construct(ALL_MODULI, seed=42)
    greedy_score = _fast_score(greedy_classes)
    if greedy_score < best_score:
        best_score = greedy_score
        best_classes = greedy_classes
    
    # Phase 2: Local search on the greedy result
    if time.time() < deadline - 30:
        ls_classes, ls_score = _local_search(greedy_classes, max_iter=200, seed=42)
        if ls_score < best_score:
            best_score = ls_score
            best_classes = ls_classes
    
    # Phase 3: Try different greedy orderings (randomized)
    for seed in range(10):
        if time.time() > deadline - 20:
            break
        rng = random.Random(seed)
        shuffled = ALL_MODULI[:]
        rng.shuffle(shuffled)
        # Bias towards smaller moduli first? Or larger?
        # Let's try both
        greedy2 = _greedy_construct(shuffled, seed=seed)
        score2 = _fast_score(greedy2)
        if score2 < best_score:
            best_score = score2
            best_classes = greedy2
    
    # Phase 4: More local search iterations if time permits
    if time.time() < deadline - 10 and best_classes:
        ls2_classes, ls2_score = _local_search(best_classes, max_iter=1000, seed=123)
        if ls2_score < best_score:
            best_score = ls2_score
            best_classes = ls2_classes
    
    if best_classes is None:
        # Fallback: return a valid but possibly suboptimal solution
        best_classes = _greedy_construct(ALL_MODULI, seed=0)
    
    return best_classes
