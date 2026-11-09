import time
from math import gcd
from fractions import Fraction

def _try_add_class(classes, new_class):
    """Check if a new class can be added to the existing set while maintaining disjointness."""
    m_new, r_new = new_class
    for m, r in classes:
        g = gcd(m_new, m)
        if g == 1:
            continue  # coprime moduli: residues mod 1 are always equal, constraint is vacuous
        if r_new % g == r % g:
            return False
    return True

def _score(classes):
    """Compute the exact rational score (sum of 1/m)."""
    total = Fraction(0, 1)
    for m, r in classes:
        total += Fraction(1, m)
    return total

def _score_float(classes):
    """Compute float score for fast comparison."""
    return sum(1.0 / m for m, r in classes)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    all_moduli = list(range(20, 201))
    
    def greedy_construct(moduli_list):
        """Greedy: add moduli in given order, pick first valid residue."""
        classes = []
        for m in moduli_list:
            found = False
            for r in range(m):
                if _try_add_class(classes, [m, r]):
                    classes.append([m, r])
                    found = True
                    break
        return classes
    
    # Strategy 1: Greedy with smallest moduli first
    best_classes = greedy_construct(all_moduli)
    best_score_f = _score_float(best_classes)
    
    # Strategy 2: Focus on even moduli (all share factor 2)
    even_moduli = [m for m in all_moduli if m % 2 == 0]
    classes_even = greedy_construct(even_moduli)
    score_even = _score_float(classes_even)
    if score_even > best_score_f:
        best_classes = classes_even
        best_score_f = score_even
    
    # Strategy 3: Multiples of 3
    mult3 = [m for m in all_moduli if m % 3 == 0]
    classes_m3 = greedy_construct(mult3)
    score_m3 = _score_float(classes_m3)
    if score_m3 > best_score_f:
        best_classes = classes_m3
        best_score_f = score_m3
    
    # Strategy 4: Multiples of 6
    mult6 = [m for m in all_moduli if m % 6 == 0]
    classes_m6 = greedy_construct(mult6)
    score_m6 = _score_float(classes_m6)
    if score_m6 > best_score_f:
        best_classes = classes_m6
        best_score_f = score_m6
    
    # Local search improvement
    def try_improve(classes, max_iters=500):
        current = [c[:] for c in classes]
        current_score_f = _score_float(current)
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            if time.time() > deadline - 5:
                break
            
            # Try adding a new modulus
            for m in all_moduli:
                if any(c[0] == m for c in current):
                    continue
                added = False
                for r in range(m):
                    if _try_add_class(current, [m, r]):
                        current = current + [[m, r]]
                        current_score_f += 1.0/m
                        improved = True
                        added = True
                        break
                if added:
                    break
            
            # Try replacing a class with a smaller modulus
            for i in range(len(current)):
                m_old = current[i][0]
                replaced = False
                for m_new in all_moduli:
                    if m_new >= m_old:
                        continue
                    if any(c[0] == m_new for c in current if c[0] != m_old):
                        continue
                    others = [c for c in current if c[0] != m_old]
                    for r_new in range(m_new):
                        if _try_add_class(others, [m_new, r_new]):
                            new_score_f = current_score_f - 1.0/m_old + 1.0/m_new
                            if new_score_f > current_score_f:
                                current = others + [[m_new, r_new]]
                                current_score_f = new_score_f
                                improved = True
                                replaced = True
                                break
                    if replaced:
                        break
                if replaced:
                    break
            
            # Try swap: replace one class with another of different size
            for i in range(len(current)):
                m_old = current[i][0]
                swapped = False
                for m_new in all_moduli:
                    if m_new == m_old:
                        continue
                    if any(c[0] == m_new for c in current if c[0] != m_old):
                        continue
                    others = [c for c in current if c[0] != m_old]
                    for r_new in range(m_new):
                        if _try_add_class(others, [m_new, r_new]):
                            new_score_f = current_score_f - 1.0/m_old + 1.0/m_new
                            if new_score_f > current_score_f:
                                current = others + [[m_new, r_new]]
                                current_score_f = new_score_f
                                improved = True
                                swapped = True
                                break
                    if swapped:
                        break
                if swapped:
                    break
                    
        return current, current_score_f
    
    best_classes, best_score_f = try_improve(best_classes, max_iters=500)
    
    # Try improving even-based solution
    if time.time() < deadline - 10:
        classes_even2, score_even2 = try_improve(classes_even, max_iters=300)
        if score_even2 > best_score_f:
            best_classes = classes_even2
            best_score_f = score_even2
    
    # Try improving mult3-based solution
    if time.time() < deadline - 10:
        classes_m32, score_m32 = try_improve(classes_m3, max_iters=300)
        if score_m32 > best_score_f:
            best_classes = classes_m32
            best_score_f = score_m32
    
    return [[m, r] for m, r in best_classes]
