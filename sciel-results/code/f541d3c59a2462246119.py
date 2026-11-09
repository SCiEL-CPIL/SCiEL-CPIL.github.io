"""Optimized disjoint residue construction: maximize sum 1/m with pairwise disjointness."""
import time
import math
import random
from fractions import Fraction

def _disjoint(m1, r1, m2, r2):
    """Check if residue classes are pairwise disjoint: r1 != r2 mod gcd(m1, m2)."""
    g = math.gcd(m1, m2)
    return (r1 % g) != (r2 % g)

def _is_valid(classes):
    """Check pairwise disjointness for a list of [m, r] pairs with distinct moduli."""
    if not classes:
        return True
    moduli = [c[0] for c in classes]
    if len(moduli) != len(set(moduli)):
        return False
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            if not _disjoint(classes[i][0], classes[i][1], classes[j][0], classes[j][1]):
                return False
    return True

def _score(classes):
    """Exact rational score, returned as float."""
    total = Fraction(0, 1)
    for m, r in classes:
        total += Fraction(1, m)
    return float(total)

def _find_valid_residues(m, selected):
    """Find all valid residues for modulus m given already selected classes."""
    valid = []
    for r in range(m):
        ok = True
        for (m2, r2) in selected:
            g = math.gcd(m, m2)
            if (r % g) == (r2 % g):
                ok = False
                break
        if ok:
            valid.append(r)
    return valid

def _local_search(classes, deadline):
    """Iterative improvement: try adding, swapping, and reassigning residues."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_moduli = set(m for m, r in classes)
        
        # Try to add a new modulus
        for m in range(20, 201):
            if time.time() > deadline:
                return classes
            if m in current_moduli:
                continue
            valid_rs = _find_valid_residues(m, classes)
            if valid_rs:
                # Try each valid residue and pick the best
                best_r = valid_rs[0]
                for r in valid_rs:
                    new_classes = classes + [[m, r]]
                    if _is_valid(new_classes):
                        best_r = r
                        break
                new_classes = classes + [[m, best_r]]
                if _is_valid(new_classes):
                    sc = _score(new_classes)
                    if sc > _score(classes):
                        classes = new_classes
                        current_moduli = set(m2 for m2, r2 in classes)
                        improved = True
                        break
        
        # Try to swap: remove one modulus, add a smaller one
        for i in range(len(classes)):
            if time.time() > deadline:
                return classes
            m_remove = classes[i][0]
            remaining = classes[:i] + classes[i+1:]
            for m_add in range(20, m_remove):
                if m_add in set(m2 for m2, r2 in remaining):
                    continue
                valid_rs = _find_valid_residues(m_add, remaining)
                if valid_rs:
                    new_classes = remaining + [[m_add, valid_rs[0]]]
                    if _is_valid(new_classes):
                        sc = _score(new_classes)
                        if sc > _score(classes):
                            classes = new_classes
                            improved = True
                            break
        
        # Try reassigning residues for existing moduli to free up space
        for i in range(len(classes)):
            if time.time() > deadline:
                return classes
            m, r = classes[i]
            others = classes[:i] + classes[i+1:]
            valid_rs = _find_valid_residues(m, others)
            if len(valid_rs) > 1:
                for new_r in valid_rs:
                    if new_r == r:
                        continue
                    new_classes = classes[:i] + [[m, new_r]] + classes[i+1:]
                    if _is_valid(new_classes):
                        # Check if this allows adding a new modulus
                        for m_add in range(20, 201):
                            if m_add in set(m2 for m2, r2 in new_classes):
                                continue
                            valid_add = _find_valid_residues(m_add, new_classes)
                            if valid_add:
                                final_classes = new_classes + [[m_add, valid_add[0]]]
                                if _is_valid(final_classes):
                                    sc = _score(final_classes)
                                    if sc > _score(classes):
                                        classes = final_classes
                                        improved = True
                                        break
                            if improved:
                                break
                    if improved:
                        break
            if improved:
                break
    
    return classes

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    best_classes = None
    best_score = 0.0
    
    # Strategy 1: Structured base-modulus packing
    # For a base g, use multiples of g with distinct residues mod g
    # This allows up to g classes with small moduli
    for g in [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30]:
        if time.time() > deadline - 5:
            break
        multiples = [m for m in range(20, 201) if m % g == 0]
        num_classes = min(g, len(multiples))
        selected_moduli = multiples[:num_classes]
        selected = [(m, i) for i, m in enumerate(selected_moduli)]
        
        # Try to add non-multiples
        for m in range(20, 201):
            if m % g == 0 or m in set(m2 for m2, r2 in selected):
                continue
            valid_rs = _find_valid_residues(m, selected)
            if valid_rs:
                selected.append((m, valid_rs[0]))
        
        classes = [[m, r] for m, r in selected]
        if _is_valid(classes):
            sc = _score(classes)
            if sc > best_score:
                best_score = sc
                best_classes = classes[:]
    
    # Strategy 2: Greedy in order of increasing modulus with different residue strategies
    for residue_strategy in range(4):
        if time.time() > deadline - 5:
            break
        selected = []
        for m in range(20, 201):
            if time.time() > deadline - 5:
                break
            valid_rs = _find_valid_residues(m, selected)
            if valid_rs:
                if residue_strategy == 0:
                    r = valid_rs[0]
                elif residue_strategy == 1:
                    r = valid_rs[-1]
                elif residue_strategy == 2:
                    r = valid_rs[len(valid_rs) // 2]
                else:
                    r = valid_rs[random.randint(0, len(valid_rs) - 1)]
                selected.append((m, r))
        
        if selected:
            classes = [[m, r] for m, r in selected]
            if _is_valid(classes):
                sc = _score(classes)
                if sc > best_score:
                    best_score = sc
                    best_classes = classes[:]
    
    # Strategy 3: Randomized multi-start with local search
    random.seed(42)
    num_restarts = 20
    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        # Random subset of moduli, then fill in
        available = list(range(20, 201))
        random.shuffle(available)
        selected = []
        for m in available:
            if time.time() > deadline - 10:
                break
            valid_rs = _find_valid_residues(m, selected)
            if valid_rs:
                r = random.choice(valid_rs)
                selected.append((m, r))
        
        if selected:
            classes = [[m, r] for m, r in selected]
            if _is_valid(classes):
                # Apply local search
                classes = _local_search(classes, deadline - 2)
                sc = _score(classes)
                if sc > best_score:
                    best_score = sc
                    best_classes = classes[:]
    
    # Strategy 4: Apply local search to the best found so far
    if best_classes and time.time() < deadline - 2:
        improved = _local_search(best_classes, deadline)
        sc = _score(improved)
        if sc > best_score:
            best_score = sc
            best_classes = improved
    
    if best_classes is None:
        # Fallback
        best_classes = [[20, 0], [40, 1], [60, 2], [80, 3], [100, 4],
                       [120, 5], [140, 6], [160, 7], [180, 8], [200, 9]]
    
    return best_classes
