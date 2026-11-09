import time
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    # Precompute gcd table for moduli 20..200
    moduli = list(range(20, 201))
    n_mod = len(moduli)
    
    # gcd[m1_idx][m2_idx]
    gcd_table = [[0] * n_mod for _ in range(n_mod)]
    for i in range(n_mod):
        for j in range(i, n_mod):
            g = gcd(moduli[i], moduli[j])
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    
    # For each pair of moduli (i, j), precompute which residue pairs conflict
    # conflict[i][j] = set of r_i values that conflict with some r_j
    # Actually, for a given r_i, the conflicting r_j values are those where r_j % gcd == r_i % gcd
    
    def can_add(classes, m_idx, r):
        """Check if adding (m_idx, r) to classes keeps them disjoint."""
        for mi, ri in classes:
            g = gcd_table[mi][m_idx]
            if ri % g == r % g:
                return False
        return True
    
    def score(classes):
        """Sum of 1/m for the classes."""
        total = Fraction(0, 1)
        for m_idx, r in classes:
            total += Fraction(1, moduli[m_idx])
        return total
    
    # Greedy construction: try to add moduli in increasing order (smaller m = larger 1/m)
    def greedy_construct():
        classes = []
        for m_idx in range(n_mod):
            m = moduli[m_idx]
            found = False
            for r in range(m):
                if can_add(classes, m_idx, r):
                    classes.append((m_idx, r))
                    found = True
                    break
        return classes
    
    # Branch and bound: process moduli in order, try all residues
    # Upper bound: current score + sum of 1/m for all remaining moduli
    # This is a weak bound, so we need smarter pruning
    
    # Better: use the fact that for each pair of moduli, some residues are forbidden
    # Precompute for each modulus m_idx, given the set of already-chosen (mi, ri),
    # which residues are still valid
    
    # Let's use a different approach: local search with add-remove-swap
    def local_search(classes, max_iter=500000):
        classes = list(classes)
        current_score = score(classes)
        improved = True
        iter_count = 0
        while improved and iter_count < max_iter and time.time() < deadline:
            improved = False
            iter_count += 1
            # Try to add a missing modulus
            for m_idx in range(n_mod):
                if m_idx in [c[0] for c in classes]:
                    continue
                m = moduli[m_idx]
                for r in range(m):
                    if can_add(classes, m_idx, r):
                        new_score = current_score + Fraction(1, moduli[m_idx])
                        if new_score > current_score:
                            classes.append((m_idx, r))
                            current_score = new_score
                            improved = True
                            break
                if time.time() > deadline - 5.0:
                    break
            if time.time() > deadline - 5.0:
                break
            
            # Add-Remove Swap: try replacing a class with a different one
            for i, (mi, ri) in enumerate(classes):
                m_old = moduli[mi]
                for m_idx in range(n_mod):
                    if m_idx == mi:
                        continue
                    if m_idx in [c[0] for c in classes if c[0] != mi]:
                        continue
                    m_new = moduli[m_idx]
                    # Only consider if this improves score
                    if m_new >= m_old:
                        continue
                    for r in range(m_new):
                        # Check if replacing classes[i] with (m_idx, r) keeps disjointness
                        test_classes = classes[:i] + [(m_idx, r)] + classes[i+1:]
                        valid = True
                        for j, (mj, rj) in enumerate(test_classes):
                            if j == i:
                                continue
                            g = gcd_table[m_idx][mj]
                            if r % g == rj % g:
                                valid = False
                                break
                        if valid:
                            new_score = current_score - Fraction(1, m_old) + Fraction(1, m_new)
                            if new_score > current_score:
                                classes[i] = (m_idx, r)
                                current_score = new_score
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
        return classes, current_score
    
    # Build initial solution greedily
    best_classes = greedy_construct()
    best_score = score(best_classes)
    
    # Local search to improve
    if time.time() < deadline - 10.0:
        improved_classes, improved_score = local_search(best_classes)
        if improved_score > best_score:
            best_classes = improved_classes
            best_score = improved_score
    
    # Convert back to [[m, r], ...] format
    result = [[moduli[m_idx], r] for m_idx, r in best_classes]
    
    # Verify validity
    for i in range(len(best_classes)):
        mi, ri = best_classes[i]
        for j in range(i + 1, len(best_classes)):
            mj, rj = best_classes[j]
            g = gcd_table[mi][mj]
            assert ri % g != rj % g, f"Conflict between {mi} and {mj}"
    assert len(set(moduli[m_idx] for m_idx, r in best_classes)) == len(best_classes)
    
    return result

if __name__ == "__main__":
    result = construct()
    print(f"Number of classes: {len(result)}")
    print(f"Score: {sum(1/m for m, r in result)}")
