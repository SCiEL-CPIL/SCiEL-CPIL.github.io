import math
import sys

def construct():
    # We want to maximize sum of 1/m for pairwise disjoint residue classes
    # Process moduli in ascending order (to prioritize small m with large 1/m)
    # Use backtracking with the constraint that residues must be distinct mod gcd
    
    moduli = list(range(20, 201))
    
    # Precompute gcd matrix for efficiency
    n = len(moduli)
    gcds = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            gcds[i][j] = math.gcd(moduli[i], moduli[j])
    
    best_score = 0
    best_classes = []
    
    # Backtracking: try to include each modulus in order
    # For each modulus, try all valid residues
    # Use memoization-like pruning: if current sum + max possible remaining < best, prune
    
    # Precompute suffix sums of 1/m for upper bound
    suffix = [0.0] * (n + 1)
    for i in range(n-1, -1, -1):
        suffix[i] = suffix[i+1] + 1.0 / moduli[i]
    
    def backtrack(idx, classes, current_score):
        nonlocal best_score, best_classes
        
        # Prune: even if we add all remaining, can we beat best?
        if current_score + suffix[idx] <= best_score:
            return
        
        if idx == n:
            if current_score > best_score:
                best_score = current_score
                best_classes = [c[:] for c in classes]
            return
        
        m = moduli[idx]
        
        # Try all possible residues for this modulus
        found_valid = False
        for r in range(m):
            valid = True
            for j, (m_j, r_j) in enumerate(classes):
                j_idx = idx - 1 - j  # index in moduli list
                g = gcds[j_idx][idx]
                if (r - r_j) % g == 0:
                    valid = False
                    break
            if valid:
                found_valid = True
                classes.append([m, r])
                backtrack(idx + 1, classes, current_score + 1.0 / m)
                classes.pop()
        
        # Also try skipping this modulus
        backtrack(idx + 1, classes, current_score)
    
    backtrack(0, [], 0.0)
    
    if best_classes:
        return best_classes
    
    # Fallback: greedy
    classes = []
    for m in range(20, 201):
        for r in range(m):
            valid = True
            for m_j, r_j in classes:
                g = math.gcd(m, m_j)
                if (r - r_j) % g == 0:
                    valid = False
                    break
            if valid:
                classes.append([m, r])
                break
    return classes
