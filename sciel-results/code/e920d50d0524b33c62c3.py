"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    base = 51975
    
    # All odd divisors of 51975 = 3^3 * 5^2 * 7 * 11, greater than 1
    # 3^a * 5^b * 7^c * 11^d where a in [0,3], b in [0,2], c in [0,1], d in [0,1]
    # and product > 1
    moduli = []
    for a in range(0, 4):
        for b in range(0, 3):
            for c in range(0, 2):
                for d in range(0, 2):
                    m = (3 ** a) * (5 ** b) * (7 ** c) * (11 ** d)
                    if m > 1:
                        moduli.append(m)
    moduli.sort()  # ascending order
    
    # Precompute residue masks for each modulus
    # mask[m][r] is a boolean array of length base indicating which residues are covered
    # But that's too much memory. Instead, we'll use a different approach.
    
    # We'll track coverage as a set of uncovered residues
    # Initially all residues are uncovered
    # We want to minimize the number of uncovered residues
    
    # Strategy: 
    # 1. Greedy initialization with multiple orders
    # 2. Local search to improve
    
    def greedy_init(order):
        """Greedy: for each modulus in given order, pick residue covering most uncovered."""
        uncovered = set(range(base))
        classes = []
        for m in order:
            best_r = 0
            best_count = -1
            for r in range(m):
                # Count how many uncovered residues are in class (m, r)
                count = sum(1 for x in range(r, base, m) if x in uncovered)
                if count > best_count:
                    best_count = count
                    best_r = r
            classes.append([m, best_r])
            # Remove covered residues
            for x in range(best_r, base, m):
                uncovered.discard(x)
        return classes, len(uncovered)
    
    # Try multiple greedy orders
    candidates = []
    
    # Ascending greedy
    classes_asc, score_asc = greedy_init(moduli)
    candidates.append((score_asc, classes_asc))
    
    # Descending greedy
    classes_desc, score_desc = greedy_init(moduli[::-1])
    candidates.append((score_desc, classes_desc))
    
    # Random order greedy (a few times)
    rng = random.Random(42)
    for _ in range(5):
        shuffled = moduli[:]
        rng.shuffle(shuffled)
        classes_r, score_r = greedy_init(shuffled)
        candidates.append((score_r, classes_r))
    
    # Select best candidate
    candidates.sort(key=lambda x: x[0])
    best_score = candidates[0][0]
    best_classes = candidates[0][1]
    
    # Local search: try to improve by changing residues
    # For each modulus, try all possible residues and see if it reduces uncovered count
    
    def get_uncovered_count(classes):
        """Compute number of uncovered residues."""
        covered = bytearray(base)
        for m, r in classes:
            for x in range(r, base, m):
                covered[x] = 1
        return base - sum(covered)
    
    def local_search(classes, max_time=100):
        """Local search to minimize uncovered count."""
        current_score = get_uncovered_count(classes)
        if current_score == 0:
            return classes, 0
        
        start_time = time.time()
        improved = True
        while improved and (time.time() - start_time) < max_time:
            improved = False
            for i, (m, r) in enumerate(classes):
                # Try all other residues for this modulus
                best_r = r
                best_score = current_score
                for r_new in range(m):
                    if r_new == r:
                        continue
                    # Temporarily change and compute score
                    old_r = classes[i][1]
                    classes[i][1] = r_new
                    new_score = get_uncovered_count(classes)
                    if new_score < best_score:
                        best_score = new_score
                        best_r = r_new
                    classes[i][1] = old_r
                if best_r != r:
                    classes[i][1] = best_r
                    current_score = best_score
                    improved = True
        return classes, current_score
    
    # Run local search on best candidate
    if time.time() < deadline - 20:
        best_classes, best_score = local_search(best_classes, max_time=100)
    
    # If we still have time, try local search on other candidates
    for score, classes in candidates[1:]:
        if time.time() >= deadline - 10:
            break
        if score >= best_score:
            continue
        # This candidate is better, but let's try to improve it
        classes2, score2 = local_search(classes, max_time=30)
        if score2 < best_score:
            best_score = score2
            best_classes = classes2
    
    # Final check
    if time.time() < deadline - 5:
        best_classes, best_score = local_search(best_classes, max_time=30)
    
    return best_classes
# EVOLVE-BLOCK-END
