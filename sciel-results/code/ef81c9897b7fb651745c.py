import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 135
    
    base = 51975
    
    # Get all divisors of 51975 greater than 1
    divisors = []
    for d in range(3, base + 1, 2):
        if base % d == 0:
            divisors.append(d)
    
    # Precompute residue arrays for each modulus (for fast coverage counting)
    # residue_positions[m] = np.arange(r, base, m) for each r... but that's too much memory
    # Instead, we'll compute on the fly with numpy
    
    def count_uncovered(covered, m, r):
        """Count uncovered positions in class (m, r) using numpy."""
        positions = np.arange(r, base, m)
        return np.sum(covered[positions] == 0)
    
    def compute_score(classes):
        """Compute number of uncovered residues."""
        covered = np.zeros(base, dtype=np.uint8)
        for m, r in classes:
            positions = np.arange(r, base, m)
            covered[positions] = 1
        return np.sum(covered == 0)
    
    def greedy_construct(divs_order, time_limit):
        """Greedy construction with given divisor order."""
        covered = np.zeros(base, dtype=np.uint8)
        classes = []
        
        for m in divs_order:
            if time.time() > time_limit:
                break
            
            # Find best residue for this modulus
            best_r = 0
            best_count = -1
            
            if m <= 100:
                # Try all residues
                for r in range(m):
                    positions = np.arange(r, base, m)
                    count = np.sum(covered[positions] == 0)
                    if count > best_count:
                        best_count = count
                        best_r = r
            else:
                # Sample residues for large moduli
                # Try structured candidates
                candidates = set()
                candidates.add(0)
                candidates.add(1)
                candidates.add(2)
                candidates.add(m // 2)
                candidates.add(m // 3)
                candidates.add(2 * m // 3)
                # Add some pseudo-random based on m
                for k in range(8):
                    candidates.add((m * 7919 + k * 104729 + len(classes) * 31) % m)
                
                for r in candidates:
                    r = r % m
                    positions = np.arange(r, base, m)
                    count = np.sum(covered[positions] == 0)
                    if count > best_count:
                        best_count = count
                        best_r = r
            
            classes.append([m, best_r])
            
            # Update coverage
            positions = np.arange(best_r, base, m)
            covered[positions] = 1
        
        return classes
    
    def local_search(classes, time_limit):
        """Improve classes by trying better residues for each modulus."""
        classes = [list(c) for c in classes]
        improved = True
        
        while improved and time.time() < time_limit:
            improved = False
            
            # Compute coverage without each class to find marginal benefit
            for i in range(len(classes)):
                if time.time() > time_limit:
                    break
                
                m, r = classes[i]
                
                # Compute coverage of all other classes
                covered_others = np.zeros(base, dtype=np.uint8)
                for j in range(len(classes)):
                    if j != i:
                        m2, r2 = classes[j]
                        positions = np.arange(r2, base, m2)
                        covered_others[positions] = 1
                
                # Current uncovered count without this class
                unc_without = np.sum(covered_others == 0)
                
                # Try to find better residue
                best_r = r
                best_unc = unc_without
                
                if m <= 80:
                    for r2 in range(m):
                        # Adding class (m, r2) to covered_others
                        positions = np.arange(r2, base, m)
                        new_covered = covered_others.copy()
                        new_covered[positions] = 1
                        unc = np.sum(new_covered == 0)
                        if unc < best_unc:
                            best_unc = unc
                            best_r = r2
                else:
                    # Sample for large moduli
                    candidates = {0, 1, 2, m//2, m//3, 2*m//3}
                    for k in range(6):
                        candidates.add((m * 7919 + k * 104729 + i * 31) % m)
                    
                    for r2 in candidates:
                        r2 = r2 % m
                        positions = np.arange(r2, base, m)
                        new_covered = covered_others.copy()
                        new_covered[positions] = 1
                        unc = np.sum(new_covered == 0)
                        if unc < best_unc:
                            best_unc = unc
                            best_r = r2
                
                if best_r != r:
                    classes[i] = [m, best_r]
                    improved = True
        
        return classes
    
    # Strategy: multiple restarts with different orderings
    best_classes = None
    best_score = base  # worst case: nothing covered
    
    # Ordering 1: descending (large moduli first)
    divs_desc = sorted(divisors, reverse=True)
    classes1 = greedy_construct(divs_desc, deadline - 60)
    classes1 = local_search(classes1, deadline - 30)
    score1 = compute_score(classes1)
    if score1 < best_score:
        best_score = score1
        best_classes = classes1
    
    # Ordering 2: ascending (small moduli first)
    divs_asc = sorted(divisors)
    classes2 = greedy_construct(divs_asc, deadline - 60)
    classes2 = local_search(classes2, deadline - 30)
    score2 = compute_score(classes2)
    if score2 < best_score:
        best_score = score2
        best_classes = classes2
    
    # Ordering 3: random
    rng = np.random.RandomState(42)
    divs_rand = list(divisors)
    rng.shuffle(divs_rand)
    classes3 = greedy_construct(divs_rand, deadline - 60)
    classes3 = local_search(classes3, deadline - 30)
    score3 = compute_score(classes3)
    if score3 < best_score:
        best_score = score3
        best_classes = classes3
    
    # Ordering 4: another random
    rng2 = np.random.RandomState(123)
    divs_rand2 = list(divisors)
    rng2.shuffle(divs_rand2)
    classes4 = greedy_construct(divs_rand2, deadline - 60)
    classes4 = local_search(classes4, deadline - 30)
    score4 = compute_score(classes4)
    if score4 < best_score:
        best_score = score4
        best_classes = classes4
    
    # If we still have time, do more local search on the best
    if best_classes is not None and time.time() < deadline - 15:
        best_classes = local_search(best_classes, deadline - 5)
    
    if best_classes is None:
        # Fallback: just use all divisors with residue 0
        best_classes = [[m, 0] for m in divisors]
    
    return best_classes
