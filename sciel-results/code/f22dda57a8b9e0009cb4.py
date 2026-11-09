import time
import numpy as np

def _get_divisors(n):
    divisors = []
    for i in range(1, int(np.sqrt(n)) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    return sorted([d for d in divisors if d > 1])

def _score_classes(classes, base=51975):
    """
    Calculate the number of uncovered residues.
    """
    if not classes:
        return base
    # Use a numpy boolean array for coverage
    covered = np.zeros(base, dtype=bool)
    for m, r in classes:
        covered[r::m] = True
    return base - np.sum(covered)

def _greedy_construct(base=51975):
    """
    Greedily construct a covering system by adding the class with maximum marginal gain.
    """
    covered = np.zeros(base, dtype=bool)
    classes = []
    divisors = [d for d in range(3, base + 1, 2) if base % d == 0]
    
    # Sort divisors to potentially pick smaller ones first? Or just iterate.
    # The greedy choice should be based on marginal gain.
    # We have to pick one class per modulus? No, the problem says "distinct moduli".
    # It implies we select a subset of moduli and assign a residue to each.
    # The seed program iterates through ALL divisors and picks the best residue for each.
    # This results in a system with one class per divisor.
    # Is it beneficial to skip a modulus?
    # If a modulus adds no new coverage, it doesn't hurt the score, but it uses up a modulus.
    # Since we want to minimize uncovered residues, adding a class that covers 0 new residues is neutral.
    # However, the constraint is "distinct moduli". It doesn't say we MUST use all.
    # But typically, more classes = better coverage.
    # Let's stick to the seed's strategy: use all available divisors.
    
    # To speed up greedy, we can precompute the gain for each (m, r)
    # But 'covered' changes.
    
    for m in divisors:
        best_r = 0
        max_gain = -1
        # We need to find r that maximizes the number of 0s in covered[r::m]
        # covered[r::m] is a view. sum(covered[r::m] == 0) is the gain.
        # This is O(base) per r, and O(m) residues. Total O(base * m) per modulus.
        # Sum of m for all divisors is roughly base * sum(1/d) which is manageable?
        # Sum of divisors of 51975 is not huge.
        # 51975 = 3^3 * 5^2 * 7 * 11.
        # Let's optimize the gain calculation.
        
        # Instead of looping r and summing, we can use numpy.
        # But slicing creates views.
        # Let's just loop r. m is small on average.
        for r in range(m):
            gain = np.sum(covered[r::m] == False)
            if gain > max_gain:
                max_gain = gain
                best_r = r
        
        classes.append([m, best_r])
        covered[best_r::m] = True
        
    return classes

def _local_search(classes, base=51975, time_limit=10):
    """
    Local search: try changing residues for existing moduli.
    Uses incremental coverage count update for speed.
    """
    start_time = time.time()
    
    # Initialize coverage counts
    # count[x] = number of classes covering residue x
    counts = np.zeros(base, dtype=np.int32)
    for m, r in classes:
        counts[r::m] += 1
    
    current_score = base - np.sum(counts > 0)
    best_score = current_score
    best_classes = [c[:] for c in classes]
    
    # Precompute indices for each modulus to speed up updates
    # indices[m] is a list of arrays? Or just use slicing.
    # Slicing counts[r::m] is fast in numpy.
    
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        # Shuffle the order of moduli to avoid bias?
        # Or just iterate.
        for i, (m, r) in enumerate(best_classes):
            if time.time() - start_time > time_limit:
                break
            
            # Try all other residues for this modulus
            # Optimization: Only try residues that might be better?
            # For now, try all.
            for new_r in range(m):
                if new_r == r:
                    continue
                
                # Calculate delta score
                # Old residues: r::m
                # New residues: new_r::m
                
                # We need to know how many residues in new_r::m are currently 0 (would become 1)
                # and how many in r::m are currently 1 (would become 0)
                
                # However, there is overlap? No, same modulus, different residues.
                # The sets {x | x % m == r} and {x | x % m == new_r} are disjoint.
                
                # So:
                # Residues in new_r::m that are 0 will become 1. Gain = count of 0s in new_r::m.
                # Residues in r::m that are 1 will become 0. Loss = count of 1s in r::m.
                # But wait, if a residue is covered by ANOTHER class, changing this class might not change its 0/1 status.
                
                # Let's look at the counts.
                # Let S_new = {x | x % m == new_r}
                # Let S_old = {x | x % m == r}
                
                # For x in S_new:
                # If counts[x] == 0, it becomes 1. (Gain 1)
                # If counts[x] > 0, it stays 1 (or becomes counts[x]+1, still 1). (Gain 0)
                
                # For x in S_old:
                # If counts[x] == 1, it becomes 0. (Loss 1)
                # If counts[x] > 1, it stays 1 (or becomes counts[x]-1, still 1). (Loss 0)
                
                # So:
                # Gain = np.sum(counts[new_r::m] == 0)
                # Loss = np.sum(counts[r::m] == 1)
                
                gain = np.sum(counts[new_r::m] == 0)
                loss = np.sum(counts[r::m] == 1)
                
                delta = gain - loss
                if delta > 0:
                    # Apply change
                    counts[r::m] -= 1
                    counts[new_r::m] += 1
                    current_score += delta
                    best_classes[i][1] = new_r
                    r = new_r
                    improved = True
                    if current_score < best_score:
                        best_score = current_score
                    break # Restart the inner loop? Or continue?
                    # Breaking the inner loop and restarting the outer loop (or continuing to next i)
                    # is a "first improvement" strategy. "Best improvement" would check all r.
                    # Let's try "best improvement" for the current modulus to be more robust.
                    # Actually, checking all r for a modulus is O(m). m is small.
                    # Let's find the best new_r for this m.
            
            # Let's refactor to find the BEST new_r for the current m
            # This is better than first-improvement.
            
    # Refactored Local Search Step
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        for i, (m, r) in enumerate(best_classes):
            if time.time() - start_time > time_limit:
                break
            
            best_new_r = r
            best_delta = 0
            
            for new_r in range(m):
                if new_r == r:
                    continue
                
                gain = np.sum(counts[new_r::m] == 0)
                loss = np.sum(counts[r::m] == 1)
                delta = gain - loss
                
                if delta > best_delta:
                    best_delta = delta
                    best_new_r = new_r
            
            if best_delta > 0:
                # Apply best change
                counts[r::m] -= 1
                counts[best_new_r::m] += 1
                current_score += best_delta
                best_classes[i][1] = best_new_r
                improved = True
                if current_score < best_score:
                    best_score = current_score
    
    return best_classes

def construct():
    base = 51975
    start_time = time.time()
    deadline = start_time + 140
    
    divisors = _get_divisors(base)
    
    # 1. Greedy Construction
    classes = _greedy_construct(base)
    
    # 2. Local Search
    remaining_time = deadline - time.time()
    if remaining_time > 2:
        classes = _local_search(classes, base, time_limit=min(remaining_time - 2, 60))
        
    # 3. Multi-Start / Perturbation if time remains
    # We can try to perturb the solution and run local search again.
    current_score = _score_classes(classes, base)
    best_score = current_score
    best_classes = [c[:] for c in classes]
    
    while time.time() < deadline - 5:
        # Perturb: change 1-3 residues randomly
        perturbed = [c[:] for c in best_classes]
        num_changes = np.random.randint(1, 4)
        for _ in range(num_changes):
            idx = np.random.randint(0, len(perturbed))
            m = perturbed[idx][0]
            new_r = np.random.randint(0, m)
            perturbed[idx][1] = new_r
        
        # Local search on perturbed
        remaining = deadline - time.time()
        if remaining < 5:
            break
        time_limit_ls = min(remaining - 5, 10) # Keep it short to allow more restarts
        refined = _local_search(perturbed, base, time_limit=time_limit_ls)
        
        score = _score_classes(refined, base)
        if score < best_score:
            best_score = score
            best_classes = [c[:] for c in refined]
            
    return best_classes
