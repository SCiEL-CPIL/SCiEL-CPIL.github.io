"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _all_odd_divisors(n):
    divs = []
    for d in range(3, n + 1, 2):
        if n % d == 0:
            divs.append(d)
    return divs

def _count_uncovered(covered, m, r):
    # covered is a numpy array of 0/1
    return covered[r::m].sum()

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    base = 51975
    # Precompute all odd divisors > 1 of base
    divisors = _all_odd_divisors(base)
    # Sort by size descending for better coverage early, but we'll use a smart order
    # Actually, let's think about the greedy strategy.
    # The seed program sorts by increasing m. Small moduli cover more residues.
    # But small moduli are "expensive" in terms of the number of classes they use?
    # No, each modulus can only be used once.
    # Small moduli cover a large fraction of residues.
    # Large moduli cover a small fraction.
    # To minimize uncovered, we want to cover as much as possible.
    # Greedy: pick the modulus that covers the most *currently uncovered* residues.
    # However, we have to pick residues too.
    
    # Let's use a randomized restart approach with a time budget.
    # Each restart:
    # 1. Shuffle divisors (or sort by some heuristic).
    # 2. For each modulus in the order, pick the residue that covers the most uncovered residues.
    # 3. Track the best solution found.
    
    best_score = base  # initially all uncovered
    best_classes = None
    rng = random.Random(42)
    
    # Precompute ranges for each modulus to speed up numpy slicing?
    # Actually, numpy slicing is fast.
    
    # Strategy 1: Greedy with sorted divisors (small to large)
    # This is the seed program's approach, but we can improve by trying different orders.
    
    # Let's try a few different heuristic orders:
    # 1. Increasing m
    # 2. Decreasing m
    # 3. Random order
    # 4. Order by m / gcd(m, something)? No, they are all divisors.
    
    # Let's also consider that choosing a residue for a small modulus is critical.
    # If we pick m=3, r=0, we cover 1/3 of residues.
    # If we pick m=5, r=0, we cover 1/5.
    # The interactions matter.
    
    # Let's implement a local search on top of a greedy initialization.
    # State: a set of (m, r) pairs.
    # Move: change r for a given m, or swap m with another m?
    # Since moduli must be distinct, we can't add new moduli if we've used them all.
    # But we don't have to use all moduli. We can choose a subset.
    # The seed program uses ALL divisors. Is that optimal?
    # Using a modulus always adds coverage (or keeps it same if already covered).
    # So using all moduli is weakly better than using a subset.
    # Therefore, we should use all divisors.
    
    # So the problem reduces to: assign a residue r_m to each divisor m > 1 of 51975
    # to minimize the number of uncovered residues.
    
    # This is a large optimization problem. Number of divisors of 51975:
    # 51975 = 3^3 * 5^2 * 7 * 11
    # Number of divisors = (3+1)*(2+1)*(1+1)*(1+1) = 4*3*2*2 = 48.
    # So we have 48 moduli to assign residues to.
    # Total residues = 51975.
    
    # Let's use simulated annealing or hill climbing on the assignment of residues.
    # State: array of residues for the 48 moduli.
    # Objective: count uncovered residues.
    
    # To evaluate the objective quickly, we can maintain a coverage count array.
    # But updating it when one residue changes is O(base/m).
    # With 48 moduli, and base=51975, this is fast enough.
    
    # Let's precompute the divisors and their indices.
    moduli = divisors
    num_moduli = len(moduli)
    
    # Initialize with greedy:
    covered = np.zeros(base, dtype=np.int8)
    current_residues = [0] * num_moduli
    for i, m in enumerate(moduli):
        # Find best r
        best_r = 0
        best_gain = -1
        for r in range(m):
            # Count how many uncovered residues are in r::m
            # We can do this by checking covered[r::m]
            gain = np.count_nonzero(covered[r::m] == 0)
            if gain > best_gain:
                best_gain = gain
                best_r = r
        current_residues[i] = best_r
        covered[best_r::m] = 1
    
    current_score = np.count_nonzero(covered == 0)
    if current_score < best_score:
        best_score = current_score
        best_classes = [[moduli[i], current_residues[i]] for i in range(num_moduli)]
    
    # Now do local search / simulated annealing
    # We'll use a simple hill climbing with random restarts of the search order.
    
    # Time check
    if time.time() > deadline:
        return best_classes
    
    # Let's try a few random restarts of the greedy phase with different random seeds for tie-breaking
    # Or just perturb the current solution.
    
    # Perturbation: pick a random modulus, change its residue to a random value.
    # Accept if it improves or is equal (to escape plateaus).
    
    temperature = 1.0
    cooling = 0.99
    min_temp = 0.01
    
    while time.time() < deadline:
        # Pick a random modulus index
        i = rng.randint(0, num_moduli - 1)
        m = moduli[i]
        old_r = current_residues[i]
        
        if m == 1:
            continue
            
        # Pick a new random residue
        new_r = rng.randint(0, m - 1)
        if new_r == old_r:
            continue
            
        # Calculate delta in score
        # Remove old coverage
        # We need to count how many residues were covered ONLY by this class.
        # This is complex. Easier to just recompute the score?
        # Recomputing score is O(base) = 51975, which is fast.
        # But we have to update the covered array.
        
        # Let's just update the covered array and recompute the score.
        # covered[old_r::m] -= 1
        # But we need to know if other classes cover those residues.
        # So we can't just subtract 1.
        
        # Alternative: maintain a count array of how many classes cover each residue.
        # counts[residue] = number of classes covering it.
        # Uncovered = count of residues where counts[residue] == 0.
        
        # Let's switch to this representation.
        break  # Exit this loop and restart with counts array
    
    # Restart with counts array representation
    counts = np.zeros(base, dtype=np.int8)
    current_residues = [0] * num_moduli
    
    # Re-initialize with greedy using counts
    for i, m in enumerate(moduli):
        best_r = 0
        best_gain = -1
        for r in range(m):
            # Gain is the number of residues with count 0 in r::m
            gain = np.count_nonzero(counts[r::m] == 0)
            if gain > best_gain:
                best_gain = gain
                best_r = r
        current_residues[i] = best_r
        counts[best_r::m] += 1
    
    current_score = np.count_nonzero(counts == 0)
    if current_score < best_score:
        best_score = current_score
        best_classes = [[moduli[i], current_residues[i]] for i in range(num_moduli)]
    
    # Simulated Annealing
    temperature = 5.0
    cooling = 0.995
    min_temp = 0.1
    
    while time.time() < deadline and temperature > min_temp:
        i = rng.randint(0, num_moduli - 1)
        m = moduli[i]
        old_r = current_residues[i]
        new_r = rng.randint(0, m - 1)
        if new_r == old_r:
            continue
        
        # Update counts
        counts[old_r::m] -= 1
        counts[new_r::m] += 1
        
        new_score = np.count_nonzero(counts == 0)
        delta = new_score - current_score
        
        if delta < 0 or rng.random() < np.exp(-delta / temperature):
            current_residues[i] = new_r
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_classes = [[moduli[i], current_residues[i]] for i in range(num_moduli)]
        else:
            # Revert
            counts[new_r::m] -= 1
            counts[old_r::m] += 1
        
        temperature *= cooling
    
    return best_classes
# EVOLVE-BLOCK-END
