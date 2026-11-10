import random
import time
from itertools import combinations

def _is_sidon(subset):
    """
    Check if a subset is a Sidon set.
    All unordered pair sums (including diagonals i<=j) must be unique.
    """
    sums = set()
    n = len(subset)
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _max_sidon_size(ambient_set):
    """
    Exhaustively find the size of the largest Sidon subset within ambient_set.
    ambient_set is a list of 12 distinct integers.
    """
    n = len(ambient_set)
    # Check from largest possible size down to 1
    for k in range(n, 0, -1):
        for combo in combinations(ambient_set, k):
            if _is_sidon(combo):
                return k
    return 1

def construct():
    deadline = time.time() + 55  # Safety margin for 60s limit
    
    # Initial random set
    rng = random.Random(42)
    
    # Start with a random set of 12 distinct integers in [0, 128]
    current = list(rng.sample(range(129), 12))
    current_score = _max_sidon_size(current)
    
    best_set = list(current)
    best_score = current_score
    
    # Local Search
    # We will perform random single-element replacements
    # If the new score is better, we accept.
    # To escape local minima, we can occasionally accept worse moves (Simulated Annealing style)
    # or just rely on the fact that the landscape is rugged and random restarts help.
    # Given the small N, simple hill climbing might get stuck.
    # Let's use a simple Simulated Annealing.
    
    T = 10.0  # Initial temperature
    T_min = 0.1
    cooling_rate = 0.999
    
    step = 0
    while time.time() < deadline:
        # Generate neighbor
        candidate = list(current)
        idx = rng.randrange(12)
        new_val = rng.randint(0, 128)
        
        # Ensure distinctness
        if new_val in candidate:
            # Try to find a valid distinct value quickly
            # Or just skip
            continue
        
        candidate[idx] = new_val
        
        new_score = _max_sidon_size(candidate)
        
        # Acceptance criterion
        if new_score < current_score:
            current = candidate
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_set = list(candidate)
                
            # Reset temperature on improvement? Or just keep cooling.
            # Keeping cooling is standard SA.
        else:
            # Metropolis criterion
            # Delta E = new_score - current_score (we want to minimize, so delta > 0 is bad)
            delta = new_score - current_score
            if delta > 0:
                # Probability of accepting worse solution
                prob = 1.0 / (1.0 + delta) # Simple heuristic
                # Or exp(-delta/T)
                # Since scores are integers (1-12), delta is small.
                # exp(-1/10) is ~0.9. exp(-5/10) is ~0.6.
                # Let's use exp
                import math
                if T > T_min:
                    p = math.exp(-delta / T)
                    if rng.random() < p:
                        current = candidate
                        current_score = new_score
                # Else reject
                
        T *= cooling_rate
        step += 1
        
        # Optional: Periodic random restart if stuck for too long?
        # For now, SA should handle it.
        
    return best_set
