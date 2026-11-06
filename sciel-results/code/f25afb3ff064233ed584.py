"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    n = 20
    max_val = 10000
    
    # Generate 2,3-smooth numbers (Hamming numbers) up to max_val
    smooths = []
    for a in range(0, 15):
        for b in range(0, 15):
            val = (2 ** a) * (3 ** b)
            if 1 <= val <= max_val:
                smooths.append(val)
    smooths = list(set(smooths))
    smooths.sort()
    
    def compute_objective(A):
        """Compute max(|A+A|, |A*A|) for set A"""
        A_set = set(A)
        
        # Compute sumset A+A
        sumset = set()
        for i in range(len(A)):
            for j in range(i, len(A)):
                sumset.add(A[i] + A[j])
        
        # Compute product set A*A
        prodset = set()
        for i in range(len(A)):
            for j in range(i, len(A)):
                prodset.add(A[i] * A[j])
        
        return max(len(sumset), len(prodset))
    
    # Initial population from smooth numbers
    # Pick 20 smooth numbers that minimize the objective
    # Use a greedy/random approach to find a good starting set
    
    best_set = None
    best_score = float('inf')
    
    # Try a few random subsets of smooth numbers
    random.seed(42)
    
    # First, try the smallest 20 smooth numbers
    initial_candidates = []
    
    # Strategy 1: Smallest smooths
    if len(smooths) >= n:
        candidate = smooths[:n]
        score = compute_objective(candidate)
        if score < best_score:
            best_score = score
            best_set = candidate[:]
    
    # Strategy 2: Random subsets of smooths
    for _ in range(50):
        if len(smooths) < n:
            break
        candidate = random.sample(smooths, n)
        candidate.sort()
        score = compute_objective(candidate)
        if score < best_score:
            best_score = score
            best_set = candidate[:]
        
        if time.time() > deadline - 10:
            break
    
    # Strategy 3: Mix of small and large smooths for balance
    if len(smooths) >= n:
        half = n // 2
        candidate = smooths[:half] + smooths[len(smooths)//2:len(smooths)//2 + (n - half)]
        candidate = sorted(set(candidate))
        if len(candidate) == n:
            score = compute_objective(candidate)
            if score < best_score:
                best_score = score
                best_set = candidate[:]
    
    if best_set is None:
        # Fallback
        best_set = list(range(1, n + 1))
        best_score = compute_objective(best_set)
    
    # Simulated Annealing refinement
    current_set = best_set[:]
    current_score = best_score
    best_set_sa = current_set[:]
    best_score_sa = current_score
    
    T = 10.0
    T_min = 0.1
    cooling = 0.995
    
    # Precompute some useful data
    def get_neighbors(A):
        """Generate possible moves"""
        moves = []
        # Single element replacement
        for i in range(len(A)):
            # Try replacing with random smooth
            for _ in range(3):
                new_val = random.choice(smooths)
                if new_val not in A:
                    moves.append((i, new_val))
            # Try small perturbations
            for delta in [-1, 1, -2, 2, -3, 3]:
                new_val = A[i] + delta
                if 1 <= new_val <= max_val and new_val not in A:
                    moves.append((i, new_val))
        return moves
    
    iteration = 0
    while time.time() < deadline - 5:
        if T < T_min:
            T = T_min
        
        # Generate a random move
        i = random.randint(0, n - 1)
        move_type = random.random()
        
        if move_type < 0.5:
            # Replace with random smooth
            new_val = random.choice(smooths)
        elif move_type < 0.75:
            # Small perturbation
            new_val = current_set[i] + random.randint(-5, 5)
        else:
            # Swap two elements
            j = random.randint(0, n - 1)
            if j == i:
                continue
            new_val = current_set[j]
            # Also perturb the other element
            current_set[i], current_set[j] = current_set[j], current_set[i]
            new_score = compute_objective(current_set)
            delta = new_score - current_score
            if delta < 0 or random.random() < np.exp(-delta / T):
                if new_score < best_score_sa:
                    best_score_sa = new_score
                    best_set_sa = current_set[:]
                current_score = new_score
            else:
                current_set[i], current_set[j] = current_set[j], current_set[i]
            T *= cooling
            iteration += 1
            continue
        
        if not (1 <= new_val <= max_val) or new_val in current_set:
            T *= cooling
            iteration += 1
            continue
        
        old_val = current_set[i]
        current_set[i] = new_val
        current_set.sort()
        
        new_score = compute_objective(current_set)
        delta = new_score - current_score
        
        if delta < 0 or random.random() < np.exp(-delta / T):
            current_score = new_score
            if new_score < best_score_sa:
                best_score_sa = new_score
                best_set_sa = current_set[:]
        else:
            current_set[i] = old_val
            current_set.sort()
        
        T *= cooling
        iteration += 1
    
    # Final local search with tighter tolerance
    if time.time() < deadline - 2:
        for _ in range(200):
            improved = False
            for i in range(n):
                for delta_val in range(-10, 11):
                    new_val = current_set[i] + delta_val
                    if not (1 <= new_val <= max_val) or new_val in current_set:
                        continue
                    old_val = current_set[i]
                    current_set[i] = new_val
                    current_set.sort()
                    new_score = compute_objective(current_set)
                    if new_score < best_score_sa:
                        best_score_sa = new_score
                        best_set_sa = current_set[:]
                        improved = True
                    else:
                        current_set[i] = old_val
                        current_set.sort()
            if not improved:
                break
    
    return best_set_sa

# EVOLVE-BLOCK-END
