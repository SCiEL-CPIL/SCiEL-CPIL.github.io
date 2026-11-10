"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from itertools import combinations

def _collision_count(ambient_set):
    """Count the number of sum collisions in the ambient set.
    A collision occurs when two different unordered pairs (i,j) with i<=j
    produce the same sum. Lower is better (0 means the whole set is Sidon)."""
    sums = {}
    n = len(ambient_set)
    for i in range(n):
        for j in range(i, n):
            s = ambient_set[i] + ambient_set[j]
            if s in sums:
                sums[s] += 1
            else:
                sums[s] = 1
    collisions = 0
    for cnt in sums.values():
        if cnt > 1:
            collisions += cnt - 1
    return collisions

def _max_sidon_size(ambient_set):
    """Find the largest Sidon subset size using branch and bound."""
    n = len(ambient_set)
    if n <= 1:
        return n
    
    # Quick check: if no collisions, whole set is Sidon
    if _collision_count(ambient_set) == 0:
        return n
    
    # Use backtracking with pruning
    best_size = 1
    current_subset = []
    
    def backtrack(start, current):
        nonlocal best_size
        if len(current) > best_size:
            best_size = len(current)
        
        if start >= n:
            return
        
        # Pruning: even if we add all remaining, can't beat best
        if len(current) + (n - start) <= best_size:
            return
        
        for i in range(start, n):
            # Check if adding ambient_set[i] to current keeps it Sidon
            valid = True
            for j in range(len(current)):
                s = current[j] + ambient_set[i]
                # Check if this sum already exists in current
                for k in range(j, len(current)):
                    if current[k] + current[j] == s:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                current.append(ambient_set[i])
                backtrack(i + 1, current)
                current.pop()
    
    backtrack(0, [])
    return best_size

def construct():
    deadline = time.time() + 55  # Safety margin before 60s limit
    rng = random.Random(42)
    
    # Initialize with a set that has high collision count
    # Using consecutive integers maximizes collisions
    current = list(range(12))
    current_collisions = _collision_count(current)
    
    best_set = list(current)
    best_collisions = current_collisions
    
    temperature = 10.0
    cooling_rate = 0.9995
    
    while time.time() < deadline:
        candidate = list(current)
        idx = rng.randrange(12)
        new_val = rng.randint(0, 128)
        if new_val in candidate:
            continue
        candidate[idx] = new_val
        
        candidate_collisions = _collision_count(candidate)
        
        if candidate_collisions < best_collisions:
            best_collisions = candidate_collisions
            best_set = list(candidate)
        
        delta = candidate_collisions - current_collisions
        
        if delta < 0 or rng.random() < pow(2.718281828, -delta / max(temperature, 0.01)):
            current = candidate
            current_collisions = candidate_collisions
        
        temperature *= cooling_rate
    
    # Final verification and optimization pass
    final_score = _max_sidon_size(best_set)
    
    # Try local improvement on the best set found
    improved = list(best_set)
    improved_score = final_score
    
    for _ in range(1000):
        if time.time() > deadline - 5:
            break
        idx = rng.randrange(12)
        new_val = rng.randint(0, 128)
        if new_val in improved:
            continue
        candidate = list(improved)
        candidate[idx] = new_val
        candidate_score = _max_sidon_size(candidate)
        
        if candidate_score < improved_score:
            improved = candidate
            improved_score = candidate_score
        elif candidate_score == improved_score and rng.random() < 0.3:
            improved = candidate
    
    if improved_score < final_score:
        best_set = improved
        final_score = improved_score
    
    return best_set
# EVOLVE-BLOCK-END
