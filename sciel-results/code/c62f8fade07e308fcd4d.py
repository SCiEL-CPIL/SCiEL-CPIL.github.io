"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import random
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 70  # safety margin for 90s limit
    
    def compute_objective(A):
        """Compute max(|A+A|, |A*A|) for set A."""
        A_set = set(A)
        sums = set()
        prods = set()
        n = len(A)
        for i in range(n):
            for j in range(i, n):
                sums.add(A[i] + A[j])
                prods.add(A[i] * A[j])
        return max(len(sums), len(prods))
    
    def is_valid(A):
        """Check if A is a valid set of 20 distinct positive integers <= 10000."""
        if len(A) != 20:
            return False
        if len(set(A)) != 20:
            return False
        if any(x < 1 or x > 10000 for x in A):
            return False
        return True
    
    # Start with the seed program's answer
    best_A = list(range(1, 21))
    best_score = compute_objective(best_A)
    
    # Generate 5-smooth numbers (of form 2^a * 3^b * 5^c) <= 10000
    def generate_5_smooth(max_val=10000):
        smooths = set()
        for a in range(0, 14):  # 2^13 = 8192
            for b in range(0, 10):  # 3^9 = 19683
                for c in range(0, 7):  # 5^6 = 15625
                    val = (2 ** a) * (3 ** b) * (5 ** c)
                    if val <= max_val and val >= 1:
                        smooths.add(val)
        return sorted(smooths)
    
    smooth_numbers = generate_5_smooth(10000)
    
    # Strategy 1: Try geometric progressions with small ratios
    def try_geometric_progression():
        candidates = []
        # Try various bases and exponents
        for base in range(2, 100):
            for start in range(1, 50):
                A = []
                val = start
                for i in range(20):
                    if val > 10000:
                        break
                    A.append(val)
                    val = val * base
                if len(A) == 20 and is_valid(A):
                    candidates.append(A)
        return candidates
    
    # Strategy 2: Random search in 5-smooth numbers
    def random_search_smooth(num_iterations=5000):
        best_local = None
        best_local_score = float('inf')
        
        for _ in range(num_iterations):
            if time.time() > deadline - 5:
                break
            
            # Pick 20 random 5-smooth numbers
            A = random.sample(smooth_numbers, 20)
            A = sorted(A)
            
            if not is_valid(A):
                continue
            
            score = compute_objective(A)
            if score < best_local_score:
                best_local_score = score
                best_local = A
        
        return best_local, best_local_score
    
    # Strategy 3: Local search / hill climbing
    def local_search(initial_A, max_iterations=10000):
        A = list(initial_A)
        current_score = compute_objective(A)
        
        for iteration in range(max_iterations):
            if time.time() > deadline - 5:
                break
            
            improved = False
            # Try swapping one element with another from the pool
            pool = smooth_numbers + list(range(1, 101))
            
            for i in range(20):
                if time.time() > deadline - 5:
                    break
                for new_val in random.sample(pool, min(50, len(pool))):
                    if new_val in A and A.index(new_val) != i:
                        continue
                    if new_val > 10000 or new_val < 1:
                        continue
                    
                    new_A = A.copy()
                    new_A[i] = new_val
                    
                    if not is_valid(new_A):
                        continue
                    
                    score = compute_objective(new_A)
                    if score < current_score:
                        A = new_A
                        current_score = score
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                # Try adding a random new element and removing one
                for _ in range(10):
                    if time.time() > deadline - 5:
                        break
                    remove_idx = random.randint(0, 19)
                    add_val = random.choice(pool)
                    
                    if add_val in A:
                        continue
                    
                    new_A = A.copy()
                    new_A.pop(remove_idx)
                    new_A.append(add_val)
                    new_A = sorted(new_A)
                    
                    if not is_valid(new_A):
                        continue
                    
                    score = compute_objective(new_A)
                    if score < current_score:
                        A = new_A
                        current_score = score
                        improved = True
                        break
        
        return A, current_score
    
    # Run geometric progression search
    for A in try_geometric_progression():
        if time.time() > deadline - 10:
            break
        score = compute_objective(A)
        if score < best_score:
            best_score = score
            best_A = A
    
    # Run random search in 5-smooth numbers
    local_best, local_score = random_search_smooth(3000)
    if local_best and local_score < best_score:
        best_score = local_score
        best_A = local_best
    
    # Run local search from best found so far
    if time.time() < deadline - 15:
        improved_A, improved_score = local_search(best_A, 5000)
        if improved_score < best_score:
            best_score = improved_score
            best_A = improved_A
    
    # Final verification
    assert is_valid(best_A), f"Invalid result: {best_A}"
    assert len(best_A) == 20
    assert all(1 <= x <= 10000 for x in best_A)
    
    return sorted(best_A)
# EVOLVE-BLOCK-END
