import time
import random

def compute_metrics(A):
    """
    Compute sizes of A+A and A*A.
    A is a list of distinct positive integers.
    """
    n = len(A)
    sums = set()
    prods = set()
    for i in range(n):
        for j in range(i, n):
            sums.add(A[i] + A[j])
            prods.add(A[i] * A[j])
    return len(sums), len(prods)

def objective(A):
    s, p = compute_metrics(A)
    return max(s, p)

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    # Initial candidate: Arithmetic Progression 1..20
    # |A+A| = 39
    # |A*A| needs calculation
    best_A = list(range(1, 21))
    best_val = objective(best_A)
    
    # Also try a geometric progression or other structured sets if time permits
    # But AP is usually strong for sumsets.
    
    # Local Search
    random.seed(42)
    
    current_A = best_A[:]
    current_val = best_val
    
    # Pre-compute some neighbors or just do random perturbations
    
    while time.time() < deadline:
        # Generate a neighbor
        # Strategy: Replace one element with a random integer in [1, 10000]
        # Or swap two elements
        # Or add/subtract small delta
        
        move_type = random.randint(0, 2)
        
        if move_type == 0:
            # Replace
            idx = random.randint(0, 19)
            new_val = random.randint(1, 10000)
            if new_val in current_A and new_val != current_A[idx]:
                continue
            new_A = current_A[:]
            new_A[idx] = new_val
            if len(set(new_A)) != 20:
                continue
            if any(x < 1 or x > 10000 for x in new_A):
                continue
        elif move_type == 1:
            # Swap
            i, j = random.sample(range(20), 2)
            new_A = current_A[:]
            new_A[i], new_A[j] = new_A[j], new_A[i]
        else:
            # Perturb
            idx = random.randint(0, 19)
            delta = random.choice([-5, -3, -2, -1, 1, 2, 3, 5, 10, -10])
            new_val = current_A[idx] + delta
            if new_val < 1 or new_val > 10000:
                continue
            if new_val in current_A and new_val != current_A[idx]:
                continue
            new_A = current_A[:]
            new_A[idx] = new_val
            if len(set(new_A)) != 20:
                continue
        
        # Evaluate
        # Optimization: Check if it's worth calculating full metrics?
        # For n=20, 210 pairs. It's fast.
        val = objective(new_A)
        
        if val < current_val:
            current_A = new_A
            current_val = val
            if current_val < best_val:
                best_A = current_A[:]
                best_val = current_val
                # Optional: restart or cool down? 
                # Simple hill climbing can get stuck. 
                # Maybe add a random restart if stuck for a while?
        
        # Simulated Annealing style? 
        # Accept worse moves with probability exp(-delta/T)?
        # Given time limit, simple hill climbing with random restarts might be better.
        
        # Let's add a simple restart mechanism if no improvement for many steps
        # But we don't track steps easily in this loop structure without counters.
        
        # Instead, let's try a few different initializations if time allows.
        
    return best_A
