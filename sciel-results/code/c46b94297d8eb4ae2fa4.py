import random
import time

def construct():
    n = 20
    max_val = 10000
    deadline = time.time() + 50  # 50s to be safe
    
    def set_sizes(A):
        sums = set()
        prods = set()
        for i in range(len(A)):
            for j in range(i, len(A)):
                sums.add(A[i] + A[j])
                prods.add(A[i] * A[j])
        return len(sums), len(prods)
    
    def objective(A):
        s, p = set_sizes(A)
        return max(s, p)
    
    # Initial set: try a geometric-like progression with small base
    # Use powers of 2 but cap at 10000, then fill
    # Actually, let's try multiple initializations
    
    best_A = None
    best_score = float('inf')
    
    def update_best(A, score):
        nonlocal best_A, best_score
        if score < best_score:
            best_score = score
            best_A = list(A)
    
    # Initial candidates
    # 1. Arithmetic progression
    A1 = list(range(1, 21))
    update_best(A1, objective(A1))
    
    # 2. Geometric progression (powers of 2)
    A2 = [2**i for i in range(14)]  # 1,2,4,...,8192 (14 elements)
    # Need 20 elements, extend
    while len(A2) < 20:
        A2.append(A2[-1] * 2)
    A2 = A2[:20]
    if max(A2) <= max_val:
        update_best(A2, objective(A2))
    
    # 3. Random sets
    random.seed(42)
    for trial in range(20):
        A = random.sample(range(1, max_val + 1), n)
        s = objective(A)
        update_best(A, s)
    
    # Local search with simulated annealing
    current_A = list(best_A)
    current_score = best_score
    temperature = 1000.0
    cooling = 0.99
    
    while time.time() < deadline:
        # Generate neighbor
        A = list(current_A)
        move_type = random.random()
        
        if move_type < 0.3:
            # Swap two elements
            i, j = random.sample(range(n), 2)
            A[i], A[j] = A[j], A[i]
        elif move_type < 0.6:
            # Replace one element
            i = random.randrange(n)
            A[i] = random.randint(1, max_val)
            if len(set(A)) < n:
                continue
        elif move_type < 0.8:
            # Perturb one element
            i = random.randrange(n)
            delta = random.randint(-50, 50)
            A[i] = max(1, min(max_val, A[i] + delta))
            if len(set(A)) < n:
                continue
        else:
            # Replace two elements
            i, j = random.sample(range(n), 2)
            A[i] = random.randint(1, max_val)
            A[j] = random.randint(1, max_val)
            if len(set(A)) < n:
                continue
        
        new_score = objective(A)
        delta = new_score - current_score
        
        if delta <= 0 or random.random() < pow(2.718281828, -delta / max(temperature, 1e-10)):
            current_A = A
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_A = list(current_A)
        
        temperature *= cooling
        if temperature < 1e-6:
            temperature = 1000.0  # Restart
            # Optional: Re-initialize to best known if stuck?
            # For now, just restart temp.
    
    return best_A
