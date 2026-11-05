import random
import time
import numpy as np

def construct():
    n = 20
    max_val = 10000
    deadline = time.time() + 55  # 55s to be safe, leaving 5s margin
    
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
    
    def validate(A):
        if len(A) != n:
            return False
        if len(set(A)) < n:
            return False
        if any(x < 1 or x > max_val for x in A):
            return False
        return True
    
    best_A = None
    best_score = float('inf')
    
    def update_best(A, score):
        nonlocal best_A, best_score
        if score < best_score:
            best_score = score
            best_A = list(A)
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. Arithmetic progression
    A1 = list(range(1, 21))
    candidates.append(A1)
    
    # 2. Geometric progression (powers of 2)
    A2 = [2**i for i in range(14)]
    while len(A2) < 20:
        A2.append(A2[-1] * 2)
    A2 = A2[:20]
    if max(A2) <= max_val:
        candidates.append(A2)
    
    # 3. Log-spaced values
    log_min, log_max = 0, np.log10(max_val)
    A3 = [int(10**x) for x in np.linspace(log_min, log_max, n)]
    # Ensure distinct and in range
    A3 = sorted(set(A3))
    while len(A3) < n:
        A3.append(max_val - len(A3))
    A3 = A3[:n]
    if validate(A3):
        candidates.append(A3)
    
    # 4. Hybrid: first half arithmetic, second half geometric
    A4 = list(range(1, 11))
    A4 += [2**i for i in range(10, 14)]
    A4 += [2**14 * i for i in range(1, 3)]  # 16384, 32768 - too big
    # Let me redo this
    A4 = list(range(1, 11))
    A4 += [2**i for i in range(4, 10)]  # 16, 32, 64, 128, 256, 512
    A4 += [1024, 2048, 4096, 8192]
    A4 = sorted(set(A4))
    if len(A4) >= n:
        A4 = A4[:n]
    if validate(A4):
        candidates.append(A4)
    
    # 5. Random sets with bias toward small values
    random.seed(42)
    for trial in range(30):
        # Use a mix of small and large values
        small = random.sample(range(1, 100), min(10, n))
        large = random.sample(range(100, max_val + 1), n - len(small))
        A = small + large
        random.shuffle(A)
        if validate(A):
            candidates.append(A)
    
    # 6. Pure geometric with small base
    for base in [2, 3]:
        A = []
        val = 1
        while len(A) < n and val <= max_val:
            A.append(val)
            val *= base
        if len(A) >= n:
            A = A[:n]
            if validate(A):
                candidates.append(A)
    
    # 7. Arithmetic progression with larger step
    for step in [2, 3, 5]:
        A = [1 + i * step for i in range(n)]
        if max(A) <= max_val and validate(A):
            candidates.append(A)
    
    # Evaluate all candidates
    for A in candidates:
        if validate(A):
            score = objective(A)
            update_best(A, score)
    
    if best_A is None:
        best_A = list(range(1, 21))
        best_score = objective(best_A)
    
    # Local search with structured moves
    current_A = list(best_A)
    current_score = best_score
    
    def local_search(start_A, start_score, time_budget):
        nonlocal best_A, best_score
        A = list(start_A)
        score = start_score
        t_start = time.time()
        no_improve_count = 0
        max_no_improve = 100
        
        while time.time() - t_start < time_budget and no_improve_count < max_no_improve:
            # Try different types of moves
            move_type = random.random()
            
            if move_type < 0.4:
                # Replace one element with a nearby value
                i = random.randrange(n)
                old_val = A[i]
                # Try values in a range around old_val
                delta = random.randint(-20, 20)
                new_val = max(1, min(max_val, old_val + delta))
                if new_val != old_val and new_val not in A:
                    A[i] = new_val
                    new_score = objective(A)
                    if new_score < score:
                        score = new_score
                        no_improve_count = 0
                        if score < best_score:
                            best_score = score
                            best_A = list(A)
                    else:
                        A[i] = old_val
                        no_improve_count += 1
                else:
                    no_improve_count += 1
            elif move_type < 0.7:
                # Swap two elements with random values
                i, j = random.sample(range(n), 2)
                old_i, old_j = A[i], A[j]
                A[i] = random.randint(1, max_val)
                A[j] = random.randint(1, max_val)
                if len(set(A)) == n and all(1 <= x <= max_val for x in A):
                    new_score = objective(A)
                    if new_score < score:
                        score = new_score
                        no_improve_count = 0
                        if score < best_score:
                            best_score = score
                            best_A = list(A)
                    else:
                        A[i], A[j] = old_i, old_j
                        no_improve_count += 1
                else:
                    A[i], A[j] = old_i, old_j
                    no_improve_count += 1
            else:
                # Perturb multiple elements slightly
                indices = random.sample(range(n), 3)
                old_vals = [A[i] for i in indices]
                for i in indices:
                    delta = random.randint(-10, 10)
                    A[i] = max(1, min(max_val, A[i] + delta))
                if len(set(A)) == n and all(1 <= x <= max_val for x in A):
                    new_score = objective(A)
                    if new_score < score:
                        score = new_score
                        no_improve_count = 0
                        if score < best_score:
                            best_score = score
                            best_A = list(A)
                    else:
                        for i, v in zip(indices, old_vals):
                            A[i] = v
                        no_improve_count += 1
                else:
                    for i, v in zip(indices, old_vals):
                        A[i] = v
                    no_improve_count += 1
        
        return A, score
    
    # Run local search from multiple starting points
    search_budget = 30  # seconds for local search
    elapsed = time.time() - (deadline - 55)
    remaining = deadline - time.time()
    
    if remaining > 10:
        # Run local search from best candidate
        local_search(current_A, current_score, min(search_budget, remaining - 5))
    
    # If we still have time, try more diverse initializations
    elapsed = time.time() - (deadline - 55)
    remaining = deadline - time.time()
    
    if remaining > 15:
        # Try more structured initializations
        for _ in range(5):
            # Create a set with controlled structure
            A = []
            # Start with a base
            base = random.randint(1, 5)
            A.append(base)
            # Add multiples
            for mult in range(2, n + 1):
                val = base * mult
                if val <= max_val and val not in A:
                    A.append(val)
            # Fill remaining with random
            while len(A) < n:
                val = random.randint(1, max_val)
                if val not in A:
                    A.append(val)
            A = A[:n]
            if validate(A):
                score = objective(A)
                update_best(A, score)
                if remaining > 10:
                    local_search(A, score, min(5, remaining - 5))
                remaining = deadline - time.time()
                if remaining < 5:
                    break
    
    return best_A
