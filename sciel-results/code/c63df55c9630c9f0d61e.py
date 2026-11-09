import time
import random
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    n = 20
    max_val = 10000
    
    # Precompute all pairs (i, j) with i <= j for n=20
    pairs_i = np.array([i for i in range(n) for j in range(i, n)], dtype=np.int32)
    pairs_j = np.array([j for i in range(n) for j in range(i, n)], dtype=np.int32)
    
    def evaluate(A):
        """Compute max(|A+A|, |A*A|) for set A using NumPy vectorization."""
        A_arr = np.array(A, dtype=np.int64)
        sums = A_arr[pairs_i] + A_arr[pairs_j]
        prods = A_arr[pairs_i] * A_arr[pairs_j]
        sumset_size = len(np.unique(sums))
        prodset_size = len(np.unique(prods))
        return max(sumset_size, prodset_size)
    
    def evaluate_full(A):
        """Compute full score details."""
        A_arr = np.array(A, dtype=np.int64)
        sums = A_arr[pairs_i] + A_arr[pairs_j]
        prods = A_arr[pairs_i] * A_arr[pairs_j]
        sumset_size = len(np.unique(sums))
        prodset_size = len(np.unique(prods))
        return max(sumset_size, prodset_size), sumset_size, prodset_size
    
    # Generate initial candidate sets
    def generate_geometric_start():
        """Geometric progression-like set."""
        # Start with 1 and multiply by small integers
        A = [1]
        for _ in range(n - 1):
            # Try to keep numbers smooth and within range
            candidates = []
            for d in range(2, 15):
                v = A[-1] * d
                if v <= max_val:
                    candidates.append(v)
            if not candidates:
                break
            A.append(random.choice(candidates))
        # If we don't have enough, fill with smooth numbers
        while len(A) < n:
            A.append(random.randint(1, 500))
        # Remove duplicates
        seen = set()
        unique_A = []
        for x in A:
            if x not in seen:
                seen.add(x)
                unique_A.append(x)
        return sorted(unique_A[:n])
    
    def generate_arithmetic_start():
        """Arithmetic progression."""
        d = random.randint(1, 10)
        a = random.randint(1, 50)
        A = [a + i * d for i in range(n)]
        # Ensure within bounds
        A = [min(x, max_val) for x in A]
        # Remove duplicates
        seen = set()
        unique_A = []
        for x in A:
            if x not in seen:
                seen.add(x)
                unique_A.append(x)
        while len(unique_A) < n:
            unique_A.append(random.randint(1, max_val))
        return sorted(unique_A[:n])
    
    def generate_smooth_start():
        """Set of smooth numbers."""
        smooths = set()
        for x in range(1, max_val + 1):
            t = x
            for p in [2, 3, 5, 7]:
                while t % p == 0:
                    t //= p
            if t == 1:
                smooths.add(x)
            if len(smooths) > 200:
                break
        smooth_list = list(smooths)
        random.shuffle(smooth_list)
        return sorted(smooth_list[:n])
    
    def generate_random_start():
        """Random set biased toward small numbers."""
        A = []
        while len(A) < n:
            v = random.randint(1, 1000)
            if v not in A:
                A.append(v)
        return sorted(A)
    
    # Generate multiple initial candidates
    initial_sets = []
    initial_sets.append(list(range(1, 21)))
    initial_sets.append(generate_geometric_start())
    initial_sets.append(generate_arithmetic_start())
    initial_sets.append(generate_smooth_start())
    initial_sets.append(generate_random_start())
    
    # Evaluate all initial sets
    best_set = None
    best_score = float('inf')
    
    for A in initial_sets:
        if len(set(A)) != n:
            continue
        score = evaluate(A)
        if score < best_score:
            best_score = score
            best_set = A[:]
    
    # Simulated Annealing with multiple restarts
    num_restarts = 5
    current_restarts = 0
    
    while current_restarts < num_restarts and time.time() < deadline:
        # Start from best found so far or a new random start
        if current_restarts == 0:
            current_set = best_set[:]
        else:
            current_set = generate_random_start()
        
        current_score = evaluate(current_set)
        
        temperature = 30.0
        cooling_rate = 0.985
        min_temp = 0.5
        
        while time.time() < deadline and temperature > min_temp:
            idx = random.randint(0, n - 1)
            old_val = current_set[idx]
            
            # Perturbation strategies
            r = random.random()
            if r < 0.5:
                # Local perturbation: change by small delta
                delta = random.randint(-20, 20)
                new_val = old_val + delta
            elif r < 0.8:
                # Moderate perturbation
                new_val = random.randint(1, 2000)
            else:
                # Large perturbation
                new_val = random.randint(1, max_val)
            
            new_val = max(1, min(max_val, new_val))
            
            if new_val == old_val:
                continue
            if new_val in current_set:
                continue
            
            current_set[idx] = new_val
            new_score = evaluate(current_set)
            
            delta_score = new_score - current_score
            
            if delta_score < 0 or random.random() < math.exp(-delta_score / temperature):
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_set = current_set[:]
            else:
                current_set[idx] = old_val
            
            temperature *= cooling_rate
            
            # Check time more frequently
            if time.time() > deadline - 5:
                break
        
        current_restarts += 1
    
    # Final local search polish
    if time.time() < deadline - 10:
        current_set = best_set[:]
        current_score = evaluate(current_set)
        
        # Greedy local search
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for idx in range(n):
                old_val = current_set[idx]
                for new_val in range(max(1, old_val - 30), min(max_val, old_val + 30) + 1):
                    if new_val == old_val or new_val in current_set:
                        continue
                    current_set[idx] = new_val
                    new_score = evaluate(current_set)
                    if new_score < current_score:
                        current_score = new_score
                        if current_score < best_score:
                            best_score = current_score
                            best_set = current_set[:]
                        improved = True
                    else:
                        current_set[idx] = old_val
    
    return sorted(best_set)
