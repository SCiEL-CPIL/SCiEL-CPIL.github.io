import random
import time
import math

def construct():
    random.seed(42)
    n = 20
    max_val = 10000
    
    def compute_scores(A):
        sumset = set()
        prodset = set()
        for i in range(n):
            for j in range(i, n):
                sumset.add(A[i] + A[j])
                prodset.add(A[i] * A[j])
        return max(len(sumset), len(prodset))
        
    # Initialize with Geometric Progression
    # 2^19 is 524288, which is > 10000.
    # So we can only have 1, 2, 4, ..., 512 (2^9). That's 10 elements.
    # We need 20 elements.
    # So pure GP with base 2 doesn't fit in 10000 for 20 elements.
    # Max power: 2^13 = 8192. 2^14 = 16384.
    # So {1, 2, ..., 8192} has 14 elements.
    # We need 6 more.
    
    # Let's use a base that allows 20 elements within 10000.
    # b^19 <= 10000 => b <= 10000^(1/19) approx 1.2.
    # So integer base must be 1? No, distinct.
    # So pure integer GP is impossible for 20 elements <= 10000 with base >= 2.
    # 2^0..2^13 is 14 elements.
    # We can extend with 2^14/2? No, must be integer.
    
    # So the "GP" strategy needs to be adapted.
    # Maybe a "Sparse GP" or "Two-Part" set.
    
    # Let's start with the current best known heuristic: 
    # The set that minimizes max(|A+A|, |A*A|) often looks like a "logarithmic" distribution.
    
    # Initialization: 
    # Use a set that spreads out logarithmically.
    # e.g., floor(10000 * (i/19)^2) + 1 ?
    
    # Let's try a random initialization but with a "log-uniform" distribution.
    # And then use Simulated Annealing.
    
    def random_log_uniform():
        # Generate a random integer between 1 and 10000 with log-uniform distribution
        # log(u) * log(10000)
        u = random.random()
        val = int(math.exp(math.log(10000) * u)) + 1
        return val

    A = []
    while len(A) < n:
        v = random_log_uniform()
        if v not in A:
            A.append(v)
    A = sorted(A)
    
    current_score = compute_scores(A)
    best_score = current_score
    best_A = A[:]
    
    start_time = time.time()
    timeout = 55
    
    # Simulated Annealing
    T = 100.0
    cooling = 0.99
    
    while time.time() - start_time < timeout:
        if T < 1e-6:
            break
            
        # Generate a move
        move_type = random.random()
        
        if move_type < 0.5:
            # Replace
            i = random.randint(0, n-1)
            old_val = A[i]
            # Smart candidate generation
            r = random.random()
            if r < 0.3:
                new_val = random.randint(1, max_val)
            elif r < 0.6:
                # Near existing
                j = random.randint(0, n-1)
                delta = random.randint(-100, 100)
                new_val = A[j] + delta
            else:
                # Sum or Product of existing
                i1 = random.randint(0, n-1)
                i2 = random.randint(0, n-1)
                if random.random() < 0.5:
                    new_val = A[i1] + A[i2]
                else:
                    new_val = A[i1] * A[i2]
            
            if new_val < 1 or new_val > max_val:
                continue
            if new_val in A and new_val != old_val:
                continue
                
            A[i] = new_val
            A_sorted = sorted(A)
            new_score = compute_scores(A_sorted)
            
            delta_score = new_score - current_score
            if delta_score < 0 or random.random() < math.exp(-delta_score / T):
                current_score = new_score
                A = A_sorted
                if current_score < best_score:
                    best_score = current_score
                    best_A = A[:]
            else:
                A[i] = old_val
                
        elif move_type < 0.8:
            # Swap
            i, j = random.sample(range(n), 2)
            A[i], A[j] = A[j], A[i]
            A_sorted = sorted(A)
            new_score = compute_scores(A_sorted)
            
            delta_score = new_score - current_score
            if delta_score < 0 or random.random() < math.exp(-delta_score / T):
                current_score = new_score
                A = A_sorted
                if current_score < best_score:
                    best_score = current_score
                    best_A = A[:]
            else:
                A[i], A[j] = A[j], A[i]
        else:
            # Perturb (small change)
            i = random.randint(0, n-1)
            old_val = A[i]
            delta = random.choice([-2, -1, 1, 2])
            new_val = old_val + delta
            if new_val < 1 or new_val > max_val or new_val in A:
                continue
            A[i] = new_val
            A_sorted = sorted(A)
            new_score = compute_scores(A_sorted)
            
            delta_score = new_score - current_score
            if delta_score < 0 or random.random() < math.exp(-delta_score / T):
                current_score = new_score
                A = A_sorted
                if current_score < best_score:
                    best_score = current_score
                    best_A = A[:]
            else:
                A[i] = old_val
                
        T *= cooling
        
    return best_A
