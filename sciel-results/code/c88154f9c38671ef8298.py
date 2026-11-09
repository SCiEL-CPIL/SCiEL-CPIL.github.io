"""Program for integer sum-product extremal sets using structured simulated annealing."""
import time
import random
import math

def construct():
    deadline = time.time() + 75.0
    best_set = None
    best_score = float('inf')
    
    def evaluate(A):
        """Compute max(|A+A|, |A*A|) for set A."""
        if len(A) != 20:
            return float('inf')
        if len(set(A)) != 20:
            return float('inf')
        if any(x < 1 or x > 10000 for x in A):
            return float('inf')
            
        A = sorted(A)
        n = len(A)
        sums = set()
        prods = set()
        for i in range(n):
            ai = A[i]
            for j in range(i, n):
                sums.add(ai + A[j])
                prods.add(ai * A[j])
        return max(len(sums), len(prods))

    def generate_initial_set(seed):
        """Generate diverse initial sets."""
        random.seed(seed)
        strategy = random.randint(0, 4)
        
        if strategy == 0:
            # Geometric progression based
            base = random.randint(1, 5)
            ratio = random.uniform(1.1, 2.0)
            A = []
            val = base
            while len(A) < 20 and val <= 10000:
                A.append(int(val))
                val *= ratio
            # Fill remaining with random large numbers to ensure distinctness and range
            while len(A) < 20:
                new_val = random.randint(1, 10000)
                if new_val not in A:
                    A.append(new_val)
            return list(set(A))[:20]
            
        elif strategy == 1:
            # Dense low range (Arithmetic-like)
            start = random.randint(1, 50)
            step = random.randint(1, 10)
            A = [start + i * step for i in range(20)]
            # Ensure distinct and within bounds
            A = list(set(A))
            while len(A) < 20:
                A.append(random.randint(1, 10000))
            return A[:20]
            
        elif strategy == 2:
            # Powers of small primes
            A = []
            for p in [2, 3, 5, 7, 11]:
                val = 1
                while val <= 10000 and len(A) < 20:
                    if val not in A:
                        A.append(val)
                    val *= p
            while len(A) < 20:
                A.append(random.randint(1, 10000))
            return list(set(A))[:20]
            
        elif strategy == 3:
            # Random small numbers
            A = list(random.sample(range(1, 200), 20))
            return A
            
        else:
            # Pure random
            return list(random.sample(range(1, 10001), 20))

    def mutate(A):
        """Perform a smart mutation."""
        A = A.copy()
        idx = random.randint(0, 19)
        old_val = A[idx]
        mut_type = random.randint(0, 3)
        
        if mut_type == 0:
            # Small perturbation
            delta = random.randint(-10, 10)
            new_val = old_val + delta
        elif mut_type == 1:
            # Geometric scaling
            factor = random.uniform(0.8, 1.2)
            new_val = int(old_val * factor)
        elif mut_type == 2:
            # Prime shift
            new_val = old_val + random.choice([2, 3, 5, 7, 11, 13])
        else:
            # Random jump
            new_val = random.randint(1, 10000)
            
        if new_val < 1 or new_val > 10000:
            return None
            
        if new_val in A and new_val != old_val:
            return None
            
        A[idx] = new_val
        return A

    # Multi-start Simulated Annealing
    num_starts = 10
    steps_per_start = 2000
    
    for start in range(num_starts):
        if time.time() > deadline:
            break
            
        current_set = generate_initial_set(start)
        current_score = evaluate(current_set)
        
        if current_score < best_score:
            best_score = current_score
            best_set = current_set.copy()
            
        # Simulated Annealing parameters
        T = 100.0
        T_min = 0.1
        cooling = 0.995
        
        steps = 0
        no_improve = 0
        while steps < steps_per_start and T > T_min and time.time() < deadline:
            steps += 1
            new_set = mutate(current_set)
            if new_set is None:
                continue
                
            new_score = evaluate(new_set)
            delta_score = new_score - current_score
            
            if delta_score < 0 or random.random() < math.exp(-delta_score / T):
                current_set = new_set
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_set = current_set.copy()
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1
                
            T *= cooling
            
            # Early stopping if stuck for too long
            if no_improve > 500:
                break

    return best_set

# EVOLVE-BLOCK-END
