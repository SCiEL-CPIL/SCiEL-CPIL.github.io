import time
import random
import math

def calc_score(A):
    """
    Compute the score for a set A: max(|A+A|, |A*A|).
    """
    if not A:
        return 0
    A = list(A)
    sumset = set(a + b for a in A for b in A)
    prodset = set(a * b for a in A for b in A)
    return max(len(sumset), len(prodset))

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 90s limit
    
    # Strategy: Simulated Annealing with structural initialization
    # 1. Start with diverse, structured seeds (small integers, geometric, etc.)
    # 2. Use SA to escape local minima by allowing "bad" moves early on
    # 3. Focus on swapping elements rather than just adding/removing to maintain size
    
    def get_initial_set():
        """Generate a diverse initial set of 20 distinct positive integers <= 10000."""
        choice = random.random()
        if choice < 0.3:
            # Small integers
            return sorted(random.sample(range(1, 101), 20))
        elif choice < 0.5:
            # Geometric-like (powers of 2 or 3, scaled)
            base = random.choice([2, 3, 4])
            A = [base ** i for i in range(20)]
            # Scale down if too large
            max_val = max(A)
            if max_val > 10000:
                scale = 10000 / max_val
                A = [max(1, int(a * scale)) for a in A]
                A = sorted(set(A))
                # If we lost elements due to collision, fill with small ints
                while len(A) < 20:
                    new_val = random.randint(1, 10000)
                    if new_val not in A:
                        A.append(new_val)
                A = sorted(A[:20])
            return A
        elif choice < 0.7:
            # Arithmetic progression
            start = random.randint(1, 10)
            diff = random.randint(1, 5)
            A = [start + i * diff for i in range(20)]
            if max(A) > 10000:
                A = sorted(random.sample(range(1, 10000), 20))
            return sorted(A)
        else:
            # Random mix
            return sorted(random.sample(range(1, 10000), 20))

    best_set = None
    best_score = float('inf')
    
    # Run multiple independent SA runs
    num_runs = 10
    
    for run in range(num_runs):
        if time.time() > deadline:
            break
            
        A = get_initial_set()
        current_score = calc_score(A)
        
        # Initialize SA parameters
        temperature = 100.0
        cooling_rate = 0.95
        min_temperature = 0.1
        
        while temperature > min_temperature:
            if time.time() > deadline:
                break
                
            # Generate a neighbor: swap two elements or replace one
            if random.random() < 0.7:
                # Swap two elements
                i, j = random.sample(range(20), 2)
                new_i = random.randint(1, 10000)
                new_j = random.randint(1, 10000)
                # Ensure distinctness
                if new_i in A and new_i != A[i]:
                    continue
                if new_j in A and new_j != A[j]:
                    continue
                if new_i == new_j:
                    continue
                    
                old_i, old_j = A[i], A[j]
                A[i], A[j] = new_i, new_j
                new_score = calc_score(A)
                delta = new_score - current_score
                
                if delta < 0 or random.random() < math.exp(-delta / temperature):
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_set = A[:]
                else:
                    A[i], A[j] = old_i, old_j
            else:
                # Replace one element
                i = random.randint(0, 19)
                new_val = random.randint(1, 10000)
                if new_val in A:
                    continue
                    
                old_val = A[i]
                A[i] = new_val
                new_score = calc_score(A)
                delta = new_score - current_score
                
                if delta < 0 or random.random() < math.exp(-delta / temperature):
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_set = A[:]
                else:
                    A[i] = old_val
                    
            temperature *= cooling_rate
            
        # Local improvement pass (hill climbing) after SA
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(20):
                if time.time() > deadline:
                    break
                old_val = A[i]
                # Try swapping with a random small number
                for _ in range(10):
                    new_val = random.randint(1, 50)
                    if new_val in A:
                        continue
                    A[i] = new_val
                    new_score = calc_score(A)
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                        if current_score < best_score:
                            best_score = current_score
                            best_set = A[:]
                    else:
                        A[i] = old_val
                A.sort()

    # Final fallback
    if not best_set:
        best_set = list(range(1, 21))
        
    # Ensure exactly 20 distinct positive integers <= 10000
    best_set = sorted(list(set(best_set)))
    if len(best_set) < 20:
        # Pad with small numbers
        for i in range(1, 100):
            if i not in best_set:
                best_set.append(i)
            if len(best_set) == 20:
                break
        best_set = sorted(best_set)
    elif len(best_set) > 20:
        best_set = best_set[:20]
        
    return best_set
