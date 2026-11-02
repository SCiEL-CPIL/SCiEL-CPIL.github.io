import time
import random
import math

def _score(U):
    """Compute the score for a given set U."""
    if not U or U[0] != 0:
        return 0.0
    # Ensure unique and sorted
    U = sorted(list(set(U)))
    n = len(U)
    if n < 2:
        return 0.0
    M = U[-1]
    if M == 0:
        return 0.0
    
    sums = set()
    diffs = set()
    # Optimization: use list comprehension or loops
    for a in U:
        for b in U:
            sums.add(a + b)
            diffs.add(a - b)
    
    num = math.log(len(diffs)) - math.log(len(sums))
    den = math.log(2 * M + 1)
    if den == 0:
        return 0.0
    return 1 + num / den

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_score = 0.0
    best_set = [0, 1, 3]
    
    # Try a range of set sizes
    # We can go up to 30 if time permits, but 20 is likely sufficient for small max U
    for n in range(3, 25):
        # Run a few independent searches for this size
        for trial in range(4):
            # Initialize
            # Strategy: Start with a geometric-like sequence or random
            if trial == 0:
                # Geometric (powers of 2 or 3)
                U = [0]
                val = 1
                for i in range(n-1):
                    U.append(val)
                    val *= 2
            elif trial == 1:
                # Prime-ish gaps
                U = [0]
                val = 0
                # Small primes
                primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89]
                for i in range(n-1):
                    val += primes[i % len(primes)]
                    U.append(val)
            elif trial == 2:
                # Random small gaps
                U = [0]
                val = 0
                for i in range(n-1):
                    val += random.randint(1, 5)
                    U.append(val)
            else:
                # Random larger gaps
                U = [0]
                val = 0
                for i in range(n-1):
                    val += random.randint(1, 15)
                    U.append(val)
            
            current_U = U[:]
            current_score = _score(current_U)
            
            if current_score > best_score:
                best_score = current_score
                best_set = current_U[:]
            
            # Simulated Annealing
            T = 1.0
            T_min = 0.01
            cooling = 0.9995
            no_improve = 0
            
            while time.time() < deadline - 5.0:
                # Pick index
                idx = random.randint(1, len(current_U) - 1)
                
                # Adaptive perturbation
                # If the value is small, use small steps. If large, use larger steps.
                # But keep it simple: mix of small and medium steps
                if current_U[idx] < 20:
                    delta = random.choice([-2, -1, 1, 2, 3])
                else:
                    delta = random.choice([-5, -3, -2, -1, 1, 2, 3, 5, 8])
                
                new_val = current_U[idx] + delta
                
                # Validity checks
                if new_val < 0:
                    continue
                if idx > 0 and new_val <= current_U[idx-1]:
                    continue
                if idx < len(current_U) - 1 and new_val >= current_U[idx+1]:
                    continue
                
                new_U = current_U[:]
                new_U[idx] = new_val
                new_score = _score(new_U)
                
                if new_score > current_score:
                    current_U = new_U
                    current_score = new_score
                    no_improve = 0
                    if current_score > best_score:
                        best_score = current_score
                        best_set = current_U[:]
                else:
                    if T > T_min:
                        prob = math.exp((new_score - current_score) / T)
                        if random.random() < prob:
                            current_U = new_U
                            current_score = new_score
                    no_improve += 1
                    if no_improve > 2000:
                        # Break early if stuck for a long time
                        break
                
                T *= cooling
                if T < T_min:
                    break
        
        # Check time
        if time.time() > deadline - 10:
            break
    
    return best_set
