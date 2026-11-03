import time
import random
from math import log
import itertools

def score_set(U):
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return 0.0
    sums = set()
    diffs = set()
    for i in range(n):
        for j in range(i, n):
            sums.add(U[i] + U[j])
        for j in range(i + 1, n):
            d = U[j] - U[i]
            diffs.add(d)
            diffs.add(-d)
    diffs.add(0)
    ratio = len(diffs) / len(sums)
    if ratio <= 1:
        return -1
    max_u = max(U)
    D = 2 * max_u + 1
    if D <= 1:
        return -1
    return 1 + log(ratio) / log(D)

def local_search(U, deadline, max_iter=500):
    current = sorted(set(U))
    current_score = score_set(current)
    if current_score < 0:
        return current, current_score
    
    improved = True
    iteration = 0
    while improved and time.time() < deadline and iteration < max_iter:
        improved = False
        iteration += 1
        n = len(current)
        
        # Strategy 1: Perturb individual elements with wider range
        for i in range(1, n):
            if time.time() > deadline:
                break
            old_val = current[i]
            # Try a wider range of nearby values, including larger jumps
            deltas = list(range(-8, 9)) + [old_val * k for k in [-0.5, -0.3, -0.1, 0.1, 0.3, 0.5]]
            deltas = [int(d) for d in deltas if d != 0 and int(d) != old_val]
            random.shuffle(deltas)
            
            for delta in deltas:
                new_val = old_val + delta
                if new_val <= 0:
                    continue
                new_set = [current[j] if j != i else new_val for j in range(n)]
                new_set = sorted(set(new_set))
                if len(new_set) != n:
                    continue
                new_score = score_set(new_set)
                if new_score > current_score:
                    current = new_set
                    current_score = new_score
                    improved = True
                    break
            if improved:
                break
        
        # Strategy 2: Try adding an element
        if time.time() < deadline and n < 20:
            max_val = max(current)
            candidates_to_try = []
            # Values near max
            for new_val in range(max_val + 1, max_val + 30):
                candidates_to_try.append(new_val)
            # Values between existing elements
            for i in range(1, n):
                gap = current[i] - current[i-1]
                if gap > 3:
                    for pos in range(current[i-1] + 1, current[i]):
                        if random.random() < 0.3:  # Sample some positions
                            candidates_to_try.append(pos)
            # Some random large values
            for _ in range(5):
                candidates_to_try.append(random.randint(max_val + 1, max_val * 2 + 10))
            
            random.shuffle(candidates_to_try)
            for new_val in candidates_to_try:
                if new_val <= 0 or new_val in current:
                    continue
                new_set = sorted(current + [new_val])
                new_score = score_set(new_set)
                if new_score > current_score:
                    current = new_set
                    current_score = new_score
                    improved = True
                    break
        
        # Strategy 3: Try removing an element
        if time.time() < deadline and n > 5:
            for i in range(1, n):
                new_set = [current[j] for j in range(n) if j != i]
                new_score = score_set(new_set)
                if new_score > current_score:
                    current = new_set
                    current_score = new_score
                    improved = True
                    break

    return current, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_set = [0, 1, 4, 9, 11]
    best_score = score_set(best_set)
    
    # Generate diverse base candidates
    candidates = []
    
    # 1. Exponential growth (2^k - 1)
    for k in range(1, 16):
        U = [0]
        val = 1
        for _ in range(k):
            U.append(val)
            val = 2 * val + 1
        candidates.append(U)
        
    # 2. Powers of 2
    for k in range(1, 16):
        U = [0]
        val = 1
        for _ in range(k):
            U.append(val)
            val = 2 * val
        candidates.append(U)
        
    # 3. Fibonacci
    a, b = 0, 1
    U = [0, 1]
    while b < 20000:
        U.append(b)
        a, b = b, a + b
    candidates.append(U)
    
    # 4. Squares
    U = [i*i for i in range(15)]
    candidates.append(U)
    
    # 5. Triangular numbers
    U = [i*(i+1)//2 for i in range(15)]
    candidates.append(U)
    
    # 6. Prime numbers
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True
    U = [0] + [i for i in range(1, 200) if is_prime(i)]
    candidates.append(U)
    
    # 7. Randomized exponential with different seeds
    for seed in [42, 123, 456, 789, 1000, 2000, 3000, 4000]:
        random.seed(seed)
        for _ in range(8):
            U = [0]
            val = random.randint(1, 8)
            for _ in range(random.randint(4, 14)):
                U.append(val)
                val = random.randint(int(val * 1.2), int(val * 3.5))
            candidates.append(U)
    
    # 8. Geometric sequences with various ratios
    for ratio in [1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5, 2.8, 3.0, 3.5, 4.0]:
        U = [0]
        val = 1
        for _ in range(12):
            U.append(val)
            val = int(val * ratio)
            if val > 100000:
                break
        candidates.append(U)
    
    # 9. Sparse random sets
    random.seed(999)
    for _ in range(20):
        n_elem = random.randint(6, 18)
        max_val = random.randint(100, 2000)
        U = [0] + sorted(random.sample(range(1, max_val), n_elem - 1))
        candidates.append(U)
    
    # 10. Mixed growth patterns
    for seed in [11, 22, 33, 44, 55]:
        random.seed(seed)
        U = [0]
        val = random.randint(1, 5)
        for _ in range(random.randint(8, 15)):
            # Alternate between different growth rates
            if random.random() < 0.5:
                val = int(val * random.uniform(1.5, 2.5))
            else:
                val += random.randint(1, max(2, int(val * 0.3)))
            U.append(val)
        candidates.append(U)
    
    # 11. Sets with controlled gaps
    for gap_pattern in [[1, 2, 4, 8, 16], [2, 3, 5, 8, 13], [1, 3, 7, 15, 31], [2, 4, 8, 16, 32]]:
        U = [0]
        val = 0
        for gap in gap_pattern * 3:  # Repeat pattern
            val += gap
            U.append(val)
            if val > 50000:
                break
        candidates.append(U)
    
    # Evaluate and refine
    # Sort candidates by initial score to prioritize better starting points
    scored_candidates = []
    for U in candidates:
        s = score_set(U)
        if s > 0:
            scored_candidates.append((s, U))
    scored_candidates.sort(reverse=True)
    
    # Process top candidates first
    for score, U in scored_candidates[:50]:
        if time.time() > deadline - 20:
            break
        refined, refined_score = local_search(U, deadline)
        if refined_score > best_score:
            best_score = refined_score
            best_set = refined
    
    # If we have time, do additional refinement on the best set
    if time.time() < deadline - 10:
        refined, refined_score = local_search(best_set, deadline)
        if refined_score > best_score:
            best_score = refined_score
            best_set = refined
    
    return best_set
