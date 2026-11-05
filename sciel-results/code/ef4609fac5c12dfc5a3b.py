import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 75  # safety margin before 90s kill
    
    def compute_sizes(A):
        s = set()
        p = set()
        n = len(A)
        for i in range(n):
            for j in range(n):
                s.add(A[i] + A[j])
                p.add(A[i] * A[j])
        return len(s), len(p)
    
    def score(A):
        ss, pp = compute_sizes(A)
        return max(ss, pp)
    
    # Initial construction: use lattice points 2^a * 3^b
    # Pick 20 distinct pairs (a,b) with a,b >= 0, value <= 10000
    lattice = []
    for a in range(0, 14):  # 2^13 = 8192
        for b in range(0, 9):  # 3^8 = 6561
            v = (2 ** a) * (3 ** b)
            if v <= 10000 and v >= 1:
                lattice.append(v)
    lattice = sorted(set(lattice))
    
    # Start with first 20 from lattice
    A = sorted(lattice[:20])
    best_A = list(A)
    best_score = score(best_A)
    
    # Simulated annealing
    T = 50.0
    cooling = 0.9995
    rng = random.Random(42)
    
    while time.time() < deadline and T > 0.01:
        # Pick a random element to replace
        idx = rng.randint(0, 19)
        old_val = A[idx]
        
        # Generate candidate: pick a random lattice point not in A
        candidates = [v for v in lattice if v != old_val and v not in A]
        if not candidates:
            break
        new_val = rng.choice(candidates)
        
        A[idx] = new_val
        new_score = score(A)
        
        if new_score < best_score:
            best_score = new_score
            best_A = list(A)
        else:
            # Accept with probability
            delta = new_score - score(list(A))
            if delta > 0 and rng.random() < (2.718281828 ** (-delta / max(T, 0.01))):
                pass  # accept
            else:
                A[idx] = old_val  # reject
        
        T *= cooling
    
    return sorted(best_A)
