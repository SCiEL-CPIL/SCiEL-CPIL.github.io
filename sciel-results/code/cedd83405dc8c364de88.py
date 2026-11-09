"""Rewritten program with a stronger algorithm for integer sum-product extremal sets."""
import time
import random
import sys

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 75.0  # Safety margin before 90s kill
    best_set = None
    best_score = float('inf')
    
    def evaluate(A):
        """Compute max(|A+A|, |A*A|) for set A."""
        A = sorted(A)
        n = len(A)
        # Sumset A+A (with equal summands allowed)
        sumset = set()
        for i in range(n):
            for j in range(i, n):
                sumset.add(A[i] + A[j])
        # Product set A*A (with equal factors allowed)
        prodset = set()
        for i in range(n):
            for j in range(i, n):
                prodset.add(A[i] * A[j])
        return max(len(sumset), len(prodset))
    
    def divisors(n):
        divs = set()
        i = 1
        while i * i <= n:
            if n % i == 0:
                divs.add(i)
                divs.add(n // i)
            i += 1
        return divs
    
    # Try structured initializations
    initial_sets = []
    
    # Sequential
    initial_sets.append(list(range(1, 21)))
    
    # Geometric progression with ratio 2
    initial_sets.append([2**i for i in range(20)])
    
    # Divisors of highly composite numbers
    for hc in [720, 1260, 2520, 5040, 10080, 15120, 25200, 50400]:
        divs = sorted([d for d in divisors(hc) if d <= 10000])
        if len(divs) >= 20:
            initial_sets.append(divs[:20])
    
    # Lattice-like: products of small primes
    # {p^a * q^b : a,b in {0,1,2,3,4}, p*q <= 10000}
    for p, q in [(2,3), (2,5), (3,5), (2,7), (3,7)]:
        lattice = set()
        for a in range(5):
            for b in range(5):
                val = (p**a) * (q**b)
                if val <= 10000:
                    lattice.add(val)
        if len(lattice) >= 20:
            initial_sets.append(sorted(lattice)[:20])
    
    # Mixed: small numbers + some structured additions
    initial_sets.append(list(range(1, 11)) + [12, 15, 18, 20, 24, 30, 36, 40, 45, 48])
    
    # Evaluate all initial sets
    for A in initial_sets:
        if len(A) != 20 or len(set(A)) != 20:
            continue
        if any(x < 1 or x > 10000 for x in A):
            continue
        score = evaluate(A)
        if score < best_score:
            best_score = score
            best_set = sorted(A)[:]
    
    if best_set is None:
        best_set = list(range(1, 21))
        best_score = evaluate(best_set)
    
    if time.time() > deadline:
        return best_set
    
    # Simulated annealing with nudge/swap moves
    random.seed(42)
    current = best_set[:]
    current_score = best_score
    temperature = 50.0
    cooling = 0.9995
    
    for iteration in range(200000):
        if time.time() > deadline:
            break
        
        move_type = random.random()
        
        if move_type < 0.6:
            # Nudge: change one element by a small amount
            i = random.randint(0, 19)
            old_val = current[i]
            nudge = random.randint(-5, 5)
            new_val = old_val + nudge
            if new_val < 1 or new_val > 10000:
                continue
            if new_val in current:
                continue
            current[i] = new_val
            current.sort()
            new_score = evaluate(current)
            delta = new_score - current_score
            if delta < 0 or random.random() < (1.0 / (1.0 + 0.1 * abs(delta))) * min(1.0, temperature / 10.0):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_set = current[:]
            else:
                current[i] = old_val
                current.sort()
        elif move_type < 0.9:
            # Swap: exchange two elements
            i, j = random.sample(range(20), 2)
            old_i, old_j = current[i], current[j]
            # Try swapping with nearby values
            nudge_i = random.randint(-3, 3)
            nudge_j = random.randint(-3, 3)
            new_i = old_i + nudge_i
            new_j = old_j + nudge_j
            if new_i < 1 or new_i > 10000 or new_j < 1 or new_j > 10000:
                continue
            if new_i in current or new_j in current:
                continue
            if new_i == new_j:
                continue
            current[i] = new_i
            current[j] = new_j
            current.sort()
            new_score = evaluate(current)
            delta = new_score - current_score
            if delta < 0 or random.random() < (1.0 / (1.0 + 0.1 * abs(delta))) * min(1.0, temperature / 10.0):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_set = current[:]
            else:
                current[i] = old_i
                current[j] = old_j
                current.sort()
        else:
            # Large jump: replace one element with a random value
            i = random.randint(0, 19)
            old_val = current[i]
            new_val = random.randint(1, 10000)
            if new_val in current:
                continue
            current[i] = new_val
            current.sort()
            new_score = evaluate(current)
            delta = new_score - current_score
            if delta < 0 or random.random() < (1.0 / (1.0 + 0.1 * abs(delta))) * min(1.0, temperature / 10.0):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_set = current[:]
            else:
                current[i] = old_val
                current.sort()
        
        temperature *= cooling
        if temperature < 0.1:
            temperature = 0.1
    
    return best_set
# EVOLVE-BLOCK-END
