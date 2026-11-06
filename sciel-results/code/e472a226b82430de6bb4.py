import random
import itertools
import time

def max_sidon_size(S):
    """
    Computes the size of the largest subset of S with all unordered pair sums unique.
    S is a list of 12 distinct integers.
    """
    n = len(S)
    
    def is_sidon(subset):
        sums = set()
        for i in range(len(subset)):
            for j in range(i, len(subset)):
                s = subset[i] + subset[j]
                if s in sums:
                    return False
                sums.add(s)
        return True

    for k in range(n, 0, -1):
        for subset in itertools.combinations(S, k):
            if is_sidon(list(subset)):
                return k
    return 1

def count_violations(S):
    """
    Count the number of quadruples {a,b,c,d} (with a<=b, c<=d, and the pair {a,b} != {c,d})
    such that a+b = c+d. This is a proxy for how "dense" the Sidon violations are.
    More violations means it's harder to find large Sidon subsets.
    """
    pair_sums = {}
    n = len(S)
    for i in range(n):
        for j in range(i, n):
            s = S[i] + S[j]
            if s not in pair_sums:
                pair_sums[s] = 0
            pair_sums[s] += 1
    
    violations = 0
    for s, count in pair_sums.items():
        if count > 1:
            violations += count * (count - 1)
    return violations

def construct():
    # Use stochastic local search to find a set of 12 integers in [0,128]
    # that maximizes the number of Sidon-violating quadruples,
    # which should minimize the largest Sidon subset.
    
    random.seed(42)
    
    def random_set():
        return sorted(random.sample(range(129), 12))
    
    # Initial candidate
    current = random_set()
    current_score = count_violations(current)
    
    best = current[:]
    best_score = current_score
    
    start_time = time.time()
    time_limit = 45  # Leave some margin for the evaluator's computation
    
    # Simulated annealing / stochastic local search
    temperature = 10.0
    cooling = 0.999
    
    while time.time() - start_time < time_limit:
        # Generate a neighbor by swapping one element with a random new value
        idx = random.randint(0, 11)
        old_val = current[idx]
        new_val = random.randint(0, 128)
        
        if new_val in current:
            continue
        
        current[idx] = new_val
        new_score = count_violations(current)
        
        delta = new_score - current_score
        
        # Accept if better, or with some probability if worse
        if delta > 0 or random.random() < pow(2.71828, delta / max(temperature, 0.1)):
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best = current[:]
        else:
            # Revert
            current[idx] = old_val
        
        temperature *= cooling
        if temperature < 0.01:
            # Restart with a new random set
            current = random_set()
            current_score = count_violations(current)
            temperature = 10.0
    
    # Verify the best set is valid
    if len(set(best)) != 12:
        return [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    if min(best) < 0 or max(best) > 128:
        return [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    
    return best
