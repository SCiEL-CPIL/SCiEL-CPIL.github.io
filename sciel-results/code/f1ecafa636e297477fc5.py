"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random
import numpy as np

# EVOLVE-BLOCK-START

def _score_of(U):
    """Compute the lower bound score for a set U."""
    U = list(U)
    if not U:
        return 0.0
    n = len(U)
    max_U = max(U)
    if max_U == 0:
        return 0.0
    # Compute sumset U+U and difference set U-U
    U_set = set(U)
    sumset = set()
    diffset = set()
    for a in U:
        for b in U:
            sumset.add(a + b)
            diffset.add(a - b)
    # Ensure 0 is in diffset (it should be since a-a=0)
    # |U-U| is the size of the difference set
    # |U+U| is the size of the sum set
    ratio = len(diffset) / len(sumset)
    if ratio <= 0:
        return 0.0
    denom = 2 * max_U + 1
    log_ratio = math.log(ratio)
    log_denom = math.log(denom)
    if log_denom == 0:
        return 0.0
    score = 1 + log_ratio / log_denom
    return score


def _score_of_fast(U):
    """Faster score computation using numpy for larger sets."""
    U = np.array(U, dtype=np.int64)
    n = len(U)
    max_U = int(U.max())
    if max_U == 0:
        return 0.0
    # Use set-based computation for moderate sizes, numpy for larger
    if n <= 2000:
        U_list = U.tolist()
        sumset = set()
        diffset = set()
        for a in U_list:
            for b in U_list:
                sumset.add(a + b)
                diffset.add(a - b)
        ratio = len(diffset) / len(sumset)
    else:
        # For larger sets, use a more efficient approach
        # Sumset: all pairs a+b
        # Difference set: all pairs a-b
        # We can use convolution-like approaches but sets are simpler
        U_list = U.tolist()
        sumset = set()
        diffset = set()
        for a in U_list:
            sumset.update(a + b for b in U_list)
            diffset.update(a - b for b in U_list)
        ratio = len(diffset) / len(sumset)
    
    if ratio <= 0:
        return 0.0
    denom = 2 * max_U + 1
    log_ratio = math.log(ratio)
    log_denom = math.log(denom)
    if log_denom == 0:
        return 0.0
    score = 1 + log_ratio / log_denom
    return score


def _construct_geometric(n_terms, base):
    """Construct a geometric-like set: {0, 1, base, base^2, ..., base^(n-1)}"""
    if n_terms <= 0:
        return [0]
    U = [0]
    current = 1
    for i in range(1, n_terms):
        U.append(current)
        current *= base
    return sorted(U)


def _construct_mixed(base1, base2, n1, n2):
    """Construct a mixed set with two geometric progressions."""
    U = [0]
    # First progression
    current = 1
    for i in range(1, n1):
        U.append(current)
        current *= base1
    # Second progression starting from a large offset
    offset = current * base1
    current = offset + 1
    for i in range(1, n2):
        U.append(current)
        current = offset + 1 + (current - offset - 1) * base2
    return sorted(U)


def _local_search(U, max_steps=5000, time_limit=10.0):
    """Local search to improve the set."""
    start_time = time.time()
    U = sorted(set(U))
    current_score = _score_of_fast(U)
    best_U = U[:]
    best_score = current_score
    max_val = max(U) if U else 0
    
    for step in range(max_steps):
        if time.time() - start_time > time_limit:
            break
        
        # Randomly modify the set
        r = random.random()
        if r < 0.3 and len(U) > 1:
            # Remove a random element (not 0)
            candidates = [x for x in U if x != 0]
            if candidates:
                idx = random.randint(0, len(candidates) - 1)
                new_U = U[:]
                new_U.remove(candidates[idx])
                new_score = _score_of_fast(new_U)
                if new_score > best_score:
                    best_score = new_score
                    best_U = new_U
                    U = new_U
                    current_score = new_score
                    max_val = max(U) if U else 0
        elif r < 0.6:
            # Add a new element
            if max_val == 0:
                new_val = 1
            else:
                new_val = random.randint(1, max_val * 2)
            if new_val not in U:
                new_U = U + [new_val]
                new_score = _score_of_fast(new_U)
                if new_score > best_score:
                    best_score = new_score
                    best_U = new_U
                    U = new_U
                    current_score = new_score
                    max_val = max(U)
        else:
            # Replace a random element
            candidates = [x for x in U if x != 0]
            if candidates:
                idx = random.randint(0, len(candidates) - 1)
                old_val = candidates[idx]
                new_val = random.randint(1, max_val * 2)
                if new_val not in U:
                    new_U = U[:]
                    new_U[new_U.index(old_val)] = new_val
                    new_score = _score_of_fast(new_U)
                    if new_score > best_score:
                        best_score = new_score
                        best_U = new_U
                        U = new_U
                        current_score = new_score
                        max_val = max(U)
    
    return best_U, best_score


def _simulated_annealing(U, time_limit=15.0):
    """Simulated annealing to improve the set."""
    start_time = time.time()
    U = sorted(set(U))
    current_score = _score_of_fast(U)
    best_U = U[:]
    best_score = current_score
    max_val = max(U) if U else 0
    
    T = 0.1  # Initial temperature
    cooling = 0.99
    
    while time.time() - start_time < time_limit:
        # Generate a neighbor
        r = random.random()
        if r < 0.3 and len(U) > 1:
            candidates = [x for x in U if x != 0]
            if candidates:
                idx = random.randint(0, len(candidates) - 1)
                new_U = U[:]
                new_U.remove(candidates[idx])
        elif r < 0.6:
            if max_val == 0:
                new_val = 1
            else:
                new_val = random.randint(1, max_val * 2)
            if new_val in U:
                continue
            new_U = U + [new_val]
        else:
            candidates = [x for x in U if x != 0]
            if candidates:
                idx = random.randint(0, len(candidates) - 1)
                old_val = candidates[idx]
                new_val = random.randint(1, max_val * 2)
                if new_val in U:
                    continue
                new_U = U[:]
                new_U[new_U.index(old_val)] = new_val
            else:
                continue
        
        new_score = _score_of_fast(new_U)
        delta = new_score - current_score
        
        if delta > 0 or random.random() < math.exp(delta / T):
            U = new_U
            current_score = new_score
            max_val = max(U) if U else 0
            if new_score > best_score:
                best_score = new_score
                best_U = U[:]
        
        T *= cooling
        if T < 0.001:
            T = 0.001
    
    return best_U, best_score


def _generate_random_set(n, max_val):
    """Generate a random set of size n containing 0."""
    if n <= 1:
        return [0]
    elements = [0] + random.sample(range(1, max_val + 1), n - 1)
    return sorted(elements)


def construct():
    """Main construction function that searches for a good set."""
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    best_U = [0, 1, 3]
    best_score = _score_of_fast(best_U)
    
    # Try various initial constructions
    initial_candidates = []
    
    # Geometric sets
    for base in [2, 3, 4, 5]:
        for n_terms in range(3, 15):
            U = _construct_geometric(n_terms, base)
            initial_candidates.append(U)
    
    # Mixed sets
    for base1 in [2, 3]:
        for base2 in [2, 3]:
            for n1 in [3, 5, 7]:
                for n2 in [3, 5, 7]:
                    U = _construct_mixed(base1, base2, n1, n2)
                    initial_candidates.append(U)
    
    # Random sets
    for n in [5, 10, 15, 20, 30]:
        for max_val in [100, 500, 1000, 5000]:
            for _ in range(5):
                U = _generate_random_set(n, max_val)
                initial_candidates.append(U)
    
    # Evaluate all initial candidates
    for U in initial_candidates:
        if time.time() > deadline - 20:
            break
        score = _score_of_fast(U)
        if score > best_score:
            best_score = score
            best_U = U[:]
    
    # Local search on the best candidate
    if time.time() < deadline - 15:
        improved_U, improved_score = _local_search(best_U, max_steps=10000, time_limit=10.0)
        if improved_score > best_score:
            best_score = improved_score
            best_U = improved_U
    
    # Simulated annealing
    if time.time() < deadline - 10:
        improved_U, improved_score = _simulated_annealing(best_U, time_limit=15.0)
        if improved_score > best_score:
            best_score = improved_score
            best_U = improved_U
    
    # Final local search
    if time.time() < deadline - 5:
        improved_U, improved_score = _local_search(best_U, max_steps=5000, time_limit=5.0)
        if improved_score > best_score:
            best_score = improved_score
            best_U = improved_U
    
    return sorted(set(best_U))

# EVOLVE-BLOCK-END
