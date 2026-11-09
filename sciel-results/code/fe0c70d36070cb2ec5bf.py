"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 5000
_START_TIME = time.time()
_TIME_LIMIT = 280  # wall-clock safety margin before 300s kill


def _primes(limit):
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]


def _is_admissible(points, primes_to_check):
    """Check if points form an admissible tuple for the given primes."""
    n = len(points)
    if n == 0:
        return True
    point_set = set(points)
    for p in primes_to_check:
        if p > n:
            break
        # Check if all residue classes mod p are covered
        covered = set(v % p for v in point_set)
        if len(covered) == p:
            return False
    return True


def _greedy_conflict_graph(points, primes_list):
    """
    Greedy approach using conflict graph concept.
    For each prime p <= len(points), we must miss at least one residue class.
    We choose which residue class to miss for each prime, then keep only elements
    that don't fall into any "missed" class.
    
    Strategy: For each prime, track which residue classes are "critical" (i.e.,
    removing them would break admissibility). Use greedy removal of least-populated
    classes, then try to recover elements.
    """
    point_set = set(points)
    current_size = len(point_set)
    
    # For each prime p <= current_size, determine which residue class to remove
    # to maintain admissibility while maximizing remaining set size
    
    # Precompute residue class counts for each prime
    prime_residue_info = {}
    for p in primes_list:
        if p > current_size:
            break
        class_counts = {}
        for v in point_set:
            r = v % p
            class_counts[r] = class_counts.get(r, 0) + 1
        prime_residue_info[p] = class_counts
    
    # Greedy: for each prime in order of increasing p, if all classes covered,
    # remove the least-populated class
    for p in sorted(prime_residue_info.keys()):
        class_counts = prime_residue_info[p]
        if len(class_counts) == p:
            # All classes covered; remove least-populated
            min_class = min(class_counts, key=class_counts.get)
            to_remove = [v for v in point_set if v % p == min_class]
            for v in to_remove:
                point_set.discard(v)
    
    return sorted(point_set)


def _local_search_improve(points, primes_list, max_iterations=10000):
    """
    Local search: try to add back removed elements or swap elements to increase size.
    """
    point_set = set(points)
    all_elements = set(range(N + 1))
    removed = all_elements - point_set
    
    current_size = len(point_set)
    best_size = current_size
    best_set = set(point_set)
    
    # Primes to check based on current size
    def get_primes_for_size(s):
        return [p for p in primes_list if p <= s]
    
    # Try adding removed elements
    improved = True
    iterations = 0
    while improved and iterations < max_iterations:
        improved = False
        iterations += 1
        if time.time() - _START_TIME > _TIME_LIMIT - 15:
            break
        
        primes_to_check = get_primes_for_size(len(point_set))
        
        # Shuffle removed elements for diversity
        removed_list = list(removed)
        random.shuffle(removed_list)
        
        for elem in removed_list:
            if time.time() - _START_TIME > _TIME_LIMIT - 15:
                break
            
            # Try adding elem
            test_set = point_set | {elem}
            test_size = len(test_set)
            
            # Check admissibility
            is_ok = True
            for p in get_primes_for_size(test_size):
                covered = set(v % p for v in test_set)
                if len(covered) == p:
                    is_ok = False
                    break
            
            if is_ok:
                point_set.add(elem)
                removed.discard(elem)
                improved = True
                if len(point_set) > best_size:
                    best_size = len(point_set)
                    best_set = set(point_set)
    
    return sorted(best_set)


def _simulated_annealing(points, primes_list, max_iterations=50000):
    """
    Simulated annealing to explore the space of admissible tuples.
    State: set of elements in [0, N]
    Moves: add/remove/swap elements
    Objective: maximize size while maintaining admissibility
    """
    point_set = set(points)
    all_elements = set(range(N + 1))
    
    def get_primes_for_size(s):
        return [p for p in primes_list if p <= s]
    
    def is_admissible_fast(s):
        size = len(s)
        for p in get_primes_for_size(size):
            covered = set(v % p for v in s)
            if len(covered) == p:
                return False
        return True
    
    current_set = set(point_set)
    best_set = set(point_set)
    best_size = len(point_set)
    
    # Temperature schedule
    T_start = 10.0
    T_end = 0.1
    cooling = (T_end / T_start) ** (1.0 / max_iterations)
    T = T_start
    
    random.seed(42)
    
    for i in range(max_iterations):
        if time.time() - _START_TIME > _TIME_LIMIT - 20:
            break
        
        # Generate a move
        move_type = random.random()
        if move_type < 0.4:
            # Add an element
            if len(current_set) < N + 1:
                candidates = list(all_elements - current_set)
                if candidates:
                    elem = random.choice(candidates)
                    new_set = current_set | {elem}
                    delta = 1  # size increase
                else:
                    continue
            else:
                continue
        elif move_type < 0.7:
            # Remove an element
            if len(current_set) > 0:
                elem = random.choice(list(current_set))
                new_set = current_set - {elem}
                delta = -1  # size decrease
            else:
                continue
        else:
            # Swap: remove one, add another
            if len(current_set) > 0 and len(current_set) < N + 1:
                remove_elem = random.choice(list(current_set))
                candidates = list(all_elements - current_set)
                if candidates:
                    add_elem = random.choice(candidates)
                    new_set = (current_set - {remove_elem}) | {add_elem}
                    delta = 0  # same size
                else:
                    continue
            else:
                continue
        
        # Check if new set is admissible
        if not is_admissible_fast(new_set):
            # Reject invalid moves
            T *= cooling
            continue
        
        # Accept or reject based on simulated annealing
        new_size = len(new_set)
        if delta >= 0 or random.random() < np.exp(delta / T):
            current_set = new_set
            if new_size > best_size:
                best_size = new_size
                best_set = set(current_set)
        
        T *= cooling
    
    return sorted(best_set)


def construct():
    """Construct the largest admissible tuple within [0, N]."""
    primes_N = _primes(N)
    
    # Strategy 1: Greedy conflict graph approach
    points = list(range(N + 1))
    greedy_result = _greedy_conflict_graph(points, primes_N)
    
    # Strategy 2: Local search improvement
    local_result = _local_search_improve(greedy_result, primes_N, max_iterations=5000)
    
    # Strategy 3: Simulated annealing for better exploration
    sa_result = _simulated_annealing(local_result, primes_N, max_iterations=30000)
    
    # Take the best result
    best = max([local_result, sa_result], key=len)
    
    # Final verification
    final_size = len(best)
    primes_final = _primes(final_size)
    for p in primes_final:
        covered = set(v % p for v in best)
        if len(covered) == p:
            # Need to fix: remove least-populated class
            class_counts = {}
            for v in best:
                r = v % p
                class_counts[r] = class_counts.get(r, 0) + 1
            min_class = min(class_counts, key=class_counts.get)
            best = [v for v in best if v % p != min_class]
    
    return best
