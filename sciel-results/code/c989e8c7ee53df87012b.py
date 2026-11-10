"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
from itertools import combinations

def _compute_largest_sidon_subset_size(S):
    """
    Compute the largest subset of S where all unordered pair sums (including diagonals) are unique.
    This is equivalent to finding the maximum independent set in a conflict graph where
    two elements conflict if they form a pair sum that collides with another pair sum.
    
    For a set S, a subset T is "Sidon with diagonals" if for all (i,j) and (k,l) in T x T
    (unordered, allowing i=j), we have i+j != k+l unless {i,j} == {k,l}.
    
    We use exhaustive search since |S| = 12 is small.
    """
    n = len(S)
    if n == 0:
        return 0
    if n == 1:
        return 1
    if n == 2:
        # Two elements a, b. Pairs: (a,a), (a,b), (b,b). Sums: 2a, a+b, 2b.
        # These are distinct iff 2a != a+b, a+b != 2b, 2a != 2b.
        # 2a != a+b => a != b (true since distinct)
        # a+b != 2b => a != b (true)
        # 2a != 2b => a != b (true)
        # So any 2-element set is valid.
        return 2
    
    # For n >= 3, use backtracking to find maximum subset
    # We'll try all subsets in descending order of size
    
    best_size = 0
    
    # Check subsets in descending order of size
    for size in range(n, best_size, -1):
        found = False
        for subset in combinations(S, size):
            if _is_valid_sidon_with_diagonals(set(subset)):
                best_size = size
                found = True
                break
        if found:
            break
    
    return best_size

def _is_valid_sidon_with_diagonals(T):
    """
    Check if T is a Sidon set with diagonals: all unordered pair sums i+j for i,j in T (i<=j)
    are unique.
    """
    if len(T) <= 1:
        return True
    
    sums = set()
    T_list = sorted(T)
    for i in range(len(T_list)):
        for j in range(i, len(T_list)):
            s = T_list[i] + T_list[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _evaluate_candidate(S):
    """Return the largest Sidon-with-diagonals subset size for candidate set S."""
    return _compute_largest_sidon_subset_size(S)

def _simulated_annealing(initial_set, max_iters=50000, start_temp=10.0, end_temp=0.1, cooling=0.9999):
    """
    Use simulated annealing to minimize the largest Sidon-with-diagonals subset size.
    Moves: swap an element in the set with a random element in [0,128].
    """
    current = list(initial_set)
    current_size = _evaluate_candidate(current)
    best = list(current)
    best_size = current_size
    
    temp = start_temp
    
    for it in range(max_iters):
        # Generate a neighbor: swap one element with a random value in [0, 128]
        idx = random.randint(0, len(current) - 1)
        old_val = current[idx]
        new_val = random.randint(0, 128)
        
        if new_val == old_val:
            temp *= cooling
            continue
        
        # Check if new_val is already in the set
        if new_val in current:
            temp *= cooling
            continue
        
        neighbor = list(current)
        neighbor[idx] = new_val
        
        neighbor_size = _evaluate_candidate(neighbor)
        
        delta = neighbor_size - current_size
        
        if delta <= 0 or random.random() < pow(2.718281828, -delta / max(temp, 1e-10)):
            current = neighbor
            current_size = neighbor_size
            
            if current_size < best_size:
                best = list(current)
                best_size = current_size
        
        temp *= cooling
    
    return best, best_size

def construct():
    """
    Return exactly 12 distinct integers in [0, 128] that minimize the largest
    Sidon-with-diagonals subset size.
    
    Strategy: Use simulated annealing to find a set where sum collisions are
    maximally distributed, making it hard to find a large Sidon subset.
    """
    random.seed(42)
    
    # Start with a random initial set
    initial = random.sample(range(0, 129), 12)
    
    # Run simulated annealing
    best_set, best_size = _simulated_annealing(
        initial, 
        max_iters=100000, 
        start_temp=15.0, 
        end_temp=0.01, 
        cooling=0.99995
    )
    
    # Try multiple restarts
    for restart in range(5):
        random.seed(42 + restart * 1000)
        initial = random.sample(range(0, 129), 12)
        best_set_r, best_size_r = _simulated_annealing(
            initial,
            max_iters=50000,
            start_temp=15.0,
            end_temp=0.01,
            cooling=0.99995
        )
        if best_size_r < best_size:
            best_set = best_set_r
            best_size = best_size_r
    
    return best_set
# EVOLVE-BLOCK-END
