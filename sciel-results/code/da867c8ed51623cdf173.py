"""Rewritten program: minimize largest Sidon subset (with diagonals) of a 12-set in [0,128]."""
import time
import random
from itertools import combinations

# Deadline guard: evaluator kills after 90s; keep margin.
_DEADLINE = 80.0  # seconds
_start_time = time.time()

def _elapsed():
    return time.time() - _start_time

def _count_sum_collisions(subset):
    """
    Count the number of unordered pairs of unordered pairs (including diagonals)
    that produce the same sum.
    For a set S of size k, we consider all pairs (a, b) with a <= b (i.e., indices i <= j).
    There are k*(k+1)/2 such pairs. We count how many sums are duplicated.
    A Sidon set (with diagonals) has all these sums unique, so collision count = 0.
    """
    sums = []
    for i in range(len(subset)):
        for j in range(i, len(subset)):
            sums.append(subset[i] + subset[j])
    # Count collisions: number of pairs of sums that are equal
    # More efficiently: count duplicates
    from collections import Counter
    cnt = Counter(sums)
    collisions = 0
    for v in cnt.values():
        if v > 1:
            # Number of colliding pairs among v identical sums
            collisions += v * (v - 1) // 2
    return collisions

def _is_sidon(subset):
    """Check if subset has all unordered pair sums (including diagonals) unique."""
    return _count_sum_collisions(subset) == 0

def _largest_sidon_subset_size(S):
    """
    Exhaustively compute the largest subset of S with all unordered pair sums unique (including diagonals).
    S is a list of distinct integers.
    """
    n = len(S)
    # Try subsets from largest size down
    # For n=12, max subset size is 12, but we want to find the maximum k such that some k-subset is Sidon.
    # We can use a branch-and-bound or just check all subsets of size k from k=n down.
    # Since n=12, the number of subsets of size 12 is 1, size 11 is 12, size 10 is 66, etc.
    # Total subsets = 2^12 = 4096, which is very manageable.
    
    # Actually, let's be smart: check from size n down to 1.
    for k in range(n, 0, -1):
        for subset in combinations(S, k):
            if _is_sidon(subset):
                return k
    return 0  # Should never happen since single element is always Sidon

def _evaluate_set(S):
    """Return the largest Sidon subset size for set S."""
    return _largest_sidon_subset_size(S)

def _random_set():
    """Generate a random set of 12 distinct integers in [0, 128]."""
    return random.sample(range(129), 12)

def _local_search(S, max_iters=1000):
    """
    Local search: try to reduce the largest Sidon subset size by modifying elements.
    We use a simulated annealing-like approach with a soft objective.
    """
    current = list(S)
    current_score = _evaluate_set(current)
    best = list(current)
    best_score = current_score
    
    T = 10.0  # Temperature
    cooling = 0.995
    
    for it in range(max_iters):
        if _elapsed() > _DEADLINE - 2:
            break
        
        # Pick a random element to change
        idx = random.randint(0, 11)
        old_val = current[idx]
        new_val = random.randint(0, 128)
        if new_val == old_val:
            continue
        if new_val in current:
            continue  # Must remain distinct
        
        current[idx] = new_val
        new_score = _evaluate_set(current)
        
        # We want to minimize the score
        delta = new_score - current_score
        
        if delta <= 0 or random.random() < pow(2.718281828, -delta / max(T, 1e-9)):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best = list(current)
        else:
            current[idx] = old_val
        
        T *= cooling
    
    return best, best_score

def construct():
    """
    Return exactly 12 distinct integers in [0, 128] that minimize the largest Sidon subset size.
    """
    global _start_time
    _start_time = time.time()
    
    random.seed(42)
    
    best_set = None
    best_score = 13  # Max possible is 12, so 13 is a safe initial upper bound
    
    # Try several random starting points with local search
    n_restarts = 20
    
    for r in range(n_restarts):
        if _elapsed() > _DEADLINE - 5:
            break
        
        S = _random_set()
        score = _evaluate_set(S)
        
        if score < best_score:
            best_score = score
            best_set = list(S)
        
        # If we already found the minimum possible (which is likely small), we can stop early
        # The minimum possible largest Sidon subset for 12 elements... 
        # Actually, for a 12-element set, the largest Sidon subset could be as low as... 
        # Let's just run local search to improve.
        
        if score <= 5:
            # Already quite good, do local search to try to improve further
            improved_set, improved_score = _local_search(S, max_iters=500)
            if improved_score < best_score:
                best_score = improved_score
                best_set = improved_set
        else:
            # Do local search
            improved_set, improved_score = _local_search(S, max_iters=300)
            if improved_score < best_score:
                best_score = improved_score
                best_set = improved_set
    
    if best_set is None:
        best_set = _random_set()
    
    # Final polish: try a few more local search iterations from the best set
    if _elapsed() < _DEADLINE - 3:
        improved_set, improved_score = _local_search(best_set, max_iters=500)
        if improved_score < best_score:
            best_score = improved_score
            best_set = improved_set
    
    return best_set
