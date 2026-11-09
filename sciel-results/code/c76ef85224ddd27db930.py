"""Optimized construction of a 3-AP-free subset of [1,256] maximizing sum of reciprocals."""
import time
from fractions import Fraction

# Deadline guard
_DEADLINE = 90.0
_START_TIME = time.time()

def _elapsed():
    return time.time() - _START_TIME

def _is_3ap_free(S):
    """Check if set S contains no nonconstant 3-term AP."""
    s = set(S)
    for a in s:
        for b in s:
            if a < b:
                c = 2 * b - a
                if c != a and c != b and c in s:
                    return False
    return True

def _reciprocal_sum(S):
    """Exact sum of reciprocals as Fraction."""
    total = Fraction(0, 1)
    for x in S:
        total += Fraction(1, x)
    return total

def _can_add(S, x):
    """Check if adding x to S keeps it 3-AP-free."""
    s = set(S)
    s.add(x)
    # Check all pairs in s where one might be x
    # For 3-term AP a, b, c with a<b<c and a+c=2b:
    # x could be a, b, or c
    for a in S:
        for b in S:
            if a < b:
                c = 2 * b - a
                # Check if (a,b,c) is now a 3-AP with x involved
                if c == x:
                    return False
                if a == x:
                    # Already checked above since x is new
                    pass
            # Also check if x is the middle term
        # x as middle: a + c = 2*x => c = 2*x - a
        c = 2 * x - a
        if c != x and c in S:
            return False
    # x as last term: a + x = 2*b => a = 2*b - x
    for b in S:
        a = 2 * b - x
        if a != x and a in S:
            return False
    return True

def _greedy_construct():
    """Greedy: add elements in order of largest reciprocal (smallest x) that don't create 3-AP."""
    selected = []
    for x in range(1, 257):
        if _can_add(selected, x):
            selected.append(x)
    return selected

def _local_search(S, max_iter=50000):
    """Try to improve by swapping elements or removing/adding."""
    S = list(S)
    S_set = set(S)
    current_sum = _reciprocal_sum(S)
    
    for _ in range(max_iter):
        if _elapsed() > _DEADLINE - 5:
            break
        
        improved = False
        # Try adding each element not in S
        for x in range(1, 257):
            if x not in S_set:
                new_S = S + [x]
                if _is_3ap_free(new_S):
                    new_sum = _reciprocal_sum(new_S)
                    if new_sum > current_sum:
                        S = new_S
                        S_set = set(S)
                        current_sum = new_sum
                        improved = True
                        break
        if improved:
            continue
        
        # Try swapping: remove one element, add another
        for i in range(len(S)):
            if _elapsed() > _DEADLINE - 5:
                return S, current_sum
            for x in range(1, 257):
                if x in S_set:
                    continue
                new_S = [S[j] for j in range(len(S)) if j != i] + [x]
                if _is_3ap_free(new_S):
                    new_sum = _reciprocal_sum(new_S)
                    if new_sum > current_sum:
                        S = new_S
                        S_set = set(S)
                        current_sum = new_sum
                        improved = True
                        break
            if improved:
                break
        if not improved:
            break
    
    return S, current_sum

def _simulated_annealing(S, max_iter=100000):
    """Simulated annealing to escape local optima."""
    import math
    import random
    
    S = list(S)
    S_set = set(S)
    current_sum = _reciprocal_sum(S)
    best_S = list(S)
    best_sum = current_sum
    
    T = 0.1  # initial temperature
    cooling = 0.999
    
    for _ in range(max_iter):
        if _elapsed() > _DEADLINE - 5:
            break
        
        T *= cooling
        if T < 1e-8:
            break
        
        # Random move: add, remove, or swap
        move_type = random.random()
        
        if move_type < 0.3:
            # Add random element
            candidates = [x for x in range(1, 257) if x not in S_set]
            if not candidates:
                continue
            x = random.choice(candidates)
            new_S = S + [x]
        elif move_type < 0.6:
            # Remove random element
            if len(S) <= 1:
                continue
            i = random.randrange(len(S))
            new_S = [S[j] for j in range(len(S)) if j != i]
        else:
            # Swap
            if len(S) == 0:
                continue
            i = random.randrange(len(S))
            candidates = [x for x in range(1, 257) if x not in S_set]
            if not candidates:
                continue
            x = random.choice(candidates)
            new_S = [S[j] for j in range(len(S)) if j != i] + [x]
        
        if not _is_3ap_free(new_S):
            continue
        
        new_sum = _reciprocal_sum(new_S)
        delta = new_sum - current_sum
        
        if delta > 0 or random.random() < math.exp(delta / T):
            S = new_S
            S_set = set(S)
            current_sum = new_sum
            if current_sum > best_sum:
                best_S = list(S)
                best_sum = current_sum
    
    return best_S, best_sum

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    # Start with greedy construction
    S = _greedy_construct()
    current_sum = _reciprocal_sum(S)
    
    # Local search polish
    S, current_sum = _local_search(S, max_iter=20000)
    
    # Simulated annealing for further improvement
    if _elapsed() < _DEADLINE - 10:
        import random
        random.seed(42)
        S, current_sum = _simulated_annealing(S, max_iter=50000)
    
    # Final local search polish
    if _elapsed() < _DEADLINE - 5:
        S, current_sum = _local_search(S, max_iter=10000)
    
    # Ensure we return a valid nonempty subset
    if not S:
        S = [1]
    
    # Verify 3-AP-free
    if not _is_3ap_free(S):
        # Fallback to greedy
        S = _greedy_construct()
    
    return S
