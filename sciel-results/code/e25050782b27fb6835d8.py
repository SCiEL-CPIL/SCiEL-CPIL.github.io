import itertools
import math
import time
import numpy as np

def _score_fast(U):
    """Fast score computation using numpy for vectorized set operations."""
    U = np.array(sorted(set(U)))
    if 0 not in U:
        return 0.0
    mx = int(U[-1])
    if mx == 0:
        return 0.0
    
    # Compute all sums and differences using broadcasting
    sums = U[:, None] + U[None, :]
    diffs = U[:, None] - U[None, :]
    
    n_plus = len(np.unique(sums))
    n_minus = len(np.unique(diffs))
    
    if n_plus == 0:
        return 0.0
    ratio = n_minus / n_plus
    if ratio <= 1:
        return 1.0
    return 1.0 + math.log(ratio) / math.log(2 * mx + 1)

def _score(U):
    """Original score function for compatibility."""
    U = list(set(U))
    if 0 not in U:
        return 0.0
    mx = max(U)
    if mx == 0:
        return 0.0
    plus = set()
    minus = set()
    for a in U:
        for b in U:
            plus.add(a + b)
            minus.add(a - b)
    n_plus = len(plus)
    n_minus = len(minus)
    if n_plus == 0:
        return 0.0
    ratio = n_minus / n_plus
    if ratio <= 1:
        return 1.0
    return 1.0 + math.log(ratio) / math.log(2 * mx + 1)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    best_score = 0.0
    best_set = [0, 1, 3]
    
    # Phase 1: Exhaustive search for small sizes
    # We search for sets containing 0.
    # Size n means n-1 other elements.
    # We limit the max value to keep the search space manageable.
    max_val_small = 30
    candidates_small = list(range(1, max_val_small + 1))
    
    for size in range(2, 7):
        if time.time() > deadline - 10:
            break
        for combo in itertools.combinations(candidates_small, size - 1):
            if time.time() > deadline - 10:
                break
            U = [0] + list(combo)
            s = _score_fast(U)
            if s > best_score:
                best_score = s
                best_set = U
    
    # Phase 2: Local Search
    # Start from the best set found and try to improve it by adding/removing/swapping elements.
    if time.time() < deadline - 5.0:
        current_set = set(best_set)
        current_score = best_score
        
        # Define a range for new elements to try
        # We can try to extend the set or modify it.
        # To keep max(U) small, we should be careful about adding large numbers.
        # However, adding a large number increases the denominator log(2max+1).
        # It also increases |U-U| and |U+U|.
        # Let's try a range up to 100.
        max_val_large = 100
        candidates_large = list(range(1, max_val_large + 1))
        
        # We will perform a series of local search steps.
        # At each step, we try to add an element, remove an element, or swap an element.
        # We accept the move if it improves the score.
        
        # To make it efficient, we don't re-score the whole set from scratch if possible,
        # but for simplicity and correctness with numpy, we will re-score.
        # The sets are small (size < 20), so numpy is fast.
        
        iterations = 0
        max_iterations = 5000
        
        while time.time() < deadline - 5.0 and iterations < max_iterations:
            iterations += 1
            improved = False
            
            # 1. Try adding a random element
            # Pick a random element from candidates_large not in current_set
            available = [x for x in candidates_large if x not in current_set]
            if available:
                x = np.random.choice(available)
                new_set = current_set | {x}
                s = _score_fast(list(new_set))
                if s > current_score:
                    current_score = s
                    current_set = new_set
                    improved = True
                    if s > best_score:
                        best_score = s
                        best_set = list(current_set)
            
            # 2. Try removing a random element (except 0)
            if len(current_set) > 1:
                removable = [x for x in current_set if x != 0]
                if removable:
                    x = np.random.choice(removable)
                    new_set = current_set - {x}
                    s = _score_fast(list(new_set))
                    if s > current_score:
                        current_score = s
                        current_set = new_set
                        improved = True
                        if s > best_score:
                            best_score = s
                            best_set = list(current_set)
            
            # 3. Try swapping a random element with another
            if len(current_set) > 1:
                removable = [x for x in current_set if x != 0]
                if removable:
                    x = np.random.choice(removable)
                    available = [y for y in candidates_large if y not in current_set]
                    if available:
                        y = np.random.choice(available)
                        new_set = (current_set - {x}) | {y}
                        s = _score_fast(list(new_set))
                        if s > current_score:
                            current_score = s
                            current_set = new_set
                            improved = True
                            if s > best_score:
                                best_score = s
                                best_set = list(current_set)
            
            # If no improvement, we might get stuck.
            # We can occasionally accept a slightly worse move to escape local minima,
            # or just stop if no improvement for a while.
            # For simplicity, we just continue trying random moves.
            # To avoid infinite loops on non-improving moves, we rely on the iteration limit.
            
            # Optional: If stuck, try a larger range or different strategy.
            # For now, just keep trying.
            
    return best_set
