<<<<<<< SEARCH
import time
import random
import math
from itertools import combinations

def construct():
    """
    Try to find the largest n with a valid 3-uniform hypergraph coloring
    where no 4 vertices have all 4 triples red, and no 5 vertices have all 10 triples blue.
    Uses algebraic constructions and local search.
    """
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Try known constructions first
    for n in [34, 33, 32, 31, 30]:
        if time.time() > deadline - 30:
            break
        
        # Try algebraic construction based on cyclic group Z_p
        result = try_algebraic(n)
        if result is not None:
            return {'n': n, 'colors': result}
        
        # Try improved local search with better initialization
        time_remaining = deadline - 5 - time.time()
        if time_remaining > 5:
            result = solve_improved(n, min(time_remaining, 20))
            if result is not None:
                return {'n': n, 'colors': result}
    
    # Fallback: try smaller n with more time
    for n in range(29, 8, -1):
        if time.time() > deadline - 5:
            break
        
        time_remaining = deadline - 5 - time.time()
        if time_remaining > 2:
            result = solve_improved(n, min(time_remaining, 10))
            if result is not None:
                return {'n': n, 'colors': result}
    
    # Guaranteed fallback for small n
    n = 9
    colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]
    return {'n': n, 'colors': colors}
=======
import time
import random
import math
from itertools import combinations

def construct():
    """
    Try to find the largest n with a valid 3-uniform hypergraph coloring
    where no 4 vertices have all 4 triples red, and no 5 vertices have all 10 triples blue.
    Uses structural initialization and targeted local search.
    """
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit
    
    # Try known constructions first
    for n in [34, 33, 32, 31, 30]:
        if time.time() > deadline - 30:
            break
        
        # Try algebraic construction based on cyclic group Z_p
        result = try_algebraic(n)
        if result is not None:
            return {'n': n, 'colors': result}
        
        # Try improved local search with better initialization
        time_remaining = deadline - 5 - time.time()
        if time_remaining > 5:
            result = solve_improved(n, min(time_remaining, 20))
            if result is not None:
                return {'n': n, 'colors': result}
    
    # Fallback: try smaller n with more time
    for n in range(29, 8, -1):
        if time.time() > deadline - 5:
            break
        
        time_remaining = deadline - 5 - time.time()
        if time_remaining > 2:
            result = solve_improved(n, min(time_remaining, 10))
            if result is not None:
                return {'n': n, 'colors': result}
    
    # Guaranteed fallback for small n
    n = 9
    colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]
    return {'n': n, 'colors': colors}
>>>>>>> REPLACE
