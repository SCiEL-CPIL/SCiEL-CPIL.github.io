"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    """
    Find a nonempty subset of [1,256] with no 3-term arithmetic progression
    that maximizes the sum of reciprocals.
    
    Uses branch-and-bound with greedy upper-bound pruning.
    """
    deadline = time.time() + 80  # 80s safety margin before 90s kill
    
    n = 256
    # Weights: 1/x for x in 1..n
    weights = [Fraction(1, i) for i in range(1, n + 1)]
    
    # Precompute conflict structure:
    # For each pair (a, b) with a < b, they conflict if there exists c such that
    # a, b, c form a 3-term AP. This happens when:
    #   - c = 2*b - a (b is middle)
    #   - a = 2*b - c => c = 2*b - a (same)
    #   - b = 2*a - c => c = 2*a - b (a is middle, but a<b so 2*a-b < a < b)
    #   - b = 2*c - a => c = (a+b)/2 (c is middle)
    #   - a = 2*c - b => c = (a+b)/2 (same)
    #   - a = 2*b - c => c = 2*b - a (b is middle)
    # So conflicts arise when:
    #   - 2*b - a is in [1, n] and != a, b (b is middle term)
    #   - 2*a - b is in [1, n] and != a, b (a is middle term, but since a<b, 2a-b < a)
    #   - (a+b) is even and (a+b)//2 is in [1, n] and != a, b (middle term between a and b)
    
    # Actually, let me think more carefully. A 3-term AP (x, y, z) with x < y < z
    # satisfies y - x = z - y, i.e., x + z = 2y.
    # So for any two elements a, b in our set (a < b), they conflict if there exists
    # a third element c such that {a, b, c} forms a 3-term AP.
    # The three cases:
    #   1. b is the middle: c = 2*b - a, need c > b and c <= n
    #   2. a is the middle: c = 2*a - b, need c < a and c >= 1
    #   3. c is the middle: c = (a+b)/2, need a < c < b and a+b even
    
    # For branch-and-bound, I'll use a different approach:
    # Process elements in order of decreasing weight (i.e., increasing x).
    # At each step, decide to include or exclude the current element.
    # Include only if it doesn't create a 3-AP with already-included elements.
    
    # Precompute for each x, which previously-considered elements it would conflict with
    # if both are included. Actually, let's precompute all triples.
    
    # Let's use a cleaner approach:
    # For each element x, precompute the set of elements that conflict with x
    # in the sense that they'd form a 3-AP together with some third element.
    
    # Actually, the cleanest way: for a set S, S is valid iff for no x, y, z in S
    # do we have x + z = 2*y (with x != y != z).
    
    # Let me precompute, for each pair (i, j) with i < j, whether they "conflict"
    # meaning there exists k such that {i, j, k} is a 3-AP.
    
    # But for branch-and-bound, I need to check if adding element x to current set S
    # creates a 3-AP. This happens iff there exist a, b in S such that:
    #   - x + a = 2*b (x and a are ends, b is middle)
    #   - x + b = 2*a (x and b are ends, a is middle)  
    #   - a + b = 2*x (a and b are ends, x is middle)
    
    # Let me precompute conflict pairs more efficiently.
    
    # For each x in 1..n, and for each y in 1..n, y != x:
    #   They conflict if there exists z in 1..n, z != x, z != y, such that
    #   {x, y, z} forms a 3-AP.
    
    # The z values that would complete a 3-AP with x and y:
    #   z1 = 2*y - x (if x < y < z1 or z1 < x < y)
    #   z2 = 2*x - y (if y < x < z2 or z2 < y < x)
    #   z3 = (x+y)//2 if x+y even (if x < z3 < y or y < z3 < x)
    
    # Two elements x, y conflict if any of z1, z2, z3 is in [1, n] and not equal to x or y.
    
    # Precompute conflict[x][y] = True if x and y cannot both be in a valid set
    # (because they'd form a 3-AP with some third element in [1,n]).
    
    conflict = [[False] * (n + 1) for _ in range(n + 1)]
    for x in range(1, n + 1):
        for y in range(x + 1, n + 1):
            # Check if there's a z that completes a 3-AP with x and y
            # z1 = 2*y - x
            z1 = 2 * y - x
            if 1 <= z1 <= n and z1 != x and z1 != y:
                conflict[x][y] = True
                conflict[y][x] = True
                continue
            # z2 = 2*x - y
            z2 = 2 * x - y
            if 1 <= z2 <= n and z2 != x and z2 != y:
                conflict[x][y] = True
                conflict[y][x] = True
                continue
            # z3 = (x+y)/2
            if (x + y) % 2 == 0:
                z3 = (x + y) // 2
                if 1 <= z3 <= n and z3 != x and z3 != y:
                    conflict[x][y] = True
                    conflict[y][x] = True
                    continue
    
    # Order elements by decreasing weight (increasing x)
    order = list(range(1, n + 1))
    # Sort by weight descending: 1/1 > 1/2 > ... so order is already 1, 2, ..., 256
    # But for branch-and-bound, we want to process high-weight elements first.
    # Weight of x is 1/x, so higher weight means smaller x.
    # So order = [1, 2, 3, ..., 256] is already in decreasing weight order.
    
    # Branch and bound
    best_set = []
    best_weight = Fraction(0)
    
    # For upper bound: sum of weights of all remaining elements that don't conflict
    # with the current partial solution.
    
    # Use a recursive approach with pruning
    # State: index in order, current set (as a boolean array), current weight
    # To speed up, use bitmasks or sets.
    
    # Let's use a set for the current solution and track conflicts.
    
    # Actually, let me use a more efficient representation.
    # Since n=256, I can use a 256-bit integer as a bitmask.
    
    # Precompute for each element x, the bitmask of elements that conflict with x.
    conflict_mask = [0] * (n + 1)
    for x in range(1, n + 1):
        mask = 0
        for y in range(1, n + 1):
            if conflict[x][y]:
                mask |= (1 << (y - 1))
        conflict_mask[x] = mask
    
    # Weight as float for quick comparison, Fraction for exact
    weight_float = [0.0] * (n + 1)
    for x in range(1, n + 1):
        weight_float[x] = 1.0 / x
    
    # Process in order of decreasing weight
    # order[i] is the i-th element to consider (0-indexed)
    # order = [1, 2, ..., 256]
    
    # For branch and bound:
    # At position i in order, with current_mask (bitmask of selected elements):
    #   - Option 1: Include order[i] if it doesn't conflict with any selected element
    #   - Option 2: Exclude order[i]
    #   - Upper bound: current_weight + sum of weights of all remaining elements
    #     that don't conflict with current_mask
    
    # To compute upper bound efficiently, precompute suffix sums of weights.
    # But the upper bound should only count elements that don't conflict with current set.
    
    # Let me use a simpler upper bound: sum of all remaining weights (ignoring conflicts).
    # This is valid but loose.
    
    suffix_weight = [0.0] * (n + 2)
    for i in range(n - 1, -1, -1):
        suffix_weight[i] = suffix_weight[i + 1] + weight_float[order[i]]
    
    # Recursive branch and bound
    # Use iterative deepening or just DFS with pruning
    
    # Let me use a stack-based DFS to avoid recursion depth issues
    # State: (index, current_mask, current_weight_float, current_weight_frac, current_list)
    
    # Actually, let's use recursion with sys.setrecursionlimit
    import sys
    sys.setrecursionlimit(10000)
    
    best_set = []
    best_weight_frac = Fraction(0)
    best_weight_float = 0.0
    
    def can_add(x, current_mask):
        """Check if x can be added to the set represented by current_mask."""
        # x conflicts with any selected element if conflict_mask[x] & current_mask != 0
        return (conflict_mask[x] & current_mask) == 0
    
    def dfs(idx, current_mask, current_weight_float, current_weight_frac, current_list):
        nonlocal best_set, best_weight_frac, best_weight_float
        
        if time.time() > deadline:
            return
        
        # Update best if current solution is better
        if current_weight_float > best_weight_float:
            best_weight_float = current_weight_float
            best_weight_frac = current_weight_frac
            best_set = list(current_list)
        
        if idx == n:
            return
        
        # Upper bound: current weight + sum of all remaining weights
        upper_bound = current_weight_float + suffix_weight[idx]
        if upper_bound <= best_weight_float:
            return
        
        x = order[idx]
        
        # Option 1: Include x if possible
        if can_add(x, current_mask):
            new_mask = current_mask | (1 << (x - 1))
            dfs(idx + 1, new_mask, 
                current_weight_float + weight_float[x],
                current_weight_frac + weights[x - 1],
                current_list + [x])
        
        # Option 2: Exclude x
        dfs(idx + 1, current_mask, 
            current_weight_float,
            current_weight_frac,
            current_list)
    
    # Start DFS
    dfs(0, 0, 0.0, Fraction(0), [])
    
    if not best_set:
        # Fallback: just return {1}
        return [1]
    
    return sorted(best_set)

# EVOLVE-BLOCK-END
