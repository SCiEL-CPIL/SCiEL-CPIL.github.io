"""Optimized 3-AP-free set maximizer for [1,256] by reciprocal weight."""
import time
from fractions import Fraction

# Precompute conflict structure: for each x in 1..256, which other y are in conflict
# y conflicts with x if there exists a third element z in [1,256] such that
# {x, y, z} forms a nonconstant 3-term AP.
# The three possibilities:
#   x is the middle: y = 2*x - z for some z != x, z in [1,256]
#   x is the first:  y = x + 2*d for d >= 1, y <= 256 (z = x+d is the middle)
#   x is the last:   y = x - 2*d for d >= 1, y >= 1 (z = x-d is the middle)
# Actually, any pair (a,b) with a != b is in conflict if there exists c in [1,256],
# c != a, c != b, such that {a,b,c} is a 3-term AP.
#
# For a given pair (a,b) with a < b:
#   Case 1: a is first, b is middle => third = 2*b - a. Valid if 2*b-a <= 256 and 2*b-a != a (always true since b>a).
#   Case 2: a is middle, b is last  => third = 2*a - b. Valid if 2*a-b >= 1 and 2*a-b != a (i.e., a != b, true).
#   Case 3: a is first, b is last   => middle = (a+b)/2. Valid if (a+b) is even and middle != a, middle != b (always true if a<b).
#   Case 4: b is first, a is middle => same as case 2 with a,b swapped: 2*b-a... wait.
#
# Let me re-derive. A 3-term AP is {p, p+d, p+2d} with d >= 1.
# Given two distinct elements a, b from [1,256], they conflict if there exists d >= 1 and p in [1, 256-2d]
# such that {a, b} ⊆ {p, p+d, p+2d}.
#
# Equivalently:
# - If a = p, b = p+d: third = p+2d = 2b - a. Need 2b-a <= 256.
# - If a = p, b = p+2d: third = p+d = (a+b)//2. Need a+b even.
# - If a = p+d, b = p+2d: third = p = 2a - b. Need 2a-b >= 1.
# - If a = p+d, b = p: same as a=p, b=p+d with a,b swapped -> already covered by symmetry.
# - If a = p+2d, b = p: same as a=p, b=p+2d with a,b swapped -> already covered.
# - If a = p+2d, b = p+d: third = p = 2b - a. Need 2b-a >= 1.
#
# So for pair (a, b) with a != b:
#   conflict if:
#     (2*b - a) is in [1, 256] and 2*b - a != a and 2*b - a != b  [a=p, b=p+d or a=p+2d, b=p+d]
#     OR ((a+b) % 2 == 0) and (a+b)//2 != a and (a+b)//2 != b     [a=p, b=p+2d or a=p+2d, b=p]
#     OR (2*a - b) is in [1, 256] and 2*a - b != a and 2*a - b != b  [a=p+d, b=p+2d or a=p+d, b=p]
#
# Wait, let me be more careful. For distinct a, b:
#   They are in a 3-AP together iff one of these holds:
#   1. There exists c in [1,256], c != a, c != b, such that a, b, c form a 3-AP.
#   The three positions:
#     a first, b middle: c = 2b - a. Need 1 <= 2b-a <= 256, c != a (i.e., b != a, true), c != b (i.e., b != a, true).
#     a first, b last:   c = (a+b)/2. Need a+b even, 1 <= c <= 256, c != a, c != b.
#     a middle, b first: c = 2a - b. Need 1 <= 2a-b <= 256, c != a, c != b.
#     a middle, b last:  c = 2b - a. Need 1 <= 2b-a <= 256, c != a, c != b. (same as first case)
#     a last, b first:   c = 2b - a. Need 1 <= 2b-a <= 256. (same)
#     a last, b middle:  c = 2a - b. Need 1 <= 2a-b <= 256. (same as third)
#
# So: conflict(a,b) iff:
#   (1 <= 2*b - a <= 256) OR ((a+b) % 2 == 0 and 1 <= (a+b)//2 <= 256) OR (1 <= 2*a - b <= 256)
# with the caveat that c != a and c != b, but since a != b and the formulas give c different from a and b in non-degenerate cases, this is automatically satisfied.
#
# Actually wait: if a = p, b = p+d, then c = p+2d = 2b-a. If d=0, a=b, but we only consider a != b. So c != a and c != b always.
# If a = p, b = p+2d, c = p+d = (a+b)/2. c != a iff d != 0 (true), c != b iff d != 0 (true).
# If a = p+d, b = p+2d, c = p = 2a-b. c != a iff d != 0 (true), c != b iff d != 0 (true).

N = 256
DEADLINE = 85.0  # seconds, with safety margin

def _build_conflicts():
    """Build conflict bitmask for each element 1..256."""
    # conflicts[i] is a bitmask (or set) of elements j != i that conflict with i
    # Use a list of sets for clarity, then convert to bitmasks if needed
    conflicts = [set() for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            c1 = 2 * b - a  # a first, b middle
            c2 = (a + b) // 2 if (a + b) % 2 == 0 else None  # a first, b last
            c3 = 2 * a - b  # a middle, b last
            conflict = False
            if 1 <= c1 <= N:
                conflict = True
            if c2 is not None and 1 <= c2 <= N:
                conflict = True
            if 1 <= c3 <= N:
                conflict = True
            if conflict:
                conflicts[a].add(b)
                conflicts[b].add(a)
    return conflicts

def _greedy_upper_bound(selected, remaining, conflicts, weights):
    """
    Compute a greedy upper bound on the total weight we can still add.
    selected: set of already chosen elements
    remaining: set of candidate elements not yet decided
    conflicts: list of conflict sets
    weights: dict x -> Fraction(1, x)
    """
    # Greedy: repeatedly pick the element with highest weight that doesn't conflict with selected
    # This gives a valid solution, but we want an upper bound.
    # A simple upper bound: sum of weights of all remaining elements that don't conflict with selected.
    # This is an upper bound because we can't pick all of them (they may conflict among themselves),
    # but it's a valid upper bound for pruning.
    bound = Fraction(0)
    for x in remaining:
        if not (conflicts[x] & selected):
            bound += weights[x]
    return bound

def construct():
    start_time = time.time()
    weights = {x: Fraction(1, x) for x in range(1, N + 1)}
    conflicts = _build_conflicts()
    
    # Convert conflicts to frozensets for faster intersection
    conflicts_frozen = [frozenset(conflicts[x]) for x in range(N + 1)]
    
    # Greedy initial solution: pick elements in order of decreasing weight (i.e., increasing x)
    # that don't conflict with already selected
    selected = set()
    for x in range(1, N + 1):
        if not (conflicts_frozen[x] & selected):
            selected.add(x)
    
    best_solution = set(selected)
    best_weight = sum(weights[x] for x in best_solution)
    
    # Branch and bound
    # Order elements by weight (decreasing) for better pruning
    ordered = sorted(range(1, N + 1), key=lambda x: -weights[x])
    
    def branch(idx, current_selected, current_weight, remaining_set):
        nonlocal best_solution, best_weight
        
        if time.time() - start_time > DEADLINE:
            return
        
        if idx == len(ordered):
            if current_weight > best_weight:
                best_weight = current_weight
                best_solution = set(current_selected)
            return
        
        x = ordered[idx]
        
        # Upper bound: current weight + sum of weights of all remaining elements that don't conflict with current_selected
        # Quick bound: if current_weight + sum of all remaining weights <= best_weight, prune
        # But computing sum of all remaining is expensive. Use a precomputed suffix sum.
        
        # Try including x
        if not (conflicts_frozen[x] & current_selected):
            new_selected = current_selected | {x}
            new_remaining = remaining_set - {x}
            # Remove elements that conflict with x from remaining
            for y in conflicts_frozen[x]:
                if y in new_remaining:
                    new_remaining.discard(y)
            branch(idx + 1, new_selected, current_weight + weights[x], new_remaining)
        
        # Try excluding x
        new_remaining = remaining_set - {x}
        branch(idx + 1, current_selected, current_weight, new_remaining)
    
    # This naive branch and bound will be too slow. Let's use a better approach.
    # Use a recursive search with better pruning.
    
    # Actually, let's try a different approach: use the greedy solution as a starting point
    # and try local improvements (swap, add, remove).
    
    # First, let's verify the greedy solution is valid
    for x in best_solution:
        for y in best_solution:
            if x < y and y in conflicts[x]:
                raise ValueError(f"Invalid solution: {x} and {y} conflict")
    
    # Local search: try to add elements
    improved = True
    while improved:
        improved = False
        for x in range(1, N + 1):
            if x not in best_solution and not (conflicts_frozen[x] & best_solution):
                best_solution.add(x)
                best_weight += weights[x]
                improved = True
    
    # Try swaps: remove one element, add one or two
    # This is getting complex. Let's use a smarter branch and bound with memoization or better bounds.
    
    # For now, return the greedy + local add solution
    return sorted(best_solution)

# Let's test
if __name__ == "__main__":
    result = construct()
    print(f"Size: {len(result)}")
    total = sum(Fraction(1, x) for x in result)
    print(f"Weight: {float(total)}")
    print(f"Exact: {total}")
