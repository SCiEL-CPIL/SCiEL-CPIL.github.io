"""Fresh design: MIS on precomputed conflict graph for reciprocal-subsum-free sets."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from itertools import combinations

_START_TIME = None
_DEADLINE = 85.0  # safety margin under 90s wall-clock

def _deadline_ok():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return (time.time() - _START_TIME) < _DEADLINE

def _is_reciprocal_subsum(u, others):
    """Check if 1/u equals sum of reciprocals of some non-empty subset of others."""
    target = Fraction(1, u)
    # others is a list of ints (the selected elements excluding u)
    # We need: exists non-empty subset S of others such that sum(1/x for x in S) == target
    # Use DP / subset enumeration with pruning
    n = len(others)
    # Precompute reciprocals
    recips = [Fraction(1, x) for x in others]
    
    # For small n, enumerate all subsets
    if n <= 20:
        # Enumerate all non-empty subsets
        # Use meet-in-the-middle for larger n
        if n <= 16:
            # Direct enumeration
            for mask in range(1, 1 << n):
                s = Fraction(0, 1)
                for i in range(n):
                    if mask & (1 << i):
                        s += recips[i]
                        if s > target:
                            break
                if s == target:
                    return True
            return False
        else:
            # Meet in the middle
            half = n // 2
            left = recips[:half]
            right = recips[half:]
            
            left_sums = set()
            for mask in range(1 << half):
                s = Fraction(0, 1)
                for i in range(half):
                    if mask & (1 << i):
                        s += left[i]
                left_sums.add(s)
            
            for mask in range(1 << (n - half)):
                s = Fraction(0, 1)
                for i in range(n - half):
                    if mask & (1 << i):
                        s += right[i]
                need = target - s
                if need in left_sums:
                    # Check that the subset is non-empty overall
                    # If mask==0 and need==0, that's empty subset -> skip
                    if mask == 0 and need == Fraction(0, 1):
                        continue
                    return True
            return False
    else:
        # For larger sets, use a smarter approach: 
        # The sum of reciprocals of a subset must equal 1/u.
        # Since all elements are >= 1, reciprocals are <= 1.
        # Use a bounded search
        # Actually for our problem n <= 24, but after excluding u, n <= 23
        # Let's use meet-in-the-middle for up to 23
        half = n // 2
        left = recips[:half]
        right = recips[half:]
        
        left_sums = set()
        for mask in range(1 << half):
            s = Fraction(0, 1)
            for i in range(half):
                if mask & (1 << i):
                    s += left[i]
            left_sums.add(s)
        
        for mask in range(1 << (n - half)):
            s = Fraction(0, 1)
            for i in range(n - half):
                if mask & (1 << i):
                    s += right[i]
            need = target - s
            if need in left_sums:
                if mask == 0 and need == Fraction(0, 1):
                    continue
                return True
        return False

def _build_conflict_graph(universe):
    """Build conflict graph: edge (u,v) if 1/u is subsum of reciprocals of (universe \\ {u}) 
       but specifically the conflict is: for a candidate set S, u conflicts with the set if
       1/u = sum of reciprocals of a subset of S\\{u}.
       
       For MIS formulation: we want a set S such that for every u in S, 1/u is NOT a subsum
       of reciprocals of S\\{u}.
       
       This is NOT simply a pairwise conflict graph because the conflict depends on the 
       specific subset of OTHER selected elements. However, we can define:
       u "attacks" v if 1/u = 1/v (i.e., u=v, impossible since distinct) OR
       more generally, we need a hypergraph.
       
       Alternative: Use a different formulation. Define conflict(u, S) where S is the set of
       other elements. For MIS, we need pairwise conflicts. Let's define:
       u and v are in conflict if there exists a subset T of [1,24]\\{u,v} such that
       1/u = 1/v + sum(1/t for t in T). But this still depends on T.
       
       Better approach: Since the constraint is "for the final set S, no u in S has 1/u 
       equal to sum of reciprocals of a subset of S\\{u}", we can use a greedy/branch-and-bound
       approach directly on the set, checking validity incrementally.
       
       Let me reformulate: Use branch-and-bound where we build the set element by element,
       and at each step check if adding a new element would create a subsum conflict with
       any existing element.
    """
    pass

def _check_valid(S):
    """Check if set S is reciprocal-subsum-free."""
    S = set(S)
    for u in S:
        others = [x for x in S if x != u]
        if _is_reciprocal_subsum(u, others):
            return False
    return True

def _greedy_add(universe):
    """Greedy addition: add elements one by one if the set remains valid."""
    S = []
    # Try adding in various orders
    for order in [sorted(universe, reverse=True), sorted(universe), 
                  sorted(universe, key=lambda x: Fraction(1,x), reverse=True)]:
        S = []
        for x in order:
            if _check_valid(S + [x]):
                S.append(x)
        if len(S) > 0:
            break
    return S

def _branch_and_bound(universe, deadline_check=None):
    """Branch and bound to find maximum valid subset."""
    universe = sorted(universe, reverse=True)
    best = []
    n = len(universe)
    
    def _valid_set(S):
        S_set = set(S)
        for u in S_set:
            others = [x for x in S_set if x != u]
            if _is_reciprocal_subsum(u, others):
                return False
        return True
    
    def _rec(S, remaining):
        nonlocal best
        if deadline_check and not deadline_check():
            return
        if len(remaining) == 0:
            if len(S) > len(best):
                best = list(S)
            return
        
        # Upper bound: len(S) + len(remaining)
        if len(S) + len(remaining) <= len(best):
            return
        
        # Try adding remaining[0]
        # Check if adding remaining[0] to S is valid
        new_S = S + [remaining[0]]
        if _valid_set(new_S):
            _rec(new_S, remaining[1:])
        
        # Try not adding remaining[0]
        _rec(S, remaining[1:])
    
    _rec([], universe)
    return best

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    universe = list(range(1, 25))
    
    # Start with a greedy solution
    best = _greedy_add(universe)
    
    # Try to improve with local search / branch and bound
    # For the full problem, branch and bound on 24 elements is expensive.
    # Let's use a smarter approach: 
    # 1. Get a good initial solution via greedy
    # 2. Try to add elements from the complement
    # 3. Try to swap elements
    
    best_set = set(best)
    
    # Try adding elements not in best
    for x in universe:
        if x not in best_set:
            if _check_valid(list(best_set | {x})):
                best_set.add(x)
    
    # Try swapping: remove one element, add another
    improved = True
    while improved and _deadline_ok():
        improved = False
        current = list(best_set)
        for i, rem in enumerate(current):
            for add in universe:
                if add in best_set:
                    continue
                new_set = set(best_set)
                new_set.discard(rem)
                new_set.add(add)
                if len(new_set) > len(best_set) and _check_valid(list(new_set)):
                    best_set = new_set
                    improved = True
                    break
            if improved:
                break
    
    # Also try removing one element and adding two
    if _deadline_ok():
        current = list(best_set)
        for rem in current:
            if not _deadline_ok():
                break
            new_set = set(best_set)
            new_set.discard(rem)
            for add1 in universe:
                if add1 in new_set:
                    continue
                for add2 in universe:
                    if add2 <= add1 or add2 in new_set:
                        continue
                    candidate = new_set | {add1, add2}
                    if len(candidate) > len(best_set) and _check_valid(list(candidate)):
                        best_set = candidate
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    
    result = sorted(best_set)
    if not result:
        result = [13]
    return result
# EVOLVE-BLOCK-END
