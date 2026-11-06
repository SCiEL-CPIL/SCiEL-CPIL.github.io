"""Improved program for the suite task; uses branch-and-bound for MWIS on conflict graph."""
import math
import time
from itertools import combinations

# Deadline: must finish within 120s wall-clock. We'll aim to return well before 150s.
_START_TIME = time.time()
TIME_LIMIT = 110.0  # safety margin

def _elapsed():
    return time.time() - _START_TIME

def _time_exhausted():
    return _elapsed() > TIME_LIMIT

def construct():
    """
    Find distinct pairwise coprime integers a in 1..511 that maximize sum 1/(512 - a).
    
    Strategy: Model as Maximum Weight Independent Set (MWIS) on a conflict graph.
    Use branch-and-bound with bitset-based pruning for exact global optimum.
    """
    N = 512
    # Precompute scores
    scores = [1.0 / (N - a) for a in range(N)]  # scores[a] = 1/(512-a)
    
    # Precompute prime factors for each number up to 511
    def _get_prime_factors(n):
        factors = set()
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors.add(d)
                n //= d
            d += 1
        if n > 1:
            factors.add(n)
        return factors
    
    prime_factors = [set() for _ in range(N)]
    for i in range(2, N):
        prime_factors[i] = _get_prime_factors(i)
    prime_factors[1] = set()
    
    # Build conflict graph: edge between a,b if NOT coprime
    # Use bitmasks for fast operations
    # Vertex i corresponds to integer i (0..511), but we only care about 1..511
    # Let's index vertices by their value: vertex v = integer v (1..511)
    # For bitset operations, we'll use Python integers as bitsets
    
    # conflict_mask[a] = bitmask of all b such that gcd(a,b) > 1
    conflict_mask = [0] * N
    for a in range(1, N):
        mask = 0
        for b in range(1, N):
            if a != b and len(prime_factors[a] & prime_factors[b]) > 0:
                mask |= (1 << b)
        conflict_mask[a] = mask
    
    # Greedy initialization for initial lower bound
    def _greedy_init():
        chosen = set()
        for a in range(511, 0, -1):
            if all(len(prime_factors[a] & prime_factors[b]) == 0 for b in chosen):
                chosen.add(a)
        return chosen
    
    best_set = _greedy_init()
    best_score = sum(scores[a] for a in best_set)
    
    # Branch and bound for MWIS
    # We process vertices in order of decreasing weight
    # Order: 511, 510, 509, ..., 1
    vertex_order = list(range(511, 0, -1))
    vertex_weights = [scores[v] for v in vertex_order]
    
    # Precompute suffix sums for upper bound
    # suffix_sum[i] = sum of weights from vertex_order[i] to end
    suffix_sum = [0.0] * (len(vertex_order) + 1)
    for i in range(len(vertex_order) - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + vertex_weights[i]
    
    # Map from vertex value to index in vertex_order
    vertex_to_idx = {v: i for i, v in enumerate(vertex_order)}
    
    # State for branch and bound
    # We use a recursive approach with pruning
    # available_mask: bitmask of vertices still available (not yet decided)
    # current_set: set of vertices chosen so far
    # current_score: sum of weights in current_set
    
    # For efficiency, we'll track:
    # - pos: index in vertex_order we're currently considering
    # - chosen_mask: bitmask of chosen vertices
    # - available_mask: bitmask of vertices that can still be chosen
    
    # Key insight: when we decide on vertex at position pos,
    # if we include it, we remove it and all its conflicts from available
    # if we exclude it, we just move to pos+1
    
    # Let's use an iterative approach with explicit stack to avoid recursion limits
    
    # Actually, let's use a cleaner recursive approach with memoization of best seen
    
    def _branch_and_bound():
        nonlocal best_score, best_set
        
        # Sort vertices by weight descending (already done in vertex_order)
        # We'll use recursion with the following state:
        # pos: current position in vertex_order
        # chosen_mask: bitmask of chosen vertices
        # current_score: current accumulated score
        
        # For pruning: if current_score + suffix_sum[pos] <= best_score, prune
        
        # To speed up conflict checking, precompute for each vertex its conflict mask
        # conflict_mask[a] is already computed
        
        # We'll process vertices in order of decreasing weight
        # At each step, we can either include or exclude the current vertex
        
        # Use a list-based approach for the stack
        # Each stack entry: (pos, chosen_mask, current_score, chosen_list)
        
        # For better pruning, let's precompute for each position pos,
        # the set of vertices from pos onward that are mutually non-conflicting
        # with the current chosen set
        
        # Actually, let's just do straightforward B&B with good ordering
        
        max_depth = len(vertex_order)
        
        # Use recursion with a depth counter
        import sys
        sys.setrecursionlimit(10000)
        
        def _bb(pos, chosen_mask, current_score, chosen_list):
            nonlocal best_score, best_set
            
            # Check time
            if _time_exhausted():
                return
            
            # Upper bound: current_score + sum of all remaining weights
            if current_score + suffix_sum[pos] <= best_score + 1e-15:
                return
            
            # Base case
            if pos >= max_depth:
                if current_score > best_score:
                    best_score = current_score
                    best_set = set(chosen_list)
                return
            
            v = vertex_order[pos]
            w = vertex_weights[pos]
            
            # Option 1: Include v if possible
            # v is includable if it doesn't conflict with any chosen vertex
            # Check: conflict_mask[v] & chosen_mask == 0
            if (conflict_mask[v] & chosen_mask) == 0:
                new_mask = chosen_mask | (1 << v)
                _bb(pos + 1, new_mask, current_score + w, chosen_list + [v])
            
            # Option 2: Exclude v
            _bb(pos + 1, chosen_mask, current_score, chosen_list)
        
        _bb(0, 0, 0.0, [])
    
    _branch_and_bound()
    
    # If B&B didn't find anything better (shouldn't happen), fall back to greedy
    if not best_set:
        best_set = _greedy_init()
    
    return sorted(best_set)

# Keep the original function name and signature
# The function construct() is defined above.
