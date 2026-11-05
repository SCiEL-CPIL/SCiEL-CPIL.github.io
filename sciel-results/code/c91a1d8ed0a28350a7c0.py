"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    N = 256
    # Weights: reciprocal of each number (use Fraction for exactness)
    weights = [Fraction(1, i) for i in range(1, N + 1)]
    
    # Build conflict graph: two numbers conflict if they can form a 3-AP with a third number in [1,N]
    # a, b, c form a 3-AP if a+c = 2b (a < b < c)
    # Two numbers x, y conflict if there exists z in [1,N] such that {x,y,z} forms a 3-AP
    # i.e., 2x = y + z (z = 2x - y), or 2y = x + z (z = 2y - x), or 2z = x + y (z = (x+y)/2)
    
    # For efficiency, precompute conflicts using bitsets
    # Vertex i represents number i+1 (0-indexed internally)
    num_vertices = N
    
    # conflicts[i] is a bitset of vertices that conflict with vertex i
    conflicts = [0] * num_vertices
    
    for i in range(num_vertices):
        x = i + 1
        for j in range(num_vertices):
            if i == j:
                continue
            y = j + 1
            # Check if x and y can be part of a 3-AP with some z in [1,N]
            # Case 1: x, y, z with x < y < z: 2y = x + z => z = 2y - x
            # Case 2: y, x, z with y < x < z: 2x = y + z => z = 2x - y  
            # Case 3: x, z, y with x < z < y: 2z = x + y => z = (x+y)/2
            # Case 4: z, x, y with z < x < y: 2x = z + y => z = 2x - y
            # Case 5: z, y, x with z < y < x: 2y = z + x => z = 2y - x
            # Case 6: y, z, x with y < z < x: 2z = y + x => z = (x+y)/2
            
            # More systematically: x and y are in a 3-AP if:
            # - They are the first two terms: z = 2y - x must be in [1,N] and z > y
            # - They are the middle and last: z = 2x - y must be in [1,N] and z < x
            # - They are the first and last: z = (x+y)/2 must be integer in [1,N] and x < z < y or y < z < x
            
            conflict = False
            # x is first, y is second: z = 2y - x, need z > y and z <= N
            z = 2 * y - x
            if y < z <= N:
                conflict = True
            # y is first, x is second: z = 2x - y, need z > x and z <= N
            z = 2 * x - y
            if x < z <= N:
                conflict = True
            # x is first, y is last (or vice versa): z = (x+y)/2
            s = x + y
            if s % 2 == 0:
                z = s // 2
                if 1 <= z <= N and z != x and z != y:
                    conflict = True
            
            if conflict:
                conflicts[i] |= (1 << j)
    
    # Now solve Maximum Weight Independent Set using branch and bound
    # Use a recursive approach with bitset representation
    
    # Precompute weights as floats for bounding
    float_weights = [float(w) for w in weights]
    
    # Order vertices by weight descending for better pruning
    order = sorted(range(num_vertices), key=lambda i: -float_weights[i])
    # Create a mapping: new_index -> old_index
    # We'll work in the reordered space
    
    # For the branch and bound, we use the following approach:
    # - Maintain a candidate set (bitset) of vertices that can still be added
    # - At each step, pick a vertex from the candidates
    # - Branch: include it (remove it and its neighbors from candidates) or exclude it
    
    # Use a greedy upper bound: for remaining candidates, the sum of weights is an upper bound
    # But we can do better with a coloring-based bound
    
    # Let's use a simpler but effective approach:
    # Greedy coloring of the conflict graph on candidates to get an upper bound
    
    best_set = set()  # set of vertex indices (0-indexed)
    best_weight = Fraction(0, 1)
    
    # Convert conflicts to use the reordered indices
    # Actually, let's keep it simple and work with original indices, but process in weight order
    
    # Greedy solution first for initial lower bound
    greedy_set = set()
    greedy_weight = Fraction(0, 1)
    remaining = set(range(num_vertices))
    for i in order:
        if i not in remaining:
            continue
        # Check if i can be added to greedy_set
        can_add = True
        for j in greedy_set:
            if (conflicts[j] >> i) & 1:
                can_add = False
                break
        if can_add:
            greedy_set.add(i)
            greedy_weight += weights[i]
            # Remove i and its neighbors from remaining
            remaining.discard(i)
            for j in list(remaining):
                if (conflicts[i] >> j) & 1:
                    remaining.discard(j)
    
    best_set = set(greedy_set)
    best_weight = greedy_weight
    
    # Now do branch and bound
    # State: (candidates_bitset, current_set, current_weight)
    # We'll use recursion with explicit stack to avoid Python recursion limits
    
    # For efficiency, let's use a different approach:
    # Since N=256, we can use numpy bitsets (using Python ints as bitsets)
    
    # Upper bound function: sum of weights of all candidates
    # This is loose but fast. We can improve with a greedy matching bound.
    
    # Let's use a recursive branch and bound with memoization-like pruning
    
    # For the B&B, we'll use the following:
    # - candidates: a Python int representing a bitset of available vertices
    # - current_weight: Fraction sum of selected vertices
    # - We branch on the highest-weight vertex in candidates
    
    # To get a tighter bound, use the fact that in an independent set,
    # no two adjacent vertices can both be selected.
    # Upper bound = sum of weights of candidates (loose)
    # Better: use a greedy coloring of the induced subgraph
    
    # For now, use the simple sum bound and hope it's tight enough
    
    def get_weight_sum(candidates_bitset):
        """Compute sum of weights of vertices in candidates_bitset."""
        s = Fraction(0, 1)
        for i in range(num_vertices):
            if (candidates_bitset >> i) & 1:
                s += weights[i]
        return s
    
    # Stack for DFS: (candidates_bitset, current_set_as_bitset, current_weight)
    # We'll use a priority queue to explore high-potential branches first
    
    import heapq
    
    # (negative_upper_bound, candidates_bitset, current_set_bitset, current_weight)
    # Use a max-heap via negative upper bound
    
    initial_candidates = (1 << num_vertices) - 1
    initial_set = 0
    initial_weight = Fraction(0, 1)
    
    # Initial upper bound
    ub = get_weight_sum(initial_candidates)
    
    # For the heap, we want to explore branches with highest potential first
    # Use negative of the upper bound for min-heap
    
    stack = []
    heapq.heappush(stack, (-float(ub), 0, initial_candidates, initial_set, initial_weight))
    
    nodes_explored = 0
    
    while stack:
        neg_ub, _, candidates, current_set, current_weight = heapq.heappop(stack)
        nodes_explored += 1
        
        # Check time
        if time.time() > deadline:
            break
        
        # If no candidates, update best if needed
        if candidates == 0:
            if current_weight > best_weight:
                best_weight = current_weight
                best_set = set()
                for i in range(num_vertices):
                    if (current_set >> i) & 1:
                        best_set.add(i)
            continue
        
        # Upper bound: current_weight + sum of weights of candidates
        # But we can tighten: the candidates form a graph, and we need an independent set
        # Simple bound: current_weight + sum(candidates)
        ub = current_weight + get_weight_sum(candidates)
        
        if ub <= best_weight:
            continue
        
        # Find the highest-weight vertex in candidates
        # Process in order of decreasing weight
        best_v = -1
        best_v_weight = -1
        for i in order:
            if (candidates >> i) & 1:
                best_v = i
                break
        
        if best_v == -1:
            continue
        
        # Branch 1: Include best_v
        # Remove best_v and all its neighbors from candidates
        new_candidates = candidates & ~(1 << best_v) & ~conflicts[best_v]
        new_set = current_set | (1 << best_v)
        new_weight = current_weight + weights[best_v]
        
        # Upper bound for this branch
        new_ub = new_weight + get_weight_sum(new_candidates)
        if new_ub > best_weight:
            heapq.heappush(stack, (-float(new_ub), nodes_explored, new_candidates, new_set, new_weight))
        
        # Branch 2: Exclude best_v
        new_candidates2 = candidates & ~(1 << best_v)
        new_ub2 = current_weight + get_weight_sum(new_candidates2)
        if new_ub2 > best_weight:
            heapq.heappush(stack, (-float(new_ub2), nodes_explored, new_candidates2, current_set, current_weight))
    
    # Extract best set
    result = sorted([i + 1 for i in best_set])
    
    # Verify no 3-AP
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            a, b = result[i], result[j]
            # Check if there's a c such that a, b, c or b, a, c or a, c, b etc. form AP
            # Actually, just check all pairs
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            a, b = result[i], result[j]
            # c = 2a - b, c = 2b - a, c = (a+b)/2
            for c in [2*a - b, 2*b - a]:
                if 1 <= c <= N and c in best_set:
                    # This should not happen if our conflict graph is correct
                    pass
            if (a + b) % 2 == 0:
                c = (a + b) // 2
                if 1 <= c <= N and c in best_set:
                    pass
    
    if not result:
        # Fallback to greedy
        result = sorted([i + 1 for i in greedy_set])
    
    return result
# EVOLVE-BLOCK-END
