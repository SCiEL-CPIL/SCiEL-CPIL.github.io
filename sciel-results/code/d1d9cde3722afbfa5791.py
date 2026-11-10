"""MWIS branch-and-bound on 3-AP conflict graph with bitset acceleration."""
import time
from fractions import Fraction
import numpy as np

N = 256

def _build_conflict_graph():
    """Build conflict bitsets: conflict[i] is a bitset of vertices that conflict with i."""
    # For each vertex i (1..256), find all j such that {i, j, k} forms a 3-AP
    # A 3-AP is (a, a+d, a+2d) with d >= 1
    # For a given i, it can be:
    #   - the smallest: i, i+d, i+2d -> conflicts with i+d and i+2d
    #   - the middle: i-d, i, i+d -> conflicts with i-d and i+d
    #   - the largest: i-2d, i-d, i -> conflicts with i-2d and i-d
    # Actually, for MWIS on the "conflict graph" where edges mean "cannot both be in set":
    # Two vertices x, y conflict if there exists z such that {x,y,z} is a 3-AP.
    # That happens when:
    #   - x + y = 2z for some z (x and y are endpoints, z is middle) -> z = (x+y)/2 must be integer
    #   - x + z = 2y (x smallest, y middle, z largest) -> z = 2y - x
    #   - y + z = 2x (y smallest, x middle, z largest) -> z = 2x - y
    # So x and y conflict if:
    #   - (x + y) is even and 1 <= (x+y)//2 <= N  [they are endpoints of a 3-AP]
    #   - 1 <= 2y - x <= N  [x, y, 2y-x is a 3-AP with y as middle]
    #   - 1 <= 2x - y <= N  [y, x, 2x-y is a 3-AP with x as middle]
    
    # We use 0-indexed internally: vertex i represents number i+1
    conflict = [0] * (N + 1)  # 1-indexed bitsets
    for x in range(1, N + 1):
        cx = 0
        for y in range(1, N + 1):
            if x == y:
                continue
            # Check if x and y are in some 3-AP together
            # Case 1: x and y are endpoints: z = (x+y)/2
            if (x + y) % 2 == 0:
                z = (x + y) // 2
                if 1 <= z <= N and z != x and z != y:
                    cx |= (1 << y)
                    continue
            # Case 2: x smallest, y middle: z = 2y - x
            z = 2 * y - x
            if 1 <= z <= N and z != x and z != y:
                cx |= (1 << y)
                continue
            # Case 3: y smallest, x middle: z = 2x - y
            z = 2 * x - y
            if 1 <= z <= N and z != x and z != y:
                cx |= (1 << y)
                continue
        conflict[x] = cx
    return conflict

# Precompute conflict graph
_conflict = _build_conflict_graph()

def _greedy_fraubound(available_bits, conflict, weights):
    """
    Compute an upper bound on the MWIS weight using greedy fractional relaxation.
    available_bits: int bitmask of available vertices
    conflict: list of conflict bitsets
    weights: list of weights (1-indexed)
    Returns: upper bound (float)
    """
    if available_bits == 0:
        return 0.0
    
    # Greedy: repeatedly pick the vertex with highest weight/degree ratio
    # This is a simplified fractional bound
    bound = 0.0
    avail = available_bits
    # To avoid infinite loops, limit iterations
    max_iter = 64
    for _ in range(max_iter):
        if avail == 0:
            break
        # Find vertex with best weight/density in avail
        best_v = -1
        best_ratio = -1.0
        v = avail
        while v:
            # Extract lowest set bit
            lsb = v & (-v)
            idx = lsb.bit_length()  # 1-indexed
            # Count conflicts in avail
            conf_in_avail = conflict[idx] & avail
            degree = conf_in_avail.bit_count()
            # Ratio: weight / (1 + degree) as a heuristic
            ratio = weights[idx] / (1 + degree)
            if ratio > best_ratio:
                best_ratio = ratio
                best_v = idx
            v ^= lsb
        
        if best_v == -1:
            break
        
        bound += weights[best_v]
        # Remove best_v and all its conflicts from avail
        avail &= ~(1 << best_v)
        avail &= ~conflict[best_v]
    
    # Add sum of remaining weights as a loose bound
    if avail:
        # Sum weights of remaining
        rem_sum = 0.0
        v = avail
        while v:
            lsb = v & (-v)
            idx = lsb.bit_length()
            rem_sum += weights[idx]
            v ^= lsb
        bound += rem_sum
    
    return bound

def _branch_and_bound(conflict, weights, deadline):
    """
    Branch and bound for MWIS.
    conflict: list of conflict bitsets (1-indexed)
    weights: list of weights (1-indexed)
    deadline: time limit
    Returns: best bitset found
    """
    n = N
    all_bits = (1 << (n + 1)) - 2  # bits 1..n set
    
    best_weight = 0.0
    best_bits = 0
    
    # Sort vertices by weight descending for branching order
    # We'll use a recursive DFS with pruning
    # To avoid recursion depth issues, use iterative or limit depth
    
    # Precompute sorted order by weight
    order = list(range(1, n + 1))
    order.sort(key=lambda x: -weights[x])
    
    # State: (current_bits, available_bits, current_weight, index_in_order)
    # We branch on each vertex in order: include or exclude
    
    # For efficiency, use a stack-based approach
    # But with 256 vertices, even with pruning, this could be huge
    # Let's use a simpler approach: recursive with memoization-like pruning
    
    # Actually, for 256 vertices, exact B&B might be too slow.
    # Let's use a hybrid: B&B on a reduced problem or use it as a local improvement
    
    # Alternative: Use B&B with aggressive pruning
    # Upper bound: sum of weights of all available vertices (loose)
    # Better: greedy fractional bound
    
    # Given time constraints, let's do a limited B&B
    # We'll branch on vertices in weight order, with strong pruning
    
    def dfs(avail, current_weight, current_bits, depth):
        nonlocal best_weight, best_bits
        
        if time.time() > deadline:
            return
        
        if avail == 0:
            if current_weight > best_weight:
                best_weight = current_weight
                best_bits = current_bits
            return
        
        # Upper bound
        ub = current_weight
        v = avail
        while v:
            lsb = v & (-v)
            idx = lsb.bit_length()
            ub += weights[idx]
            v ^= lsb
        
        if ub <= best_weight:
            return
        
        # Greedy fractional bound for tighter pruning
        # But this is expensive; use it selectively
        if depth < 30:
            gb = _greedy_fraubound(avail, conflict, weights)
            if current_weight + gb <= best_weight:
                return
        
        # Pick a vertex to branch on
        # Choose the one with highest weight in avail
        best_v = -1
        best_w = -1.0
        v = avail
        while v:
            lsb = v & (-v)
            idx = lsb.bit_length()
            if weights[idx] > best_w:
                best_w = weights[idx]
                best_v = idx
            v ^= lsb
        
        if best_v == -1:
            return
        
        # Branch 1: Include best_v
        new_avail = avail & ~(1 << best_v) & ~conflict[best_v]
        dfs(new_avail, current_weight + weights[best_v], current_bits | (1 << best_v), depth + 1)
        
        if time.time() > deadline:
            return
        
        # Branch 2: Exclude best_v
        new_avail2 = avail & ~(1 << best_v)
        dfs(new_avail2, current_weight, current_bits, depth + 1)
    
    dfs(all_bits, 0.0, 0, 0)
    
    return best_bits, best_weight

def _local_search(bits, conflict, weights, deadline):
    """
    Local search: try to improve by adding/removing/swapping vertices.
    bits: current solution bitset
    Returns: improved bitset
    """
    current_weight = 0.0
    v = bits
    while v:
        lsb = v & (-v)
        idx = lsb.bit_length()
        current_weight += weights[idx]
        v ^= lsb
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try adding each vertex not in set
        for x in range(1, N + 1):
            if time.time() > deadline:
                break
            if bits & (1 << x):
                continue
            if bits & conflict[x]:
                continue
            # Can add x
            new_weight = current_weight + weights[x]
            if new_weight > current_weight:
                bits |= (1 << x)
                current_weight = new_weight
                improved = True
        
        # Try removing one and adding another
        for x in range(1, N + 1):
            if time.time() > deadline:
                break
            if not (bits & (1 << x)):
                continue
            # Remove x
            bits_without_x = bits & ~(1 << x)
            # Try adding each y not in bits_without_x
            for y in range(1, N + 1):
                if time.time() > deadline:
                    break
                if bits_without_x & (1 << y):
                    continue
                if bits_without_x & conflict[y]:
                    continue
                new_weight = current_weight - weights[x] + weights[y]
                if new_weight > current_weight:
                    bits = bits_without_x | (1 << y)
                    current_weight = new_weight
                    improved = True
                    break  # Restart the removal loop
    
    return bits, current_weight

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    # Weights: reciprocal of the number
    weights = [0.0] * (N + 1)
    for i in range(1, N + 1):
        weights[i] = 1.0 / i
    
    # Phase 1: Greedy initialization
    # Sort by weight descending (smallest numbers first)
    order = list(range(1, N + 1))
    order.sort(key=lambda x: -weights[x])
    
    greedy_bits = 0
    for x in order:
        if time.time() > deadline - 20:
            break
        if not (greedy_bits & conflict[x]):
            greedy_bits |= (1 << x)
    
    # Phase 2: Branch and bound (limited)
    bnb_deadline = deadline - 10  # Leave 10s for local search
    bnb_bits, bnb_weight = _branch_and_bound(conflict, weights, bnb_deadline)
    
    # Use the better of greedy and B&B
    if bnb_weight > 0:
        # Compare with greedy
        greedy_weight = 0.0
        v = greedy_bits
        while v:
            lsb = v & (-v)
            idx = lsb.bit_length()
            greedy_weight += weights[idx]
            v ^= lsb
        if bnb_weight > greedy_weight:
            current_bits = bnb_bits
        else:
            current_bits = greedy_bits
    else:
        current_bits = greedy_bits
    
    # Phase 3: Local search
    ls_deadline = deadline - 2
    current_bits, _ = _local_search(current_bits, conflict, weights, ls_deadline)
    
    # Convert to list
    result = []
    v = current_bits
    while v:
        lsb = v & (-v)
        idx = lsb.bit_length()
        result.append(idx)
        v ^= lsb
    
    if not result:
        result = [1]
    
    return sorted(result)
