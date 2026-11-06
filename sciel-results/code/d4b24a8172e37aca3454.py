import time
import sys
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    N = 256
    
    # Precompute weights as Fractions for exact arithmetic
    weights = [Fraction(0)] * (N + 1)
    for i in range(1, N + 1):
        weights[i] = Fraction(1, i)
    
    # Precompute conflict sets: for each x, which other elements form a 3-AP with x
    # conflict[x] = set of y such that x and y cannot both be in S (they form a 3-AP with some z)
    # Actually, we need: if x is in S, which elements are forbidden?
    # Better: precompute for each pair (x, y), whether they conflict (form part of a 3-AP)
    # Two elements x, y conflict if there exists z such that {x, y, z} is a 3-AP
    # This happens if:
    #   - x, y are endpoints: z = (x+y)/2 exists and is integer, 1 <= z <= N
    #   - x is middle: y, x, z => z = 2x - y, 1 <= z <= N
    #   - y is middle: x, y, z => z = 2y - x, 1 <= z <= N
    
    # For branch and bound, we'll track which elements are "blocked" by current choices
    # Use a bitset or boolean array for efficiency
    
    # Precompute all 3-APs
    # An element x is blocked if it forms a 3-AP with any two elements in S
    # More precisely: x can be added to S iff for all pairs (a, b) in S, {x, a, b} is not a 3-AP
    # Equivalently: for all a in S, check if x forms a 3-AP with a and some c in S
    
    # Let's use a different approach: maintain a "blocked" set
    # When we add x to S, we block all y such that x, y, z form a 3-AP for some z in S ∪ {x}
    # Actually, simpler: x and y are incompatible if they are in the same 3-AP
    
    # Precompute incompatibility: incompat[x][y] = True if x and y cannot both be in S
    # This is true if there exists z such that {x, y, z} is a 3-AP
    # Cases:
    # 1. x < y < z: 2y = x + z => z = 2y - x. Check 1 <= z <= N
    # 2. y < x < z: 2x = y + z => z = 2x - y. Check 1 <= z <= N  
    # 3. x < z < y: 2z = x + y => z = (x+y)/2. Check x+y even and 1 <= z <= N
    # 4. z < x < y: 2x = z + y => z = 2x - y. Same as case 2
    # 5. z < y < x: 2y = z + x => z = 2y - x. Same as case 1
    # 6. x < y and x > z: covered
    # Actually, for any distinct x, y, they conflict if:
    #   - (x+y) is even and 1 <= (x+y)//2 <= N (they are endpoints of a 3-AP)
    #   - OR 2x - y is in [1,N] and 2x - y != x and 2x - y != y (x is middle)
    #   - OR 2y - x is in [1,N] and 2y - x != x and 2y - x != y (y is middle)
    
    # Precompute conflict matrix
    conflict = [[False] * (N + 1) for _ in range(N + 1)]
    for x in range(1, N + 1):
        for y in range(x + 1, N + 1):
            c = False
            # Case 1: x, y are endpoints, middle is (x+y)/2
            s = x + y
            if s % 2 == 0:
                m = s // 2
                if 1 <= m <= N and m != x and m != y:
                    c = True
            # Case 2: x is middle, y and z are endpoints: z = 2x - y
            z = 2 * x - y
            if 1 <= z <= N and z != x and z != y:
                c = True
            # Case 3: y is middle, x and z are endpoints: z = 2y - x
            z = 2 * y - x
            if 1 <= z <= N and z != x and z != y:
                c = True
            conflict[x][y] = conflict[y][x] = c
    
    # Greedy initialization: add elements in order of decreasing weight (i.e., increasing index)
    in_S = [False] * (N + 1)
    S = set()
    
    for x in range(1, N + 1):
        # Check if x can be added
        valid = True
        for a in S:
            if conflict[x][a]:
                valid = False
                break
        if valid:
            S.add(x)
            in_S[x] = True
    
    # Compute initial score
    best_score = sum(weights[x] for x in S)
    best_set = set(S)
    
    # Branch and bound
    # We'll explore decisions for elements in order of decreasing weight (1, 2, 3, ...)
    # At each step, we decide to include or exclude element i
    # Bound: current_score + sum of weights of all remaining elements that could potentially be added
    
    # Precompute suffix sums of weights for bounding
    # suffix_sum[i] = sum of weights[i] + weights[i+1] + ... + weights[N]
    suffix_sum = [Fraction(0)] * (N + 2)
    for i in range(N, 0, -1):
        suffix_sum[i] = suffix_sum[i + 1] + weights[i]
    
    # Stack for DFS: (current_index, current_S, current_score, blocked_mask)
    # blocked_mask: bitmask of elements that are blocked (cannot be added)
    # Use Python integers as bitsets
    
    def get_blocked_set(S_set):
        """Get the set of elements blocked by S_set"""
        blocked = set()
        for x in S_set:
            for y in range(1, N + 1):
                if y not in S_set and conflict[x][y]:
                    blocked.add(y)
        return blocked
    
    # Use iterative DFS with explicit stack to avoid recursion limit
    # State: (idx, S_set, score, blocked_set)
    # idx: next element to consider (1-indexed)
    
    # Start with the greedy solution as initial incumbent
    # Then do DFS to find better solutions
    
    # For efficiency, we'll use a different strategy:
    # 1. Start with empty set
    # 2. Process elements in order 1, 2, 3, ..., N
    # 3. At each element, try adding it (if not blocked) or skipping it
    # 4. Use bound: score + sum of weights of unprocessed elements not blocked
    
    # To make this efficient, we need good pruning.
    # The key insight: if we process in order of decreasing weight, we can bound tightly.
    
    # Let's implement a recursive DFS with memoization of blocked states (if feasible)
    # Or just use iterative deepening / branch and bound
    
    # Given the time constraint, let's do a focused search:
    # Start from the greedy solution and try local improvements (swaps, 2-swaps)
    # This is similar to the original approach but with better move evaluation
    
    # Actually, let's implement a proper branch and bound:
    # Process elements in order 1, 2, ..., N
    # At each step i, we decide include or exclude i
    # Bound: current_score + sum_{j >= i, j not blocked} weights[j]
    # If bound <= best_score, prune
    
    # To track blocked elements efficiently, use a frozenset or tuple
    # But this might be expensive. Let's use a boolean array and copy it.
    
    # Alternative: use bit operations. Represent blocked as a set of integers.
    
    # Let's implement a clean DFS:
    
    def dfs(idx, S_set, score, blocked_set):
        nonlocal best_score, best_set
        
        if time.time() > deadline:
            return
        
        # Base case: all elements considered
        if idx > N:
            if score > best_score:
                best_score = score
                best_set = set(S_set)
            return
        
        # Bound: even if we add all remaining unblocked elements, can we beat best_score?
        # Upper bound on additional score: sum of weights[j] for j >= idx, j not in blocked_set
        ub = score
        for j in range(idx, N + 1):
            if j not in blocked_set:
                ub += weights[j]
                if ub <= best_score:
                    break
        
        if ub <= best_score:
            return
        
        # Option 1: Exclude element idx
        dfs(idx + 1, S_set, score, blocked_set)
        
        if time.time() > deadline:
            return
        
        # Option 2: Include element idx (if not blocked)
        if idx not in blocked_set:
            # Check if idx conflicts with any element in S_set
            valid = True
            for a in S_set:
                if conflict[idx][a]:
                    valid = False
                    break
            
            if valid:
                new_S = S_set | {idx}
                # Compute new blocked set
                new_blocked = set(blocked_set)
                for a in S_set:
                    if conflict[idx][a]:
                        # This shouldn't happen if valid is True
                        pass
                # Add elements that conflict with idx
                for y in range(idx + 1, N + 1):
                    if conflict[idx][y] and y not in new_blocked:
                        new_blocked.add(y)
                
                dfs(idx + 1, new_S, score + weights[idx], new_blocked)
    
    # Start DFS from empty set
    # But this might be too slow. Let's use the greedy solution as a starting point
    # and do a local search from there.
    
    # Actually, let's do a hybrid: use the greedy solution, then try to improve it
    # by removing some elements and re-running DFS on the remaining elements.
    
    # For now, let's just do the local search approach but more thoroughly.
    
    # Local search: try all single swaps and 2-swaps
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        best_gain = Fraction(0)
        best_move = None
        
        # Try single additions
        for x in range(1, N + 1):
            if in_S[x]:
                continue
            # Check if x can be added
            valid = True
            for a in S:
                if conflict[x][a]:
                    valid = False
                    break
            if valid:
                gain = weights[x]
                if gain > best_gain:
                    best_gain = gain
                    best_move = ('add', x)
        
        # Try single swaps: remove u, add v
        for v in range(1, N + 1):
            if in_S[v]:
                continue
            # Check if v can be added after removing some u in S
            for u in S:
                # Check if v is valid in S \ {u}
                valid = True
                for a in S:
                    if a == u:
                        continue
                    if conflict[v][a]:
                        valid = False
                        break
                if valid:
                    gain = weights[v] - weights[u]
                    if gain > best_gain:
                        best_gain = gain
                        best_move = ('swap', u, v)
        
        # Try 2-swaps: remove u1, u2, add v1, v2
        # This is expensive, so only do it if we have time
        if time.time() < deadline - 10.0:
            for v1 in range(1, N + 1):
                if in_S[v1]:
                    continue
                for v2 in range(v1 + 1, N + 1):
                    if in_S[v2]:
                        continue
                    # Check if v1 and v2 conflict with each other
                    if conflict[v1][v2]:
                        continue
                    # Check if v1 and v2 can be added after removing some u1, u2 in S
                    # Try all pairs (u1, u2) in S
                    S_list = list(S)
                    for i in range(len(S_list)):
                        u1 = S_list[i]
                        for j in range(i + 1, len(S_list)):
                            u2 = S_list[j]
                            # Check if v1, v2 are valid in S \ {u1, u2}
                            valid = True
                            for a in S:
                                if a == u1 or a == u2:
                                    continue
                                if conflict[v1][a] or conflict[v2][a]:
                                    valid = False
                                    break
                            if valid:
                                gain = weights[v1] + weights[v2] - weights[u1] - weights[u2]
                                if gain > best_gain:
                                    best_gain = gain
                                    best_move = ('2swap', u1, u2, v1, v2)
        
        if best_move is not None:
            if best_move[0] == 'add':
                x = best_move[1]
                S.add(x)
                in_S[x] = True
            elif best_move[0] == 'swap':
                u, v = best_move[1], best_move[2]
                S.remove(u)
                in_S[u] = False
                S.add(v)
                in_S[v] = True
            elif best_move[0] == '2swap':
                u1, u2, v1, v2 = best_move[1], best_move[2], best_move[3], best_move[4]
                S.remove(u1)
                in_S[u1] = False
                S.remove(u2)
                in_S[u2] = False
                S.add(v1)
                in_S[v1] = True
                S.add(v2)
                in_S[v2] = True
            
            best_score = sum(weights[x] for x in S)
            best_set = set(S)
            improved = True
    
    return list(best_set)
