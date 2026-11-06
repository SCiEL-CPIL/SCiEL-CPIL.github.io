import time
import sys
from fractions import Fraction
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit (90s wall clock, but evaluator may kill at 60s)
    
    N = 256
    
    # Precompute weights as Fractions for exact arithmetic
    weights = [Fraction(0)] * (N + 1)
    for i in range(1, N + 1):
        weights[i] = Fraction(1, i)
    
    # Precompute conflict: two elements x, y conflict if they form a 3-AP with some z
    # i.e., there exists z in [1,N], z != x, z != y such that {x, y, z} is a 3-AP
    # This happens if:
    #   - (x+y) is even and (x+y)//2 in [1,N] and (x+y)//2 not in {x,y}
    #   - 2x - y in [1,N] and 2x - y not in {x,y}
    #   - 2y - x in [1,N] and 2y - x not in {x,y}
    
    # For efficient checking, precompute for each x, the set of elements that conflict with x
    # But we can also check on the fly for speed
    
    # Let's precompute a conflict matrix using bitsets for fast membership
    # Actually, let's precompute for each x, a frozenset of conflicting y's
    conflict_sets = [frozenset() for _ in range(N + 1)]
    for x in range(1, N + 1):
        conflicting = set()
        for y in range(1, N + 1):
            if x == y:
                continue
            s = x + y
            if s % 2 == 0:
                m = s // 2
                if 1 <= m <= N and m != x and m != y:
                    conflicting.add(y)
                    continue
            z1 = 2 * x - y
            if 1 <= z1 <= N and z1 != x and z1 != y:
                conflicting.add(y)
                continue
            z2 = 2 * y - x
            if 1 <= z2 <= N and z2 != x and z2 != y:
                conflicting.add(y)
                continue
        conflict_sets[x] = frozenset(conflicting)
    
    # Suffix sums for bounding
    suffix_sum = [Fraction(0)] * (N + 2)
    for i in range(N, 0, -1):
        suffix_sum[i] = suffix_sum[i + 1] + weights[i]
    
    # Greedy initialization: process in order of decreasing weight (increasing index)
    in_S = [False] * (N + 1)
    S = set()
    
    for x in range(1, N + 1):
        valid = True
        for a in S:
            if a in conflict_sets[x]:
                valid = False
                break
        if valid:
            S.add(x)
            in_S[x] = True
    
    best_score = sum(weights[x] for x in S)
    best_set = set(S)
    
    # Branch and Bound DFS
    # Process elements in order 1, 2, ..., N
    # State: current index, current S (as frozenset for hashing if needed), current score, blocked elements
    
    # To make DFS efficient, use a different approach:
    # We'll do a recursive search with good pruning
    
    def is_valid_addition(x, S_current):
        """Check if x can be added to S_current without creating a 3-AP"""
        for a in S_current:
            if a in conflict_sets[x]:
                return False
        return True
    
    def get_blocked(S_current):
        """Get set of elements blocked by S_current"""
        blocked = set()
        for x in S_current:
            for y in conflict_sets[x]:
                if y not in S_current:
                    blocked.add(y)
        return blocked
    
    # For DFS, we need to track which elements are still available
    # Use a bitmask or a set
    
    # Let's implement an iterative DFS with explicit stack
    # State: (idx, S_tuple, score, blocked_frozen)
    # But copying sets is expensive. Let's use a different approach.
    
    # Better: use recursion with backtracking, maintaining a mutable state
    
    # We'll use a global mutable state and recurse
    # Process in order 1, 2, ..., N
    
    # Actually, for the time budget, let's do a focused local search instead
    # which is more likely to find improvements within the time limit
    
    # Local search with multiple restarts and perturbation
    
    def local_search(initial_set):
        """Perform local search from initial_set"""
        S_local = set(initial_set)
        in_local = [False] * (N + 1)
        for x in S_local:
            in_local[x] = True
        
        score_local = sum(weights[x] for x in S_local)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            best_gain = Fraction(0)
            best_move = None
            
            # Try single additions
            for x in range(1, N + 1):
                if in_local[x]:
                    continue
                valid = True
                for a in S_local:
                    if a in conflict_sets[x]:
                        valid = False
                        break
                if valid:
                    gain = weights[x]
                    if gain > best_gain:
                        best_gain = gain
                        best_move = ('add', x)
            
            # Try single swaps
            for v in range(1, N + 1):
                if in_local[v]:
                    continue
                for u in list(S_local):
                    valid = True
                    for a in S_local:
                        if a == u:
                            continue
                        if a in conflict_sets[v]:
                            valid = False
                            break
                    if valid:
                        gain = weights[v] - weights[u]
                        if gain > best_gain:
                            best_gain = gain
                            best_move = ('swap', u, v)
            
            # Try 2-for-2 swaps if time permits
            if time.time() < deadline - 5.0:
                S_list = list(S_local)
                for v1 in range(1, N + 1):
                    if in_local[v1]:
                        continue
                    for v2 in range(v1 + 1, N + 1):
                        if in_local[v2]:
                            continue
                        # Check v1 and v2 don't conflict
                        if v2 in conflict_sets[v1]:
                            continue
                        # Try removing pairs
                        for i in range(len(S_list)):
                            u1 = S_list[i]
                            for j in range(i + 1, len(S_list)):
                                u2 = S_list[j]
                                # Check v1, v2 valid in S_local \ {u1, u2}
                                valid = True
                                for a in S_local:
                                    if a == u1 or a == u2:
                                        continue
                                    if a in conflict_sets[v1] or a in conflict_sets[v2]:
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
                    S_local.add(x)
                    in_local[x] = True
                elif best_move[0] == 'swap':
                    u, v = best_move[1], best_move[2]
                    S_local.remove(u)
                    in_local[u] = False
                    S_local.add(v)
                    in_local[v] = True
                elif best_move[0] == '2swap':
                    u1, u2, v1, v2 = best_move[1], best_move[2], best_move[3], best_move[4]
                    S_local.remove(u1)
                    in_local[u1] = False
                    S_local.remove(u2)
                    in_local[u2] = False
                    S_local.add(v1)
                    in_local[v1] = True
                    S_local.add(v2)
                    in_local[v2] = True
                
                score_local = sum(weights[x] for x in S_local)
                improved = True
        
        return S_local, score_local
    
    # Run local search from greedy
    S_ls, score_ls = local_search(best_set)
    if score_ls > best_score:
        best_score = score_ls
        best_set = S_ls
    
    # Try multiple random restarts
    random.seed(42)
    for restart in range(10):
        if time.time() > deadline - 5.0:
            break
        
        # Create a random perturbation of the best set
        S_pert = set(best_set)
        # Remove 1-3 random elements
        to_remove = min(random.randint(1, 3), len(S_pert))
        remove_list = random.sample(list(S_pert), to_remove)
        for x in remove_list:
            S_pert.discard(x)
        
        S_ls2, score_ls2 = local_search(S_pert)
        if score_ls2 > best_score:
            best_score = score_ls2
            best_set = S_ls2
    
    # Also try a different greedy order: random shuffles
    for trial in range(20):
        if time.time() > deadline - 3.0:
            break
        
        # Random greedy
        S_rand = set()
        in_rand = [False] * (N + 1)
        order = list(range(1, N + 1))
        random.shuffle(order)
        
        for x in order:
            valid = True
            for a in S_rand:
                if a in conflict_sets[x]:
                    valid = False
                    break
            if valid:
                S_rand.add(x)
                in_rand[x] = True
        
        S_ls3, score_ls3 = local_search(S_rand)
        if score_ls3 > best_score:
            best_score = score_ls3
            best_set = S_ls3
    
    return list(best_set)
