import time
import math
import random
import sys

N = 128

# Precompute conflicts: conflicts[x] = set of (u, v) pairs such that {x, u, v} is a forbidden triple
conflicts = [set() for _ in range(N + 1)]
# Also build the 2-shadow graph: shadow[x] = set of y such that x and y appear in some forbidden triple
shadow = [set() for _ in range(N + 1)]

def precompute_conflicts():
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            L_ab = math.lcm(a, b)
            for c in range(b + 1, N + 1):
                if L_ab % c != 0:
                    continue
                if math.lcm(a, c) == L_ab and math.lcm(b, c) == L_ab:
                    conflicts[a].add((b, c))
                    conflicts[b].add((a, c))
                    conflicts[c].add((a, b))
                    shadow[a].add(b)
                    shadow[a].add(c)
                    shadow[b].add(a)
                    shadow[b].add(c)
                    shadow[c].add(a)
                    shadow[c].add(b)

precompute_conflicts()

def can_add(x, current_set):
    for u, v in conflicts[x]:
        if u in current_set and v in current_set:
            return False
    return True

def greedy_color_bound(remaining):
    """
    Compute greedy coloring upper bound on the 2-shadow graph induced by remaining.
    The number of colors used is an upper bound on the independence number.
    """
    if not remaining:
        return 0
    
    # Greedy coloring: assign colors in order, use the smallest available color
    # We use a simple sequential greedy coloring
    color_assignment = {}
    max_color = 0
    rem_list = list(remaining)
    
    for v in rem_list:
        # Find used colors by neighbors of v in remaining
        used_colors = set()
        for u in shadow[v]:
            if u in color_assignment:
                used_colors.add(color_assignment[u])
        # Assign smallest available color
        c = 0
        while c in used_colors:
            c += 1
        color_assignment[v] = c
        max_color = max(max_color, c + 1)
    
    return max_color

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    best_set = set()
    best_size = 0
    
    # Initial greedy to get a good lower bound
    def greedy_construct():
        s = set()
        for _ in range(200):
            order = list(range(1, N + 1))
            random.shuffle(order)
            cur = set()
            for x in order:
                if can_add(x, cur):
                    cur.add(x)
            if len(cur) > len(s):
                s = cur
        return s
    
    initial_set = greedy_construct()
    best_set = initial_set
    best_size = len(best_set)
    
    # B&B with greedy coloring bound
    def bnb_rec(current_set, remaining, depth):
        nonlocal best_set, best_size
        
        if time.time() > deadline:
            return False
        
        if not remaining:
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = current_set.copy()
            return True
        
        # Upper bound using greedy coloring of 2-shadow
        ub = len(current_set) + greedy_color_bound(remaining)
        if ub <= best_size:
            return True
        
        # Pick branching variable: most constrained (highest degree in shadow within remaining)
        best_x = None
        best_deg = -1
        for x in remaining:
            deg = len(shadow[x] & remaining)
            if deg > best_deg:
                best_deg = deg
                best_x = x
        
        rest = [x for x in remaining if x != best_x]
        
        # Branch 1: Include best_x
        new_set = current_set | {best_x}
        valid_rest_include = [y for y in rest if can_add(y, new_set)]
        
        bnb_rec(new_set, valid_rest_include, depth + 1)
        
        if time.time() > deadline:
            return False
        
        # Branch 2: Exclude best_x
        bnb_rec(current_set, rest, depth + 1)
        
        return True

    # Run multiple B&B restarts with different orderings
    for attempt in range(30):
        if time.time() > deadline:
            break
        
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        
        # Start B&B from scratch with shuffled order
        # The branching heuristic will re-order based on degree
        bnb_rec(set(), candidates, 0)
        
        if time.time() > deadline:
            break

    result = list(best_set)
    if not result:
        result = [1]
    
    return result
