import time
import math
import random
from functools import lru_cache

N = 128
# conflicts[x] will be a set of integers y such that x and y are in a forbidden triple with some z.
# Actually, for B&B on hypergraph, we need to know which triples are forbidden.
# But checking "can I add x?" requires checking if adding x creates a forbidden triple with any two existing elements.
# This is equivalent to: for all pairs (u, v) in current_set, is (x, u, v) a forbidden triple?
# Precomputing all forbidden triples is O(N^3) which is ~2M. This is fast enough if done once.
# Let's store forbidden triples as a set of frozensets or sorted tuples.
# Or better, for each x, store the set of pairs (u, v) such that {x, u, v} is forbidden.
# Then, to check if x can be added to S:
# For each (u, v) in conflicts[x], if u in S and v in S, then x cannot be added.

conflicts = [set() for _ in range(N + 1)]
# To speed up the "can add" check, we can also precompute for each pair (u, v) the list of x that conflict with them.
# But the check "is (u,v) in S" is fast if S is a set or bitset.

def precompute_conflicts():
    # Iterate through all triples a < b < c
    # Check if lcm(a,b) == lcm(a,c) == lcm(b,c)
    # Optimization: 
    # lcm(a,b) = L.
    # c must be a divisor of L.
    # Also lcm(a,c) = L and lcm(b,c) = L.
    
    # Let's iterate L from 1 to some max.
    # Max LCM of two numbers <= 128 is 128*127 = 16256 (if coprime).
    # But c <= 128.
    # For a fixed L, let D_L = {d in [1, 128] : d divides L and L/d is integer? No, d divides L.}
    # Actually, if lcm(a,b)=L, then a|L and b|L.
    # So we can group numbers by their "LCM signature"? No.
    
    # Direct triple iteration is 128*127*126/6 ~ 340k iterations.
    # Each iteration does 3 lcm calls.
    # 340k * 3 = 1M lcm calls. This is very fast in Python (<0.1s).
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            L_ab = math.lcm(a, b)
            # If L_ab is large, c must divide L_ab.
            # c < b (since a<b<c is not required, just distinct. We can assume a<b<c for generation to avoid duplicates)
            # Wait, the triple is {a,b,c}. Order doesn't matter.
            # Let's generate a < b < c.
            # If we do that, we only check 1/6 of the space.
            
            # For a < b, find c > b such that lcm(a,c)=L_ab and lcm(b,c)=L_ab.
            # c must be a divisor of L_ab.
            # Let's find all divisors of L_ab that are > b and <= N.
            
            # To make this faster, let's just iterate c from b+1 to N?
            # That would be O(N^3) ~ 2M. Still fast.
            # Let's stick to the simple O(N^3) but optimized with early breaks if possible?
            # No, just run the 340k unique triples.
            
    # Let's restart the precomputation logic to be cleaner.
    # We will iterate a < b < c.
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            L_ab = math.lcm(a, b)
            # c must be > b and <= N
            # And c must divide L_ab? 
            # If lcm(a,c) = L_ab, then c must divide L_ab.
            # So c is a divisor of L_ab.
            # Let's find divisors of L_ab that are > b and <= N.
            
            # Efficient divisor finding for L_ab
            # We can precompute divisors for all numbers up to max_LCM?
            # Max LCM is ~16k.
            # Let's just compute divisors on the fly for each L_ab encountered.
            # There are at most ~8k pairs (a,b).
            
            # Optimization: Precompute divisors for all possible LCM values?
            # Or just iterate c from b+1 to N and check if c divides L_ab?
            # That is O(N) per pair. Total O(N^3). 2M ops. Fine.
            
            for c in range(b + 1, N + 1):
                if L_ab % c != 0:
                    continue
                if math.lcm(a, c) == L_ab and math.lcm(b, c) == L_ab:
                    # Found forbidden triple (a, b, c)
                    conflicts[a].add((b, c))
                    conflicts[b].add((a, c))
                    conflicts[c].add((a, b))

# Run precomputation
precompute_conflicts()

def can_add(x, current_set):
    """
    Check if x can be added to current_set (a set of ints).
    Returns True if no conflict.
    """
    for u, v in conflicts[x]:
        if u in current_set and v in current_set:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    # We will use Branch and Bound.
    # State: 
    #   current_set: set of ints
    #   remaining: list of ints not in current_set and not excluded
    #   best_size: global variable
    #   best_set: global variable
    
    # To make B&B effective, we need a good ordering and a good bound.
    # Bound: current_size + len(remaining) is weak.
    # Better bound: current_size + max_independent_set_size(remaining_conflict_graph)
    # Computing MIS of remaining graph at each step is expensive.
    # Approximation: Greedy coloring of remaining graph.
    
    # Let's try a different structural approach: 
    # Since N=128 is small, we can try to construct a large set using a greedy algorithm with random restarts and local search, 
    # but the prompt asks for a structural change to B&B.
    
    # Let's implement a simple B&B.
    # We sort numbers by "conflict degree" (number of pairs in conflicts[x]).
    # High degree nodes are more restrictive.
    
    # Actually, let's look at the structure of "equal LCM triples".
    # If a, b, c form a triple, they are all divisors of L and pairwise LCM is L.
    # This implies that for each prime factor p of L, at least two of a,b,c must have the full power of p present in L?
    # No, lcm(a,b)=L means for each p^k || L, max(v_p(a), v_p(b)) = k.
    # For three numbers, max(v_p(a), v_p(b), v_p(c)) = k is not enough.
    # We need max(v_p(a), v_p(b)) = k, max(v_p(a), v_p(c)) = k, max(v_p(b), v_p(c)) = k.
    # This implies that at least two of the three numbers must have v_p(x) = k.
    # If only one has v_p(x) = k, say a, then v_p(b) < k and v_p(c) < k.
    # Then max(v_p(b), v_p(c)) < k, so lcm(b,c) != L.
    # So, for a triple to exist, for every prime power p^k dividing L, at least two of the elements must be divisible by p^k.
    
    # This structure might allow for a smarter construction.
    # However, B&B is requested.
    
    # Let's implement a recursive B&B with a tight time check.
    
    best_set = set()
    best_size = 0
    
    # Initial greedy to get a good lower bound
    def greedy_construct():
        s = set()
        # Try different orderings
        for _ in range(100):
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
    
    # B&B
    # We represent the state by:
    # 1. The set of chosen elements (current)
    # 2. The list of candidate elements (remaining)
    
    # To prune, we need an upper bound on how many more we can add.
    # A simple bound: len(remaining).
    # If len(current) + len(remaining) <= best_size, return.
    
    # We can improve the bound by looking at conflicts.
    # If a node x is in remaining, and it conflicts with many nodes in current, it's already excluded.
    # We maintain `remaining` as a list of valid candidates.
    
    # Let's use a recursive function.
    # To avoid deep recursion issues and overhead, we might use an iterative stack or limit depth.
    # Depth is at most 128.
    
    def solve_bnb(current_set, remaining_list, depth):
        nonlocal best_set, best_size
        
        if time.time() > deadline:
            return
            
        if not remaining_list:
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = current_set.copy()
            return
        
        # Bound
        if len(current_set) + len(remaining_list) <= best_size:
            return
        
        # Heuristic: Pick the next element to branch on.
        # Choose an element from remaining_list.
        # To make it efficient, we can pick the first one, or one with high conflict potential.
        
        # Let's try a randomized B&B or multiple starts.
        # Given the time limit, a single deep B&B might get stuck.
        # Let's try a few restarts of B&B with different random orderings of candidates.
        
        pass # Placeholder

    # Since a full B&B might be too slow or complex to implement robustly within the edit constraints,
    # and the previous failure was due to precomputation time, 
    # the "Structural Change" directive might be satisfied by a much faster, smarter Greedy + Local Search 
    # that actually runs to completion, OR a very tight B&B.
    
    # Let's try a specific B&B strategy:
    # 1. Order candidates by "pessimistic" impact.
    # 2. Recurse.
    
    # Actually, let's look at the time. 55 seconds is a lot.
    # The previous code spent all time in precompute.
    # Now precompute is ~0.1s.
    # We have ~50s for search.
    
    # Let's implement a simple recursive B&B.
    
    def bnb_rec(current_set, remaining):
        nonlocal best_set, best_size
        
        if time.time() > deadline:
            return False # Signal timeout
        
        if len(current_set) + len(remaining) <= best_size:
            return True # Pruned
        
        if not remaining:
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = current_set.copy()
            return True
            
        # Pick an element x from remaining
        # Strategy: Pick x that conflicts with the most other remaining elements?
        # Or just pick the first one?
        # Let's pick the first one for simplicity, but shuffle remaining initially.
        
        x = remaining[0]
        rest = remaining[1:]
        
        # Branch 1: Include x
        new_set = current_set | {x}
        # Filter rest to remove elements that conflict with x
        # An element y in rest conflicts with x if there exists z in current_set such that (x,y,z) is a forbidden triple?
        # No, y conflicts with x if adding y to new_set would create a conflict.
        # y can be added to new_set if for all (u,v) in conflicts[y], not (u in new_set and v in new_set).
        
        valid_rest_include = []
        for y in rest:
            if can_add(y, new_set):
                valid_rest_include.append(y)
        
        bnb_rec(new_set, valid_rest_include)
        
        if time.time() > deadline:
            return False
            
        # Branch 2: Exclude x
        bnb_rec(current_set, rest)
        
        return True

    # To make B&B work, we need a good initial ordering.
    # Let's try a few random shuffles of the initial list.
    
    for _ in range(50): # 50 restarts of B&B
        if time.time() > deadline:
            break
            
        # Randomize the order of candidates
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        
        # We can also filter out elements that are "obviously bad"? No.
        
        # Run B&B
        # Note: bnb_rec modifies best_set/best_size globally.
        # It returns False if timeout.
        
        # We need to pass the initial empty set.
        # But we can start with the greedy solution as a "seed" for the B&B?
        # B&B usually starts from scratch to prove optimality, but here we just want a good solution.
        # Starting from scratch with a good lower bound (from greedy) helps pruning.
        
        # Let's just run B&B from scratch.
        bnb_rec(set(), candidates)
        
        if time.time() > deadline:
            break

    # Convert best_set to list
    result = list(best_set)
    
    if not result:
        result = [1]
        
    return result
