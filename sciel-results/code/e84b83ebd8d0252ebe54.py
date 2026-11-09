import time
import random
from itertools import combinations

def _check_addition(x, S):
    """
    Check if x can be added to S.
    S is a list of integers.
    Condition: No element in S U {x} divides the sum of any 3 distinct OTHER elements.
    Since S is valid, we only need to check triples involving x.
    1. x divides sum of 3 distinct elements from S.
    2. Any a in S divides sum of x and 2 distinct elements from S \ {a}.
    """
    if len(S) < 3:
        return True
    
    # 1. Check if x divides sum of any 3 distinct elements from S
    # Sum of 3 elements from S
    for a, b, c in combinations(S, 3):
        if (a + b + c) % x == 0:
            return False
            
    # 2. Check if any a in S divides (x + b + c) where b, c are distinct from a and x
    for a in S:
        # We need to pick 2 distinct elements b, c from S \ {a}
        # Sum is x + b + c
        # Check if a divides x + b + c
        for b in S:
            if b == a:
                continue
            for c in S:
                if c == a or c == b:
                    continue
                if (x + b + c) % a == 0:
                    return False
    return True

def _count_conflicts(x, S):
    """
    Count how many elements in S would conflict with x if x were added.
    This is used for heuristic ordering (Most Constrained Variable).
    A conflict exists if adding x violates the condition for x or any a in S.
    """
    conflicts = 0
    if len(S) >= 3:
        # Check if x divides any triple sum in S
        for a, b, c in combinations(S, 3):
            if (a + b + c) % x == 0:
                conflicts += 1
                break # One violation is enough to mark it as conflicting for this x
        
        # Check if any a in S divides x + b + c
        for a in S:
            violated = False
            for b in S:
                if b == a:
                    continue
                for c in S:
                    if c == a or c == b:
                        continue
                    if (x + b + c) % a == 0:
                        violated = True
                        break
                if violated:
                    break
            if violated:
                conflicts += 1
    return conflicts

def _backtrack_search(candidates, S, depth, target_size, deadline):
    """
    Recursive backtracking search.
    candidates: list of available numbers
    S: current valid set
    depth: current recursion depth
    target_size: desired size (48)
    deadline: time limit
    """
    if time.time() > deadline:
        return S
    
    if len(S) == target_size:
        return S
    
    if not candidates:
        return S
    
    # Heuristic: Sort candidates by "constraint"
    # We prefer candidates that have FEWER conflicts with the current S? 
    # Actually, in CSP, we often pick the most constrained variable first.
    # Here, "conflict count" tells us how many existing elements would be violated if we added x.
    # If conflict count is 0, it's a "safe" move.
    # If conflict count > 0, it's an "invalid" move immediately.
    # Wait, _check_addition returns False if ANY conflict exists.
    # So _count_conflicts > 0 means _check_addition would return False.
    # Therefore, we should only consider candidates with 0 conflicts.
    
    valid_candidates = []
    for x in candidates:
        if _check_addition(x, S):
            valid_candidates.append(x)
            
    if not valid_candidates:
        return S
        
    # Among valid candidates, which one to pick next?
    # Strategy: Pick the one that "locks in" the most constraints?
    # Or pick the one that leaves the most options?
    # Let's try: Pick the candidate that has the highest value (largest numbers are harder to place later?)
    # Or pick the one that minimizes the number of future conflicts?
    
    # Simple heuristic: Pick the largest valid candidate first (greedy within valid set)
    # This tends to place large numbers early, which might constrain the small numbers more effectively?
    # Or pick the smallest?
    # Let's try a mix. For now, let's stick to a deterministic choice to avoid randomness in recursion depth.
    # We can randomize the selection of valid candidates to explore different branches.
    
    # To make it a proper search, we should try multiple options if possible, but that's expensive.
    # Instead, we use a randomized restart strategy with a greedy-like backtracker.
    
    # Let's pick a random valid candidate to explore different paths
    # Or better: pick the one that seems "most critical".
    # Let's just pick the first one in a random shuffle of valid_candidates for diversity.
    
    random.shuffle(valid_candidates)
    x = valid_candidates[0]
    
    new_S = S + [x]
    new_candidates = [c for c in candidates if c != x]
    
    # Recurse
    result = _backtrack_search(new_candidates, new_S, depth + 1, target_size, deadline)
    
    # If we didn't reach target, and we have time, we could backtrack and try another x.
    # But deep backtracking is expensive. 
    # Let's implement a limited backtracking: if we fail to reach target, try the next valid candidate?
    # This is essentially a depth-first search with backtracking.
    
    # To keep it simple and fast, let's do a single pass backtracking (greedy with lookahead) first.
    # If that fails, we rely on the outer loop restarting with different random seeds/shuffles.
    
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s limit
    
    best_set = []
    
    # 1. Deterministic Greedy (Descending) as baseline
    order = list(range(192, 0, -1))
    S = []
    for x in order:
        if _check_addition(x, S):
            S.append(x)
    if len(S) > len(best_set):
        best_set = S
    if len(best_set) == 48:
        return best_set
        
    # 2. Backtracking Search with Randomized Heuristics
    # We will run multiple iterations of a backtracking search.
    # Each iteration starts with a random subset of candidates or a randomized order.
    
    for i in range(200): # Max iterations, limited by time
        if time.time() > deadline:
            break
            
        # Randomize the initial candidate pool order or selection
        # Strategy: Start with all numbers, but shuffle them.
        # The backtracker will pick valid ones.
        
        # To introduce diversity, we can bias the initial "valid_candidates" selection.
        # Let's just shuffle the full range and let the backtracker do its thing.
        
        candidates = list(range(1, 193))
        random.shuffle(candidates)
        
        # Run backtracking search
        # We pass the shuffled list. The search function will filter valid ones.
        S = _backtrack_search(candidates, [], 0, 48, deadline)
        
        if len(S) > len(best_set):
            best_set = S
            if len(best_set) == 48:
                return best_set
                
    return best_set

# To improve performance, we can optimize _check_addition using numpy or bitsets?
# For n=192, O(N^3) is 192^3 ~ 7 million ops per check. 
# If we do 48 checks, that's ~350 million ops. 
# Python might be too slow for 200 iterations of full backtracking if each step is expensive.
# Let's optimize _check_addition slightly or reduce iterations.

# Actually, the current _check_addition is O(K^3) where K is size of S.
# Max K=48. 48^3 = 110,592. 
# For each candidate x (192 candidates), we check 110k ops. 
# Total per greedy pass: 192 * 110k ~ 21 million ops.
# 200 iterations * 21 million = 4.2 billion ops. 
# This is likely too slow for Python in 50s.
# We need to be more efficient or reduce iterations.

# Let's reduce iterations to 20 and make the backtracking smarter.
# Or, use the greedy approach but with a "lookahead" of 1 or 2 steps?

# Let's stick to a hybrid: 
# 1. Fast greedy to get a good baseline.
# 2. Local search / Hill climbing on the baseline.

def _local_search(S, deadline):
    """
    Try to improve S by swapping elements or adding new ones.
    """
    S = list(S)
    current_size = len(S)
    remaining = [x for x in range(1, 193) if x not in S]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try adding any remaining element
        for x in remaining:
            if _check_addition(x, S):
                S.append(x)
                remaining.remove(x)
                improved = True
                break
        
        if not improved:
            # Try swapping: remove one element, add another
            # This is expensive. Let's just try adding for now.
            # If we can't add, we might be stuck.
            # We can try removing a "bad" element to allow a "good" one?
            # Too complex for now.
            pass
            
    return S

def construct_optimized():
    start_time = time.time()
    deadline = start_time + 50.0
    
    best_set = []
    
    # 1. Deterministic Greedy (Descending)
    order = list(range(192, 0, -1))
    S = []
    for x in order:
        if _check_addition(x, S):
            S.append(x)
    if len(S) > len(best_set):
        best_set = S
        
    # 2. Randomized Greedy with Local Search
    for i in range(50):
        if time.time() > deadline:
            break
            
        order = list(range(1, 193))
        random.shuffle(order)
        
        S = []
        for x in order:
            if _check_addition(x, S):
                S.append(x)
                
        # Local search: try to add more
        S = _local_search(S, deadline)
        
        if len(S) > len(best_set):
            best_set = S
            if len(best_set) == 48:
                return best_set
                
    return best_set

# We will use construct_optimized as the main construct
# But wait, the task says "Keep the function names and signatures of the seed program".
# The seed program has `construct`.
# I will replace the body of `construct` with the optimized version.

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    best_set = []
    
    # 1. Deterministic Greedy (Descending)
    order = list(range(192, 0, -1))
    S = []
    for x in order:
        if _check_addition(x, S):
            S.append(x)
    if len(S) > len(best_set):
        best_set = S
        
    # 2. Randomized Greedy with Local Search
    for i in range(50):
        if time.time() > deadline:
            break
            
        order = list(range(1, 193))
        random.shuffle(order)
        
        S = []
        for x in order:
            if _check_addition(x, S):
                S.append(x)
                
        # Local search: try to add more
        # We need a helper for local search
        improved = True
        while improved and time.time() < deadline:
            improved = False
            remaining = [x for x in range(1, 193) if x not in S]
            for x in remaining:
                if _check_addition(x, S):
                    S.append(x)
                    improved = True
                    break
                    
        if len(S) > len(best_set):
            best_set = S
            if len(best_set) == 48:
                return best_set
                
    return best_set
