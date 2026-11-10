import math
import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    n = 128
    elements = list(range(1, n + 1))
    
    def lcm(a, b):
        return a * b // math.gcd(a, b)
    
    # 1. Precompute bad triples
    # A triple (a,b,c) is bad if lcm(a,b) == lcm(a,c) == lcm(b,c)
    bad_triples = []
    # For efficient checking, we also store for each element x, the list of (y, z) 
    # such that (x,y,z) is a bad triple.
    # Actually, for adding x to S, we need to check if there exists (y,z) in S 
    # such that (x,y,z) is bad.
    # So for each x, we store a list of sets {y,z} for each bad triple containing x.
    conflicts = {x: [] for x in elements}
    
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            lab = lcm(a, b)
            for c in range(b + 1, n + 1):
                if lcm(a, c) == lab and lcm(b, c) == lab:
                    bad_triples.append((a, b, c))
                    conflicts[a].append((b, c))
                    conflicts[b].append((a, c))
                    conflicts[c].append((a, b))
    
    # 2. Helper to check if adding x to S is valid
    # S is a set of selected elements
    def can_add(x, S):
        for y, z in conflicts[x]:
            if y in S and z in S:
                return False
        return True

    # 3. Greedy Initialization
    def greedy_init(order):
        S = set()
        for x in order:
            if can_add(x, S):
                S.add(x)
        return S

    # 4. Local Search (Hill Climbing)
    # We try to improve the set by swapping elements.
    def local_search(S, max_iters=1000):
        S = set(S)
        best_S = set(S)
        best_size = len(S)
        current_size = len(S)
        
        # Precompute list of elements not in S for quick access
        # We'll just iterate over all elements for swaps
        
        no_improve_count = 0
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            improved = False
            
            # Try 1-swap: remove x in S, add y not in S
            # We iterate through all x in S and y not in S
            # To speed up, we can just iterate all y not in S and check if we can add y,
            # and if not, try to remove some x to make it possible?
            # Simpler: Try all pairs (x in S, y not in S).
            
            # Optimization: Only try swaps if we might gain size.
            # But we only gain size if we can add without removing.
            # So first, try to just add any element.
            for y in elements:
                if y not in S and can_add(y, S):
                    S.add(y)
                    current_size += 1
                    improved = True
                    if current_size > best_size:
                        best_size = current_size
                        best_S = set(S)
                    break # Restart search after an addition
            
            if improved:
                no_improve_count = 0
                continue
            
            # If no single addition, try swaps
            # We want to swap x for y such that the new set is valid.
            # This is equivalent to: S' = (S \ {x}) U {y} is valid.
            # This requires that no bad triple is fully contained in S'.
            # Bad triples in S' are either:
            # 1. Bad triples in S that do not involve x. (These were already valid in S, so they are valid in S')
            # 2. Bad triples involving y and two elements from S \ {x}.
            # 3. Bad triples involving x? No, x is removed.
            
            # So we just need to check if adding y to (S \ {x}) is valid.
            # But we don't want to try all x for all y. That's O(N^2) checks per iteration.
            # N=128, so 128*128 = 16384 checks. Each check is O(conflicts[y]).
            # This is feasible.
            
            found_swap = False
            # Randomize order of swaps to escape local optima
            xs = list(S)
            random.shuffle(xs)
            ys = [y for y in elements if y not in S]
            random.shuffle(ys)
            
            for y in ys:
                if found_swap:
                    break
                for x in xs:
                    if time.time() > deadline:
                        return best_S
                    
                    # Check if S \ {x} U {y} is valid
                    # This is equivalent to checking if y can be added to S \ {x}
                    # We can simulate this:
                    # Temporarily remove x, check can_add(y), then restore x?
                    # Or just check directly:
                    # A bad triple in S \ {x} U {y} must involve y.
                    # Let the triple be (y, u, v).
                    # u, v must be in S \ {x}.
                    # So we check if there exists (u, v) in conflicts[y] such that u in S and v in S and u != x and v != x.
                    
                    valid_swap = True
                    for u, v in conflicts[y]:
                        if u == x or v == x:
                            continue
                        if u in S and v in S:
                            valid_swap = False
                            break
                    
                    if valid_swap:
                        S.remove(x)
                        S.add(y)
                        improved = True
                        found_swap = True
                        # Size remains same, so best_size doesn't change unless we added later
                        # But we might be in a position to add more later.
                        break
            
            if improved:
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count > 50:
                    break
        
        return best_S

    best_result = set()
    
    # Multiple restarts with different initializations
    random.seed(42)
    
    # 1. Natural order
    S = greedy_init(elements)
    best_result = local_search(S)
    
    # 2. Reverse order
    S = greedy_init(list(reversed(elements)))
    if len(local_search(S)) > len(best_result):
        best_result = local_search(S)
    
    # 3. Random orders
    for i in range(100):
        if time.time() > deadline:
            break
        
        order = elements[:]
        random.shuffle(order)
        S = greedy_init(order)
        
        # Quick local search
        improved_S = local_search(S, max_iters=200)
        if len(improved_S) > len(best_result):
            best_result = improved_S
    
    # Final polish on the best result with more iterations
    if time.time() < deadline - 5:
        final_S = local_search(best_result, max_iters=2000)
        if len(final_S) > len(best_result):
            best_result = final_S
    
    return list(best_result)
