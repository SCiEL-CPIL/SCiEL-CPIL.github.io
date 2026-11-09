import time
import random
import math

def construct():
    start_time = time.time()
    DEADLINE = 145  # Safety margin for 150s limit

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def is_sum_free(subset):
        s = set(subset)
        for a in subset:
            for b in subset:
                if a + b in s:
                    return False
        return True

    def solve_instance(values, budget):
        n = len(values)
        if n <= 2:
            return list(values)
        
        v_set = set(values)
        
        # Precompute conflict graph
        # Edge (u, v) exists if there is a w in values such that u + v = w, u + w = v, or v + w = u
        # This is equivalent to: one is the sum of the other two.
        # We want to find a maximum independent set in this graph.
        
        # For n=240, building the graph is O(n^2 * log n) or O(n^2) with set lookups.
        # We can represent the graph as a list of bitmasks or sets of neighbors.
        # Since n <= 240, we can use Python integers as bitmasks if n <= 63? No, 240 > 63.
        # We can use a list of sets of indices.
        
        index_to_val = {v: i for i, v in enumerate(values)}
        
        # Build adjacency list
        # adj[i] is a set of indices j such that (values[i], values[j]) is a conflict.
        adj = [set() for _ in range(n)]
        
        # For each pair (i, j), check if they conflict.
        # Conflict if:
        # 1. values[i] + values[j] in v_set
        # 2. values[i] - values[j] in v_set (implies values[j] + (values[i]-values[j]) = values[i]) -> wait.
        #    If a+b=c, then {a,b,c} is a triangle in the conflict graph?
        #    Let's check:
        #    If a+b=c:
        #    Pair (a,b): a+b=c in v_set -> conflict.
        #    Pair (a,c): a+c = 2a+b? No. a+c = a+a+b. Is a+a+b in v_set? Not necessarily.
        #    But wait, the condition is "no a+b=c".
        #    The conflict graph should have an edge between u and v if u+v is in the set?
        #    No, if we pick u and v, and u+v is also in our set, that's a violation.
        #    So if we include u and v, we MUST exclude u+v.
        #    This is a 3-uniform hypergraph independent set problem.
        #    Max Independent Set in a 3-uniform hypergraph.
        #    
        #    However, we can transform this.
        #    If we select a subset S, it is sum-free if for all a,b in S, a+b is NOT in S.
        #    This is equivalent to: for every triple (a,b,c) with a+b=c, we cannot have all three in S.
        #    
        #    Let's stick to the penalty function approach but optimize it.
        #    The current SA uses get_penalty which is O(|S|^2).
        #    For n=240, |S|~80, 80^2 = 6400 ops per step.
        #    10,000 steps = 64M ops. Might be slow in Python.
        #    
        #    Alternative: Use a bitset representation for S.
        #    But values are in [-10000, 10000]. We can map them to indices.
        #    Let's try to speed up the penalty calculation or use a different search.
        
        # Greedy Heuristic:
        # 1. Start with empty set.
        # 2. Sort values by some heuristic (e.g., magnitude, or random).
        # 3. Add a value if it doesn't conflict with current set.
        #    Conflict check: adding x to S is valid if:
        #    - 2x not in S
        #    - for all y in S, x+y not in S and y-x not in S (which means x+y not in S and x-(y) not in S? No, y+x not in S and x+y not in S is same).
        #    Actually, just check: for all y in S, is x+y in S? Is x-y in S? Is y-x in S?
        #    Wait, if we add x, new triples are (x, x, 2x), (x, y, x+y), (y, x, y+x).
        #    So we need: 2x not in S, and for all y in S, x+y not in S and y-x not in S.
        #    (y-x not in S is equivalent to x+y not in S if we consider the set is closed under subtraction? No.)
        #    Let's just check:
        #    for y in S:
        #       if x + y in S: conflict
        #       if y - x in S: conflict  (because y = x + (y-x))
        #    Also check 2x in S.
        
        def can_add(x, S):
            if 2 * x in S:
                return False
            for y in S:
                if x + y in S:
                    return False
                if y - x in S:
                    return False
            return True

        # Multiple restarts of Greedy + Local Search
        best_S = set()
        best_size = 0
        
        # Time per restart
        num_restarts = 10
        time_per_restart = budget / num_restarts
        
        # Different sorting strategies
        strategies = [
            lambda v: v, # Identity
            lambda v: -v, # Negative
            lambda v: abs(v), # Magnitude
            lambda v: -abs(v), # Inverse Magnitude
            lambda v: random.random(), # Random
        ]
        
        for i in range(num_restarts):
            if time.time() - start_time > DEADLINE - 5:
                break
            
            # Choose a strategy
            if i < len(strategies):
                sort_key = strategies[i]
            else:
                sort_key = random.choice(strategies)
            
            # Shuffle values slightly to break ties or randomize
            vals_copy = values[:]
            random.shuffle(vals_copy)
            vals_copy.sort(key=sort_key)
            
            S = set()
            for x in vals_copy:
                if can_add(x, S):
                    S.add(x)
            
            # Local Search: Try to add more elements
            # Iterate through values not in S, try to add if possible.
            # If not possible, maybe remove one element from S to make room?
            # Simple 1-swap: Remove y from S, add x if can_add(x, S-{y}) and |S-{y}+{x}| > |S| (always true if sizes equal, but we want to maintain size or grow).
            # Actually, we want to maximize size.
            # So try to swap y out and x in if it allows adding x and doesn't reduce size (it doesn't, size stays same) but maybe allows future additions?
            # Better: Try to add x. If conflict, find a y in S that conflicts with x. Remove y. Try to add x.
            
            changed = True
            while changed and time.time() - start_time < DEADLINE - 5:
                changed = False
                # Try to add each element not in S
                for x in values:
                    if x in S:
                        continue
                    if can_add(x, S):
                        S.add(x)
                        changed = True
                    else:
                        # Find a blocker
                        # We can try to remove a random blocker and see if we can add x
                        # To save time, only do this for a few candidates
                        pass # Skip complex swap for now, greedy add is fast.
            
            if len(S) > best_size:
                best_size = len(S)
                best_S = set(S)
            
            # Simulated Annealing Post-processor on best_S found so far?
            # Or run SA from this S?
            
            # Let's run a short SA from the current best_S if time permits
            # This is expensive, so maybe just do it once at the end?
            
        # Run SA on the best greedy result
        if best_size > 0:
            S = set(best_S)
            T = 50.0
            T_min = 0.01
            cooling_rate = 0.99
            
            # Fast penalty check
            def get_penalty(S):
                count = 0
                s_list = list(S)
                for a in s_list:
                    for b in s_list:
                        if a + b in S:
                            count += 1
                return count
            
            penalty = get_penalty(S)
            
            # SA Loop
            while T > T_min and time.time() - start_time < DEADLINE - 2:
                x = random.choice(values)
                if x in S:
                    S_new = S - {x}
                    delta_size = -1
                else:
                    S_new = S | {x}
                    delta_size = 1
                
                penalty_new = get_penalty(S_new)
                delta_penalty = penalty_new - penalty
                
                delta_score = delta_size - 100 * delta_penalty
                
                if delta_score >= 0:
                    accept = True
                else:
                    accept = random.random() < math.exp(delta_score / T)
                
                if accept:
                    S = S_new
                    penalty = penalty_new
                    if penalty == 0 and len(S) > best_size:
                        best_size = len(S)
                        best_S = set(S)
                
                T *= cooling_rate
        
        if best_size == 0:
            return [x for x in values if x % 2 != 0]
            
        return list(best_S)

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        remaining = DEADLINE - (time.time() - start_time)
        if remaining < 5:
            vals = instance(size, seed)
            results.append([x for x in vals if x % 2])
            continue

        # Allocate budget proportionally to problem size
        budget = min(remaining - 2, 40 * (size / 240))
        budget = max(budget, 5)

        vals = instance(size, seed)
        res = solve_instance(vals, budget)

        if not is_sum_free(res):
            res = [x for x in vals if x % 2 != 0]

        results.append(res)

    return results
