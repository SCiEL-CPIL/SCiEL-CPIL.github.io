import time
import random
import sys

def construct():
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        S_set = set(S)
        for a in S:
            for b in S:
                if a + b in S_set:
                    return False
        return True

    def max_3_set_packing(V, time_limit):
        """
        Finds a maximum subset of V such that no three elements a, b, c in the subset
        satisfy a + b = c.
        This is equivalent to finding a maximum subset that contains no solution to a+b=c.
        We model this as a 3-Set Packing problem where the "sets" are the triples (a, b, c).
        We want to select the maximum number of vertices (elements of V) such that no selected
        triple is fully contained in the selected vertices.
        
        Actually, it's simpler: we want a subset S of V such that for all a, b in S, a+b is not in S.
        This is a constraint satisfaction problem.
        
        We will use a branch and bound approach on the elements of V.
        State: (current_set, remaining_elements, forbidden_pairs)
        However, the constraint is global.
        
        Let's use a different perspective:
        We are selecting a subset S.
        The condition is: for all a, b in S, a+b not in S.
        
        We can view this as: We have a set of "bad triples" T = {(a, b, c) | a+b=c, a,b,c in V}.
        We want to choose the largest subset S of V such that no triple in T is a subset of S.
        This is exactly the Maximum 3-Set Packing problem (or rather, its complement: finding the largest set that avoids containing any hyperedge).
        Wait, standard 3-Set Packing is: given a collection of sets of size 3, find the largest subcollection of pairwise disjoint sets.
        That's not it.
        
        Let's re-read: "no a + b = c with a, b, c in it".
        This means S is an independent set in the 3-uniform hypergraph defined by the triples.
        This is the Maximum Independent Set in a 3-uniform hypergraph.
        
        Algorithm:
        1. Precompute all triples (a, b, c) with a <= b, a+b=c, all in V.
        2. Branch on elements.
        
        Heuristic for branching:
        Pick an element v.
        Branch 1: Include v. Then we must exclude any pair (a, b) such that a+b=v or v+a=b or v+b=a.
        Specifically, if v is in S:
          - For any a in S, v+a cannot be in S.
          - For any a in S, v-a cannot be in S (if v-a is in V).
          - For any a in S, a-v cannot be in S (if a-v is in V).
        
        This suggests a recursive search where we maintain:
        - `chosen`: set of elements currently in S.
        - `forbidden`: set of elements that CANNOT be in S given `chosen`.
        - `available`: set of elements that can still be considered.
        
        When we add x to `chosen`:
        - Remove x from `available`.
        - For each y in `chosen`:
            - If x+y is in V, add x+y to `forbidden`.
            - If x-y is in V, add x-y to `forbidden`.
            - If y-x is in V, add y-x to `forbidden`.
        - Also, x itself is removed from `available`.
        
        We want to maximize |chosen|.
        
        Upper bound:
        |chosen| + |available| is a trivial bound.
        A tighter bound: 
        Consider the remaining `available` elements.
        If we pick a set of elements from `available`, they must not form a triple with each other or with `chosen`.
        Actually, the interaction with `chosen` is already handled by `forbidden`.
        So the problem reduces to: Find the largest subset of `available` that is sum-free AND disjoint from `forbidden`.
        But `available` is already disjoint from `forbidden` by definition (we remove from available when adding to forbidden).
        So we just need the largest sum-free subset of `available`.
        
        This is the same problem! So simple recursion doesn't give a good bound unless we use coloring or packing bounds.
        
        Let's use a greedy coloring bound on the hypergraph of `available`.
        Or, more simply, since n is small (<=240), we can use a heuristic search with good pruning.
        
        Let's try a different structural approach:
        The problem is equivalent to finding a maximum independent set in a graph where edges represent "conflicts"?
        No, because the conflict is a 3-way constraint.
        
        However, we can convert it to a graph MIS problem?
        If a+b=c, then a and c conflict? No, if a and c are in S, b might not be.
        The constraint is only violated if ALL THREE are in S.
        So it's not a graph problem. It's a hypergraph problem.
        
        Let's implement a branch and bound for the hypergraph MIS.
        
        State: (available, chosen_count, forbidden_mask)
        We represent `available` as a bitset (int) for speed.
        We represent `forbidden` as a bitset.
        
        Precomputation:
        For each element x in V, precompute:
        - `adds[x]`: set of y such that x+y in V. (y -> x+y)
        - `subs[x]`: set of y such that x-y in V. (y -> x-y)
        - `rsubs[x]`: set of y such that y-x in V. (y -> y-x)
        
        Actually, we can just precompute for each x, the set of pairs (u, v) that would be forbidden if x is chosen.
        If x is chosen, then for any u in chosen, u+x, u-x, x-u become forbidden.
        This is dynamic.
        
        Let's use a simpler recursive structure:
        def solve(available_bits, chosen_bits):
            if not available_bits:
                return popcount(chosen_bits)
            
            # Upper bound
            # We can use a greedy matching bound on the hyperedges within available.
            # Or just |available| is a weak bound.
            
            # Pick a pivot element from available.
            # Good pivot: one that participates in the most triples within available.
            
            # Branch 1: Include pivot
            # Branch 2: Exclude pivot
        """
        n = len(V)
        val_to_idx = {v: i for i, v in enumerate(V)}
        
        # Precompute triples
        # triple_mask[i] is a bitmask of the other two elements that form a triple with i?
        # No, let's precompute for each element i, the list of pairs (j, k) such that V[i]+V[j]=V[k] or V[i]+V[k]=V[j] etc.
        # Actually, we need to know: if we pick i, which pairs (j, k) in available become forbidden?
        # If i is in S, then for any j in S, i+j cannot be in S.
        # So if we pick i, and we already have some set C of chosen elements, 
        # then for each c in C, i+c, i-c, c-i are forbidden.
        
        # This state-dependency makes it hard to precompute a static bound.
        
        # Alternative: Use the fact that the graph of "sums" is sparse.
        # Let's try a simple recursive search with memoization on `available_bits`?
        # 2^240 is too big.
        
        # Let's use a heuristic:
        # 1. Greedy construction to get a lower bound.
        # 2. Local search to improve.
        
        # The current program uses greedy removal + hill climbing.
        # Inspiration 1 uses exact MIS on a conflict graph.
        # Wait, the conflict graph approach:
        # If a+b=c, then a and c are NOT necessarily in conflict. They can both be in S if b is not.
        # So the "conflict graph" where an edge means "cannot both be in S" is WRONG for sum-free sets.
        # Example: {1, 2, 3}. 1+2=3.
        # Can {1, 3} be sum-free? Yes. 1+1=2 (not in {1,3}), 1+3=4 (not), 3+3=6 (not).
        # So 1 and 3 do NOT conflict.
        # The conflict is between the TRIPLE {1, 2, 3}.
        # So the current program's `build_conflict_graph` is fundamentally flawed for this problem definition!
        # It treats it as a graph MIS, but it's a hypergraph MIS.
        # This explains why the score is not higher.
        
        # So, we MUST solve the hypergraph problem.
        
        # Strategy:
        # 1. Generate all triples (a, b, c) with a+b=c, a<=b.
        # 2. Use a branch and bound algorithm.
        
        # To make it fast:
        # - Use bitsets.
        # - Precompute for each element x, the set of elements y such that x+y is in V (let's call it Sums[x]).
        # - Precompute for each element x, the set of elements y such that x-y is in V (Diffs[x]).
        
        # Recursive function:
        # search(available, chosen_count)
        # available: int bitmask of elements still available to be chosen.
        # chosen_count: number of elements already chosen.
        
        # Upper bound:
        # We can compute a greedy independent set size on the hypergraph induced by `available`.
        # Or, simpler:
        # If we pick an element x from available:
        #   New available = available - {x} - (forbidden elements)
        #   Forbidden elements depend on `chosen` set, which we don't track explicitly in the signature.
        
        # We need to track `chosen` to compute forbidden elements.
        # But `chosen` can be large.
        
        # Let's try a different angle:
        # The problem is equivalent to: Maximize |S| such that S is sum-free.
        
        # We can use a SAT solver? No, stdlib only.
        
        # Let's use a heuristic search:
        # 1. Start with empty set.
        # 2. Greedily add elements that create the fewest new conflicts?
        #    Conflict of adding x to S:
        #    - x+x in S?
        #    - x+y in S for any y in S?
        #    - y+x in S for any y in S?
        #    - x-y in S for any y in S?
        #    - y-x in S for any y in S?
        
        # This is the `can_add` logic in the current program.
        
        # The key difference in the new strategy:
        # Instead of just hill climbing, we will use a **Randomized Restart with Simulated Annealing** 
        # on the set of excluded elements, or better, a **Local Search on the Selected Set** 
        # with **Swap Moves** (remove one, add one) and **K-Swap Moves**.
        
        # However, the most promising structural change is to correctly model the **Hypergraph**.
        # We will implement a **Branch and Bound** that is aware of the hyperedges.
        
        # Precompute:
        # triples: list of (i, j, k) indices such that V[i]+V[j]=V[k].
        
        # For each element i, let `neighbors[i]` be the set of elements j such that (i, j, k) is a triple for some k.
        # This is the 1-shadow of the hypergraph.
        
        # Algorithm:
        # 1. Sort elements by degree (number of triples they are part of).
        # 2. Branch on the element with highest degree.
        # 3. If we include it, we must exclude any pair (j, k) that completes a triple with it?
        #    No, if we include i, we don't immediately exclude j or k. We only exclude a triple if ALL THREE are present.
        #    So if we include i, we can still include j, provided we don't include k (if i+j=k).
        
        # This is tricky to prune.
        
        # Let's look at the size: n=240.
        # A simple recursive B&B might be too slow without strong bounds.
        
        # Let's use the **Maximum 2-Set Packing** approximation?
        
        # Actually, there is a known result: The sum-free subset problem is equivalent to finding an independent set in a 3-uniform hypergraph.
        
        # Let's try a **Genetic Algorithm** or **Memetic Algorithm**?
        
        # Given the 120s limit, and n=240, a heuristic local search is likely the only feasible way to get near-optimal.
        
        # The current program's flaw is the incorrect conflict graph.
        # We will remove the incorrect MIS solver and replace it with a **Robust Local Search** 
        # that correctly handles the sum-free constraint.
        
        # We will use **Simulated Annealing** to escape local optima.
        
        pass

    def local_search(V, time_limit):
        """
        Local search with Simulated Annealing.
        State: A set S of elements.
        Move: Remove an element, or Add an element, or Swap an element.
        Objective: Maximize |S| subject to sum-free constraint.
        """
        V_set = set(V)
        # Start with a greedy solution
        S = set()
        # Greedy: Add elements in random order if they don't violate constraint
        V_list = list(V)
        random.shuffle(V_list)
        for x in V_list:
            if can_add(x, S):
                S.add(x)
        
        best_S = set(S)
        best_size = len(S)
        
        # Simulated Annealing
        T = 10.0
        cooling = 0.999
        min_T = 0.01
        
        while time.time() < start_time + time_limit and T > min_T:
            # Generate a neighbor
            # 1. Remove a random element from S
            # 2. Add a random element from V\S
            # 3. Swap: Remove a from S, Add b from V\S
            
            if S:
                r = random.random()
                if r < 0.3:
                    # Remove
                    a = random.choice(list(S))
                    S.remove(a)
                elif r < 0.6:
                    # Add
                    candidates = [x for x in V if x not in S]
                    if candidates:
                        b = random.choice(candidates)
                        if can_add(b, S):
                            S.add(b)
                        else:
                            # If can't add, maybe swap?
                            pass
                else:
                    # Swap
                    if S and len(V_set - S) > 0:
                        a = random.choice(list(S))
                        candidates = [x for x in V if x not in S]
                        b = random.choice(candidates)
                        # Check if swap is valid
                        S.remove(a)
                        if can_add(b, S):
                            S.add(b)
                        else:
                            S.add(a) # revert
            else:
                # S is empty, add a random element
                b = random.choice(V_list)
                S.add(b)
            
            # Evaluate
            current_size = len(S)
            if current_size > best_size:
                best_size = current_size
                best_S = set(S)
            
            # Acceptance criterion
            # We always accept improvements.
            # We accept worse solutions with probability exp(-delta/T)
            # But here, the "energy" is -size. So delta = - (current_size - best_size)?
            # No, we compare to the PREVIOUS state, not the best.
            # We need to track previous size.
            
            # Let's restart the SA loop properly.
            break # Placeholder for now, will implement full SA below
            
        return list(best_S)

    def can_add(x, S):
        # Check if x can be added to S while keeping it sum-free
        S_set = S
        # Check x+x
        if x + x in S_set:
            return False
        # Check x+a and a+x (same) and x-a and a-x
        for a in S_set:
            if x + a in S_set:
                return False
            if x - a in S_set:
                return False
            if a - x in S_set:
                return False
        return True

    # Full Simulated Annealing Implementation
    
    def solve_instance(V, time_limit):
        V_set = set(V)
        
        # Greedy initialization
        S = set()
        V_list = list(V)
        random.shuffle(V_list)
        for x in V_list:
            if can_add(x, S):
                S.add(x)
        
        best_S = set(S)
        best_size = len(S)
        
        # Simulated Annealing
        T = 10.0
        cooling = 0.9995
        min_T = 0.01
        end_time = start_time + time_limit
        
        while time.time() < end_time and T > min_T:
            # Generate a move
            # 1. Add an element
            # 2. Remove an element
            # 3. Swap an element
            
            move_type = random.random()
            
            new_S = set(S)
            
            if move_type < 0.3:
                # Remove
                if S:
                    a = random.choice(list(S))
                    new_S.remove(a)
            elif move_type < 0.6:
                # Add
                candidates = [x for x in V if x not in S]
                if candidates:
                    b = random.choice(candidates)
                    if can_add(b, new_S):
                        new_S.add(b)
                    else:
                        # Revert to S if invalid
                        new_S = set(S)
            else:
                # Swap
                if S and len(V_set - S) > 0:
                    a = random.choice(list(S))
                    candidates = [x for x in V if x not in S]
                    b = random.choice(candidates)
                    new_S.remove(a)
                    if can_add(b, new_S):
                        new_S.add(b)
                    else:
                        new_S = set(S)
            
            # Evaluate
            new_size = len(new_S)
            current_size = len(S)
            
            if new_size > current_size:
                S = new_S
                if new_size > best_size:
                    best_size = new_size
                    best_S = set(S)
            else:
                # Accept with probability
                delta = new_size - current_size # negative or zero
                if delta < 0:
                    prob = 2.71828 ** (delta / T)
                    if random.random() < prob:
                        S = new_S
                    else:
                        # Keep old S
                        pass
                else:
                    # delta == 0, accept
                    S = new_S
            
            T *= cooling
        
        return list(best_S)

    results = []
    
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        V = instance(size, seed)
        
        # Allocate time
        elapsed = time.time() - start_time
        remaining = DEADLINE - elapsed
        if remaining < 1:
            # Emergency fallback
            S = set()
            for x in V:
                if can_add(x, S):
                    S.add(x)
            results.append(list(S))
            continue
            
        # Distribute time
        if size <= 60:
            t_alloc = min(20.0, remaining * 0.4)
        elif size <= 120:
            t_alloc = min(40.0, remaining * 0.4)
        else:
            t_alloc = min(80.0, remaining * 0.5)
            
        # Run multiple restarts of SA
        best_res = []
        best_len = 0
        restarts = 0
        while time.time() < start_time + DEADLINE - 5:
            res = solve_instance(V, t_alloc / (restarts + 1)) # Decreasing time per restart?
            # Actually, let's just run one long SA or multiple short ones.
            # Let's do 3 restarts with equal time.
            if len(res) > best_len:
                best_len = len(res)
                best_res = res
            restarts += 1
            if restarts >= 3:
                break
                
        # Verify
        if not is_sum_free(best_res):
            # Fallback
            S = set()
            for x in V:
                if can_add(x, S):
                    S.add(x)
            best_res = list(S)
            
        results.append(best_res)
        
    return results

# The above code has a flaw: `can_add` is called inside `solve_instance` but `S` is a set.
# `can_add` iterates over `S_set`. This is O(N).
# Total complexity of one SA step is O(N).
# Number of steps is limited by time.
# This should be fast enough for N=240.

# Let's refine the `can_add` to be faster if possible, but O(N) is fine.

# One issue: `solve_instance` uses `start_time` from the outer scope.
# This is correct.

# Let's double check the `can_add` logic.
# S is sum-free. We want to add x.
# New set is S U {x}.
# Violation if:
# 1. x + x in S
# 2. x + a in S for some a in S
# 3. a + x in S for some a in S (same as 2)
# 4. x - a in S for some a in S (implies a + (x-a) = x, so a, x-a, x in S U {x})
# 5. a - x in S for some a in S (implies x + (a-x) = a, so x, a-x, a in S U {x})
# Yes, the logic is correct.

# Let's run a quick mental check on performance.
# N=240.
# `can_add` is 240 iterations.
# SA step: 1 `can_add` call.
# 100,000 steps/sec?
# 120s -> 12,000,000 steps.
# Plenty of time.

# The key improvement is that we are NOT using the flawed MIS on the conflict graph.
# We are using a direct local search on the sum-free constraint.

# We will also add a "Greedy + Hill Climbing" baseline to ensure we have a valid solution.

def construct():
    start_time = time.time()
    DEADLINE = 115.0

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        S_set = set(S)
        for a in S:
            for b in S:
                if a + b in S_set:
                    return False
        return True

    def can_add(x, S):
        S_set = S
        if x + x in S_set:
            return False
        for a in S_set:
            if x + a in S_set:
                return False
            if x - a in S_set:
                return False
            if a - x in S_set:
                return False
        return True

    def solve_instance(V, time_limit):
        V_set = set(V)
        
        # Greedy initialization
        S = set()
        V_list = list(V)
        random.shuffle(V_list)
        for x in V_list:
            if can_add(x, S):
                S.add(x)
        
        best_S = set(S)
        best_size = len(S)
        
        # Simulated Annealing
        T = 10.0
        cooling = 0.9995
        min_T = 0.01
        end_time = start_time + time_limit
        
        while time.time() < end_time and T > min_T:
            move_type = random.random()
            new_S = set(S)
            
            if move_type < 0.3:
                if S:
                    a = random.choice(list(S))
                    new_S.remove(a)
            elif move_type < 0.6:
                candidates = [x for x in V if x not in S]
                if candidates:
                    b = random.choice(candidates)
                    if can_add(b, new_S):
                        new_S.add(b)
                    else:
                        new_S = set(S)
            else:
                if S and len(V_set - S) > 0:
                    a = random.choice(list(S))
                    candidates = [x for x in V if x not in S]
                    b = random.choice(candidates)
                    new_S.remove(a)
                    if can_add(b, new_S):
                        new_S.add(b)
                    else:
                        new_S = set(S)
            
            new_size = len(new_S)
            current_size = len(S)
            
            if new_size > current_size:
                S = new_S
                if new_size > best_size:
                    best_size = new_size
                    best_S = set(S)
            else:
                if new_size < current_size:
                    delta = new_size - current_size
                    prob = 2.71828 ** (delta / T)
                    if random.random() < prob:
                        S = new_S
                else:
                    S = new_S
            
            T *= cooling
        
        return list(best_S)

    results = []
    
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        V = instance(size, seed)
        
        elapsed = time.time() - start_time
        remaining = DEADLINE - elapsed
        if remaining < 1:
            S = set()
            for x in V:
                if can_add(x, S):
                    S.add(x)
            results.append(list(S))
            continue
            
        if size <= 60:
            t_alloc = min(20.0, remaining * 0.4)
        elif size <= 120:
            t_alloc = min(40.0, remaining * 0.4)
        else:
            t_alloc = min(80.0, remaining * 0.5)
            
        best_res = []
        best_len = 0
        restarts = 0
        while time.time() < start_time + DEADLINE - 5:
            res = solve_instance(V, t_alloc / (restarts + 1))
            if len(res) > best_len:
                best_len = len(res)
                best_res = res
            restarts += 1
            if restarts >= 3:
                break
                
        if not is_sum_free(best_res):
            S = set()
            for x in V:
                if can_add(x, S):
                    S.add(x)
            best_res = list(S)
            
        results.append(best_res)
        
    return results
