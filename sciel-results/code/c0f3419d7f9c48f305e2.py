"""Sum-free subset solver using conflict-graph local search with simulated annealing."""
import time
import random

def construct():
    start = time.time()
    deadline = start + 140.0  # safety margin before 150s limit

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def solve_one(values, seed):
        n = len(values)
        val_set = set(values)
        idx_map = {v: i for i, v in enumerate(values)}
        
        # Precompute conflict hyperedges: for each pair (i,j), check if values[i]+values[j] is in set
        # Store as: conflicts[i] = set of j such that values[i] + values[j] is in val_set
        # Also need: for each i, which j have 2*values[i] == values[j] (self-conflicts)
        
        # Build conflict structure:
        # conflict[i][j] = True if values[i] + values[j] is in val_set (i.e., there exists k with values[k] = values[i]+values[j])
        # For sum-free: we need no i,j,k (not necessarily distinct) with values[i]+values[j]=values[k] all in our subset
        
        # More efficient: for each element i, precompute:
        # - add_conflicts[i]: set of j such that values[i] + values[j] is in val_set
        #   (if both i and j are in subset, and values[i]+values[j] is also in subset, conflict)
        # - Actually we need: if i and j are in subset, then values[i]+values[j] must NOT be in subset
        #   So: conflict exists if i, j, k all in subset and values[i]+values[j] = values[k]
        
        # Let's build: for each pair (i, j), does values[i]+values[j] exist in values?
        # If yes, then we cannot have i, j, and the index of values[i]+values[j] all in our subset.
        
        # For local search, we need fast evaluation of: given a subset S, how many conflicts does it have?
        # A conflict is a triple (i,j,k) all in S with values[i]+values[j]=values[k]
        
        # Precompute for each i: 
        #   partners[i] = list of (j, k) such that values[i]+values[j]=values[k]
        # Then conflict count = sum over i in S of (number of j in S such that (j,k) in partners[i] and k in S)
        # But this overcounts; each triple counted 3 times (once for each element as the "sum" element)
        # Actually: for triple (i,j,k) with i+j=k, it's counted when we look at i (find j,k), 
        # when we look at j (find i,k), and when we look at k (find i,j where i+j=k... no, k is the sum)
        
        # Let me redefine: 
        # For each element k, precompute: pairs[k] = list of (i,j) such that values[i]+values[j]=values[k]
        # Then a conflict exists if k in S and there exist i,j in S with (i,j) in pairs[k]
        
        # For fast local search:
        # Maintain: for each k, count[k] = number of pairs (i,j) with i,j in S and i+j=k
        # Conflict count = number of k in S with count[k] > 0
        
        # When we add element x to S:
        #   - For each y in S: if x+y in S, increment count[x+y]
        #   - For each y in S: if x+y not in S but x+y in val_set, increment count[x+y] (but x+y not in S so no new conflict from this)
        #   - Check if count[x] > 0 (was x already a sum of two elements in S?)
        #   - New conflicts = (count[x] > 0) + (number of y in S such that x+y in S)
        
        # When we remove element x from S:
        #   - For each y in S: if x+y in S, decrement count[x+y]
        #   - Decrement count[x] by... wait, count[x] counts pairs summing to x, not pairs involving x
        #   - Actually removing x: for each y in S (after removal), if x+y was in S, decrement count[x+y]
        #   - And x itself may have been a sum: count[x] doesn't change from removing x (x is not a pair sum unless some pair sums to x)
        
        # This is getting complex. Let me use a simpler but still efficient approach:
        # For n <= 240, we can afford O(n) per move evaluation.
        
        # Precompute: for each i, the set of j such that values[i] + values[j] is in val_set
        # And for each i, the index of values[i]+values[j] if it exists
        
        sum_idx = {}  # (i,j) -> k if values[i]+values[j] = values[k]
        for i in range(n):
            for j in range(i, n):
                s = values[i] + values[j]
                if s in idx_map:
                    sum_idx[(i, j)] = idx_map[s]
        
        # For fast conflict check: 
        # Given subset S (as a set of indices), count conflicts:
        # A conflict is a triple (i,j,k) in S with i+j=k (values[i]+values[j]=values[k])
        # Note: i and j can be the same (2i=k)
        
        def count_conflicts(S):
            """Count number of conflicting triples in S."""
            S_set = set(S)
            count = 0
            for k in S:
                # Check if there exist i, j in S with values[i]+values[j] = values[k]
                # i.e., for some i in S, values[k]-values[i] is in S
                for i in S:
                    if i == k:
                        continue  # skip, handled separately or included
                    diff = values[k] - values[i]
                    if diff in val_set:
                        j = idx_map[diff]
                        if j in S_set:
                            count += 1
            # Each triple (i,j,k) with i+j=k is counted once here (for each k, each i that works)
            # But (i,j,k) and (j,i,k) are the same triple if we consider unordered pairs
            # Actually in the sum-free condition, a and b are ordered? No, a+b=c is symmetric in a,b.
            # The condition is: no a,b,c in S with a+b=c. So (a,b,c) and (b,a,c) are the same violation.
            # In my count, for a given k and i, I check if j=k-i is in S. This counts each unordered pair {i,j} once per k.
            # But if i=j, it's counted once. If i≠j, it's counted once (for the specific i we iterate over).
            # Wait, I iterate over all i in S, so for triple (i,j,k) with i≠j, it's counted when i is the loop variable AND when j is the loop variable? No:
            # When i is the loop variable: check if k-i = j is in S -> yes, count++
            # When j is the loop variable: check if k-j = i is in S -> yes, count++
            # So each unordered pair {i,j} with i≠j is counted twice!
            # And {i,i} is counted once.
            # For sum-free, we just need to know if ANY conflict exists, not the exact count.
            # But for local search, having a good conflict count helps.
            # Let me just check if any conflict exists (return 0 or 1 for simplicity, or count properly).
            return count
        
        # For local search, let's use a different approach:
        # Maintain the subset as a set. For each move (add/remove), check if it creates/removes conflicts.
        
        # Start with all odds
        odds = [i for i in range(n) if values[i] % 2 != 0]
        evens = [i for i in range(n) if values[i] % 2 == 0]
        
        current = set(odds)
        
        # Verify current is sum-free
        def is_sum_free_indices(S):
            S_set = set(S)
            for a in S:
                for b in S:
                    s = values[a] + values[b]
                    if s in val_set:
                        k = idx_map[s]
                        if k in S_set:
                            return False
            return True
        
        # Greedy: try to add evens
        for e in evens:
            if time.time() > deadline:
                break
            test_set = current | {e}
            if is_sum_free_indices(test_set):
                current = test_set
        
        best_set = set(current)
        best_size = len(current)
        
        # Local search with simulated annealing
        rng = random.Random(seed * 104729 + 1)
        T = 1.0
        cooling = 0.9995
        T_min = 0.01
        
        for step in range(5000):
            if time.time() > deadline:
                break
            if T < T_min:
                break
            
            # Choose a random move: add a random element not in current, or remove a random element in current
            if rng.random() < 0.5 and len(current) < n:
                # Add
                candidates = [i for i in range(n) if i not in current]
                if not candidates:
                    continue
                v = rng.choice(candidates)
                test_set = current | {v}
            else:
                # Remove
                if not current:
                    continue
                v = rng.choice(list(current))
                test_set = current - {v}
            
            # Evaluate
            if is_sum_free_indices(test_set):
                # Accept if sum-free
                if len(test_set) > best_size:
                    best_size = len(test_set)
                    best_set = set(test_set)
                current = test_set
                T = max(T * cooling, T_min)
            else:
                # Reject (or accept with probability exp(-delta/T) if we had a cost function)
                # For now, just reject non-sum-free sets
                T = max(T * cooling, T_min)
        
        # Multi-start: try different initial subsets
        for restart in range(300):
            if time.time() > deadline:
                break
            
            # Different initialization strategies
            if restart % 5 == 0:
                init = set(odds)
            elif restart % 5 == 1:
                # Random subset of size n//3
                k = max(1, n // 3)
                init = set(rng.sample(range(n), k))
            elif restart % 5 == 2:
                # All odds, random subset
                k = max(1, len(odds) // 2)
                init = set(rng.sample(odds, k))
            elif restart % 5 == 3:
                # Large random subset
                k = n // 2
                init = set(rng.sample(range(n), k))
            else:
                # Perturb best
                init = set(best_set)
                for _ in range(3):
                    if init and rng.random() < 0.5:
                        v = rng.choice(list(init))
                        init.discard(v)
                    else:
                        candidates = [i for i in range(n) if i not in init]
                        if candidates:
                            init.add(rng.choice(candidates))
            
            # Greedy fill
            for v in range(n):
                if v in init:
                    continue
                test_set = init | {v}
                if is_sum_free_indices(test_set):
                    init = test_set
            
            # Local search
            T_local = 0.5
            for step in range(2000):
                if time.time() > deadline:
                    break
                if T_local < 0.01:
                    break
                
                if rng.random() < 0.5 and len(init) < n:
                    candidates = [i for i in range(n) if i not in init]
                    if not candidates:
                        continue
                    v = rng.choice(candidates)
                    test_set = init | {v}
                else:
                    if not init:
                        continue
                    v = rng.choice(list(init))
                    test_set = init - {v}
                
                if is_sum_free_indices(test_set):
                    if len(test_set) > best_size:
                        best_size = len(test_set)
                        best_set = set(test_set)
                    init = test_set
                    T_local = max(T_local * 0.999, 0.01)
                else:
                    T_local = max(T_local * 0.999, 0.01)
            
            if len(init) > best_size:
                best_size = len(init)
                best_set = set(init)
        
        return sorted([values[i] for i in best_set])

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        if time.time() > deadline:
            vals = instance(size, seed)
            results.append([x for x in vals if x % 2])
            continue
        vals = instance(size, seed)
        results.append(solve_one(vals, seed))

    # Final validation
    for i, subset in enumerate(results):
        sv_set = set(subset)
        valid = True
        for a in subset:
            for b in subset:
                if a + b in sv_set:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            size, seed = ((60, 801), (120, 802), (240, 803))[i]
            vals = instance(size, seed)
            results[i] = [x for x in vals if x % 2]

    return results
