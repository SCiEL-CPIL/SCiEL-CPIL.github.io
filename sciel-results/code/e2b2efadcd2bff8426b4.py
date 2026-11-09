import time
from random import Random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    def instance(size, seed, limit=10000):
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)
    
    def solve_instance(vals):
        n = len(vals)
        if n == 0:
            return []
        
        # Precompute: for each pair (i,j), does vals[i] + vals[j] exist in vals?
        # And for each i, which j make vals[i] + vals[j] in vals?
        val_set = set(vals)
        
        # conflict[i] = set of j such that vals[i] + vals[j] is in val_set
        # This means if we have both i and j, we can't have the element vals[i]+vals[j]
        # But for the sum-free property, we need: no a,b,c in subset with a+b=c
        # So if a and b are in subset, then a+b must NOT be in subset
        
        # Let's precompute for each index i, the list of indices j such that vals[i]+vals[j] is in vals
        # And also for each index i, the list of pairs (j,k) where vals[j]+vals[k]=vals[i]
        
        # Actually, let me think differently. For a candidate set S:
        # S is sum-free iff for all a,b in S: a+b not in S
        # Equivalent: for all a in S: for all b in S: a+b not in S
        
        # For efficient checking, I'll maintain the set of values and check incrementally.
        
        # Precompute sum relationships:
        # sum_to[i] = set of j such that vals[j] + vals[i] is in val_set
        # i.e., if we have j and i, we conflict with vals[j]+vals[i]
        
        sum_to = [set() for _ in range(n)]
        for i in range(n):
            for j in range(n):
                s = vals[i] + vals[j]
                if s in val_set:
                    # Find index of s
                    # We need to know which index has value s
                    pass
        
        # Better: precompute val_to_idx
        val_to_idx = {v: i for i, v in enumerate(vals)}
        
        # conflict[i] = set of j such that vals[i] + vals[j] is in vals
        # If both i and j are in S, then vals[i]+vals[j] must not be in S
        # But vals[i]+vals[j] might be some vals[k], so we'd need k not in S
        
        # Let me precompute: for each i, the set of k such that there exists j with vals[i]+vals[j]=vals[k]
        # i.e., k = index of vals[i]+vals[j] for some j
        
        # Actually, the cleanest approach: 
        # For a set S (as set of indices), S is sum-free iff:
        # For all i,j in S: if vals[i]+vals[j] is in val_set, then the index of vals[i]+vals[j] is not in S
        
        # Precompute: sum_index[i][j] = k if vals[i]+vals[j] == vals[k], else -1
        # This is O(n^2) which is fine for n up to 240
        
        sum_index = [[-1] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                s = vals[i] + vals[j]
                if s in val_to_idx:
                    sum_index[i][j] = val_to_idx[s]
        
        def is_sum_free(idx_set):
            """Check if the set of indices is sum-free."""
            for i in idx_set:
                for j in idx_set:
                    k = sum_index[i][j]
                    if k != -1 and k in idx_set:
                        return False
            return True
        
        def can_add(idx, current_set):
            """Check if adding idx to current_set keeps it sum-free."""
            # Need to check: for all j in current_set, if vals[idx]+vals[j] in val_set, 
            # then the index of that sum is not in current_set
            # Also: for all j in current_set, if vals[j]+vals[idx] in val_set (same thing)
            # And: for all j in current_set, if vals[idx]-vals[j] in val_set, then... 
            # Wait, the condition is a+b=c, so we need to check all pairs involving idx
            
            for j in current_set:
                # Check idx + j
                k = sum_index[idx][j]
                if k != -1 and k in current_set:
                    return False
                # Check j + idx (same as above since addition is commutative, but sum_index might differ)
                # Actually sum_index[j][idx] should be same as sum_index[idx][j] since vals[j]+vals[idx]=vals[idx]+vals[j]
                # But let's be safe
                k2 = sum_index[j][idx]
                if k2 != -1 and k2 in current_set:
                    return False
            return True
        
        # Greedy initialization with multiple strategies
        def greedy_fill(order):
            current = set()
            for idx in order:
                if can_add(idx, current):
                    current.add(idx)
            return current
        
        # Generate multiple orderings
        orderings = []
        
        # 1. By absolute value descending
        orderings.append(sorted(range(n), key=lambda i: abs(vals[i]), reverse=True))
        
        # 2. Positive values descending, then negative ascending
        pos = [i for i in range(n) if vals[i] > 0]
        neg = [i for i in range(n) if vals[i] < 0]
        orderings.append(sorted(pos, key=lambda i: vals[i], reverse=True) + sorted(neg, key=lambda i: vals[i]))
        
        # 3. Positive values ascending, then negative descending
        orderings.append(sorted(pos, key=lambda i: vals[i]) + sorted(neg, key=lambda i: vals[i], reverse=True))
        
        # 4. Random orderings
        rng = Random(42)
        for _ in range(20):
            order = list(range(n))
            rng.shuffle(order)
            orderings.append(order)
        
        # 5. Order by "conflict degree" - number of conflicts with other elements
        # An element with fewer conflicts is easier to include
        conflict_count = [0] * n
        for i in range(n):
            for j in range(n):
                if i != j and sum_index[i][j] != -1:
                    conflict_count[i] += 1
        
        orderings.append(sorted(range(n), key=lambda i: conflict_count[i]))
        orderings.append(sorted(range(n), key=lambda i: -conflict_count[i]))
        
        # Find best greedy solution
        best_set = set()
        best_size = 0
        for order in orderings:
            if time.time() > deadline - 30:
                break
            s = greedy_fill(order)
            if len(s) > best_size:
                best_size = len(s)
                best_set = s
        
        # Local search: 1-swap moves (remove one, add one)
        # and 2-swap moves (remove one, add two, or remove two, add one)
        
        def local_search(current_set, max_rounds=50):
            current = set(current_set)
            improved = True
            rounds = 0
            while improved and rounds < max_rounds and time.time() < deadline - 15:
                improved = False
                rounds += 1
                
                # Try 1-swap: remove i, add j
                # For each i in current, for each j not in current, try swap
                # This is O(n^2) per round, which is fine for n=240
                
                # To speed up, we can try in random order
                remove_order = list(current)
                rng.shuffle(remove_order)
                
                for i in remove_order:
                    if time.time() > deadline - 15:
                        return current
                    
                    # Try to add each j not in current
                    for j in range(n):
                        if j in current:
                            continue
                        
                        # Try: remove i, add j
                        new_set = current - {i}
                        new_set.add(j)
                        
                        # Check if new_set is sum-free
                        # We can do a quick check
                        valid = True
                        for a in new_set:
                            for b in new_set:
                                k = sum_index[a][b]
                                if k != -1 and k in new_set:
                                    valid = False
                                    break
                            if not valid:
                                break
                        
                        if valid and len(new_set) > len(current):
                            current = new_set
                            improved = True
                            break
                    if improved:
                        break
                
                # Also try pure additions: for each j not in current, try adding
                if not improved:
                    add_order = [j for j in range(n) if j not in current]
                    rng.shuffle(add_order)
                    for j in add_order:
                        if time.time() > deadline - 15:
                            return current
                        if can_add(j, current):
                            current.add(j)
                            improved = True
        
        # Run local search from best greedy
        if time.time() < deadline - 20:
            best_set = local_search(best_set)
            best_size = len(best_set)
        
        # Simulated annealing for further improvement
        if time.time() < deadline - 10:
            current = set(best_set)
            current_size = len(current)
            best_sa = set(current)
            best_sa_size = current_size
            
            T = 5.0
            T_min = 0.1
            cooling = 0.995
            max_iters = 5000
            
            for it in range(max_iters):
                if time.time() > deadline - 5:
                    break
                if T < T_min:
                    break
                
                # Generate a move: remove a random element from current, or add a random element not in current
                if current:
                    move_type = rng.random()
                    if move_type < 0.5 and len(current) > 1:
                        # Remove a random element
                        i = rng.choice(list(current))
                        new_set = current - {i}
                    elif move_type < 0.8:
                        # Add a random element not in current
                        candidates = [j for j in range(n) if j not in current]
                        if not candidates:
                            continue
                        j = rng.choice(candidates)
                        new_set = current | {j}
                    else:
                        # Swap: remove one, add one
                        if current:
                            i = rng.choice(list(current))
                            candidates = [j for j in range(n) if j not in current]
                            if not candidates:
                                continue
                            j = rng.choice(candidates)
                            new_set = current - {i}
                            new_set.add(j)
                        else:
                            candidates = list(range(n))
                            j = rng.choice(candidates)
                            new_set = {j}
                else:
                    j = rng.randrange(n)
                    new_set = {j}
                
                # Check if new_set is sum-free
                valid = True
                for a in new_set:
                    for b in new_set:
                        k = sum_index[a][b]
                        if k != -1 and k in new_set:
                            valid = False
                            break
                    if not valid:
                        break
                
                if not valid:
                    continue
                
                new_size = len(new_set)
                delta = new_size - current_size
                
                if delta >= 0 or rng.random() < pow(2.718281828, delta / T):
                    current = new_set
                    if new_size > best_sa_size:
                        best_sa_size = new_size
                        best_sa = set(current)
                
                T *= cooling
            
            if best_sa_size > best_size:
                best_set = best_sa
                best_size = best_sa_size
        
        return [vals[i] for i in sorted(best_set)]
    
    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        if time.time() > deadline - 5:
            # Fallback
            vals = instance(size, seed)
            results.append([x for x in vals if x % 2 != 0])
            continue
        
        vals = instance(size, seed)
        results.append(solve_instance(vals))
    
    return results
