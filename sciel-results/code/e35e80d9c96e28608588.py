import time
import random

def construct():
    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def can_add(x, S_set):
        # Check if adding x to S_set keeps it sum-free
        for a in S_set:
            if (x - a) in S_set:
                return False
        for a in S_set:
            if (a + x) in S_set:
                return False
        if (2 * x) in S_set:
            return False
        return True

    def get_suffix(S):
        n = len(S)
        best_k = 1
        for k in range(1, n + 1):
            if 2 * S[n - k] > S[n - 1]:
                best_k = k
            else:
                break
        return S[n - best_k : n]

    def compute_conflicts(S_set):
        """Return dict: element -> set of elements it conflicts with (a+b=c type)"""
        conflicts = {x: set() for x in S_set}
        for a in S_set:
            for b in S_set:
                c = a + b
                if c in S_set:
                    conflicts[a].add(c)
                    conflicts[b].add(c)
                    conflicts[c].add(a)
                    conflicts[c].add(b)
                # Also handle a+b where a==b
                if a == b:
                    c2 = 2 * a
                    if c2 in S_set:
                        conflicts[a].add(c2)
                        conflicts[c2].add(a)
        return conflicts

    def count_conflicts(S_set):
        """Count number of violated sum-free constraints"""
        count = 0
        for a in S_set:
            for b in S_set:
                if a + b in S_set:
                    count += 1
        return count

    def walk_sat_search(S, time_limit, rng):
        """WalkSAT-style local search with restarts"""
        n = len(S)
        S_set = set(S)
        remaining = [x for x in S if x not in S_set]
        
        start_time = time.time()
        best_size = len(S_set)
        best_set = set(S_set)
        
        # Precompute all elements
        all_elements = set(S)
        
        max_restarts = 50
        max_steps_per_restart = 2000
        
        for restart in range(max_restarts):
            if time.time() - start_time > time_limit:
                break
            
            # Initialize: start from current best or greedy
            if restart == 0:
                current = set(best_set)
            else:
                # Perturbation: randomly remove some elements and re-add
                current = set(best_set)
                if current:
                    to_remove = min(len(current) // 10, 5)
                    for _ in range(to_remove):
                        if current:
                            current.remove(rng.choice(list(current)))
                # Try to add some random elements
                candidates = [x for x in all_elements if x not in current]
                for _ in range(min(10, len(candidates))):
                    if candidates:
                        x = rng.choice(candidates)
                        candidates.remove(x)
                        current.add(x)
            
            # Local search phase
            for step in range(max_steps_per_restart):
                if time.time() - start_time > time_limit:
                    break
                
                # Try to add elements
                add_candidates = [x for x in all_elements if x not in current]
                if add_candidates:
                    # Weight by how many conflicts they'd create (we want 0)
                    # Just try random additions
                    rng.shuffle(add_candidates)
                    added = False
                    for x in add_candidates[:20]:
                        if can_add(x, current):
                            current.add(x)
                            added = True
                            break
                    if added:
                        if len(current) > best_size:
                            best_size = len(current)
                            best_set = set(current)
                        continue
                
                # Try swaps: remove one, add one
                # Pick element with most conflicts to remove
                if current:
                    # Find elements involved in conflicts
                    conflict_elements = set()
                    for a in current:
                        for b in current:
                            if a + b in current:
                                conflict_elements.add(a)
                                conflict_elements.add(b)
                                conflict_elements.add(a + b)
                    
                    if conflict_elements:
                        # Remove one conflict element
                        rem_elem = rng.choice(list(conflict_elements))
                        current.discard(rem_elem)
                        # Try to add a new element
                        add_cands = [x for x in all_elements if x not in current]
                        if add_cands:
                            rng.shuffle(add_cands)
                            for x in add_cands[:15]:
                                if can_add(x, current):
                                    current.add(x)
                                    break
                        if len(current) > best_size:
                            best_size = len(current)
                            best_set = set(current)
                    else:
                        # No conflicts, try pure additions
                        add_cands = [x for x in all_elements if x not in current]
                        if add_cands:
                            rng.shuffle(add_cands)
                            for x in add_cands[:20]:
                                if can_add(x, current):
                                    current.add(x)
                                    break
                            if len(current) > best_size:
                                best_size = len(current)
                                best_set = set(current)
                        else:
                            break  # Can't add anything more
                else:
                    break
        
        return list(best_set)

    def solve_instance(S, time_limit):
        # Multiple initialization strategies
        # 1. Odds
        odds = [x for x in S if x % 2 != 0]
        
        # 2. Suffix (largest values)
        suffix = get_suffix(S)
        
        # 3. Greedy sorted
        greedy_init = []
        S_set = set()
        for x in S:
            if can_add(x, S_set):
                S_set.add(x)
                greedy_init.append(x)
        
        # 4. Greedy by absolute value (small first)
        greedy_abs = []
        S_set2 = set()
        for x in sorted(S, key=abs):
            if can_add(x, S_set2):
                S_set2.add(x)
                greedy_abs.append(x)
        
        # 5. Greedy by negative values first
        greedy_neg = []
        S_set3 = set()
        for x in sorted(S, reverse=True):
            if can_add(x, S_set3):
                S_set3.add(x)
                greedy_neg.append(x)
        
        candidates = [odds, suffix, greedy_init, greedy_abs, greedy_neg]
        best_init = max(candidates, key=len)
        
        # Run WalkSAT search with multiple seeds
        rng = random.Random()
        improved = walk_sat_search(best_init, time_limit, rng)
        
        return improved

    # Instances
    instances = [
        instance(60, 801),
        instance(120, 802),
        instance(240, 803)
    ]
    
    results = []
    total_time = 145  # Safety margin from 150s kill
    start = time.time()
    
    # Allocate time proportionally to difficulty
    time_allocs = [15, 30, 95]  # 60, 120, 240 instances
    
    for i, S in enumerate(instances):
        elapsed = time.time() - start
        remaining_time = total_time - elapsed - 3  # 3s safety margin
        
        if remaining_time <= 0:
            # Fallback to odds
            results.append([x for x in S if x % 2 != 0])
            continue
        
        # Use allocated time or remaining, whichever is smaller
        alloc = time_allocs[i] if i < len(time_allocs) else remaining_time
        t_limit = min(alloc, remaining_time)
        
        if t_limit <= 0:
            results.append([x for x in S if x % 2 != 0])
            continue
            
        res = solve_instance(S, t_limit)
        results.append(res)
        
    return results

# EVOLVE-BLOCK-END
