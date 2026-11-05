"""Simulated annealing approach for large sum-free subsets."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    start_time = time.time()
    DEADLINE = 145  # Safety margin before 150s kill

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def search_instance(elements, seed_val, time_budget):
        """Simulated annealing to find large sum-free subset."""
        rng = random.Random(seed_val)
        n = len(elements)
        
        # Precompute all sums: for each pair (i,j), what index does elements[i]+elements[j] map to?
        # This allows O(1) lookup of whether a+b=c
        elem_to_idx = {v: i for i, v in enumerate(elements)}
        
        # Precompute sum targets: sum_target[i][j] = index of elements[i]+elements[j] or -1
        # Actually, let's just use a set for fast lookup
        elem_set = set(elements)
        
        def count_violations(mask):
            """Count number of (a,b,c) with a+b=c, all in mask. Use a=b allowed."""
            S = set()
            for i in range(n):
                if mask[i]:
                    S.add(elements[i])
            count = 0
            for a in S:
                for b in S:
                    if a + b in S:
                        count += 1
            return count

        # Better: maintain violation count incrementally
        # Let's use a different approach: track for each element c, how many pairs (a,b) in S satisfy a+b=c
        # When we add x: new violations = #{(a,b) in S: a+b=x} + #{(a) in S: a+x in S} + (2x in S)
        # When we remove x: remove those violations
        
        # Actually for n up to 240, O(n^2) per evaluation is fine.
        # Let's just do full evaluation but be smart about it.
        
        def objective(mask):
            """Return (size, -violation_count). We want to maximize size subject to violation=0."""
            S = [elements[i] for i in range(n) if mask[i]]
            size = len(S)
            S_set = set(S)
            viol = 0
            for a in S:
                for b in S:
                    if a + b in S_set:
                        viol += 1
            return size, viol

        # Simulated annealing
        # State: mask (list of 0/1)
        # Move: flip one bit
        # Objective: minimize penalty = max(0, viol) * BIG - size
        # Actually: we want max size with viol=0. Use penalty method:
        # score = size - penalty * viol where penalty is large
        
        # Initialize: random mask with ~60% inclusion
        p_init = 0.6
        mask = [1 if rng.random() < p_init else 0 for _ in range(n)]
        
        # Compute initial state
        def compute_stats(mask):
            S = [elements[i] for i in range(n) if mask[i]]
            size = len(S)
            S_set = set(S)
            viol = 0
            for a in S:
                for b in S:
                    if a + b in S_set:
                        viol += 1
            return size, viol

        cur_size, cur_viol = compute_stats(mask)
        cur_score = cur_size - 1000 * cur_viol  # Heavy penalty for violations
        
        best_mask = mask[:]
        best_score = cur_score
        best_size = cur_size if cur_viol == 0 else 0
        
        # If initial is sum-free, record it
        if cur_viol == 0:
            best_size = cur_size
        
        # Annealing schedule
        T_start = 50.0
        T_end = 0.01
        # Number of iterations - use time budget
        # Estimate: each iteration is O(n^2) for recompute. For n=240, that's ~57600 ops.
        # In Python, maybe ~50us per iteration? Let's be conservative: 100us
        # For 240 elements, we can do maybe 100k iterations in 10 seconds.
        # Let's just run until time budget.
        
        max_iter = 50000
        alpha = (T_end / T_start) ** (1.0 / max_iter)
        
        for it in range(max_iter):
            if time.time() - start_time > DEADLINE:
                break
            if it % 1000 == 0 and time.time() - start_time > DEADLINE - 5:
                break
                
            T = T_start * (alpha ** it)
            
            # Choose a random position to flip
            pos = rng.randint(0, n - 1)
            old_val = mask[pos]
            mask[pos] = 1 - old_val
            
            new_size, new_viol = compute_stats(mask)
            new_score = new_size - 1000 * new_viol
            
            delta = new_score - cur_score
            
            if delta > 0 or rng.random() < math.exp(delta / T):
                # Accept
                cur_size, cur_viol = new_size, new_viol
                cur_score = new_score
                if cur_viol == 0 and cur_size > best_size:
                    best_size = cur_size
                    best_mask = mask[:]
                    best_score = cur_score
            else:
                # Reject
                mask[pos] = old_val
        
        # Final: if best_mask is sum-free, try to add more elements greedily
        result = [elements[i] for i in range(n) if best_mask[i]]
        
        # Greedy addition pass
        result_set = set(result)
        changed = True
        while changed:
            changed = False
            for x in elements:
                if x not in result_set:
                    # Check if adding x creates a violation
                    can_add = True
                    for a in result:
                        if x - a in result_set:
                            can_add = False
                            break
                        if a + x in result_set:
                            can_add = False
                            break
                    if can_add and 2 * x not in result_set:
                        result.append(x)
                        result_set.add(x)
                        changed = True
        
        return result

    # Generate instances
    instances = []
    for size, seed in [(60, 801), (120, 802), (240, 803)]:
        instances.append(instance(size, seed))
    
    results = []
    for i, elems in enumerate(instances):
        if time.time() - start_time > DEADLINE:
            # Fallback: odd numbers
            results.append([x for x in elems if x % 2 != 0])
            continue
        
        # Allocate time budget: more time for larger instances
        remaining = DEADLINE - (time.time() - start_time)
        # Give proportionally more time to larger instances
        if i == 0:
            time_budget = remaining * 0.2
        elif i == 1:
            time_budget = remaining * 0.3
        else:
            time_budget = remaining * 0.5
        
        result = search_instance(elems, seed_val=801 + i, time_budget=time_budget)
        results.append(result)
    
    return results
# EVOLVE-BLOCK-END
