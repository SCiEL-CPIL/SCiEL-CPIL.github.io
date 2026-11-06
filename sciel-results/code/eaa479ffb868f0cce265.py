import time
import random
import math

def compute_metrics(A):
    """Compute max(|A+A|, |A*A|) for a set A of distinct positive integers."""
    sum_set = set()
    prod_set = set()
    n = len(A)
    for i in range(n):
        a = A[i]
        for j in range(i, n):
            b = A[j]
            sum_set.add(a + b)
            prod_set.add(a * b)
    return max(len(sum_set), len(prod_set))

def compute_metrics_incremental(A, old_val, new_val):
    """
    Compute new score when replacing old_val with new_val in set A.
    A is a list of 20 distinct positive integers.
    Returns the new max(|A+A|, |A*A|).
    """
    # First, compute the current sum and product sets without old_val
    # Then add contributions of new_val
    n = len(A)
    
    # Build sum_set and prod_set from A without old_val
    sum_set = set()
    prod_set = set()
    others = [x for x in A if x != old_val]
    
    for i in range(len(others)):
        a = others[i]
        for j in range(i, len(others)):
            b = others[j]
            sum_set.add(a + b)
            prod_set.add(a * b)
    
    # Now add contributions involving new_val
    sum_set_new = set(sum_set)
    prod_set_new = set(prod_set)
    
    sum_set_new.add(2 * new_val)  # new_val + new_val
    prod_set_new.add(new_val * new_val)  # new_val * new_val
    
    for x in others:
        sum_set_new.add(new_val + x)
        prod_set_new.add(new_val * x)
    
    return max(len(sum_set_new), len(prod_set_new))

def generate_smooth_pool(max_val=10000):
    """Generate 2,3,5-smooth numbers <= max_val."""
    smooth = set()
    for i in range(0, 14):  # 2^i
        for j in range(0, 10):  # 3^j
            for k in range(0, 7):  # 5^k
                val = (1 << i) * (3 ** j) * (5 ** k)
                if val <= max_val and val >= 1:
                    smooth.add(val)
    return sorted(smooth)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    # Generate smooth number pool (2,3,5-smooth)
    smooth_pool = generate_smooth_pool(10000)
    
    # Also add some small numbers and other structured values
    extra_pool = set()
    for v in range(1, 101):
        extra_pool.add(v)
    for base in [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30, 40, 50, 60, 80, 100]:
        for k in range(1, 101):
            v = base * k
            if v <= 10000:
                extra_pool.add(v)
    
    full_pool = sorted(set(smooth_pool) | extra_pool)
    
    # Initial candidates: subsets of smooth numbers
    initial_candidates = []
    
    # Take first 20 smooth numbers
    if len(smooth_pool) >= 20:
        initial_candidates.append(smooth_pool[:20])
    
    # Take evenly spaced smooth numbers
    if len(smooth_pool) >= 20:
        step = max(1, len(smooth_pool) // 20)
        cand = [smooth_pool[i * step] for i in range(20)]
        if len(set(cand)) == 20 and all(x <= 10000 for x in cand):
            initial_candidates.append(sorted(cand))
    
    # Powers of 2 mixed with small numbers
    A_powers = [2**i for i in range(14)]
    if len(A_powers) < 20:
        next_val = A_powers[-1] + 1
        while len(A_powers) < 20:
            A_powers.append(next_val)
            next_val += 1
        if all(x <= 10000 for x in A_powers):
            initial_candidates.append(sorted(A_powers))
    
    # Random subsets from smooth pool
    random.seed(42)
    for _ in range(10):
        if len(smooth_pool) >= 20:
            cand = sorted(random.sample(smooth_pool, 20))
            initial_candidates.append(cand)
    
    # Evaluate initial candidates
    best_set = None
    best_score = float('inf')
    for cand in initial_candidates:
        if len(set(cand)) != 20 or any(x < 1 or x > 10000 for x in cand):
            continue
        s = compute_metrics(cand)
        if s < best_score:
            best_score = s
            best_set = cand[:]
    
    if not best_set:
        best_set = list(range(1, 21))
        best_score = compute_metrics(best_set)
    
    # Multi-start simulated annealing
    num_restarts = 3
    time_per_restart = (deadline - start_time) / num_restarts
    
    for restart in range(num_restarts):
        if time.time() > deadline - 2.0:
            break
        
        restart_deadline = start_time + (restart + 1) * time_per_restart
        
        if restart == 0:
            current_set = best_set[:]
            current_score = best_score
        else:
            # New random start from smooth pool
            if len(smooth_pool) >= 20:
                current_set = sorted(random.sample(smooth_pool, 20))
            else:
                current_set = best_set[:]
            current_score = compute_metrics(current_set)
            if current_score < best_score:
                best_score = current_score
                best_set = current_set[:]
        
        # Simulated annealing with incremental updates
        T_start = 8.0
        T_end = 0.005
        max_steps = 50000
        
        step = 0
        while step < max_steps:
            if time.time() > restart_deadline - 1.0:
                break
            
            step += 1
            
            T = T_start * (T_end / T_start) ** (step / max_steps)
            
            # Choose move type: 70% single replace, 30% double replace
            if random.random() < 0.7:
                idx = random.randint(0, 19)
                old_val = current_set[idx]
                new_val = random.choice(full_pool)
                
                if new_val == old_val or new_val in current_set:
                    continue
                
                # Use incremental computation
                new_score = compute_metrics_incremental(current_set, old_val, new_val)
                delta = new_score - current_score
                
                if delta <= 0 or random.random() < math.exp(-delta / T):
                    current_set[idx] = new_val
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_set = current_set[:]
                else:
                    pass  # reject, no change needed
            else:
                # Double replacement
                idx1 = random.randint(0, 19)
                idx2 = random.randint(0, 19)
                if idx1 == idx2:
                    continue
                
                old_val1 = current_set[idx1]
                old_val2 = current_set[idx2]
                new_val1 = random.choice(full_pool)
                new_val2 = random.choice(full_pool)
                
                if new_val1 in current_set or new_val2 in current_set or new_val1 == new_val2:
                    continue
                if new_val1 == old_val1 and new_val2 == old_val2:
                    continue
                
                # Compute new score
                temp_set = current_set[:]
                temp_set[idx1] = new_val1
                temp_set[idx2] = new_val2
                new_score = compute_metrics(temp_set)
                delta = new_score - current_score
                
                if delta <= 0 or random.random() < math.exp(-delta / T):
                    current_set = temp_set
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_set = current_set[:]
        
        # Local search polish after each restart
        if time.time() < deadline - 3.0:
            improved = True
            while improved and time.time() < deadline - 1.5:
                improved = False
                for i in range(20):
                    old_val = best_set[i]
                    # Try replacements from a smaller focused pool
                    for new_val in full_pool[:200]:  # Smaller pool for speed
                        if new_val in best_set:
                            continue
                        new_score = compute_metrics_incremental(best_set, old_val, new_val)
                        if new_score < best_score:
                            best_set[i] = new_val
                            best_score = new_score
                            improved = True
                        else:
                            best_set[i] = old_val
                    best_set[i] = old_val
    
    return sorted(best_set)
