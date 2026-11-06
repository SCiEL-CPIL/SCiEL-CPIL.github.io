# EVOLVE-BLOCK-START
import random
import time
import itertools
import numpy as np

def _is_sidon_fast(subset):
    """Fast Sidon check with early termination using a set of sums."""
    n = len(subset)
    if n <= 1:
        return True
    seen = set()
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in seen:
                return False
            seen.add(s)
    return True

def _max_sidon_size_exact(ambient, time_limit=None, start_time=None):
    """Find largest Sidon subset exactly by trying all combinations from large to small.
    Returns the maximum k such that some k-subset is Sidon."""
    if start_time is None:
        start_time = time.time()
    n = len(ambient)
    ambient = list(ambient)
    
    # Check if full set is Sidon
    if _is_sidon_fast(ambient):
        return n
    
    # Try decreasing k
    for k in range(n - 1, 1, -1):
        if time_limit and (time.time() - start_time) > time_limit:
            break
        found = False
        for combo in itertools.combinations(ambient, k):
            if _is_sidon_fast(list(combo)):
                return k
            if time_limit and (time.time() - start_time) > time_limit:
                break
        if time_limit and (time.time() - start_time) > time_limit:
            break
    return 2

def _collision_score(ambient):
    """Compute a score based on sum collision density. Higher = more collisions = harder for Sidon."""
    sum_count = {}
    for i in range(len(ambient)):
        for j in range(i, len(ambient)):
            s = ambient[i] + ambient[j]
            sum_count[s] = sum_count.get(s, 0) + 1
    # Score: sum of (count * (count - 1)) / 2 for each sum value
    # This counts the number of colliding pairs of pairs
    score = 0
    for count in sum_count.values():
        score += count * (count - 1) // 2
    return score

def _is_3ap_free(ambient):
    """Check if ambient set is 3-AP free (no x, y, z with x+z=2y)."""
    s = set(ambient)
    for x in ambient:
        for y in ambient:
            if x == y:
                continue
            z = 2*y - x
            if z in s and z != x and z != y:
                return False
    return True

def _count_3aps(ambient):
    """Count number of 3-term APs in ambient."""
    s = set(ambient)
    count = 0
    for x in ambient:
        for y in ambient:
            if x == y:
                continue
            z = 2*y - x
            if z in s and z != x and z != y:
                count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    random.seed(42)
    
    best_set = None
    best_sidon_size = 999
    best_collision = -1
    
    evaluated_sets = set()
    
    def evaluate(candidate):
        nonlocal best_set, best_sidon_size, best_collision
        if len(candidate) != 12 or len(set(candidate)) != 12:
            return
        if any(x < 0 or x > 128 for x in candidate):
            return
        key = tuple(sorted(candidate))
        if key in evaluated_sets:
            return
        evaluated_sets.add(key)
        
        # Quick collision score check
        col_score = _collision_score(candidate)
        
        # If collision score is not promising, skip expensive Sidon check
        # But always check if it could beat current best
        if col_score <= best_collision and best_sidon_size <= 3:
            return
        
        sidon_size = _max_sidon_size_exact(candidate, time_limit=1.5, start_time=start_time)
        
        if sidon_size < best_sidon_size:
            best_sidon_size = sidon_size
            best_set = list(candidate)
            best_collision = col_score
        elif sidon_size == best_sidon_size and col_score > best_collision:
            # Same Sidon size but more collisions - might be more robust
            best_collision = col_score
    
    # Strategy 1: Dense consecutive sets (most collisions)
    for start in range(0, 120):
        if time.time() > deadline:
            break
        candidate = list(range(start, start + 12))
        if candidate[-1] <= 128:
            evaluate(candidate)
    
    # Strategy 2: Arithmetic progressions with small step
    for step in range(1, 6):
        for start in range(0, 129 - 11 * step + 1):
            if time.time() > deadline:
                break
            candidate = [start + i * step for i in range(12)]
            evaluate(candidate)
    
    # Strategy 3: Dense clusters - groups of consecutive numbers
    for gap in range(1, 15):
        for cluster_size in range(1, 7):
            num_clusters = 12 // cluster_size
            remainder = 12 % cluster_size
            if time.time() > deadline:
                break
            for base in range(0, 50):
                candidate = []
                for c in range(num_clusters + (1 if remainder else 0)):
                    cs = cluster_size if c < num_clusters else remainder
                    for i in range(cs):
                        val = base + c * gap + i
                        if val <= 128:
                            candidate.append(val)
                if len(candidate) == 12:
                    evaluate(candidate)
    
    # Strategy 4: Modular structures from inspiration 2
    # These can create dense collision patterns
    for mod in range(2, 20):
        for base in range(0, 20):
            if time.time() > deadline:
                break
            candidate = []
            for i in range(12):
                val = base + (i // 3) * mod + (i % 3)
                if val <= 128:
                    candidate.append(val)
                else:
                    candidate = None
                    break
            if candidate and len(candidate) == 12 and len(set(candidate)) == 12:
                evaluate(candidate)
    
    # Strategy 5: Simulated annealing to maximize collision density
    # Start from a dense random set
    current = random.sample(range(0, 50), 12)
    current_collision = _collision_score(current)
    current_sidon = _max_sidon_size_exact(current, time_limit=1.0, start_time=start_time)
    
    if current_sidon < best_sidon_size:
        best_sidon_size = current_sidon
        best_set = list(current)
        best_collision = current_collision
    
    T = 10.0  # Initial temperature
    cooling_rate = 0.995
    
    for iteration in range(3000):
        if time.time() > deadline:
            break
        
        idx = random.randint(0, 11)
        old_val = current[idx]
        
        # Perturb: try nearby values
        if random.random() < 0.7:
            new_val = random.randint(max(0, old_val - 5), min(128, old_val + 5))
        else:
            new_val = random.randint(0, 128)
        
        if new_val == old_val or new_val in current:
            continue
        
        current[idx] = new_val
        new_collision = _collision_score(current)
        
        # Accept based on collision improvement (with some randomness)
        delta_collision = new_collision - current_collision
        
        if delta_collision >= 0 or random.random() < np.exp(delta_collision / T):
            # Check Sidon size if collision improved
            if delta_collision > 0:
                new_sidon = _max_sidon_size_exact(current, time_limit=1.0, start_time=start_time)
                if new_sidon < best_sidon_size:
                    best_sidon_size = new_sidon
                    best_set = list(current)
                    best_collision = new_collision
            current_collision = new_collision
        else:
            current[idx] = old_val
        
        T *= cooling_rate
    
    # Strategy 6: Local search from best found
    if best_set is not None and best_sidon_size > 2:
        current = list(best_set)
        current_sidon = best_sidon_size
        for iteration in range(5000):
            if time.time() > deadline:
                break
            
            idx = random.randint(0, 11)
            old_val = current[idx]
            # Try values near current position for denser sets
            if random.random() < 0.5:
                new_val = random.randint(max(0, old_val - 10), min(128, old_val + 10))
            else:
                new_val = random.randint(0, 128)
            
            if new_val == old_val or new_val in current:
                continue
            
            current[idx] = new_val
            sidon_size = _max_sidon_size_exact(current, time_limit=1.5, start_time=start_time)
            
            if sidon_size < best_sidon_size:
                best_sidon_size = sidon_size
                best_set = list(current)
                current_sidon = sidon_size
            else:
                current[idx] = old_val
    
    # Strategy 7: 3-AP-free sets - these have minimum possible Sidon size (3)
    # Search for 12-element 3-AP-free sets in [0, 128]
    # Greedy construction
    def greedy_3ap_free(size, max_val):
        """Greedy construction of 3-AP-free set."""
        result = []
        for v in range(max_val + 1):
            # Check if adding v creates a 3-AP
            valid = True
            for i in range(len(result)):
                for j in range(len(result)):
                    if i == j:
                        continue
                    a, b = result[i], result[j]
                    # Check if (a, b, v) or (a, v, b) or (v, a, b) or (v, b, a) forms AP
                    # 2*b = a + v => v = 2*b - a
                    # 2*a = b + v => v = 2*a - b
                    # 2*v = a + b => v = (a+b)/2
                    if 2*b == a + v or 2*a == b + v:
                        valid = False
                        break
                    if (a + b) % 2 == 0 and (a + b) // 2 == v:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                result.append(v)
                if len(result) == size:
                    break
        return result
    
    # Try multiple random seeds for greedy
    for seed in range(50):
        if time.time() > deadline:
            break
        random.seed(seed)
        # Randomized greedy
        result = []
        candidates = list(range(129))
        random.shuffle(candidates)
        for v in candidates:
            if len(result) == 12:
                break
            valid = True
            for a in result:
                for b in result:
                    if a == b:
                        continue
                    if 2*b == a + v or 2*a == b + v:
                        valid = False
                        break
                    if (a + b) % 2 == 0 and (a + b) // 2 == v:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                result.append(v)
        if len(result) == 12:
            evaluate(result)
    
    # Strategy 8: Targeted search for sets with max Sidon size = 3
    # We want sets where every 4-element subset has a collision
    # This is equivalent to: no 4-element Sidon subset exists
    # Use local search to minimize max Sidon size directly
    if best_sidon_size > 3:
        # Try to find sets with Sidon size 3
        for _ in range(200):
            if time.time() > deadline:
                break
            # Start from a dense set
            base = random.randint(0, 30)
            current = [base + random.randint(0, 8) for _ in range(12)]
            # Make distinct
            current = sorted(set(current))
            while len(current) < 12:
                current.append(random.randint(0, 128))
                current = sorted(set(current))
                if len(current) > 12:
                    current = current[:12]
            
            if len(current) != 12:
                continue
            
            current_sidon = _max_sidon_size_exact(current, time_limit=1.0, start_time=start_time)
            
            if current_sidon < best_sidon_size:
                best_sidon_size = current_sidon
                best_set = list(current)
                best_collision = _collision_score(current)
            
            # Local search to reduce Sidon size
            for iteration in range(100):
                if time.time() > deadline:
                    break
                
                idx = random.randint(0, 11)
                old_val = current[idx]
                new_val = random.randint(0, 128)
                
                if new_val == old_val or new_val in current:
                    continue
                
                current[idx] = new_val
                new_sidon = _max_sidon_size_exact(current, time_limit=0.5, start_time=start_time)
                
                if new_sidon < current_sidon:
                    current_sidon = new_sidon
                    if new_sidon < best_sidon_size:
                        best_sidon_size = new_sidon
                        best_set = list(current)
                        best_collision = _collision_score(current)
                else:
                    current[idx] = old_val
    
    # Final fallback
    if best_set is None or len(set(best_set)) != 12 or any(x < 0 or x > 128 for x in best_set):
        return list(range(12))
    
    return best_set
# EVOLVE-BLOCK-END
