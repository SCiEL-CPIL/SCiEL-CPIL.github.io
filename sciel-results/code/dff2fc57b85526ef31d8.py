import time
import random
import sys
import math

sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # Safety margin before 90s limit
    
    # Fast subset sum computation
    def compute_sums(elements):
        """Compute all non-empty subset sums. Returns set or None if duplicates."""
        sums = set()
        for x in elements:
            new_sums = set()
            for s in sums:
                ns = s + x
                if ns in sums:
                    return None
                new_sums.add(ns)
            if x in sums:
                return None
            new_sums.add(x)
            sums = sums.union(new_sums)
        return sums
    
    # Check divisibility constraint on a set of sums
    def check_divisibility(sums):
        """Check no two distinct sums divide each other. Returns True if valid."""
        s_list = sorted(sums)
        n = len(s_list)
        for i in range(n):
            si = s_list[i]
            for j in range(i+1, n):
                sj = s_list[j]
                if sj % si == 0:
                    return False
        return True
    
    # Full validation
    def is_valid(elements):
        if not elements:
            return True
        if len(elements) != len(set(elements)):
            return False
        if any(x < 1 or x > 256 for x in elements):
            return False
        sums = compute_sums(elements)
        if sums is None:
            return False
        return check_divisibility(sums)
    
    # Score function: negative count of divisibility violations (lower is better)
    def score(elements):
        """Returns (is_valid, penalty). Lower penalty is better."""
        if not elements:
            return True, 0
        sums = compute_sums(elements)
        if sums is None:
            return False, 1000  # Duplicate sums
        s_list = sorted(sums)
        n = len(s_list)
        violations = 0
        for i in range(n):
            si = s_list[i]
            for j in range(i+1, n):
                sj = s_list[j]
                if sj % si == 0:
                    violations += 1
        return violations == 0, violations
    
    # Simulated annealing
    def simulated_annealing(initial_elements, time_limit):
        if time.time() > time_limit:
            return None
        
        current = list(initial_elements)
        current_valid, current_penalty = score(current)
        
        best = list(current)
        best_valid, best_penalty = current_valid, current_penalty
        
        all_numbers = list(range(1, 257))
        
        # Temperature schedule
        T_start = 10.0
        T_end = 0.01
        total_steps = 50000
        
        rng = random.Random()
        
        for step in range(total_steps):
            if time.time() > time_limit:
                break
            
            # Temperature
            progress = step / total_steps
            T = T_start * (T_end / T_start) ** progress
            
            # Choose a move
            move_type = rng.random()
            
            new_elements = list(current)
            
            if move_type < 0.4:
                # Add an element
                available = [x for x in all_numbers if x not in current]
                if available:
                    x = rng.choice(available)
                    new_elements.append(x)
            elif move_type < 0.7:
                # Remove an element
                if current:
                    idx = rng.randrange(len(current))
                    new_elements.pop(idx)
            else:
                # Swap an element
                if current:
                    idx = rng.randrange(len(current))
                    old_x = current[idx]
                    available = [x for x in all_numbers if x not in current]
                    if available:
                        new_x = rng.choice(available)
                        new_elements[idx] = new_x
            
            new_valid, new_penalty = score(new_elements)
            
            # Metropolis acceptance
            if new_penalty < current_penalty:
                # Accept improvement
                current = new_elements
                current_valid, current_penalty = new_valid, new_penalty
                if new_penalty < best_penalty:
                    best = list(current)
                    best_valid, best_penalty = new_valid, new_penalty
                    if best_valid and len(best) > 0:
                        # Found valid set, keep improving
                        pass
            else:
                # Accept worsening with probability exp(-delta/T)
                delta = new_penalty - current_penalty
                if T > 0:
                    prob = math.exp(-delta / T)
                    if rng.random() < prob:
                        current = new_elements
                        current_valid, current_penalty = new_valid, new_penalty
        
        return best if best_valid else None
    
    # Build initial solution using greedy
    def greedy_initial(target_size):
        current = []
        sums = set()
        
        # Start with larger numbers (fewer divisibility issues)
        candidates = list(range(256, 1, -1))
        rng = random.Random()
        rng.shuffle(candidates)
        
        for x in candidates:
            if time.time() > deadline:
                break
            if len(current) >= target_size:
                break
            
            # Try adding x
            new_sums = set()
            for s in sums:
                ns = s + x
                if ns in sums:
                    break
                new_sums.add(ns)
            else:
                if x in sums:
                    break
                new_sums.add(x)
                all_sums = sums.union(new_sums)
                
                # Quick divisibility check
                valid = True
                s_list = sorted(all_sums)
                for i in range(len(s_list)):
                    si = s_list[i]
                    for j in range(i+1, len(s_list)):
                        sj = s_list[j]
                        if sj % si == 0:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    current.append(x)
                    sums = all_sums
        
        return current
    
    # Try to find a valid set of size k using SA
    def find_valid_set(k, time_limit):
        # Generate several initial solutions and run SA on each
        best_result = None
        
        num_initials = min(5, max(1, int((time_limit - time.time()) / 2)))
        
        for i in range(num_initials):
            if time.time() > time_limit:
                break
            
            # Generate initial solution
            initial = greedy_initial(k + 2)  # Try to start larger
            if len(initial) < 2:
                # Fallback: random small set
                initial = random.sample(range(100, 257), min(3, k))
            
            # Run SA
            result = simulated_annealing(initial, time_limit)
            
            if result is not None and len(result) >= k:
                if best_result is None or len(result) > len(best_result):
                    best_result = result
            
            if best_result is not None and len(best_result) >= k:
                break
        
        return best_result
    
    # Main search: try to find largest valid set
    best_set = []
    
    # First, find a valid set using SA starting from various sizes
    for k in [8, 7, 6, 5, 4, 3, 2]:
        if time.time() > deadline:
            break
        
        remaining_time = deadline - time.time()
        if remaining_time < 2:
            break
        
        time_alloc = min(remaining_time * 0.7, 15.0)
        current_deadline = time.time() + time_alloc
        
        result = find_valid_set(k, current_deadline)
        
        if result is not None and len(result) >= k:
            if is_valid(result):
                best_set = result
                break
    
    # If we found a set, try to improve it with more SA iterations
    if best_set:
        remaining_time = deadline - time.time()
        if remaining_time > 5:
            improved = simulated_annealing(best_set, time.time() + min(remaining_time * 0.8, 20.0))
            if improved is not None and len(improved) > len(best_set):
                if is_valid(improved):
                    best_set = improved
    
    # Fallback valid sets
    if not best_set:
        fallbacks = [
            [100, 101, 103],
            [100, 101],
            [100, 102],
            [100, 103],
            [100],
            [200, 201],
            [250, 251],
            [255, 256],
        ]
        for fs in fallbacks:
            if is_valid(fs):
                best_set = fs
                break
    
    if not best_set:
        best_set = [100]
    
    # Ensure valid Python ints in range
    best_set = [int(x) for x in best_set if 1 <= x <= 256]
    seen = set()
    unique = []
    for x in best_set:
        if x not in seen:
            seen.add(x)
            unique.append(x)
    best_set = unique
    
    return best_set
