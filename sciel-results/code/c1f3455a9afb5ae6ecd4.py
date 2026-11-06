import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    N = 1000
    MAX_SUM = 2 * N
    
    def repair_random_set(initial_set, sum_counts):
        """
        Repair a random set by removing elements that cause sum counts > 2.
        Uses a greedy removal strategy: when a sum has > 2 representations,
        remove the element whose removal causes the least damage.
        """
        chosen = list(initial_set)
        chosen_set = set(chosen)
        
        # We need to track which pairs contribute to each sum
        # For efficiency, we'll rebuild sum info after each removal
        # but use a smarter approach: track pairs per sum
        
        # Build pair tracking: for each sum, list of (a, b) pairs
        sum_pairs = {}
        for i in range(len(chosen)):
            for j in range(i, len(chosen)):
                s = chosen[i] + chosen[j]
                if s not in sum_pairs:
                    sum_pairs[s] = []
                sum_pairs[s].append((chosen[i], chosen[j]))
        
        # Find violations and repair
        changed = True
        while changed and time.time() < deadline:
            changed = False
            
            # Find all sums with > 2 representations
            violations = []
            for s, pairs in sum_pairs.items():
                if len(pairs) > 2:
                    violations.append((s, pairs))
            
            if not violations:
                break
            
            # For each violation, decide which element to remove
            # Strategy: remove the element that appears in the most violations
            # but whose removal causes the least new violations
            element_violation_count = {}
            for s, pairs in violations:
                for a, b in pairs:
                    element_violation_count[a] = element_violation_count.get(a, 0) + 1
                    if a != b:
                        element_violation_count[b] = element_violation_count.get(b, 0) + 1
            
            # Pick the element with highest violation count to remove
            if not element_violation_count:
                break
                
            # Try removing the element with max violation count
            candidates = sorted(element_violation_count.items(), key=lambda x: -x[1])
            removed = False
            
            for elem, count in candidates:
                if elem not in chosen_set:
                    continue
                    
                # Tentatively remove elem
                new_chosen = [x for x in chosen if x != elem]
                new_chosen_set = set(new_chosen)
                
                # Check if this removal helps (reduces violations)
                # Quick check: count how many violations involve elem
                violations_with_elem = 0
                for s, pairs in violations:
                    for a, b in pairs:
                        if a == elem or b == elem:
                            violations_with_elem += 1
                            break
                
                if violations_with_elem == 0:
                    continue
                
                # Remove elem
                chosen = new_chosen
                chosen_set = new_chosen_set
                removed = True
                changed = True
                break
            
            if not removed:
                # Can't repair further, break
                break
            
            # Rebuild sum_pairs after removal
            sum_pairs = {}
            for i in range(len(chosen)):
                for j in range(i, len(chosen)):
                    s = chosen[i] + chosen[j]
                    if s not in sum_pairs:
                        sum_pairs[s] = []
                    sum_pairs[s].append((chosen[i], chosen[j]))
        
        return chosen, chosen_set
    
    def run_probabilistic(p, seed=42):
        """
        Randomly select each integer with probability p, then repair.
        """
        random.seed(seed)
        
        # Random selection
        initial_set = [x for x in range(1, N + 1) if random.random() < p]
        
        if not initial_set:
            return []
        
        # Repair
        repaired, _ = repair_random_set(initial_set, None)
        return repaired
    
    def run_greedy(noise_scale=0.0, seed=42):
        random.seed(seed)
        chosen = []
        sum_counts = [0] * (MAX_SUM + 1)
        chosen_set = set()
        
        while True:
            if time.time() > deadline:
                break
                
            best_x = None
            best_score = float('inf')
            
            for x in range(1, N + 1):
                if x in chosen_set:
                    continue
                
                valid = True
                tight_count = 0
                
                c_double = sum_counts[2 * x]
                if c_double >= 2:
                    valid = False
                elif c_double == 1:
                    tight_count += 1
                
                if valid:
                    for y in chosen:
                        c_sum = sum_counts[x + y]
                        if c_sum >= 2:
                            valid = False
                            break
                        elif c_sum == 1:
                            tight_count += 1
                
                if not valid:
                    continue
                
                score = tight_count + random.random() * noise_scale
                
                if score < best_score:
                    best_score = score
                    best_x = x
            
            if best_x is None:
                break
                
            chosen.append(best_x)
            chosen_set.add(best_x)
            
            sum_counts[2 * best_x] += 1
            for y in chosen:
                if y != best_x:
                    sum_counts[best_x + y] += 1
        
        return sorted(chosen)
    
    best_set = []
    best_size = 0
    
    # Phase 1: Probabilistic constructions with different p values
    # Theoretical density is ~3/2, so p should be around 0.15-0.25
    # But after repair, we lose some elements, so start higher
    
    p_values = [0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30, 0.35]
    
    for p in p_values:
        if time.time() > deadline:
            break
        
        # Try multiple seeds for each p
        for seed in range(10):
            if time.time() > deadline:
                break
            
            s = run_probabilistic(p, seed)
            if len(s) > best_size:
                best_size = len(s)
                best_set = s
    
    # Phase 2: Greedy constructions as fallback
    for seed in range(5):
        if time.time() > deadline:
            break
        s = run_greedy(noise_scale=0.0, seed=seed)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
    
    # Phase 3: Randomized greedy restarts
    while time.time() < deadline:
        s = run_greedy(noise_scale=1.0, seed=random.randint(0, 10000))
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
        
        # Also try probabilistic with random p
        if random.random() < 0.3:
            p = random.uniform(0.15, 0.35)
            s = run_probabilistic(p, random.randint(0, 100000))
            if len(s) > best_size:
                best_size = len(s)
                best_set = s
    
    return best_set
