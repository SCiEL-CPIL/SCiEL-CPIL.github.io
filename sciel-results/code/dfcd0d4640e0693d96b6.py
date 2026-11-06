import random
import time
import sys

def construct():
    """
    Finds a Sidon set containing 0, covering differences 1..30, 
    with at most 32 elements, minimizing the largest element.
    """
    # Deadline guard
    start_time = time.time()
    time_limit = 145.0  # Leave 5s margin for safety
    
    # 1. Initial Solution: Greedy
    # This provides a valid starting point (Sidon + Coverage)
    a = [0]
    def get_diffs(s):
        return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
        
    # Greedy construction
    for d in range(1, 31):
        if d in get_diffs(sorted(a)):
            continue
        # Try to add x such that x or x-d is in a
        found = False
        # Candidates: existing elements +/- d
        candidates = set()
        for v in a:
            candidates.add(v + d)
            candidates.add(v - d)
        
        for x in sorted(candidates):
            if x < 0 or x in a:
                continue
            trial = sorted(a + [x])
            diffs = get_diffs(trial)
            if len(set(diffs)) == len(diffs):
                a = trial
                found = True
                break
        if not found:
            # Fallback: add two new elements far away
            y = 2 * max(a) + 3 * d + 1
            while True:
                trial = sorted(a + [y, y + d])
                diffs = get_diffs(trial)
                if len(set(diffs)) == len(diffs):
                    a = trial
                    break
                y += 1
    
    best_set = a[:]
    best_val = max(a)
    
    # 2. Simulated Annealing to minimize diameter
    # We maintain a set of integers.
    # Energy: If invalid, huge penalty. If valid, penalty = max element.
    
    current_set = a[:]
    current_max = best_val
    
    def calculate_energy(s):
        """
        Returns (is_valid, diameter, diff_set, dup_count, miss_count)
        """
        if len(s) > 32:
            return False, float('inf'), set(), 0, 0
        
        diffs = [y - x for i, x in enumerate(s) for y in s[i + 1:]]
        diff_set = set(diffs)
        
        # Check Sidon property (distinct positive differences)
        # Note: differences are always positive in this generation loop
        if len(diff_set) != len(diffs):
            # Count duplicates
            dup_count = len(diffs) - len(diff_set)
            return False, float('inf'), diff_set, dup_count, 0
        
        # Check coverage 1..30
        miss_count = 0
        for d in range(1, 31):
            if d not in diff_set:
                miss_count += 1
        
        if miss_count > 0:
            return False, float('inf'), diff_set, 0, miss_count
        
        return True, max(s), diff_set, 0, 0

    # Initial energy
    is_valid, current_diam, curr_diffs, curr_dup, curr_miss = calculate_energy(current_set)
    
    T = 50.0  # Initial temperature
    T_min = 0.1
    cooling_rate = 0.995
    
    steps = 0
    max_steps = 500000
    
    while time.time() - start_time < time_limit and steps < max_steps:
        if T < T_min:
            break
            
        # Generate a neighbor
        # Strategy: Pick a random element (not 0) and perturb it
        if len(current_set) > 1:
            idx = random.randint(1, len(current_set) - 1)
            old_val = current_set[idx]
            
            # Perturbation: small change or larger jump
            if random.random() < 0.7:
                # Small jitter
                delta = random.randint(-10, 10)
                new_val = old_val + delta
            else:
                # Larger jump, biased towards smaller values to compress
                # Try to move towards 0 or random within current range
                if random.random() < 0.5:
                    new_val = random.randint(1, max(1, current_max // 2))
                else:
                    new_val = random.randint(1, current_max + 10)
            
            # Ensure non-negative and not duplicate
            if new_val < 0 or new_val == 0:
                continue
            if new_val in current_set:
                continue
                
            new_set = current_set[:]
            new_set[idx] = new_val
            new_set.sort()
            
            # Evaluate new state
            new_valid, new_diam, new_diffs, new_dup, new_miss = calculate_energy(new_set)
            
            # Calculate energy difference
            # We use a composite score for acceptance probability
            # Score = Diameter + Penalty
            curr_score = current_diam + (curr_dup * 1000 + curr_miss * 1000)
            new_score = new_diam + (new_dup * 1000 + new_miss * 1000)
            
            delta_e = new_score - curr_score
            
            # Metropolis criterion
            if delta_e < 0 or random.random() < pow(2.718281828, -delta_e / T):
                current_set = new_set
                current_max = max(current_set)
                is_valid, current_diam, curr_diffs, curr_dup, curr_miss = new_valid, new_diam, new_diffs, new_dup, new_miss
                
                # Update best if valid
                if is_valid:
                    if current_diam < best_val:
                        best_val = current_diam
                        best_set = current_set[:]
            
        else:
            # If set is too small, add an element (unlikely given greedy start)
            new_val = random.randint(1, current_max + 10)
            if new_val not in current_set:
                new_set = current_set + [new_val]
                new_set.sort()
                if len(new_set) <= 32:
                    # Accept if it improves or is valid (simplified)
                    current_set = new_set
                    current_max = max(current_set)
                    is_valid, current_diam, curr_diffs, curr_dup, curr_miss = calculate_energy(current_set)
                    if is_valid and current_diam < best_val:
                        best_val = current_diam
                        best_set = current_set[:]

        T *= cooling_rate
        steps += 1
        
        # Periodic cooling restart if stuck? 
        # Simple approach: if no improvement for 1000 steps, cool faster
        # For now, standard cooling is fine.

    return best_set

if __name__ == "__main__":
    res = construct()
    print(res)
    print("Max:", max(res))
