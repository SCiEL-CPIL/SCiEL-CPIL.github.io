import time
import random

def construct():
    start_time = time.time()
    timeout = 115.0  # Safety margin
    
    def evaluate(S):
        S_set = set(S)
        valid_diffs = set()
        S_list = sorted(S)
        n = len(S_list)
        for i in range(n):
            a = S_list[i]
            for j in range(i + 1, n):
                b = S_list[j]
                d = b - a
                c = a + 2 * d
                if c in S_set:
                    valid_diffs.add(d)
        return len(valid_diffs)
    
    # Initialize with a good heuristic or random
    # Seed: 1..32 (score 15)
    # Let's start with a random set of 32 elements from 1..512
    # Or better, start with 1..32 and improve, as it's a good local structure.
    # Actually, mixing dense and sparse is key.
    
    def random_set():
        return set(random.sample(range(1, 513), 32))
    
    # We will use Iterated Local Search
    best_set = None
    best_score = 0
    
    # Initial population of a few candidates to pick the best start
    candidates = []
    candidates.append(set(range(1, 33))) # Dense start
    candidates.append(random_set())
    candidates.append(random_set())
    
    for c in candidates:
        s = evaluate(c)
        if s > best_score:
            best_score = s
            best_set = c
    
    current_set = best_set
    current_score = best_score
    
    no_improvement_count = 0
    max_no_improvement = 1000
    
    while time.time() - start_time < timeout:
        # Local Search Phase
        improved = False
        # Try a number of random moves
        # We can limit the number of moves per iteration to allow restarts
        # Or just run until stuck
        
        # To allow perturbation, we run a fixed number of attempts or until stuck
        # Let's run until stuck or a large number of steps
        
        steps_in_iteration = 0
        max_steps_per_iter = 5000
        
        while steps_in_iteration < max_steps_per_iter:
            # Pick a random element to remove
            remove_idx = random.randint(0, 31)
            x_to_remove = list(current_set)[remove_idx]
            
            # Pick a random element to add
            y_to_add = random.randint(1, 512)
            
            if y_to_add in current_set:
                steps_in_iteration += 1
                continue
            
            new_set = set(current_set)
            new_set.remove(x_to_remove)
            new_set.add(y_to_add)
            
            new_score = evaluate(new_set)
            
            if new_score > current_score:
                current_set = new_set
                current_score = new_score
                improved = True
                no_improvement_count = 0
                # Update global best
                if current_score > best_score:
                    best_score = current_score
                    best_set = current_set
            else:
                no_improvement_count += 1
                
            steps_in_iteration += 1
            
            # Check time inside inner loop
            if time.time() - start_time > timeout - 5.0:
                break
        
        if time.time() - start_time > timeout - 5.0:
            break
            
        # If we didn't improve in this iteration, or just to explore, perturb
        # If we are stuck (no improvement in a while), do a bigger perturbation
        if not improved:
            # Perturb: 5 random swaps
            for _ in range(5):
                remove_idx = random.randint(0, 31)
                x_to_remove = list(current_set)[remove_idx]
                y_to_add = random.randint(1, 512)
                if y_to_add not in current_set:
                    current_set.remove(x_to_remove)
                    current_set.add(y_to_add)
            current_score = evaluate(current_set)
            # If perturbation led to a worse state, we might want to go back?
            # ILS usually keeps the perturbed state and continues searching from there.
            # But if we go back to best, we might get stuck in the same basin.
            # Let's keep the perturbed state.
        else:
            # We improved, so we continue from the new local optimum
            pass

    return list(best_set)
