import time
import random
import math

def calculate_score(U):
    if not U:
        return 0.0
    U = sorted(list(set(U)))
    if 0 not in U:
        U = sorted(list(set(U) | {0}))
        
    max_u = max(U)
    if max_u == 0:
        return 1.0
        
    sums = set()
    diffs = set()
    n = len(U)
    for i in range(n):
        ui = U[i]
        for j in range(n):
            uj = U[j]
            sums.add(ui + uj)
            diffs.add(ui - uj)
            
    num = len(diffs)
    den = len(sums)
    if den == 0:
        return 0.0
    ratio = num / den
    if ratio <= 0:
        return 0.0
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * max_u + 1)
    if log_denom == 0:
        return 1.0
    return 1 + log_ratio / log_denom

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_score = 1.0
    best_set = [0, 1, 3]
    
    # Initial candidates: Golomb rulers and variations
    initial_sets = [
        [0, 1, 3],
        [0, 1, 4, 6],
        [0, 1, 4, 9, 11],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 18, 23, 28],
        [0, 1, 4, 10, 18, 24, 28, 34],
        [0, 1, 4, 10, 18, 24, 28, 34, 42]
    ]
    
    for s in initial_sets:
        s = sorted(set(s))
        score = calculate_score(s)
        if score > best_score:
            best_score = score
            best_set = s
            
    # Local Search
    random.seed(42)
    
    for restart in range(100):
        if time.time() > deadline:
            break
            
        # Start from best or random
        if restart % 3 == 0:
            current_set = list(best_set)
        else:
            # Random Golomb-like set
            n = random.randint(3, 8)
            current_set = [0]
            last = 0
            for i in range(1, n):
                # Add a gap
                gap = random.randint(1, 10)
                last += gap
                current_set.append(last)
            current_set = sorted(current_set)
            
        current_score = calculate_score(current_set)
        if current_score > best_score:
            best_score = current_score
            best_set = list(current_set)
            
        no_improvement = 0
        while time.time() < deadline and no_improvement < 100:
            neighbor = list(current_set)
            move = random.randint(0, 2)
            
            if move == 0 and len(neighbor) > 1:
                # Perturb
                idx = random.randint(1, len(neighbor)-1)
                max_val = max(neighbor)
                step = random.choice([-3, -2, -1, 1, 2, 3, -5, 5])
                new_val = neighbor[idx] + step
                if new_val >= 0 and new_val not in neighbor:
                    neighbor[idx] = new_val
                else:
                    no_improvement += 1
                    continue
            elif move == 1 and len(neighbor) < 9:
                # Add
                new_val = random.randint(1, max(neighbor) + 10)
                if new_val not in neighbor:
                    neighbor.append(new_val)
                else:
                    no_improvement += 1
                    continue
            elif move == 2 and len(neighbor) > 2:
                # Remove
                idx = random.randint(1, len(neighbor)-1)
                neighbor.pop(idx)
            else:
                no_improvement += 1
                continue
                
            neighbor = sorted(set(neighbor))
            if 0 not in neighbor:
                neighbor = sorted(set(neighbor) | {0})
                
            new_score = calculate_score(neighbor)
            
            if new_score > current_score:
                current_set = neighbor
                current_score = new_score
                no_improvement = 0
                if current_score > best_score:
                    best_score = current_score
                    best_set = list(current_set)
            else:
                no_improvement += 1
                
    # Final Polish: Try small steps on best_set
    if time.time() < deadline:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(1, len(best_set)):
                for delta in [-2, -1, 1, 2]:
                    candidate = list(best_set)
                    new_val = candidate[i] + delta
                    if new_val >= 0 and new_val not in candidate:
                        candidate[i] = new_val
                        candidate = sorted(set(candidate))
                        if 0 not in candidate:
                            candidate = sorted(set(candidate) | {0})
                        s = calculate_score(candidate)
                        if s > best_score:
                            best_score = s
                            best_set = candidate
                            improved = True
                            # Restart polish loop with new best
                            break
                if improved:
                    break
            # Also try adding/removing in polish
            if not improved:
                # Try add
                for delta in [1, 2, 3]:
                    new_val = max(best_set) + delta
                    candidate = sorted(set(best_set) | {new_val})
                    s = calculate_score(candidate)
                    if s > best_score:
                        best_score = s
                        best_set = candidate
                        improved = True
                # Try remove
                if len(best_set) > 2:
                    for i in range(1, len(best_set)):
                        candidate = list(best_set)
                        candidate.pop(i)
                        candidate = sorted(set(candidate))
                        if 0 not in candidate:
                            candidate = sorted(set(candidate) | {0})
                        s = calculate_score(candidate)
                        if s > best_score:
                            best_score = s
                            best_set = candidate
                            improved = True

    return best_set
