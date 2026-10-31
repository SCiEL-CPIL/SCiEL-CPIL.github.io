import time
import random
import math
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 130.0  # Leave 20s safety margin for 150s limit
    
    def compute_score(U):
        if not U or len(U) <= 1:
            return 1.0
        max_u = max(U)
        if max_u == 0:
            return 1.0
        
        u_arr = np.array(U, dtype=np.int64)
        n = len(u_arr)
        
        # Vectorized computation of sumset and diffset
        # All pairwise sums (i <= j to avoid duplicates, but we'll use set anyway)
        sums = np.add.outer(u_arr, u_arr)
        sumset = set(np.unique(sums).tolist())
        
        # All pairwise differences
        diffs = np.subtract.outer(u_arr, u_arr)
        diffset = set(np.unique(diffs).tolist())
        
        size_sum = len(sumset)
        size_diff = len(diffset)
        
        if size_sum == 0:
            return 1.0
            
        ratio = size_diff / size_sum
        if ratio <= 1.0:
            return 1.0
        
        denom = math.log(2 * max_u + 1)
        if denom == 0:
            return 1.0
            
        return 1.0 + math.log(ratio) / denom

    # Known optimal Golomb rulers (these have all distinct positive differences)
    # These maximize |U-U| for given max(U), which helps the ratio
    golomb_rulers = {
        1: [0],
        2: [0, 1],
        3: [0, 1, 3],
        4: [0, 1, 4, 6],
        5: [0, 1, 4, 9, 11],
        6: [0, 1, 4, 10, 12, 17],
        7: [0, 1, 4, 10, 12, 17, 25],
        8: [0, 1, 4, 10, 12, 17, 25, 29],
        9: [0, 1, 4, 10, 12, 17, 25, 29, 34],
        10: [0, 1, 4, 10, 12, 17, 25, 29, 34, 41],
    }
    
    # Additional candidates: subsets of Golomb rulers, and some hand-crafted sets
    initial_sets = []
    for n in range(1, 11):
        if n in golomb_rulers:
            initial_sets.append(golomb_rulers[n])
    
    # Add some variations
    extra_sets = [
        [0, 1, 3, 7],
        [0, 1, 4, 6, 10],
        [0, 2, 5, 11],
        [0, 1, 5, 11],
        [0, 3, 7, 12],
        [0, 1, 4, 9, 11, 18],
        [0, 2, 7, 12, 18],
        [0, 1, 5, 10, 16],
    ]
    initial_sets.extend(extra_sets)
    
    best_score = 0.0
    best_set = [0]
    
    for s in initial_sets:
        score = compute_score(s)
        if score > best_score:
            best_score = score
            best_set = list(s)
    
    # Local search with multiple restarts
    max_restarts = 8
    best_overall_score = best_score
    best_overall_set = best_set
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Start from best set or a random Golomb ruler
        if restart == 0:
            current_set = list(best_overall_set)
        else:
            # Pick a random Golomb ruler as starting point
            n = random.randint(3, 10)
            if n in golomb_rulers:
                current_set = list(golomb_rulers[n])
            else:
                current_set = list(best_overall_set)
        
        current_score = compute_score(current_set)
        
        # Local search phase
        no_improve_count = 0
        max_no_improve = 800
        
        while time.time() < deadline and no_improve_count < max_no_improve:
            # Generate a neighbor
            neighbor = list(current_set)
            op = random.randint(0, 5)
            
            if op == 0 and len(neighbor) > 1:
                # Remove a random element (not 0)
                idx = random.randint(1, len(neighbor) - 1)
                neighbor.pop(idx)
                if len(neighbor) == 0:
                    continue
                if 0 not in neighbor:
                    neighbor.append(0)
                    neighbor.sort()
            elif op == 1:
                # Add a random element
                max_val = max(neighbor) if neighbor else 0
                upper_bound = max(2 * max_val + 20, 30)
                new_val = random.randint(1, upper_bound)  # Avoid 0
                if new_val not in neighbor:
                    neighbor.append(new_val)
                    neighbor.sort()
            elif op == 2:
                # Modify a random element (not 0) to a nearby value
                if len(neighbor) > 1:
                    idx = random.randint(1, len(neighbor) - 1)
                    old_val = neighbor[idx]
                    # Try small perturbations
                    deltas = [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]
                    random.shuffle(deltas)
                    for delta in deltas:
                        new_val = old_val + delta
                        if new_val >= 0 and new_val not in neighbor:
                            neighbor[idx] = new_val
                            neighbor.sort()
                            break
                    else:
                        continue
            elif op == 3:
                # Swap: remove one element and add another
                if len(neighbor) > 1:
                    idx = random.randint(1, len(neighbor) - 1)
                    removed_val = neighbor.pop(idx)
                    max_val = max(neighbor) if neighbor else 0
                    upper_bound = max(2 * max_val + 20, 30)
                    new_val = random.randint(1, upper_bound)
                    if new_val not in neighbor:
                        neighbor.append(new_val)
                        neighbor.sort()
                    else:
                        # Revert
                        neighbor.append(removed_val)
                        neighbor.sort()
                        continue
            elif op == 4:
                # Add two elements
                max_val = max(neighbor) if neighbor else 0
                upper_bound = max(2 * max_val + 20, 30)
                new_val1 = random.randint(1, upper_bound)
                new_val2 = random.randint(1, upper_bound)
                if new_val1 not in neighbor and new_val2 not in neighbor and new_val1 != new_val2:
                    neighbor.append(new_val1)
                    neighbor.append(new_val2)
                    neighbor.sort()
            else:
                # Remove two elements
                if len(neighbor) > 3:
                    idx1 = random.randint(1, len(neighbor) - 1)
                    idx2 = random.randint(1, len(neighbor) - 1)
                    if idx1 == idx2:
                        continue
                    # Sort indices to handle correctly
                    if idx1 > idx2:
                        idx1, idx2 = idx2, idx1
                    neighbor.pop(idx1)
                    neighbor.pop(idx2)
                    if len(neighbor) == 0:
                        continue
                    if 0 not in neighbor:
                        neighbor.append(0)
                        neighbor.sort()
            
            if not neighbor or 0 not in neighbor:
                continue
                
            new_score = compute_score(neighbor)
            
            if new_score > current_score:
                current_set = neighbor
                current_score = new_score
                no_improve_count = 0
                if current_score > best_overall_score:
                    best_overall_score = current_score
                    best_overall_set = list(current_set)
            else:
                no_improve_count += 1
                # Small probability of accepting worse move to escape local minima
                if random.random() < 0.003:
                    current_set = neighbor
                    current_score = new_score
    
    # Final polish: systematically try small perturbations on the best set
    if time.time() < deadline - 5:
        polish_set = list(best_overall_set)
        polish_score = compute_score(polish_set)
        
        # Try removing each element one at a time
        for i in range(1, len(polish_set)):
            if time.time() > deadline - 2:
                break
            candidate = list(polish_set)
            candidate.pop(i)
            if len(candidate) > 0 and 0 in candidate:
                candidate.sort()
                s = compute_score(candidate)
                if s > polish_score:
                    polish_set = candidate
                    polish_score = s
        
        # Try adding elements near existing ones
        if time.time() < deadline - 2:
            max_val = max(polish_set)
            for base in list(polish_set):
                for delta in range(-3, 4):
                    if time.time() > deadline - 2:
                        break
                    new_val = base + delta
                    if new_val >= 0 and new_val not in polish_set:
                        candidate = list(polish_set)
                        candidate.append(new_val)
                        candidate.sort()
                        s = compute_score(candidate)
                        if s > polish_score:
                            polish_set = candidate
                            polish_score = s
        
        if polish_score > best_overall_score:
            best_overall_score = polish_score
            best_overall_set = polish_set
    
    return best_overall_set
# EVOLVE-BLOCK-END
