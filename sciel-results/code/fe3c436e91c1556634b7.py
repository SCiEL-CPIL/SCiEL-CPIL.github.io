import time
import random

L = 5354228880  # lcm(1..24)

def subset_sum_exists(weights, target):
    """Check if target can be formed by a nonempty subset of weights."""
    if target <= 0:
        return False
    weights = [w for w in weights if 0 < w <= target]
    if not weights:
        return False
    if target in weights:
        return True
    
    n = len(weights)
    mid = n // 2
    left = weights[:mid]
    right = weights[mid:]
    
    left_sums = {0}
    for w in left:
        left_sums |= {s + w for s in left_sums if s + w <= target}
    
    right_sums = {0}
    for w in right:
        right_sums |= {s + w for s in right_sums if s + w <= target}
    
    for ls in left_sums:
        if target - ls in right_sums:
            return True
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 10s safety margin
    
    weights = [0] * 25
    for i in range(1, 25):
        weights[i] = L // i
    
    # Try multiple greedy orders
    orders = [
        list(range(24, 0, -1)),  # descending
        list(range(1, 25)),      # ascending
        list(range(1, 25, 2)) + list(range(2, 25, 2)),  # odd then even
        list(range(24, 0, -2)) + list(range(23, 0, -2)),  # even then odd
    ]
    
    best_S = set()
    
    for order in orders:
        if time.time() > deadline - 15:
            break
        
        S = set()
        for x in order:
            if time.time() > deadline - 15:
                break
            target = weights[x]
            current_weights = [weights[z] for z in S if weights[z] <= target]
            if not subset_sum_exists(current_weights, target):
                S.add(x)
        
        if len(S) > len(best_S):
            best_S = set(S)
    
    # Local search on best_S
    S = set(best_S)
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for x in range(1, 25):
            if x in S:
                continue
            if time.time() > deadline - 5:
                break
            # Try to add x
            target_x = weights[x]
            conflict_with_S = subset_sum_exists([weights[z] for z in S], target_x)
            
            if not conflict_with_S:
                # Check if adding x breaks any existing element
                valid = True
                for z in S:
                    t = weights[z]
                    others = [weights[yy] for yy in S if yy != z] + [weights[x]]
                    if subset_sum_exists(others, t):
                        valid = False
                        break
                if valid:
                    S.add(x)
                    improved = True
                    break  # Restart the loop with new S
                else:
                    # Try removing one element to fix
                    for z in sorted(S, reverse=True):
                        new_S = S - {z}
                        if not subset_sum_exists([weights[zz] for zz in new_S], target_x):
                            valid2 = True
                            for zz in new_S:
                                t = weights[zz]
                                others = [weights[yy] for yy in new_S if yy != zz] + [weights[x]]
                                if subset_sum_exists(others, t):
                                    valid2 = False
                                    break
                            if valid2:
                                S = new_S
                                S.add(x)
                                improved = True
                                break
            else:
                # Conflict: try removing one element
                for z in sorted(S, reverse=True):
                    new_S = S - {z}
                    if not subset_sum_exists([weights[zz] for zz in new_S], target_x):
                        valid2 = True
                        for zz in new_S:
                            t = weights[zz]
                            others = [weights[yy] for yy in new_S if yy != zz] + [weights[x]]
                            if subset_sum_exists(others, t):
                                valid2 = False
                                break
                        if valid2:
                            S = new_S
                            S.add(x)
                            improved = True
                            break
    
    if len(S) > len(best_S):
        best_S = S
    
    # If time remains, try random restarts
    if time.time() < deadline - 10:
        for _ in range(5):
            if time.time() > deadline - 5:
                break
            random.seed(int(time.time() * 1000))
            order = list(range(1, 25))
            random.shuffle(order)
            
            S = set()
            for x in order:
                if time.time() > deadline - 5:
                    break
                target = weights[x]
                current_weights = [weights[z] for z in S if weights[z] <= target]
                if not subset_sum_exists(current_weights, target):
                    S.add(x)
            
            if len(S) > len(best_S):
                best_S = set(S)
    
    return list(best_S)
