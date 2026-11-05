# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    n = 4096
    target_size = 80
    
    def try_greedy(candidates, max_size=80, deadline=deadline):
        """Try to build a B_3[1] set by adding elements in the given order."""
        selected = []
        triple_sums = set()
        
        for x in candidates:
            if time.time() > deadline:
                break
            if len(selected) >= max_size:
                break
            
            # Check if adding x creates any collision
            # New triples involving x (x is the largest element)
            # For each pair (i,j) from selected (including x itself), check sum
            valid = True
            new_sums = set()
            
            # (x, x, x)
            s = 3 * x
            if s in triple_sums:
                valid = False
            else:
                new_sums.add(s)
            
            # (x, x, y) and (x, y, y) and (x, y, z) for y,z in selected
            s_list = list(selected)
            for i in range(len(s_list)):
                for j in range(i, len(s_list)):
                    total = x + s_list[i] + s_list[j]
                    if total in triple_sums or total in new_sums:
                        valid = False
                        break
                    new_sums.add(total)
                if not valid:
                    break
            
            if valid:
                selected.append(x)
                triple_sums.update(new_sums)
        
        return selected
    
    # First try simple greedy in increasing order
    result = try_greedy(range(1, n+1))
    
    if len(result) >= target_size:
        return result[:target_size]
    
    # Try random restarts with different orderings
    best = result
    attempt = 0
    
    while time.time() < deadline - 2 and len(best) < target_size:
        attempt += 1
        random.seed(attempt * 7919)
        
        # Mix of strategies: sometimes try from larger values, sometimes random
        if attempt % 3 == 0:
            # Try descending
            candidates = list(range(n, 0, -1))
        elif attempt % 3 == 1:
            # Random shuffle
            candidates = list(range(1, n+1))
            random.shuffle(candidates)
        else:
            # Weighted toward middle
            candidates = list(range(1, n+1))
            random.shuffle(candidates)
            # Move some smaller values to front
            for _ in range(20):
                idx = random.randint(0, len(candidates)-1)
                candidates.insert(0, candidates.pop(idx))
        
        result = try_greedy(candidates, target_size, deadline)
        
        if len(result) > len(best):
            best = result
            if len(best) >= target_size:
                break
    
    return best[:target_size] if len(best) >= target_size else best

# EVOLVE-BLOCK-END
