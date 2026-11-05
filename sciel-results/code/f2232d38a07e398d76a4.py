import random
import time
import math

def _score_from_list(lst):
    """
    Calculate the number of distinct d > 0 such that a 3-AP with difference d exists in lst.
    Uses bitset for efficiency.
    """
    if not lst:
        return 0
    
    # Create bitmask
    # lst contains distinct integers in 1..512
    mask = 0
    for x in lst:
        mask |= (1 << x)
        
    valid_ds = 0
    # d can range from 1 to 255 (since max element is 512, min is 1, max span 511, 2d <= 511 => d <= 255)
    for d in range(1, 256):
        # Check if there is any a such that a, a+d, a+2d are in the set
        # mask has bit i set if i is in S
        # mask >> d has bit i set if i+d is in S
        # mask >> 2d has bit i set if i+2d is in S
        # Intersection non-zero means there is such an i (which is a)
        if (mask & (mask >> d) & (mask >> 2 * d)) != 0:
            valid_ds += 1
            
    return valid_ds

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    best_score = 0
    best_list = []
    
    def init_random():
        return random.sample(range(1, 513), 32)
        
    def init_spread():
        # Try to initialize with some spread to encourage large differences
        points = []
        # Use a jittered grid
        step = 512 / 32.0
        for i in range(32):
            p = int(i * step) + random.randint(0, int(step))
            if p < 1: p = 1
            if p > 512: p = 512
            points.append(p)
        # Ensure distinctness
        while len(set(points)) < 32:
            idx = random.randint(0, 31)
            points[idx] = random.randint(1, 512)
        return points

    def init_clustered():
        # Start with a cluster and expand
        base = random.randint(1, 400)
        points = []
        for i in range(32):
            p = base + random.randint(-50, 50)
            if p < 1: p = 1
            if p > 512: p = 512
            points.append(p)
        while len(set(points)) < 32:
            idx = random.randint(0, 31)
            points[idx] = random.randint(1, 512)
        return points

    # Pre-generate some initial candidates
    initial_sets = []
    for _ in range(5):
        initial_sets.append(init_random())
    for _ in range(5):
        initial_sets.append(init_spread())
    for _ in range(5):
        initial_sets.append(init_clustered())

    for init in initial_sets:
        if time.time() > deadline:
            break
            
        current_list = init.copy()
        current_score = _score_from_list(current_list)
        
        if current_score > best_score:
            best_score = current_score
            best_list = current_list.copy()
            
        # Simulated Annealing
        temperature = 5.0
        min_temperature = 0.01
        cooling_rate = 0.9995
        
        # Use a set for O(1) membership check in current_list
        current_set = set(current_list)
        
        while temperature > min_temperature and time.time() < deadline:
            # Choose a move
            move_type = random.random()
            
            if move_type < 0.6:
                # Replace one element
                idx = random.randint(0, 31)
                old_val = current_list[idx]
                
                # Generate new value
                new_val = random.randint(1, 512)
                if new_val == old_val:
                    continue
                if new_val in current_set:
                    continue
                    
                # Apply move
                current_list[idx] = new_val
                current_set.remove(old_val)
                current_set.add(new_val)
                
                new_score = _score_from_list(current_list)
                delta = new_score - current_score
                
                if delta > 0 or random.random() < math.exp(delta / temperature):
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_list = current_list.copy()
                else:
                    # Revert
                    current_list[idx] = old_val
                    current_set.remove(new_val)
                    current_set.add(old_val)
                    
            elif move_type < 0.9:
                # Swap two elements
                idx1 = random.randint(0, 31)
                idx2 = random.randint(0, 31)
                if idx1 == idx2:
                    continue
                
                v1 = current_list[idx1]
                v2 = current_list[idx2]
                
                # Swap
                current_list[idx1] = v2
                current_list[idx2] = v1
                
                new_score = _score_from_list(current_list)
                delta = new_score - current_score
                
                if delta > 0 or random.random() < math.exp(delta / temperature):
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_list = current_list.copy()
                else:
                    # Revert swap
                    current_list[idx1] = v1
                    current_list[idx2] = v2
                    
            else:
                # Shift one element by a small amount
                idx = random.randint(0, 31)
                old_val = current_list[idx]
                shift = random.randint(-10, 10)
                if shift == 0:
                    continue
                new_val = old_val + shift
                if new_val < 1 or new_val > 512:
                    continue
                if new_val in current_set:
                    continue
                    
                current_list[idx] = new_val
                current_set.remove(old_val)
                current_set.add(new_val)
                
                new_score = _score_from_list(current_list)
                delta = new_score - current_score
                
                if delta > 0 or random.random() < math.exp(delta / temperature):
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_list = current_list.copy()
                else:
                    # Revert
                    current_list[idx] = old_val
                    current_set.remove(new_val)
                    current_set.add(old_val)
                    
            temperature *= cooling_rate
            
    if not best_list:
        return list(range(1, 33))
        
    return best_list

# Test/Debug block (not part of construct)
if __name__ == "__main__":
    # Quick test
    res = construct()
    print(f"Result: {res}")
    print(f"Score: {_score_from_list(res)}")
