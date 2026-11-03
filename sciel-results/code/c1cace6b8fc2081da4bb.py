import time
import random
import math
import numpy as np

def _score(U):
    """Compute the score for a given set U."""
    if not U:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 1.0
    u_arr = np.array(U, dtype=np.int64)
    n = len(u_arr)
    
    sums = u_arr[:, None] + u_arr[None, :]
    diffs = u_arr[:, None] - u_arr[None, :]
    
    sumset_size = len(np.unique(sums))
    diffset_size = len(np.unique(diffs))
    
    ratio = diffset_size / sumset_size
    if ratio <= 1.0:
        return 1.0
    denom = math.log(2 * max_u + 1)
    if denom == 0:
        return 1.0
    return 1.0 + math.log(ratio) / denom

def _score_fast(U_list):
    """Fast score using numpy - U_list is a list of ints."""
    if not U_list:
        return 0.0
    max_u = max(U_list)
    if max_u == 0:
        return 1.0
    u_arr = np.array(U_list, dtype=np.int64)
    n = len(u_arr)
    
    sums = u_arr[:, None] + u_arr[None, :]
    diffs = u_arr[:, None] - u_arr[None, :]
    
    sumset_size = len(np.unique(sums))
    diffset_size = len(np.unique(diffs))
    
    ratio = diffset_size / sumset_size
    if ratio <= 1.0:
        return 1.0
    denom = math.log(2 * max_u + 1)
    if denom == 0:
        return 1.0
    return 1.0 + math.log(ratio) / denom

def _generate_algebraic_starts():
    """Generate diverse algebraic starting sets."""
    starts = []
    
    # Geometric-like sequences
    for base in [2, 3, 4, 5]:
        for n in range(4, 10):
            s = [0]
            val = 1
            for i in range(n - 1):
                s.append(val)
                val *= base
            starts.append(sorted(s))
    
    # Quadratic sequences
    for a in [1, 2, 3]:
        for n in range(5, 12):
            s = [0]
            for i in range(1, n):
                s.append(a * i * i)
            starts.append(sorted(s))
    
    # Mixed: geometric with perturbations
    for base in [2, 3]:
        for n in range(5, 10):
            s = [0]
            val = 1
            for i in range(n - 1):
                s.append(val + random.randint(-1, 2))
                val *= base
            s = sorted(set(s))
            if 0 in s:
                starts.append(s)
    
    # Arithmetic with gaps
    for step in [2, 3, 5]:
        for n in range(5, 15):
            s = [0]
            for i in range(1, n):
                s.append(step * i)
            starts.append(sorted(s))
    
    # Sparse geometric: powers of 2 with some skipped
    for n in range(5, 12):
        s = [0]
        powers = [2**i for i in range(n + 2)]
        random.shuffle(powers)
        s = sorted(powers[:n])
        if 0 not in s:
            s = [0] + s
        starts.append(s)
    
    return starts

def _simulated_annealing(start_set, deadline, initial_temp=100.0, cooling=0.95):
    """Simulated annealing to optimize the set."""
    current = list(start_set)
    current_score = _score_fast(current)
    best = list(current)
    best_score = current_score
    
    temp = initial_temp
    rng = random.Random(42)
    
    while time.time() < deadline:
        # Choose a random move
        move_type = rng.choice(['add', 'remove', 'move'])
        
        if move_type == 'add':
            if len(current) > 25:
                continue
            max_c = max(current) if current else 0
            # Try adding near existing elements or at gaps
            if current and rng.random() < 0.7:
                base = rng.choice(current)
                x = base + rng.randint(1, max(5, max_c // 10))
            else:
                x = rng.randint(1, 2 * max_c + 20)
            
            if x in current or x < 0:
                continue
            new_set = current + [x]
            new_score = _score_fast(new_set)
            
        elif move_type == 'remove':
            if len(current) <= 3:
                continue
            x = rng.choice([y for y in current if y != 0])
            if x == 0:
                continue
            new_set = [y for y in current if y != x]
            new_score = _score_fast(new_set)
            
        else:  # move
            if len(current) <= 3:
                continue
            idx = rng.randint(1, len(current) - 1)  # don't move 0
            x = current[idx]
            max_c = max(current) if current else 0
            # Move to nearby or random position
            if rng.random() < 0.5:
                y = x + rng.randint(-max(3, x // 10), max(3, x // 10))
            else:
                y = rng.randint(1, 2 * max_c + 10)
            
            if y < 0 or y in current or y == x:
                continue
            new_set = list(current)
            new_set[idx] = y
            new_score = _score_fast(new_set)
        
        # Acceptance criterion
        delta = new_score - current_score
        if delta > 0 or (temp > 0.01 and rng.random() < math.exp(delta / temp)):
            current = new_set
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best = list(current)
        
        # Cool down
        temp *= cooling
        if temp < 0.01:
            temp = 0.01
    
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 105  # Safety margin before 120s limit
    
    best_set = [0, 1, 3]
    best_score = _score_fast(best_set)
    
    # Generate diverse starting points
    starts = _generate_algebraic_starts()
    
    # Add some random starts
    for _ in range(10):
        size = random.randint(4, 12)
        s = {0}
        max_val = 100
        while len(s) < size:
            s.add(random.randint(1, max_val))
        starts.append(sorted(s))
    
    # Shuffle starts for diversity
    random.shuffle(starts)
    
    # Run simulated annealing on each start with time allocation
    num_starts = len(starts)
    time_per_start = (deadline - time.time()) / max(1, num_starts)
    
    for i, start in enumerate(starts):
        remaining = deadline - time.time()
        if remaining < 2:
            break
        
        # Allocate time based on remaining budget
        alloc_time = min(time_per_start, remaining - 1)
        if alloc_time < 1:
            break
        
        sa_deadline = time.time() + alloc_time
        sa_set, sa_score = _simulated_annealing(start, sa_deadline)
        
        if sa_score > best_score:
            best_score = sa_score
            best_set = sa_set
    
    # Final local polish on the best set
    if time.time() < deadline - 5:
        polish_deadline = deadline - 2
        polished, polish_score = _simulated_annealing(best_set, polish_deadline, initial_temp=10.0, cooling=0.99)
        if polish_score > best_score:
            best_score = polish_score
            best_set = polished
    
    return best_set
