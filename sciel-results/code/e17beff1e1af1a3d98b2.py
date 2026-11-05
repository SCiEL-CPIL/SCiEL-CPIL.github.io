import time
import random
import math

def _count_conflicts(seq):
    n = len(seq)
    if n == 0:
        return 0
    sums = {}
    conflicts = 0
    for i in range(n):
        acc = 0
        for j in range(i, n):
            acc += seq[j]
            if acc in sums:
                conflicts += 1
            else:
                sums[acc] = True
    return conflicts

def _greedy_valid(order):
    vals = []
    for x in order:
        pos = 0
        while pos < len(vals) and vals[pos] < x:
            pos += 1
        if pos < len(vals) and vals[pos] == x:
            continue
        new_seq = vals[:pos] + [x] + vals[pos:]
        if _count_conflicts(new_seq) == 0:
            vals = new_seq
    return vals

def construct():
    start = time.time()
    deadline = start + 85.0
    
    # Get initial valid solution
    best_seq = _greedy_valid(range(1, 257))
    best_len = len(best_seq)
    
    # Try a few greedy variants
    for _ in range(20):
        if time.time() > deadline - 15.0:
            break
        order = random.sample(range(1, 257), 256)
        seq = _greedy_valid(order)
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq
    
    # Simulated annealing
    if time.time() < deadline - 3.0:
        current = list(best_seq)
        current_energy = 0  # We start from a valid solution
        used = set(current)
        
        T = 3.0
        T_min = 0.005
        cooling = 0.99995
        no_improve_count = 0
        max_no_improve = 5000
        
        while time.time() < deadline - 2.0:
            if current_energy == 0:
                # Prefer add moves
                if random.random() < 0.75:
                    # Add
                    unused = [x for x in range(1, 257) if x not in used]
                    if not unused:
                        break
                    x = random.choice(unused)
                    pos = 0
                    while pos < len(current) and current[pos] < x:
                        pos += 1
                    if pos < len(current) and current[pos] == x:
                        T *= cooling
                        continue
                    new_seq = current[:pos] + [x] + current[pos:]
                    new_energy = _count_conflicts(new_seq)
                    delta = new_energy
                    if delta == 0:
                        current = new_seq
                        used.add(x)
                        if len(current) > best_len:
                            best_len = len(current)
                            best_seq = list(current)
                            no_improve_count = 0
                        else:
                            no_improve_count += 1
                    elif random.random() < math.exp(-delta / T):
                        current = new_seq
                        current_energy = new_energy
                        used.add(x)
                        no_improve_count += 1
                    else:
                        no_improve_count += 1
                else:
                    # Remove
                    if not current:
                        break
                    pos = random.randint(0, len(current) - 1)
                    x = current[pos]
                    new_seq = current[:pos] + current[pos+1:]
                    new_energy = _count_conflicts(new_seq)
                    # Removing from a valid seq should keep it valid
                    if new_energy == 0:
                        current = new_seq
                        used.discard(x)
                        no_improve_count += 1
            else:
                # Prefer remove moves to get back to valid
                if random.random() < 0.6:
                    # Remove
                    if not current:
                        current = list(best_seq)
                        current_energy = 0
                        used = set(current)
                        continue
                    pos = random.randint(0, len(current) - 1)
                    x = current[pos]
                    new_seq = current[:pos] + current[pos+1:]
                    new_energy = _count_conflicts(new_seq)
                    delta = new_energy - current_energy
                    if delta <= 0 or random.random() < math.exp(-delta / T):
                        current = new_seq
                        current_energy = new_energy
                        used.discard(x)
                        if new_energy == 0 and len(current) > best_len:
                            best_len = len(current)
                            best_seq = list(current)
                            no_improve_count = 0
                else:
                    # Add or swap
                    unused = [x for x in range(1, 257) if x not in used]
                    if unused:
                        x = random.choice(unused)
                        pos = 0
                        while pos < len(current) and current[pos] < x:
                            pos += 1
                        if pos < len(current) and current[pos] == x:
                            T *= cooling
                            continue
                        new_seq = current[:pos] + [x] + current[pos:]
                        new_energy = _count_conflicts(new_seq)
                        delta = new_energy - current_energy
                        if delta <= 0 or random.random() < math.exp(-delta / T):
                            current = new_seq
                            current_energy = new_energy
                            used.add(x)
            
            T *= cooling
            if T < T_min:
                # Restart
                T = 3.0
                if no_improve_count > max_no_improve:
                    # Reset to best
                    current = list(best_seq)
                    current_energy = 0
                    used = set(current)
                    no_improve_count = 0
        
        # Final cleanup: if current is valid and longer than best, update
        if current_energy == 0 and len(current) > best_len:
            best_len = len(current)
            best_seq = list(current)
    
    # Validate
    if _count_conflicts(best_seq) != 0:
        best_seq = _greedy_valid(range(1, 257))
    
    return best_seq
