import time
import random
from itertools import combinations

N = 256

def construct():
    """
    Construct a subset of [1,256] with no 4-term AP, maximizing 3-term APs.
    Uses bitmask-based simulated annealing for speed and effectiveness.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # Precompute all 3-term APs and 4-term APs
    three_aps = []
    for a in range(1, N + 1):
        for d in range(1, (N - a) // 2 + 1):
            b = a + d
            c = a + 2 * d
            if c <= N:
                three_aps.append((a, b, c))
    
    four_aps = []
    for a in range(1, N + 1):
        for d in range(1, (N - a) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e <= N:
                four_aps.append((a, b, c, e))
    
    # For each element, precompute which 3APs and 4APs it participates in
    elem_3aps = {x: [] for x in range(1, N + 1)}
    for idx, (a, b, c) in enumerate(three_aps):
        elem_3aps[a].append(idx)
        elem_3aps[b].append(idx)
        elem_3aps[c].append(idx)
    
    elem_4aps = {x: [] for x in range(1, N + 1)}
    for idx, (a, b, c, e) in enumerate(four_aps):
        elem_4aps[a].append(idx)
        elem_4aps[b].append(idx)
        elem_4aps[c].append(idx)
        elem_4aps[e].append(idx)
    
    # Initialize with greedy construction
    selected = set()
    for x in range(1, N + 1):
        # Check if adding x creates a 4-AP
        valid = True
        for d in range(1, (x - 1) // 3 + 1):
            if all(x - j * d in selected for j in (1, 2, 3)):
                valid = False
                break
        if valid:
            selected.add(x)
    
    # Convert to bitmask for fast operations
    mask = 0
    for x in selected:
        mask |= (1 << (x - 1))
    
    def count_3aps(mask):
        count = 0
        for (a, b, c) in three_aps:
            if (mask & (1 << (a - 1))) and (mask & (1 << (b - 1))) and (mask & (1 << (c - 1))):
                count += 1
        return count
    
    def has_4ap(mask):
        for (a, b, c, e) in four_aps:
            if (mask & (1 << (a - 1))) and (mask & (1 << (b - 1))) and (mask & (1 << (c - 1))) and (mask & (1 << (e - 1))):
                return True
        return False
    
    current_score = count_3aps(mask)
    best_mask = mask
    best_score = current_score
    
    # Simulated annealing
    temperature = 5.0
    cooling_rate = 0.9995
    min_temp = 0.1
    
    while time.time() < deadline and temperature > min_temp:
        # Choose a random move: add or remove an element
        if random.random() < 0.5:
            # Remove a random element
            elements = [x for x in range(1, N + 1) if mask & (1 << (x - 1))]
            if not elements:
                temperature *= cooling_rate
                continue
            x = random.choice(elements)
            new_mask = mask & ~(1 << (x - 1))
            if has_4ap(new_mask):
                continue
            new_score = count_3aps(new_mask)
        else:
            # Add a random element
            not_in = [x for x in range(1, N + 1) if not (mask & (1 << (x - 1)))]
            if not not_in:
                temperature *= cooling_rate
                continue
            x = random.choice(not_in)
            new_mask = mask | (1 << (x - 1))
            if has_4ap(new_mask):
                continue
            new_score = count_3aps(new_mask)
        
        delta = new_score - current_score
        if delta > 0 or random.random() < (10 ** ((delta - 1) / temperature)):
            mask = new_mask
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_mask = mask
        
        temperature *= cooling_rate
    
    # Extract result from best mask
    result = [x for x in range(1, N + 1) if best_mask & (1 << (x - 1))]
    return sorted(result)
