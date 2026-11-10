"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def _has_4ap(selected, n=256):
    """Check if selected contains a nonconstant 4-term AP."""
    sel = set(selected)
    for a in sel:
        for d in range(1, (n - a) // 3 + 1):
            if a + d in sel and a + 2*d in sel and a + 3*d in sel:
                return True
    return False

def _count_3aps(selected, n=256):
    """Count distinct 3-term APs (a, a+d, a+2d) with d>0."""
    sel = set(selected)
    count = 0
    for a in sel:
        for d in range(1, (n - a) // 2 + 1):
            if a + d in sel and a + 2*d in sel:
                count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 75  # 75s deadline with 15s margin
    n = 256
    
    # Initialize with greedy construction as seed
    selected = set()
    for x in range(1, n + 1):
        if all(not all(x - j*d in selected for j in (1, 2, 3)) for d in range(1, (x-1)//3 + 1)):
            selected.add(x)
    
    current_score = _count_3aps(selected)
    best_selected = set(selected)
    best_score = current_score
    
    # Simulated annealing
    random.seed(42)
    temperature = 10.0
    cooling = 0.9995
    min_temp = 0.01
    
    while time.time() < deadline and temperature > min_temp:
        # Generate a random move
        move_type = random.random()
        
        if move_type < 0.4:
            # Add an element
            candidates = [x for x in range(1, n+1) if x not in selected]
            if not candidates:
                break
            x = random.choice(candidates)
            # Check if adding x creates a 4-AP
            new_selected = selected | {x}
            if not _has_4ap(new_selected):
                new_score = _count_3aps(new_selected)
                delta = new_score - current_score
                if delta >= 0 or random.random() < np.exp(delta / temperature):
                    selected = new_selected
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_selected = set(selected)
        elif move_type < 0.8:
            # Remove an element
            if not selected:
                break
            x = random.choice(list(selected))
            new_selected = selected - {x}
            new_score = _count_3aps(new_selected)
            delta = new_score - current_score
            if delta >= 0 or random.random() < np.exp(delta / temperature):
                selected = new_selected
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_selected = set(selected)
        else:
            # Swap: remove one, add another
            if len(selected) < 2:
                break
            x_out = random.choice(list(selected))
            candidates = [x for x in range(1, n+1) if x not in selected]
            if not candidates:
                break
            x_in = random.choice(candidates)
            new_selected = (selected - {x_out}) | {x_in}
            if not _has_4ap(new_selected):
                new_score = _count_3aps(new_selected)
                delta = new_score - current_score
                if delta >= 0 or random.random() < np.exp(delta / temperature):
                    selected = new_selected
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_selected = set(selected)
        
        temperature *= cooling
    
    # Final local search pass: try to add any element that doesn't create 4-AP
    for x in range(1, n + 1):
        if x not in best_selected:
            test = best_selected | {x}
            if not _has_4ap(test):
                best_selected = test
                best_score = _count_3aps(best_selected)
    
    return sorted(best_selected)
# EVOLVE-BLOCK-END
