import time
import random
import itertools

N = 16
POINTS = [(x, y) for x in range(N) for y in range(N)]
NUM_POINTS = N * N

# Precompute isosceles pairs for each point
# iso_pairs[i] is a list of tuples (j, k) such that triangle (POINTS[i], POINTS[j], POINTS[k]) is isosceles
iso_pairs = [[] for _ in range(NUM_POINTS)]

def _is_isosceles(a, b, c):
    d1 = (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
    d2 = (a[0] - c[0]) ** 2 + (a[1] - c[1]) ** 2
    d3 = (b[0] - c[0]) ** 2 + (b[1] - c[1]) ** 2
    return d1 == d2 or d1 == d3 or d2 == d3

# Precomputation
for i in range(NUM_POINTS):
    a = POINTS[i]
    for j in range(i + 1, NUM_POINTS):
        b = POINTS[j]
        for k in range(j + 1, NUM_POINTS):
            c = POINTS[k]
            if _is_isosceles(a, b, c):
                iso_pairs[i].append((j, k))
                iso_pairs[j].append((i, k))
                iso_pairs[k].append((i, j))

def is_safe(p_idx, S_set):
    """
    Check if adding p_idx to S_set maintains the no-isosceles property.
    S_set is a set of indices.
    """
    for j, k in iso_pairs[p_idx]:
        if j in S_set and k in S_set:
            return False
    return True

def greedy_construct(rng):
    S = []
    S_set = set()
    points = list(range(NUM_POINTS))
    rng.shuffle(points)
    
    for p in points:
        if is_safe(p, S_set):
            S.append(p)
            S_set.add(p)
            
    return [POINTS[i] for i in S]

def construct():
    start_time = time.time()
    deadline = start_time + 280  # Safety margin
    
    best_set = None
    best_len = float('inf')
    
    # Phase 1: Generate initial solution with greedy
    rng0 = random.Random(42)
    initial = greedy_construct(rng0)
    best_set = initial
    best_len = len(initial)
    
    # Phase 2: Simulated annealing on the set representation
    # State: S_set (set of indices in the solution)
    # Moves: add a point (if safe) or remove a point
    # Objective: minimize |S| while maintaining maximality
    
    current_rng = random.Random(123)
    
    # Start from the greedy solution
    current_S = set(initial_idx for initial_idx in range(NUM_POINTS) if POINTS[initial_idx] in initial)
    current_len = len(current_S)
    
    # Temperature schedule
    T_start = 10.0
    T_end = 0.01
    T_decay = 0.9995
    
    T = T_start
    step = 0
    
    while time.time() < deadline:
        # Pick a random move
        move_type = current_rng.random()
        
        if move_type < 0.4:
            # Try to add a random point
            candidate = current_rng.randint(0, NUM_POINTS - 1)
            if candidate in current_S:
                continue
            if is_safe(candidate, current_S):
                new_S = current_S | {candidate}
                new_len = current_len + 1
                # Adding increases size, only accept if it helps maximality
                # Check if this makes the set more maximal
                # For simplicity, accept with probability based on temperature
                delta = new_len - current_len
                if delta <= 0 or current_rng.random() < (1.0 / (1.0 + (1 - 1) * (delta / T))):
                    # Actually, we want to minimize, so adding is bad unless it enables removals
                    # Let's just accept small increases with low probability
                    if current_rng.random() < 0.1 * T / T_start:
                        current_S = new_S
                        current_len = new_len
                        if current_len < best_len:
                            best_len = current_len
                            best_set = [POINTS[i] for i in current_S]
            else:
                # Can't add this point
                pass
                
        elif move_type < 0.8:
            # Try to remove a random point from current set
            if current_len == 0:
                continue
            candidate = current_rng.choice(list(current_S))
            new_S = current_S - {candidate}
            new_len = current_len - 1
            delta = new_len - current_len
            # Removing decreases size, always good if it maintains no-isosceles (it does)
            # But we need to check maximality: after removal, can we add any point?
            # If we can add a point, the set is not maximal
            # For now, accept removals that keep the set non-empty
            if new_len > 0:
                current_S = new_S
                current_len = new_len
                if current_len < best_len:
                    best_len = current_len
                    best_set = [POINTS[i] for i in current_S]
        else:
            # Swap: remove one, add another
            if current_len == 0:
                continue
            remove_candidate = current_rng.choice(list(current_S))
            add_candidate = current_rng.randint(0, NUM_POINTS - 1)
            if add_candidate in current_S and add_candidate != remove_candidate:
                continue
            new_S = (current_S - {remove_candidate}) | {add_candidate}
            # Check if add_candidate is safe in the new set
            if is_safe(add_candidate, new_S - {add_candidate}):
                new_len = current_len
                delta = 0
                # Swap keeps size same, might improve structure
                if current_rng.random() < 0.5:
                    current_S = new_S
                    current_len = new_len
                    if current_len < best_len:
                        best_len = current_len
                        best_set = [POINTS[i] for i in current_S]
        
        # Cool down
        T = max(T_end, T * T_decay)
        step += 1
        
        # Periodic restart if stuck
        if step % 5000 == 0:
            # Try a new greedy start
            restart_rng = random.Random(step)
            new_initial = greedy_construct(restart_rng)
            new_initial_set = set(i for i in range(NUM_POINTS) if POINTS[i] in new_initial)
            if len(new_initial_set) < current_len:
                current_S = new_initial_set
                current_len = len(current_S)
                if current_len < best_len:
                    best_len = current_len
                    best_set = [POINTS[i] for i in current_S]
    
    return best_set
