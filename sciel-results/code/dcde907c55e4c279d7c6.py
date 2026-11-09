import itertools
import random
import time
import numpy as np

N = 16
POINTS = list(itertools.product(range(N), repeat=2))
POINT_TO_ID = {p: i for i, p in enumerate(POINTS)}
ID_TO_POINT = {i: p for i, p in enumerate(POINTS)}
NUM_POINTS = len(POINTS)

# Precompute squared distances
DIST = np.zeros((NUM_POINTS, NUM_POINTS), dtype=int)
for i in range(NUM_POINTS):
    x1, y1 = POINTS[i]
    for j in range(i+1, NUM_POINTS):
        x2, y2 = POINTS[j]
        d = (x1-x2)**2 + (y1-y2)**2
        DIST[i][j] = d
        DIST[j][i] = d

def can_add(p_id, current_ids):
    """
    Check if p_id can be added to current_ids without forming an isosceles triangle.
    """
    n = len(current_ids)
    if n < 2:
        return True
    
    # Use numpy for vectorized distance checks?
    # d_pa = DIST[p_id][current_ids]
    # d_ab = DIST[np.ix_(current_ids, current_ids)]
    # Check if any d_pa[i] == d_pa[j] or d_pa[i] == d_ab[i][j]
    
    # For small n, loop is faster due to overhead of numpy indexing.
    # But let's try a hybrid: if n > 5, use numpy.
    if n > 10:
        ids_arr = np.array(current_ids)
        d_pa = DIST[p_id, ids_arr]
        d_ab = DIST[np.ix_(ids_arr, ids_arr)]
        
        # d_pa == d_pa.T (symmetric, so check upper triangle)
        # Actually, we just need to check if there exists i != j such that d_pa[i] == d_pa[j]
        # or d_pa[i] == d_ab[i, j]
        
        # Check 1: d_pa[i] == d_pa[j] for i != j
        # We can use a set of d_pa values. If len(set) < n, there's a duplicate.
        if len(set(d_pa)) < n:
            return False
            
        # Check 2: d_pa[i] == d_ab[i, j] for i != j
        # This is harder to vectorize simply without broadcasting.
        # d_ab is n x n. d_pa is n.
        # We want to know if there exists i, j (i!=j) such that d_pa[i] == d_ab[i, j]
        # Note d_ab[i, i] = 0, d_pa[i] > 0 (since p != a).
        
        # Let's stick to the loop for simplicity and correctness, it's O(n^2) which is fine for n~10-20.
        pass

    for i in range(n):
        a = current_ids[i]
        d_pa = DIST[p_id][a]
        for j in range(i + 1, n):
            b = current_ids[j]
            d_pb = DIST[p_id][b]
            d_ab = DIST[a][b]
            
            if d_pa == d_pb or d_pa == d_ab or d_pb == d_ab:
                return False
    return True

def get_addable_points(current_ids):
    """
    Return list of points not in current_ids that can be added.
    """
    current_set = set(current_ids)
    addable = []
    for p_id in range(NUM_POINTS):
        if p_id in current_set:
            continue
        if can_add(p_id, current_ids):
            addable.append(p_id)
    return addable

def greedy_fill(initial_ids, order=None):
    """
    Greedily add points to make the set maximal.
    """
    if order is None:
        order = list(range(NUM_POINTS))
        random.shuffle(order)
        
    current_set = list(initial_ids)
    current_set_set = set(current_set)
    
    for p_id in order:
        if p_id in current_set_set:
            continue
        if can_add(p_id, current_set):
            current_set.append(p_id)
            current_set_set.add(p_id)
            
    return current_set

def calculate_cost(current_ids, addable_count=None):
    """
    Cost = size + lambda * (number of addable points)
    """
    if addable_count is None:
        addable_count = len(get_addable_points(current_ids))
    # Lambda should be large to enforce maximality
    # If we find a set of size 10 with 0 addable, cost 10.
    # If we find a set of size 5 with 10 addable, cost 5 + 10*100 = 1005.
    # We want to avoid non-maximal sets.
    LAMBDA = 1000.0
    return len(current_ids) + LAMBDA * addable_count

def simulated_annealing(initial_ids, start_time, deadline):
    """
    Run Simulated Annealing.
    """
    current_set = list(initial_ids)
    current_addable = get_addable_points(current_set)
    current_cost = calculate_cost(current_set, len(current_addable))
    
    best_set = list(current_set)
    best_cost = current_cost
    
    T = 100.0  # Initial temperature
    T_min = 0.1
    cooling_rate = 0.999
    
    while time.time() < deadline and T > T_min:
        # Generate a neighbor
        move_type = random.random()
        
        if move_type < 0.4:
            # Remove a point
            if not current_set:
                continue
            idx = random.randint(0, len(current_set) - 1)
            p_to_remove = current_set[idx]
            new_set = [p for i, p in enumerate(current_set) if i != idx]
            
        elif move_type < 0.8:
            # Add a point
            if not current_addable:
                continue
            p_to_add = random.choice(current_addable)
            new_set = current_set + [p_to_add]
            
        else:
            # Swap: remove one, add one
            if not current_set or not current_addable:
                continue
            p_to_remove = random.choice(current_set)
            p_to_add = random.choice(current_addable)
            # Check if p_to_add is still valid after removing p_to_remove
            temp_set = [p for p in current_set if p != p_to_remove]
            if can_add(p_to_add, temp_set):
                new_set = temp_set + [p_to_add]
            else:
                continue
        
        # Calculate new cost
        # Note: new_set is always no-isosceles by construction of moves.
        new_addable = get_addable_points(new_set)
        new_cost = calculate_cost(new_set, len(new_addable))
        
        # Acceptance
        delta = new_cost - current_cost
        if delta < 0 or random.random() < np.exp(-delta / T):
            current_set = new_set
            current_addable = new_addable
            current_cost = new_cost
            
            # Update best
            # We only care about maximal sets for the final answer, 
            # but for tracking progress, we can track the best "effective" size.
            # If current_set is maximal (addable empty), its size is a candidate.
            if len(current_addable) == 0:
                if len(current_set) < len(best_set) or (len(current_set) == len(best_set) and current_cost < best_cost):
                    best_set = list(current_set)
                    best_cost = current_cost
            else:
                # Even if not maximal, if it's "close" to being small, it might lead to a good maximal set.
                # But strictly, we want the best maximal set found so far.
                pass
                
        T *= cooling_rate
        
    # At the end, ensure best_set is maximal
    # If best_set is not maximal, fill it
    if get_addable_points(best_set):
        best_set = greedy_fill(best_set)
        
    return best_set

def construct():
    start_time = time.time()
    deadline = start_time + 290.0
    
    best_set = []
    best_size = float('inf')
    
    # Initial greedy runs to find a good starting point
    for _ in range(5):
        init_set = greedy_fill([])
        if len(init_set) < best_size:
            best_size = len(init_set)
            best_set = init_set
            
    # Run SA
    # Use the best greedy set as initial state
    sa_set = simulated_annealing(best_set, start_time, deadline)
    
    if len(sa_set) < best_size:
        best_size = len(sa_set)
        best_set = sa_set
        
    # If time remains, do more SA restarts?
    # SA is stochastic, so restarts can help.
    while time.time() < deadline - 10.0:
        # Start from a random greedy set or the best set?
        # Starting from best set might get stuck. Start from random.
        init_set = greedy_fill([])
        sa_set = simulated_annealing(init_set, start_time, deadline - 5.0) # Leave 5s margin
        if len(sa_set) < best_size:
            best_size = len(sa_set)
            best_set = sa_set
            
    return [ID_TO_POINT[i] for i in best_set]
