import time
import random

N = 512
K = 4

# Precompute all 4-AP constraints
def precompute_constraints():
    """Return list of tuples (a, b, c, d) for all 4-APs in [1, N]."""
    constraints = []
    for a in range(1, N - 3):
        for d in range(1, (N - a) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e <= N:
                constraints.append((a, b, c, e))
    return constraints

def is_valid_addition(x, selected, constraints_by_x):
    """Check if adding x to selected creates any 4-AP."""
    for (a, b, c, d) in constraints_by_x.get(x, []):
        if (a in selected and b in selected and c in selected) or \
           (a in selected and b in selected and d in selected) or \
           (a in selected and c in selected and d in selected) or \
           (b in selected and c in selected and d in selected):
            return False
    return True

def greedy_construct():
    """Simple greedy: iterate 1..N, add if no 4-AP formed."""
    selected = set()
    for x in range(1, N + 1):
        if all(not all(x - j * d in selected for j in range(1, 4)) for d in range(1, (x - 1) // 3 + 1)):
            selected.add(x)
    return selected

def simulated_annealing(initial, constraints_by_x, time_budget):
    """SA with periodic fill passes to escape local optima."""
    start_time = time.time()
    selected = set(initial)
    best_selected = set(selected)
    best_size = len(selected)
    
    T = 50.0
    cooling = 0.995
    min_T = 0.01
    
    while time.time() - start_time < time_budget:
        # Try a random move: remove one, add another, or swap
        if len(selected) < N:
            # Try adding a random unselected element
            unselected = [x for x in range(1, N + 1) if x not in selected]
            if unselected:
                x = random.choice(unselected)
                if is_valid_addition(x, selected, constraints_by_x):
                    selected.add(x)
                    if len(selected) > best_size:
                        best_size = len(selected)
                        best_selected = set(selected)
                else:
                    # With probability T, accept a bad move (remove something to make room)
                    if random.random() < T / 100:
                        # Remove a random element to potentially allow addition
                        to_remove = random.choice(list(selected))
                        selected.discard(to_remove)
                        if is_valid_addition(x, selected, constraints_by_x):
                            selected.add(x)
        else:
            # Swap: remove one, add another
            to_remove = random.choice(list(selected))
            unselected = [x for x in range(1, N + 1) if x not in selected]
            if not unselected:
                break
            x = random.choice(unselected)
            selected.discard(to_remove)
            if is_valid_addition(x, selected, constraints_by_x):
                selected.add(x)
                if len(selected) > best_size:
                    best_size = len(selected)
                    best_selected = set(selected)
            else:
                # Revert
                selected.discard(x)
                selected.add(to_remove)
                if random.random() < T / 200:
                    # Accept a bad swap
                    selected.discard(to_remove)
                    selected.add(x)
        
        T *= cooling
        if T < min_T:
            T = min_T
            # Periodic fill pass: try to add elements greedily
            for x in range(1, N + 1):
                if x not in selected and is_valid_addition(x, selected, constraints_by_x):
                    selected.add(x)
            if len(selected) > best_size:
                best_size = len(selected)
                best_selected = set(selected)
    
    return best_selected

def construct():
    deadline = 50  # seconds, leave margin for 60s limit
    
    # Precompute constraints by element
    constraints_by_x = {x: [] for x in range(1, N + 1)}
    for (a, b, c, d) in precompute_constraints():
        for x in (a, b, c, d):
            constraints_by_x[x].append((a, b, c, d))
    
    # Start with greedy
    initial = greedy_construct()
    best_result = set(initial)
    best_size = len(initial)
    
    start_time = time.time()
    
    # Run SA for remaining time
    if time.time() - start_time < deadline - 5:
        sa_result = simulated_annealing(initial, constraints_by_x, deadline - 5)
        if len(sa_result) > best_size:
            best_result = sa_result
            best_size = len(sa_result)
    
    # Final greedy fill
    for x in range(1, N + 1):
        if x not in best_result and is_valid_addition(x, best_result, constraints_by_x):
            best_result.add(x)
    
    return sorted(best_result)
