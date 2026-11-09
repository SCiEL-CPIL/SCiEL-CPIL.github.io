import time
from fractions import Fraction
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Precompute reciprocal values as Fractions
    reciprocals = [Fraction(1, d) for d in denominators]
    
    # Greedy initialization: pick largest reciprocals first
    selected = []
    total = Fraction(0)
    for i in range(n):
        if total + reciprocals[i] < 1:
            selected.append(i)
            total += reciprocals[i]
    
    best_indices = list(selected)
    best_sum = total
    best_deficit = 1 - best_sum
    
    # Simulated annealing approach
    # State: set of selected indices
    # Move: add/remove one index, or swap two
    
    def get_deficit(indices, current_sum):
        return 1 - current_sum
    
    # Use a set for fast lookup
    current_set = set(best_indices)
    current_sum = best_sum
    current_deficit = best_deficit
    
    temperature = 0.01
    cooling_rate = 0.9995
    min_temperature = 1e-10
    
    step = 0
    while time.time() < deadline and temperature > min_temperature:
        step += 1
        
        # Generate a random move
        move_type = random.random()
        
        if move_type < 0.4:
            # Remove a random selected index
            if not current_set:
                temperature *= cooling_rate
                continue
            i = random.choice(list(current_set))
            new_set = current_set - {i}
            new_sum = current_sum - reciprocals[i]
            new_deficit = 1 - new_sum
        elif move_type < 0.8:
            # Add a random unselected index
            unselected = [i for i in range(n) if i not in current_set]
            if not unselected:
                temperature *= cooling_rate
                continue
            j = random.choice(unselected)
            new_sum = current_sum + reciprocals[j]
            if new_sum >= 1:
                temperature *= cooling_rate
                continue
            new_set = current_set | {j}
            new_deficit = 1 - new_sum
        else:
            # Swap: remove one, add one
            if not current_set:
                temperature *= cooling_rate
                continue
            i = random.choice(list(current_set))
            unselected = [j for j in range(n) if j not in current_set]
            if not unselected:
                temperature *= cooling_rate
                continue
            j = random.choice(unselected)
            new_sum = current_sum - reciprocals[i] + reciprocals[j]
            if new_sum >= 1:
                temperature *= cooling_rate
                continue
            new_set = (current_set - {i}) | {j}
            new_deficit = 1 - new_sum
        
        # Accept or reject based on simulated annealing
        delta = new_deficit - current_deficit  # positive means worse
        if delta < 0:
            # Better solution, accept
            current_set = new_set
            current_sum = new_sum
            current_deficit = new_deficit
            if current_deficit < best_deficit:
                best_deficit = current_deficit
                best_sum = current_sum
                best_indices = list(current_set)
        else:
            # Worse solution, accept with probability exp(-delta/temperature)
            if delta == 0:
                current_set = new_set
                current_sum = new_sum
                current_deficit = new_deficit
            else:
                try:
                    prob = math.exp(-delta / temperature)
                except OverflowError:
                    prob = 0.0
                if random.random() < prob:
                    current_set = new_set
                    current_sum = new_sum
                    current_deficit = new_deficit
        
        temperature *= cooling_rate
    
    return best_indices
