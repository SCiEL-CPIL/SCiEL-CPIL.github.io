import random
import math
import time
from fractions import Fraction

def construct():
    N = 256
    # Precompute reciprocals as Fractions for exact arithmetic
    recip = [Fraction(0)] * (N + 1)
    for i in range(1, N + 1):
        recip[i] = Fraction(1, i)
    
    # Precompute conflict: for each x, which pairs (a,b) in S would form a 3-AP with x
    # Actually, let's precompute for each x, the set of y such that x,y form a 3-AP with some z
    # Better: precompute for each pair (a,b) with a<b, the middle m=(a+b)/2 if integer
    # and the two endpoints for each middle
    
    # Precompute: for each x, list of (y, z) such that {x,y,z} forms a 3-AP
    # This is expensive. Instead, for a given set S and candidate x, check conflicts:
    # For each y in S: check if 2*x - y is in S (x is middle or endpoint)
    # Actually: 3-AP means x,y,z with 2y = x+z (y is middle)
    # So for candidate x and existing S:
    #   - x is middle: need y,z in S with 2x = y+z
    #   - y is middle: need z in S with 2y = x+z for some y,z in S
    #   - z is middle: need y in S with 2z = x+y for some y,z in S
    # Simpler: for each y in S, check if 2*y - x is in S (and != y, != x)
    #          and for each y in S, check if x + 2*(y-x) is in S... no
    # Let me think: a 3-AP is {a, a+d, a+2d} for d>0
    # If we add x to S, we need to check:
    #   For each d>0: if x-d in S and x+d in S -> x is middle
    #   For each d>0: if x+2d in S and x-d in S -> x is start... no
    #   Actually: if x is in S, and we add y, check if there exists z in S such that
    #   {x,y,z} is a 3-AP. The three cases:
    #   1. y is middle: 2y = x+z => z = 2y-x, check if z in S
    #   2. x is middle: 2x = y+z => z = 2x-y, check if z in S  
    #   3. z is middle: 2z = x+y => z = (x+y)/2, check if z in S and z != x, z != y
    
    # For efficiency, use a boolean array for membership
    
    def is_valid_addition(S_bool, S_list, x):
        # S_bool: list of bools, S_bool[i] = True if i in S
        # S_list: list of elements in S
        # Check if adding x creates a 3-AP
        for y in S_list:
            # Case 1: y is middle, x is endpoint: z = 2y - x
            z = 2 * y - x
            if z != x and z != y and 1 <= z <= N and S_bool[z]:
                return False
            # Case 2: x is middle: z = 2x - y
            z = 2 * x - y
            if z != x and z != y and 1 <= z <= N and S_bool[z]:
                return False
            # Case 3: z is middle: z = (x+y)/2
            if (x + y) % 2 == 0:
                z = (x + y) // 2
                if z != x and z != y and 1 <= z <= N and S_bool[z]:
                    return False
        return True
    
    # Greedy initialization: add elements in order of decreasing reciprocal
    S_bool = [False] * (N + 1)
    S_list = []
    candidates = sorted(range(1, N + 1), key=lambda x: -recip[x])
    for x in candidates:
        if is_valid_addition(S_bool, S_list, x):
            S_bool[x] = True
            S_list.append(x)
    
    # Compute initial score
    current_score = sum(recip[x] for x in S_list)
    best_score = current_score
    best_S_list = list(S_list)
    best_S_bool = list(S_bool)
    
    start_time = time.time()
    time_limit = 85.0  # Leave 5s margin
    
    # Simulated annealing with better move generation
    T = 0.05
    T_min = 0.001
    cooling_rate = 0.9995
    
    # Precompute for each x, the list of y such that adding y would conflict with x given current S
    # Actually, let's just do local search with targeted moves
    
    iteration = 0
    while time.time() - start_time < time_limit and T > T_min:
        iteration += 1
        
        # Choose move type: 0=add, 1=remove, 2=swap
        move_type = random.choices([0, 1, 2], weights=[1, 1, 2])[0]
        
        new_S_bool = None
        new_S_list = None
        delta = Fraction(0)
        
        if move_type == 0:  # Add
            # Try to add a random element not in S
            x = random.randint(1, N)
            if not S_bool[x]:
                if is_valid_addition(S_bool, S_list, x):
                    new_S_bool = list(S_bool)
                    new_S_bool[x] = True
                    new_S_list = list(S_list)
                    new_S_list.append(x)
                    delta = recip[x]
                else:
                    T *= cooling_rate
                    continue
            else:
                T *= cooling_rate
                continue
                
        elif move_type == 1:  # Remove
            if not S_list:
                T *= cooling_rate
                continue
            x = random.choice(S_list)
            new_S_bool = list(S_bool)
            new_S_bool[x] = False
            new_S_list = [y for y in S_list if y != x]
            delta = -recip[x]
            
        else:  # Swap
            if not S_list:
                T *= cooling_rate
                continue
            x = random.choice(S_list)
            y = random.randint(1, N)
            if S_bool[y]:
                T *= cooling_rate
                continue
            # Check if S - {x} + {y} is valid
            temp_bool = list(S_bool)
            temp_bool[x] = False
            temp_list = [z for z in S_list if z != x]
            if is_valid_addition(temp_bool, temp_list, y):
                new_S_bool = temp_bool
                new_S_bool[y] = True
                new_S_list = temp_list
                new_S_list.append(y)
                delta = recip[y] - recip[x]
            else:
                T *= cooling_rate
                continue
        
        if new_S_bool is None:
            T *= cooling_rate
            continue
            
        # Acceptance criterion
        if delta >= 0 or random.random() < math.exp(float(delta) / T):
            S_bool = new_S_bool
            S_list = new_S_list
            current_score += delta
            if current_score > best_score:
                best_score = current_score
                best_S_list = list(S_list)
                best_S_bool = list(S_bool)
        
        T *= cooling_rate
        
    # Final polish: try all possible single additions
    for x in range(1, N + 1):
        if not best_S_bool[x]:
            if is_valid_addition(best_S_bool, best_S_list, x):
                new_score = best_score + recip[x]
                if new_score > best_score:
                    best_score = new_score
                    best_S_bool[x] = True
                    best_S_list.append(x)
    
    # Try all possible swaps
    changed = True
    while changed and time.time() - start_time < time_limit - 1.0:
        changed = False
        for x in list(best_S_list):
            for y in range(1, N + 1):
                if best_S_bool[y]:
                    continue
                # Check if replacing x with y improves score
                if recip[y] <= recip[x]:
                    continue
                temp_bool = list(best_S_bool)
                temp_bool[x] = False
                temp_list = [z for z in best_S_list if z != x]
                if is_valid_addition(temp_bool, temp_list, y):
                    new_score = best_score - recip[x] + recip[y]
                    if new_score > best_score:
                        best_S_bool[x] = False
                        best_S_bool[y] = True
                        best_S_list.remove(x)
                        best_S_list.append(y)
                        best_score = new_score
                        changed = True
                        break
            if changed:
                break
    
    return list(best_S_list)
