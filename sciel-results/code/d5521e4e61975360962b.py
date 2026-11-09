import time
import random
import math

def construct():
    N = 1000
    max_val = 1000
    deadline = time.time() + 115  # Safety margin
    
    # State
    in_S = [False] * (max_val + 1)
    S = []
    counts = [0] * (2 * max_val + 1)
    overloaded = 0  # Number of sums with count > 2
    
    def add_element(x):
        nonlocal overloaded
        if in_S[x]:
            return
        in_S[x] = True
        S.append(x)
        # Update counts for sums x + y where y in S, y != x
        # Note: We iterate over current S which includes x now? 
        # Better to iterate over S before adding x, then handle x+x?
        # Let's revert S append to after update logic or handle carefully.
        S.pop() # Remove x from S for now to iterate over existing elements
        
        for y in S:
            s = x + y
            old_count = counts[s]
            new_count = old_count + 1
            counts[s] = new_count
            if old_count == 2:
                overloaded += 1
            # If old_count < 2 or > 2, overloaded doesn't change
        
        # Handle 2*x
        s = 2 * x
        old_count = counts[s]
        new_count = old_count + 1
        counts[s] = new_count
        if old_count == 2:
            overloaded += 1
            
        S.append(x)
        in_S[x] = True

    def remove_element(x):
        nonlocal overloaded
        if not in_S[x]:
            return
        in_S[x] = False
        S.remove(x)
        
        for y in S:
            s = x + y
            old_count = counts[s]
            new_count = old_count - 1
            counts[s] = new_count
            if old_count == 3:
                overloaded -= 1
            
        s = 2 * x
        old_count = counts[s]
        new_count = old_count - 1
        counts[s] = new_count
        if old_count == 3:
            overloaded -= 1

    # 1. Greedy Initialization
    # We use a standard greedy approach: add smallest possible number first.
    # This tends to produce dense sets at the lower end.
    for i in range(1, max_val + 1):
        # Check if adding i is valid
        valid = True
        # Check 2*i
        if counts[2 * i] >= 2:
            valid = False
        if valid:
            for y in S:
                s = i + y
                if counts[s] >= 2:
                    valid = False
                    break
        if valid:
            add_element(i)
            
    best_size = len(S)
    best_S = S.copy()
    
    # 2. Simulated Annealing
    # Objective: Minimize Energy = -size + penalty * overloaded
    # We want overloaded == 0 and size max.
    # Start with current state (which is valid, so overloaded=0).
    # We will allow moves that increase overloaded, hoping to find a larger valid set later.
    
    T = 10.0
    T_min = 0.01
    cooling_rate = 0.9999
    
    # We track the best valid solution found so far
    current_size = len(S)
    current_overloaded = overloaded
    current_energy = -current_size + 100.0 * current_overloaded
    
    while time.time() < deadline:
        # Choose a random move: add or remove
        if random.random() < 0.5:
            # Add random element
            x = random.randint(1, max_val)
            if in_S[x]:
                continue
            # Calculate delta
            delta_energy = 0.0
            # Simulate add
            # For each y in S, s = x+y
            # If counts[s] == 2, overloaded increases by 1
            # If counts[2x] == 2, overloaded increases by 1
            # Size increases by 1 -> energy decreases by 1
            
            # Check if this move is always good?
            # We just do it and calculate delta properly
            
            old_size = len(S)
            old_overloaded = overloaded
            
            add_element(x)
            
            new_size = len(S)
            new_overloaded = overloaded
            
            delta_energy = (-new_size + 100.0 * new_overloaded) - (-old_size + 100.0 * old_overloaded)
            
            if delta_energy < 0 or random.random() < math.exp(-delta_energy / T):
                current_size = new_size
                current_overloaded = new_overloaded
                current_energy = -current_size + 100.0 * current_overloaded
                
                if current_overloaded == 0 and current_size > best_size:
                    best_size = current_size
                    best_S = S.copy()
            else:
                # Reject move
                remove_element(x)
        else:
            # Remove random element
            if not S:
                continue
            x = random.choice(S)
            if not in_S[x]:
                continue
            
            old_size = len(S)
            old_overloaded = overloaded
            
            remove_element(x)
            
            new_size = len(S)
            new_overloaded = overloaded
            
            delta_energy = (-new_size + 100.0 * new_overloaded) - (-old_size + 100.0 * old_overloaded)
            
            if delta_energy < 0 or random.random() < math.exp(-delta_energy / T):
                current_size = new_size
                current_overloaded = new_overloaded
                current_energy = -current_size + 100.0 * current_overloaded
                
                if current_overloaded == 0 and current_size > best_size:
                    best_size = current_size
                    best_S = S.copy()
            else:
                # Reject move
                add_element(x)
                
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min

    # Final polish: Ensure best_S is valid and return it
    # best_S is already a valid set from the tracking logic
    return best_S
