import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    # Try to find a valid coloring for n=34 down to n=1
    # The known lower bound is 34.
    
    for n in range(34, 0, -1):
        if time.time() > deadline - 15:
            break
        success, colors = find_valid_coloring_smooth(n, deadline)
        if success:
            return {'n': n, 'colors': colors}
            
    # Fallback: return a valid small coloring if we ran out of time
    # n=9 from seed is valid.
    n = 9
    return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]}

def find_valid_coloring_smooth(n, deadline):
    # Precompute structures
    triples = list(combinations(range(n), 3))
    num_triples = len(triples)
    triple_index = {t: i for i, t in enumerate(triples)}
    
    # Precompute K4s and K5s as lists of triple indices
    k4s = []
    for quad in combinations(range(n), 4):
        idxs = [triple_index[t] for t in combinations(quad, 3)]
        k4s.append(idxs)
        
    k5s = []
    for quint in combinations(range(n), 5):
        idxs = [triple_index[t] for t in combinations(quint, 3)]
        k5s.append(idxs)
        
    num_k4s = len(k4s)
    num_k5s = len(k5s)
    
    # For each triple, precompute which K4s and K5s it belongs to
    triple_to_k4s = [[] for _ in range(num_triples)]
    triple_to_k5s = [[] for _ in range(num_triples)]
    
    for ki, idxs in enumerate(k4s):
        for idx in idxs:
            triple_to_k4s[idx].append(ki)
            
    for ki, idxs in enumerate(k5s):
        for idx in idxs:
            triple_to_k5s[idx].append(ki)
            
    # Simulated Annealing with SMOOTH OBJECTIVE
    # Objective: Minimize sum(r_k4^2) + sum(b_k5^2)
    # This penalizes "almost" violations, providing a gradient.
    
    best_energy = float('inf')
    best_colors = None
    
    # Run multiple restarts
    max_restarts = 50
    restarts_done = 0
    
    while time.time() < deadline - 5 and restarts_done < max_restarts:
        # Random initialization
        colors = [random.randint(0, 1) for _ in range(num_triples)]
        
        # Compute initial counts
        k4_red_count = [0] * num_k4s
        k5_blue_count = [0] * num_k5s
        
        for ki, idxs in enumerate(k4s):
            for idx in idxs:
                if colors[idx] == 0:
                    k4_red_count[ki] += 1
                    
        for ki, idxs in enumerate(k5s):
            for idx in idxs:
                if colors[idx] == 1:
                    k5_blue_count[ki] += 1
        
        # Initial energy: sum of squares
        energy = sum(c*c for c in k4_red_count) + sum(c*c for c in k5_blue_count)
        
        # Check if we already have a solution (unlikely with random init for n>10)
        # A valid solution requires max red in K4 < 4 and max blue in K5 < 10.
        # Actually, we need NO K4 with 4 red and NO K5 with 10 blue.
        # The smooth objective is 0 only if all counts are 0, which is impossible.
        # We need to check the hard constraint separately.
        
        def is_valid():
            for c in k4_red_count:
                if c == 4:
                    return False
            for c in k5_blue_count:
                if c == 10:
                    return False
            return True
            
        if is_valid():
            return True, colors
            
        if energy < best_energy:
            best_energy = energy
            best_colors = colors[:]
        
        # Simulated Annealing
        temperature = 50.0  # Start high to allow exploration
        cooling = 0.999
        no_improvement = 0
        max_no_improvement = 2000
        
        while no_improvement < max_no_improvement and time.time() < deadline - 5:
            # Pick random triple to flip
            i = random.randint(0, num_triples - 1)
            old_color = colors[i]
            new_color = 1 - old_color
            
            # Calculate delta energy using (r+1)^2 - r^2 = 2r + 1
            # and (r-1)^2 - r^2 = -2r + 1
            delta = 0
            
            # Update K4s containing triple i
            for ki in triple_to_k4s[i]:
                old_red = k4_red_count[ki]
                if old_color == 0:
                    # Was red, now blue: red count decreases
                    # delta = (old_red - 1)^2 - old_red^2 = -2*old_red + 1
                    delta += -2 * old_red + 1
                    k4_red_count[ki] = old_red - 1
                else:
                    # Was blue, now red: red count increases
                    # delta = (old_red + 1)^2 - old_red^2 = 2*old_red + 1
                    delta += 2 * old_red + 1
                    k4_red_count[ki] = old_red + 1
                    
            # Update K5s containing triple i
            for ki in triple_to_k5s[i]:
                old_blue = k5_blue_count[ki]
                if old_color == 1:
                    # Was blue, now red: blue count decreases
                    # delta = (old_blue - 1)^2 - old_blue^2 = -2*old_blue + 1
                    delta += -2 * old_blue + 1
                    k5_blue_count[ki] = old_blue - 1
                else:
                    # Was red, now blue: blue count increases
                    # delta = (old_blue + 1)^2 - old_blue^2 = 2*old_blue + 1
                    delta += 2 * old_blue + 1
                    k5_blue_count[ki] = old_blue + 1
            
            energy += delta
            
            # Accept or reject
            if delta <= 0 or random.random() < pow(2.718281828, -delta / max(temperature, 0.001)):
                colors[i] = new_color
                if delta < 0:
                    no_improvement = 0
                    if energy < best_energy:
                        best_energy = energy
                        best_colors = colors[:]
                else:
                    no_improvement += 1
                    
                # Check for valid solution
                if is_valid():
                    return True, colors
                    
            else:
                # Reject: revert the counts
                for ki in triple_to_k4s[i]:
                    if old_color == 0:
                        k4_red_count[ki] += 1
                    else:
                        k4_red_count[ki] -= 1
                for ki in triple_to_k5s[i]:
                    if old_color == 1:
                        k5_blue_count[ki] += 1
                    else:
                        k5_blue_count[ki] -= 1
                energy -= delta
                no_improvement += 1
            
            temperature *= cooling
        
        restarts_done += 1
        
    # If we didn't find a valid solution, return False
    # Note: best_colors might be a "near" solution, but we must return a VALID one or None
    # The caller handles the fallback.
    return False, None
