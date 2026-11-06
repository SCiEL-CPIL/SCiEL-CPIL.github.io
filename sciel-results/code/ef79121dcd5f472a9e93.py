# EVOLVE-BLOCK-START
import numpy as np
import random
import time

def construct():
    """
    Construct a regular tournament where every 4 vertices have a common dominator.
    Strategy: Randomized Local Search (Simulated Annealing) on the space of tournaments,
    with a penalty for non-regularity to ensure the final result is regular.
    We target n=67 first. If not found, we try larger n.
    """
    
    start_time = time.time()
    time_limit = 110.0  # Leave 10s for buffer
    
    def generate_random_regular_tournament(n):
        """
        Generate a random regular tournament.
        Method: Start with a random tournament, then fix degrees by swapping edges.
        Simpler: Use a random permutation and create a cyclic tournament?
        Cyclic tournament is regular.
        Or just random and accept if degrees are close, then fix?
        For simplicity and speed, we'll start with a random tournament and use the search
        to fix both regularity and the 4-set property.
        """
        wins = [0] * n
        # Initialize random
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.5:
                    wins[i] |= (1 << j)
                else:
                    wins[j] |= (1 << i)
        return wins

    def is_valid_regular(wins, n):
        """
        Check if the tournament is regular and every 4-set has a dominator.
        """
        target_deg = (n - 1) // 2
        # Check regularity
        for v in range(n):
            if bin(wins[v]).count('1') != target_deg:
                return False
        
        # Check 4-sets
        # Sample if n is large? No, we need to be sure.
        # For n=67, number of 4-sets is ~700,000.
        # Checking all is fast with bitmasks.
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    for d in range(c+1, n):
                        # Check if any of a,b,c,d dominates the other 3
                        # a dominates b,c,d?
                        if (wins[a] >> b) & 1 and (wins[a] >> c) & 1 and (wins[a] >> d) & 1:
                            continue
                        # b dominates a,c,d?
                        if (wins[b] >> a) & 1 and (wins[b] >> c) & 1 and (wins[b] >> d) & 1:
                            continue
                        # c dominates a,b,d?
                        if (wins[c] >> a) & 1 and (wins[c] >> b) & 1 and (wins[c] >> d) & 1:
                            continue
                        # d dominates a,b,c?
                        if (wins[d] >> a) & 1 and (wins[d] >> b) & 1 and (wins[d] >> c) & 1:
                            continue
                        return False
        return True

    def get_cost(wins, n):
        """
        Cost = num_bad_4sets + 1000 * num_degree_violations.
        """
        target_deg = (n - 1) // 2
        deg_viol = 0
        for v in range(n):
            deg = bin(wins[v]).count('1')
            deg_viol += abs(deg - target_deg)
        
        bad_sets = 0
        # Sample a subset of 4-sets for speed during optimization?
        # If we sample, we might miss the exact 0.
        # But we verify at the end.
        # Let's sample 1000 4-sets for the cost function in the loop.
        # This makes the search faster.
        sample_size = 500
        for _ in range(sample_size):
            s = random.sample(range(n), 4)
            a, b, c, d = s
            valid = False
            if (wins[a] >> b) & 1 and (wins[a] >> c) & 1 and (wins[a] >> d) & 1:
                valid = True
            elif (wins[b] >> a) & 1 and (wins[b] >> c) & 1 and (wins[b] >> d) & 1:
                valid = True
            elif (wins[c] >> a) & 1 and (wins[c] >> b) & 1 and (wins[c] >> d) & 1:
                valid = True
            elif (wins[d] >> a) & 1 and (wins[d] >> b) & 1 and (wins[d] >> c) & 1:
                valid = True
            if not valid:
                bad_sets += 1
        
        return bad_sets + 1000 * deg_viol

    # Try to find a solution for n=67
    n_target = 67
    wins = generate_random_regular_tournament(n_target)
    
    # Initial cost
    current_cost = get_cost(wins, n_target)
    
    T = 50.0
    min_T = 0.01
    cooling_rate = 0.99
    
    best_wins = wins[:]
    best_cost = current_cost
    
    while time.time() - start_time < time_limit:
        # Pick a random edge to flip
        u = random.randint(0, n_target - 1)
        v = random.randint(0, n_target - 1)
        if u == v:
            continue
            
        # Check current direction
        # If u beats v, bit v is set in wins[u]
        if (wins[u] >> v) & 1:
            # Flip: u no longer beats v, v beats u
            old_wins_u = wins[u]
            old_wins_v = wins[v]
            
            wins[u] = old_wins_u & ~(1 << v)
            wins[v] = old_wins_v | (1 << u)
            
            new_cost = get_cost(wins, n_target)
            delta = new_cost - current_cost
            
            if delta < 0 or random.random() < np.exp(-delta / T):
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_wins = wins[:]
            else:
                # Revert
                wins[u] = old_wins_u
                wins[v] = old_wins_v
                
        else:
            # v beats u. Flip: v no longer beats u, u beats v
            old_wins_u = wins[u]
            old_wins_v = wins[v]
            
            wins[u] = old_wins_u | (1 << v)
            wins[v] = old_wins_v & ~(1 << u)
            
            new_cost = get_cost(wins, n_target)
            delta = new_cost - current_cost
            
            if delta < 0 or random.random() < np.exp(-delta / T):
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_wins = wins[:]
            else:
                # Revert
                wins[u] = old_wins_u
                wins[v] = old_wins_v
                
        T *= cooling_rate
        if T < min_T:
            T = min_T
            
        # If we found a perfect sample cost, verify fully
        if best_cost == 0:
            if is_valid_regular(best_wins, n_target):
                # Convert to orientation list
                orientation = []
                for i in range(n_target):
                    for j in range(i+1, n_target):
                        if (best_wins[i] >> j) & 1:
                            orientation.append(0) # i -> j
                        else:
                            orientation.append(1) # j -> i
                return {'n': n_target, 'orientation': orientation}

    # If n=67 failed, try n=69?
    # For now, fallback to a known valid structure if we have time.
    # Paley 83 is a safe fallback.
    p = 83
    squares = {x*x % p for x in range(1, p)}
    orientation = []
    for i in range(p):
        for j in range(i+1, p):
            orientation.append(0 if (j-i) % p in squares else 1)
    return {'n': p, 'orientation': orientation}

# EVOLVE-BLOCK-END
