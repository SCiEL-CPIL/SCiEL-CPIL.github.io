import time
from math import gcd, lcm
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    # 90s limit, use 85s for safety
    deadline = start_time + 85.0
    
    N = 256
    values = list(range(2, N + 1))
    n = len(values)
    
    # 1. Precompute Conflict Graph
    # conflict[i][j] = True if lcm(values[i], values[j]) <= 256
    # We want an independent set in this graph (no edges between selected nodes)
    # Using a list of lists of neighbors for fast compatibility checks
    neighbors = [set() for _ in range(n)]
    for i in range(n):
        vi = values[i]
        for j in range(i + 1, n):
            vj = values[j]
            # lcm(a,b) = a*b//gcd(a,b)
            # To avoid overflow or just for speed, check lcm <= 256
            g = gcd(vi, vj)
            if (vi // g) * vj <= 256:
                neighbors[i].add(j)
                neighbors[j].add(i)
                
    weights = [1.0 / values[i] for i in range(n)]
    
    # Helper to check if a vertex can be added to a set
    # Representing the set as a boolean array or a set of indices
    # For speed in SA, we maintain a set of chosen indices and a set of forbidden indices?
    # Or just a set of chosen indices and check neighbors.
    
    def is_compatible(v, chosen_set):
        for u in chosen_set:
            if u in neighbors[v]:
                return False
        return True

    def score_of_set(chosen_set):
        return sum(weights[i] for i in chosen_set)

    # Randomized Greedy Initialization
    def random_greedy_init():
        avail = set(range(n))
        chosen = set()
        while avail:
            # Pick a random vertex from available
            # To bias towards high weights, we could sample with probability proportional to weight,
            # but simple random selection from a shuffled list is a good diverse start.
            # Let's pick the highest weight among a random subset to balance greediness and randomness
            items = list(avail)
            random.shuffle(items)
            # Pick the first one that is "good enough" or just the first?
            # Let's just pick a random one, but sort by weight desc and pick from top K?
            # Actually, standard randomized greedy: sort by weight desc, then pick random from top 50%?
            # Let's stick to: sort by weight desc, iterate, if random() < 0.5 pick it?
            # Better: Just pick a random element from avail.
            v = random.choice(items)
            chosen.add(v)
            # Remove v and its neighbors from avail
            to_remove = set(neighbors[v])
            to_remove.add(v)
            avail -= to_remove
        return chosen

    # Simulated Annealing
    def simulated_annealing(init_set, time_limit):
        current_set = set(init_set)
        current_score = score_of_set(current_set)
        best_set = set(current_set)
        best_score = current_score
        
        # Temperature parameters
        T_start = 0.01
        T_end = 0.00001
        cooling_rate = 0.99
        
        T = T_start
        no_improve_count = 0
        
        while T > T_end:
            if time.time() > time_limit:
                break
            
            # Generate a neighbor
            # Moves: 
            # 1. Add a random compatible vertex
            # 2. Remove a random vertex
            # 3. Swap a random vertex with a compatible one
            move_type = random.random()
            
            new_set = set(current_set)
            
            if move_type < 0.33:
                # Add
                candidates = [i for i in range(n) if i not in current_set and is_compatible(i, current_set)]
                if candidates:
                    v = random.choice(candidates)
                    new_set.add(v)
            elif move_type < 0.66:
                # Remove
                if current_set:
                    v = random.choice(list(current_set))
                    new_set.remove(v)
            else:
                # Swap
                if current_set:
                    v_out = random.choice(list(current_set))
                    temp_set = current_set - {v_out}
                    candidates = [i for i in range(n) if i not in current_set and is_compatible(i, temp_set)]
                    if candidates:
                        v_in = random.choice(candidates)
                        new_set.remove(v_out)
                        new_set.add(v_in)
                    else:
                        # No valid swap, try a simple add/remove or skip
                        # Fallback to add if possible
                        candidates = [i for i in range(n) if i not in current_set and is_compatible(i, current_set)]
                        if candidates:
                            v_in = random.choice(candidates)
                            new_set.add(v_in)
                        else:
                            # Skip move
                            T *= cooling_rate
                            continue

            new_score = score_of_set(new_set)
            delta = new_score - current_score
            
            # Acceptance probability
            if delta > 0:
                accept = True
            else:
                # Avoid overflow in exp
                if T > 1e-15:
                    accept_prob = min(1.0, (1.0 / T) * delta * 100) # Scale delta?
                    # Standard: exp(delta/T)
                    # Since weights are small (~1/2 to 1/256), delta is small.
                    # Let's use standard Boltzmann
                    import math
                    try:
                        accept_prob = math.exp(delta / T)
                    except OverflowError:
                        accept_prob = 0.0
                    accept = random.random() < accept_prob
            
            if accept:
                current_set = new_set
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_set = set(current_set)
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1
                
            # Cool down
            T *= cooling_rate
            
        return best_set, best_score

    # Local Search to polish the SA result
    def local_search_polish(S):
        S_set = set(S)
        current_score = score_of_set(S_set)
        candidates = [i for i in range(n) if i not in S_set]
        
        improved = True
        while improved:
            if time.time() > deadline - 5:
                break
            improved = False
            
            # Try add
            best_add_v = -1
            best_add_w = -1.0
            for v in candidates:
                if weights[v] > best_add_w:
                    if is_compatible(v, S_set):
                        best_add_w = weights[v]
                        best_add_v = v
            if best_add_v != -1:
                S_set.add(best_add_v)
                candidates.remove(best_add_v)
                current_score += best_add_w
                improved = True
                continue
                
            # Try swap
            best_swap_u = -1
            best_swap_v = -1
            best_gain = 1e-9
            for u in list(S_set):
                temp_set = S_set - {u}
                for v in candidates:
                    if weights[v] <= weights[u]:
                        continue
                    if is_compatible(v, temp_set):
                        gain = weights[v] - weights[u]
                        if gain > best_gain:
                            best_gain = gain
                            best_swap_u = u
                            best_swap_v = v
            if best_swap_u != -1 and best_swap_v != -1:
                S_set.remove(best_swap_u)
                S_set.add(best_swap_v)
                candidates.remove(best_swap_v)
                candidates.append(best_swap_u)
                current_score += best_gain
                improved = True
                continue
                
        return list(S_set), current_score

    # Main Loop: Multi-Start SA
    best_global_score = 0.0
    best_global_set = []
    
    # Initial greedy run to set a baseline
    init_set = random_greedy_init()
    init_set_list, init_score = local_search_polish(init_set)
    best_global_score = init_score
    best_global_set = init_set_list
    
    # Run multiple SA restarts
    num_restarts = 1000 # High number, but each is fast
    restart_time_budget = 0.5 # 0.5s per restart approx
    
    for i in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        # Initialize with a random greedy set
        init = random_greedy_init()
        
        # Run SA with a small time limit
        sa_limit = time.time() + restart_time_budget
        sa_set, sa_score = simulated_annealing(init, sa_limit)
        
        # Polish with local search
        polished_list, polished_score = local_search_polish(sa_set)
        
        if polished_score > best_global_score:
            best_global_score = polished_score
            best_global_set = polished_list

    # If time remains, do a final global local search on the best set found
    if time.time() < deadline - 5:
        final_list, final_score = local_search_polish(best_global_set)
        if final_score > best_global_score:
            best_global_score = final_score
            best_global_set = final_list

    # Return the values
    return [values[i] for i in best_global_set]

if __name__ == "__main__":
    res = construct()
    print(res)
