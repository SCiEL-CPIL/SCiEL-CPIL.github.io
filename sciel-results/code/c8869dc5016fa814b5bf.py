import time
import random
import math

def construct():
    start_time = time.time()
    limit = 55.0  # Safety margin for 60s limit
    
    n = 256
    # Weights: 1/i
    weights = [0.0] + [1.0 / i for i in range(1, n + 1)]
    
    # Build Conflict Graph
    # conflict[i] is a set of j such that i, j conflict
    conflict = [set() for _ in range(n + 1)]
    
    for u in range(1, n + 1):
        for v in range(u + 1, n + 1):
            # Check 3-AP conditions
            # 1. w = 2u - v
            w = 2 * u - v
            if 1 <= w <= n and w != u and w != v:
                conflict[u].add(v)
                conflict[v].add(u)
                continue
            # 2. w = 2v - u
            w = 2 * v - u
            if 1 <= w <= n and w != u and w != v:
                conflict[u].add(v)
                conflict[v].add(u)
                continue
            # 3. w = (u + v) / 2
            if (u + v) % 2 == 0:
                w = (u + v) // 2
                if 1 <= w <= n and w != u and w != v:
                    conflict[u].add(v)
                    conflict[v].add(u)

    # Helper to check if x can be added to set S (represented as boolean array or set)
    # For speed in SA, we'll use a boolean array `in_set` and a set `S` for iteration.
    
    def is_compatible(x, S_set):
        for y in S_set:
            if y in conflict[x]:
                return False
        return True

    # Initial Solution: Greedy by weight (descending)
    # Sort indices 1..n by weight descending (i.e., index ascending)
    # Actually, 1/1 > 1/2 > ... so we just iterate 1 to n.
    S_init = set()
    for i in range(1, n + 1):
        if is_compatible(i, S_init):
            S_init.add(i)
    
    best_S = set(S_init)
    best_score = sum(weights[i] for i in best_S)
    
    # Simulated Annealing
    # State: S (set of indices)
    # Score: sum(weights[i] for i in S)
    
    current_S = set(best_S)
    current_score = best_score
    
    # Precompute list of all indices
    all_indices = list(range(1, n + 1))
    
    # Initial temperature
    # Max possible score change is roughly 1/1 = 1.0
    T = 0.5
    T_min = 0.001
    cooling_rate = 0.999
    
    # We will run until time limit
    while time.time() - start_time < limit:
        # Generate a random move
        move_type = random.random()
        
        if move_type < 0.4:
            # Try to add a random non-member
            non_members = [i for i in all_indices if i not in current_S]
            if not non_members:
                continue
            v = random.choice(non_members)
            if is_compatible(v, current_S):
                new_S = set(current_S)
                new_S.add(v)
                new_score = current_score + weights[v]
                
                # Accept if better or with probability exp((new_score - current_score)/T)
                # Since new_score > current_score, always accept
                current_S = new_S
                current_score = new_score
                
                if current_score > best_score:
                    best_score = current_score
                    best_S = set(current_S)
            else:
                # If not compatible, we can't add it. 
                # Optionally, we could try to remove conflicts to add it, but that's a different move.
                pass
                
        elif move_type < 0.7:
            # Try to remove a random member
            if not current_S:
                continue
            u = random.choice(list(current_S))
            new_S = set(current_S)
            new_S.remove(u)
            new_score = current_score - weights[u]
            
            # Accept if better (it's not, since we remove weight) or with prob exp((new-old)/T)
            delta = new_score - current_score
            if delta > 0 or random.random() < math.exp(delta / T):
                current_S = new_S
                current_score = new_score
                # Update best if improved (unlikely for removal)
                if current_score > best_score:
                    best_score = current_score
                    best_S = set(current_S)
                
        else:
            # Try to swap a random member u with a random non-member v
            if not current_S:
                continue
            non_members = [i for i in all_indices if i not in current_S]
            if not non_members:
                continue
            
            u = random.choice(list(current_S))
            v = random.choice(non_members)
            
            # Check if v is compatible with all members except u
            # And u is not needed for compatibility (it's being removed)
            compatible = True
            for y in current_S:
                if y == u:
                    continue
                if y in conflict[v]:
                    compatible = False
                    break
            
            if compatible:
                new_S = set(current_S)
                new_S.remove(u)
                new_S.add(v)
                new_score = current_score - weights[u] + weights[v]
                
                delta = new_score - current_score
                if delta > 0 or random.random() < math.exp(delta / T):
                    current_S = new_S
                    current_score = new_score
                    
                    if current_score > best_score:
                        best_score = current_score
                        best_S = set(current_S)
        
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
        # Occasionally reheat if stuck? 
        # Simple strategy: if no improvement for a while, reset T?
        # For now, just cool down.
        
    # Final Polish: Greedy add any possible
    for i in range(1, n + 1):
        if i not in best_S:
            if is_compatible(i, best_S):
                best_S.add(i)
                # Update score if needed, but we just return the set
                # The evaluator calculates the score.
    
    return list(best_S)

if __name__ == "__main__":
    construct()
