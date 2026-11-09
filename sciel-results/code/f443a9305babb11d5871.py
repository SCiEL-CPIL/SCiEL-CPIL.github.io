# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from itertools import combinations

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    # Compute forbidden differences
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)
    
    # Symmetrize: if d is forbidden, -d is also effectively forbidden
    forbidden_sym = set()
    for d in forbidden:
        forbidden_sym.add(d)
        forbidden_sym.add((-d) % m)
    forbidden_sym.discard(0)
    
    # Precompute for each element x, the set of elements it conflicts with
    # conflict[x] = set of y such that (x-y)%m is in forbidden_sym
    conflict = [set() for _ in range(m)]
    for x in range(m):
        for y in range(m):
            if x == y:
                continue
            d = (x - y) % m
            if d in forbidden_sym:
                conflict[x].add(y)
    
    def is_valid_addition(S_set, x):
        """Check if x can be added to set S_set using precomputed conflicts"""
        for s in S_set:
            if s in conflict[x]:
                return False
        return True
    
    def greedy_construct(order):
        """Construct a set greedily based on the given order of candidates"""
        S = []
        S_set = set()
        for x in order:
            if is_valid_addition(S_set, x):
                S.append(x)
                S_set.add(x)
        return S, S_set
    
    def local_search(S, S_set):
        """
        Local search with Add and Swap (remove one, add one) moves.
        """
        improved = True
        while improved:
            improved = False
            # Phase 1: Try to add any possible element
            for x in range(m):
                if x not in S_set:
                    if is_valid_addition(S_set, x):
                        S.append(x)
                        S_set.add(x)
                        improved = True
            
            # Phase 2: If no additions, try swap moves (remove u, add x)
            if not improved:
                for u in list(S):
                    for x in range(m):
                        if x in S_set:
                            continue
                        # Try removing u and adding x
                        S_set_copy = S_set - {u}
                        if is_valid_addition(S_set_copy, x):
                            # Check if the new set is valid (it should be by construction)
                            # Accept if it leads to a larger set (it won't, since we're swapping)
                            # But maybe it allows further additions? Let's do a quick refill.
                            S_copy = [s for s in S if s != u]
                            S_copy.append(x)
                            S_set_copy.add(x)
                            
                            # Try to add more
                            added = True
                            while added:
                                added = False
                                for y in range(m):
                                    if y not in S_set_copy:
                                        if is_valid_addition(S_set_copy, y):
                                            S_copy.append(y)
                                            S_set_copy.add(y)
                                            added = True
                            
                            if len(S_copy) > len(S):
                                S = S_copy
                                S_set = S_set_copy
                                improved = True
                                
        return S, S_set
    
    def simulated_annealing(S, S_set, temp=1.0, steps=1000):
        """
        Simulated annealing to escape local optima.
        Moves: Add, Remove, Swap.
        """
        current_S = list(S)
        current_set = set(S_set)
        best_S = list(current_S)
        best_set = set(current_set)
        best_len = len(best_S)
        
        for i in range(steps):
            t = temp * (0.99 ** (i // 10))  # Cooling schedule
            if t < 0.01:
                break
            
            # Choose a random move
            move_type = random.choice(['add', 'remove', 'swap'])
            
            new_S = list(current_S)
            new_set = set(current_set)
            
            if move_type == 'add':
                candidates = [x for x in range(m) if x not in new_set]
                if not candidates:
                    continue
                x = random.choice(candidates)
                if is_valid_addition(new_set, x):
                    new_S.append(x)
                    new_set.add(x)
                else:
                    # Invalid move, skip
                    continue
            
            elif move_type == 'remove':
                if not new_S:
                    continue
                u = random.choice(new_S)
                new_S.remove(u)
                new_set.remove(u)
            
            elif move_type == 'swap':
                if not new_S:
                    continue
                u = random.choice(new_S)
                candidates = [x for x in range(m) if x not in new_set]
                if not candidates:
                    continue
                x = random.choice(candidates)
                
                # Try the swap
                temp_set = new_set - {u}
                if is_valid_addition(temp_set, x):
                    new_S.remove(u)
                    new_set.remove(u)
                    new_S.append(x)
                    new_set.add(x)
                else:
                    continue
            
            # Evaluate
            delta = len(new_S) - len(current_S)
            
            if delta >= 0 or random.random() < np.exp(delta / t):
                current_S = new_S
                current_set = new_set
                
                if len(current_S) > best_len:
                    best_S = list(current_S)
                    best_set = set(current_set)
                    best_len = len(best_S)
        
        return best_S, best_set
    
    best_set = []
    best_set_set = set()
    
    # Deadline guard: 630s limit, keep safety margin
    deadline = time.time() + 610.0
    
    # Strategy 1: Randomized Greedy + Local Search
    for i in range(1000):
        if time.time() > deadline:
            break
        
        order = list(range(m))
        random.shuffle(order)
        
        S, S_set = greedy_construct(order)
        S, S_set = local_search(S, S_set)
        
        if len(S) > len(best_set):
            best_set = S
            best_set_set = S_set
    
    # Strategy 2: Simulated Annealing from best found so far and random starts
    for i in range(100):
        if time.time() > deadline:
            break
        
        # Start from best set found so far
        if best_set:
            S, S_set = simulated_annealing(best_set, best_set_set, temp=1.0, steps=500)
            if len(S) > len(best_set):
                best_set = S
                best_set_set = S_set
        else:
            # Random start
            order = list(range(m))
            random.shuffle(order)
            S, S_set = greedy_construct(order)
            S, S_set = simulated_annealing(S, S_set, temp=1.0, steps=500)
            if len(S) > len(best_set):
                best_set = S
                best_set_set = S_set
        
        # Also try random starts
        order = list(range(m))
        random.shuffle(order)
        S, S_set = greedy_construct(order)
        S, S_set = simulated_annealing(S, S_set, temp=1.5, steps=1000)
        if len(S) > len(best_set):
            best_set = S
            best_set_set = S_set
    
    # Fallback: if no set found (should not happen), return [0]
    if not best_set:
        best_set = [0]
        
    return np.array(best_set, dtype=np.int64)

# EVOLVE-BLOCK-END
