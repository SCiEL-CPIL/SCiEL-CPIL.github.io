"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    N = 128
    elements = list(range(1, N + 1))
    
    # Precompute LCM for all pairs
    lcm = [[0]*(N+1) for _ in range(N+1)]
    for i in range(1, N+1):
        for j in range(i, N+1):
            lcm[i][j] = i * j // gcd(i, j)
    
    # Find all bad triples
    bad_triples = set()
    for a in range(1, N+1):
        for b in range(a+1, N+1):
            lab = lcm[a][b]
            for c in range(b+1, N+1):
                if lcm[a][c] == lab and lcm[b][c] == lab:
                    bad_triples.add(frozenset([a, b, c]))
    
    # For each element x, store list of (y, z) pairs that form bad triple with x
    conflicts = {x: [] for x in range(1, N+1)}
    for bt in bad_triples:
        a, b, c = list(bt)
        conflicts[a].append((b, c))
        conflicts[b].append((a, c))
        conflicts[c].append((a, b))
    
    # Precompute for each element x: a list of bitmask pairs (1<<y | 1<<z)
    # where {x, y, z} is a bad triple
    conflict_masks = {x: [] for x in range(1, N+1)}
    for x in range(1, N+1):
        for (y, z) in conflicts[x]:
            mask = (1 << y) | (1 << z)
            conflict_masks[x].append(mask)
    
    # Precompute conflict count for ordering heuristics
    conflict_counts = {x: len(conflict_masks[x]) for x in range(1, N+1)}
    
    def can_add_mask(x, S_mask):
        """Check if x can be added to set represented by S_mask (bitmask)."""
        for cm in conflict_masks[x]:
            if (S_mask & cm) == cm:
                return False
        return True
    
    # Greedy fill with bitmask
    def greedy_fill_mask(S_mask, order):
        for x in order:
            if (S_mask >> x) & 1 == 0 and can_add_mask(x, S_mask):
                S_mask |= (1 << x)
        return S_mask
    
    # Simulated Annealing with bitmask
    def simulated_annealing(S_mask, max_iters=3000, t_start=1.5, t_end=0.01, cooling=0.9997):
        """SA to escape local minima. Returns improved S_mask."""
        current = S_mask
        current_size = bin(current).count('1')
        best = current
        best_size = current_size
        
        t = t_start
        for it in range(max_iters):
            if time.time() > deadline - 3:
                break
                
            # Pick a random move: add, remove, or swap
            move_type = random.random()
            
            if move_type < 0.35:
                # Try to add a random element not in set
                not_in = [x for x in range(1, N+1) if (current >> x) & 1 == 0]
                if not not_in:
                    continue
                x = random.choice(not_in)
                if can_add_mask(x, current):
                    new_mask = current | (1 << x)
                    new_size = bin(new_mask).count('1')
                    delta = new_size - current_size
                    if delta > 0 or random.random() < math.exp(delta / t):
                        current = new_mask
                        current_size = new_size
                        if current_size > best_size:
                            best = current
                            best_size = current_size
            elif move_type < 0.65:
                # Try to remove a random element from set
                in_set = [x for x in range(1, N+1) if (current >> x) & 1 == 1]
                if not in_set:
                    continue
                x = random.choice(in_set)
                new_mask = current & ~(1 << x)
                new_size = bin(new_mask).count('1')
                delta = new_size - current_size
                if delta > 0 or random.random() < math.exp(delta / t):
                    current = new_mask
                    current_size = new_size
                    if current_size > best_size:
                        best = current
                        best_size = current_size
            else:
                # Swap: remove one, try to add another
                in_set = [x for x in range(1, N+1) if (current >> x) & 1 == 1]
                not_in = [x for x in range(1, N+1) if (current >> x) & 1 == 0]
                if not in_set or not not_in:
                    continue
                x = random.choice(in_set)
                y = random.choice(not_in)
                new_mask = (current & ~(1 << x)) | (1 << y)
                # Check validity of the swap
                valid = True
                for cm in conflict_masks[y]:
                    if (new_mask & cm) == cm:
                        valid = False
                        break
                if valid:
                    new_size = bin(new_mask).count('1')
                    delta = new_size - current_size
                    if delta > 0 or random.random() < math.exp(delta / t):
                        current = new_mask
                        current_size = new_size
                        if current_size > best_size:
                            best = current
                            best_size = current_size
            
            t *= cooling
        
        return best
    
    # Local search: 1-removal and re-greedy-fill
    def local_search(S_mask, max_rounds=80):
        """Try 1-removal and 2-removal moves to improve."""
        current = S_mask
        current_size = bin(current).count('1')
        best = current
        best_size = current_size
        
        for round_num in range(max_rounds):
            if time.time() > deadline - 3:
                break
            
            improved = False
            in_set = [x for x in range(1, N+1) if (current >> x) & 1 == 1]
            random.shuffle(in_set)
            
            # Try 1-removal moves
            for u in in_set:
                if time.time() > deadline - 3:
                    break
                
                # Remove u
                temp = current & ~(1 << u)
                
                # Re-fill greedily
                candidates = [x for x in range(1, N+1) if (temp >> x) & 1 == 0]
                random.shuffle(candidates)
                
                for x in candidates:
                    if can_add_mask(x, temp):
                        temp |= (1 << x)
                
                new_size = bin(temp).count('1')
                if new_size > current_size:
                    current = temp
                    current_size = new_size
                    if current_size > best_size:
                        best = current
                        best_size = current_size
                    improved = True
                    break
                elif new_size == current_size and random.random() < 0.15:
                    current = temp
                    current_size = new_size
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try 2-removal moves (2-opt style)
            if len(in_set) >= 2:
                pairs = random.sample(list(zip(in_set, in_set)), min(20, len(in_set)*(len(in_set)-1)//2))
                for u, v in pairs:
                    if u >= v or time.time() > deadline - 3:
                        continue
                    
                    # Remove u and v
                    temp = current & ~(1 << u) & ~(1 << v)
                    
                    # Re-fill greedily
                    candidates = [x for x in range(1, N+1) if (temp >> x) & 1 == 0]
                    random.shuffle(candidates)
                    
                    for x in candidates:
                        if can_add_mask(x, temp):
                            temp |= (1 << x)
                    
                    new_size = bin(temp).count('1')
                    if new_size > current_size:
                        current = temp
                        current_size = new_size
                        if current_size > best_size:
                            best = current
                            best_size = current_size
                        improved = True
                        break
                    elif new_size == current_size and random.random() < 0.1:
                        current = temp
                        current_size = new_size
                        improved = True
                        break
        
        return best
    
    # Final exact polish: try all single additions and 2-swaps
    def final_polish(S_mask):
        """Exhaustive single-add and 2-swap polish."""
        current = S_mask
        current_size = bin(current).count('1')
        
        # Try all single additions
        for x in range(1, N+1):
            if (current >> x) & 1 == 0 and can_add_mask(x, current):
                current |= (1 << x)
                current_size += 1
        
        # Try all 2-swaps (remove one, add one)
        in_set = [x for x in range(1, N+1) if (current >> x) & 1 == 1]
        not_in = [x for x in range(1, N+1) if (current >> x) & 1 == 0]
        
        for u in in_set:
            for y in not_in:
                new_mask = (current & ~(1 << u)) | (1 << y)
                # Check validity
                valid = True
                for cm in conflict_masks[y]:
                    if (new_mask & cm) == cm:
                        valid = False
                        break
                if valid and bin(new_mask).count('1') > current_size:
                    current = new_mask
                    current_size = bin(current).count('1')
                    not_in = [x for x in range(1, N+1) if (current >> x) & 1 == 0]
        
        return current
    
    # Main search loop
    best_mask = 0
    best_size = 0
    
    # Precompute different orderings
    order_by_conflict = sorted(elements, key=lambda x: conflict_counts[x])
    order_by_conflict_desc = sorted(elements, key=lambda x: -conflict_counts[x])
    order_reverse = list(range(N, 0, -1))
    order_forward = list(range(1, N+1))
    
    num_restarts = 500
    
    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        # Random restart with diverse seeding
        S_mask = 0
        
        # Choose ordering strategy based on restart number
        strategy = r % 5
        if strategy == 0:
            order = elements[:]
            random.shuffle(order)
        elif strategy == 1:
            order = order_by_conflict[:]
        elif strategy == 2:
            order = order_by_conflict_desc[:]
        elif strategy == 3:
            order = order_reverse[:]
        else:
            order = order_forward[:]
        
        # Sometimes seed with a small random set
        if r % 4 == 0:
            seed_size = random.randint(1, 3)
            seed_elements = random.sample(elements, seed_size)
            for x in seed_elements:
                if can_add_mask(x, S_mask):
                    S_mask |= (1 << x)
        
        # Greedy fill
        S_mask = greedy_fill_mask(S_mask, order)
        
        # Local search with 1-removal and 2-removal
        S_mask = local_search(S_mask)
        
        # Simulated annealing to escape local minima
        S_mask = simulated_annealing(S_mask)
        
        # Final polish: exhaustive single-add and 2-swap
        S_mask = final_polish(S_mask)
        
        # Update best
        current_size = bin(S_mask).count('1')
        if current_size > best_size:
            best_size = current_size
            best_mask = S_mask
    
    # Convert best_mask back to list
    best_set = [x for x in range(1, N+1) if (best_mask >> x) & 1 == 1]
    
    if not best_set:
        best_set = [1]
    
    return best_set
# EVOLVE-BLOCK-END
