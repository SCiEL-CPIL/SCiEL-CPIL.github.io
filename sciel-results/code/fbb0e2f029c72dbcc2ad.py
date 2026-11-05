"""
Optimized construction for max 3-AP, no 4-AP subset of [1,256].
Uses multi-strategy initialization + aggressive local search with simulated annealing.
"""
import time
import random
import sys

def construct():
    N = 256
    deadline = time.time() + 85.0  # Safety margin before 90s kill
    
    # Precompute 4-AP conflicts: for each x, list of (y, z, w) such that {x,y,z,w} forms a 4-AP
    conflicts = [[] for _ in range(N)]
    for a in range(N):
        max_d = (N - 1 - a) // 3
        for d in range(1, max_d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            # x=a conflicts with {b,c,e}
            conflicts[a].append((b, c, e))
            # x=b conflicts with {a,c,e}
            conflicts[b].append((a, c, e))
            # x=c conflicts with {a,b,e}
            conflicts[c].append((a, b, e))
            # x=e conflicts with {a,b,c}
            conflicts[e].append((a, b, c))
    
    # Precompute 3-AP contribution info for delta calculations
    # For each x, precompute lists of (y, z) pairs where x is part of a 3-AP
    # x as first term: (x, x+d, x+2d) -> need both x+d and x+2d
    # x as middle term: (x-d, x, x+d) -> need both x-d and x+d
    # x as last term: (x-2d, x-d, x) -> need both x-2d and x-d
    
    def is_valid_add(mask, x):
        """Check if adding x to mask creates a 4-AP."""
        for (y, z, w) in conflicts[x]:
            if (mask >> y) & 1 and (mask >> z) & 1 and (mask >> w) & 1:
                return False
        return True
    
    def count_3aps(mask):
        """Count 3-term APs in the set represented by mask."""
        count = 0
        for i in range(N):
            if not ((mask >> i) & 1):
                continue
            max_d = (N - 1 - i) // 2
            for d in range(1, max_d + 1):
                j = i + d
                k = i + 2 * d
                if ((mask >> j) & 1) and ((mask >> k) & 1):
                    count += 1
        return count
    
    def delta_3aps_add(mask, x):
        """Calculate increase in 3-AP count if x is added to mask."""
        delta = 0
        # Case 1: x is the start: x, x+d, x+2d
        max_d = (N - 1 - x) // 2
        for d in range(1, max_d + 1):
            if ((mask >> (x + d)) & 1) and ((mask >> (x + 2 * d)) & 1):
                delta += 1
        # Case 2: x is the middle: x-d, x, x+d
        max_d = min(x, N - 1 - x)
        for d in range(1, max_d + 1):
            if ((mask >> (x - d)) & 1) and ((mask >> (x + d)) & 1):
                delta += 1
        # Case 3: x is the end: x-2d, x-d, x
        max_d = x // 2
        for d in range(1, max_d + 1):
            if ((mask >> (x - 2 * d)) & 1) and ((mask >> (x - d)) & 1):
                delta += 1
        return delta
    
    def delta_3aps_remove(mask, x):
        """Calculate decrease in 3-AP count if x is removed from mask."""
        # Same as delta_3aps_add since we're counting APs that x participates in
        return delta_3aps_add(mask, x)
    
    # Strategy 1: Forward greedy with scoring
    def greedy_forward():
        mask = 0
        elems = list(range(N))
        random.shuffle(elems)
        for x in elems:
            if is_valid_add(mask, x):
                score = delta_3aps_add(mask, x)
                if score > 0 or random.random() < 0.4:
                    mask |= (1 << x)
        return mask
    
    # Strategy 2: Reverse greedy (start with all, remove 4-AP violators)
    def greedy_reverse():
        mask = (1 << N) - 1  # All elements
        # Repeatedly find a 4-AP and remove one element from it
        removed = set()
        while True:
            found = False
            # Find a 4-AP in current set
            for a in range(N):
                if (mask >> a) & 1 == 0:
                    continue
                max_d = (N - 1 - a) // 3
                for d in range(1, max_d + 1):
                    b = a + d
                    c = a + 2 * d
                    e = a + 3 * d
                    if ((mask >> b) & 1) and ((mask >> c) & 1) and ((mask >> e) & 1):
                        # Found 4-AP (a, b, c, e), remove one randomly
                        candidates = [a, b, c, e]
                        # Prefer removing the one with least 3-AP contribution
                        best_to_remove = min(candidates, key=lambda x: delta_3aps_remove(mask, x))
                        mask &= ~(1 << best_to_remove)
                        found = True
                        break
                if found:
                    break
            if not found:
                break
        return mask
    
    # Strategy 3: Random dense subset then cleanup
    def random_dense_init():
        mask = 0
        # Start with random subset of density ~0.5
        for i in range(N):
            if random.random() < 0.5:
                mask |= (1 << i)
        # Remove 4-AP violators
        changed = True
        while changed:
            changed = False
            for a in range(N):
                if (mask >> a) & 1 == 0:
                    continue
                max_d = (N - 1 - a) // 3
                for d in range(1, max_d + 1):
                    b = a + d
                    c = a + 2 * d
                    e = a + 3 * d
                    if ((mask >> b) & 1) and ((mask >> c) & 1) and ((mask >> e) & 1):
                        # Remove one from the 4-AP
                        x = random.choice([a, b, c, e])
                        mask &= ~(1 << x)
                        changed = True
                        break
                if changed:
                    break
        return mask
    
    # Local search: add/remove/swap moves
    def local_search(mask, score, T=2.0, max_steps=5000):
        """Aggressive local search with simulated annealing."""
        current_mask = mask
        current_score = score
        best_mask = mask
        best_score = score
        T_min = 0.01
        cooling = 0.9995
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            move_type = random.random()
            new_mask = current_mask
            new_score = current_score
            valid_move = False
            
            if move_type < 0.4:
                # Add an element
                x = random.randint(0, N - 1)
                if not ((current_mask >> x) & 1):
                    if is_valid_add(current_mask, x):
                        new_mask = current_mask | (1 << x)
                        new_score = current_score + delta_3aps_add(current_mask, x)
                        valid_move = True
            elif move_type < 0.7:
                # Remove an element
                x = random.randint(0, N - 1)
                if (current_mask >> x) & 1:
                    new_mask = current_mask & ~(1 << x)
                    new_score = current_score - delta_3aps_remove(current_mask, x)
                    valid_move = True
            else:
                # Swap
                x = random.randint(0, N - 1)
                if (current_mask >> x) & 1:
                    y = random.randint(0, N - 1)
                    if not ((current_mask >> y) & 1):
                        temp_mask = current_mask & ~(1 << x)
                        if is_valid_add(temp_mask, y):
                            d_rem = delta_3aps_remove(current_mask, x)
                            d_add = delta_3aps_add(temp_mask, y)
                            new_score = current_score - d_rem + d_add
                            new_mask = temp_mask | (1 << y)
                            valid_move = True
            
            if valid_move:
                if new_score >= current_score:
                    current_mask = new_mask
                    current_score = new_score
                    if new_score > best_score:
                        best_score = new_score
                        best_mask = new_mask
                else:
                    # Accept with probability exp((new - current) / T)
                    delta = new_score - current_score
                    if random.random() < pow(2.718281828, delta / T):
                        current_mask = new_mask
                        current_score = new_score
            
            T = max(T_min, T * cooling)
        
        return best_mask, best_score
    
    # Multi-start local search
    def multi_start_local_search(init_strategy, num_starts=3):
        best_mask = 0
        best_score = -1
        for _ in range(num_starts):
            if time.time() > deadline:
                break
            m = init_strategy()
            s = count_3aps(m)
            if s > best_score:
                best_score = s
                best_mask = m
            # Local search
            lm, ls = local_search(m, s, T=3.0, max_steps=3000)
            if ls > best_score:
                best_score = ls
                best_mask = lm
        return best_mask, best_score
    
    # Main: try multiple strategies
    best_mask = 0
    best_score = -1
    
    # Try forward greedy
    m, s = multi_start_local_search(greedy_forward, num_starts=2)
    if s > best_score:
        best_score = s
        best_mask = m
    
    # Try reverse greedy
    m, s = multi_start_local_search(greedy_reverse, num_starts=2)
    if s > best_score:
        best_score = s
        best_mask = m
    
    # Try random dense init
    m, s = multi_start_local_search(random_dense_init, num_starts=2)
    if s > best_score:
        best_score = s
        best_mask = m
    
    # Final polish: more aggressive local search on the best found
    if time.time() < deadline - 10:
        lm, ls = local_search(best_mask, best_score, T=1.0, max_steps=10000)
        if ls > best_score:
            best_score = ls
            best_mask = lm
    
    # Convert best mask to result
    result = [i + 1 for i in range(N) if (best_mask >> i) & 1]
    if not result:
        result = [1]
    return sorted(result)
