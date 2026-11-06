import time
import random
from math import lcm

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill

    universe = list(range(1, 23))
    
    # Precompute scaled reciprocals using a common denominator
    BASE = 232792560
    scaled = {i: BASE // i for i in range(1, 23)}

    best_set = []
    best_size = 0

    def update_best(S):
        nonlocal best_set, best_size
        if len(S) > best_size:
            best_set = list(S)
            best_size = len(S)

    def is_dissociated_fast(S):
        """
        Check if all subset sums of reciprocals of S are distinct.
        Uses integer arithmetic with scaling factor BASE.
        Early termination on collision.
        """
        if not S:
            return True
        n = len(S)
        
        sums = [0]
        for x in S:
            val = scaled[x]
            current_len = len(sums)
            for i in range(current_len):
                new_sum = sums[i] + val
                if new_sum in sums:
                    return False
                sums.append(new_sum)
        
        return True

    def is_dissociated(S):
        """
        Check if all subset sums of reciprocals of S are distinct.
        Uses a set for O(1) lookup.
        """
        if not S:
            return True
        n = len(S)
        
        sums = [0]
        for x in S:
            val = scaled[x]
            current_len = len(sums)
            for i in range(current_len):
                sums.append(sums[i] + val)
        
        return len(sums) == len(set(sums))

    def find_collision(S):
        """
        Find a collision in subset sums. Returns two masks that produce the same sum.
        Returns (mask1, mask2) or None.
        """
        n = len(S)
        sums = [0]
        masks = [0]
        
        for i, x in enumerate(S):
            val = scaled[x]
            current_len = len(sums)
            for j in range(current_len):
                new_sum = sums[j] + val
                new_mask = masks[j] | (1 << i)
                sums.append(new_sum)
                masks.append(new_mask)
                
        seen = {}
        for i, s in enumerate(sums):
            if s in seen:
                return (masks[i], seen[s])
            else:
                seen[s] = masks[i]
        return None

    # Strategy 1: Greedy from large to small
    current_S = []
    for x in sorted(universe, reverse=True):
        if time.time() > deadline:
            break
        new_S = current_S + [x]
        if is_dissociated_fast(new_S):
            current_S = new_S
    update_best(current_S)

    # Strategy 2: Randomized Greedy with Restarts
    random.seed(42)
    num_restarts = 100
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        perm = universe.copy()
        random.shuffle(perm)
        
        current_S = []
        for x in perm:
            if time.time() > deadline:
                break
            new_S = current_S + [x]
            if is_dissociated_fast(new_S):
                current_S = new_S
        
        update_best(current_S)
        
        # Local Search: Try to add any remaining elements
        rem = [x for x in universe if x not in current_S]
        rem.sort(reverse=True)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for x in rem:
                if x in current_S:
                    continue
                new_S = current_S + [x]
                if is_dissociated_fast(new_S):
                    current_S = new_S
                    rem.remove(x)
                    update_best(current_S)
                    improved = True
                    break
            
            if not improved:
                # Try swaps: remove one, add one
                rem2 = [x for x in universe if x not in current_S]
                for a in current_S:
                    if time.time() > deadline:
                        break
                    for b in rem2:
                        new_S = [x for x in current_S if x != a] + [b]
                        if is_dissociated_fast(new_S):
                            current_S = new_S
                            update_best(current_S)
                            improved = True
                            break
                    if improved:
                        break

    # Strategy 3: Iterative improvement on the best set found
    current_S = list(best_set)
    rem = [x for x in universe if x not in current_S]
    rem.sort(reverse=True)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in rem:
            if x in current_S:
                continue
            new_S = current_S + [x]
            if is_dissociated_fast(new_S):
                current_S = new_S
                rem.remove(x)
                update_best(current_S)
                improved = True
                break

    # Strategy 4: More aggressive swaps on the best set
    current_S = list(best_set)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        rem = [x for x in universe if x not in current_S]
        for a in current_S:
            if time.time() > deadline:
                break
            for b in rem:
                new_S = [x for x in current_S if x != a] + [b]
                if is_dissociated_fast(new_S):
                    current_S = new_S
                    update_best(current_S)
                    improved = True
                    # After a swap, try to add more
                    rem2 = [x for x in universe if x not in current_S]
                    for c in rem2:
                        if time.time() > deadline:
                            break
                        new_S2 = current_S + [c]
                        if is_dissociated_fast(new_S2):
                            current_S = new_S2
                            update_best(current_S)
                    break
            if improved:
                break

    # Strategy 5: Systematic pair swaps to escape local optima
    current_S = list(best_set)
    rem = [x for x in universe if x not in current_S]
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try removing two and adding two
        n = len(current_S)
        for i in range(n):
            if time.time() > deadline:
                break
            for j in range(i + 1, n):
                if time.time() > deadline:
                    break
                base_S = [x for k, x in enumerate(current_S) if k != i and k != j]
                for a in rem:
                    if time.time() > deadline:
                        break
                    for b in rem:
                        if a == b:
                            continue
                        new_S = base_S + [a, b]
                        if is_dissociated_fast(new_S):
                            current_S = new_S
                            update_best(current_S)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break

    # Strategy 6: Try to build up from scratch with different orderings
    for trial in range(20):
        if time.time() > deadline:
            break
        
        # Use a mix of sorted and random orders
        if trial % 3 == 0:
            order = sorted(universe)
        elif trial % 3 == 1:
            order = sorted(universe, reverse=True)
        else:
            order = universe.copy()
            random.shuffle(order)
        
        current_S = []
        for x in order:
            if time.time() > deadline:
                break
            new_S = current_S + [x]
            if is_dissociated_fast(new_S):
                current_S = new_S
        
        update_best(current_S)
        
        # Local search
        rem = [x for x in universe if x not in current_S]
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for x in rem:
                if x in current_S:
                    continue
                new_S = current_S + [x]
                if is_dissociated_fast(new_S):
                    current_S = new_S
                    rem.remove(x)
                    update_best(current_S)
                    improved = True
                    break
            if not improved:
                rem2 = [x for x in universe if x not in current_S]
                for a in current_S:
                    if time.time() > deadline:
                        break
                    for b in rem2:
                        new_S = [x for x in current_S if x != a] + [b]
                        if is_dissociated_fast(new_S):
                            current_S = new_S
                            update_best(current_S)
                            improved = True
                            break
                    if improved:
                        break

    # Strategy 7: Simulated annealing-style approach
    current_S = list(best_set)
    temperature = 10
    for step in range(500):
        if time.time() > deadline:
            break
        
        if not current_S:
            break
        
        rem = [x for x in universe if x not in current_S]
        
        # Try random move: add, remove, or swap
        move_type = random.choice(['add', 'remove', 'swap'])
        
        if move_type == 'add' and rem:
            x = random.choice(rem)
            new_S = current_S + [x]
            if is_dissociated_fast(new_S):
                current_S = new_S
                update_best(current_S)
        
        elif move_type == 'remove' and len(current_S) > 1:
            x = random.choice(current_S)
            new_S = [y for y in current_S if y != x]
            if is_dissociated_fast(new_S):
                # Accept with probability based on temperature
                if random.random() < 0.3:  # Bias toward keeping larger sets
                    current_S = new_S
        
        elif move_type == 'swap' and rem:
            a = random.choice(current_S)
            b = random.choice(rem)
            new_S = [x for x in current_S if x != a] + [b]
            if is_dissociated_fast(new_S):
                current_S = new_S
                update_best(current_S)
        
        temperature *= 0.99

    if not best_set:
        best_set = [22]

    return best_set

if __name__ == "__main__":
    print(construct())
