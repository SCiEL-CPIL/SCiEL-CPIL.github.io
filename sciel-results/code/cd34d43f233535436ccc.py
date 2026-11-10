import time
import random

def construct():
    N = 1000
    max_sum = 2 * N
    best_set = []
    best_len = 0
    
    def check_and_add(chosen, counts, x):
        """Check if x can be added. Return True if added."""
        for y in chosen:
            if counts[x + y] + 1 > 2:
                return False
        if counts[2 * x] + 1 > 2:
            return False
        for y in chosen:
            counts[x + y] += 1
        counts[2 * x] += 1
        chosen.append(x)
        return True
    
    def remove_from_set(chosen, counts, x):
        """Remove x from chosen set."""
        for y in chosen:
            if y != x:
                counts[x + y] -= 1
        counts[2 * x] -= 1
        chosen.remove(x)
    
    def run_greedy(order='random'):
        if order == 'random':
            candidates = list(range(1, N + 1))
            random.shuffle(candidates)
        elif order == 'ascending':
            candidates = list(range(1, N + 1))
        elif order == 'descending':
            candidates = list(range(N, 0, -1))
        elif order == 'sparse':
            # Start with sparser numbers
            candidates = list(range(1, N + 1, 2))
            random.shuffle(candidates)
            candidates += list(range(2, N + 1, 2))
            random.shuffle(candidates)
        else:
            candidates = list(range(1, N + 1))
            random.shuffle(candidates)
        
        chosen = []
        counts = [0] * (max_sum + 1)
        
        for x in candidates:
            check_and_add(chosen, counts, x)
                
        return chosen, counts
    
    def local_search(chosen, counts):
        """Try to improve by swapping elements."""
        chosen_set = set(chosen)
        improved = True
        max_iters = 50
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            # Try to add any element not in chosen
            for x in range(1, N + 1):
                if x in chosen_set:
                    continue
                # Check if x can be added
                can_add = True
                for y in chosen:
                    if y != x and counts[x + y] + 1 > 2:
                        can_add = False
                        break
                if can_add and counts[2 * x] + 1 > 2:
                    can_add = False
                if can_add:
                    check_and_add(chosen, counts, x)
                    chosen_set.add(x)
                    improved = True
                    break
            if improved:
                continue
            # Try swapping: remove one element, try to add another
            for i, x in enumerate(chosen[:]):
                for y in range(1, N + 1):
                    if y in chosen_set:
                        continue
                    # Temporarily remove x
                    remove_from_set(chosen, counts, x)
                    chosen_set.discard(x)
                    # Try to add y
                    if check_and_add(chosen, counts, y):
                        chosen_set.add(y)
                        improved = True
                        break
                    else:
                        # Put x back
                        check_and_add(chosen, counts, x)
                        chosen_set.add(x)
                if improved:
                    break
        return chosen, counts
    
    start_time = time.time()
    timeout = 115 # Safety margin
    
    orders = ['random', 'random', 'random', 'ascending', 'descending', 'sparse']
    
    while time.time() - start_time < timeout:
        order = random.choice(orders)
        current_set, current_counts = run_greedy(order)
        current_set, current_counts = local_search(current_set, current_counts)
        if len(current_set) > best_len:
            best_len = len(current_set)
            best_set = current_set[:]
            
    return best_set
