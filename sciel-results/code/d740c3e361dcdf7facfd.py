import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    target = 200
    max_val = 100
    all_elements = list(range(1, max_val + 1))
    mask_limit = (1 << (target + 1)) - 1
    
    def compute_mask(elements):
        mask = 1
        for x in elements:
            if x > target:
                continue
            mask = mask | (mask << x)
            mask &= mask_limit
        return mask

    def is_valid_mask(mask):
        return not ((mask >> target) & 1)

    def find_conflict(elements):
        reachable = [False] * (target + 1)
        parent = [None] * (target + 1)
        reachable[0] = True
        for x in elements:
            if x > target:
                continue
            for s in range(target, x - 1, -1):
                if not reachable[s] and reachable[s - x]:
                    reachable[s] = True
                    parent[s] = (s - x, x)
        if not reachable[target]:
            return None
        subset = []
        s = target
        while s > 0:
            prev_s, x = parent[s]
            subset.append(x)
            s = prev_s
        return subset

    def make_valid(elements):
        """Remove elements until the set is valid."""
        current = list(elements)
        current_mask = compute_mask(current)
        while not is_valid_mask(current_mask):
            if time.time() > deadline - 5.0:
                break
            conflict = find_conflict(current)
            if conflict is None:
                break
            idx = random.randint(0, len(conflict) - 1)
            to_remove = conflict[idx]
            current.remove(to_remove)
            current_mask = compute_mask(current)
        return current, current_mask

    def greedy_add(current, current_mask, order):
        """Add elements from order if they don't create a conflict."""
        for x in order:
            if x in current:
                continue
            new_mask = current_mask | (current_mask << x)
            new_mask &= mask_limit
            if is_valid_mask(new_mask):
                current.append(x)
                current_mask = new_mask
        return current, current_mask

    def try_swaps(current, current_mask, max_iterations=30):
        """Try to improve by swapping elements."""
        current = list(current)
        current_mask = current_mask
        for _ in range(max_iterations):
            if time.time() > deadline - 5.0:
                break
            improved = False
            missing = [x for x in range(1, 101) if x not in current]
            if not missing:
                break
            # Try adding any missing element
            for x in missing:
                new_mask = current_mask | (current_mask << x)
                new_mask &= mask_limit
                if is_valid_mask(new_mask):
                    current.append(x)
                    current_mask = new_mask
                    improved = True
                    break
            if improved:
                continue
            # Try swaps: remove y, add x
            for y in list(current):
                if time.time() > deadline - 5.0:
                    break
                mask_without_y = compute_mask([z for z in current if z != y])
                for x in missing:
                    if x == y:
                        continue
                    new_mask = mask_without_y | (mask_without_y << x)
                    new_mask &= mask_limit
                    if is_valid_mask(new_mask):
                        current.remove(y)
                        current.append(x)
                        current_mask = new_mask
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        return current, current_mask

    best_set = list(range(66, 101))
    best_size = len(best_set)
    
    # Multiple starting strategies
    starting_configs = [
        list(range(66, 101)),
        list(range(67, 101)),
        list(range(60, 101)),
        list(range(55, 101)),
        list(range(50, 101)),
    ]
    
    addition_orders = [
        list(range(1, 66)),
        list(range(65, 0, -1)),
        sorted(random.sample(range(1, 66), 65)),
        sorted(random.sample(range(1, 66), 65), reverse=True),
    ]
    
    for start_set in starting_configs:
        if time.time() > deadline - 10.0:
            break
        
        current, current_mask = make_valid(start_set)
        
        if len(current) > best_size:
            best_size = len(current)
            best_set = current[:]
        
        for order in addition_orders:
            if time.time() > deadline - 10.0:
                break
            
            added, added_mask = greedy_add(current[:], current_mask, order)
            
            if len(added) > best_size:
                best_size = len(added)
                best_set = added[:]
            
            # Try swaps to improve further
            swapped, final_mask = try_swaps(added, added_mask, max_iterations=20)
            
            if len(swapped) > best_size:
                best_size = len(swapped)
                best_set = swapped[:]
    
    # Final check
    final_mask = compute_mask(best_set)
    if not is_valid_mask(final_mask):
        best_set = list(range(66, 101))
    
    return best_set

if __name__ == "__main__":
    res = construct()
    print(f"Best size: {len(res)}")
    print(f"Set: {sorted(res)}")
    mask = 1
    for x in res:
        mask = mask | (mask << x)
        mask &= (1 << 201) - 1
    print(f"Valid (no sum 200): {not ((mask >> 200) & 1)}")
