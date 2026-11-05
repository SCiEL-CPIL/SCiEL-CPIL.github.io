import random
import time
import heapq

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10-second safety margin
    
    best_set = set()
    
    # Helper function to check if adding x to current_set is valid
    def can_add(x, current_set, counts):
        if counts.get(2*x, 0) >= 2:
            return False
        for y in current_set:
            s = x + y
            if counts.get(s, 0) >= 2:
                return False
        return True

    def update_counts(x, current_set, counts):
        counts[2*x] = counts.get(2*x, 0) + 1
        for y in current_set:
            s = x + y
            counts[s] = counts.get(s, 0) + 1

    def revert_counts(x, current_set, counts):
        counts[2*x] -= 1
        if counts[2*x] == 0:
            del counts[2*x]
        for y in current_set:
            s = x + y
            counts[s] -= 1
            if counts[s] == 0:
                del counts[s]

    # Compute conflict cost for candidate x given current state
    def conflict_cost(x, current_set, counts):
        """
        Return a tuple (num_saturated, num_near_saturated, x) for priority queue.
        num_saturated: count of sums s where counts[s] == 2 (would become 3 if we add x)
        num_near_saturated: count of sums s where counts[s] == 1 (would become 2)
        Lower is better (fewer conflicts).
        """
        if x in current_set:
            return (float('inf'), float('inf'), x)
        
        saturated = 0
        near_saturated = 0
        
        # Check 2x
        c = counts.get(2*x, 0)
        if c >= 2:
            return (float('inf'), float('inf'), x)
        elif c == 1:
            near_saturated += 1
        
        # Check x+y for y in current_set
        for y in current_set:
            s = x + y
            c = counts.get(s, 0)
            if c >= 2:
                return (float('inf'), float('inf'), x)
            elif c == 1:
                near_saturated += 1
        
        return (saturated, near_saturated, x)

    # Local improvement: try to add elements that were skipped, or swap
    def local_improve(current_set, counts, max_attempts=200):
        """
        Try to grow the set by finding elements that can be added,
        or by removing one element and adding two others.
        """
        improved = True
        attempts = 0
        while improved and attempts < max_attempts:
            improved = False
            attempts += 1
            
            # Try adding any element not in the set
            for x in range(1, 1001):
                if x not in current_set and can_add(x, current_set, counts):
                    current_set.add(x)
                    update_counts(x, current_set, counts)
                    improved = True
                    break
            
            # Try swap: remove one, add two
            if not improved:
                found_swap = False
                for remove_x in list(current_set):
                    # Temporarily remove
                    revert_counts(remove_x, current_set, counts)
                    current_set.discard(remove_x)
                    
                    added_elements = []
                    for add_x in range(1, 1001):
                        if add_x == remove_x or add_x in current_set:
                            continue
                        if can_add(add_x, current_set, counts):
                            current_set.add(add_x)
                            update_counts(add_x, current_set, counts)
                            added_elements.append(add_x)
                            if len(added_elements) >= 2:
                                break
                    
                    if len(added_elements) >= 2:
                        found_swap = True
                        break
                    else:
                        # Revert added elements
                        for add_x in added_elements:
                            revert_counts(add_x, current_set, counts)
                            current_set.discard(add_x)
                        # Re-add remove_x
                        current_set.add(remove_x)
                        update_counts(remove_x, current_set, counts)
                
                if found_swap:
                    improved = True
        
        return current_set, counts

    # Priority queue based greedy
    def priority_greedy(random_seed=None):
        if random_seed is not None:
            random.seed(random_seed)
        
        current_set = set()
        counts = {}
        candidates = list(range(1, 1001))
        random.shuffle(candidates)
        
        # Build priority queue with initial costs
        # We'll use a lazy deletion approach: push all candidates, pop min,
        # check if still valid, if not skip, if valid add and update
        
        # Initial: all candidates have cost (0, 0, x) since set is empty
        # But we want randomness, so we shuffle and use that order as tiebreaker
        
        # Use a heap with (saturated, near_saturated, random_key, x)
        # random_key ensures diversity in tie-breaking
        
        heap = []
        for i, x in enumerate(candidates):
            random_key = random.random()
            heap.append((0, 0, random_key, x))
        heapq.heapify(heap)
        
        in_heap = set(candidates)
        
        while heap and time.time() < deadline:
            sat, near, rkey, x = heapq.heappop(heap)
            
            if x not in in_heap:
                continue
            
            if x in current_set:
                continue
            
            # Recompute actual cost
            actual_sat, actual_near, _ = conflict_cost(x, current_set, counts)
            
            if actual_sat == float('inf'):
                in_heap.discard(x)
                continue
            
            # Add x
            current_set.add(x)
            update_counts(x, current_set, counts)
            in_heap.discard(x)
            
            # Update costs for remaining candidates
            # For efficiency, we just re-push with updated costs lazily
            # Actually, let's just recompute on the fly when popping
            
        return current_set, counts

    # Try multiple strategies
    num_restarts = 0
    while time.time() < deadline:
        num_restarts += 1
        
        # Strategy 1: Priority queue greedy
        current_set, counts = priority_greedy(random_seed=num_restarts * 12345)
        
        # Local improvement
        current_set, counts = local_improve(current_set, counts)
        
        if len(current_set) > len(best_set):
            best_set = current_set.copy()
        
        # Check if we have time for another restart
        if time.time() > deadline - 5.0:
            break
    
    return list(best_set)
