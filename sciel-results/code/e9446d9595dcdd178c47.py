import random
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin
    
    def elapsed():
        return time.time() - start_time
    
    def build_greedy(order, counts_init=None, chosen_init=None):
        """Build a B2[2] set by greedily adding elements in given order."""
        if chosen_init is None:
            chosen = []
            counts = {}
        else:
            chosen = list(chosen_init)
            counts = dict(counts_init)
        
        chosen_set = set(chosen)
        for x in order:
            if x in chosen_set:
                continue
            # Check if adding x violates B2[2]
            new_sums = []
            for y in chosen:
                s = x + y
                new_sums.append(s)
            new_sums.append(2 * x)
            
            valid = True
            for s in new_sums:
                if counts.get(s, 0) + 1 > 2:
                    valid = False
                    break
            
            if valid:
                chosen.append(x)
                chosen_set.add(x)
                for s in new_sums:
                    counts[s] = counts.get(s, 0) + 1
        
        return chosen, counts
    
    def get_conflict_count(x, chosen, counts):
        """Count how many sums would exceed 2 if x were added."""
        conflicts = 0
        for y in chosen:
            s = x + y
            if counts.get(s, 0) >= 2:
                conflicts += 1
        s = 2 * x
        if counts.get(s, 0) >= 2:
            conflicts += 1
        return conflicts
    
    def local_search(chosen, counts, max_iter=5000):
        """Try to improve the set via swaps and additions."""
        chosen_set = set(chosen)
        n = len(chosen)
        best_size = n
        
        for iteration in range(max_iter):
            if elapsed() > deadline:
                break
            
            improved = False
            
            # Try adding a random element not in the set
            candidates = [x for x in range(1, 1001) if x not in chosen_set]
            if candidates:
                # Prioritize elements with fewer conflicts
                scored = []
                for x in candidates:
                    c = get_conflict_count(x, chosen, counts)
                    scored.append((c, x))
                scored.sort()
                
                # Try the best few candidates
                for c, x in scored[:5]:
                    if c == 0:
                        # Can add directly
                        new_sums = [x + y for y in chosen] + [2 * x]
                        new_counts = dict(counts)
                        for s in new_sums:
                            new_counts[s] = new_counts.get(s, 0) + 1
                        chosen.append(x)
                        chosen_set.add(x)
                        counts = new_counts
                        best_size = len(chosen)
                        improved = True
                        break
                    elif c == 1:
                        # One conflict: find which sum is at limit and see if we can swap
                        # This is complex, skip for now
                        pass
            
            # Try swaps: remove one element, add another
            if not improved and len(chosen) > 10:
                # Pick a random element to potentially remove
                remove_idx = random.randint(0, len(chosen) - 1)
                removed = chosen[remove_idx]
                
                # Compute counts without this element
                new_counts = dict(counts)
                for y in chosen:
                    if y == removed:
                        continue
                    s = removed + y
                    if s in new_counts:
                        new_counts[s] -= 1
                        if new_counts[s] == 0:
                            del new_counts[s]
                s = 2 * removed
                if s in new_counts:
                    new_counts[s] -= 1
                    if new_counts[s] == 0:
                        del new_counts[s]
                
                # Try to add a different element
                candidates = [x for x in range(1, 1001) if x not in chosen_set or x == removed]
                scored = []
                for x in candidates:
                    if x == removed:
                        continue
                    c = get_conflict_count(x, chosen, new_counts)
                    scored.append((c, x))
                scored.sort()
                
                added = False
                for c, x in scored[:3]:
                    if c == 0:
                        new_chosen = [y for y in chosen if y != removed]
                        new_chosen.append(x)
                        new_chosen_set = set(new_chosen)
                        
                        final_counts = dict(new_counts)
                        for y in new_chosen:
                            if y == x:
                                continue
                            s = x + y
                            final_counts[s] = final_counts.get(s, 0) + 1
                        s = 2 * x
                        final_counts[s] = final_counts.get(s, 0) + 1
                        
                        if len(new_chosen) > len(chosen):
                            chosen = new_chosen
                            chosen_set = new_chosen_set
                            counts = final_counts
                            best_size = len(chosen)
                            improved = True
                            added = True
                            break
            
            if not improved and random.random() < 0.1:
                # Occasionally accept a neutral move to escape local optima
                pass
        
        return chosen, counts
    
    # Phase 1: Multiple randomized greedy starts
    best_chosen = []
    best_counts = {}
    
    num_restarts = 50
    for r in range(num_restarts):
        if elapsed() > deadline:
            break
        
        # Different random orderings
        if r == 0:
            order = list(range(1, 1001))
        elif r == 1:
            order = list(range(1, 1001))
            random.shuffle(order)
        elif r == 2:
            # Prefer smaller numbers (they create fewer large sums)
            order = list(range(1, 1001))
            random.shuffle(order)
            order.sort(key=lambda x: x)
        elif r == 3:
            # Prefer numbers with prime-like properties
            order = list(range(1, 1001))
            random.shuffle(order)
            # Weight by 1/x to prefer smaller
            order.sort(key=lambda x: random.random() * x)
        else:
            order = list(range(1, 1001))
            random.shuffle(order)
        
        chosen, counts = build_greedy(order)
        
        if len(chosen) > len(best_chosen):
            best_chosen = chosen
            best_counts = counts
    
    # Phase 2: Local search from the best solution
    if elapsed() < deadline - 10:
        best_chosen, best_counts = local_search(best_chosen, best_counts, max_iter=20000)
    
    # Phase 3: More local search iterations if time permits
    while elapsed() < deadline - 5:
        chosen_before = len(best_chosen)
        best_chosen, best_counts = local_search(best_chosen, best_counts, max_iter=5000)
        if len(best_chosen) <= chosen_before:
            # No improvement, try a different approach
            # Perturb and restart local search
            if elapsed() > deadline - 10:
                break
            # Randomly remove a few elements and try to refill
            chosen = list(best_chosen)
            counts = dict(best_counts)
            for _ in range(3):
                if not chosen:
                    break
                idx = random.randint(0, len(chosen) - 1)
                removed = chosen.pop(idx)
                # Update counts
                for y in chosen:
                    s = removed + y
                    if s in counts:
                        counts[s] -= 1
                        if counts[s] == 0:
                            del counts[s]
                s = 2 * removed
                if s in counts:
                    counts[s] -= 1
                    if counts[s] == 0:
                        del counts[s]
            
            # Refill greedily
            order = list(range(1, 1001))
            random.shuffle(order)
            best_chosen, best_counts = build_greedy(order, counts, chosen)
            
            if len(best_chosen) > chosen_before:
                continue
            else:
                break
    
    return best_chosen
