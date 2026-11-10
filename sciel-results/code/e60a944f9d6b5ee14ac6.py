import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin for 150s limit
    N = 1000
    
    def greedy_init(order=None):
        """Greedy construction with a given order of elements."""
        if order is None:
            order = list(range(1, N + 1))
        
        S = set()
        counts = {}
        for x in order:
            valid = True
            for y in S:
                s = x + y
                if counts.get(s, 0) >= 2:
                    valid = False
                    break
            if valid and counts.get(2*x, 0) >= 2:
                valid = False
            if valid:
                S.add(x)
                for y in S:
                    if y == x:
                        continue
                    s = x + y
                    counts[s] = counts.get(s, 0) + 1
                counts[2*x] = counts.get(2*x, 0) + 1
        return S, counts
    
    def try_insert(S, counts, x):
        """Try to insert x into S, return True if successful."""
        if x in S:
            return True
        valid = True
        for y in S:
            s = x + y
            if counts.get(s, 0) >= 2:
                valid = False
                break
        if valid and counts.get(2*x, 0) >= 2:
            valid = False
        
        if valid:
            S.add(x)
            for y in list(S):
                if y == x:
                    continue
                s = x + y
                counts[s] = counts.get(s, 0) + 1
            counts[2*x] = counts.get(2*x, 0) + 1
            return True
        return False
    
    def remove_from(S, counts, x):
        """Remove x from S and update counts."""
        if x not in S:
            return
        S.remove(x)
        for y in list(S):
            s = x + y
            counts[s] = counts.get(s, 0) - 1
            if counts[s] <= 0:
                del counts[s]
        counts[2*x] = counts.get(2*x, 0) - 1
        if counts[2*x] <= 0:
            del counts[2*x]
    
    def local_search(S, counts, max_iters=50000):
        """Local search to improve the set."""
        best_S = frozenset(S)
        best_size = len(S)
        current_S = set(S)
        current_counts = dict(counts)
        
        # Precompute list of all elements
        all_elements = list(range(1, N + 1))
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Try insertion first (higher probability)
            if random.random() < 0.4:
                # Try to add an element
                candidates = [x for x in all_elements if x not in current_S]
                if not candidates:
                    continue
                x = random.choice(candidates)
                if try_insert(current_S, current_counts, x):
                    if len(current_S) > best_size:
                        best_S = frozenset(current_S)
                        best_size = len(current_S)
            
            else:
                # Try a swap: remove one, add another
                if not current_S:
                    continue
                x = random.choice(list(current_S))
                remove_from(current_S, current_counts, x)
                
                # Try to add a different element
                candidates = [x2 for x2 in all_elements if x2 not in current_S]
                if candidates:
                    # Try multiple candidates to find a valid one
                    random.shuffle(candidates)
                    found = False
                    for x2 in candidates[:10]:
                        if try_insert(current_S, current_counts, x2):
                            found = True
                            if len(current_S) > best_size:
                                best_S = frozenset(current_S)
                                best_size = len(current_S)
                            break
                    
                    if not found:
                        # Put x back
                        try_insert(current_S, current_counts, x)
        
        return best_S, best_size
    
    # Multiple restarts with different orderings
    random.seed(42)
    
    best_S_overall = frozenset()
    best_size_overall = 0
    
    # Different greedy orderings to explore different basins
    orderings = [
        list(range(1, N + 1)),           # ascending
        list(range(N, 0, -1)),           # descending
        sorted(range(1, N + 1), key=lambda x: random.random()),  # random
        list(range(1, N + 1, 2)) + list(range(2, N + 1, 2)),  # odds then evens
        list(range(2, N + 1, 2)) + list(range(1, N + 1, 2)),  # evens then odds
    ]
    
    for ordering in orderings:
        if time.time() > deadline:
            break
        
        S, counts = greedy_init(ordering)
        
        # Local search
        best_S, best_size = local_search(S, counts, max_iters=30000)
        
        if best_size > best_size_overall:
            best_size_overall = best_size
            best_S_overall = best_S
    
    # Final polish: try to insert any remaining elements
    if time.time() < deadline - 5:
        final_S = set(best_S_overall)
        final_counts = {}
        for x in final_S:
            for y in final_S:
                if y < x:
                    continue
                s = x + y
                final_counts[s] = final_counts.get(s, 0) + 1
        
        all_elements = list(range(1, N + 1))
        for x in all_elements:
            if x in final_S:
                continue
            if try_insert(final_S, final_counts, x):
                if len(final_S) > best_size_overall:
                    best_size_overall = len(final_S)
                    best_S_overall = frozenset(final_S)
    
    return list(best_S_overall)
