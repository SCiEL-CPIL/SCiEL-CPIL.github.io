<<<<<<< SEARCH
    def calculate_score(pts):
        n = len(pts)
        if n < 3:
            return n
        
        # Precompute conflicts
        # conflicts[i] = list of (j, k) such that i, j, k are collinear
        conflicts = [[] for _ in range(n)]
        
        # To speed up, we can just check all triples i<j<k
        # and add to conflicts[i], conflicts[j], conflicts[k]
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                for k in range(j + 1, n):
                    x3, y3 = pts[k]
                    # Check collinearity
                    if (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1):
                        conflicts[i].append((j, k))
                        conflicts[j].append((i, k))
                        conflicts[k].append((i, j))
                        
        # Sort indices by degree (number of conflicts) descending
        # Points with more conflicts are "more constrained", putting them earlier in B&B
        # might help prune branches where they are excluded? 
        # Actually, if we include a high-degree point, it blocks many others.
        # If we exclude it, we might be able to include more others?
        # Standard heuristic for Independent Set: Process high degree first.
        order = sorted(range(n), key=lambda i: len(conflicts[i]), reverse=True)
        
        best_score = 0
        
        def bbb(step, current_mask, current_size):
            nonlocal best_score
            if elapsed() > deadline - 5:
                return
            if current_size > best_score:
                best_score = current_size
            if step == n:
                return
            
            # Pruning: even if we add all remaining points, can we beat best?
            # Note: current_mask does not determine the set of remaining points,
            # but 'step' does if we follow 'order'.
            if current_size + (n - step) <= best_score:
                return
                
            idx = order[step]
            
            # Option 1: Exclude idx
            bbb(step + 1, current_mask, current_size)
            
            # Option 2: Include idx
            # Check if adding idx creates a triple with any two points in current_mask
            valid = True
            for a, b in conflicts[idx]:
                if (current_mask & (1 << a)) and (current_mask & (1 << b)):
                    valid = False
                    break
            
            if valid:
                bbb(step + 1, current_mask | (1 << idx), current_size + 1)
                
        bbb(0, 0, 0)
        return best_score
=======
    def calculate_score(pts):
        n = len(pts)
        if n < 3:
            return n
        
        # Precompute conflicts
        conflicts = [[] for _ in range(n)]
        
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                for k in range(j + 1, n):
                    x3, y3 = pts[k]
                    if (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1):
                        conflicts[i].append((j, k))
                        conflicts[j].append((i, k))
                        conflicts[k].append((i, j))
                        
        best_score = 0
        
        # Greedy Lower Bound to initialize best_score
        def greedy_lower_bound():
            nonlocal best_score
            for _ in range(5):
                selected = set()
                remaining = list(range(n))
                random.shuffle(remaining)
                for idx in remaining:
                    valid = True
                    for a, b in conflicts[idx]:
                        if a in selected and b in selected:
                            valid = False
                            break
                    if valid:
                        selected.add(idx)
                if len(selected) > best_score:
                    best_score = len(selected)

        greedy_lower_bound()

        order = sorted(range(n), key=lambda i: len(conflicts[i]), reverse=True)
        
        def bbb(step, current_mask, current_size):
            nonlocal best_score
            if elapsed() > deadline - 5:
                return
            if current_size > best_score:
                best_score = current_size
            if step == n:
                return
            
            # Pruning: Count compatible remaining points
            max_additional = 0
            for k in range(step, n):
                idx = order[k]
                # Check if idx is compatible with current_mask
                compat = True
                for a, b in conflicts[idx]:
                    if (current_mask & (1 << a)) and (current_mask & (1 << b)):
                        compat = False
                        break
                if compat:
                    max_additional += 1
            
            if current_size + max_additional <= best_score:
                return

            idx = order[step]
            
            # Option 1: Exclude idx
            bbb(step + 1, current_mask, current_size)
            
            # Option 2: Include idx
            valid = True
            for a, b in conflicts[idx]:
                if (current_mask & (1 << a)) and (current_mask & (1 << b)):
                    valid = False
                    break
            
            if valid:
                bbb(step + 1, current_mask | (1 << idx), current_size + 1)
                
        bbb(0, 0, 0)
        return best_score
>>>>>>> REPLACE

I will also increase the `max_iterations` in `local_search` to allow more steps if the score calculation is faster.

