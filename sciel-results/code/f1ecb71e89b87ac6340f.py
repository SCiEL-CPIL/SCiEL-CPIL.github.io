import time
import random
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 52  # safety margin before 60s limit (evaluator kills at 90s but task says 60s)

    def is_valid(seq):
        if not seq:
            return True
        L = len(seq)
        P = [0] * (L + 1)
        for i in range(L):
            P[i + 1] = P[i] + seq[i]
        sums = set()
        for i in range(L):
            for j in range(i + 1, L + 1):
                s = P[j] - P[i]
                if s in sums:
                    return False
                sums.add(s)
        return True

    def get_all_sums(seq):
        L = len(seq)
        P = [0] * (L + 1)
        for i in range(L):
            P[i + 1] = P[i] + seq[i]
        sums = set()
        for i in range(L):
            for j in range(i + 1, L + 1):
                s = P[j] - P[i]
                sums.add(s)
        return sums

    def can_add(seq, x):
        """Check if x can be added to sorted seq without creating duplicate consecutive block sums.
        Returns (True, new_sums) or (False, None)."""
        if not seq:
            return (True, {x})
        
        k = bisect.bisect_left(seq, x)
        L = len(seq)
        
        new_sums = {x}
        
        # Left extensions: blocks ending at x
        left_sum = 0
        for i in range(k - 1, -1, -1):
            left_sum += seq[i]
            s = left_sum + x
            if s in new_sums:
                return (False, None)
            new_sums.add(s)
        
        # Right extensions: blocks starting at x
        right_sum = 0
        for j in range(k, L):
            right_sum += seq[j]
            s = x + right_sum
            if s in new_sums:
                return (False, None)
            new_sums.add(s)
        
        # Both sides: blocks containing x with elements on both sides
        left_sum = 0
        for i in range(k - 1, -1, -1):
            left_sum += seq[i]
            right_sum = 0
            for j in range(k, L):
                right_sum += seq[j]
                s = left_sum + x + right_sum
                if s in new_sums:
                    return (False, None)
                new_sums.add(s)
        
        return (True, new_sums)

    # Greedy initialization - multiple strategies
    def greedy_forward():
        """Try adding values from 1 to 256 in order."""
        values = []
        sums = set()
        for x in range(1, 257):
            valid, new_sums = can_add(values, x)
            if valid:
                k = bisect.bisect_left(values, x)
                values = values[:k] + [x] + values[k:]
                sums.update(new_sums)
        return values

    def greedy_reverse():
        """Try adding values from 256 to 1 in order."""
        values = []
        sums = set()
        for x in range(256, 0, -1):
            valid, new_sums = can_add(values, x)
            if valid:
                k = bisect.bisect_left(values, x)
                values = values[:k] + [x] + values[k:]
                sums.update(new_sums)
        return values

    def greedy_random(seed):
        """Try adding values in random order."""
        rng = random.Random(seed)
        values = []
        sums = set()
        candidates = list(range(1, 257))
        rng.shuffle(candidates)
        for x in candidates:
            valid, new_sums = can_add(values, x)
            if valid:
                k = bisect.bisect_left(values, x)
                values = values[:k] + [x] + values[k:]
                sums.update(new_sums)
        return values

    def greedy_gapped(step):
        """Try adding values with gaps to reduce conflicts."""
        values = []
        sums = set()
        for x in range(1, 257, step):
            valid, new_sums = can_add(values, x)
            if valid:
                k = bisect.bisect_left(values, x)
                values = values[:k] + [x] + values[k:]
                sums.update(new_sums)
        return values

    # Generate initial candidates from multiple greedy strategies
    candidates = []
    candidates.append(greedy_forward())
    candidates.append(greedy_reverse())
    
    # Several random-order greedy restarts
    for s in range(20):
        candidates.append(greedy_random(s))
    
    # Gapped strategies
    for step in [2, 3, 4, 5]:
        candidates.append(greedy_gapped(step))
    
    # Pick the best initial solution
    best_seq = max(candidates, key=len)
    best_len = len(best_seq)
    best_final = list(best_seq)
    
    # Local search with multiple strategies
    current_seq = list(best_final)
    current_set = set(current_seq)
    all_values = set(range(1, 257))
    
    # Precompute for faster validity checking
    def fast_is_valid(seq):
        L = len(seq)
        if L < 2:
            return True
        P = [0] * (L + 1)
        for i in range(L):
            P[i + 1] = P[i] + seq[i]
        seen = set()
        for i in range(L):
            for j in range(i + 1, L + 1):
                s = P[j] - P[i]
                if s in seen:
                    return False
                seen.add(s)
        return True

    # Simulated annealing style local search
    def local_search(seq, time_budget):
        current = list(seq)
        current_set = set(current)
        best = list(current)
        best_len = len(current)
        end_time = time.time() + time_budget
        
        while time.time() < end_time:
            r = random.random()
            
            if r < 0.6:  # Add
                available = [x for x in all_values if x not in current_set]
                if available:
                    x = random.choice(available)
                    new_seq = current + [x]
                    new_seq.sort()
                    if fast_is_valid(new_seq):
                        current = new_seq
                        current_set = set(current)
                        if len(current) > best_len:
                            best_len = len(current)
                            best = list(current)
            
            elif r < 0.85:  # Swap
                if current:
                    available = [x for x in all_values if x not in current_set]
                    if available:
                        x_rem = random.choice(current)
                        x_add = random.choice(available)
                        new_seq = [x for x in current if x != x_rem]
                        new_seq.append(x_add)
                        new_seq.sort()
                        if fast_is_valid(new_seq):
                            current = new_seq
                            current_set = set(current)
                            if len(current) > best_len:
                                best_len = len(current)
                                best = list(current)
            
            elif r < 0.95:  # Remove (to escape local minima)
                if len(current) > 2:
                    x_rem = random.choice(current)
                    current.remove(x_rem)
                    current_set = set(current)
            
            else:  # Double swap (remove two, add two different ones)
                if len(current) >= 2:
                    available = [x for x in all_values if x not in current_set]
                    if len(available) >= 2:
                        x_rem1, x_rem2 = random.sample(current, 2)
                        x_add1, x_add2 = random.sample(available, 2)
                        new_seq = [x for x in current if x not in (x_rem1, x_rem2)]
                        new_seq.extend([x_add1, x_add2])
                        new_seq.sort()
                        if fast_is_valid(new_seq):
                            current = new_seq
                            current_set = set(current)
                            if len(current) > best_len:
                                best_len = len(current)
                                best = list(current)
        
        return best

    # Run local search from the best initial solution
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        improved = local_search(best_final, remaining_time - 2)
        if len(improved) > best_len:
            best_final = improved
            best_len = len(best_final)
    
    # Try local search from other good initial solutions if time permits
    remaining_time = deadline - time.time()
    if remaining_time > 10:
        # Sort candidates by length, try local search on next best
        sorted_candidates = sorted(candidates, key=len, reverse=True)
        for cand in sorted_candidates[1:5]:
            if time.time() > deadline - 3:
                break
            if len(cand) >= best_len - 1:  # Only try if close to best
                t_budget = min(remaining_time / 4, 8)
                if t_budget > 1:
                    improved = local_search(cand, t_budget)
                    if len(improved) > best_len:
                        best_final = improved
                        best_len = len(improved)
                    remaining_time = deadline - time.time()
                    if remaining_time < 5:
                        break
    
    return sorted(best_final)
