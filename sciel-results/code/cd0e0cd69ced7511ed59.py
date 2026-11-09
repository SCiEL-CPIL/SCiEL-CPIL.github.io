import time
import random

def construct():
    start = time.time()
    DEADLINE = start + 85  # Safety margin before 90s limit

    def compute_block_sums(s):
        """Return set of all consecutive block sums."""
        sums = set()
        n = len(s)
        for i in range(n):
            acc = 0
            for j in range(i, n):
                acc += s[j]
                sums.add(acc)
        return sums

    def is_valid(s):
        """Check if sequence is valid: strictly increasing, values in [1,256], distinct block sums."""
        n = len(s)
        if n == 0:
            return True
        for i in range(n - 1):
            if s[i] >= s[i + 1]:
                return False
        for v in s:
            if v < 1 or v > 256:
                return False
        sums = compute_block_sums(s)
        expected = n * (n + 1) // 2
        return len(sums) == expected

    def can_extend(s, x, pos):
        """Check if inserting x at pos keeps sequence valid."""
        # Check increasing order
        if pos > 0 and x <= s[pos - 1]:
            return False
        if pos < len(s) and x >= s[pos]:
            return False
        # Check all new block sums are distinct
        n = len(s)
        # Compute all existing block sums
        existing_sums = set()
        for i in range(n):
            acc = 0
            for j in range(i, n):
                acc += s[j]
                existing_sums.add(acc)
        
        # New block sums involving x
        new_sums = set()
        # Blocks starting at pos
        acc = x
        for j in range(pos, n):
            new_sums.add(acc)
            acc += s[j]
        # Blocks ending at pos (but not starting at pos, already counted)
        acc = x
        for i in range(pos - 1, -1, -1):
            acc += s[i]
            new_sums.add(acc)
        # The single element x
        new_sums.add(x)
        
        # Check no conflicts
        if new_sums & existing_sums:
            return False
        if len(new_sums) != len(new_sums):  # Should always be true, but check for internal conflicts
            return False
        
        return True

    def greedy_build(direction='smallest'):
        """Build a sequence greedily."""
        values = []
        if direction == 'smallest':
            candidates = range(1, 257)
        else:
            candidates = range(256, 0, -1)
        
        for x in candidates:
            if time.time() > DEADLINE - 2:
                break
            # Try appending at end (pos = len(values))
            n = len(values)
            if n == 0 or x > values[-1]:
                # Check if appending is valid
                ok = True
                sums = set()
                for i in range(n):
                    acc = 0
                    for j in range(i, n):
                        acc += values[j]
                        sums.add(acc)
                # New sums when appending x
                new_sums = set()
                acc = x
                new_sums.add(acc)
                for a in reversed(values):
                    acc += a
                    new_sums.add(acc)
                if not (new_sums & sums) and len(new_sums) == len(new_sums):
                    values.append(x)
        
        return sorted(values)

    def backtracking_search(max_depth=12, time_limit=80):
        """Use backtracking with constraint propagation to find long sequences."""
        best = []
        
        def dfs(current, depth, max_d):
            nonlocal best
            if time.time() > time_limit:
                return
            if len(current) > len(best):
                best = list(current)
            if depth >= max_d:
                return
            
            # Try to extend by adding elements
            # Start from last element + 1 up to 256
            last = current[-1] if current else 0
            for x in range(last + 1, 257):
                if time.time() > time_limit:
                    return
                if can_extend(current, x, len(current)):
                    current.append(x)
                    dfs(current, depth + 1, max_d)
                    current.pop()
        
        dfs([], 0, max_depth)
        return best

    def simulated_annealing(start_seq, iterations=10000, start_temp=10.0, end_temp=0.01):
        """Simulated annealing to improve sequence length."""
        current = list(start_seq)
        best = list(current)
        best_len = len(current)
        
        def get_score(s):
            """Score: negative of number of conflicts + bonus for length."""
            if not s:
                return 0
            # Check increasing
            for i in range(len(s) - 1):
                if s[i] >= s[i + 1]:
                    return -1000
            # Check range
            for v in s:
                if v < 1 or v > 256:
                    return -1000
            # Count distinct block sums
            sums = compute_block_sums(s)
            expected = len(s) * (len(s) + 1) // 2
            conflicts = expected - len(sums)
            return len(s) * 10 - conflicts * 100
        
        current_score = get_score(current)
        
        for i in range(iterations):
            if time.time() > DEADLINE - 5:
                break
            temp = start_temp * (end_temp / start_temp) ** (i / iterations)
            
            # Generate neighbor
            neighbor = list(current)
            move_type = random.choice(['insert', 'delete', 'swap'])
            
            if move_type == 'insert' and len(neighbor) < 15:
                x = random.randint(1, 256)
                pos = random.randint(0, len(neighbor))
                if pos == 0 or (pos > 0 and x > neighbor[pos-1] and (pos == len(neighbor) or x < neighbor[pos])):
                    neighbor.insert(pos, x)
            elif move_type == 'delete' and len(neighbor) > 1:
                pos = random.randint(0, len(neighbor) - 1)
                neighbor.pop(pos)
            elif move_type == 'swap' and len(neighbor) >= 2:
                i1, i2 = random.sample(range(len(neighbor)), 2)
                neighbor[i1], neighbor[i2] = neighbor[i2], neighbor[i1]
                # Fix ordering
                neighbor.sort()
                # Check if still increasing (sort ensures this)
            
            neighbor_score = get_score(neighbor)
            
            if neighbor_score >= current_score:
                current = neighbor
                current_score = neighbor_score
                if len(current) > best_len:
                    best = list(current)
                    best_len = len(current)
            else:
                delta = neighbor_score - current_score
                if random.random() < pow(2.718281828, delta / max(temp, 0.001)):
                    current = neighbor
                    current_score = neighbor_score
        
        return best

    # Strategy 1: Backtracking with increasing depth
    best = backtracking_search(max_depth=10, time_limit=30)
    
    # Strategy 2: Simulated annealing from multiple starts
    starts = [best]
    
    # Add greedy starts
    starts.append(greedy_build('smallest'))
    starts.append(greedy_build('largest'))
    
    # Add random starts
    for _ in range(3):
        if time.time() > DEADLINE - 10:
            break
        rand_seq = []
        used = set()
        for _ in range(random.randint(5, 10)):
            x = random.randint(1, 256)
            if x in used:
                continue
            if rand_seq and x <= rand_seq[-1]:
                continue
            if can_extend(rand_seq, x, len(rand_seq)):
                rand_seq.append(x)
                used.add(x)
        starts.append(rand_seq)
    
    for start_seq in starts:
        if time.time() > DEADLINE - 5:
            break
        result = simulated_annealing(start_seq, iterations=5000)
        if len(result) > len(best):
            best = result
    
    # Final verification
    if not is_valid(best):
        n = len(best)
        for i in range(n):
            if is_valid(best[:i + 1]):
                best = best[:i + 1]
            else:
                break
        else:
            best = best[:i]
    
    return best
