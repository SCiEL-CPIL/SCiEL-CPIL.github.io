import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill

    # Maximum possible consecutive block sum: 256 * (256+1)/2 = 32896
    # But for a sequence of length n with values in [1,256], max sum is n*256
    # We'll use a bitset for fast sum tracking
    MAX_SUM = 256 * 256  # generous upper bound

    def all_block_sums(seq):
        """Return set of all consecutive block sums."""
        sums = set()
        n = len(seq)
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += seq[j]
                sums.add(s)
        return sums

    def can_insert(seq, x, pos, sums_set):
        """Check if inserting x at pos in seq keeps all block sums distinct.
        Returns (new_sums, valid) where new_sums is the updated sum set."""
        n = len(seq)
        # New blocks involving x:
        # 1. Blocks ending at x: sum of seq[i..pos-1] + x for i in 0..pos-1, and x alone
        # 2. Blocks starting at x: sum of x + seq[pos..j] for j in pos..n-1, and x alone
        # 3. Blocks spanning x: sum of seq[i..pos-1] + x + seq[pos..j] for i in 0..pos-1, j in pos..n-1
        
        # Quick check: x alone must not be in sums_set
        if x in sums_set:
            return None, False
        
        new_sums = set(sums_set)
        new_sums.add(x)
        
        # Blocks ending at x (from left)
        total = x
        for i in range(pos - 1, -1, -1):
            total += seq[i]
            if total in new_sums:
                return None, False
            new_sums.add(total)
        
        # Blocks starting at x (to right)
        total = x
        for j in range(pos, n):
            total += seq[j]
            if total in new_sums:
                return None, False
            new_sums.add(total)
        
        # Blocks spanning x
        # For each i from pos-1 down to 0:
        #   left_sum = sum(seq[i..pos-1])
        #   For each j from pos to n-1:
        #     total = left_sum + x + sum(seq[pos..j])
        left_sums = [x]  # left_sums[k] = sum of last k elements before pos + x... 
        # Actually let's compute more carefully
        # left_part[i] = sum(seq[i..pos-1]) for i = 0..pos-1, and 0 for "no left"
        # right_part[j] = sum(seq[pos..j]) for j = pos..n-1, and 0 for "no right"
        
        left_sums = []
        s = 0
        for i in range(pos - 1, -1, -1):
            s += seq[i]
            left_sums.append(s)
        # left_sums[k] = sum of seq[pos-1-k .. pos-1], so left_sums[0] = seq[pos-1], etc.
        
        right_sums = []
        s = 0
        for j in range(pos, n):
            s += seq[j]
            right_sums.append(s)
        # right_sums[k] = sum of seq[pos..pos+k]
        
        for ls in left_sums:
            total_base = ls + x
            for rs in right_sums:
                total = total_base + rs
                if total in new_sums:
                    return None, False
                new_sums.add(total)
        
        return new_sums, True

    def greedy_asc():
        """Greedy in ascending order."""
        values = []
        sums = set()
        for x in range(1, 257):
            tails = [x]
            total = x
            for a in reversed(values):
                total += a
                tails.append(total)
            if not sums.intersection(tails):
                values.append(x)
                sums.update(tails)
        return values

    def greedy_desc():
        """Greedy in descending order, then sort."""
        values = []
        sums = set()
        for x in range(256, 0, -1):
            tails = [x]
            total = x
            for a in reversed(values):
                total += a
                tails.append(total)
            if not sums.intersection(tails):
                values.append(x)
                sums.update(tails)
        values.sort()
        return values

    def greedy_weighted():
        """Greedy with weighting: prefer numbers that create fewer new sums."""
        values = []
        sums = set()
        remaining = list(range(1, 257))
        
        while remaining:
            best_x = None
            best_conflicts = float('inf')
            for x in remaining:
                tails = [x]
                total = x
                for a in reversed(values):
                    total += a
                    tails.append(total)
                conflicts = len(sums.intersection(tails))
                if conflicts == 0:
                    best_x = x
                    best_conflicts = 0
                    break
                elif conflicts < best_conflicts:
                    best_x = x
                    best_conflicts = conflicts
            if best_x is None:
                break
            remaining.remove(best_x)
            values.append(best_x)
            tails = [best_x]
            total = best_x
            for a in reversed(values[:-1]):
                total += a
                tails.append(total)
            sums.update(tails)
        
        values.sort()
        return values

    # Get initial solutions from different greedy strategies
    sol_asc = greedy_asc()
    sol_desc = greedy_desc()
    sol_weighted = greedy_weighted()
    
    best = max([sol_asc, sol_desc, sol_weighted], key=len)
    best_len = len(best)
    
    if time.time() > deadline - 5:
        return best

    # Now try to improve using DFS with pruning
    # We'll do a time-bounded search
    
    rng = random.Random(42)
    
    def dfs_improve(initial_seq, time_limit):
        """Try to improve initial_seq by DFS insertion."""
        seq = initial_seq[:]
        sums = all_block_sums(seq)
        used = set(seq)
        unused = [x for x in range(1, 257) if x not in used]
        best_local = seq[:]
        best_local_len = len(seq)
        
        # Shuffle unused for diversity
        rng.shuffle(unused)
        
        # Try inserting elements one by one
        for x in unused:
            if time.time() > time_limit:
                break
            
            # Try all positions
            for pos in range(len(seq) + 1):
                new_sums, valid = can_insert(seq, x, pos, sums)
                if valid:
                    # Try this insertion
                    new_seq = seq[:pos] + [x] + seq[pos:]
                    new_sums_local = new_sums
                    new_used = used | {x}
                    
                    # Recursively try to add more
                    remaining_unused = [y for y in unused if y != x and y not in new_used]
                    rng.shuffle(remaining_unused)
                    
                    # Try to add more elements
                    improved = True
                    while improved and time.time() < time_limit:
                        improved = False
                        for y in remaining_unused[:10]:  # limit for speed
                            for p2 in range(len(new_seq) + 1):
                                new_sums2, valid2 = can_insert(new_seq, y, p2, new_sums_local)
                                if valid2:
                                    new_seq2 = new_seq[:p2] + [y] + new_seq[p2:]
                                    new_sums_local = new_sums2
                                    new_used.add(y)
                                    remaining_unused.remove(y)
                                    improved = True
                                    if len(new_seq2) > best_local_len:
                                        best_local = new_seq2[:]
                                        best_local_len = len(new_seq2)
                                    break
                            if improved:
                                break
                    
                    if len(new_seq) > best_local_len:
                        best_local = new_seq[:]
                        best_local_len = len(new_seq)
        
        return best_local

    # Try DFS improvement on each initial solution
    for init in [sol_asc, sol_desc, sol_weighted]:
        if time.time() > deadline - 10:
            break
        
        result = dfs_improve(init, deadline - 10)
        if len(result) > best_len:
            best = result
            best_len = len(result)
    
    # Try random restarts with greedy + improvement
    for trial in range(30):
        if time.time() > deadline - 5:
            break
        
        candidates = list(range(1, 257))
        rng.shuffle(candidates)
        
        values = []
        sums = set()
        for x in candidates:
            tails = [x]
            total = x
            for a in reversed(values):
                total += a
                tails.append(total)
            if not sums.intersection(tails):
                values.append(x)
                sums.update(tails)
        
        values.sort()
        if len(values) > best_len:
            best = values
            best_len = len(values)
        
        # Try to improve
        if len(values) >= best_len - 2:
            result = dfs_improve(values, deadline - 5)
            if len(result) > best_len:
                best = result
                best_len = len(result)
    
    return best
