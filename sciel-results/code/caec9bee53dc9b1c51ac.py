import random
import time

def construct():
    start_time = time.time()
    DEADLINE = 85.0

    def elapsed():
        return time.time() - start_time

    def within_deadline():
        return elapsed() < DEADLINE

    best_seq = []
    best_len = 0

    def update_best(seq):
        nonlocal best_seq, best_len
        if len(seq) > best_len:
            best_seq = list(seq)
            best_len = len(seq)

    def check_validity(seq):
        if not seq:
            return True
        n = len(seq)
        if len(seq) != len(set(seq)):
            return False
        if seq != sorted(seq):
            return False
        if seq[0] < 1 or seq[-1] > 256:
            return False
        sums = set()
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += seq[j]
                if s in sums:
                    return False
                sums.add(s)
        return True

    def get_valid_candidates(current_seq, last_val):
        """
        Returns a list of valid values x > last_val such that 
        current_seq + [x] is valid.
        """
        if not current_seq:
            # Any x in [1, 256] is valid for length 1
            return list(range(1, 257))
        
        n = len(current_seq)
        # Pre-compute suffix sums of current_seq
        # Suffix sums are sums of a_i...a_n
        suffix_sums = [0] * (n + 1)
        s = 0
        for i in range(n - 1, -1, -1):
            s += current_seq[i]
            suffix_sums[i] = s
        # suffix_sums[i] = sum(current_seq[i:])
        # The new sums added by x are:
        # x, x + current_seq[n-1], x + current_seq[n-1] + current_seq[n-2], ...
        # These are x + suffix_sums[i] for i in 0..n-1? No.
        # Block ending at x:
        # j=n: x
        # j=n-1: x + a[n-1]
        # j=n-2: x + a[n-1] + a[n-2]
        # ...
        # j=0: x + sum(a)
        # So new sums are { x + suffix_sums[i] for i in range(n) } U {x}?
        # suffix_sums[n-1] is a[n-1].
        # suffix_sums[0] is sum(a).
        # So new sums are { x + suffix_sums[i] for i in range(n) }.
        # Wait, the block of length 1 is just x.
        # Is x in the set { x + suffix_sums[i] }?
        # suffix_sums[i] is at least a[n-1] > 0. So x is not in that set.
        # So we must also check x itself.
        # Actually, let's just compute the set of new sums.
        
        # Existing sums
        existing_sums = set()
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += current_seq[j]
                existing_sums.add(s)
                
        valid = []
        # We can iterate x from last_val + 1 to 256
        for x in range(last_val + 1, 257):
            # Check if x collides with existing_sums
            if x in existing_sums:
                continue
            
            # Check if x + suffix collides with existing_sums
            # New sums are x + suffix_sums[i] for i in 0..n-1
            # Note: suffix_sums[i] is sum from i to n-1.
            # The blocks ending at x are:
            # a[i]...a[n-1], x. Sum = suffix_sums[i] + x.
            # So we check if (suffix_sums[i] + x) is in existing_sums.
            
            conflict = False
            for i in range(n):
                if (suffix_sums[i] + x) in existing_sums:
                    conflict = True
                    break
            if not conflict:
                valid.append(x)
                
        return valid

    def greedy_build_deterministic():
        seq = []
        last = 0
        while True:
            candidates = get_valid_candidates(seq, last)
            if not candidates:
                break
            # Pick smallest
            x = candidates[0]
            seq.append(x)
            last = x
        return seq

    def randomized_greedy_with_backtracking(max_depth=15, branching_factor=5, num_runs=1000):
        best_local = []
        
        def backtrack(seq, depth):
            nonlocal best_local
            if len(seq) > len(best_local):
                best_local = list(seq)
            
            if not within_deadline():
                return
                
            if depth >= max_depth:
                return

            last = seq[-1] if seq else 0
            candidates = get_valid_candidates(seq, last)
            
            if not candidates:
                return

            # Shuffle and take top branching_factor
            random.shuffle(candidates)
            # We might want to bias towards smaller or larger?
            # Smaller leaves more room. Larger might be needed if small are blocked.
            # Let's just take a random subset.
            selected = candidates[:branching_factor]
            
            for x in selected:
                if not within_deadline():
                    break
                backtrack(seq + [x], depth + 1)

        for _ in range(num_runs):
            if not within_deadline():
                break
            # Random start? No, start empty.
            backtrack([], 0)
            
        return best_local

    # Phase 1: Deterministic Greedy
    seq = greedy_build_deterministic()
    update_best(seq)
    
    # Phase 2: Randomized Greedy (Fast, no backtracking, just restarts)
    def fast_randomized_greedy(num_restarts=500):
        best_local = []
        for _ in range(num_restarts):
            if not within_deadline():
                break
            seq = []
            last = 0
            while True:
                if not within_deadline():
                    break
                candidates = get_valid_candidates(seq, last)
                if not candidates:
                    break
                # Pick random candidate
                x = random.choice(candidates)
                seq.append(x)
                last = x
            if len(seq) > len(best_local):
                best_local = seq
        return best_local

    seq = fast_randomized_greedy(num_restarts=1000)
    update_best(seq)

    # Phase 3: Backtracking Search
    # Use a smaller max_depth and branching to allow many runs
    # The seed program didn't have backtracking. This is a structural change.
    seq = randomized_greedy_with_backtracking(max_depth=12, branching_factor=3, num_runs=50)
    update_best(seq)

    # Phase 4: Local Search on Best Seq
    # Try to extend by inserting
    def try_extend(seq):
        seq = list(seq)
        n = len(seq)
        # Try inserting at each position
        # We can optimize by only checking validity of the new sequence
        # But check_validity is O(n^2).
        # Let's just try random insertions.
        
        # Try to find a gap where we can insert a value
        for pos in range(n + 1):
            min_val = (seq[pos-1] + 1) if pos > 0 else 1
            max_val = (seq[pos] - 1) if pos < n else 256
            if min_val > max_val:
                continue
            
            # Try a few values
            # Randomize the order
            candidates = list(range(min_val, max_val + 1))
            random.shuffle(candidates)
            for x in candidates[:20]: # Limit tries per pos
                if not within_deadline():
                    break
                candidate = seq[:pos] + [x] + seq[pos:]
                if check_validity(candidate):
                    return candidate
        return seq

    if within_deadline():
        extended = try_extend(best_seq)
        update_best(extended)
        
        # Iterate extension
        for _ in range(10):
            if not within_deadline():
                break
            extended = try_extend(best_seq)
            if len(extended) == len(best_seq):
                break
            update_best(extended)

    return best_seq
