import time
import random

def construct():
    start_time = time.time()
    time_limit = 85.0  # Safety margin for 90s kill

    def validate_seq(seq):
        """Validate a sequence: strictly increasing, in [1,256], all consecutive block sums distinct"""
        n = len(seq)
        if n == 0:
            return True
        for i in range(n):
            if seq[i] < 1 or seq[i] > 256:
                return False
            if i > 0 and seq[i] <= seq[i-1]:
                return False
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i+1] = prefix[i] + seq[i]
        sums = set()
        for i in range(n):
            for j in range(i+1, n+1):
                s = prefix[j] - prefix[i]
                if s in sums:
                    return False
                sums.add(s)
        return True

    def can_add_fast(prefix_sums, used_sums, x, last_val):
        """Check if x can be appended given current prefix_sums and used_sums.
        Returns (new_prefix, new_sums) if valid, else None."""
        if x <= last_val or x > 256:
            return None
        new_prefix = prefix_sums[-1] + x
        new_sums = []
        for p in prefix_sums:
            s = new_prefix - p
            if s in used_sums:
                return None
            new_sums.append(s)
        # Check new_sums are all distinct
        seen = set()
        for s in new_sums:
            if s in seen:
                return None
            seen.add(s)
        return (new_prefix, new_sums)
    
    def greedy_smallest(start_val=1):
        """Greedy: always pick the smallest valid next element"""
        seq = []
        prefix_sums = [0]
        used_sums = set()
        last_val = 0
        for _ in range(256):
            found = False
            for x in range(start_val, 257):
                result = can_add_fast(prefix_sums, used_sums, x, last_val)
                if result is not None:
                    new_prefix, new_sums = result
                    seq.append(x)
                    prefix_sums.append(new_prefix)
                    used_sums.update(new_sums)
                    last_val = x
                    start_val = x + 1
                    found = True
                    break
            if not found:
                break
        return seq
    
    def greedy_randomized(start_val=1, seed_offset=0, bias_power=1.0):
        """Randomized greedy: at each step, pick from valid candidates with bias"""
        seq = []
        prefix_sums = [0]
        used_sums = set()
        last_val = 0
        rng = random.Random(seed_offset)
        for _ in range(256):
            valid_candidates = []
            for x in range(start_val, 257):
                result = can_add_fast(prefix_sums, used_sums, x, last_val)
                if result is not None:
                    valid_candidates.append((x, result))
            if not valid_candidates:
                break
            if len(valid_candidates) <= 3:
                x, (new_prefix, new_sums) = valid_candidates[0]
            else:
                # Weighted selection: favor smaller values with adjustable bias
                weights = [(1.0 / (i + 1)) ** bias_power for i in range(len(valid_candidates))]
                total_w = sum(weights)
                r = rng.random() * total_w
                cum = 0
                chosen = 0
                for i, w in enumerate(weights):
                    cum += w
                    if r <= cum:
                        chosen = i
                        break
                x, (new_prefix, new_sums) = valid_candidates[chosen]
            seq.append(x)
            prefix_sums.append(new_prefix)
            used_sums.update(new_sums)
            last_val = x
            start_val = x + 1
        return seq
    
    def local_search(seq, max_iters=200, time_budget=3.0):
        """Try to extend or improve a sequence by local modifications.
        Attempts to swap, insert, or replace elements to get a longer valid sequence."""
        start_ls = time.time()
        best = list(seq)
        best_len = len(best)
        rng = random.Random(hash(tuple(seq)) % 10000)
        
        for it in range(max_iters):
            if time.time() - start_ls > time_budget:
                break
            if len(best) >= 256:
                break
            
            # Try to extend the sequence
            if len(best) < 256:
                prefix_sums = [0] * (len(best) + 1)
                for i in range(len(best)):
                    prefix_sums[i+1] = prefix_sums[i] + best[i]
                used_sums = set()
                for i in range(len(best)):
                    for j in range(i+1, len(best)+1):
                        used_sums.add(prefix_sums[j] - prefix_sums[i])
                
                # Try to find a value to append
                last_val = best[-1] if best else 0
                found = False
                for x in range(last_val + 1, 257):
                    result = can_add_fast(prefix_sums, used_sums, x, last_val)
                    if result is not None:
                        new_prefix, new_sums = result
                        candidate = best + [x]
                        if len(candidate) > best_len:
                            best = candidate
                            best_len = len(best)
                        found = True
                        break
                if found:
                    continue
            
            # Try to modify the sequence: pick a random position and try replacing
            if len(best) > 0:
                pos = rng.randint(0, len(best) - 1)
                # Try replacing best[pos] with a different value
                old_val = best[pos]
                # Determine valid range for this position
                min_val = (best[pos-1] + 1) if pos > 0 else 1
                max_val = (best[pos+1] - 1) if pos < len(best) - 1 else 256
                
                if min_val <= max_val:
                    # Try a few random replacements
                    for _ in range(10):
                        if time.time() - start_ls > time_budget:
                            break
                        new_val = rng.randint(min_val, max_val)
                        if new_val == old_val:
                            continue
                        candidate = best[:pos] + [new_val] + best[pos+1:]
                        if validate_seq(candidate):
                            if len(candidate) > best_len:
                                best = candidate
                                best_len = len(best)
                            break
        
        return best
    
    def csp_backtrack(target_len, max_time=5.0, heuristic='smallest'):
        """CSP-style backtracking with forward checking."""
        seq = []
        prefix_sums = [0]
        used_sums = set()
        deadline = start_time + max_time
        found = [False]

        def dfs(last_val, depth):
            if found[0]:
                return
            if time.time() > deadline:
                return
            if depth == target_len:
                found[0] = True
                return

            for x in range(last_val + 1, 257):
                if found[0] or time.time() > deadline:
                    return

                new_prefix = prefix_sums[-1] + x
                new_sums = []
                valid = True
                for p in prefix_sums:
                    s = new_prefix - p
                    if s in used_sums:
                        valid = False
                        break
                    new_sums.append(s)

                if not valid:
                    continue

                seen = set()
                for s in new_sums:
                    if s in seen:
                        valid = False
                        break
                    seen.add(s)

                if not valid:
                    continue

                seq.append(x)
                prefix_sums.append(new_prefix)
                used_sums.update(new_sums)

                dfs(x, depth + 1)

                seq.pop()
                prefix_sums.pop()
                for s in new_sums:
                    used_sums.discard(s)

                if found[0]:
                    return

        dfs(0, 0)
        if found[0]:
            return list(seq)
        return None

    # Strategy 1: Pure greedy smallest
    greedy_result = greedy_smallest()
    best_seq = list(greedy_result)
    best_len = len(best_seq)

    # Strategy 2: Try different starting points with greedy
    for start in range(1, 50):
        if time.time() - start_time > time_limit - 20:
            break
        seq = greedy_smallest(start)
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = list(seq)

    # Strategy 3: Many randomized greedy attempts with various biases
    for attempt in range(200):
        if time.time() - start_time > time_limit - 25:
            break
        bias = 0.5 + (attempt % 5) * 0.3  # Vary bias from 0.5 to 1.7
        start_val = 1 + (attempt % 10)
        seq = greedy_randomized(start_val=start_val, seed_offset=attempt * 7 + 1, bias_power=bias)
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = list(seq)

    # Strategy 4: Local search on best sequences found
    if time.time() - start_time < time_limit - 15:
        improved = local_search(best_seq, max_iters=500, time_budget=5.0)
        if len(improved) > best_len:
            best_len = len(improved)
            best_seq = list(improved)

    # Strategy 5: CSP backtracking to extend beyond greedy
    for target in range(best_len + 1, min(best_len + 6, 35)):
        if time.time() - start_time > time_limit - 10:
            break
        remaining_time = min(4.0, time_limit - 10 - (time.time() - start_time))
        if remaining_time < 0.5:
            break
        result = csp_backtrack(target, max_time=remaining_time)
        if result is not None and len(result) > best_len:
            best_len = len(result)
            best_seq = list(result)
            # Try to extend further
            if time.time() - start_time < time_limit - 8:
                improved = local_search(best_seq, max_iters=200, time_budget=3.0)
                if len(improved) > best_len:
                    best_len = len(improved)
                    best_seq = list(improved)

    # Strategy 6: More diverse randomized attempts
    for attempt in range(100):
        if time.time() - start_time > time_limit - 5:
            break
        start_val = random.randint(1, 30)
        bias = random.uniform(0.3, 2.0)
        seq = greedy_randomized(start_val=start_val, seed_offset=5000 + attempt * 17, bias_power=bias)
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = list(seq)

    # Final validation
    if not validate_seq(best_seq):
        best_seq = [1, 2, 4, 8, 16, 32, 64, 128]
        assert validate_seq(best_seq), "Fallback sequence invalid"

    return best_seq
