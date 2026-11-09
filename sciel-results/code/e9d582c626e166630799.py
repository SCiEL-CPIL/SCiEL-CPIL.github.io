import time
import random
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    if n <= 1:
        return "".join(map(str, range(1, n+1)))
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        # Classic is 33. Optimal is 33? 
        # 1!+2!+3!+4! = 33.
        # Let's check if 33 is optimal for n=4.
        # Yes, 33 is the known optimal for n=4.
        pass

    if n > 4:
        # For n >= 5, use the optimization strategy
        all_perms = list(permutations(range(1, n+1)))
        N = len(all_perms)
        
        # Precompute overlaps? 
        # overlap(A, B) is the largest k such that A[-k:] == B[:k]
        # We can represent perms as tuples of ints.
        
        def get_overlap(A, B):
            # A, B are tuples of length n
            # Max possible overlap is n-1
            for k in range(n - 1, 0, -1):
                if A[-k:] == B[:k]:
                    return k
            return 0

        # Greedy Initialization
        def greedy_init():
            unvisited = set(range(N))
            # Map perm to index
            # This is expensive if we do it naively.
            # Let's just keep the list of permutations and indices.
            # To make it faster, we can use a list of available indices.
            
            # Random start
            start_idx = random.randint(0, N - 1)
            seq = [start_idx]
            unvisited.remove(start_idx)
            
            # To speed up greedy, we need to find max overlap quickly.
            # With N=5040, checking all unvisited at each step is 5040*5040/2 ~ 12.5M operations.
            # Each operation is O(n) comparison. 12.5M * 7 ~ 87.5M ops.
            # In Python, this might take a few seconds. Acceptable.
            
            while unvisited:
                last_idx = seq[-1]
                last_perm = all_perms[last_idx]
                best_idx = -1
                best_ov = -1
                
                # We can iterate over a random subset or all?
                # All is safer for quality.
                # Optimization: if we find overlap n-1, we can stop early?
                # No, we just want *a* max overlap. If multiple, pick random.
                
                candidates = []
                max_ov_found = -1
                
                for idx in unvisited:
                    ov = get_overlap(last_perm, all_perms[idx])
                    if ov > max_ov_found:
                        max_ov_found = ov
                        candidates = [idx]
                    elif ov == max_ov_found:
                        candidates.append(idx)
                        
                if candidates:
                    next_idx = random.choice(candidates)
                else:
                    next_idx = random.choice(list(unvisited))
                
                seq.append(next_idx)
                unvisited.remove(next_idx)
                
            return seq

        # Calculate total length (or total overlap)
        def calc_score(seq):
            # Length = N * n - sum(overlaps)
            # We want to maximize sum(overlaps)
            total_ov = 0
            for i in range(len(seq) - 1):
                total_ov += get_overlap(all_perms[seq[i]], all_perms[seq[i+1]])
            return total_ov

        # Local Search: 2-opt (swap adjacent)
        def local_search(seq, max_time):
            current_score = calc_score(seq)
            improved = True
            while improved and time.time() < max_time:
                improved = False
                # Randomize start position to avoid local minima?
                # Or just scan left to right.
                for i in range(len(seq) - 1):
                    if time.time() > max_time:
                        return seq, current_score
                    
                    # Swap seq[i] and seq[i+1]
                    old_score = current_score
                    # Remove contributions of i, i+1
                    # Note: boundaries
                    ov_left = get_overlap(all_perms[seq[i-1]], all_perms[seq[i]]) if i > 0 else 0
                    ov_mid = get_overlap(all_perms[seq[i]], all_perms[seq[i+1]])
                    ov_right = get_overlap(all_perms[seq[i+1]], all_perms[seq[i+2]]) if i + 2 < len(seq) else 0
                    
                    # New connections
                    new_ov_left = get_overlap(all_perms[seq[i-1]], all_perms[seq[i+1]]) if i > 0 else 0
                    new_ov_mid = get_overlap(all_perms[seq[i+1]], all_perms[seq[i]])
                    new_ov_right = get_overlap(all_perms[seq[i]], all_perms[seq[i+2]]) if i + 2 < len(seq) else 0
                    
                    delta = (new_ov_left + new_ov_mid + new_ov_right) - (ov_left + ov_mid + ov_right)
                    
                    if delta > 0:
                        seq[i], seq[i+1] = seq[i+1], seq[i]
                        current_score += delta
                        improved = True
                        # After swap, check previous position again?
                        # Standard 2-opt usually restarts or continues.
                        # Let's continue.
                        
            return seq, current_score

        # Run multiple restarts
        best_seq = None
        best_score = -1
        
        # Time budget allocation
        # 1. Greedy init (fast)
        # 2. Local search (slow)
        
        num_restarts = 10 # Adjust based on time
        restart_time_limit = (deadline - time.time()) / num_restarts
        
        for r in range(num_restarts):
            if time.time() > deadline - 10:
                break
            
            seq = greedy_init()
            seq, score = local_search(seq, time.time() + restart_time_limit * 0.8)
            
            if score > best_score:
                best_score = score
                best_seq = seq[:]
            
            # If we found a very good score, we can reduce restarts
            # The theoretical max score is (N-1)*(n-1)
            # Max length is N*n - MaxScore
            # For n=7, N=5040, MaxScore = 5039 * 6 = 30234
            # Min Length = 35280 - 30234 = 5046? 
            # Wait. 
            # If all overlaps were 6, length = 7 + 5039*1 = 5046.
            # But we know optimal is ~5905.
            # So we cannot have all overlaps 6.
            # The structure of the graph limits this.
            
        # Construct final word
        if best_seq is None:
            # Fallback to classic if something went wrong
            return "".join(map(str, classic_construct(n)))
            
        word = list(all_perms[best_seq[0]])
        for i in range(1, len(best_seq)):
            prev = all_perms[best_seq[i-1]]
            curr = all_perms[best_seq[i]]
            ov = get_overlap(prev, curr)
            word.extend(curr[ov:])
            
        return "".join(map(str, word))

    else:
        # For small n, we can use the known optimal or classic
        if n == 4:
             # Classic is 33.
             # Let's just return classic for n=4 to be safe, or run the optimizer if time permits.
             # But 24 perms is small, optimizer is fast.
             pass

    # If n is small, we might have skipped the logic above if I structured it badly.
    # Let's ensure n=4 is handled by the general logic or specific case.
    # The `if n > 4` block handles n>=5.
    # We need to handle n=4 in the general logic or add it there.
    # Let's move the logic to handle n >= 4.

    # Re-evaluating structure:
    # The code above puts n>4 in a block. I should change it to n>=4 or just let it run for all n>3.
    # For n=3, 6 perms. Very fast.
    # For n=4, 24 perms. Very fast.
    
    # I will rewrite the block to handle n >= 4.

    # Wait, the SEARCH/REPLACE must match the current code.
    # The current code has `if n <= 1`, `if n == 2`, `if n == 3`, `if n == 4: pass`.
    # Then it defines `classic_construct` and runs it.
    # I will replace the entire logic from `# For n up to 7...` to the end.

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    if n <= 1:
        return "".join(map(str, range(1, n+1)))
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    
    # For n >= 4, use optimization
    all_perms = list(permutations(range(1, n+1)))
    N = len(all_perms)
    
    def get_overlap(A, B):
        for k in range(n - 1, 0, -1):
            if A[-k:] == B[:k]:
                return k
        return 0

    def greedy_init():
        unvisited = set(range(N))
        start_idx = random.randint(0, N - 1)
        seq = [start_idx]
        unvisited.remove(start_idx)
        
        while unvisited:
            last_idx = seq[-1]
            last_perm = all_perms[last_idx]
            best_idx = -1
            best_ov = -1
            candidates = []
            max_ov_found = -1
            
            for idx in unvisited:
                ov = get_overlap(last_perm, all_perms[idx])
                if ov > max_ov_found:
                    max_ov_found = ov
                    candidates = [idx]
                elif ov == max_ov_found:
                    candidates.append(idx)
                        
            if candidates:
                next_idx = random.choice(candidates)
            else:
                next_idx = random.choice(list(unvisited))
                
            seq.append(next_idx)
            unvisited.remove(next_idx)
        return seq

    def calc_score(seq):
        total_ov = 0
        for i in range(len(seq) - 1):
            total_ov += get_overlap(all_perms[seq[i]], all_perms[seq[i+1]])
        return total_ov

    def local_search(seq, max_time):
        current_score = calc_score(seq)
        improved = True
        while improved and time.time() < max_time:
            improved = False
            for i in range(len(seq) - 1):
                if time.time() > max_time:
                    return seq, current_score
                ov_left = get_overlap(all_perms[seq[i-1]], all_perms[seq[i]]) if i > 0 else 0
                ov_mid = get_overlap(all_perms[seq[i]], all_perms[seq[i+1]])
                ov_right = get_overlap(all_perms[seq[i+1]], all_perms[seq[i+2]]) if i + 2 < len(seq) else 0
                new_ov_left = get_overlap(all_perms[seq[i-1]], all_perms[seq[i+1]]) if i > 0 else 0
                new_ov_mid = get_overlap(all_perms[seq[i+1]], all_perms[seq[i]])
                new_ov_right = get_overlap(all_perms[seq[i]], all_perms[seq[i+2]]) if i + 2 < len(seq) else 0
                delta = (new_ov_left + new_ov_mid + new_ov_right) - (ov_left + ov_mid + ov_right)
                if delta > 0:
                    seq[i], seq[i+1] = seq[i+1], seq[i]
                    current_score += delta
                    improved = True
        return seq, current_score

    best_seq = None
    best_score = -1
    
    # Allocate time
    # Greedy is O(N^2 * n). For N=5040, this is ~12M * 7 = 84M ops.
    # Python might take ~5-10s for one greedy init.
    # Local search is O(N * Iterations).
    
    # Strategy:
    # 1. Run 1 greedy init + local search.
    # 2. If time remains, run more.
    
    random.seed(42)
    
    # First attempt
    if time.time() < deadline - 20:
        seq = greedy_init()
        seq, score = local_search(seq, deadline - 10)
        best_seq = seq
        best_score = score

    # Subsequent attempts if time allows
    # Each restart needs ~5-10s for greedy + some local search time
    while time.time() < deadline - 15:
        seq = greedy_init()
        # Give local search a smaller budget for restarts
        ls_budget = min(10, deadline - time.time() - 5)
        if ls_budget > 1:
             seq, score = local_search(seq, time.time() + ls_budget)
        else:
             score = calc_score(seq)
             
        if score > best_score:
            best_seq = seq
            best_score = score
            
    if best_seq is None:
        # Fallback
        word = [1]
        for m in range(2, n + 1):
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                overlap = 0
                max_ov = min(len(new), len(block))
                for o in range(max_ov, -1, -1):
                    if o == 0 or new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        return "".join(map(str, word))

    word = list(all_perms[best_seq[0]])
    for i in range(1, len(best_seq)):
        prev = all_perms[best_seq[i-1]]
        curr = all_perms[best_seq[i]]
        ov = get_overlap(prev, curr)
        word.extend(curr[ov:])
        
    return "".join(map(str, word))
