# EVOLVE-BLOCK-START
def construct():
    import time
    import math
    import random
    from functools import lru_cache

    start_time = time.time()
    deadline = start_time + 75.0  # Safety margin before 90s kill

    def sieve_primes(n):
        is_prime = bytearray([1]) * (n + 1)
        is_prime[0:2] = b'\x00\x00'
        for i in range(2, int(math.isqrt(n)) + 1):
            if is_prime[i]:
                is_prime[i*i::i] = b'\x00' * len(range(i*i, n+1, i))
        return [i for i in range(n + 1) if is_prime[i]]

    primes = sieve_primes(10000)
    P = len(primes)
    prime_set = set(primes)

    def is_valid(seq):
        if len(seq) < 2:
            return True
        gaps = [seq[i+1] - seq[i] for i in range(len(seq)-1)]
        return all(gaps[i+1] >= gaps[i] for i in range(len(gaps)-1))

    # Greedy initialization
    def greedy_construct():
        values = []
        for p in primes:
            if len(values) < 2 or p - values[-1] >= values[-1] - values[-2]:
                values.append(p)
        return values

    best_seq = greedy_construct()
    best_len = len(best_seq)

    # Try to extend/insert using local search
    def try_insert(seq, candidate):
        # Try inserting candidate at every position
        for i in range(len(seq) + 1):
            new_seq = seq[:i] + [candidate] + seq[i:]
            if is_valid(new_seq):
                return new_seq
        return None

    def local_search(seq, max_time=None):
        current = list(seq)
        improved = True
        while improved:
            improved = False
            for i in range(len(current)):
                # Try replacing current[i] with a nearby prime
                old_val = current[i]
                # Try primes slightly before and after
                for offset in range(1, 50):
                    for new_val in [old_val + offset, old_val - offset]:
                        if new_val not in prime_set or new_val < 2 or new_val > 10000:
                            continue
                        if new_val in current:
                            continue
                        test = current[:i] + [new_val] + current[i+1:]
                        if test != sorted(test):
                            continue
                        if is_valid(test):
                            # Check if it's actually better (longer or same length but... just accept for now)
                            # Since we want to maximize length, replacements don't help length.
                            # We need insertions.
                            pass
            # Try insertions
            for p in primes:
                if p in current:
                    continue
                new_seq = try_insert(current, p)
                if new_seq is not None and len(new_seq) > len(current):
                    current = new_seq
                    improved = True
                    if max_time and time.time() > max_time:
                        break
            if max_time and time.time() > max_time:
                break
        return current

    # More aggressive: Start from greedy, then try random restarts of greedy with different starting points?
    # Actually, let's do a dynamic programming approach but optimized.
    # State: (last_prime_index, last_gap) -> max_length
    # But last_gap can be large.
    
    # Let's try a different DP:
    # dp[i][j] = max length of valid sequence ending at prime[i] with last gap being prime[j] - prime[i-1]? No.
    # Let's use the finding: "Greedy local search with insertion heuristics"
    
    # Let's implement a proper local search that tries to insert primes.
    def improve_sequence(seq):
        current = list(seq)
        changed = True
        while changed:
            changed = False
            # Try inserting every prime not in current
            for p in primes:
                if p in current:
                    continue
                # Find valid insertion point
                # We can binary search or just check all
                for i in range(len(current) + 1):
                    new_seq = current[:i] + [p] + current[i:]
                    if is_valid(new_seq):
                        current = new_seq
                        changed = True
                        break  # Try next prime? Or restart?
                        # Restarting is safer to find more
            # This is O(P^3) worst case. P=1229. 1229^3 is too big.
            # We need to be smarter.
            
        return current

    # Faster insertion check
    def can_insert(seq, p, i):
        # Insert p at index i
        # Check gap before: if i > 0, p - seq[i-1] >= (seq[i-1] - seq[i-2]) if i > 1, else just gap
        # Check gap after: if i < len(seq), seq[i] - p >= (p - seq[i-1]) if i > 0
        # Actually, gaps are:
        # ... seq[i-1], p, seq[i] ...
        # Gap 1: p - seq[i-1]
        # Gap 2: seq[i] - p
        # Previous gap (before seq[i-1]): g_prev = seq[i-1] - seq[i-2] (if i>1)
        # Next gap (after seq[i]): g_next = seq[i+1] - seq[i] (if i < len-1)
        
        # We need:
        # 1. If i > 0: p - seq[i-1] >= (seq[i-1] - seq[i-2]) if i > 1, else always true for first gap?
        #    No, the condition is that the sequence of gaps is non-decreasing.
        #    Let gaps be g_0, g_1, ..., g_{k-1}
        #    Inserting p creates new gaps g'_i.
        #    If i=0: new gap is p - (nothing) -> just p. Next gap is seq[0] - p.
        #            Original gap 0 was seq[0] - seq[-1]? No, seq is 1D.
        #            Original gaps: seq[1]-seq[0], seq[2]-seq[1]...
        #            New gaps: p - (start), seq[0]-p, seq[1]-seq[0]...
        #            Wait, the sequence is just the list of primes.
        #            If we insert p at index i:
        #            New list: L' = L[:i] + [p] + L[i:]
        #            Gaps:
        #            If i > 0: g_new_1 = p - L[i-1]
        #            If i < len(L): g_new_2 = L[i] - p
        #            
        #            Constraints:
        #            If i > 1: g_new_1 >= L[i-1] - L[i-2]  (prev gap)
        #            If 0 < i < len(L): g_new_2 >= g_new_1
        #            If i < len(L) - 1: L[i+1] - L[i] >= g_new_2  (next gap)
        #            
        #            Edge cases:
        #            i == 0:
        #              g_new_2 = L[0] - p
        #              Constraint: L[1] - L[0] >= g_new_2 (if len(L) > 1)
        #            i == len(L):
        #              g_new_1 = p - L[-1]
        #              Constraint: g_new_1 >= L[-1] - L[-2] (if len(L) > 1)
            
        if i == 0:
            if len(seq) == 0:
                return True
            if len(seq) == 1:
                return True # Any single gap is valid
            g_new = seq[0] - p
            if g_new <= 0: return False
            g_next = seq[1] - seq[0]
            return g_next >= g_new
        elif i == len(seq):
            if len(seq) == 0:
                return True
            if len(seq) == 1:
                return True
            g_new = p - seq[-1]
            if g_new <= 0: return False
            g_prev = seq[-1] - seq[-2]
            return g_new >= g_prev
        else:
            # 0 < i < len(seq)
            if len(seq) == 2 and i == 1:
                 # seq = [a, b]. insert p at 1 -> [a, p, b]
                 g1 = p - a
                 g2 = b - p
                 if g1 <= 0 or g2 <= 0: return False
                 return g2 >= g1
            # General
            g_prev = seq[i-1] - (seq[i-2] if i > 1 else 0) # Careful with i=1
            # If i=1, there is no gap before seq[0]. So g_prev constraint is vacuous?
            # The sequence of gaps starts with seq[1]-seq[0].
            # If we insert at i=1: [a, p, b, c]. Gaps: p-a, b-p, c-b.
            # Constraint: b-p >= p-a AND c-b >= b-p.
            # There is no constraint on p-a from the left.
            
            if i > 1:
                g_left = seq[i-1] - seq[i-2]
                g_new1 = p - seq[i-1]
                if g_new1 < g_left:
                    return False
            else:
                g_new1 = p - seq[i-1]
                if g_new1 <= 0:
                    return False
            
            g_new2 = seq[i] - p
            if g_new2 <= 0:
                return False
            
            if g_new2 < g_new1:
                return False
            
            if i < len(seq) - 1:
                g_right = seq[i+1] - seq[i]
                if g_right < g_new2:
                    return False
            
            return True

    def fast_improve(seq):
        current = list(seq)
        improved = True
        while improved:
            improved = False
            for p in primes:
                if p in current:
                    continue
                # Binary search or linear scan for valid insertion
                for i in range(len(current) + 1):
                    if can_insert(current, p, i):
                        current = current[:i] + [p] + current[i:]
                        improved = True
                        break
            if time.time() > deadline - 5.0:
                break
        return current

    # Try multiple initializations
    # 1. Greedy
    best = greedy_construct()
    
    # 2. Start with small primes, then greedy?
    # The greedy already does this effectively.
    
    # 3. Randomized greedy?
    # Sometimes skipping an early prime allows for a much longer sequence later.
    # Let's try a few random restarts where we skip the first K primes.
    for start_skip in range(0, 20):
        if time.time() > deadline - 10.0:
            break
        seq = []
        for p in primes[start_skip:]:
            if len(seq) < 2 or p - seq[-1] >= seq[-1] - seq[-2]:
                seq.append(p)
        if len(seq) > len(best):
            best = seq
    
    # Now improve the best one
    best = fast_improve(best)
    
    # Try improving other candidates if time permits
    for start_skip in range(0, 10):
        if time.time() > deadline - 5.0:
            break
        seq = []
        for p in primes[start_skip:]:
            if len(seq) < 2 or p - seq[-1] >= seq[-1] - seq[-2]:
                seq.append(p)
        if len(seq) >= len(best):
            improved = fast_improve(seq)
            if len(improved) > len(best):
                best = improved

    return best
# EVOLVE-BLOCK-END
