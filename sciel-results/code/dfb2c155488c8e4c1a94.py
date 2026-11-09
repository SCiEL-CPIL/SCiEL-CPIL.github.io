import time

def construct():
    deadline = time.time() + 80.0  # 10s safety margin before 90s kill
    n = 128
    full_mask = (1 << (n + 1)) - 2  # bits 1..128

    def has_bad(s):
        """Return True if set s (bitset) contains a forbidden triple."""
        bits = [i for i in range(1, n + 1) if (s >> i) & 1]
        L = len(bits)
        for i in range(L):
            a = bits[i]
            for j in range(i + 1, L):
                b = bits[j]
                # c must be in S, c != a, c != b, and (c+a) in S, (c+b) in S
                # (s ^ a) has bits of S shifted by a (where valid)
                # (s ^ b) has bits of S shifted by b
                # intersection gives c such that c+a in S and c+b in S
                # exclude a and b themselves
                bad_c = s & (s ^ a) & (s ^ b)
                bad_c &= ~(1 << a) & ~(1 << b)
                if bad_c:
                    return True
        return False

    def add_ok(s, x):
        """Check if adding x to s creates no forbidden triple."""
        if (s >> x) & 1:
            return True  # already in set
        new_s = s | (1 << x)
        return not has_bad(new_s)

    # --- Phase 1: Greedy construction ---
    s = 0
    # Try adding elements in a heuristic order: larger elements first
    # (pair sums of large elements exceed 128 more often)
    for x in range(n, 0, -1):
        if time.time() > deadline - 5.0:
            break
        if add_ok(s, x):
            s |= (1 << x)

    # --- Phase 2: Local search (swap moves) ---
    best_s = s
    best_size = bin(s).count('1')
    no_improve_count = 0

    while time.time() < deadline - 2.0:
        improved = False
        # Get current elements
        cur_bits = [i for i in range(1, n + 1) if (s >> i) & 1]
        # Try each swap: remove r, add a
        for r in cur_bits:
            if time.time() > deadline - 3.0:
                break
            s_without = s & ~(1 << r)
            for a in range(1, n + 1):
                if (s_without >> a) & 1:
                    continue
                new_s = s_without | (1 << a)
                if has_bad(new_s):
                    continue
                new_size = bin(new_s).count('1')
                if new_size > best_size:
                    best_s = new_s
                    best_size = new_size
                    s = new_s
                    improved = True
                    no_improve_count = 0
                    break
            if improved:
                break
        if not improved:
            no_improve_count += 1
            if no_improve_count >= 3:
                break

    # --- Phase 3: Try greedy add on the best found ---
    for x in range(n, 0, -1):
        if time.time() > deadline - 1.0:
            break
        if add_ok(best_s, x):
            best_s |= (1 << x)

    # --- Phase 4: More swap rounds on best ---
    for _ in range(50):
        if time.time() > deadline - 1.0:
            break
        cur_bits = [i for i in range(1, n + 1) if (best_s >> i) & 1]
        improved = False
        for r in cur_bits:
            if time.time() > deadline - 1.0:
                break
            s_without = best_s & ~(1 << r)
            for a in range(1, n + 1):
                if (s_without >> a) & 1:
                    continue
                new_s = s_without | (1 << a)
                if has_bad(new_s):
                    continue
                new_size = bin(new_s).count('1')
                if new_size > best_size:
                    best_s = new_s
                    best_size = new_size
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break

    # Final: try adding any remaining elements
    for x in range(n, 0, -1):
        if time.time() > deadline - 0.5:
            break
        if add_ok(best_s, x):
            best_s |= (1 << x)

    # Convert to list of Python ints
    result = [i for i in range(1, n + 1) if (best_s >> i) & 1]
    return result
