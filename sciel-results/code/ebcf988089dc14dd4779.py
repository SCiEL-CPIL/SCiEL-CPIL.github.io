import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    n = 256

    def is_valid_set(S):
        """Check if S satisfies the property: for all a<b<=c in S, a does NOT divide b+c."""
        S_list = sorted(S)
        m = len(S_list)
        for i in range(m):
            a = S_list[i]
            for j in range(i + 1, m):
                b = S_list[j]
                if (b + b) % a == 0:
                    return False
                for k in range(j + 1, m):
                    c = S_list[k]
                    if (b + c) % a == 0:
                        return False
        return True

    def can_add_greedy(S, pair_sums_set, x):
        """Fast check when all elements in S are > x (descending greedy).
        Only need: no pair b,c in S with b<=c and x | (b+c)."""
        # Check if any pair sum is divisible by x
        max_sum = 512  # 256 + 256
        for s in range(2 * x, max_sum + 1, x):
            if s in pair_sums_set:
                return False
        return True

    def can_add_full(S, x):
        """Full check: x can be added to S satisfying a<b<=c, a ∤ (b+c)."""
        if x in S:
            return False
        S_list = sorted(S)
        m = len(S_list)
        # Case 1 & 2: x is 'b' or 'c', some a in S with a < x
        for i in range(m):
            a = S_list[i]
            if a >= x:
                continue
            # a < x
            # Case 1: x is 'c'. Need a < b <= x for b in S.
            for j in range(m):
                b = S_list[j]
                if a < b <= x:
                    if (b + x) % a == 0:
                        return False
            # Case 2: x is 'b'. Need a < x <= c for c in S.
            for j in range(m):
                c = S_list[j]
                if c >= x:
                    if (x + c) % a == 0:
                        return False
        # Case 3: x is 'a'. Need x < b <= c for b, c in S.
        for i in range(m):
            b = S_list[i]
            if b > x:
                if (b + b) % x == 0:
                    return False
                for j in range(i + 1, m):
                    c = S_list[k] if False else S_list[j]
                    if (b + c) % x == 0:
                        return False
        return True

    def build_pair_sums(S):
        """Build set of all pair sums b+c where b<=c in S."""
        S_list = sorted(S)
        m = len(S_list)
        sums = set()
        for i in range(m):
            b = S_list[i]
            for j in range(i, m):
                c = S_list[j]
                sums.add(b + c)
        return sums

    def local_search(S, rng, time_limit):
        """Try to improve S by swapping elements for better ones with residue-aware pruning."""
        S = set(S)
        improved = True
        # Precompute residues for pruning: for each small modulus, track which residues are "blocked"
        while improved and time.time() < time_limit:
            improved = False
            S_list = sorted(S)
            for idx in range(len(S_list)):
                if time.time() > time_limit:
                    return S
                a = S_list[idx]
                S.remove(a)
                # Try to add a smaller element
                for x in range(1, a):
                    if x not in S:
                        if can_add_full(S, x):
                            S.add(x)
                            improved = True
                            break
                if not improved:
                    S.add(a)
        return S

    def local_search_swap(S, rng, time_limit):
        """Try to improve S by swapping one element for another (keeping size)."""
        S = set(S)
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            S_list = sorted(S)
            for idx in range(len(S_list)):
                if time.time() > time_limit:
                    return S
                a = S_list[idx]
                S.remove(a)
                for x in range(1, n + 1):
                    if x not in S and x != a:
                        if can_add_full(S, x):
                            S.add(x)
                            improved = True
                            break
                if not improved:
                    S.add(a)
        return S

    best_size = 0
    best_S = set()

    # Strategy 1: Descending greedy with optimized pair-sums check
    if time.time() < deadline - 5.0:
        S = set()
        pair_sums = set()
        for x in range(n, 0, -1):
            if time.time() > deadline - 3.0:
                break
            if can_add_greedy(S, pair_sums, x):
                # Update pair sums: add x + s for all s in S
                S_list = sorted(S)
                for s in S_list:
                    pair_sums.add(x + s)
                pair_sums.add(2 * x)
                S.add(x)
        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)

    # Strategy 2: Start with upper half, greedily add from below
    for restart in range(50):
        if time.time() > deadline - 5.0:
            break

        rng = random.Random(restart * 1000 + 42)

        base_start = n - rng.randint(40, 80)
        S = set(range(base_start, n + 1))

        if not is_valid_set(S):
            S_list = sorted(S)
            valid_S = set()
            for x in S_list:
                if can_add_full(valid_S, x):
                    valid_S.add(x)
            S = valid_S

        candidates = [x for x in range(1, base_start) if x not in S]
        rng.shuffle(candidates)
        for x in candidates:
            if time.time() > deadline - 3.0:
                break
            if can_add_full(S, x):
                S.add(x)

        candidates2 = [x for x in range(1, n + 1) if x not in S]
        rng.shuffle(candidates2)
        for x in candidates2:
            if time.time() > deadline - 3.0:
                break
            if can_add_full(S, x):
                S.add(x)

        # Local search refinement
        if time.time() < deadline - 5.0:
            S = local_search(S, rng, deadline - 3.0)

        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)

    # Strategy 3: Start with a random subset, greedily add
    for restart in range(50):
        if time.time() > deadline - 5.0:
            break

        rng = random.Random(restart * 2000 + 99)

        S = set(rng.sample(range(1, n + 1), rng.randint(20, 40)))

        if not is_valid_set(S):
            S_list = sorted(S)
            valid_S = set()
            for x in S_list:
                if can_add_full(valid_S, x):
                    valid_S.add(x)
            S = valid_S

        candidates = [x for x in range(1, n + 1) if x not in S]
        rng.shuffle(candidates)
        for x in candidates:
            if time.time() > deadline - 3.0:
                break
            if can_add_full(S, x):
                S.add(x)

        candidates2 = [x for x in range(1, n + 1) if x not in S]
        rng.shuffle(candidates2)
        for x in candidates2:
            if time.time() > deadline - 3.0:
                break
            if can_add_full(S, x):
                S.add(x)

        # Local search refinement
        if time.time() < deadline - 5.0:
            S = local_search(S, rng, deadline - 3.0)

        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)

    # Strategy 4: Start with all odd numbers, then add evens
    if time.time() < deadline - 5.0:
        S = set(range(1, n + 1, 2))
        if not is_valid_set(S):
            S_list = sorted(S)
            valid_S = set()
            for x in S_list:
                if can_add_full(valid_S, x):
                    valid_S.add(x)
            S = valid_S

        candidates = [x for x in range(2, n + 1, 2) if x not in S]
        rng = random.Random(12345)
        rng.shuffle(candidates)
        for x in candidates:
            if time.time() > deadline - 3.0:
                break
            if can_add_full(S, x):
                S.add(x)

        # Local search refinement
        if time.time() < deadline - 5.0:
            S = local_search(S, rng, deadline - 3.0)

        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)

    # Strategy 5: Start with numbers > n/3, then add smaller
    if time.time() < deadline - 5.0:
        S = set(range(86, n + 1))
        if not is_valid_set(S):
            S_list = sorted(S)
            valid_S = set()
            for x in S_list:
                if can_add_full(valid_S, x):
                    valid_S.add(x)
            S = valid_S

        candidates = [x for x in range(1, 86) if x not in S]
        rng = random.Random(54321)
        rng.shuffle(candidates)
        for x in candidates:
            if time.time() > deadline - 3.0:
                break
            if can_add_full(S, x):
                S.add(x)

        # Local search refinement
        if time.time() < deadline - 5.0:
            S = local_search(S, rng, deadline - 3.0)

        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)

    # Strategy 6: Descending greedy with different starting points
    for restart in range(20):
        if time.time() > deadline - 5.0:
            break

        rng = random.Random(restart * 3000 + 777)

        upper_size = rng.randint(30, 60)
        upper_start = n - upper_size + 1
        S = set(range(upper_start, n + 1))

        if not is_valid_set(S):
            S_list = sorted(S)
            valid_S = set()
            for x in S_list:
                if can_add_full(valid_S, x):
                    valid_S.add(x)
            S = valid_S

        # Descending greedy from n-1 down to 1
        pair_sums = build_pair_sums(S)
        for x in range(n - 1, 0, -1):
            if time.time() > deadline - 3.0:
                break
            if x not in S and can_add_greedy(S, pair_sums, x):
                S_list = sorted(S)
                for s in S_list:
                    pair_sums.add(x + s)
                pair_sums.add(2 * x)
                S.add(x)

        # Local search refinement
        if time.time() < deadline - 5.0:
            S = local_search(S, rng, deadline - 3.0)

        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)

    # Strategy 7: Residue-aware construction - track residues mod small numbers
    if time.time() < deadline - 5.0:
        # For each modulus m in [2..16], track which residues are "used" by pairs
        # An element x is blocked if for some m dividing potential conflicts,
        # the pair-sum residues already cover x's residue class
        S = set()
        # Track: for each small modulus q, the set of residues r such that
        # some pair in S sums to a value ≡ r (mod q)
        mod_residues = {}
        for q in range(2, 17):
            mod_residues[q] = set()

        for x in range(n, 0, -1):
            if time.time() > deadline - 3.0:
                break
            # Check if x is blocked: is there any pair sum divisible by x?
            # Equivalent: for each multiple k*x, is there a pair summing to k*x?
            # Use residue tracking as a quick filter
            blocked = False
            for q in range(2, 17):
                if q > x:
                    continue
                # If x shares a factor with q, the residue tracking is less useful
                # Quick check: if 0 mod q is in mod_residues[q], and x | q * something...
                # Actually, let's just do the direct check but with early termination
                pass

            # Direct check using pair_sums for correctness
            # Build pair sums incrementally
            S_list = sorted(S)
            m = len(S_list)
            can_add = True
            for i in range(m):
                b = S_list[i]
                for j in range(i, m):
                    c = S_list[j]
                    if (b + c) % x == 0:
                        can_add = False
                        break
                if not can_add:
                    break

            if can_add:
                # Update residue tracking
                S_list = sorted(S)
                for s in S_list:
                    for q in range(2, 17):
                        mod_residues[q].add((x + s) % q)
                for q in range(2, 17):
                    mod_residues[q].add((2 * x) % q)
                S.add(x)

        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)

    # Strategy 8: Hybrid - start with large evens, add odds strategically
    if time.time() < deadline - 5.0:
        S = set(range(130, n + 1, 2))  # Large evens
        if not is_valid_set(S):
            S_list = sorted(S)
            valid_S = set()
            for x in S_list:
                if can_add_full(valid_S, x):
                    valid_S.add(x)
            S = valid_S

        # Add large odds
        candidates = [x for x in range(130, n + 1, 2) if x not in S]
        rng = random.Random(77777)
        rng.shuffle(candidates)
        for x in candidates:
            if time.time() > deadline - 3.0:
                break
            if can_add_full(S, x):
                S.add(x)

        # Add smaller elements
        candidates2 = [x for x in range(1, 130) if x not in S]
        rng.shuffle(candidates2)
        for x in candidates2:
            if time.time() > deadline - 3.0:
                break
            if can_add_full(S, x):
                S.add(x)

        if time.time() < deadline - 5.0:
            S = local_search(S, rng, deadline - 3.0)

        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)

    if best_size == 0:
        return list(range(129, 257))

    if not is_valid_set(best_S):
        S = set(range(129, 257))
        return sorted(S)

    return sorted(best_S)
