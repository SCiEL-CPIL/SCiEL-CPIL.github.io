"""Program: large 3-sum-free-pair-sum set in 1..128 via bit-parallel greedy independent set."""
# EVOLVE-BLOCK-START
import random

N = 128

def construct():
    # We want a large subset S of {1..128} such that there do NOT exist three
    # distinct a,b,c in S with a+b, a+c, b+c all in S.
    #
    # Bit-parallel greedy independent set on the 3-uniform hypergraph where
    # hyperedges are triples whose pair sums fall in the domain.
    # Use randomized vertex ordering with multiple restarts.

    best = None
    best_len = 0

    rng = random.Random(12345)

    def can_add(x, S_mask, S_set):
        # Check if adding x to S creates a forbidden triple.
        # A forbidden triple {x, a, b} with a < b in S requires:
        #   x+a in S, x+b in S, and (a+b in S or a+b == x)
        # Since x, a, b >= 1 and distinct, x+a > x, x+b > x, so x+a != x, x+b != x.
        # We need x+a in S_set, x+b in S_set, and (a+b in S_set or a+b == x).
        
        elems = list(S_set)
        n = len(elems)
        for i in range(n):
            a = elems[i]
            xa = x + a
            if xa > 128:
                continue
            if xa not in S_set:
                continue
            for j in range(i+1, n):
                b = elems[j]
                xb = x + b
                if xb > 128:
                    continue
                if xb not in S_set:
                    continue
                ab = a + b
                if ab == x or ab in S_set:
                    return False
        return True

    # Try multiple strategies: random permutations and sorted orders
    for trial in range(3000):
        if trial % 2 == 0:
            perm = list(range(1, 129))
            rng.shuffle(perm)
        else:
            # Try reverse-sorted or other deterministic orders
            perm = list(range(128, 0, -1))
            if trial % 4 == 1:
                # Slightly perturb reverse order
                idx = rng.randint(0, 127)
                perm[idx], perm[min(idx+1, 127)] = perm[min(idx+1, 127)], perm[idx]

        S_set = set()
        S_mask = 0

        for x in perm:
            if can_add(x, S_mask, S_set):
                S_set.add(x)
                S_mask |= (1 << x)

        if len(S_set) > best_len:
            best_len = len(S_set)
            best = list(S_set)

        if best_len >= 85:  # early stop if we get very large
            break

    # Verify best
    def is_valid(S):
        S_set = set(S)
        S_list = list(S)
        for i in range(len(S_list)):
            for j in range(i+1, len(S_list)):
                for k in range(j+1, len(S_list)):
                    a, b, c = S_list[i], S_list[j], S_list[k]
                    if (a+b in S_set) and (a+c in S_set) and (b+c in S_set):
                        return False
        return True

    if best is None:
        best = list(range(65, 129))

    if not is_valid(best):
        # Fallback to safe construction: top half
        best = list(range(65, 129))

    return list(best)
# EVOLVE-BLOCK-END
