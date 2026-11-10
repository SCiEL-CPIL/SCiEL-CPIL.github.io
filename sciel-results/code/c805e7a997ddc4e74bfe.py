import time
import random
import numpy as np

def construct():
    N = 1000
    start_time = time.time()
    deadline = start_time + 140  # 10 second margin

    best_size = 0
    best_set = []

    def _is_valid_addition(S, x, counts):
        """Check if adding x to S keeps all sum counts <= 2."""
        if x in S:
            return False
        # Check sums x + y for y in S
        for y in S:
            s = x + y
            if counts[s] >= 2:
                return False
        # Check sum 2x
        s = 2 * x
        if counts[s] >= 2:
            return False
        return True

    def _add_to_set(S, x, counts):
        """Add x to S and update counts."""
        for y in S:
            counts[x + y] += 1
        counts[2 * x] += 1
        S.append(x)

    def _remove_from_set(S, x, counts):
        """Remove x from S and update counts."""
        S.remove(x)
        for y in S:
            counts[x + y] -= 1
        counts[2 * x] -= 1

    def greedy_build(order):
        S = []
        counts = [0] * (2 * N + 1)
        for x in order:
            if _is_valid_addition(S, x, counts):
                _add_to_set(S, x, counts)
        return S

    def try_order(order):
        nonlocal best_size, best_set
        S = greedy_build(order)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]

    # 1. Natural order
    try_order(list(range(1, N + 1)))

    # 2. Reverse order
    if time.time() < deadline:
        try_order(list(range(N, 0, -1)))

    # 3. Try several structured orderings
    if time.time() < deadline:
        def is_prime(n):
            if n < 2:
                return False
            for i in range(2, int(n**0.5) + 1):
                if n % i == 0:
                    return False
            return True
        
        primes = [x for x in range(1, N+1) if is_prime(x)]
        composites = [x for x in range(1, N+1) if not is_prime(x)]
        try_order(primes + composites)

    # 4. Sparse pattern: powers of 2 first, then fill in
    if time.time() < deadline:
        powers = [2**k for k in range(10) if 2**k <= N]
        rest = [x for x in range(1, N+1) if x not in set(powers)]
        try_order(powers + rest)

    # 5. Random shuffles with different seeds
    rng = random.Random(42)
    while time.time() < deadline - 30:  # Leave 30s for local search
        order = list(range(1, N + 1))
        rng.shuffle(order)
        try_order(order)
        if best_size >= 750:
            break

    # 6. Local search: try to add elements that fit, then try swaps
    if time.time() < deadline - 5:
        S = best_set[:]
        counts = [0] * (2 * N + 1)
        S_set = set(S)
        for x in S:
            for y in S:
                if x <= y:
                    counts[x + y] += 1

        # Phase 1: Try adding any element that fits
        added = True
        while added and time.time() < deadline - 5:
            added = False
            for x in range(1, N + 1):
                if x not in S_set and _is_valid_addition(S, x, counts):
                    _add_to_set(S, x, counts)
                    S_set.add(x)
                    added = True

        # Phase 2: Try swaps - remove one element, try adding two
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            S_copy = S[:]
            S_set_copy = set(S)
            for i, x in enumerate(S_copy):
                # Try removing x
                _remove_from_set(S, x, counts)
                S_set.discard(x)
                
                # Try adding two new elements
                new_added = 0
                for y in range(1, N + 1):
                    if y not in S_set and _is_valid_addition(S, y, counts):
                        _add_to_set(S, y, counts)
                        S_set.add(y)
                        new_added += 1
                        if new_added == 1:
                            break
                
                if new_added >= 1:
                    improved = True
                else:
                    # Put x back
                    _add_to_set(S, x, counts)
                    S_set.add(x)

        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]

    # 7. Final polish: try to add more elements
    if time.time() < deadline - 2:
        S = best_set[:]
        counts = [0] * (2 * N + 1)
        S_set = set(S)
        for x in S:
            for y in S:
                if x <= y:
                    counts[x + y] += 1

        for x in range(1, N + 1):
            if x not in S_set and _is_valid_addition(S, x, counts):
                _add_to_set(S, x, counts)
                S_set.add(x)

        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]

    return best_set
