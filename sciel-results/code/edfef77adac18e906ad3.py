"""Construct a small inclusion-maximal Sidon set in [1,256]."""
import time
import random

N = 256

# EVOLVE-BLOCK-START

def construct():
    """Return an inclusion-maximal Sidon set in [1,256] with small cardinality."""
    deadline = time.time() + 27.0  # Safety margin before 30s limit
    
    # Precompute for fast operations
    MAX_SUM = 2 * N  # 512
    
    def can_add_fast(x, sums_bitset, S):
        """Check if x can be added to S given sums as a bitset."""
        if (sums_bitset >> (2 * x)) & 1:
            return False
        for a in S:
            if (sums_bitset >> (x + a)) & 1:
                return False
        return True

    def get_sums_bitset(S):
        """Compute the sum bitset for set S."""
        sums = 0
        n = len(S)
        for i in range(n):
            for j in range(i, n):
                sums |= (1 << (S[i] + S[j]))
        return sums

    def is_maximal_fast(S, sums_bitset=None):
        """Check if S is inclusion-maximal using bitset for sums."""
        if not S:
            return False
        if sums_bitset is None:
            sums_bitset = get_sums_bitset(S)
        S_set = set(S)
        for x in range(1, N + 1):
            if x in S_set:
                continue
            if (sums_bitset >> (2 * x)) & 1:
                continue
            addable = True
            for a in S:
                if (sums_bitset >> (x + a)) & 1:
                    addable = False
                    break
            if addable:
                return False
        return True

    def greedy_construct(order):
        """Greedy construction with given order."""
        S = []
        sums = 0  # bitset
        for x in order:
            if can_add_fast(x, sums, S):
                S.append(x)
                sums |= (1 << (2 * x))
                for a in S[:-1]:
                    sums |= (1 << (x + a))
        return S

    def prune_aggressive(S):
        """Aggressively prune: try removing each element, check maximality."""
        S = list(S)
        improved = True
        while improved:
            improved = False
            if time.time() > deadline:
                return S
            # Try removing each element, starting from ones that seem most removable
            # Sort by "removability" - elements whose removal still leaves maximality
            for i in range(len(S) - 1, -1, -1):
                if time.time() > deadline:
                    return S
                new_S = S[:i] + S[i+1:]
                if not new_S:
                    continue
                new_sums = get_sums_bitset(new_S)
                if is_maximal_fast(new_S, new_sums):
                    S = new_S
                    improved = True
                    break  # Restart with new indices
        return S

    def local_search(S, max_iters=100):
        """Local search: try swapping elements to reduce size while maintaining maximality."""
        S = list(S)
        best = S
        best_size = len(S)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            # Try removing a random element
            if len(S) <= 1:
                break
            i = random.Random(len(S) * 7 + _).randint(0, len(S) - 1)
            new_S = S[:i] + S[i+1:]
            if new_S:
                new_sums = get_sums_bitset(new_S)
                if is_maximal_fast(new_S, new_sums):
                    S = new_S
                    if len(S) < best_size:
                        best_size = len(S)
                        best = list(S)
        
        return best

    # Phase 1: Multiple deterministic orders
    orders = [
        list(range(N, 0, -1)),  # descending
        list(range(1, N + 1)),  # ascending
        list(range(N // 2, 0, -1)) + list(range(N // 2 + 1, N + 1)),
        list(range(1, N + 1, 2)) + list(range(2, N + 1, 2)),
        list(range(2, N + 1, 2)) + list(range(1, N + 1, 2)),
        # Interleaved high-low
        [x for x in range(1, N + 1) if x % 3 != 0] + [x for x in range(1, N + 1) if x % 3 == 0],
        # Powers of 2 first, then rest
        [2**i for i in range(1, 9) if 2**i <= N] + [x for x in range(1, N + 1) if x not in [2**i for i in range(1, 9) if 2**i <= N]],
    ]
    
    best_set = None
    best_size = N + 1
    
    for order in orders:
        if time.time() > deadline:
            break
        S = greedy_construct(order)
        S = prune_aggressive(S)
        if len(S) < best_size:
            best_size = len(S)
            best_set = S
    
    # Phase 2: Randomized restarts with local search
    rng = random.Random(42)
    all_elements = list(range(1, N + 1))
    
    iterations = 0
    while time.time() < deadline - 2.0:  # Leave 2s for final checks
        iterations += 1
        # Various random order strategies
        r = rng.random()
        if r < 0.25:
            order = list(all_elements)
            rng.shuffle(order)
        elif r < 0.4:
            # Bias toward larger numbers
            order = sorted(all_elements, key=lambda x: -x + rng.gauss(0, 80))
        elif r < 0.55:
            # Bias toward smaller numbers
            order = sorted(all_elements, key=lambda x: x + rng.gauss(0, 80))
        elif r < 0.7:
            # Weighted: favor numbers with fewer divisors (harder to block)
            import math
            def num_divisors(n):
                count = 0
                for i in range(1, int(n**0.5) + 1):
                    if n % i == 0:
                        count += 1
                        if i != n // i:
                            count += 1
                return count
            order = sorted(all_elements, key=lambda x: num_divisors(x) + rng.gauss(0, 3))
        else:
            # Random with occasional reversal
            order = list(all_elements)
            rng.shuffle(order)
            if rng.random() < 0.3:
                order.reverse()
        
        S = greedy_construct(order)
        S = prune_aggressive(S)
        
        # Local search on the pruned set
        if len(S) < best_size:
            S_ls = local_search(S, max_iters=20)
            if len(S_ls) < best_size:
                best_size = len(S_ls)
                best_set = S_ls
        elif len(S) == best_size:
            # Even if same size, local search might find better structure
            S_ls = local_search(S, max_iters=10)
            if len(S_ls) < best_size:
                best_size = len(S_ls)
                best_set = S_ls
    
    # Final: try more aggressive local search on best found
    if best_set is None:
        best_set = greedy_construct(list(range(N, 0, -1)))
    
    # One more round of local search with more iterations
    if time.time() < deadline - 1.0:
        final_ls = local_search(best_set, max_iters=50)
        if len(final_ls) < len(best_set):
            best_set = final_ls
    
    # Final verification
    if not is_maximal_fast(best_set):
        # Fallback to a known valid construction
        best_set = greedy_construct(list(range(N, 0, -1)))
    
    return best_set

# EVOLVE-BLOCK-END
