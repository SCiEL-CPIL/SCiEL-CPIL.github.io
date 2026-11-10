import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s limit

    n = 512
    nodes = list(range(1, n + 1))

    # 1. Build adjacency bitsets
    adj = [0] * (n + 1)

    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            g = gcd(i, j)
            l = (i // g) * j
            if l <= n:
                adj[i] |= (1 << (j - 1))
                adj[j] |= (1 << (i - 1))

        if time.time() > deadline:
            return [1, 2]

    # 2. Generate initial best clique from divisors of highly composite numbers
    def get_divisors(L):
        divs = []
        for k in range(1, int(L**0.5) + 1):
            if L % k == 0:
                divs.append(k)
                if k != L // k:
                    divs.append(L // k)
        return divs

    # Find L with max divisors
    best_div_count = 0
    best_L = 1
    for L in range(1, n + 1):
        d = get_divisors(L)
        if len(d) > best_div_count:
            best_div_count = len(d)
            best_L = L

    best_seed = get_divisors(best_L)
    best_clique_size = len(best_seed)
    best_clique = best_seed

    # 3. Bron-Kerbosch with bitsets and improved pivot
    def bron_kerbosch(R, P, X):
        nonlocal best_clique, best_clique_size

        if time.time() > deadline:
            return

        # Pruning
        if len(R) + P.bit_count() <= best_clique_size:
            return

        if P == 0:
            if X == 0:
                if len(R) > best_clique_size:
                    best_clique_size = len(R)
                    best_clique = list(R)
            return

        # Pivot selection: choose u in P | X that maximizes (P & adj[u]).bit_count()
        pivot = -1
        max_count = -1

        px = P | X
        while px:
            lsb = px & -px
            u = lsb.bit_length()
            count = (P & adj[u]).bit_count()
            if count > max_count:
                max_count = count
                pivot = u
            px ^= lsb

        if pivot == -1:
            return

        # Candidates: P & ~adj[pivot]
        candidates = P & ~adj[pivot]

        while candidates:
            lsb = candidates & -candidates
            v = lsb.bit_length()

            new_P = P & adj[v]
            new_X = X & adj[v]

            R.append(v)
            bron_kerbosch(R, new_P, new_X)
            R.pop()

            P ^= lsb
            X |= lsb
            candidates ^= lsb

            if time.time() > deadline:
                return

    # 4. Greedy improvement: try to add nodes to best_clique using bitsets
    # Sort candidates by compatibility (fewest conflicts first)
    def greedy_improve(clique_set):
        improved_set = set(clique_set)
        changed = True
        while changed and time.time() < deadline - 2.0:
            changed = False
            # Find all candidates not in set
            candidates = [v for v in range(1, n + 1) if v not in improved_set]
            # Sort by number of conflicts (ascending)
            candidates.sort(key=lambda v: sum(1 for u in improved_set if not ((adj[v] >> (u - 1)) & 1)))
            for v in candidates:
                # Check if v is compatible with all in improved
                compat_with_all = True
                for u in improved_set:
                    if not (adj[v] >> (u - 1)) & 1:
                        compat_with_all = False
                        break
                if compat_with_all:
                    improved_set.add(v)
                    changed = True
                    break  # Restart scan after adding
        return improved_set

    # Run BK
    initial_P = (1 << n) - 1
    initial_X = 0
    initial_R = []

    bron_kerbosch(initial_R, initial_P, initial_X)

    # Apply greedy improvement
    if time.time() < deadline - 2.0:
        improved = greedy_improve(set(best_clique))
        if len(improved) > best_clique_size:
            best_clique_size = len(improved)
            best_clique = list(improved)

    # 5. Multi-start: try BK from different initial seeds
    # Use divisors of several highly composite numbers as starting points
    highly_composite = [480, 504, 420, 360, 336, 288, 240, 216, 180, 168, 144, 120, 96, 72, 60, 48, 40, 36, 30, 24, 20, 18, 16, 15, 14, 12, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    
    for L in highly_composite:
        if time.time() > deadline - 5.0:
            break
        seed_set = set(get_divisors(L))
        # Greedy improve from this seed
        improved = greedy_improve(seed_set)
        if len(improved) > best_clique_size:
            best_clique_size = len(improved)
            best_clique = list(improved)
        # Run BK from this seed's complement
        if time.time() > deadline - 3.0:
            break
        # Build P as nodes compatible with all in seed_set
        P = 0
        for v in range(1, n + 1):
            if v in seed_set:
                continue
            compat = True
            for u in seed_set:
                if not ((adj[v] >> (u - 1)) & 1):
                    compat = False
                    break
            if compat:
                P |= (1 << (v - 1))
        
        R = list(seed_set)
        bron_kerbosch(R, P, 0)
        # Greedy improve again
        if time.time() < deadline - 2.0:
            improved = greedy_improve(set(best_clique))
            if len(improved) > best_clique_size:
                best_clique_size = len(improved)
                best_clique = list(improved)

    # 6. Local search: try removing one element and adding two compatible ones
    if time.time() < deadline - 3.0:
        improved_set = set(best_clique)
        best_local_size = len(improved_set)
        
        for _ in range(50):
            if time.time() > deadline - 1.0:
                break
            
            progress = False
            for rem in list(improved_set):
                if len(improved_set) - 1 + 2 <= best_local_size:
                    continue
                
                candidate_set = improved_set - {rem}
                
                # Find compatible elements to add
                addable = []
                for v in range(1, n + 1):
                    if v in candidate_set:
                        continue
                    compat = True
                    for u in candidate_set:
                        if not (adj[v] >> (u - 1)) & 1:
                            compat = False
                            break
                    if compat:
                        addable.append(v)
                
                if len(addable) < 2:
                    continue
                
                # Try pairs from addable
                found_pair = False
                for i in range(min(len(addable), 30)):
                    for j in range(i + 1, min(len(addable), 30)):
                        a, b = addable[i], addable[j]
                        if (adj[a] >> (b - 1)) & 1:
                            new_set = candidate_set | {a, b}
                            if len(new_set) > best_local_size:
                                improved_set = new_set
                                best_local_size = len(new_set)
                                progress = True
                                found_pair = True
                                break
                    if found_pair:
                        break
            
            if not progress:
                break
        
        if best_local_size > best_clique_size:
            best_clique_size = best_local_size
            best_clique = list(improved_set)

    # 7. Swap local search: remove one, add one
    if time.time() < deadline - 2.0:
        improved_set = set(best_clique)
        best_swap_size = len(improved_set)
        
        for _ in range(30):
            if time.time() > deadline - 1.0:
                break
            
            progress = False
            for rem in list(improved_set):
                candidate_set = improved_set - {rem}
                
                addable = []
                for v in range(1, n + 1):
                    if v in candidate_set:
                        continue
                    compat = True
                    for u in candidate_set:
                        if not (adj[v] >> (u - 1)) & 1:
                            compat = False
                            break
                    if compat:
                        addable.append(v)
                
                if not addable:
                    continue
                
                for v in addable:
                    new_set = candidate_set | {v}
                    if len(new_set) > best_swap_size:
                        improved_set = new_set
                        best_swap_size = len(new_set)
                        progress = True
                        break
            
            if not progress:
                break
        
        if best_swap_size > best_clique_size:
            best_clique_size = best_swap_size
            best_clique = list(improved_set)

    # Final validation
    best_clique = list(set(best_clique))
    for i in range(len(best_clique)):
        for j in range(i + 1, len(best_clique)):
            a, b = best_clique[i], best_clique[j]
            if (a // gcd(a, b)) * b > n:
                # Invalid, fall back to divisors of best_L
                return get_divisors(best_L)

    return best_clique
