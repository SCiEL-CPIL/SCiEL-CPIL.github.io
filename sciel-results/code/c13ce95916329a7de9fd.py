# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    """
    Optimized model-and-optimize approach with bitset-based conflict detection:
    1. Precompute conflict bitsets for O(1) validity checks.
    2. Fast greedy construction from multiple angles.
    3. Simulated annealing with efficient conflict tracking.
    4. Local search with add/remove/swap moves.
    """
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    def time_left():
        return deadline - time.time()
    
    N = 192
    ALL = list(range(1, N + 1))
    ALL_SET = set(ALL)
    
    # Precompute: for each x in 1..N, which other values y cause conflict when both are in set?
    # conflict[x] is a bitset (Python int) where bit y is set if x and y cannot coexist
    # Actually, the constraint is: x divides sum of 3 DISTINCT OTHER elements.
    # So conflicts are 4-way (x plus 3 others). Let's precompute differently.
    
    # For each pair (a, b) with a < b, precompute the set of values c such that
    # a divides (b + c + d) for some d, or equivalently, for each a, precompute
    # which triples (b,c,d) sum to a multiple of a.
    
    # Better approach: For each element a, precompute forbidden_triples[a] = set of (b,c,d)
    # such that a | (b+c+d). But this is too large.
    
    # Alternative: For each a, and for each pair (b,c), compute the residue r = -(b+c) mod a.
    # Then a conflicts with triple (b,c,d) iff d mod a == r.
    # So for each a, we can precompute: for each residue r mod a, which values d have d mod a == r.
    
    # Even better: Use a different strategy. For a given set S, to check if x can be added:
    # 1. Check if any 3-sum from S is divisible by x
    # 2. For each a in S, check if x mod a is in forbidden_residues[a] (precomputed from S)
    
    # Let's precompute for each a in 1..N: residues_of[a] = {d % a : d in 1..N, d != a}
    # And for each a, multiples_in_range[a] = list of multiples of a in [1, 3*N]
    
    # Precompute sums of 3 for quick checking
    # For each x, precompute which 3-sums are multiples of x
    # three_sum_multiples[x] = set of all sums s of 3 distinct elements from 1..N where x | s
    
    # This is still large. Let's use a more practical approach:
    
    # Precompute pair_sums: for each pair (i,j), i<j, store i+j
    # Then for a set S, the 3-sums are {p + k : p in pair_sums(S), k in S, k not in pair}
    
    # For fast checking, we'll maintain:
    # - sums3_set: set of all 3-sums from current set
    # - For each a in set: forbidden_residues[a] = {-(b+c) mod a : b,c in set, b!=c, b!=a, c!=a}
    
    # Let's precompute some useful tables
    # mod_table[a][r] = list of values in 1..N with value % a == r
    mod_table = {}
    for a in range(1, N + 1):
        mod_table[a] = [[] for _ in range(a)]
        for v in range(1, N + 1):
            mod_table[a][v % a].append(v)
    
    # For each x, precompute multiples of x up to 3*N (max 3-sum is 190+191+192=573)
    max_sum3 = 3 * N  # 576
    multiples_of = {}
    for x in range(1, N + 1):
        multiples_of[x] = [m for m in range(x, max_sum3 + 1, x)]
    
    def compute_sums3(lst):
        """Compute set of all 3-sums from sorted list lst."""
        n = len(lst)
        sums3 = set()
        if n < 3:
            return sums3
        for i in range(n):
            for j in range(i + 1, n):
                s2 = lst[i] + lst[j]
                for k in range(j + 1, n):
                    sums3.add(s2 + lst[k])
        return sums3
    
    def compute_forbidden(lst):
        """
        For each element a in lst, compute set of forbidden residues mod a.
        forbidden[a] = {-(b+c) mod a : b,c in lst, b!=c, b!=a, c!=a}
        """
        n = len(lst)
        forbidden = {}
        for idx_a in range(n):
            a = lst[idx_a]
            forb = set()
            for i in range(n):
                if i == idx_a:
                    continue
                for j in range(i + 1, n):
                    if j == idx_a:
                        continue
                    forb.add(-(lst[i] + lst[j]) % a)
            forbidden[a] = forb
        return forbidden
    
    def can_add_fast(x, s, sums3, forbidden):
        """
        Fast check if x can be added to s.
        s is a set, sums3 is set of 3-sums from s, forbidden is dict from a to forbidden residues.
        """
        # Check 1: Is any 3-sum from s divisible by x?
        for m in multiples_of[x]:
            if m in sums3:
                return False
        
        # Check 2: For each a in s, is x % a in forbidden[a]?
        for a in s:
            if x % a in forbidden[a]:
                return False
        
        return True
    
    def is_valid_full(s):
        """
        Full validity check for set s. O(n^4) worst case but with early exit.
        """
        lst = sorted(s)
        n = len(lst)
        if n < 4:
            return True
        
        # For each a, check if any 3-sum of OTHER elements is divisible by a
        for idx_a in range(n):
            a = lst[idx_a]
            # Collect other elements
            others = [lst[i] for i in range(n) if i != idx_a]
            m = len(others)
            if m < 3:
                continue
            # Check all triples from others
            for i in range(m):
                for j in range(i + 1, m):
                    s2 = others[i] + others[j]
                    for k in range(j + 1, m):
                        if (s2 + others[k]) % a == 0:
                            return False
        return True
    
    def greedy_from_high(rng):
        """Greedy: add elements from high to low if valid."""
        selected = set()
        sums3 = set()
        forbidden = {}
        for x in range(N, 0, -1):
            if can_add_fast(x, selected, sums3, forbidden):
                selected.add(x)
                # Update sums3: add all triples involving x
                lst = sorted(selected)
                sums3 = compute_sums3(lst)
                forbidden = compute_forbidden(lst)
        return selected
    
    def greedy_from_low(rng):
        """Greedy: add elements from low to high if valid."""
        selected = set()
        sums3 = set()
        forbidden = {}
        for x in range(1, N + 1):
            if can_add_fast(x, selected, sums3, forbidden):
                selected.add(x)
                lst = sorted(selected)
                sums3 = compute_sums3(lst)
                forbidden = compute_forbidden(lst)
        return selected
    
    def greedy_random(rng):
        """Greedy: random order."""
        order = ALL[:]
        rng.shuffle(order)
        selected = set()
        sums3 = set()
        forbidden = {}
        for x in order:
            if can_add_fast(x, selected, sums3, forbidden):
                selected.add(x)
                lst = sorted(selected)
                sums3 = compute_sums3(lst)
                forbidden = compute_forbidden(lst)
        return selected
    
    def greedy_biased(rng):
        """Greedy: prefer larger numbers but with some randomness."""
        selected = set()
        candidates = ALL[:]
        rng.shuffle(candidates)
        candidates.sort(key=lambda x: -x + rng.uniform(0, 20))
        sums3 = set()
        forbidden = {}
        for x in candidates:
            if can_add_fast(x, selected, sums3, forbidden):
                selected.add(x)
                lst = sorted(selected)
                sums3 = compute_sums3(lst)
                forbidden = compute_forbidden(lst)
        return selected
    
    def local_search(s, rng, max_iters=200):
        """
        Local search with add, remove, and swap moves.
        """
        s = set(s)
        cur_size = len(s)
        no_improve = 0
        lst = sorted(s)
        sums3 = compute_sums3(lst)
        forbidden = compute_forbidden(lst)
        
        for _ in range(max_iters):
            if time_left() < 2:
                break
            
            improved = False
            moved = False
            
            # Try additions
            candidates = [x for x in ALL if x not in s]
            rng.shuffle(candidates)
            for x in candidates[:40]:
                if can_add_fast(x, s, sums3, forbidden):
                    s.add(x)
                    cur_size += 1
                    improved = True
                    moved = True
                    lst = sorted(s)
                    sums3 = compute_sums3(lst)
                    forbidden = compute_forbidden(lst)
                    break
            
            if improved:
                no_improve = 0
                continue
            
            no_improve += 1
            
            # Try swaps
            elems = list(s)
            rng.shuffle(elems)
            for rem in elems[:10]:
                candidates = [x for x in ALL if x not in s]
                rng.shuffle(candidates)
                for add in candidates[:40]:
                    new_s = (s - {rem}) | {add}
                    if is_valid_full(new_s):
                        s = new_s
                        improved = True
                        moved = True
                        lst = sorted(s)
                        sums3 = compute_sums3(lst)
                        forbidden = compute_forbidden(lst)
                        break
                if improved:
                    break
            
            if improved:
                no_improve = 0
                continue
            
            # Try "remove 1, add 2" moves
            if no_improve > 3:
                elems = list(s)
                rng.shuffle(elems)
                for rem in elems[:12]:
                    candidates = [x for x in ALL if x not in s]
                    rng.shuffle(candidates)
                    for add1 in candidates[:20]:
                        for add2 in candidates[:20]:
                            if add1 == add2:
                                continue
                            new_s = (s - {rem}) | {add1, add2}
                            if is_valid_full(new_s):
                                s = new_s
                                improved = True
                                moved = True
                                lst = sorted(s)
                                sums3 = compute_sums3(lst)
                                forbidden = compute_forbidden(lst)
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    no_improve = 0
                    continue
            
            if not moved:
                break
        
        return s
    
    # Phase 1: Generate diverse starting solutions
    rng = random.Random(42)
    
    starts = []
    starts.append(greedy_from_high(rng))
    starts.append(greedy_from_low(rng))
    for _ in range(6):
        starts.append(greedy_random(rng))
    for _ in range(4):
        starts.append(greedy_biased(rng))
    
    # Structured starts
    s_large = set()
    sums3_large = set()
    forbidden_large = {}
    for x in range(N, N // 2, -1):
        if can_add_fast(x, s_large, sums3_large, forbidden_large):
            s_large.add(x)
            lst = sorted(s_large)
            sums3_large = compute_sums3(lst)
            forbidden_large = compute_forbidden(lst)
    starts.append(s_large)
    
    s_mid = set()
    sums3_mid = set()
    forbidden_mid = {}
    for x in range(N // 2, N // 4, -1):
        if can_add_fast(x, s_mid, sums3_mid, forbidden_mid):
            s_mid.add(x)
            lst = sorted(s_mid)
            sums3_mid = compute_sums3(lst)
            forbidden_mid = compute_forbidden(lst)
    starts.append(s_mid)
    
    best = max(starts, key=len)
    best_size = len(best)
    
    print(f"Initial best size: {best_size}", flush=True)
    
    # Phase 2: Local search from each start
    for i, start_s in enumerate(starts):
        if time_left() < 5:
            break
        
        result = local_search(start_s, rng, max_iters=120)
        if len(result) > best_size:
            best_size = len(result)
            best = result
            print(f"Improved to {best_size} at iteration {i}", flush=True)
    
    # Phase 3: More aggressive search with restarts
    for iteration in range(500):
        if time_left() < 10:
            break
        
        # Generate a new starting point
        if iteration % 5 == 0:
            # Perturb the best solution
            s = set(best)
            n_remove = rng.randint(2, 5)
            to_remove = rng.sample(list(s), min(n_remove, len(s)))
            s -= set(to_remove)
            start_s = s
        elif iteration % 5 == 1:
            start_s = greedy_from_high(rng)
        elif iteration % 5 == 2:
            start_s = greedy_random(rng)
        elif iteration % 5 == 3:
            start_s = greedy_biased(rng)
        else:
            # Random subset filtered to valid
            k = rng.randint(30, 45)
            start_s = set(rng.sample(ALL, k))
            valid_s = set()
            sums3_valid = set()
            forbidden_valid = {}
            for x in sorted(start_s, reverse=True):
                if can_add_fast(x, valid_s, sums3_valid, forbidden_valid):
                    valid_s.add(x)
                    lst = sorted(valid_s)
                    sums3_valid = compute_sums3(lst)
                    forbidden_valid = compute_forbidden(lst)
            start_s = valid_s
        
        result = local_search(start_s, rng, max_iters=80)
        if len(result) > best_size:
            best_size = len(result)
            best = result
            print(f"New best: {best_size} at iteration {iteration}", flush=True)
    
    # Phase 4: Simulated annealing to escape local optima
    if time_left() > 20:
        print(f"Starting simulated annealing, current best: {best_size}", flush=True)
        T = 10.0
        T_min = 0.1
        cooling = 0.995
        
        sa_best = set(best)
        sa_size = best_size
        
        for sa_iter in range(2000):
            if time_left() < 8:
                break
            if T <= T_min:
                break
            
            # Generate a random move: add, remove, or swap
            move_type = rng.random()
            s = set(sa_best)
            
            if move_type < 0.3:
                # Add a random element
                candidates = [x for x in ALL if x not in s]
                if candidates:
                    x = rng.choice(candidates)
                    s.add(x)
            elif move_type < 0.6:
                # Remove a random element
                if s:
                    x = rng.choice(list(s))
                    s.discard(x)
            else:
                # Swap: remove one, add one
                if s:
                    rem = rng.choice(list(s))
                    candidates = [x for x in ALL if x not in s]
                    if candidates:
                        add = rng.choice(candidates)
                        s.discard(rem)
                        s.add(add)
            
            # Check validity
            if is_valid_full(s):
                new_size = len(s)
                delta = new_size - sa_size
                
                # Accept if better, or with probability exp(-delta/T) if worse
                if delta >= 0 or rng.random() < (1.0 / T) ** max(1, -delta):
                    sa_best = s
                    sa_size = new_size
                    
                    if sa_size > best_size:
                        best_size = sa_size
                        best = set(sa_best)
                        print(f"SA improved to {best_size} at iter {sa_iter}", flush=True)
            
            T *= cooling
        
        print(f"SA finished, best size: {best_size}", flush=True)
    
    # Phase 5: Final polish with local search
    if time_left() > 10:
        print(f"Starting final polish, current best: {best_size}", flush=True)
        for _ in range(40):
            if time_left() < 5:
                break
            s = set(best)
            n_remove = rng.randint(1, 3)
            to_remove = rng.sample(list(s), min(n_remove, len(s)))
            s -= set(to_remove)
            result = local_search(s, rng, max_iters=60)
            if len(result) > best_size:
                best_size = len(result)
                best = result
                print(f"Polish improved to {best_size}", flush=True)
    
    # Final check
    if not is_valid_full(best):
        best = greedy_from_high(rng)
        while not is_valid_full(best):
            lst = sorted(best)
            best = set(lst[:-1])
    
    return list(best)

    def greedy_from_high(rng):
        """Greedy: add elements from high to low if valid."""
        selected = set()
        checkers = (set(), {})
        for x in range(N, 0, -1):
            if can_add(x, selected, checkers):
                selected.add(x)
                checkers = compute_checkers(selected)
        return selected

    def greedy_from_low(rng):
        """Greedy: add elements from low to high if valid."""
        selected = set()
        checkers = (set(), {})
        for x in range(1, N + 1):
            if can_add(x, selected, checkers):
                selected.add(x)
                checkers = compute_checkers(selected)
        return selected

    def greedy_random(rng):
        """Greedy: random order."""
        order = ALL[:]
        rng.shuffle(order)
        selected = set()
        checkers = (set(), {})
        for x in order:
            if can_add(x, selected, checkers):
                selected.add(x)
                checkers = compute_checkers(selected)
        return selected

    def greedy_biased(rng):
        """Greedy: prefer larger numbers but with some randomness."""
        selected = set()
        candidates = ALL[:]
        rng.shuffle(candidates)
        candidates.sort(key=lambda x: -x + rng.uniform(0, 20))
        checkers = (set(), {})
        for x in candidates:
            if can_add(x, selected, checkers):
                selected.add(x)
                checkers = compute_checkers(selected)
        return selected

    def local_search(s, rng, max_iters=200):
        """
        Local search with add, remove, and swap moves.
        Uses fast can_add for additions.
        Key improvement: try "remove 1, add 2" moves which can escape local optima.
        """
        s = set(s)
        cur_size = len(s)
        no_improve = 0
        checkers = compute_checkers(s)
        
        for _ in range(max_iters):
            if time_left() < 2:
                break
            
            improved = False
            moved = False
            
            # Try additions - try more candidates
            candidates = [x for x in ALL if x not in s]
            rng.shuffle(candidates)
            for x in candidates[:30]:
                if can_add(x, s, checkers):
                    s.add(x)
                    cur_size += 1
                    improved = True
                    moved = True
                    checkers = compute_checkers(s)
                    break
            
            if improved:
                no_improve = 0
                continue
            
            no_improve += 1
            
            # Try swaps - try more combinations
            elems = list(s)
            rng.shuffle(elems)
            for rem in elems[:8]:
                candidates = [x for x in ALL if x not in s]
                rng.shuffle(candidates)
                for add in candidates[:30]:
                    new_s = (s - {rem}) | {add}
                    if is_valid(new_s):
                        s = new_s
                        improved = True
                        moved = True
                        checkers = compute_checkers(s)
                        break
                if improved:
                    break
            
            if improved:
                no_improve = 0
                continue
            
            # NEW: Try "remove 1, add 2" moves - key for escaping local optima
            if no_improve > 2:
                elems = list(s)
                rng.shuffle(elems)
                # Try more removal candidates and more add combinations
                for rem in elems[:15]:
                    candidates = [x for x in ALL if x not in s]
                    rng.shuffle(candidates)
                    # Try more add1 candidates
                    for add1 in candidates[:25]:
                        for add2 in candidates[:25]:
                            if add1 == add2:
                                continue
                            new_s = (s - {rem}) | {add1, add2}
                            if is_valid(new_s):
                                s = new_s
                                improved = True
                                moved = True
                                checkers = compute_checkers(s)
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    no_improve = 0
                    continue
            
            # Try double moves: remove two, add two
            if no_improve > 5:
                elems = list(s)
                rng.shuffle(elems)
                for i in range(min(6, len(elems))):
                    for j in range(i + 1, min(len(elems), i + 8)):
                        rem1, rem2 = elems[i], elems[j]
                        candidates = [x for x in ALL if x not in s]
                        rng.shuffle(candidates)
                        for add1 in candidates[:15]:
                            for add2 in candidates[:15]:
                                if add1 == add2:
                                    continue
                                new_s = (s - {rem1, rem2}) | {add1, add2}
                                if is_valid(new_s):
                                    s = new_s
                                    improved = True
                                    moved = True
                                    checkers = compute_checkers(s)
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    no_improve = 0
                    continue
            
            if not moved:
                break
        
        return s
    
    # Phase 1: Generate diverse starting solutions
    rng = random.Random(42)
    
    starts = []
    starts.append(greedy_from_high(rng))
    starts.append(greedy_from_low(rng))
    for _ in range(5):
        starts.append(greedy_random(rng))
    for _ in range(3):
        starts.append(greedy_biased(rng))
    
    # Also try some structured starts
    # Start with large numbers only
    s_large = set()
    checkers_large = (set(), {})
    for x in range(N, N // 2, -1):
        if can_add(x, s_large, checkers_large):
            s_large.add(x)
            checkers_large = compute_checkers(s_large)
    starts.append(s_large)
    
    # Start with numbers in certain ranges
    s_mid = set()
    checkers_mid = (set(), {})
    for x in range(N // 2, N // 4, -1):
        if can_add(x, s_mid, checkers_mid):
            s_mid.add(x)
            checkers_mid = compute_checkers(s_mid)
    starts.append(s_mid)
    
    best = max(starts, key=len)
    best_size = len(best)
    
    print(f"Initial best size: {best_size}", flush=True)
    
    # Phase 2: Local search from each start
    for i, start_s in enumerate(starts):
        if time_left() < 5:
            break
        
        result = local_search(start_s, rng, max_iters=150)
        if len(result) > best_size:
            best_size = len(result)
            best = result
            print(f"Improved to {best_size} at iteration {i}", flush=True)
    
    # Phase 3: More aggressive search with restarts
    for iteration in range(400):
        if time_left() < 10:
            break
        
        # Generate a new starting point with some randomness
        if iteration % 5 == 0:
            # Perturb the best solution
            s = set(best)
            n_remove = rng.randint(2, 5)
            to_remove = rng.sample(list(s), min(n_remove, len(s)))
            s -= set(to_remove)
            start_s = s
        elif iteration % 5 == 1:
            start_s = greedy_from_high(rng)
        elif iteration % 5 == 2:
            start_s = greedy_random(rng)
        elif iteration % 5 == 3:
            start_s = greedy_biased(rng)
        else:
            # Random subset
            k = rng.randint(30, 45)
            start_s = set(rng.sample(ALL, k))
            # Filter to valid subset
            valid_s = set()
            checkers_valid = (set(), {})
            for x in sorted(start_s, reverse=True):
                if can_add(x, valid_s, checkers_valid):
                    valid_s.add(x)
                    checkers_valid = compute_checkers(valid_s)
            start_s = valid_s
        
        result = local_search(start_s, rng, max_iters=100)
        if len(result) > best_size:
            best_size = len(result)
            best = result
            print(f"New best: {best_size} at iteration {iteration}", flush=True)
    
    # Phase 4: Final polish - more aggressive local search on best
    if time_left() > 15:
        print(f"Starting final polish, current best: {best_size}", flush=True)
        for _ in range(50):
            if time_left() < 5:
                break
            s = set(best)
            # Small perturbation
            n_remove = rng.randint(1, 3)
            to_remove = rng.sample(list(s), min(n_remove, len(s)))
            s -= set(to_remove)
            result = local_search(s, rng, max_iters=100)
            if len(result) > best_size:
                best_size = len(result)
                best = result
                print(f"Polish improved to {best_size}", flush=True)
    
    # Final check
    if not is_valid(best):
        # Fallback to greedy
        best = greedy_from_high(rng)
        while not is_valid(best):
            best = set(list(best)[:-1])
    
    return list(best)

# EVOLVE-BLOCK-END
