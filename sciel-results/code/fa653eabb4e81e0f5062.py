import random
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    N = 128
    # Precompute GCD table
    gcd_table = [[0]*(N+1) for _ in range(N+1)]
    for i in range(1, N+1):
        for j in range(i, N+1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g

    def is_bad_triple(a, b, c):
        """Check if three elements form an equal-gcd triple."""
        return gcd_table[a][b] == gcd_table[b][c] == gcd_table[c][a]

    def can_add(S, x, S_set):
        """Check if x can be added to S without forming an equal-gcd triple."""
        for i in range(len(S)):
            a = S[i]
            g_ax = gcd_table[a][x]
            for j in range(i+1, len(S)):
                b = S[j]
                if g_ax == gcd_table[b][x] == gcd_table[a][b]:
                    return False
        return True

    def greedy_construct(order):
        S = []
        S_set = set()
        for x in order:
            if can_add(S, x, S_set):
                S.append(x)
                S_set.add(x)
        return S, S_set

    def conflict_count(x, S):
        """Count how many pairs in S would form a bad triple with x."""
        count = 0
        for i in range(len(S)):
            a = S[i]
            g_ax = gcd_table[a][x]
            for j in range(i+1, len(S)):
                b = S[j]
                if g_ax == gcd_table[b][x] == gcd_table[a][b]:
                    count += 1
        return count

    def greedy_by_conflict(items):
        """Greedy that at each step picks the candidate with minimum conflict count."""
        S = []
        S_set = set()
        available = list(items)
        while available and time.time() < deadline:
            best_x = None
            best_conflicts = float('inf')
            for x in available:
                c = conflict_count(x, S)
                if c < best_conflicts:
                    best_conflicts = c
                    best_x = x
            if best_x is None or best_conflicts > 0:
                # No zero-conflict element; pick the one with minimum conflicts
                # But only if it's zero (i.e., valid)
                if best_conflicts > 0:
                    break
            S.append(best_x)
            S_set.add(best_x)
            available.remove(best_x)
        return S, S_set

    def removal_search(S, S_set, items_base):
        """Try to improve by removing an element and re-adding others."""
        S = list(S)
        S_set = set(S)
        improved = True
        no_improve_count = 0
        while improved and time.time() < deadline:
            improved = False
            no_improve_count += 1
            if no_improve_count > 50:
                break
            
            # Try adding any remaining element
            remaining = [x for x in items_base if x not in S_set]
            random.shuffle(remaining)
            for x in remaining:
                if can_add(S, x, S_set):
                    S.append(x)
                    S_set.add(x)
                    improved = True
                    break
            
            if improved:
                no_improve_count = 0
                continue
            
            # Try removal-based improvement: remove one element, then greedily re-add
            if len(S) > 2:
                # Pick a random element to remove
                u_idx = random.randint(0, len(S)-1)
                u = S[u_idx]
                S.remove(u)
                S_set.discard(u)
                
                # Now try to add elements back (including u) in random order
                all_candidates = [x for x in items_base if x not in S_set]
                random.shuffle(all_candidates)
                added_any = False
                for x in all_candidates:
                    if can_add(S, x, S_set):
                        S.append(x)
                        S_set.add(x)
                        added_any = True
                
                # Accept only if we gained size or same size but different composition
                # Actually, we should only accept if size increased
                # But we already removed one, so we need at least one add to break even
                # Let's accept if size >= original (which means we added at least 1)
                # This is tricky; let's just keep the best
                pass
        
        return S, S_set

    def local_search_add_only(S, S_set, items_base):
        """Local search that only tries to add elements in random order."""
        S = list(S)
        S_set = set(S)
        remaining = [x for x in items_base if x not in S_set]
        random.shuffle(remaining)
        for x in remaining:
            if can_add(S, x, S_set):
                S.append(x)
                S_set.add(x)
        return S, S_set

    items_base = list(range(2, N+1))
    best_size = 0
    best_set = []

    # Precompute primes for strategy
    primes = [x for x in items_base if all(x % i != 0 for i in range(2, int(x**0.5)+1))]

    def get_order(strategy):
        items = list(items_base)
        if strategy == 0:
            random.shuffle(items)
        elif strategy == 1:
            # Primes first
            rest = [x for x in items if x not in primes]
            random.shuffle(rest)
            items = list(primes) + rest
        elif strategy == 2:
            # Descending
            items.sort(reverse=True)
        elif strategy == 3:
            # Ascending
            items.sort()
        elif strategy == 4:
            # Random with prime bias
            rest = [x for x in items if x not in primes]
            random.shuffle(rest)
            items = list(primes) + rest
        return items

    # Phase 1: Multiple greedy restarts with different strategies
    while time.time() < deadline - 15:
        strategy = random.randint(0, 4)
        order = get_order(strategy)
        S, S_set = greedy_construct(order)
        
        # Local search: try adding remaining in random order
        S, S_set = local_search_add_only(S, S_set, items_base)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = list(S)
            if best_size == N - 1:
                break

    # Phase 2: Try conflict-based greedy from scratch
    while time.time() < deadline - 5:
        # Conflict-based greedy
        S, S_set = greedy_by_conflict(items_base)
        if len(S) > best_size:
            best_size = len(S)
            best_set = list(S)
            if best_size == N - 1:
                break

    # Phase 3: Local search on best set
    if time.time() < deadline - 2:
        S_set = set(best_set)
        S, S_set = local_search_add_only(best_set, S_set, items_base)
        if len(S) > best_size:
            best_size = len(S)
            best_set = list(S)

    # Final: try adding 1
    if best_set and time.time() < deadline - 1:
        # Check if 1 can be added: for all a,b in best_set, gcd(a,b) != 1
        can_add_one = True
        for i in range(len(best_set)):
            for j in range(i+1, len(best_set)):
                if gcd_table[best_set[i]][best_set[j]] == 1:
                    can_add_one = False
                    break
            if not can_add_one:
                break
        if can_add_one:
            best_set.append(1)
            best_size += 1

    return best_set
