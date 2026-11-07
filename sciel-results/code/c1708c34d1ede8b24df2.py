import math
import time
from itertools import combinations

def construct():
    start = time.time()
    deadline = start + 82  # safety margin under 90s

    N = 256

    # Sieve for primes
    def sieve(limit):
        is_p = [True] * (limit + 1)
        is_p[0] = is_p[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if is_p[i]:
                for j in range(i * i, limit + 1, i):
                    is_p[j] = False
        return [i for i in range(2, limit + 1) if is_p[i]]

    primes = sieve(N)
    prime_set = set(primes)
    prime_index = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)

    # Prime powers: largest p^k <= N for each prime p
    prime_powers = {}
    for p in primes:
        power = p
        while power * p <= N:
            power *= p
        prime_powers[p] = power

    # Precompute prime factorization (as frozenset of distinct primes) for all numbers 1..N
    F = [None] * (N + 1)
    F[1] = frozenset()
    for i in range(2, N + 1):
        x = i
        factors = set()
        d = 2
        while d * d <= x:
            if x % d == 0:
                factors.add(d)
                while x % d == 0:
                    x //= d
            else:
                d += 1
        if x > 1:
            factors.add(x)
        F[i] = frozenset(factors)

    # Base solution: 1 + all prime powers
    base_numbers = [1] + [prime_powers[p] for p in primes]
    base_sum = sum(base_numbers)

    # Generate candidate moves
    moves = []
    
    def get_mask(ps):
        mask = 0
        for p in ps:
            mask |= (1 << prime_index[p])
        return mask

    best_c_for_ps = {}
    
    for c in range(2, N + 1):
        if c in prime_set:
            continue
        ps = F[c]
        if len(ps) < 2:
            continue
        if ps not in best_c_for_ps or c > best_c_for_ps[ps]:
            best_c_for_ps[ps] = c

    for ps, c in best_c_for_ps.items():
        base_cost = sum(prime_powers[p] for p in ps)
        gain = c - base_cost
        if gain > 0:
            mask = get_mask(ps)
            moves.append((mask, c, gain, ps))

    # Sort moves by gain descending for branch and bound efficiency
    moves.sort(key=lambda x: -x[2])
    
    n_moves = len(moves)
    suffix_gains = [0] * (n_moves + 1)
    for i in range(n_moves - 1, -1, -1):
        suffix_gains[i] = suffix_gains[i+1] + moves[i][2]

    best_total_gain = 0
    best_selected_moves = []
    
    def solve(idx, used_mask, current_gain, selected):
        nonlocal best_total_gain, best_selected_moves
        
        if time.time() > deadline - 1:
            return

        if current_gain + suffix_gains[idx] <= best_total_gain:
            return

        if idx == n_moves:
            if current_gain > best_total_gain:
                best_total_gain = current_gain
                best_selected_moves = selected[:]
            return

        # Option 1: Skip move idx
        solve(idx + 1, used_mask, current_gain, selected)

        # Option 2: Take move idx
        mask, c, gain, ps = moves[idx]
        if mask & used_mask == 0:
            selected.append((mask, c, gain, ps))
            solve(idx + 1, used_mask | mask, current_gain + gain, selected)
            selected.pop()

    solve(0, 0, 0, [])

    # Local search to improve the B&B result
    def compute_used_mask(selected):
        m = 0
        for move in selected:
            m |= move[0]
        return m

    def is_compatible(move, used_mask):
        return (move[0] & used_mask) == 0

    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        current_selected = best_selected_moves[:]
        current_gain = best_total_gain
        used_mask = compute_used_mask(current_selected)
        
        # Try adding each unselected move
        for i, move in enumerate(moves):
            if time.time() > deadline - 2:
                break
            if move in current_selected:
                continue
            if is_compatible(move, used_mask):
                new_gain = current_gain + move[2]
                if new_gain > best_total_gain:
                    best_total_gain = new_gain
                    best_selected_moves = current_selected + [move]
                    improved = True
                    used_mask = compute_used_mask(best_selected_moves)
                    current_selected = best_selected_moves[:]
                    current_gain = best_total_gain
        
        # Try removing each selected move and see if we can add a better one
        for i, move in enumerate(current_selected):
            if time.time() > deadline - 2:
                break
            new_selected = [m for j, m in enumerate(current_selected) if j != i]
            new_used_mask = compute_used_mask(new_selected)
            new_gain = current_gain - move[2]
            
            # Try adding any compatible move to the reduced set
            for j, cand in enumerate(moves):
                if time.time() > deadline - 2:
                    break
                if cand in new_selected:
                    continue
                if is_compatible(cand, new_used_mask):
                    total_gain = new_gain + cand[2]
                    if total_gain > best_total_gain:
                        best_total_gain = total_gain
                        best_selected_moves = new_selected + [cand]
                        improved = True
                        break
        
        # Try pairwise swaps: remove one selected, add one unselected
        for i, move in enumerate(current_selected):
            if time.time() > deadline - 2:
                break
            new_selected = [m for j, m in enumerate(current_selected) if j != i]
            new_used_mask = compute_used_mask(new_selected)
            new_gain = current_gain - move[2]
            
            for j, cand in enumerate(moves):
                if time.time() > deadline - 2:
                    break
                if cand in new_selected:
                    continue
                if is_compatible(cand, new_used_mask):
                    total_gain = new_gain + cand[2]
                    if total_gain > best_total_gain:
                        best_total_gain = total_gain
                        best_selected_moves = new_selected + [cand]
                        improved = True
                        break
    
    # Construct final solution
    final_sol = set(base_numbers)
    
    used_primes_in_moves = set()
    for mask, c, gain, ps in best_selected_moves:
        for p in ps:
            used_primes_in_moves.add(p)
            final_sol.remove(prime_powers[p])
        
    for mask, c, gain, ps in best_selected_moves:
        final_sol.add(c)
        
    result = [int(x) for x in final_sol]
    
    # Verification
    def verify(lst):
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                if math.gcd(lst[i], lst[j]) != 1:
                    return False
        return True

    if not verify(result):
        result = [int(x) for x in base_numbers]
        
    return result
