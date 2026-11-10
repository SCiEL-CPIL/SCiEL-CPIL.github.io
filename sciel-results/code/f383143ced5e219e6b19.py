"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import combinations
from collections import defaultdict
import random

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill

    n = 128
    numbers = list(range(1, n + 1))

    # Precompute LCM table
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            l = _lcm(i, j)
            lcm_table[i][j] = l
            lcm_table[j][i] = l

    # Find all bad triples: (a,b,c) with a<b<c and lcm(a,b)==lcm(a,c)==lcm(b,c)
    bad_triples = []
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            lab = lcm_table[a][b]
            for c in range(b + 1, n + 1):
                if lcm_table[a][c] == lab and lcm_table[b][c] == lab:
                    bad_triples.append((a, b, c))

    if time.time() > deadline:
        return numbers

    # Build conflict graph: edge (x,y) exists if there's a bad triple containing both
    # Two elements conflict if they appear together in some bad triple
    conflict = [set() for _ in range(n + 1)]
    for (a, b, c) in bad_triples:
        conflict[a].add(b)
        conflict[a].add(c)
        conflict[b].add(a)
        conflict[b].add(c)
        conflict[c].add(a)
        conflict[c].add(b)

    # Precompute for each element the set of bad triples it's in
    elem_triples = defaultdict(set)
    for t in bad_triples:
        for x in t:
            elem_triples[x].add(t)

    def is_bad_triple(a, b, c):
        lab = lcm_table[a][b]
        lac = lcm_table[a][c]
        lbc = lcm_table[b][c]
        return lab == lac == lbc

    def can_add(selected_set, candidate):
        """Check if candidate can be added to selected_set without forming a bad triple."""
        for x in selected_set:
            if is_bad_triple(x, candidate, 0) if False else True:
                pass
        # Check all pairs in selected_set with candidate
        for x in selected_set:
            for y in selected_set:
                if x >= y:
                    continue
                if is_bad_triple(x, y, candidate):
                    return False
        return True

    def greedy_add(selected, selected_set, candidate):
        """Check if candidate can be added to selected (list) / selected_set (set)."""
        for i in range(len(selected)):
            for j in range(i + 1, len(selected)):
                if is_bad_triple(selected[i], selected[j], candidate):
                    return False
        return True

    def validate_set(s):
        """Check that no three elements in s form a bad triple."""
        s_set = set(s)
        for t in bad_triples:
            if all(x in s_set for x in t):
                return False
        return True

    best_result = []

    # === Approach 1: Greedy hitting set (remove elements covering most uncovered triples) ===
    removed = set()
    uncovered = set(bad_triples)

    while uncovered and time.time() < deadline - 15:
        best_x = None
        best_size = 0
        for x in range(1, n + 1):
            if x in removed:
                continue
            size = len(elem_triples[x] & uncovered)
            if size > best_size:
                best_size = size
                best_x = x
        if best_x is None or best_size == 0:
            break
        removed.add(best_x)
        uncovered -= elem_triples[best_x]

    hs_result = [x for x in range(1, n + 1) if x not in removed]

    # Try to add back removed elements if possible
    removed_list = sorted(removed, key=lambda x: len(elem_triples[x]))
    for x in removed_list:
        if time.time() > deadline - 10:
            break
        can_add = True
        for a, b in combinations(hs_result, 2):
            if is_bad_triple(a, b, x):
                can_add = False
                break
        if can_add:
            removed.discard(x)
            hs_result.append(x)
            hs_result.sort()

    if validate_set(hs_result) and len(hs_result) > len(best_result):
        best_result = list(hs_result)

    # === Approach 2: Multiple greedy orderings ===
    def divisor_count(x):
        count = 0
        for d in range(1, int(x ** 0.5) + 1):
            if x % d == 0:
                count += 1
                if d != x // d:
                    count += 1
        return count

    def is_prime(x):
        if x < 2:
            return False
        for d in range(2, int(x ** 0.5) + 1):
            if x % d == 0:
                return False
        return True

    # Ordering 1: Natural order
    selected = []
    for x in numbers:
        if time.time() > deadline - 8:
            break
        if greedy_add(selected, set(selected), x):
            selected.append(x)
    if len(selected) > len(best_result):
        best_result = list(selected)

    # Ordering 2: Reverse order
    selected = []
    for x in reversed(numbers):
        if time.time() > deadline - 8:
            break
        if greedy_add(selected, set(selected), x):
            selected.append(x)
    if len(selected) > len(best_result):
        best_result = list(selected)

    # Ordering 3: Few divisors first
    sorted_by_divs = sorted(numbers, key=lambda x: divisor_count(x))
    selected = []
    for x in sorted_by_divs:
        if time.time() > deadline - 8:
            break
        if greedy_add(selected, set(selected), x):
            selected.append(x)
    if len(selected) > len(best_result):
        best_result = list(selected)

    # Ordering 4: Large numbers first
    selected = []
    for x in sorted(numbers, reverse=True):
        if time.time() > deadline - 8:
            break
        if greedy_add(selected, set(selected), x):
            selected.append(x)
    if len(selected) > len(best_result):
        best_result = list(selected)

    # Ordering 5: Primes first
    primes = [x for x in numbers if is_prime(x)]
    composites = [x for x in numbers if not is_prime(x)]
    order5 = primes + composites
    selected = []
    for x in order5:
        if time.time() > deadline - 8:
            break
        if greedy_add(selected, set(selected), x):
            selected.append(x)
    if len(selected) > len(best_result):
        best_result = list(selected)

    # Ordering 6: Conflict-degree ascending (least connected first)
    sorted_by_conflict = sorted(numbers, key=lambda x: len(conflict[x]))
    selected = []
    for x in sorted_by_conflict:
        if time.time() > deadline - 8:
            break
        if greedy_add(selected, set(selected), x):
            selected.append(x)
    if len(selected) > len(best_result):
        best_result = list(selected)

    # Ordering 7: Conflict-degree descending (most connected first - skip them)
    sorted_by_conflict_desc = sorted(numbers, key=lambda x: -len(conflict[x]))
    selected = []
    for x in sorted_by_conflict_desc:
        if time.time() > deadline - 8:
            break
        if greedy_add(selected, set(selected), x):
            selected.append(x)
    if len(selected) > len(best_result):
        best_result = list(selected)

    # === Approach 3: Randomized greedy with multiple restarts ===
    random.seed(42)
    for restart in range(50):
        if time.time() > deadline - 5:
            break
        order = numbers[:]
        random.shuffle(order)
        selected = []
        for x in order:
            if time.time() > deadline - 5:
                break
            if greedy_add(selected, set(selected), x):
                selected.append(x)
        if len(selected) > len(best_result):
            best_result = list(selected)

    # === Approach 4: Local search - try swaps to improve ===
    def local_search_improve(result):
        """Try to improve by swapping one element out and two in, or adding one."""
        result_set = set(result)
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            # Try adding any missing element
            for x in numbers:
                if time.time() > deadline - 3:
                    break
                if x not in result_set:
                    if greedy_add(result, result_set, x):
                        result.append(x)
                        result_set.add(x)
                        improved = True
            # Try 1-for-1 swaps
            for x in list(result):
                if time.time() > deadline - 3:
                    break
                for y in numbers:
                    if time.time() > deadline - 3:
                        break
                    if y in result_set or y == x:
                        continue
                    # Try swapping x for y
                    new_result = [z for z in result if z != x]
                    new_result_set = result_set - {x}
                    if greedy_add(new_result, new_result_set, y):
                        new_result.append(y)
                        new_result.sort()
                        result = new_result
                        result_set = set(result)
                        improved = True
        return result

    best_result = local_search_improve(best_result)

    # === Approach 5: More randomized restarts with local search ===
    random.seed(123)
    for restart in range(30):
        if time.time() > deadline - 3:
            break
        order = numbers[:]
        random.shuffle(order)
        selected = []
        for x in order:
            if time.time() > deadline - 3:
                break
            if greedy_add(selected, set(selected), x):
                selected.append(x)
        # Local search on this result
        selected = local_search_improve(selected)
        if len(selected) > len(best_result):
            best_result = list(selected)

    # === Final polish: try to add any remaining numbers ===
    changed = True
    while changed and time.time() < deadline - 1:
        changed = False
        for x in numbers:
            if time.time() > deadline - 1:
                break
            if x not in best_result:
                if greedy_add(best_result, set(best_result), x):
                    best_result.append(x)
                    changed = True

    if not best_result:
        best_result = [1]

    return best_result
# EVOLVE-BLOCK-END
