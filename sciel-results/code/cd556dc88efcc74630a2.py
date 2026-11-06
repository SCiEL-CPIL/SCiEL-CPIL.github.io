import time
import random
from math import gcd

N = 128
lcm_table = [[0] * (N + 1) for _ in range(N + 1)]
for i in range(1, N + 1):
    for j in range(1, N + 1):
        lcm_table[i][j] = i * j // gcd(i, j)

# conflicts[a][b] is a set of x such that {a, b, x} is a bad triple
conflicts = [[set() for _ in range(N + 1)] for _ in range(N + 1)]

for a in range(1, N + 1):
    for b in range(a + 1, N + 1):
        L_ab = lcm_table[a][b]
        for x in range(1, N + 1):
            if x == a or x == b:
                continue
            if lcm_table[a][x] == L_ab and lcm_table[b][x] == L_ab:
                conflicts[a][b].add(x)
                conflicts[b][a].add(x)

def is_valid_addition(x, selected_set):
    for a in selected_set:
        for b in selected_set:
            if a == b:
                continue
            if x in conflicts[a][b]:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    best_size = 0
    best_set = set()

    orders = []
    base = list(range(1, N + 1))
    orders.append(base)

    random.seed(42)
    for _ in range(20):
        shuffled = base.copy()
        random.shuffle(shuffled)
        orders.append(shuffled)

    def num_divisors(n):
        count = 0
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                count += 1
                if i != n // i:
                    count += 1
        return count

    sorted_by_divs = sorted(base, key=num_divisors)
    orders.append(sorted_by_divs)
    orders.append(sorted(base, key=num_divisors, reverse=True))

    for order in orders:
        if time.time() > deadline:
            break

        current_set = set()
        for x in order:
            if time.time() > deadline:
                break
            if is_valid_addition(x, current_set):
                current_set.add(x)

        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set.copy()

    if time.time() < deadline:
        current_set = best_set.copy()
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for x in range(1, N + 1):
                if x not in current_set:
                    if is_valid_addition(x, current_set):
                        current_set.add(x)
                        improved = True
                        break

        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set.copy()

    if time.time() < deadline:
        for _ in range(100):
            if time.time() > deadline:
                break

            trial_set = best_set.copy()
            if not trial_set:
                continue

            rem_elem = random.choice(list(trial_set))
            trial_set.remove(rem_elem)

            candidates = [x for x in range(1, N + 1) if x not in trial_set]
            random.shuffle(candidates)

            for x in candidates:
                if time.time() > deadline:
                    break
                if is_valid_addition(x, trial_set):
                    trial_set.add(x)

            if len(trial_set) > best_size:
                best_size = len(trial_set)
                best_set = trial_set.copy()

    return list(best_set)

if not construct():
    construct = lambda: [1]
