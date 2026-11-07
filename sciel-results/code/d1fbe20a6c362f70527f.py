import time
import random

def construct():
    deadline = time.time() + 80

    def can_add(S_set, x):
        S = S_set | {x}
        for y in S_set:
            if y == x:
                continue
            for z in S_set:
                if z == x or z == y:
                    continue
                if (x + y) in S and (x + z) in S and (y + z) in S:
                    return False
        for b in S_set:
            a = x - b
            if a in S_set and a != b and a != x and b != x:
                for c in S_set:
                    if c == a or c == b or c == x:
                        continue
                    if (a + c) in S and (b + c) in S:
                        return False
        return True

    best_set = set(range(63, 129))

    for x in range(1, 63):
        if can_add(best_set, x):
            best_set.add(x)

    random.seed(42)
    current_set = set(best_set)
    for _ in range(20000):
        if time.time() > deadline:
            break
        candidates = [x for x in range(1, 63) if x not in current_set]
        if not candidates:
            break
        x = random.choice(candidates)
        if can_add(current_set, x):
            current_set.add(x)
            if len(current_set) > len(best_set):
                best_set = set(current_set)
        else:
            for y in list(current_set):
                if y < 63:
                    if can_add(current_set - {y}, x):
                        current_set = current_set - {y} | {x}
                        if len(current_set) > len(best_set):
                            best_set = set(current_set)
                        break

    return list(best_set)
