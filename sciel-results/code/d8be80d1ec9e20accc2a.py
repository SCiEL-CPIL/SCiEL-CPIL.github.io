from fractions import Fraction
import time

def construct():
    start = time.time()
    deadline = start + 85

    def is_valid(A):
        for i in A:
            relevant = [j for j in A if j != i and j > i]
            if not relevant:
                continue
            target = Fraction(1, i)
            n = len(relevant)
            for mask in range(1, 1 << n):
                s = Fraction(0)
                for j in range(n):
                    if mask & (1 << j):
                        s += Fraction(1, relevant[j])
                if s == target:
                    return False
        return True

    # Greedy addition, large to small
    current = set()
    for elem in range(24, 0, -1):
        if time.time() > deadline:
            break
        trial = current | {elem}
        if is_valid(trial):
            current = trial

    result = sorted(current)
    if not result:
        result = list(range(13, 25))
    return result
