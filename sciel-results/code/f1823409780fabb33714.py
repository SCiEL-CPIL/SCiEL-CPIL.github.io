import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    B = frozenset(range(65, 129))
    random.seed(42)
    
    def is_valid_A(A):
        A_set = set(A)
        S = A_set | set(B)
        A_list = sorted(A_set)
        for i in range(len(A_list)):
            for j in range(i + 1, len(A_list)):
                a, c = A_list[i], A_list[j]
                if a > 63 or c > 63:
                    continue
                s = a + c
                if s >= 65:
                    return False
                elif s in A_set:
                    return False
        n = len(A_list)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    a, b, c = A_list[i], A_list[j], A_list[k]
                    if a + b in S and a + c in S and b + c in S:
                        return False
        return True

    best_A = set()
    best_size = 0

    for attempt in range(30):
        if time.time() > deadline - 5:
            break

        A = set()
        order = list(range(1, 65))
        if attempt > 0:
            random.shuffle(order)

        for x in order:
            if time.time() > deadline - 5:
                break
            A_test = A | {x}
            if is_valid_A(A_test):
                A = A_test

        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for to_remove in list(A):
                if time.time() > deadline - 5:
                    break
                A_no = A - {to_remove}
                for to_add in range(1, 65):
                    if to_add in A_no:
                        continue
                    A_test = A_no | {to_add}
                    if is_valid_A(A_test):
                        if len(A_test) > len(A_no):
                            A = A_test
                            improved = True
                            break
                if improved:
                    break

        if len(A) > best_size:
            best_size = len(A)
            best_A = set(A)

    result = [int(x) for x in sorted(best_A | B)]
    return result
