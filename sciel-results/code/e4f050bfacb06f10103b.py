def construct():
    best_set = []
    for m in range(1, 513):
        divisors = []
        for i in range(1, int(m**0.5) + 1):
            if m % i == 0:
                divisors.append(i)
                if i != m // i:
                    divisors.append(m // i)
        if len(divisors) > len(best_set):
            best_set = divisors
    return best_set
