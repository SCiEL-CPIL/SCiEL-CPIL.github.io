"""Solve 1/x + 1/y + 1/z = 4/n for n=3..256 with distinct denominators <= 10^12."""
import time

def construct():
    start_time = time.time()
    deadline = start_time + 65.0  # safety margin before 90s kill
    MAX_DEN = 10**12

    def get_divisors(n):
        divs = []
        i = 1
        while i * i <= n:
            if n % i == 0:
                divs.append(i)
                if i != n // i:
                    divs.append(n // i)
            i += 1
        return divs

    def find_solution(n):
        # Parametrization: x = n + d (d | n), y = d + (n+d)*k, z = d + (n+d)*(n//d)*k
        # This exactly satisfies 1/x+1/y+1/z = 4/n.
        # Distinctness z>y>x holds iff k > d/n (i.e. k >= floor(d/n)+1).
        divs = get_divisors(n)
        for d in divs:
            if time.time() > deadline:
                return None
            k_min = d // n + 1
            x = n + d
            if x > MAX_DEN:
                continue
            y = d + (n + d) * k_min
            if y > MAX_DEN:
                continue
            z = d + (n + d) * (n // d) * k_min
            if z > MAX_DEN:
                continue
            if x == y or x == z or y == z:
                continue
            # Verify by cross multiplication: n*(y*z + x*z + x*y) == 4*x*y*z
            lhs = n * (y * z + x * z + x * y)
            rhs = 4 * x * y * z
            if lhs == rhs:
                return (x, y, z)
        return None

    answers = []
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
        sol = find_solution(n)
        answers.append(sol)

    return answers
