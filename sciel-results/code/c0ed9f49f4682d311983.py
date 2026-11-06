"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 60s limit

    def count_divisors(n):
        count = 0
        i = 1
        while i * i <= n:
            if n % i == 0:
                count += 1
                if i != n // i:
                    count += 1
            i += 1
        return count

    def get_divisors(n):
        divs = []
        i = 1
        while i * i <= n:
            if n % i == 0:
                divs.append(i)
                if i != n // i:
                    divs.append(n // i)
            i += 1
        return sorted(divs)

    best_m = 1
    max_divs = 0
    for m in range(1, 513):
        if time.time() > deadline:
            break
        d = count_divisors(m)
        if d > max_divs:
            max_divs = d
            best_m = m

    # If we ran out of time, best_m might not be the global max, 
    # but for 1..512 it's very fast. 
    # To be safe, if we didn't finish, we can rely on the fact that 
    # the loop is extremely fast (<0.1s usually).
    
    return get_divisors(best_m)
# EVOLVE-BLOCK-END
