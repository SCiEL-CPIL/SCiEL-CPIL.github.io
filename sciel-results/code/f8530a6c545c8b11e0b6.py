def construct():
    start_time = time.time()
    deadline = start_time + 80

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

    # Search for 4-term solution
    for a in range(5, 100):
        if time.time() > deadline:
            break
        for b in range(a + 1, 2000):
            if time.time() > deadline:
                break
            N = a * b - a - b
            P = a * b
            if N <= 0:
                continue
            NP = N * P
            # cN - P must divide NP, be in (bN-P, P), and (cN-P+P)%N==0
            # c = (m+P)/N where m = cN-P
            # Need: m | NP, m > bN-P, m < P, (m+P) % N == 0, (m+P)//N > b
            # Also d = cP/m > c => m < P (already have)
            min_m = b * N - P + 1
            if min_m < 1:
                min_m = 1
            max_m = P - 1
            
            # Get divisors of NP
            divs = get_divisors(NP)
            for m in divs:
                if m < min_m or m > max_m:
                    continue
                if (m + P) % N != 0:
                    continue
                c = (m + P) // N
                if c <= b:
                    continue
                d = c * P // m
                if d > c and d >= 5 and d <= 10000000:
                    # Verify
                    if a * b * c * d == b * c * d + a * c * d + a * b * d + a * b * c:
                        return [a, b, c, d]
    
    # If no 4-term found, try 5-term with similar approach
    raise RuntimeError("no representation found")
