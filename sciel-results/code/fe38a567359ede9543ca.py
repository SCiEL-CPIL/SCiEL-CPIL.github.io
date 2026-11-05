import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    # Primes up to enough to cover product <= 10^18
    # Product of first 15 primes (2..47) is ~6.1e17, next (53) makes it > 3e19
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]

    def compute_score(powers):
        """
        powers: list of (prime, exponent) with exponent > 0.
        Returns (score, n). n is the product.
        """
        n = 1
        for p, e in powers:
            if e == 0:
                continue
            # Check overflow before multiplying
            if n > (10**18) // (p ** e):
                return -1, -1
            n *= (p ** e)
        if n < 2:
            return 0, n

        # Generate divisors
        divs = [1]
        for p, e in powers:
            if e == 0:
                continue
            current_len = len(divs)
            power = 1
            for _ in range(e):
                power *= p
                for i in range(current_len):
                    divs.append(divs[i] * power)

        if len(divs) < 2:
            return 0, n

        divs.sort()

        s = 0
        # Optimization: gcd(d_i, d_{i+1}) == 1 iff they share no prime factor.
        # Since divs are sorted, we can compute this.
        for i in range(len(divs) - 1):
            if math.gcd(divs[i], divs[i + 1]) == 1:
                s += 1
        return s, n

    # 1. Determine the maximum number of distinct primes (square-free kernel size)
    # We want the largest k such that product of first k primes <= 10^18
    max_k = 0
    prod = 1
    for i, p in enumerate(primes):
        if prod > (10**18) // p:
            break
        prod *= p
        max_k = i + 1

    best_score = -1
    best_n = 2
    best_powers = []

    def update_best(powers):
        nonlocal best_score, best_n, best_powers
        s, n = compute_score(powers)
        if s > best_score:
            best_score = s
            best_n = n
            best_powers = [tuple(x) for x in powers]

    # Initial guess: Square free product of first max_k primes
    # This usually yields a high score because the divisor lattice is a hypercube.
    initial_powers = [(primes[i], 1) for i in range(max_k)]
    update_best(initial_powers)

    # 2. Local Search / Hill Climbing
    # We will try to modify the exponents.
    # Moves:
    # A. Increase exponent of an existing prime
    # B. Decrease exponent of an existing prime (if > 1)
    # C. Add a new prime (if not present) - usually not helpful if we start with max_k,
    #    but might be if we reduced some exponents significantly? Unlikely to beat max_k square-free.
    # D. Remove a prime (set exponent to 0) - allows adding others?
    
    # Actually, the most promising direction is increasing exponents of small primes
    # because it creates more divisors that are "close" in value but potentially coprime?
    # Wait, increasing exponents of small primes creates divisors like p^k.
    # Adjacent divisors in a hypercube-like structure are often coprime.
    # Example: n = 2*3*5 = 30. Divisors: 1, 2, 3, 5, 6, 10, 15, 30.
    # Pairs: (1,2)gcd1, (2,3)gcd1, (3,5)gcd1, (5,6)gcd1, (6,10)gcd2, (10,15)gcd5, (15,30)gcd15.
    # Score: 4.
    # If n = 2^2*3*5 = 60. Divisors: 1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60.
    # Pairs: (1,2)1, (2,3)1, (3,4)1, (4,5)1, (5,6)1, (6,10)2, (10,12)2, (12,15)3, (15,20)5, (20,30)10, (30,60)30.
    # Score: 5. (Improved from 4)
    
    # So increasing exponents seems good.
    
    current_powers = [list(x) for x in initial_powers]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try all possible moves
        # 1. Increase exponent
        for i in range(len(current_powers)):
            p, e = current_powers[i]
            # Check if we can increase
            n_curr = 1
            for pp, ee in current_powers:
                n_curr *= (pp ** ee)
            if n_curr * p <= 10**18:
                new_powers = [list(x) for x in current_powers]
                new_powers[i][1] = e + 1
                s, n = compute_score(new_powers)
                if s > best_score:
                    best_score = s
                    best_n = n
                    best_powers = [tuple(x) for x in new_powers]
                    current_powers = new_powers
                    improved = True
                    break # Restart loop with new state
        
        if improved:
            continue

        # 2. Decrease exponent (might open up space for other primes or different structure?)
        # This is less likely to improve score directly, but might allow a different local optimum.
        # Given time constraints, we focus on increasing exponents of small primes first.
        
        # 3. Try adding a new prime if we have room?
        # If we started with max_k, we are likely at the limit.
        # But if we decreased some exponents, we might have room.
        # Let's check if we can add a prime that is not in the list.
        active_primes = {p for p, e in current_powers if e > 0}
        for p in primes:
            if p in active_primes:
                continue
            n_curr = 1
            for pp, ee in current_powers:
                n_curr *= (pp ** ee)
            if n_curr * p <= 10**18:
                new_powers = [list(x) for x in current_powers]
                new_powers.append([p, 1])
                s, n = compute_score(new_powers)
                if s > best_score:
                    best_score = s
                    best_n = n
                    best_powers = [tuple(x) for x in new_powers]
                    current_powers = new_powers
                    improved = True
                    break
            # Heuristic: if we can't add small primes, skip larger ones
            if p > 100:
                break

        if improved:
            continue

        # If no simple move improves, try a more complex move:
        # Increase exponent of a small prime and decrease a large prime?
        # Or just increase exponent of the smallest prime repeatedly?
        # The loop above already handles increasing any prime.
        
        # If we are stuck, we might be at a local maximum.
        # Let's try a perturbation: reduce the largest prime's exponent by 1 (if > 1)
        # and see if we can add it back or increase others.
        # Actually, if we can't increase any exponent, we are likely at the global max for this structure.
        
        # One more check: swap a prime?
        # Replace a large prime with a smaller one? No, we started with smallest primes.
        
        # If we haven't improved in a full pass, we are done.
        # To avoid infinite loops if no improvement is found, we break.
        # The 'improved' flag stays False, so loop exits.

    return best_n

# Quick test logic to ensure it doesn't crash
# if __name__ == "__main__":
#     n = construct()
#     print(n)
