import sys
sys.setrecursionlimit(10000)

def construct():
    candidates = list(range(2, 41))
    
    # We will try to find the maximum k such that there exists a subset of size k
    # with distinct subset products.
    
    # Start from a high estimate and go down, or start low and go up.
    # Given the previous score was 18, let's try to verify if 19 is possible.
    # If 19 is not possible, we return a valid set of 18.
    
    # To be safe and fast, we can use a greedy approach to get a good initial solution,
    # then try to improve it via local search or backtracking.
    
    # Let's use a simple greedy first to get a baseline, then try to add more.
    # Actually, a direct backtracking for k=19 might be too slow if not pruned well.
    
    # Let's try a different heuristic: 
    # Sort candidates by "cost". The "cost" of adding a number x to a set S 
    # is the number of new products it creates that might collide.
    
    # Alternative: Use the fact that if we pick numbers that are powers of distinct primes,
    # we are guaranteed distinct products. The current program does this.
    # Can we do better?
    # The current program picks:
    # 2, 4, 16 (powers of 2)
    # 3, 9 (powers of 3)
    # 5, 25 (powers of 5)
    # 7
    # 11, 13, 17, 19, 23, 29, 31, 37
    # Total: 3 + 2 + 2 + 1 + 8 = 16? 
    # Wait, let's recount the current program's output.
    # sel_2 = [2, 4, 16] (3)
    # sel_3 = [3, 9] (2)
    # sel_5 = [5, 25] (2)
    # sel_7 = [7] (1)
    # sel_others = [11, 13, 17, 19, 23, 29, 31, 37] (8)
    # Total = 3+2+2+1+8 = 16.
    
    # The prompt says the current program has score 1.3333.
    # The seed program likely returned a lower number, maybe 13 or 14?
    # If seed returned 13, 16/13 = 1.23. If seed returned 12, 16/12 = 1.33.
    # So the current program returns 16.
    # The task is to maximize the number.
    # Can we get more than 16?
    
    # Let's try to construct a set of size 17, 18, etc.
    
    # We can use a greedy algorithm that tries to add the "most independent" numbers.
    
    def can_add(products, num):
        new_products = set(products)
        for p in products:
            new_products.add(p * num)
        return len(new_products) == 2 * len(products)
    
    # Greedy approach:
    # 1. Start with empty set.
    # 2. Try to add numbers in an order that maximizes future potential.
    #    Small primes are good.
    
    # Let's try a randomized greedy or multiple greedy passes.
    # But we need a deterministic solution for the contract? 
    # "Each evaluation must finish within 60 s."
    # Randomness is risky if it fails.
    
    # Let's use a systematic backtracking for small k, or a very good greedy.
    
    # Heuristic: Numbers with fewer prime factors are better?
    # Or numbers that are "large" to avoid small product collisions?
    # Actually, small numbers create small products, which are less likely to collide with large products?
    # No, collisions happen if $a \cdot b = c \cdot d$.
    
    # Let's try to improve the current 16.
    # The current selection uses all primes >= 7 except 2,3,5 which are handled specially.
    # Primes: 2,3,5,7,11,13,17,19,23,29,31,37. (12 primes)
    # If we just picked all 12 primes, we would have 12 numbers.
    # By picking powers of 2, 3, 5, we increased the count.
    # For 2: 2, 4, 16 (3 numbers) vs 2 (1 number). Gain +2.
    # For 3: 3, 9 (2 numbers) vs 3 (1 number). Gain +1.
    # For 5: 5, 25 (2 numbers) vs 5 (1 number). Gain +1.
    # Total gain +4. Base 12 primes -> 16 numbers.
    
    # Can we gain more?
    # Prime 2: Powers 2, 4, 8, 16, 32.
    # Exponents: 1, 2, 3, 4, 5.
    # Distinct subset sums for exponents?
    # {1, 2, 4} -> sums 0,1,2,3,4,5,6,7. (8 sums, 3 numbers).
    # Can we add 8 (exp 3)? Sums of {1,2,3,4}: 0..9. Distinct?
    # 1, 2, 3, 4. Subsets:
    # 0
    # 1, 2, 3, 4
    # 1+2=3 (collides with 3) -> No.
    # So {1,2,4} is maximal for powers of 2?
    # What about {1, 3, 4}? Sums: 0, 1, 3, 4, 1+3=4 (collides).
    # What about {2, 3, 4}? Sums: 0, 2, 3, 4, 2+3=5, 2+4=6, 3+4=7, 2+3+4=9. Distinct?
    # Yes. So {2,3,4} works. Numbers: 4, 8, 16.
    # Is it better than {1,2,4}? Same count (3).
    # Can we do 4 numbers for prime 2?
    # Need 4 exponents with distinct subset sums.
    # Max sum of 4 exponents $e_1+e_2+e_3+e_4 \le 5$ (since $2^5=32 \le 40$, $2^6=64 > 40$).
    # Max possible distinct sums is $2^4=16$.
    # But max sum is $5$. Range $0..5$. Only 6 values.
    # So we can have at most 6 distinct subset sums?
    # Wait, the subset sums must be distinct integers.
    # If max sum is 5, the sums are in $\{0, 1, \dots, 5\}$.
    # So we can have at most 6 distinct sums.
    # We have $2^k$ sums. So $2^k \le 6 \implies k \le 2$.
    # So we can only pick at most 2 powers of 2?
    # Let's re-verify.
    # Powers of 2 available: $2^1, 2^2, 2^3, 2^4, 2^5$.
    # If we pick $2^a$ and $2^b$, subset sums of exponents $\{a, b\}$ are $0, a, b, a+b$.
    # These must be distinct.
    # If we pick 3 powers $2^a, 2^b, 2^c$, sums are $0, a, b, c, a+b, a+c, b+c, a+b+c$.
    # All must be distinct.
    # Max exponent is 5.
    # Let's try to pick 3 exponents from $\{1,2,3,4,5\}$.
    # Try $\{1, 2, 4\}$. Sums: $0, 1, 2, 4, 3, 5, 6, 7$. All distinct. Max sum 7.
    # Is 7 allowed? The product is $2^7 = 128$.
    # The constraint is that the *numbers* must be in $2..40$.
    # The *products* can be anything.
    # So the exponents just need to be such that $2^{\text{sum}}$ is an integer (always true).
    # There is no upper bound on the product value, only on the input numbers.
    # So for prime 2, we can pick ANY subset of exponents $\{1,2,3,4,5\}$ such that their subset sums are distinct.
    # We found $\{1,2,4\}$ works.
    # Can we add 3? $\{1,2,3,4\}$. Sums: $0,1,2,3,4, 1+2=3$ (collision). No.
    # Can we pick $\{1,3,4\}$? Sums: $0,1,3,4, 1+3=4$ (collision). No.
    # Can we pick $\{2,3,4\}$? Sums: $0,2,3,4, 2+3=5, 2+4=6, 3+4=7, 2+3+4=9$. Distinct.
    # So $\{2,3,4\}$ works. Numbers: 4, 8, 16.
    # Can we pick 4 exponents?
    # We need 16 distinct sums.
    # Max sum of 4 exponents from $\{1..5\}$ is $5+4+3+2=14$.
    # Sums in $0..14$. 15 values.
    # We need 16 distinct sums. Impossible.
    # So max 3 powers of 2.
    
    # Similarly for prime 3: Powers $3^1, 3^2, 3^3$ (3, 9, 27).
    # Exponents $\{1,2,3\}$.
    # Try 3 exponents: $\{1,2,3\}$. Sums: $0,1,2,3, 1+2=3$ (collision).
    # Try $\{1,2\}$. Sums: $0,1,2,3$. Distinct.
    # Try $\{1,3\}$. Sums: $0,1,3,4$. Distinct.
    # Try $\{2,3\}$. Sums: $0,2,3,5$. Distinct.
    # Max 2 powers of 3.
    
    # Prime 5: Powers $5^1, 5^2$ (5, 25).
    # Exponents $\{1,2\}$.
    # Try $\{1,2\}$. Sums: $0,1,2,3$. Distinct.
    # Max 2 powers of 5.
    
    # Prime 7: Powers $7^1$ (7).
    # Max 1.
    
    # Other primes: Only 1st power.
    
    # So the maximum number of numbers we can get from this "independent prime power" strategy is:
    # 3 (from 2) + 2 (from 3) + 2 (from 5) + 1 (from 7) + 8 (others) = 16.
    
    # Is it possible to get MORE than 16 by mixing primes?
    # Yes, if we use composite numbers that don't interfere.
    # But the "independent prime power" strategy is very strong because it guarantees no collisions.
    # To beat 16, we need a strategy that packs more numbers into the same "space".
    
    # Consider the set of all primes: 12 primes.
    # If we replace a prime $p$ with two numbers $a, b$ such that $a,b$ are powers of $p$, we gain 1.
    # We did this for 2, 3, 5.
    # Can we do this for 7?
    # Powers of 7: 7, 49 (49 > 40). So only 7. No gain.
    
    # Can we use composites like 6?
    # If we use 6, it shares factors with 2 and 3.
    # If we have 2 and 3, and we add 6, do we get collisions?
    # Products of $\{2,3\}$: 1, 2, 3, 6.
    # Add 6: New products: 6, 12, 18, 36.
    # Collision: 6 is already in the set. So we cannot add 6 if 2 and 3 are present.
    # So we can't just add composites.
    
    # What if we REMOVE 2 and 3, and ADD 6?
    # Set: $\{6, \text{others}\}$.
    # We lose 2 numbers (2,3) and gain 1 (6). Net -1. Bad.
    
    # What if we replace 2 with 4?
    # Set: $\{4, 3, \dots\}$.
    # Products of $\{4,3\}$: 1, 4, 3, 12. Distinct.
    # We lost 2, gained 4. Same count.
    
    # It seems 16 is a strong local maximum for this type of construction.
    # Can we find a set of 17?
    
    # Let's try a computational search for 17.
    # We can use the fact that the set of products must be distinct.
    # We can try to construct a set of 17 numbers.
    
    # Given the time constraint, I will provide a program that uses a **Greedy Algorithm with Backtracking** to try to find a set larger than 16. If it fails to find 17 within a time limit, it returns the best found (likely 16 or 17).
    
    # However, to ensure a score improvement, we should at least match 16.
    # The current program returns 16.
    # If I return 16, the score is the same.
    # I need to return > 16.
    
    # Let's try to find 17.
    # One idea: Use a set of primes and replace some with powers, but maybe mix in a composite that allows more density?
    
    # Actually, there is a known set of 18 integers with distinct subset products?
    # Let's check the literature or try a search.
    
    # I will write a program that performs a **Beam Search** or **Greedy with Restart** to try to find a set of size 17 or 18.
    
    import random
    import time
    
    def check_distinct(products):
        return len(products) == len(set(products))
    
    def greedy_construct(order):
        selected = []
        products = {1}
        for num in order:
            new_products = set(products)
            for p in products:
                new_products.add(p * num)
            if len(new_products) == 2 * len(products):
                selected.append(num)
                products = new_products
        return selected
    
    # Try multiple random orders and a sorted order
    candidates = list(range(2, 41))
    
    best_set = []
    
    # 1. Sorted order
    s = greedy_construct(sorted(candidates))
    if len(s) > len(best_set):
        best_set = s
        
    # 2. Reverse sorted
    s = greedy_construct(sorted(candidates, reverse=True))
    if len(s) > len(best_set):
        best_set = s
        
    # 3. Random shuffles
    start_time = time.time()
    for _ in range(1000):
        if time.time() - start_time > 50:
            break
        shuffled = candidates[:]
        random.shuffle(shuffled)
        s = greedy_construct(shuffled)
        if len(s) > len(best_set):
            best_set = s
            if len(best_set) >= 18:
                break
                
    # 4. Try to improve best_set by swapping
    # This is getting complex. Let's just return the best found.
    
    # If the best found is 16, we return 16.
    # If we find 17, we return 17.
    
    # To guarantee validity, we must verify the result.
    def verify(lst):
        prods = {1}
        for x in lst:
            new_prods = set(prods)
            for p in prods:
                new_prods.add(p * x)
            if len(new_prods) != 2 * len(prods):
                return False
            prods = new_prods
        return True
        
    if not verify(best_set):
        # Fallback to the known 16
        best_set = [2, 4, 16, 3, 9, 5, 25, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        
    return best_set
