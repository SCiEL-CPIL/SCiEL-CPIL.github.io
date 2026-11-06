import random
import math
import itertools

def construct():
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    def is_pairwise_coprime(sums):
        # sums is a list of distinct values
        n = len(sums)
        for i in range(n):
            for j in range(i+1, n):
                if gcd(sums[i], sums[j]) > 1:
                    return False
        return True

    def evaluate(A, B):
        sums = set()
        for a in A:
            for b in B:
                sums.add(a + b)
        if not sums:
            return 0
        if not is_pairwise_coprime(list(sums)):
            return 0
        return len(sums)

    best_score = 0
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    
    # Seed check
    s = evaluate(best_A, best_B)
    if s > best_score:
        best_score = s

    # Local Search / Random Search
    # We want to maximize distinct coprime sums.
    # Range 1..128
    
    # Helper to generate random set
    def random_set(size, low, high):
        return random.sample(range(low, high+1), size)

    # Try to find a solution with more distinct sums
    # Focus on small numbers where primes are dense
    # and differences are small to keep sums small?
    # Actually, larger sums have more chance to be prime? No, prime density decreases.
    # But coprime-ness is easier if numbers are prime.
    
    # Strategy: 
    # 1. Try random A, B in [1, 100]
    # 2. If valid, keep.
    # 3. Mutate one element and re-evaluate.
    
    max_iters = 20000
    current_A = random_set(3, 1, 100)
    current_B = random_set(3, 1, 100)
    current_score = evaluate(current_A, current_B)
    
    if current_score > best_score:
        best_score = current_score
        best_A = current_A
        best_B = current_B

    for _ in range(max_iters):
        # Mutation
        if random.random() < 0.5:
            idx = random.randint(0, 2)
            old_val = current_A[idx]
            current_A[idx] = random.randint(1, 128)
            new_score = evaluate(current_A, current_B)
            if new_score > current_score:
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_A = current_A
                    best_B = current_B
            else:
                # Revert or accept with small prob (simulated annealing style)
                # For simplicity, revert if not better, unless we are stuck
                if new_score == 0 and current_score > 0:
                    current_A[idx] = old_val
                elif new_score == 0 and current_score == 0:
                    # Stuck in invalid state, keep trying
                    pass
                else:
                    current_A[idx] = old_val
        else:
            idx = random.randint(0, 2)
            old_val = current_B[idx]
            current_B[idx] = random.randint(1, 128)
            new_score = evaluate(current_A, current_B)
            if new_score > current_score:
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_A = current_A
                    best_B = current_B
            else:
                if new_score == 0 and current_score > 0:
                    current_B[idx] = old_val
                elif new_score == 0 and current_score == 0:
                    pass
                else:
                    current_B[idx] = old_val

        # Occasional restart
        if _ % 1000 == 0:
            current_A = random_set(3, 1, 100)
            current_B = random_set(3, 1, 100)
            current_score = evaluate(current_A, current_B)
            if current_score > best_score:
                best_score = current_score
                best_A = current_A
                best_B = current_B

    # Specific targeted search for 9 sums (all distinct primes)
    # If we can find 3x3 sums that are all distinct primes, that's 9.
    # This is very hard.
    # Let's try to find 8 or 7.
    
    # Final return
    return [sorted(best_A), sorted(best_B)]

# EVOLVE-BLOCK-START
# The code above is the logic. I will wrap it properly.
# EVOLVE-BLOCK-END
