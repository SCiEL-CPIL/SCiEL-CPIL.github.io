"""Improved program: search for 12 distinct positive integers <= 10000 minimizing distinct primes in pairwise sums."""
import time
import random
import math
from itertools import combinations

# Deadline guard
_DEADLINE = 90.0
_START_TIME = None

def _time_left():
    if _START_TIME is None:
        return _DEADLINE
    return _DEADLINE - (time.time() - _START_TIME)

def _factorize(n, primes_set):
    """Return set of prime factors of n. Only care about primes in primes_set for speed,
    but we need all prime factors to build the union. So we do full factorization."""
    factors = set()
    d = 2
    while d * d <= n:
        if n % d == 0:
            factors.add(d)
            while n % d == 0:
                n //= d
        d += 1 if d == 2 else 2  # skip evens after 2
    if n > 1:
        factors.add(n)
    return factors

def _score(arr):
    """Compute the number of distinct primes dividing all pairwise sums of distinct elements."""
    prime_union = set()
    n = len(arr)
    for i in range(n):
        for j in range(i+1, n):
            s = arr[i] + arr[j]
            # Factor s
            d = 2
            while d * d <= s:
                if s % d == 0:
                    prime_union.add(d)
                    while s % d == 0:
                        s //= d
                d += 1 if d == 2 else 2
            if s > 1:
                prime_union.add(s)
    return len(prime_union)

def _score_fast(arr, known_primes):
    """Faster score if we already have a set of candidate primes.
    Returns (score, new_primes_found) where new_primes are primes not in known_primes."""
    prime_union = set(known_primes)
    n = len(arr)
    for i in range(n):
        for j in range(i+1, n):
            s = arr[i] + arr[j]
            d = 2
            while d * d <= s:
                if s % d == 0:
                    prime_union.add(d)
                    while s % d == 0:
                        s //= d
                d += 1 if d == 2 else 2
            if s > 1:
                prime_union.add(s)
    return len(prime_union)

def _greedy_improve(arr, max_iters=200):
    """Local search: try replacing one element with another to reduce score."""
    arr = list(arr)
    best_score = _score(arr)
    n = len(arr)
    max_val = 10000
    
    for iteration in range(max_iters):
        if _time_left() < 5.0:
            break
        improved = False
        # Try swapping each element with a random candidate
        for i in range(n):
            if _time_left() < 5.0:
                break
            old_val = arr[i]
            # Try a few random replacements
            for _ in range(20):
                new_val = random.randint(1, max_val)
                if new_val == old_val or new_val in arr:
                    continue
                arr[i] = new_val
                s = _score(arr)
                if s < best_score:
                    best_score = s
                    old_val = new_val
                    improved = True
                else:
                    arr[i] = old_val
            # Reset to old_val if no improvement found for this i
            arr[i] = old_val
        
        if not improved:
            break
    
    return arr, best_score

def _hill_climbing(arr, max_rounds=50):
    """More thorough local search."""
    arr = list(arr)
    best_score = _score(arr)
    n = len(arr)
    max_val = 10000
    
    for rnd in range(max_rounds):
        if _time_left() < 5.0:
            break
        improved = False
        for i in range(n):
            if _time_left() < 5.0:
                break
            old_val = arr[i]
            best_for_i = old_val
            best_score_i = best_score
            # Try random candidates
            for _ in range(50):
                new_val = random.randint(1, max_val)
                if new_val == old_val or new_val in arr:
                    continue
                arr[i] = new_val
                s = _score(arr)
                if s < best_score_i:
                    best_score_i = s
                    best_for_i = new_val
                arr[i] = old_val
            if best_for_i != old_val:
                arr[i] = best_for_i
                best_score = best_score_i
                improved = True
        
        if not improved:
            break
    
    return arr, best_score

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    best_arr = list(range(1, 13))
    best_score = _score(best_arr)
    
    random.seed(42)
    
    # Strategy: try to find sets where all pairwise sums have few distinct prime factors.
    # Key insight: if all numbers are multiples of a small set of primes, sums will too.
    # But we need distinct positive integers <= 10000.
    
    # Try various initial configurations and improve
    attempts = 0
    max_attempts = 200
    
    # First, try some structured initial guesses
    # Idea: use numbers that are all even -> sums are even, so 2 is always a factor.
    # If we can make all sums have only primes from a small set, that's good.
    
    # Try: multiples of small numbers
    structured_guesses = [
        [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24],  # all even, sums are even
        [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36],  # multiples of 3
        [6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72],  # multiples of 6
        [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46],  # 2 mod 4
        [4, 10, 16, 22, 28, 34, 40, 46, 52, 58, 64, 70],  # 2 mod 6
    ]
    
    for guess in structured_guesses:
        if _time_left() < 10.0:
            break
        arr, sc = _hill_climbing(guess, max_rounds=20)
        if sc < best_score:
            best_score = sc
            best_arr = arr
    
    # Random restarts with hill climbing
    for attempt in range(max_attempts):
        if _time_left() < 10.0:
            break
        attempts += 1
        
        # Generate random initial set
        if attempt % 3 == 0:
            # Random even numbers
            candidates = list(range(2, 10001, 2))
            arr = random.sample(candidates, 12)
        elif attempt % 3 == 1:
            # Random multiples of 6
            candidates = list(range(6, 10001, 6))
            arr = random.sample(candidates, 12)
        else:
            # Fully random
            arr = random.sample(range(1, 10001), 12)
        
        # Quick score check
        sc = _score(arr)
        if sc < best_score:
            best_score = sc
            best_arr = arr
        
        # Hill climb
        arr2, sc2 = _hill_climbing(arr, max_rounds=10)
        if sc2 < best_score:
            best_score = sc2
            best_arr = arr2
    
    # Final polish on best solution
    if _time_left() > 15.0:
        best_arr, best_score = _hill_climbing(best_arr, max_rounds=30)
    
    if _time_left() > 10.0:
        best_arr, best_score = _greedy_improve(best_arr, max_iters=100)
    
    return best_arr
