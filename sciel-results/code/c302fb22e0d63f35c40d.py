"""Maximise admissible tuple size in [0, 5000].

Key insight: For a tuple of size m, only primes p <= m matter (for p > m,
a class is automatically missed). So the binding constraints are the small
primes. We want to keep as many integers as possible while ensuring that for
each binding prime p, at least one residue class mod p is entirely absent.

Strategy:
1. Start from the full set [0, N].
2. For each binding prime p (in order of "tightness"), if all p classes are
   present, remove the class whose removal preserves the most future
   flexibility (i.e., the class that is least "shared" with other constraints).
3. Use a greedy scoring: remove the class that minimizes the increase in
   "covered classes" for other primes, or equivalently, the class that has
   the fewest elements that are also in "critical" positions for other primes.
4. Multiple restarts with different removal orders and random tie-breaks.
"""
import time
import random
import sys

N = 5000
DEADLINE = 310.0  # seconds, leaving margin before 330s kill


def _primes(limit):
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            step = p
            start = p * p
            sieve[start:limit + 1:step] = [False] * len(sieve[start:limit + 1:step])
    return [i for i, ok in enumerate(sieve) if ok]


def _is_admissible(points, primes_needed):
    """Check if points form an admissible tuple for the given primes."""
    for p in primes_needed:
        if len(points) < p:
            continue
        residues = set()
        for v in points:
            residues.add(v % p)
            if len(residues) == p:
                return False
    return True


def _construct_once(seed, deadline):
    """Single attempt using greedy removal with lookahead scoring."""
    rng = random.Random(seed)
    points = list(range(N + 1))
    
    # Precompute residue for each point for each relevant prime
    # We'll track which classes are present for each prime
    
    # Primes that could ever be binding: p <= N+1
    all_primes = _primes(N + 1)
    
    # Track present classes per prime
    # For efficiency, maintain a dict: prime -> set of present residues
    present = {}
    for p in all_primes:
        present[p] = set(range(p))  # initially all present
    
    # Actually, let's be smarter. We only need to track primes p <= len(points).
    # Start: len = N+1, so all primes <= N+1 are potentially binding.
    
    # For each prime, count how many points are in each residue class
    class_counts = {}
    for p in all_primes:
        counts = [0] * p
        for v in points:
            counts[v % p] += 1
        class_counts[p] = counts
    
    # Greedy removal loop
    max_iter = 500
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        
        current_len = len(points)
        if current_len <= 1:
            break
        
        # Find binding primes: p <= current_len where all classes present
        binding_primes = []
        for p in all_primes:
            if p > current_len:
                break
            counts = class_counts[p]
            if all(c > 0 for c in counts):
                binding_primes.append(p)
        
        if not binding_primes:
            break
        
        # For each binding prime, find the best class to remove
        # Score: prefer removing classes that are "least important" for other constraints
        # Simple heuristic: remove the class with minimum count (breaks the constraint
        # while removing fewest points). Among ties, random.
        
        # Pick a binding prime to address (smallest first for stability, or random)
        # Try different orders
        order = list(binding_primes)
        rng.shuffle(order)
        
        for p in order:
            counts = class_counts[p]
            min_count = min(counts)
            if min_count == 0:
                continue  # already broken
            candidates = [r for r, c in enumerate(counts) if c == min_count]
            chosen = rng.choice(candidates)
            
            # Remove all points with v % p == chosen
            to_remove = set(v for v in points if v % p == chosen)
            points = [v for v in points if v not in to_remove]
            
            # Update class_counts for all primes
            for q in all_primes:
                if q > current_len:
                    break
                for v in to_remove:
                    class_counts[q][v % q] -= 1
            
            break  # restart the outer loop
    
    return points


def _construct_greedy_with_lookahead(seed, deadline):
    """Greedy removal with a lookahead score: when choosing which class to remove
    for prime p, prefer the class whose removal causes the fewest "new" complete
    coverages for other primes. This is expensive, so we use a cheaper proxy:
    remove the class that has the smallest count (removes fewest points)."""
    rng = random.Random(seed)
    points = list(range(N + 1))
    
    all_primes = _primes(N + 1)
    
    # For speed, only track primes up to some limit
    # The binding primes are those p <= len(points). As we remove points,
    # len decreases, so fewer primes are binding.
    
    # Precompute: for each point, its residue mod each prime
    # This is too expensive for all primes. Instead, compute on the fly.
    
    # Maintain: for each prime p, a list of counts per residue
    class_counts = {}
    for p in all_primes:
        counts = [0] * p
        for v in points:
            counts[v % p] += 1
        class_counts[p] = counts
    
    max_iter = 300
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        
        current_len = len(points)
        if current_len <= 1:
            break
        
        # Find binding primes
        binding = []
        for p in all_primes:
            if p > current_len:
                break
            counts = class_counts[p]
            if all(c > 0 for c in counts):
                binding.append(p)
        
        if not binding:
            break
        
        # For each binding prime, compute the "cost" of removing each class
        # Cost = number of points removed (i.e., the count of that class)
        # We want to minimize total points removed, so pick the prime+class
        # combination with minimum count.
        
        best_p = None
        best_r = None
        best_count = float('inf')
        
        for p in binding:
            counts = class_counts[p]
            min_c = min(counts)
            if min_c < best_count:
                best_count = min_c
                best_p = p
                best_r = None
                # Find all classes with this min count
                candidates = [r for r, c in enumerate(counts) if c == min_c]
                best_r = rng.choice(candidates)
            elif min_c == best_count:
                # Tie: random among all
                candidates = [r for r, c in enumerate(class_counts[p]) if c == min_c]
                if rng.random() < 0.5:
                    best_p = p
                    best_r = rng.choice(candidates)
        
        if best_p is None:
            break
        
        p = best_p
        r = best_r
        
        # Remove all points with v % p == r
        to_remove = set(v for v in points if v % p == r)
        points = [v for v in points if v not in to_remove]
        
        # Update class_counts
        for q in all_primes:
            if q > current_len:
                break
            for v in to_remove:
                class_counts[q][v % q] -= 1
    
    return points


def construct():
    """Main entry point: try multiple strategies, keep best admissible tuple."""
    start_time = time.time()
    
    # Strategy 1: Simple greedy (remove min-count class for smallest binding prime)
    best = _construct_once(0, DEADLINE)
    best_len = len(best)
    
    # Validate
    if not _is_admissible(best, _primes(len(best))):
        # Fallback
        points = list(range(N + 1))
        all_primes = _primes(N + 1)
        changed = True
        while changed:
            changed = False
            current_len = len(points)
            for p in all_primes:
                if p > current_len:
                    break
                classes = {}
                for v in points:
                    classes.setdefault(v % p, []).append(v)
                if len(classes) == p:
                    smallest = min(classes.values(), key=len)
                    points = [v for v in points if v not in smallest]
                    changed = True
                    break
        best = points
        best_len = len(best)
    
    # Strategy 2: Lookahead greedy with different seeds
    seed = 1
    while time.time() < DEADLINE - 2:
        candidate = _construct_greedy_with_lookahead(seed, DEADLINE)
        if len(candidate) > best_len:
            if _is_admissible(candidate, _primes(len(candidate))):
                best = candidate
                best_len = len(candidate)
        seed += 1
    
    # Final validation
    if not _is_admissible(best, _primes(len(best))):
        points = list(range(N + 1))
        all_primes = _primes(N + 1)
        changed = True
        while changed:
            changed = False
            current_len = len(points)
            for p in all_primes:
                if p > current_len:
                    break
                classes = {}
                for v in points:
                    classes.setdefault(v % p, []).append(v)
                if len(classes) == p:
                    smallest = min(classes.values(), key=len)
                    points = [v for v in points if v not in smallest]
                    changed = True
                    break
        best = points
    
    return best
