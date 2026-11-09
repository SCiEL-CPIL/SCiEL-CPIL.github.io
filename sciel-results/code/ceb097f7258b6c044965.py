"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 5000

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

_ALL_PRIMES = _primes(N)

def construct():
    start_time = time.time()
    deadline = start_time + 280  # Safety margin before 330s kill
    
    # Precompute the full set as numpy array
    full_array = np.arange(N + 1)
    
    best_points = None
    best_size = 0
    
    # CSP approach: for each prime p <= k, choose one residue class to remove
    # The final set is points that avoid all removed residues
    # Use backtracking with forward checking
    
    # We'll process primes in order. For a target tuple size k, we need primes <= k.
    # Strategy: try different target sizes k, and for each, find if we can assign
    # residues to primes <= k such that the surviving set has size >= k.
    
    # Actually, let's think differently. The size of the final set depends on which
    # residues we remove. We want to maximize this size.
    
    # Alternative: for a given set of removed residues (one per prime up to some P),
    # the surviving count is the number of x in [0,5000] not in any removed class.
    # This can be computed via inclusion-exclusion or by direct counting.
    
    # Let's use a different strategy: greedy with local search on residue choices.
    # Start with a reasonable set of primes, choose residues greedily to maximize
    # surviving points, then do local search to improve.
    
    # For efficiency, precompute for each prime p and residue r, the count of points
    # in [0,5000] that are in residue class r mod p.
    # But we need to handle multiple primes simultaneously.
    
    # Let's implement the CSP backtracking as directed.
    
    def count_surviving(residue_choices):
        """
        residue_choices: dict mapping prime p -> residue r to remove
        Returns count of x in [0,5000] such that x % p != r for all (p,r) in residue_choices
        """
        if not residue_choices:
            return N + 1
        
        # Use numpy for efficiency
        mask = np.ones(N + 1, dtype=bool)
        for p, r in residue_choices.items():
            mask &= (full_array % p != r)
        return int(np.sum(mask))
    
    # Backtracking search
    # Process primes in increasing order
    # Prune if surviving count < next_prime (can't achieve tuple size >= next_prime)
    
    max_primes_to_consider = 50  # Limit to avoid explosion
    primes_to_use = _ALL_PRIMES[:max_primes_to_consider]
    
    # We'll do iterative deepening: try to find assignments for first k primes
    # such that surviving count >= p_{k+1} (the next prime)
    
    # Actually, the constraint is: if final tuple size is K, then for all primes p <= K,
    # one residue class mod p is missed. So we need to cover all primes <= K.
    
    # Let's try a different approach: 
    # 1. Pick a target K
    # 2. For all primes p <= K, we must remove one residue class
    # 3. Find residue choices that maximize surviving points
    # 4. Check if surviving points >= K
    # 5. If yes, we have a valid tuple of size >= K
    
    # To maximize, we want to choose residues that remove as few points as possible
    # overall. This is a complex optimization.
    
    # Simpler: use the repair+add approach but with better initialization.
    # Start with full set, repair it, then add points.
    
    def repair(points):
        points = list(points)
        changed = True
        while changed:
            if time.time() > deadline:
                break
            changed = False
            current_len = len(points)
            for p in _ALL_PRIMES:
                if p > current_len:
                    break
                counts = [0] * p
                for v in points:
                    counts[v % p] += 1
                if all(c > 0 for c in counts):
                    min_size = min(counts)
                    target_residues = [r for r in range(p) if counts[r] == min_size]
                    target_res = min(target_residues)
                    points = [v for v in points if v % p != target_res]
                    changed = True
                    break
        return points
    
    def add_points(points):
        points = list(points)
        point_set = set(points)
        
        M = {}
        for p in _ALL_PRIMES:
            counts = [0] * p
            for v in points:
                counts[v % p] += 1
            M[p] = set(r for r in range(p) if counts[r] == 0)
            
        candidates = [i for i in range(N + 1) if i not in point_set]
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() > deadline:
                break
            
            new_len = len(points) + 1
            safe = True
            for p in _ALL_PRIMES:
                if p > new_len:
                    break
                if M[p] == {x % p}:
                    safe = False
                    break
            
            if safe:
                points.append(x)
                point_set.add(x)
                for p in _ALL_PRIMES:
                    if p > new_len:
                        break
                    r = x % p
                    if r in M[p]:
                        M[p].remove(r)
        
        return points
    
    # Strategy 1: Start with full set, repair, then add
    full_set = list(range(N + 1))
    repaired_full = repair(full_set)
    added_full = add_points(repaired_full)
    
    if added_full and len(added_full) > best_size:
        best_points = added_full
        best_size = len(added_full)
    
    # Strategy 2: Try several random initializations
    for attempt in range(10):
        if time.time() > deadline:
            break
        
        size = 2000 + random.randint(0, 1000)
        initial_points = random.sample(range(N + 1), min(size, N + 1))
        
        repaired = repair(initial_points)
        added = add_points(repaired)
        
        if added and len(added) > best_size:
            best_points = added
            best_size = len(added)
    
    # Strategy 3: Local search - try to improve best_points by swapping points
    if best_points and time.time() < deadline - 30:
        points = list(best_points)
        point_set = set(points)
        
        # Precompute M
        M = {}
        for p in _ALL_PRIMES:
            counts = [0] * p
            for v in points:
                counts[v % p] += 1
            M[p] = set(r for r in range(p) if counts[r] == 0)
        
        # Try removing a point and adding two others
        for _ in range(100):
            if time.time() > deadline - 5:
                break
            
            if len(points) < 2:
                break
            
            # Pick a random point to remove
            idx = random.randint(0, len(points) - 1)
            removed = points[idx]
            points.pop(idx)
            point_set.discard(removed)
            
            # Update M after removal
            for p in _ALL_PRIMES:
                if p > len(points):
                    break
                r = removed % p
                if r not in M[p]:
                    # Check if this was the only point in this residue class
                    count = sum(1 for v in points if v % p == r)
                    if count == 0:
                        M[p].add(r)
            
            # Try to add two new points
            added_count = 0
            candidates = [i for i in range(N + 1) if i not in point_set]
            random.shuffle(candidates)
            
            for x in candidates:
                if added_count >= 2:
                    break
                if time.time() > deadline - 5:
                    break
                
                new_len = len(points) + 1
                safe = True
                for p in _ALL_PRIMES:
                    if p > new_len:
                        break
                    if M[p] == {x % p}:
                        safe = False
                        break
                
                if safe:
                    points.append(x)
                    point_set.add(x)
                    added_count += 1
                    for p in _ALL_PRIMES:
                        if p > new_len:
                            break
                        r = x % p
                        if r in M[p]:
                            M[p].remove(r)
            
            if len(points) > best_size:
                best_points = points
                best_size = len(points)
    
    if best_points is None:
        # Fallback: return empty or a minimal valid set
        return []
    
    return list(best_points)
# EVOLVE-BLOCK-END
