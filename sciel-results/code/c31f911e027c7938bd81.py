import time
import math
import random
from itertools import combinations, product

def compute_score(U):
    U_set = set(U)
    n = len(U_set)
    if n == 0:
        return 0
    max_val = max(U_set)
    if max_val == 0:
        return 1.0
    sumset = set()
    diffset = set()
    for a in U_set:
        for b in U_set:
            sumset.add(a + b)
            diffset.add(a - b)
    ratio = len(diffset) / len(sumset)
    if ratio <= 1:
        return 1.0
    denom = math.log(2 * max_val + 1)
    if denom == 0:
        return 1.0
    return 1 + math.log(ratio) / denom

def geometric_set(base, k):
    """Generate geometric set {0, base^0, base^1, ..., base^(k-1)}"""
    if k <= 0:
        return [0]
    return [0] + [base**i for i in range(k)]

def mixed_geometric(base1, base2, k):
    """Generate mixed geometric set alternating between two bases"""
    U = [0]
    val = 1
    for i in range(k):
        U.append(val)
        if i % 2 == 0:
            val *= base1
        else:
            val *= base2
    return U

def generate_candidates(deadline):
    """Generate diverse candidate sets"""
    candidates = []
    
    # Pure geometric sets
    for base in range(2, 15):
        for k in range(1, 25):
            if time.time() > deadline:
                return candidates
            U = geometric_set(base, k)
            candidates.append(U)
    
    # Mixed geometric sets
    for b1 in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]:
        for b2 in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]:
            if b1 == b2:
                continue
            for k in range(1, 20):
                if time.time() > deadline:
                    return candidates
                U = mixed_geometric(b1, b2, k)
                candidates.append(U)
    
    # Fibonacci-like sets
    for k in range(1, 30):
        if time.time() > deadline:
            return candidates
        U = [0]
        a, b = 1, 1
        for _ in range(k):
            U.append(a)
            a, b = b, a + b
        candidates.append(U)
    
    # Lucas-like sets
    for k in range(1, 30):
        if time.time() > deadline:
            return candidates
        U = [0]
        a, b = 2, 1
        for _ in range(k):
            U.append(a)
            a, b = b, a + b
        candidates.append(U)
    
    # Triangular numbers
    for k in range(1, 30):
        if time.time() > deadline:
            return candidates
        U = [0]
        for i in range(1, k + 1):
            U.append(i * (i + 1) // 2)
        candidates.append(U)
    
    # Squares
    for k in range(1, 30):
        if time.time() > deadline:
            return candidates
        U = [0]
        for i in range(1, k + 1):
            U.append(i * i)
        candidates.append(U)
    
    # Primes (first k primes + 0)
    for k in range(1, 30):
        if time.time() > deadline:
            return candidates
        primes = []
        n = 2
        while len(primes) < k:
            is_prime = True
            for p in primes:
                if p * p > n:
                    break
                if n % p == 0:
                    is_prime = False
                    break
            if is_prime:
                primes.append(n)
            n += 1
        U = [0] + primes
        candidates.append(U)
    
    # Random sets with structure
    random.seed(42)
    for trial in range(50):
        if time.time() > deadline:
            return candidates
        k = random.randint(3, 15)
        max_val = random.randint(50, 5000)
        # Start with 0 and some structured points
        U = {0}
        # Add some geometric points
        base = random.randint(2, 5)
        for i in range(random.randint(1, 5)):
            U.add(base**i)
        # Add some random points
        while len(U) < k:
            U.add(random.randint(0, max_val))
        candidates.append(sorted(U))
    
    return candidates

def local_search_dynamic(U, deadline, max_iters=1000):
    """Local search: try to improve the set by adding/removing elements"""
    best_set = list(U)
    best_score = compute_score(U)
    
    if time.time() > deadline:
        return best_set, best_score
    
    max_val = max(best_set) if best_set else 0
    search_range = max(2 * max_val + 10, 100)
    
    for iteration in range(max_iters):
        if time.time() > deadline:
            break
        
        improved = False
        # Try adding a random element
        if len(best_set) < 25:
            candidate_val = random.randint(0, search_range)
            if candidate_val not in best_set:
                new_set = best_set + [candidate_val]
                new_score = compute_score(new_set)
                if new_score > best_score:
                    best_set = new_set
                    best_score = new_score
                    improved = True
        
        # Try removing an element (if set is large enough)
        if len(best_set) > 3 and random.random() < 0.3:
            idx = random.randint(0, len(best_set) - 1)
            if best_set[idx] != 0:  # Keep 0
                new_set = best_set[:idx] + best_set[idx+1:]
                new_score = compute_score(new_set)
                if new_score > best_score:
                    best_set = new_set
                    best_score = new_score
                    improved = True
        
        # Try replacing an element
        if len(best_set) > 3 and random.random() < 0.5:
            idx = random.randint(1, len(best_set) - 1)  # Don't replace 0
            candidate_val = random.randint(0, search_range)
            if candidate_val not in best_set:
                new_set = best_set.copy()
                new_set[idx] = candidate_val
                new_set.sort()
                new_score = compute_score(new_set)
                if new_score > best_score:
                    best_set = new_set
                    best_score = new_score
                    improved = True
        
        if not improved and iteration % 50 == 0:
            # Occasionally reset search range
            max_val = max(best_set) if best_set else 0
            search_range = max(2 * max_val + 10, 100)
    
    return best_set, best_score

def construct():
    deadline = time.time() + 140  # safety margin
    random.seed(12345)
    
    best_score = 0
    best_set = [0, 1, 3]
    
    # Phase 1: Generate and evaluate diverse candidates
    candidates = generate_candidates(deadline)
    
    for U in candidates:
        if time.time() > deadline - 30:
            break
        s = compute_score(U)
        if s > best_score:
            best_score = s
            best_set = list(U)
    
    # Phase 2: Local search from the best candidate
    if time.time() < deadline - 10:
        best_set, best_score = local_search_dynamic(best_set, deadline, max_iters=500)
    
    # Phase 3: Try a few more local searches from other good candidates
    if time.time() < deadline - 5:
        # Sort candidates by score and try local search on top ones
        scored_candidates = []
        for U in candidates[:20]:
            s = compute_score(U)
            scored_candidates.append((s, U))
        scored_candidates.sort(reverse=True)
        
        for s, U in scored_candidates[:5]:
            if time.time() > deadline - 2:
                break
            if s < best_score * 0.95:  # Only try if reasonably close
                continue
            new_set, new_score = local_search_dynamic(U, deadline, max_iters=200)
            if new_score > best_score:
                best_score = new_score
                best_set = new_set
    
    # Ensure 0 is in the set
    if 0 not in best_set:
        best_set.insert(0, 0)
    
    return sorted(set(best_set))
