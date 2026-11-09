import itertools
import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # 15s safety margin before 90s kill
    
    def is_sidon_subset(elems):
        """Check if elems is Sidon including diagonals with early termination."""
        sums = set()
        n = len(elems)
        for i in range(n):
            for j in range(i, n):
                s = elems[i] + elems[j]
                if s in sums:
                    return False
                sums.add(s)
        return True
    
    def largest_sidon_size_brute(S):
        """Brute force: try all subsets from largest to smallest."""
        n = len(S)
        for size in range(n, 0, -1):
            for subset in itertools.combinations(range(n), size):
                elems = [S[i] for i in subset]
                if is_sidon_subset(elems):
                    return size
        return 0
    
    # Generate structured candidates with high collision density
    structured_candidates = []
    
    # Arithmetic progressions - these have many midpoint collisions
    for start in range(0, 100):
        for d in range(1, 12):
            S = [start + i*d for i in range(12)]
            if all(0 <= x <= 128 for x in S):
                structured_candidates.append(S)
    
    # Quadratic-like sequences (more sum collisions)
    for a in range(1, 10):
        for b in range(0, 50):
            for c in range(0, 30):
                S = [a*i*i + b*i + c for i in range(12)]
                if len(set(S)) == 12 and all(0 <= x <= 128 for x in S):
                    structured_candidates.append(S)
    
    # Mixed: combine two APs interleaved
    for base1 in range(0, 20):
        for d1 in range(1, 10):
            for base2 in range(0, 20):
                for d2 in range(1, 10):
                    S = []
                    for i in range(6):
                        S.append(base1 + i*d1)
                        S.append(base2 + i*d2)
                    if len(set(S)) == 12 and all(0 <= x <= 128 for x in S):
                        structured_candidates.append(S)
    
    # Evaluate structured candidates
    best_S = list(range(12))
    best_score = largest_sidon_size_brute(best_S)
    
    for S in structured_candidates:
        if time.time() > deadline:
            break
        score = largest_sidon_size_brute(S)
        if score < best_score:
            best_score = score
            best_S = S[:]
        if best_score <= 3:
            break
    
    # Local search with multiple restarts
    random.seed(42)
    
    def random_neighbor(S):
        """Create a random neighbor by changing one element."""
        S_new = S[:]
        idx = random.randint(0, 11)
        old_val = S_new[idx]
        for _ in range(30):
            new_val = random.randint(0, 128)
            if new_val != old_val and new_val not in S_new:
                S_new[idx] = new_val
                return S_new
        return None
    
    def two_element_neighbor(S):
        """Create a neighbor by changing two elements."""
        S_new = S[:]
        i, j = random.sample(range(12), 2)
        for _ in range(20):
            new_i = random.randint(0, 128)
            new_j = random.randint(0, 128)
            if new_i == new_j:
                continue
            if new_i in S_new or new_j in S_new:
                continue
            S_new[i] = new_i
            S_new[j] = new_j
            if len(set(S_new)) == 12:
                return S_new
        return None
    
    # Phase 1: Local search from best structured
    for iteration in range(5000):
        if time.time() > deadline:
            break
        
        if random.random() < 0.03:
            # Random restart
            S = [random.randint(0, 128) for _ in range(12)]
            if len(set(S)) < 12:
                continue
        elif random.random() < 0.7:
            S = random_neighbor(best_S)
        else:
            S = two_element_neighbor(best_S)
        
        if S is None:
            continue
        
        score = largest_sidon_size_brute(S)
        if score < best_score:
            best_score = score
            best_S = S[:]
            if best_score <= 3:
                break
    
    # Phase 2: Targeted improvement - try to force more collisions
    if best_score > 3 and time.time() < deadline:
        # Try to create sets where many triples are in AP
        for _ in range(2000):
            if time.time() > deadline:
                break
            
            # Build set with many 3-term APs
            S = []
            # Pick 4 base points
            bases = random.sample(range(0, 128), 4)
            d = random.randint(1, 8)
            
            for b in bases:
                for k in range(3):
                    val = b + k*d
                    if 0 <= val <= 128 and val not in S:
                        S.append(val)
            
            # Fill remaining with random
            while len(S) < 12:
                val = random.randint(0, 128)
                if val not in S:
                    S.append(val)
            
            score = largest_sidon_size_brute(S)
            if score < best_score:
                best_score = score
                best_S = S[:]
                if best_score <= 3:
                    break
    
    return best_S

# EVOLVE-BLOCK-END

