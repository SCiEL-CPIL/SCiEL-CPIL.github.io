import itertools
import math
import time
import random

def score_set(U):
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n == 0:
        return -1
    sumset = set()
    diffset = set()
    for a in U:
        for b in U:
            sumset.add(a + b)
            diffset.add(a - b)
    if len(sumset) == 0:
        return -1
    ratio = len(diffset) / len(sumset)
    if ratio <= 0:
        return -1
    max_u = max(U)
    if max_u == 0:
        return 1.0
    denom = math.log(2 * max_u + 1)
    if denom == 0:
        return 1.0
    return 1 + math.log(ratio) / denom

def _golomb_ruler(n):
    """Generate a Golomb ruler of order n using known optimal constructions."""
    known = {
        1: [0],
        2: [0, 1],
        3: [0, 1, 3],
        4: [0, 1, 4, 6],
        5: [0, 1, 4, 10, 11],
        6: [0, 1, 4, 10, 12, 17],
        7: [0, 1, 4, 10, 12, 17, 22],
        8: [0, 1, 4, 10, 12, 17, 22, 29],
        9: [0, 1, 4, 10, 12, 17, 22, 29, 36],
        10: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44],
        11: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52],
        12: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60],
        13: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 68],
        14: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 68, 76],
        15: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 68, 76, 84],
    }
    if n in known:
        return known[n]
    # Greedy construction for larger n
    ruler = [0]
    for i in range(1, n):
        pos = 1
        while True:
            candidate = ruler + [pos]
            diffs = set()
            valid = True
            for j in range(len(candidate)):
                for k in range(j + 1, len(candidate)):
                    d = candidate[k] - candidate[j]
                    if d in diffs:
                        valid = False
                        break
                    diffs.add(d)
                if not valid:
                    break
            if valid:
                ruler.append(pos)
                break
            pos += 1
    return ruler

def _local_search(U, deadline, max_iters=3000):
    """Improve set U via local search with swap, add, and remove moves."""
    U = sorted(set(U))
    best_score = score_set(U)
    best_U = U[:]
    max_u = max(U)
    upper_bound = max_u + 60  # search space for new elements
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        improved = False
        
        # Try swapping one element
        for i in range(len(best_U)):
            old_val = best_U[i]
            for new_val in range(0, upper_bound + 1):
                if new_val == old_val:
                    continue
                new_U = best_U[:]
                new_U[i] = new_val
                if len(set(new_U)) != len(new_U):
                    continue
                if 0 not in new_U:
                    continue
                s = score_set(new_U)
                if s > best_score + 1e-9:
                    best_score = s
                    best_U = new_U[:]
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            # Try adding an element
            if len(best_U) < 25:
                for new_val in range(0, upper_bound + 1):
                    if new_val in best_U:
                        continue
                    new_U = best_U + [new_val]
                    s = score_set(new_U)
                    if s > best_score + 1e-9:
                        best_score = s
                        best_U = new_U[:]
                        improved = True
                        break
        
        if not improved:
            # Try removing an element (if size > 3)
            if len(best_U) > 3:
                for i in range(len(best_U)):
                    if best_U[i] == 0:
                        continue  # must keep 0
                    new_U = best_U[:i] + best_U[i+1:]
                    s = score_set(new_U)
                    if s > best_score + 1e-9:
                        best_score = s
                        best_U = new_U[:]
                        improved = True
                        break
        
        if not improved:
            break
    
    return sorted(set(best_U)), best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    best_score = 1.0
    best_set = [0, 1, 3]
    
    # Generate candidates from Golomb rulers and other structures
    candidates = []
    
    # Known optimal Golomb rulers
    for n in range(3, 16):
        if time.time() > deadline:
            break
        gr = _golomb_ruler(n)
        candidates.append(gr)
    
    # Try some structured sets based on quadratic residues mod p
    # For prime p, QR(p) = {x^2 mod p : x in 1..p-1} union {0}
    # These have good additive properties
    for p in [7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]:
        if time.time() > deadline:
            break
        # Quadratic residues mod p
        qr = set()
        for x in range(p):
            qr.add((x * x) % p)
        qr = sorted(qr)
        if 0 in qr:
            candidates.append(qr)
        else:
            candidates.append([0] + qr)
    
    # Try multiplicative subgroups
    for p in [7, 11, 13, 17, 19, 23, 29, 31]:
        if time.time() > deadline:
            break
        # Find generators and subgroups
        for g in range(2, p):
            # Check if g is a primitive root
            order = 1
            val = g
            while val != 1:
                val = (val * g) % p
                order += 1
                if order >= p:
                    break
            if order == p - 1:
                # g is primitive root, take subgroup of size d
                for d in [2, 3, 4, 5, 6]:
                    if (p - 1) % d == 0:
                        k = (p - 1) // d
                        subgroup = set()
                        val = 1
                        for i in range(d):
                            subgroup.add(val)
                            val = (val * g) % p
                        subgroup = sorted(subgroup)
                        if 0 in subgroup:
                            candidates.append(subgroup)
                        else:
                            candidates.append([0] + subgroup)
    
    # Random sets for diversity
    random.seed(42)
    for trial in range(30):
        if time.time() > deadline:
            break
        n = random.randint(3, 15)
        max_val = random.randint(10, 150)
        U = [0] + random.sample(range(1, max_val + 1), n - 1)
        U = sorted(U)
        candidates.append(U)
    
    # Evaluate all candidates
    for U in candidates:
        if time.time() > deadline:
            break
        s = score_set(U)
        if s > best_score:
            best_score = s
            best_set = U[:]
    
    # Local search on the best candidate
    if time.time() < deadline:
        best_set, best_score = _local_search(best_set, deadline, max_iters=2000)
    
    # Try local search on other promising candidates
    for U in candidates:
        if time.time() > deadline:
            break
        if U == best_set:
            continue
        s = score_set(U)
        if s > best_score - 0.02:  # only if close
            improved_U, improved_score = _local_search(U, deadline, max_iters=1000)
            if improved_score > best_score:
                best_score = improved_score
                best_set = improved_U
    
    # Final polish: try more aggressive local search
    if time.time() < deadline:
        best_set, best_score = _local_search(best_set, deadline, max_iters=1500)
    
    return sorted(set(best_set))
