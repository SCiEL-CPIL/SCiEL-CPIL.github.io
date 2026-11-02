import time
import math
import random

def _score(U):
    """Compute the score for set U."""
    U = sorted(set(U))
    n = len(U)
    if n <= 1:
        return 1.0
    max_u = max(U)
    if max_u == 0:
        return 1.0
    # Compute sumset and diffset
    sumset = set()
    diffset = set()
    for i in range(n):
        for j in range(n):
            sumset.add(U[i] + U[j])
            diffset.add(U[i] - U[j])
    ratio = len(diffset) / len(sumset)
    if ratio <= 1.0:
        return 1.0
    return 1.0 + math.log(ratio) / math.log(2 * max_u + 1)

def _golomb_greedy(n, max_val=5000):
    """Greedy Golomb ruler construction - place elements to maximize distinct differences."""
    U = [0]
    diffs = set([0])
    for _ in range(1, n):
        best_pos = -1
        best_score = -1
        for pos in range(1, max_val + 1):
            # Check if adding pos creates duplicate differences
            valid = True
            for u in U:
                d = pos - u
                if abs(d) in diffs:
                    valid = False
                    break
            if valid:
                # Score: prefer smaller positions (tighter packing)
                score = -pos
                if score > best_score:
                    best_score = score
                    best_pos = pos
        if best_pos == -1:
            break
        U.append(best_pos)
        for u in U[:-1]:
            diffs.add(abs(best_pos - u))
    return sorted(U)

def _local_search(U, deadline, max_iter=100000):
    """Local search to improve the score."""
    U = sorted(set(U))
    best_score = _score(U)
    best_U = U[:]
    no_improve_count = 0
    max_u = max(best_U) if best_U else 1
    
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        
        improved = False
        n = len(best_U)
        max_u = max(best_U) if best_U else 1
        
        # Try adding an element
        if n < 50:
            for c in range(1, min(max_u + 20, 3000)):
                if c not in best_U:
                    new_U = best_U + [c]
                    s = _score(new_U)
                    if s > best_score + 1e-15:
                        best_U = new_U
                        best_score = s
                        improved = True
                        break
            if improved:
                no_improve_count = 0
                continue
        
        # Try removing an element
        if n > 2:
            for i, c in enumerate(best_U):
                if c == 0:
                    continue
                new_U = [x for x in best_U if x != c]
                s = _score(new_U)
                if s > best_score + 1e-15:
                    best_U = new_U
                    best_score = s
                    improved = True
                    break
            if improved:
                no_improve_count = 0
                continue
        
        # Try swapping an element
        for i, c in enumerate(best_U):
            if c == 0:
                continue
            for d in range(1, min(max_u + 20, 3000)):
                if d not in best_U:
                    new_U = [x for x in best_U if x != c] + [d]
                    s = _score(new_U)
                    if s > best_score + 1e-15:
                        best_U = new_U
                        best_score = s
                        improved = True
                        break
            if improved:
                break
        if improved:
            no_improve_count = 0
            continue
        
        # No improvement - try perturbation
        no_improve_count += 1
        if no_improve_count > 50:
            # Random perturbation
            if len(best_U) > 2:
                idx = random.randint(1, len(best_U) - 1)
                best_U[idx] = random.randint(1, max(best_U) + 15)
                best_U = sorted(set(best_U))
                s = _score(best_U)
                if s > best_score:
                    best_score = s
            no_improve_count = 0
    
    return best_U, best_score

def construct():
    start = time.time()
    deadline = start + 115
    
    best_score = 0
    best_U = [0, 1]
    
    # Phase 1: Try Golomb rulers of various sizes
    for n in range(2, 25):
        if time.time() > deadline - 30:
            break
        U = _golomb_greedy(n, max_val=2000)
        if len(U) >= 2:
            s = _score(U)
            if s > best_score:
                best_score = s
                best_U = U
    
    # Phase 2: Local search from best Golomb ruler
    if time.time() < deadline - 10:
        best_U, best_score = _local_search(best_U, deadline, max_iter=50000)
    
    # Phase 3: Try multiple random restarts with local search
    remaining = deadline - time.time()
    if remaining > 15:
        num_restarts = max(1, int(remaining / 8))
        for r in range(num_restarts):
            if time.time() > deadline - 5:
                break
            
            # Generate a random set
            n = random.randint(3, 20)
            max_range = random.randint(10, 500)
            U = [0] + random.sample(range(1, max_range + 1), n - 1)
            U = sorted(U)
            
            s = _score(U)
            if s > best_score:
                best_score = s
                best_U = U
            
            # Local search from this start
            if time.time() < deadline - 5:
                local_deadline = min(deadline - 3, time.time() + 5)
                U2, s2 = _local_search(U, local_deadline, max_iter=10000)
                if s2 > best_score:
                    best_score = s2
                    best_U = U2
    
    # Phase 4: Final polish - try to tighten the set
    if time.time() < deadline - 3:
        # Try removing elements that don't help
        changed = True
        while changed and time.time() < deadline - 2:
            changed = False
            for c in best_U[:]:
                if c == 0:
                    continue
                new_U = [x for x in best_U if x != c]
                if len(new_U) >= 2:
                    s = _score(new_U)
                    if s >= best_score - 1e-15:
                        if s > best_score:
                            best_U = new_U
                            best_score = s
                            changed = True
                            break
                        else:
                            # Same score but smaller set - prefer smaller
                            best_U = new_U
                            changed = True
                            break
        
        # Try shifting/squeezing
        if time.time() < deadline - 2:
            # Try to reduce max by checking if we can shift elements
            for _ in range(50):
                if time.time() > deadline - 2:
                    break
                improved = False
                max_u = max(best_U)
                # Try moving a non-zero element closer to 0
                for i, c in enumerate(best_U):
                    if c == 0:
                        continue
                    for d in range(1, c):
                        if d not in best_U:
                            new_U = best_U[:]
                            new_U[i] = d
                            new_U = sorted(set(new_U))
                            s = _score(new_U)
                            if s > best_score + 1e-15:
                                best_U = new_U
                                best_score = s
                                improved = True
                                break
                    if improved:
                        break
                if not improved:
                    break
    
    return sorted(set(best_U))
