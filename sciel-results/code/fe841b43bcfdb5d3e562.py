"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s guard for 90s limit
    
    n = 12
    max_val = 1048576
    
    # Greedy construction: build set element by element
    # Start with smallest possible, then add next smallest that maintains distinct subset sums
    
    def check_distinct_sums(elements):
        """Check if all subset sums are distinct. Returns True if valid."""
        sums = set()
        m = len(elements)
        # Use bit manipulation to enumerate all subsets
        for mask in range(1 << m):
            s = 0
            for i in range(m):
                if mask & (1 << i):
                    s += elements[i]
            if s in sums:
                return False
            sums.add(s)
        return True
    
    # Greedy: start with [1], then find smallest next element
    best_set = [1 << i for i in range(n)]  # fallback: powers of 2
    best_max = best_set[-1]
    
    # Try greedy construction
    current = [1]
    for step in range(1, n):
        if time.time() > deadline - 5.0:
            break
        found = False
        # Try smallest possible value
        for candidate in range(current[-1] + 1, max_val + 1):
            if time.time() > deadline - 5.0:
                break
            test_set = current + [candidate]
            if check_distinct_sums(test_set):
                current.append(candidate)
                found = True
                break
        if not found:
            break
    
    if len(current) == n:
        if current[-1] < best_max:
            best_set = current
            best_max = current[-1]
    
    # Try to improve by local search / different starting points
    # Try different first elements
    for first in range(1, 6):
        if time.time() > deadline - 5.0:
            break
        current = [first]
        valid = True
        for step in range(1, n):
            if time.time() > deadline - 5.0:
                valid = False
                break
            found = False
            for candidate in range(current[-1] + 1, max_val + 1):
                if time.time() > deadline - 5.0:
                    break
                test_set = current + [candidate]
                if check_distinct_sums(test_set):
                    current.append(candidate)
                    found = True
                    break
            if not found:
                valid = False
                break
        if valid and len(current) == n:
            if current[-1] < best_max:
                best_set = current
                best_max = current[-1]
    
    # Local optimization: try to reduce individual elements
    improved = True
    while improved and time.time() < deadline - 10.0:
        improved = False
        for i in range(n):
            if time.time() > deadline - 10.0:
                break
            orig = best_set[i]
            for candidate in range(1, orig):
                if candidate == best_set[i]:
                    continue
                # Check distinctness
                new_set = best_set[:]
                new_set[i] = candidate
                # Check all elements are distinct
                if len(set(new_set)) != n:
                    continue
                if new_set[i] <= 0:
                    continue
                if check_distinct_sums(new_set):
                    if new_set[-1] < best_set[-1] or (new_set[-1] == best_set[-1] and sum(new_set) < sum(best_set)):
                        best_set = new_set
                        improved = True
                        break
    
    return best_set
# EVOLVE-BLOCK-END
