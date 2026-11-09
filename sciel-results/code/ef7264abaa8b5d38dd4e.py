"""
Few prime factors across pairwise sums.

Goal: choose 12 distinct positive integers <= 10000 such that the number of
distinct primes dividing any sum a_i + a_j (i != j) is as small as possible.

Strategy:
- Precompute the set of "good" numbers: integers x in [1, 10000] such that
  for some small prime set P (e.g., {2,3,5,7}), every sum x + y for y in a
  candidate pool is P-smooth (i.e., has no prime factors outside P).
  
  A simpler and more tractable approach: 
  - Restrict the pool to numbers that are "compatible" with many others.
  - Build a graph where edge (a, b) exists if a + b is P-smooth for P = {2,3,5,7}.
  - Find a clique of size 12 in this graph.
  
- If a 12-clique exists with P = {2,3,5,7}, we achieve score 4 (only 4 distinct primes).
- If not, try P = {2,3,5,7,11} (score 5), etc.

To stay within time limits:
- Precompute smoothness for sums up to 20000 using a sieve-like method.
- Use backtracking with forward checking to find a clique of size 12.
- If no clique found quickly, fall back to a greedy heuristic.

We'll use P = {2, 3, 5, 7} first (4 primes). If that fails, expand.
"""
import time
import random
from collections import defaultdict

_MAX = 10000
_N = 12
_DEADLINE = 85  # seconds, leave 5s margin

def _is_smooth(n, primes):
    """Check if n is smooth over given primes."""
    for p in primes:
        while n % p == 0:
            n //= p
    return n == 1

def _precompute_smooth(max_val, primes):
    """Precompute which numbers up to max_val are smooth over primes."""
    smooth = [False] * (max_val + 1)
    for i in range(1, max_val + 1):
        smooth[i] = _is_smooth(i, primes)
    return smooth

def _build_compatibility_graph(numbers, primes, smooth_table):
    """Build adjacency: a and b are compatible if a+b is smooth."""
    n = len(numbers)
    # Map value to index
    val_to_idx = {v: i for i, v in enumerate(numbers)}
    adj = [set() for _ in range(n)]
    
    for i in range(n):
        for j in range(i + 1, n):
            s = numbers[i] + numbers[j]
            if s <= max(smooth_table) and smooth_table[s]:
                adj[i].add(j)
                adj[j].add(i)
    return adj

def _find_clique(adj, size):
    """Backtracking with forward checking to find a clique of given size."""
    n = len(adj)
    # Sort by degree (descending) to guide search
    order = sorted(range(n), key=lambda x: -len(adj[x]))
    
    # Remap to new indices
    remap = {old: new for new, old in enumerate(order)}
    new_adj = [set() for _ in range(n)]
    for i in range(n):
        for j in adj[i]:
            new_adj[remap[i]].add(remap[j])
    
    adj = new_adj
    n = len(adj)
    
    best_clique = []
    
    def backtrack(current_clique, candidates):
        nonlocal best_clique
        if len(current_clique) == size:
            best_clique = current_clique[:]
            return True
        if not candidates:
            return False
        
        # Forward checking: filter candidates that are connected to all in current_clique
        filtered = []
        for c in candidates:
            if all(c in adj[x] for x in current_clique):
                filtered.append(c)
        
        if len(filtered) < size - len(current_clique):
            return False
        
        # Try each candidate
        for c in filtered:
            # New candidates must be connected to c
            new_candidates = [x for x in filtered if x in adj[c] and x > c]
            # Actually we should pass all that are > c and connected to all in clique
            new_candidates = [x for x in filtered if x in adj[c]]
            if backtrack(current_clique + [c], new_candidates):
                return True
        return False
    
    backtrack([], list(range(n)))
    return best_clique

def _greedy_clique(adj, size):
    """Greedy approach: iteratively add the node with max degree to remaining candidates."""
    n = len(adj)
    # Start with the highest degree node
    order = sorted(range(n), key=lambda x: -len(adj[x]))
    current = [order[0]]
    candidates = set(order[1:])
    
    while len(current) < size:
        # For each candidate, count how many it's connected to in current
        best_node = None
        best_score = -1
        for c in candidates:
            score = sum(1 for x in current if c in adj[x])
            if score == len(current) and score > best_score:
                best_score = score
                best_node = c
        if best_node is None:
            break
        current.append(best_node)
        candidates.discard(best_node)
        # Filter candidates: must be connected to best_node
        candidates = {c for c in candidates if c in adj[best_node]}
    
    return current if len(current) == size else []

def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE
    
    # Try with increasing prime sets
    prime_sets = [
        [2, 3, 5, 7],
        [2, 3, 5, 7, 11],
        [2, 3, 5, 7, 11, 13],
    ]
    
    for primes in prime_sets:
        if time.time() > deadline:
            break
        
        # Precompute smooth table up to 2*MAX
        smooth_table = _precompute_smooth(2 * _MAX, primes)
        
        # Find candidate numbers: those that have at least _N-1 smooth sums
        # i.e., for number x, count how many y in [1, MAX] have x+y smooth
        # To speed up, only consider x where many sums are smooth
        
        # Build list of candidate numbers
        candidates = []
        for x in range(1, _MAX + 1):
            count = 0
            for y in range(1, _MAX + 1):
                if x == y:
                    continue
                s = x + y
                if smooth_table[s]:
                    count += 1
            if count >= _N - 1:
                candidates.append(x)
        
        if len(candidates) < _N:
            continue
        
        # Build compatibility graph on candidates
        adj = [set() for _ in range(len(candidates))]
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                s = candidates[i] + candidates[j]
                if smooth_table[s]:
                    adj[i].add(j)
                    adj[j].add(i)
        
        # Try to find a clique of size _N
        clique = _find_clique(adj, _N)
        
        if len(clique) == _N:
            result = [candidates[i] for i in clique]
            result.sort()
            # Verify: all pairs have smooth sums
            valid = True
            for i in range(len(result)):
                for j in range(i + 1, len(result)):
                    s = result[i] + result[j]
                    if not smooth_table[s]:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                return result
        
        # Try greedy if backtracking failed
        clique = _greedy_clique(adj, _N)
        if len(clique) == _N:
            result = [candidates[i] for i in clique]
            result.sort()
            valid = True
            for i in range(len(result)):
                for j in range(i + 1, len(result)):
                    s = result[i] + result[j]
                    if not smooth_table[s]:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                return result
    
    # Fallback: return a set that minimizes distinct primes using a heuristic
    # Use numbers that are powers of 2: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048
    # Sums will have prime factors that are limited
    result = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    return result
