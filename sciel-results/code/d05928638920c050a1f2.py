import time
import random
import math

def get_primes(n):
    """Return the set of prime factors of n."""
    if n == 1:
        return set()
    primes = set()
    d = 2
    while d * d <= n:
        while n % d == 0:
            primes.add(d)
            n //= d
        d += 1
    if n > 1:
        primes.add(n)
    return primes

def evaluate(set_a):
    """Calculate the union of prime factors of all pairwise sums."""
    all_primes = set()
    n = len(set_a)
    for i in range(n):
        for j in range(i + 1, n):
            s = set_a[i] + set_a[j]
            all_primes.update(get_primes(s))
    return all_primes

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s limit
    
    max_val = 10000
    target_size = 12
    
    # Precompute prime factors for all possible sums (2 to 20000)
    sum_primes_cache = [None] * (2 * max_val + 1)
    for s in range(2, 2 * max_val + 1):
        sum_primes_cache[s] = get_primes(s)
    
    best_set = None
    best_prime_count = float('inf')
    
    # Strategy: Use modular arithmetic to find residue classes where
    # pairwise sums have few prime factors
    # Try different prime sets P and find compatible residue classes mod M
    
    # For each candidate prime set, find residue classes r1, r2, ..., r12
    # such that for all i < j, (r_i + r_j) mod M is "smooth" (only has primes in P)
    
    prime_set_options = [
        [2],
        [2, 3],
        [2, 3, 5],
        [2, 3, 5, 7],
        [2, 3, 5, 7, 11],
    ]
    
    for P in prime_set_options:
        if time.time() > deadline - 15:
            break
        
        M = 1
        for p in P:
            M *= p
        
        # Find all residues r in [0, M) such that there exist 12 distinct residues
        # where all pairwise sums are P-smooth
        # A number is P-smooth if all its prime factors are in P
        
        def is_P_smooth(n, P):
            """Check if n is P-smooth (all prime factors in P)."""
            for p in P:
                while n % p == 0:
                    n //= p
            return n == 1
        
        # Find all residue pairs (r1, r2) where (r1 + r2) mod M is P-smooth
        # Build a graph where nodes are residues and edges connect compatible pairs
        compatible = [[False] * M for _ in range(M)]
        for r1 in range(M):
            for r2 in range(r1, M):
                s = (r1 + r2) % M
                if s == 0:
                    s = M  # 0 mod M means the sum is divisible by M
                if is_P_smooth(s, P):
                    compatible[r1][r2] = True
                    compatible[r2][r1] = True
        
        # Find a clique of size 12 in this graph
        # Use backtracking
        def find_clique(size, graph, M):
            """Find a clique of given size in the graph using backtracking."""
            if size <= 0:
                return []
            if size == 1:
                return [0]
            
            # Try adding nodes one by one
            for start in range(M):
                if not graph[start][start]:
                    continue
                # Check if we can build a clique of size 'size' starting with 'start'
                remaining = size - 1
                candidates = [i for i in range(start + 1, M) if graph[start][i]]
                if len(candidates) < remaining:
                    continue
                
                # Recursively find clique in the subgraph induced by candidates
                subgraph = [[False] * len(candidates) for _ in range(len(candidates))]
                for i in range(len(candidates)):
                    for j in range(i + 1, len(candidates)):
                        subgraph[i][j] = graph[candidates[i]][candidates[j]]
                        subgraph[j][i] = graph[candidates[i]][candidates[j]]
                
                # Check if subgraph has a clique of size 'remaining'
                # Simple DFS
                def dfs_clique(idx, current_clique, subgraph, remaining):
                    if len(current_clique) == remaining:
                        return current_clique[:]
                    if idx >= len(subgraph):
                        return None
                    # Option 1: skip idx
                    result = dfs_clique(idx + 1, current_clique, subgraph, remaining)
                    if result is not None:
                        return result
                    # Option 2: include idx if compatible with all in current_clique
                    if all(subgraph[idx][j] for j in current_clique):
                        result = dfs_clique(idx + 1, current_clique + [idx], subgraph, remaining)
                        if result is not None:
                            return result
                    return None
                
                sub_clique = dfs_clique(0, [], subgraph, remaining)
                if sub_clique is not None:
                    return [start] + [candidates[i] for i in sub_clique]
            
            return None
        
        residue_clique = find_clique(target_size, compatible, M)
        
        if residue_clique is not None:
            # We found 12 compatible residues. Now find actual integers in these classes.
            # For each residue r, the integers are r, r+M, r+2M, ..., up to max_val
            # We need to pick 12 distinct integers, one from each residue class
            
            residue_classes = []
            for r in residue_clique:
                # All integers x in [1, max_val] such that x % M == r
                # If r == 0, integers are M, 2M, 3M, ...
                # If r > 0, integers are r, r+M, r+2M, ...
                if r == 0:
                    start = M
                else:
                    start = r
                nums = []
                x = start
                while x <= max_val:
                    nums.append(x)
                    x += M
                if nums:
                    residue_classes.append(nums)
            
            if len(residue_classes) >= target_size:
                # Greedily pick one from each class to minimize total primes
                # Try multiple random selections
                for attempt in range(100):
                    if time.time() > deadline - 10:
                        break
                    
                    selected = []
                    valid = True
                    for i in range(target_size):
                        if not residue_classes[i]:
                            valid = False
                            break
                        # Pick a random element from this class
                        idx = random.randint(0, len(residue_classes[i]) - 1)
                        val = residue_classes[i][idx]
                        if val in selected:
                            valid = False
                            break
                        selected.append(val)
                    
                    if not valid or len(set(selected)) != target_size:
                        continue
                    
                    primes = evaluate(selected)
                    if len(primes) < best_prime_count:
                        best_prime_count = len(primes)
                        best_set = selected[:]
    
    # Fallback: if no good solution found via modular approach, use greedy
    if best_set is None:
        def greedy_construct(start_val, seed):
            """Build a set greedily, choosing each new element to minimize new primes."""
            rng = random.Random(seed)
            current_set = [start_val]
            current_primes = set()
            
            for _ in range(target_size - 1):
                best_candidate = None
                best_new_primes_count = float('inf')
                best_resulting_primes = None
                
                # Try a sample of candidates to keep it fast
                candidates = rng.sample(range(1, max_val + 1), 200)
                
                for candidate in candidates:
                    if candidate in current_set:
                        continue
                    
                    # Calculate new primes introduced
                    new_primes = set()
                    for existing in current_set:
                        s = candidate + existing
                        if s <= 2 * max_val:
                            new_primes.update(sum_primes_cache[s])
                    
                    total_primes = current_primes | new_primes
                    new_count = len(total_primes - current_primes)
                    
                    if new_count < best_new_primes_count:
                        best_new_primes_count = new_count
                        best_candidate = candidate
                        best_resulting_primes = total_primes
                
                if best_candidate is not None:
                    current_set.append(best_candidate)
                    current_primes = best_resulting_primes
            
            return current_set, current_primes
        
        for seed in range(200):
            if time.time() > deadline - 5:
                break
            
            start_val = random.Random(seed).randint(1, max_val)
            result_set, result_primes = greedy_construct(start_val, seed)
            
            if len(set(result_set)) == target_size and all(1 <= x <= max_val for x in result_set):
                actual_primes = evaluate(result_set)
                if len(actual_primes) < best_prime_count:
                    best_prime_count = len(actual_primes)
                    best_set = result_set[:]
                    if best_prime_count <= 4:
                        break
    
    if best_set is None:
        best_set = random.sample(range(1, max_val + 1), target_size)
        best_prime_count = len(evaluate(best_set))
    
    # Local search improvement
    if time.time() < deadline - 10:
        current_set = best_set[:]
        current_primes = evaluate(current_set)
        current_count = len(current_primes)
        
        for _ in range(3000):
            if time.time() > deadline - 5:
                break
            
            idx = random.randint(0, target_size - 1)
            old_val = current_set[idx]
            new_val = random.randint(1, max_val)
            
            if new_val == old_val:
                continue
            if new_val in current_set and current_set.index(new_val) != idx:
                continue
            
            new_set = current_set[:]
            new_set[idx] = new_val
            new_primes = evaluate(new_set)
            new_count = len(new_primes)
            
            if new_count < current_count:
                current_set = new_set
                current_primes = new_primes
                current_count = new_count
                
                if current_count < best_prime_count:
                    best_prime_count = current_count
                    best_set = current_set[:]
    
    return best_set
