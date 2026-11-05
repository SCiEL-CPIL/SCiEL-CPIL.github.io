"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill

    # Range [2, 256]
    nums = list(range(2, 257))
    n = len(nums)
    weights = [1.0 / x for x in nums]

    # Identify conflicting pairs where lcm(a, b) <= 256
    conflict_pairs = []
    for i in range(n):
        ai = nums[i]
        for j in range(i + 1, n):
            aj = nums[j]
            if _lcm(ai, aj) <= 256:
                conflict_pairs.append((i, j))

    if time.time() - start_time > deadline:
        return [2]

    # Build conflict graph as adjacency sets
    adj = [set() for _ in range(n)]
    for i, j in conflict_pairs:
        adj[i].add(j)
        adj[j].add(i)
    
    # Omega function (number of prime factors with multiplicity)
    def omega(x):
        count = 0
        d = 2
        while d * d <= x:
            while x % d == 0:
                count += 1
                x //= d
            d += 1
        if x > 1:
            count += 1
        return count

    omega_parity = [omega(x) % 2 for x in nums]
    
    # Partition by Omega parity
    even_nodes = [i for i in range(n) if omega_parity[i] == 0]
    odd_nodes = [i for i in range(n) if omega_parity[i] == 1]
    
    # Sort each partition by weight descending (smaller number = larger weight)
    even_nodes.sort(key=lambda i: -weights[i])
    odd_nodes.sort(key=lambda i: -weights[i])
    
    selected = set()
    
    def can_add(idx, sel):
        for s in sel:
            if s in adj[idx]:
                return False
        return True
    
    # Phase 1: Greedy add from even-omega nodes
    for idx in even_nodes:
        if can_add(idx, selected):
            selected.add(idx)
        if time.time() - start_time > deadline:
            break
    
    # Phase 2: Greedy add from odd-omega nodes
    if time.time() - start_time < deadline:
        for idx in odd_nodes:
            if can_add(idx, selected):
                selected.add(idx)
            if time.time() - start_time > deadline:
                break
    
    # Local search: try to add any remaining node
    if time.time() - start_time < deadline:
        improved = True
        while improved and time.time() - start_time < deadline:
            improved = False
            for idx in range(n):
                if idx in selected:
                    continue
                if can_add(idx, selected):
                    selected.add(idx)
                    improved = True
    
    # Local search: 1-for-1 swaps
    if time.time() - start_time < deadline:
        improved = True
        while improved and time.time() - start_time < deadline:
            improved = False
            for idx in range(n):
                if idx in selected:
                    continue
                # Try to find a selected node s such that removing s allows adding idx
                for s in list(selected):
                    # Check if idx conflicts with any node in (selected - {s})
                    if can_add(idx, selected - {s}):
                        gain = weights[idx] - weights[s]
                        if gain > 1e-15:  # Only swap if it improves
                            selected.discard(s)
                            selected.add(idx)
                            improved = True
                            break
                if improved:
                    break
    
    if not selected:
        return [256]
    
    return [nums[i] for i in selected]

# EVOLVE-BLOCK-END
