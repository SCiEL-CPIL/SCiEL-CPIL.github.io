import time
import random
from math import gcd

# EVOLVE-BLOCK-START
def construct():
    """
    Find a large subset of [1, 128] with no three distinct elements whose
    three pairwise gcds are all equal.
    
    Strategy: 
    1. Precompute conflict triples.
    2. Use a bitset-based representation for fast set operations.
    3. Initialize with a strong greedy heuristic.
    4. Perform extensive local search with 1-removal and 2-removal moves.
    5. Use random restarts if time permits.
    """
    N = 128
    deadline = time.time() + 85.0  # safety margin

    # Precompute gcd table
    gt = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            gt[i][j] = g
            gt[j][i] = g

    # Identify bad triples
    # A triple (a, b, c) is bad if gt[a][b] == gt[b][c] == gt[c][a]
    # We store for each vertex x, the set of pairs (u, v) such that {x, u, v} is bad.
    # To make lookup fast, we can represent the set of "neighbors" u such that (v, u) is a conflict.
    # Actually, for a candidate set S, x can be added if for all u, v in S, (u, v) is not a conflict pair for x.
    # This is equivalent to checking if any bad triple exists within S.
    
    # Let's precompute for each x, a list of (u, v) pairs.
    # Since N is small, we can also just precompute a 3D boolean tensor or a set of frozensets.
    # Set of frozensets is easier to debug but slower.
    # Let's use a bitset for the set S.
    # Conflict check: for x not in S, check all pairs (u, v) in S.
    # This is O(|S|^2) per check.
    
    # Optimization: For each x, precompute a bitset `conflict_mask[x]` is not enough because conflicts are pairwise with respect to x.
    # A better precomputation:
    # For each x, and each u, let `bad_v[x][u]` be a bitset of v such that {x, u, v} is bad.
    # Then x can be added to S if for all u in S, (bad_v[x][u] & S) is empty? No, that's not right.
    # We need to ensure no pair (u, v) in S is in the conflict list of x.
    # Conflict list of x is a set of pairs P_x = {(u, v) | {x, u, v} is bad}.
    # x is compatible with S if P_x contains no pair (u, v) where u, v in S.
    
    # Let's precompute P_x as a list of (u, v).
    conflict_pairs = {i: [] for i in range(1, N + 1)}
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            g_ab = gt[a][b]
            for c in range(b + 1, N + 1):
                if gt[a][c] == g_ab and gt[b][c] == g_ab:
                    # Triple (a, b, c) is bad
                    conflict_pairs[a].append((b, c))
                    conflict_pairs[b].append((a, c))
                    conflict_pairs[c].append((a, b))
                    
    def can_add(x, S):
        """S is a set of vertices. Returns True if x can be added."""
        # S is a Python set for simplicity, but we might switch to bitmask if needed.
        # With N=128, a Python set is fine.
        for u, v in conflict_pairs[x]:
            if u in S and v in S:
                return False
        return True

    def polish(S):
        """Greedy addition of any compatible vertex."""
        S = set(S)
        changed = True
        while changed:
            changed = False
            for x in range(1, N + 1):
                if x not in S and can_add(x, S):
                    S.add(x)
                    changed = True
        return S

    def local_search(S, max_iters=10000):
        """
        Local search with 1-removal and 2-removal moves.
        """
        S = set(S)
        best_size = len(S)
        best_S = set(S)
        
        # Precompute degrees or just iterate
        all_vertices = set(range(1, N + 1))
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
                
            improved = False
            
            # Try adding
            for x in all_vertices - S:
                if can_add(x, S):
                    S.add(x)
                    improved = True
                    if len(S) > best_size:
                        best_size = len(S)
                        best_S = set(S)
                    break # Restart add loop
            
            if improved:
                continue
                
            # Try removing 1 and adding
            # We want to remove x and add y (or more) if it leads to improvement
            # Since adding is greedy, we just remove x and polish.
            # But polishing is O(N^2) or more.
            # Let's try removing x, then greedily adding.
            
            # To speed up, only consider removing x if it's "blocking" many vertices?
            # Or just random.
            
            # 1-removal move
            removed = random.choice(list(S))
            S.remove(removed)
            # Greedily add
            added_any = False
            for x in all_vertices - S:
                if can_add(x, S):
                    S.add(x)
                    added_any = True
                    # Don't break, keep adding to maximize immediate gain
            # Actually, the previous `polish` does this. Let's use it.
            # But we want to track if it's better.
            
            # This approach is slow because `polish` is expensive.
            # Let's refine the move:
            # Remove x. Check if any y in (all_vertices - S) can be added.
            # If yes, add y.
            # If size increases, keep it. If same, keep with prob.
            
            # Let's implement a simpler 1-swap or 1-add-1-remove
            # Current S is maximal (no adds possible).
            # Try remove x, add y.
            
            # We iterate over all x in S.
            # For each x, find y not in S such that {y} is compatible with S - {x}.
            # This is expensive.
            
            # Alternative: Random restarts of greedy with different orders.
            pass

    # Given the complexity of efficient local search in Python without C extensions,
    # we rely on multiple random restarts of a smart greedy algorithm.
    
    best_S = set()
    best_size = 0
    
    # Initial strong seed: All primes and prime powers?
    # Or just random.
    
    # Let's try a specific structural seed.
    # Numbers with few prime factors?
    
    # Strategy: Randomized Greedy with Backtracking-like restarts.
    
    for trial in range(1000):
        if time.time() > deadline - 5:
            break
            
        rng = random.Random(42 + trial)
        order = list(range(1, N + 1))
        rng.shuffle(order)
        
        S = set()
        for x in order:
            if can_add(x, S):
                S.add(x)
                
        S = polish(S)
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)
            
        # If we found a good one, do some local search on it?
        # For now, just accumulate best greedy result.

    # If we have time, do a local search on the best result
    if time.time() < deadline - 10 and best_size > 0:
        S_local = set(best_S)
        # Simple 1-removal local search
        for _ in range(500):
            if time.time() > deadline - 5:
                break
            improved = False
            # Try removing each element and seeing if we can add more
            # This is O(N * (cost of can_add))
            # Cost of can_add is O(degree).
            
            # Let's just do a few random removals and polishes
            for _ in range(10):
                if time.time() > deadline - 5:
                    break
                to_remove = random.choice(list(S_local))
                S_local.remove(to_remove)
                S_new = polish(S_local)
                if len(S_new) > len(S_local):
                    S_local = S_new
                    improved = True
                elif len(S_new) == len(S_local) and random.random() < 0.5:
                    S_local = S_new
                else:
                    S_local.add(to_remove) # Restore if not improved? 
                    # Actually, if we removed to_remove, we should check if adding it back immediately is best.
                    # If S_new is not better, we lost to_remove.
                    # Let's restore to_remove if S_new is not strictly better.
                    # But polish(S_local) after removal might have added others.
                    # If we restore to_remove, we might block those.
                    # So if not improved, we discard the move.
                    pass
                    
            if len(S_local) > best_size:
                best_size = len(S_local)
                best_S = set(S_local)
                
    return sorted(best_S)
# EVOLVE-BLOCK-END
