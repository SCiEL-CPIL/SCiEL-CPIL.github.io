import time
import itertools

def _paley_tournament_bits(p):
    """Generate orientation bits for a Paley tournament of size p.
    For each pair (i,j) with i<j in lexicographic order, bit is 0 if i->j, 1 if j->i.
    Paley: i -> j iff (j-i) is a nonzero quadratic residue mod p."""
    p2 = (p - 1) // 2
    # Precompute which residues are squares
    is_sq = [False] * p
    for x in range(1, p):
        is_sq[(x * x) % p] = True
    
    # Build orientation as a list of bits in lexicographic order
    orientation = [0] * (p * (p - 1) // 2)
    idx = 0
    for i in range(p):
        for j in range(i + 1, p):
            diff = (j - i) % p
            # 0 means i -> j, which happens if diff is a square
            orientation[idx] = 0 if is_sq[diff] else 1
            idx += 1
    return orientation

def _check_validity_fast(n, orientation):
    """Check if every 4-set has a dominator. Returns True if valid."""
    # Build adjacency list: adj[u] is a list of v such that u->v
    adj = [[] for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if orientation[idx] == 0:
                # i -> j
                adj[i].append(j)
            else:
                # j -> i
                adj[j].append(i)
            idx += 1
    
    # Precompute all 4-sets and their indices
    all_4sets = list(itertools.combinations(range(n), 4))
    num_4sets = len(all_4sets)
    idx_map = {t: i for i, t in enumerate(all_4sets)}
    
    covered = bytearray(num_4sets)
    
    # For each vertex u, mark all 4-sets in its out-neighborhood
    for u in range(n):
        neighbors = adj[u]
        # If out-degree < 4, it can't dominate any 4-set
        if len(neighbors) < 4:
            continue
        # Generate all 4-subsets of neighbors
        for combo in itertools.combinations(neighbors, 4):
            # combo is already sorted because neighbors are generated in increasing order?
            # No, neighbors list is built by iterating j > i, so for a fixed i, j increases.
            # So adj[i] is sorted.
            # Thus combo from itertools.combinations(sorted_list, 4) is sorted.
            tup = tuple(combo)
            idx = idx_map[tup]
            covered[idx] = 1
            
    # Check if all are covered
    for val in covered:
        if val == 0:
            return False
    return True

def construct():
    deadline = time.time() + 110  # 10s safety margin
    
    # Try smaller Paley tournaments first (primes p ≡ 3 mod 4)
    # Candidates based on counting argument:
    # p=31: possible
    # p=43: possible
    # p=47: possible
    # p=59: possible
    # p=67: known to work
    candidates = [31, 43, 47, 59, 67]
    
    for p in candidates:
        if time.time() > deadline:
            break
        
        # Generate bits
        orientation = _paley_tournament_bits(p)
        
        # Verify
        if _check_validity_fast(p, orientation):
            return {'n': p, 'orientation': orientation}
            
    # Fallback to 67 if verification failed or timed out (should not happen for 67)
    p = 67
    orientation = _paley_tournament_bits(p)
    return {'n': p, 'orientation': orientation}
