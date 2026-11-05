"""Maximize edges of a Q3-free graph on 16 vertices using bitmask Q3 detection."""
import time
import random
import sys

sys.setrecursionlimit(100000)

N = 16
N_PAIRS = N * (N - 1) // 2

# Precompute pair index
def pidx(i, j):
    if i > j:
        i, j = j, i
    return i * 15 - i * (i - 1) // 2 - 1 + (j - i - 1)

# Precompute all 8-subsets and for each, the list of 12 edges that would form Q3
# under each of the 840 labelings. Store as bitmasks of edge indices.
# 12870 subsets * 840 labelings = 10.8M bitmasks. Too many to precompute all.

# Instead, for a given graph (as a 120-bit integer), check Q3-freeness by:
# For each 8-subset, check if it contains Q3.
# Use the 3-regular bipartite filter first.

from itertools import combinations

ALL_8_SUBSETS = list(combinations(range(N), 8))

# For each 8-subset, precompute:
# - The 28 possible edges within it (as pair indices)
# - The degree filter: which edges must be present for 3-regularity
# Actually, let's just precompute the 840 Q3 edge patterns per subset as frozensets of pair indices.
# But 10.8M * 12 entries is too much memory.

# Alternative: for each 8-subset, precompute the bipartition structure and use the 24-perm check.
# Let me just make the check as fast as possible.

def construct():
    deadline = time.time() + 145
    
    # Use adjacency as a list of 16 integers (bitmasks of neighbors)
    # adj[i] is a 16-bit integer where bit j is set if edge (i,j) exists
    
    all_edges = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    def adj_to_set(adj):
        s = set()
        for i in range(N):
            for j in range(i+1, N):
                if adj[i] & (1 << j):
                    s.add((i, j))
        return s
    
    def contains_q3_fast(adj):
        """adj: list of 16 ints. Check Q3-freeness."""
        for subset in ALL_8_SUBSETS:
            # Degree check
            ok = True
            for v in subset:
                deg = 0
                for u in subset:
                    if u != v and (adj[v] & (1 << u)):
                        deg += 1
                if deg != 3:
                    ok = False
                    break
            if not ok:
                continue
            
            # Bipartition
            v0 = subset[0]
            neighbors = [v for v in subset if v != v0 and (adj[v0] & (1 << v))]
            if len(neighbors) != 3:
                continue
            part_A = set([v0] + [v for v in subset if v != v0 and not (adj[v0] & (1 << v))])
            part_B = set(neighbors)
            if len(part_A) != 4 or len(part_B) != 4:
                continue
            
            # Verify no edges within parts
            valid = True
            for u in part_A:
                for v in part_A:
                    if u < v and (adj[u] & (1 << v)):
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                for u in part_B:
                    for v in part_B:
                        if u < v and (adj[u] & (1 << v)):
                            valid = False
                            break
                    if not valid:
                        break
            if not valid:
                continue
            
            A = tuple(sorted(part_A))
            B = tuple(sorted(part_B))
            
            # Check 24 perms
            for p in [(0,1,2,3),(0,1,3,2),(0,2,1,3),(0,2,3,1),(0,3,1,2),(0,3,2,1),
                      (1,0,2,3),(1,0,3,2),(1,2,0,3),(1,2,3,0),(1,3,0,2),(1,3,2,0),
                      (2,0,1,3),(2,0,3,1),(2,1,0,3),(2,1,3,0),(2,3,0,1),(2,3,1,0),
                      (3,0,1,2),(3,0,2,1),(3,1,0,2),(3,1,2,0),(3,2,0,1),(3,2,1,0)]:
                if ((adj[A[0]] & (1 << B[p[0]])) and (adj[A[0]] & (1 << B[p[1]])) and (adj[A[0]] & (1 << B[p[2]])) and
                    (adj[A[1]] & (1 << B[p[0]])) and (adj[A[1]] & (1 << B[p[1]])) and (adj[A[1]] & (1 << B[p[3]])) and
                    (adj[A[2]] & (1 << B[p[0]])) and (adj[A[2]] & (1 << B[p[2]])) and (adj[A[2]] & (1 << B[p[3]])) and
                    (adj[A[3]] & (1 << B[p[1]])) and (adj[A[3]] & (1 << B[p[2]])) and (adj[A[3]] & (1 << B[p[3]]))):
                    return True
        return False
    
    # Seed: K3,13
    adj = [0] * N
    for a in range(3):
        for b in range(3, 16):
            adj[a] |= (1 << b)
            adj[b] |= (1 << a)
    
    best_adj = adj[:]
    best_count = sum(adj[i].bit_count() for i in range(N)) // 2
    
    for trial in range(300):
        if time.time() > deadline - 15:
            break
        
        random.shuffle(all_edges)
        cur_adj = [0] * N
        
        for edge in all_edges:
            if time.time() > deadline - 10:
                break
            u, v = edge
            cur_adj[u] |= (1 << v)
            cur_adj[v] |= (1 << u)
            if contains_q3_fast(cur_adj):
                cur_adj[u] &= ~(1 << v)
                cur_adj[v] &= ~(1 << u)
        
        count = sum(cur_adj[i].bit_count() for i in range(N)) // 2
        if count > best_count:
            best_count = count
            best_adj = cur_adj[:]
            print(f"Trial {trial}: {best_count} edges", file=sys.stderr)
    
    # Local improvement
    for _ in range(500):
        if time.time() > deadline - 5:
            break
        missing = [e for e in all_edges if not (best_adj[e[0]] & (1 << e[1]))]
        if not missing:
            break
        random.shuffle(missing)
        improved = False
        for edge in missing:
            u, v = edge
            best_adj[u] |= (1 << v)
            best_adj[v] |= (1 << u)
            if not contains_q3_fast(best_adj):
                best_count = sum(best_adj[i].bit_count() for i in range(N)) // 2
                improved = True
                break
            else:
                best_adj[u] &= ~(1 << v)
                best_adj[v] &= ~(1 << u)
        if not improved:
            break
    
    result = []
    for i in range(N):
        for j in range(i+1, N):
            if best_adj[i] & (1 << j):
                result.append([i, j])
    return result
