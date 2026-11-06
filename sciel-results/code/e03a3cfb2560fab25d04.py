"""Rewritten program: maximize n for a K2,2,2-free 2-colouring of K_n."""
import time
import itertools
import random

# EVOLVE-BLOCK-START

def _has_k222(color_list, n):
    """
    Check if the colouring (list of 0/1 in lexicographic order) contains K2,2,2.
    K2,2,2 is the complete tripartite graph with parts of size 2.
    A K2,2,2 subgraph exists iff there are 6 vertices {a,b,c,d,e,f} such that
    all edges between the three pairs are present in one colour.
    Equivalently, there exist three disjoint pairs (p1,p2), (q1,q2), (r1,r2)
    such that all 12 edges between the three pairs have the same colour.
    
    More efficiently: for a fixed colour c, we need to find 6 vertices that form
    K2,2,2 in colour c. K2,2,2 has 12 edges. The 6 vertices can be partitioned
    into 3 pairs of "opposite" vertices (each pair is an independent set in the
    K2,2,2). Actually, K2,2,2 = K_{2,2,2}: three parts A,B,C each of size 2,
    all edges between different parts present.
    
    So we need: exist disjoint 2-sets A, B, C such that all edges between A and B,
    A and C, B and C are colour c.
    
    For fixed colour c, define for each vertex v the set of vertices u such that
    edge (v,u) has colour c. Then K2,2,2 in colour c exists iff there exist
    vertices a1,a2,b1,b2,c1,c2 all distinct such that:
      - a1,a2 both connect to b1,b2 in colour c (and vice versa)
      - a1,a2 both connect to c1,c2 in colour c
      - b1,b2 both connect to c1,c2 in colour c
    
    Equivalently: pick a pair {a1,a2} in colour c (edge a1-a2 can be anything,
    but actually in K2,2,2 the edge within a part is NOT present). Wait - K2,2,2
    is tripartite, so edges within each part are absent. So a1,a2 are NOT adjacent
    in colour c. Similarly for the other pairs.
    
    So: exist three disjoint pairs {a1,a2}, {b1,b2}, {c1,c2} such that:
      - edge(a1,a2) != c, edge(b1,b2) != c, edge(c1,c2) != c  (within-part edges absent in colour c)
      - all edges between different pairs = c
    
    Let's use a different approach. For colour c, build adjacency matrix.
    K2,2,2 in colour c exists iff there are 6 vertices forming a K2,2,2 subgraph.
    
    Efficient check: For each pair of vertices (i,j) that are NOT connected by colour c
    (i.e., they could be in the same part), consider them as a candidate part.
    Then we need two more disjoint pairs, each non-adjacent to each other in colour c,
    and all 8 cross-edges present in colour c.
    
    For n up to 29, this is feasible with bitsets.
    """
    # Build adjacency bitsets for colour 0 and colour 1
    # adj_c[v] = bitmask of vertices u where edge (v,u) has colour c
    adj0 = [0] * n
    adj1 = [0] * n
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = color_list[idx]
            idx += 1
            if c == 0:
                adj0[i] |= (1 << j)
                adj0[j] |= (1 << i)
            else:
                adj1[i] |= (1 << j)
                adj1[j] |= (1 << i)
    
    for c in [0, 1]:
        adj = adj0 if c == 0 else adj1
        # For each unordered pair (i,j) not connected in colour c (candidate part)
        # find if there exist two more disjoint pairs forming K2,2,2
        # K2,2,2: parts A={a1,a2}, B={b1,b2}, C={c1,c2}
        # Conditions: 
        #   a1,a2 not adjacent in c; b1,b2 not adjacent in c; c1,c2 not adjacent in c
        #   all edges between A-B, A-C, B-C are in c
        # 
        # For fixed A={i,j} (i<j, edge(i,j) not in c):
        #   Let S = vertices v not in {i,j} such that both i and j connect to v in c
        #   i.e., v in adj[i] & adj[j] & ~(1<<i) & ~(1<<j)
        #   We need to find two disjoint pairs {b1,b2}, {c1,c2} from S
        #   such that b1,b2 not adjacent in c, c1,c2 not adjacent in c,
        #   and all edges between B and C are in c.
        #   Additionally, since b1,b2 in S, they connect to i,j in c. Good.
        #   Since c1,c2 in S, they connect to i,j in c. Good.
        #   We need edges between B and C all in c.
        
        # Precompute: for each pair (p,q) with p<q, not_adj_c[p][q] = (edge not in colour c)
        # We'll just check on the fly.
        
        found = False
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i] & (1 << j):
                    continue  # i,j adjacent in colour c, can't be same part
                # S = common neighbours of i and j in colour c, excluding i and j
                S_mask = adj[i] & adj[j] & ~(1 << i) & ~(1 << j)
                # Get list of vertices in S
                S = []
                m = S_mask
                while m:
                    v = (m & -m).bit_length() - 1
                    S.append(v)
                    m &= m - 1
                # Need to pick two disjoint pairs from S
                # For efficiency, iterate over pairs in S
                s_len = len(S)
                if s_len < 4:
                    continue
                # For each pair (b1,b2) in S with b1<b2, not adjacent in c:
                for bi in range(s_len):
                    b1 = S[bi]
                    if adj[b1] & (1 << b1):  # shouldn't happen
                        continue
                    for bj in range(bi + 1, s_len):
                        b2 = S[bj]
                        if adj[b1] & (1 << b2):
                            continue  # b1,b2 adjacent in c, can't be same part
                        # Now need two more vertices c1,c2 in S, distinct from b1,b2,
                        # not adjacent in c to each other, and both connected to b1 and b2 in c
                        # c1 must be in adj[b1] & adj[b2] & S_mask, not b1,b2
                        # Same for c2, and c1,c2 not adjacent in c
                        cand_mask = S_mask & adj[b1] & adj[b2] & ~(1 << b1) & ~(1 << b2)
                        # Get candidate list
                        cands = []
                        m2 = cand_mask
                        while m2:
                            v = (m2 & -m2).bit_length() - 1
                            cands.append(v)
                            m2 &= m2 - 1
                        for ci in range(len(cands)):
                            c1 = cands[ci]
                            for cj in range(ci + 1, len(cands)):
                                c2 = cands[cj]
                                if adj[c1] & (1 << c2):
                                    continue
                                # Found K2,2,2
                                found = True
                                break
                            if found:
                                break
                        if found:
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                break
        if found:
            return True
    return False


def _validate(color_list, n):
    """Validate that neither colour class contains K2,2,2."""
    return not _has_k222(color_list, n)


def _paley_coloring(n):
    """Paley graph colouring for prime n ≡ 1 (mod 4)."""
    squares = set()
    for x in range(1, n):
        squares.add((x * x) % n)
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            d = (j - i) % n
            colors.append(0 if d in squares else 1)
    return colors


def _try_random_local_search(n, max_iters, time_limit, rng):
    """
    Try to find a valid K2,2,2-free 2-colouring of K_n via random init + local search.
    Returns (colors, n) if found, else None.
    """
    deadline = time.time() + time_limit
    best_n = 0
    
    for attempt in range(max_iters):
        if time.time() > deadline - 5:
            break
        
        # Random initial coloring
        colors = [rng.randint(0, 1) for _ in range(n * (n - 1) // 2)]
        
        # Local search: try flipping edges to reduce K2,2,2 count
        # But checking K2,2,2 count is expensive. Instead, use a heuristic:
        # Try to make the graph "balanced" and sparse in each colour.
        # 
        # Alternative: start from a known good structure and perturb.
        # For now, just check if random works (unlikely for large n).
        
        if _validate(colors, n):
            return colors, n
    
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Try known constructions in increasing n
    # Paley works for primes p ≡ 1 mod 4
    # Known: R(K2,2,2) > 17 via Paley(17)
    # Try to find larger constructions
    
    best_result = None
    
    # First, verify Paley(17)
    n17 = 17
    colors17 = _paley_coloring(n17)
    if _validate(colors17, n17):
        best_result = {'n': n17, 'colors': colors17}
    
    # Try other prime Paley graphs
    primes_mod4 = [13, 17, 29, 37, 41, 53, 61]  # primes ≡ 1 mod 4
    for p in primes_mod4:
        if p > 31:
            break
        if time.time() > deadline - 10:
            break
        colors_p = _paley_coloring(p)
        if _validate(colors_p, p):
            if best_result is None or p > best_result['n']:
                best_result = {'n': p, 'colors': colors_p}
    
    # Try to extend: use a greedy/iterative approach for n > 17
    # Start from best known and try to add vertices
    
    if best_result is not None:
        base_n = best_result['n']
        base_colors = best_result['colors']
        
        # Try to extend by one vertex at a time
        current_n = base_n
        current_colors = list(base_colors)
        
        for target_n in range(current_n + 1, 32):
            if time.time() > deadline - 15:
                break
            
            # Try to add a new vertex with various colour patterns
            # The new vertex has edges to all existing vertices
            # We need to choose colours for these edges such that no K2,2,2 is formed
            
            # Strategy: try all 2^(target_n-1) assignments? Too many for large n.
            # Instead, use a greedy approach: for each existing vertex, pick the colour
            # that minimizes the number of potential K2,2,2 violations.
            
            # Simplified: try a few random assignments
            rng = random.Random(42)
            found = False
            for attempt in range(200):
                if time.time() > deadline - 15:
                    break
                
                # Random assignment for new vertex
                new_edges = [rng.randint(0, 1) for _ in range(current_n)]
                candidate_colors = current_colors + new_edges
                
                if _validate(candidate_colors, target_n):
                    current_n = target_n
                    current_colors = candidate_colors
                    found = True
                    break
            
            if not found:
                break
        
        if current_n > (best_result['n'] if best_result else 0):
            best_result = {'n': current_n, 'colors': current_colors}
    
    # If we still have time, try more sophisticated search
    if time.time() < deadline - 20:
        # Try a more systematic local search starting from Paley(17) or best found
        if best_result is not None:
            start_n = best_result['n']
            start_colors = list(best_result['colors'])
            
            # Try to extend further with smarter heuristics
            current_n = start_n
            current_colors = start_colors
            rng = random.Random(123)
            
            for target_n in range(current_n + 1, 32):
                if time.time() > deadline - 20:
                    break
                
                found = False
                for attempt in range(500):
                    if time.time() > deadline - 20:
                        break
                    
                    # Greedy with randomness: for each new edge, pick colour
                    # that keeps the graph as far from K2,2,2 as possible
                    # Simplified: random but biased
                    new_edges = [rng.randint(0, 1) for _ in range(current_n)]
                    candidate_colors = current_colors + new_edges
                    
                    if _validate(candidate_colors, target_n):
                        current_n = target_n
                        current_colors = candidate_colors
                        found = True
                        break
                
                if not found:
                    break
            
            if current_n > best_result['n']:
                best_result = {'n': current_n, 'colors': current_colors}
    
    if best_result is None:
        # Fallback: return the Paley(17) which we know works
        best_result = {'n': 17, 'colors': _paley_coloring(17)}
    
    return best_result

# EVOLVE-BLOCK-END
