import time
import random
from functools import lru_cache

_start_time = time.time()

def construct():
    deadline = _start_time + 85
    n = 12
    m = 40
    total_masks = 1 << n  # 4096
    
    # Precompute comparability: comp[i][j] = True if i and j are comparable
    # Two masks are comparable if one is a subset of the other
    # i is subset of j iff (i & j) == i, i.e., i | j == j
    # They are comparable if (i | j) == i or (i | j) == j
    
    # Build adjacency: adj[i] is a set of j such that i and j are comparable
    # We'll use bitsets for efficiency
    adj = [0] * total_masks
    for i in range(total_masks):
        for j in range(i + 1, total_masks):
            if (i | j) == i or (i | j) == j:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # Convert to list of bitsets for faster access
    # adj[i] is an integer whose bit j is 1 if i and j are comparable
    
    # We want to find a 40-node clique in this graph
    # Maximize the number of edges in the clique (which is all pairs, so it's fixed at C(40,2)=780)
    # Wait, the score is the number of comparable pairs, which for a clique of size 40 is always C(40,2)=780
    # But not all 40-node cliques exist! The graph may not have a 40-clique.
    # So we need to find the largest clique, or if no 40-clique exists, find the best family.
    
    # Actually, the problem asks for exactly 40 distinct subsets that maximize comparable pairs.
    # A "clique" in the comparability graph means all pairs are comparable.
    # But we don't need all pairs to be comparable - we just want to maximize the count.
    
    # So the problem is: select 40 vertices to maximize the number of edges among them.
    # This is the maximum k-vertex subgraph problem, which is NP-hard.
    
    # Branch and bound approach:
    # State: current set of selected vertices, candidate set of remaining vertices
    # Bound: current_score + max_possible_additional_score
    
    # Let's use a different approach: greedy + local search with better initialization
    
    # Precompute degree of each vertex (number of comparable vertices)
    degree = [bin(adj[i]).count('1') for i in range(total_masks)]
    
    # Greedy construction: start with the vertex of highest degree, then greedily add
    # vertices that have the most connections to already-selected vertices
    def greedy_construct():
        selected = set()
        remaining = set(range(total_masks))
        
        # Start with the vertex of highest degree
        best_start = max(remaining, key=lambda x: degree[x])
        selected.add(best_start)
        remaining.remove(best_start)
        
        while len(selected) < m:
            best_v = None
            best_score = -1
            for v in remaining:
                # Count how many in selected are comparable with v
                cnt = 0
                for s in selected:
                    if (v | s) == v or (v | s) == s:
                        cnt += 1
                if cnt > best_score:
                    best_score = cnt
                    best_v = v
            if best_v is None:
                break
            selected.add(best_v)
            remaining.remove(best_v)
        
        return list(selected)
    
    # Compute score of a family
    def compute_score(fam):
        cnt = 0
        for i in range(len(fam)):
            for j in range(i + 1, len(fam)):
                a, b = fam[i], fam[j]
                if (a | b) == a or (a | b) == b:
                    cnt += 1
        return cnt
    
    # Initial solution from greedy
    family = greedy_construct()
    if len(family) < m:
        # Fill with random
        seen = set(family)
        for mask in random.sample(range(total_masks), total_masks):
            if mask not in seen:
                family.append(mask)
                seen.add(mask)
                if len(family) == m:
                    break
    
    cur_score = compute_score(family)
    best_family = list(family)
    best_score = cur_score
    
    # Local search with incremental updates
    def get_degree_in_family(mask, fam, exclude_idx=None):
        cnt = 0
        for i, s in enumerate(fam):
            if i == exclude_idx:
                continue
            if (mask | s) == mask or (mask | s) == s:
                cnt += 1
        return cnt
    
    # Precompute for faster local search
    random.seed(42)
    
    # Multiple passes of local search
    for pass_num in range(20):
        if time.time() > deadline - 10:
            break
        
        improved = True
        while improved:
            if time.time() > deadline - 10:
                break
            improved = False
            for i in range(m):
                if time.time() > deadline - 10:
                    break
                old = family[i]
                # Try swapping with candidates
                # Use a pool of candidates sorted by potential
                candidates = []
                for new in range(total_masks):
                    if new == old or new in family:
                        continue
                    candidates.append(new)
                
                if not candidates:
                    continue
                
                # Sample candidates for efficiency
                sample_size = min(500, len(candidates))
                sampled = random.sample(candidates, sample_size)
                
                best_new = None
                best_delta = 0
                deg_old = get_degree_in_family(old, family, i)
                
                for new in sampled:
                    deg_new = get_degree_in_family(new, family, i)
                    delta = deg_new - deg_old
                    if delta > best_delta:
                        best_delta = delta
                        best_new = new
                
                if best_new is not None:
                    family[i] = best_new
                    cur_score += best_delta
                    improved = True
                    if cur_score > best_score:
                        best_score = cur_score
                        best_family = list(family)
                    break  # Restart from beginning
    
    # Final polish: try all possible single swaps
    for i in range(m):
        if time.time() > deadline - 5:
            break
        old = family[i]
        deg_old = get_degree_in_family(old, family, i)
        for new in range(total_masks):
            if new == old or new in family:
                continue
            deg_new = get_degree_in_family(new, family, i)
            delta = deg_new - deg_old
            if delta > 0:
                family[i] = new
                cur_score += delta
                if cur_score > best_score:
                    best_score = cur_score
                    best_family = list(family)
    
    # Convert to lists of ints
    result = []
    for mask in best_family:
        lst = [int(i) for i in range(n) if (mask >> i) & 1]
        result.append(lst)
    
    return result

if __name__ == "__main__":
    t0 = time.time()
    res = construct()
    t1 = time.time()
    assert len(res) == 40
    seen = set()
    for r in res:
        fr = frozenset(r)
        assert fr not in seen
        seen.add(fr)
        for v in r:
            assert isinstance(v, int)
            assert 0 <= v <= 11
    # Count comparables
    masks = []
    for r in res:
        m = 0
        for v in r:
            m |= (1 << v)
        masks.append(m)
    cnt = 0
    for i in range(40):
        for j in range(i + 1, 40):
            a, b = masks[i], masks[j]
            if (a | b) == a or (a | b) == b:
                cnt += 1
    print(f"Time: {t1 - t0:.2f}s, Comparables: {cnt}")
