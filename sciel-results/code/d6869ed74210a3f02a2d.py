"""Hexagon-free graph construction on 40 vertices – SAT-based edge maximization."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

N = 40
ALL_PAIRS = [(i, j) for i in range(N) for j in range(i + 1, N)]
NUM_PAIRS = len(ALL_PAIRS)

def _build_adj(edges):
    adj = [set() for _ in range(N)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _has_c6_adj(adj, n):
    """Check if graph with adjacency sets has a 6-cycle."""
    for u in range(n):
        for v in adj[u]:
            if v <= u:
                continue
            for w in adj[v]:
                if w == u:
                    continue
                for x in adj[w]:
                    if x == u or x == v:
                        continue
                    for y in adj[x]:
                        if y == u or y == v or y == w:
                            continue
                        for z in adj[y]:
                            if z == u or z == v or z == w or z == x:
                                continue
                            if u in adj[z]:
                                return True
    return False

def _seed_graph_8K5():
    """8 disjoint K5s: 100 edges."""
    edges = []
    for k in range(8):
        base = 5 * k
        for a in range(5):
            for b in range(a + 1, 5):
                edges.append([base + a, base + b])
    return edges

def _precompute_c6_constraints():
    """
    For each pair of edges (e1, e2) that share a vertex, compute the set of
    edge indices that would complete a C6 through those two edges.
    Actually, we need: for each triple of edge positions in a potential C6,
    the constraint is that not all three can be present.
    
    A C6: v0-v1-v2-v3-v4-v5-v0 (6 edges)
    Edges: (v0,v1), (v1,v2), (v2,v3), (v3,v4), (v4,v5), (v5,v0)
    
    For SAT: for every 6-tuple of distinct vertices (v0..v5) forming a cycle,
    we add a clause: NOT(all 6 edges present).
    
    But 40!/(40-6)! is too many. Instead, we use a smarter approach:
    Precompute for each pair of edges that are "compatible" (share a vertex or 
    are adjacent in a path), what other edges would complete a C6.
    
    Actually, for the backtracking approach, we'll use an incremental check.
    Let's precompute all C6s as tuples of 6 edge indices.
    """
    # Enumerate all 6-cycles as sets of 6 edge indices
    # A 6-cycle is 6 vertices v0,v1,v2,v3,v4,v5 with edges (v0,v1),(v1,v2),...,(v5,v0)
    # We need all distinct 6-vertex cycles.
    # Number of 6-cycles in K40: C(40,6) * 5! / 2 = 3,838,380 * 60 / 2 = 115,151,400
    # That's too many to enumerate upfront.
    
    # Instead, for the backtracking solver, we'll use incremental C6 checking.
    # Precompute for each edge index, which other edges it's "adjacent to" 
    # (share a vertex) to speed up the check.
    
    # For each edge (u,v), precompute neighbors: edges that share u or v
    edge_adj = [[] for _ in range(NUM_PAIRS)]
    edge_to_idx = {}
    for idx, (u, v) in enumerate(ALL_PAIRS):
        edge_to_idx[(u, v)] = idx
        # Find all edges sharing u or v
        for other_idx, (a, b) in enumerate(ALL_PAIRS):
            if other_idx == idx:
                continue
            if a == u or a == v or b == u or b == v:
                edge_adj[idx].append(other_idx)
    
    return edge_adj, edge_to_idx

def _sat_backtracking_search(target_edges, edge_order, seed=42):
    """
    Backtracking search: can we select exactly target_edges edges (from the 
    given order, taking the first target_edges in order) with no C6?
    
    Actually, a better formulation: we have NUM_PAIRS edges. We want to select
    a subset of size target_edges with no C6. We process edges in a fixed order
    and decide include/exclude.
    
    For efficiency: process edges in order. At each step, decide if to include.
    Use the constraint that we must select exactly target_edges total.
    """
    random.seed(seed)
    
    # We'll use a different approach: iterative deepening with greedy + repair
    # Actually, let's use a proper backtracking with pruning.
    
    # Process edges in the given order. State: which edges are selected.
    # Pruning: if selected + remaining < target, fail.
    # C6 check: incremental.
    
    adj = [set() for _ in range(N)]
    selected = [False] * NUM_PAIRS
    selected_count = 0
    target = target_edges
    
    # We need to select exactly `target` edges from NUM_PAIRS.
    # Process in order. At position i, we can include or exclude.
    # Prune: selected_count + (NUM_PAIRS - i) < target -> can't reach target
    #        selected_count > target -> too many
    
    def backtrack(i, selected_count):
        if time.time() > _deadline:
            return None
        if selected_count == target:
            # Verify no C6 (should be guaranteed by incremental check)
            return [ALL_PAIRS[j] for j in range(NUM_PAIRS) if selected[j]]
        if i == NUM_PAIRS:
            return None
        # Pruning
        remaining = NUM_PAIRS - i
        if selected_count + remaining < target:
            return None
        if selected_count > target:
            return None
        
        # Try including edge i
        u, v = ALL_PAIRS[i]
        # Check if adding (u,v) creates a C6
        if _creates_c6(adj, u, v):
            # Must exclude
            result = backtrack(i + 1, selected_count)
            return result
        else:
            # Try including
            adj[u].add(v)
            adj[v].add(u)
            selected[i] = True
            result = backtrack(i + 1, selected_count + 1)
            if result is not None:
                return result
            # Backtrack
            adj[u].discard(v)
            adj[v].discard(u)
            selected[i] = False
            
            # Try excluding
            result = backtrack(i + 1, selected_count)
            return result
    
    _deadline = time.time() + 5  # 5 second limit per SAT call
    result = backtrack(0, 0)
    return result

def _creates_c6(adj, u, v):
    """Check if adding edge (u,v) creates a C6 in the current graph."""
    # A C6 through (u,v): u-v-w-x-y-z-u
    # Need path of length 4 from v to u through w,x,y,z
    # where all of u,v,w,x,y,z are distinct
    for w in adj[v]:
        if w == u:
            continue
        for x in adj[w]:
            if x == u or x == v:
                continue
            for y in adj[x]:
                if y == u or y == v or y == w:
                    continue
                for z in adj[y]:
                    if z == u or z == v or z == w or z == x:
                        continue
                    if u in adj[z]:
                        return True
    return False

def _greedy_fill(adj, edge_order, max_edges):
    """Greedy: go through edge_order, add edge if no C6, until max_edges reached."""
    count = 0
    for u, v in edge_order:
        if count >= max_edges:
            break
        if v in adj[u]:
            continue
        if _creates_c6(adj, u, v):
            continue
        adj[u].add(v)
        adj[v].add(u)
        count += 1
    return count

def _local_search_2opt(edges, time_limit, seed=0):
    """
    2-opt local search: try to swap one edge for another (or add/remove)
    to improve edge count while maintaining C6-free property.
    """
    random.seed(seed)
    adj = _build_adj(edges)
    edge_set = set()
    for u, v in edges:
        edge_set.add((u, v))
    current_count = len(edge_set)
    best_count = current_count
    best_edges = list(edge_set)
    
    start = time.time()
    no_improve = 0
    max_no_improve = 500
    
    while time.time() - start < time_limit:
        if no_improve > max_no_improve:
            break
        
        # Try random add
        improved = False
        for _ in range(50):
            u, v = random.choice(ALL_PAIRS)
            if (u, v) in edge_set:
                continue
            if _creates_c6(adj, u, v):
                continue
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add((u, v))
            current_count += 1
            improved = True
            no_improve = 0
            if current_count > best_count:
                best_count = current_count
                best_edges = list(edge_set)
            break
        
        if not improved:
            # Try swap: remove edge e, try to add edge f
            if edge_set:
                u, v = random.choice(list(edge_set))
                adj[u].discard(v)
                adj[v].discard(u)
                edge_set.discard((u, v))
                current_count -= 1
                
                added = False
                for _ in range(50):
                    a, b = random.choice(ALL_PAIRS)
                    if a == b or (a, b) in edge_set:
                        continue
                    if _creates_c6(adj, a, b):
                        continue
                    adj[a].add(b)
                    adj[b].add(a)
                    edge_set.add((a, b))
                    current_count += 1
                    added = True
                    no_improve = 0
                    if current_count > best_count:
                        best_count = best_count
                        best_edges = list(edge_set)
                    break
                
                if not added:
                    # Revert
                    adj[u].add(v)
                    adj[v].add(u)
                    edge_set.add((u, v))
                    current_count += 1
                no_improve += 1
    
    return [[u, v] for u, v in best_edges], best_count

def _structured_seed():
    """
    Create a structured starting graph using a 5-partite construction.
    Split 40 vertices into 5 parts of 8. Connect parts in a C5 pattern.
    This gives 8*8*5 = 320 potential inter-part edges, but we only connect
    adjacent parts in C5: 5 pairs of adjacent parts, each with 8*8=64 edges.
    Total: 5*64 = 320 edges? No wait, that's too many and likely has C6.
    
    Actually, let's use a different structure. 
    Use 4 parts of 10 vertices. Connect all pairs of parts (complete 4-partite).
    That's C(4,2)*10*10 = 6*100 = 600 edges. Way too many, definitely has C6.
    
    Better: start with 8-K5 (100 edges) and add inter-block edges carefully.
    """
    edges = _seed_graph_8K5()
    return edges

def _diverse_seeds():
    """Generate diverse starting configurations."""
    configs = []
    
    # Config 1: 8-K5
    configs.append(_seed_graph_8K5())
    
    # Config 2: 5 parts of 8, connect in C5 pattern (each part connects to 2 neighbors)
    # Parts: 0-7, 8-15, 16-23, 24-31, 32-39
    # Connect part i to part (i+1)%5 and part (i+4)%5
    edges = []
    parts = [[i*8 + j for j in range(8)] for i in range(5)]
    for i in range(5):
        for j in [(i+1)%5, (i+4)%5]:
            if i < j:  # avoid double counting
                for a in parts[i]:
                    for b in parts[j]:
                        edges.append([a, b])
    # This gives 5 * 64 = 320 edges... but likely has C6
    # Let's just use 8-K5 for now and rely on search
    configs.append(edges)
    
    # Config 3: Random dense-ish graph (start with ~100 random edges, filter C6s)
    random.seed(123)
    random_edges = random.sample(ALL_PAIRS, 110)
    adj = [set() for _ in range(N)]
    valid_edges = []
    for u, v in random_edges:
        if _creates_c6(adj, u, v):
            continue
        adj[u].add(v)
        adj[v].add(u)
        valid_edges.append([u, v])
    configs.append(valid_edges)
    
    return configs

def construct():
    n = N
    deadline = 130  # Safety margin before 150s kill
    start_time = time.time()
    
    best_edges = _seed_graph_8K5()
    best_count = len(best_edges)
    
    # Phase 1: Greedy fill from 8-K5 with multiple random orderings
    remaining = deadline - 10 - (time.time() - start_time)
    per_phase = remaining / 3.0
    
    # Phase 1a: Greedy with multiple orderings (first 1/3 of time)
    t_phase1 = time.time() + per_phase
    for seed in [1, 7, 42, 99]:
        if time.time() > t_phase1:
            break
        adj = _build_adj(best_edges)
        # Get non-edges
        edge_set = set((u, v) for u, v in best_edges)
        non_edges = [p for p in ALL_PAIRS if p not in edge_set]
        random.seed(seed)
        random.shuffle(non_edges)
        
        count = len(edge_set)
        for u, v in non_edges:
            if time.time() > t_phase1:
                break
            if _creates_c6(adj, u, v):
                continue
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add((u, v))
            count += 1
        
        if count > best_count:
            best_count = count
            best_edges = [[u, v] for u, v in edge_set]
    
    # Phase 1b: Local search from best (middle 1/3)
    t_phase2 = time.time() + per_phase
    edges, count = _local_search_2opt(best_edges, time.time() + per_phase - time.time(), seed=777)
    if count > best_count:
        best_count = count
        best_edges = edges
    
    # Phase 1c: More local search with different seeds
    t_phase3 = time.time() + per_phase
    for seed in [123, 456, 789]:
        if time.time() > t_phase3:
            break
        chunk = min(3, t_phase3 - time.time())
        if chunk <= 0:
            break
        edges, count = _local_search_2opt(best_edges, chunk, seed=seed)
        if count > best_count:
            best_count = count
            best_edges = edges
    
    # Phase 2: SAT-based search for higher edge counts
    remaining = deadline - 5 - (time.time() - start_time)
    if remaining > 10:
        # Try to prove we can get best_count + 1, +2, ... edges
        for delta in range(1, 6):
            target = best_count + delta
            if time.time() > deadline - 5:
                break
            # Use a smart edge ordering: edges that are "most needed" first
            # Order by: edges between different K5 blocks first (more likely to be addable)
            edge_order = list(ALL_PAIRS)
            # Sort: cross-block edges first
            def block_of(v):
                return v // 5
            def edge_key(e):
                u, v = e
                if block_of(u) == block_of(v):
                    return (1, 0)  # intra-block (already in K5)
                else:
                    return (0, random.random())  # cross-block
            random.seed(42)
            edge_order.sort(key=edge_key)
            
            # Time-limited backtracking
            _deadline_global = time.time() + min(8, deadline - 5 - time.time())
            global _deadline
            _deadline = _deadline_global
            
            result = _sat_backtracking_search(target, edge_order, seed=42)
            if result is not None:
                # Verify
                adj = _build_adj(result)
                if not _has_c6_adj(adj, n):
                    best_edges = result
                    best_count = target
    
    # Final verification
    adj = _build_adj(best_edges)
    if _has_c6_adj(adj, n):
        return _seed_graph_8K5()
    
    return best_edges

# Global deadline for SAT backtracking
_deadline = float('inf')
# EVOLVE-BLOCK-END
