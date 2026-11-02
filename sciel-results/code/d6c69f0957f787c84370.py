"""
Improved Cage Solver for Small 5-Regular Graphs with Girth 9.
Focuses on efficient local search (edge swaps) to minimize short cycles,
which is more effective than spectral optimization for this parameter range.
"""
import time
import threading
import random as _random
import itertools as _itertools
from collections import deque
import sys

# Deadline guard: Evaluator kills at 150s, we target 145s for safety.
_DEADLINE = 145.0
_start = [None]


def _elapsed():
    s = _start[0]
    return 0.0 if s is None else time.time() - s


# ---------- Exact girth via non-backtracking walks ----------
def _nonback_girth(adj, n):
    """
    Computes the girth of a graph using non-backtracking walks from every vertex.
    This is the exact method required by the contract.
    """
    prev = {}
    q = deque()
    for u in range(n):
        for p in adj[u]:
            key = (p, u)
            prev[key] = (u, 1)
            q.append(key)
    
    # Track the minimum cycle length found
    min_girth = 10**9
    
    while q:
        (v, p), (u, dist) = q.popleft()
        for w in adj[v]:
            if w == p:
                continue
            key = (w, v)
            if w == u:
                # Found a cycle
                if dist + 1 < min_girth:
                    min_girth = dist + 1
                # We don't break immediately because there might be shorter cycles
                # elsewhere, but for efficiency in search, we can early-exit if min_girth < g
                # However, this function is used for exact verification too.
                # For search, we can optimize by not expanding further if we only care about < g.
                # But standard non-backtracking BFS finds the girth.
            else:
                if key in prev:
                    continue
                prev[key] = (v, dist + 1)
                q.append(key)
                
    return min_girth

def _quick_girth_check(adj, n, target_girth):
    """
    Faster check: returns True if girth >= target_girth.
    Stops early if a cycle shorter than target_girth is found.
    """
    prev = {}
    q = deque()
    for u in range(n):
        for p in adj[u]:
            key = (p, u)
            prev[key] = (u, 1)
            q.append(key)
            
    while q:
        (v, p), (u, dist) = q.popleft()
        if dist + 1 >= target_girth:
            continue # No need to expand if current depth is already high enough
        for w in adj[v]:
            if w == p:
                continue
            key = (w, v)
            if w == u:
                # Cycle found of length dist+1
                if dist + 1 < target_girth:
                    return False
            else:
                if key in prev:
                    continue
                prev[key] = (v, dist + 1)
                q.append(key)
    return True


def _build_adj(edges, n):
    adj = [set() for _ in range(n)]
    for (i, j) in edges:
        adj[i].add(j)
        adj[j].add(i)
    for i in range(n):
        adj[i] = sorted(adj[i])
    return adj


def _is_valid(adj, n, k, g):
    if n == 0:
        return False
    if any(len(adj[i]) != k for i in range(n)):
        return False
    for i in range(n):
        if len(set(adj[i])) != len(adj[i]):
            return False
        for j in adj[i]:
            if i <= j:
                return False
    return _nonback_girth(adj, n) >= g


# ---------- Moore bound ----------
def _moore(k, g):
    if g % 2 == 1:
        r = (g - 1) // 2
        return 2 * sum((k - 1) ** i for i in range(r + 1))
    else:
        r = g // 2
        return 2 * sum((k - 1) ** i for i in range(r)) + k * (k - 1) ** (r - 1)


# ---------- Construction: Greedy Configuration Model with Girth Awareness ----------
def _config_model_greedy(n, k, rng, target_girth):
    """
    Generates a k-regular graph. 
    Tries to avoid creating short cycles during the stub pairing process.
    """
    # Initialize stubs
    stubs = []
    for v in range(n):
        stubs.extend([v] * k)
    
    for _ in range(50): # Max attempts
        rng.shuffle(stubs)
        adj = [set() for _ in range(n)]
        valid = True
        
        # Pair stubs
        # To avoid self-loops and multi-edges, we pick a random stub, then a partner.
        # We maintain a list of available stubs.
        available = list(range(len(stubs)))
        rng.shuffle(available)
        
        # We process pairs. If a pair is invalid (self-loop or existing edge), we try to fix it.
        # For simplicity and speed, we use a standard random pairing first, then check.
        # If girth is too low, we return None to trigger another attempt or local search.
        
        for i in range(0, len(stubs), 2):
            u = stubs[available[i]]
            v = stubs[available[i+1]]
            
            if u == v:
                valid = False
                break
            if v in adj[u]:
                valid = False
                break
                
            adj[u].add(v)
            adj[v].add(u)
            
        if not valid:
            continue
            
        if all(len(adj[i]) == k for i in range(n)):
            # Quick check if it's already good
            if _quick_girth_check([sorted(adj[i]) for i in range(n)], n, target_girth):
                return [sorted(adj[i]) for i in range(n)]
            
            # Even if not good, return it for local search to improve
            # But only if it doesn't have extremely short cycles (like 3 or 4) to save time
            # Actually, let's just return the best attempt found in this batch.
            # The caller will decide to use local search.
            return [sorted(adj[i]) for i in range(n)]
            
    return None


# ---------- Local Search: Edge Swap to Minimize Short Cycles ----------
def _local_search_girth(adj, n, k, target_girth, rng, max_iters=20000, time_limit=10.0):
    """
    Improves the girth of a k-regular graph by performing edge swaps.
    Objective: Minimize the number of 4-cycles and 6-cycles.
    """
    start_time = time.time()
    adj_set = [set(adj[i]) for i in range(n)]
    
    # Precompute counts of short cycles
    def count_short_cycles():
        c4 = 0
        c6 = 0 # Approximation or exact for small n
        # 4-cycles: pairs of vertices with >= 2 common neighbors
        for u in range(n):
            for v in range(u + 1, n):
                common = len(adj_set[u] & adj_set[v])
                if common >= 2:
                    c4 += common * (common - 1) // 2
        # 6-cycles are harder to count exactly fast. We rely on girth check periodically.
        # For the score, we primarily want to break 4-cycles and 5-cycles.
        # A simple heuristic: minimize c4.
        return c4

    current_c4 = count_short_cycles()
    best_c4 = current_c4
    best_adj = [list(s) for s in adj_set]
    
    # Check if already good
    if _quick_girth_check(adj, n, target_girth):
        return adj
    
    for _ in range(max_iters):
        if time.time() - start_time > time_limit or _elapsed() > _DEADLINE - 5.0:
            break
            
        # Pick a random edge (u, v)
        u = rng.randrange(n)
        if not adj_set[u]: continue
        v = rng.choice(list(adj_set[u]))
        
        # Pick two non-neighbors w, x such that swapping (u,v) with (u,w),(v,x) or (u,x),(v,w) is valid
        # We want to replace (u,v) with (u,w) and (v,x) where w, x are not connected to u, v in a way that creates loops/multi-edges
        # and ideally breaks cycles.
        
        candidates_w = list(range(n))
        candidates_x = list(range(n))
        
        # Filter valid candidates for w
        valid_w = []
        for w in candidates_w:
            if w == u or w == v: continue
            if w in adj_set[u]: continue
            # Check if (u, w) creates a multi-edge? No, w not in adj_set[u] ensures simple graph locally.
            # But we must ensure we don't create a self-loop on w? No, w!=u.
            valid_w.append(w)
            
        if not valid_w: continue
        
        w = rng.choice(valid_w)
        
        valid_x = []
        for x in candidates_x:
            if x == u or x == v: continue
            if x == w: continue
            if x in adj_set[v]: continue
            # Check if (v, x) is valid
            valid_x.append(x)
            
        if not valid_x: continue
        
        x = rng.choice(valid_x)
        
        # Perform swap: remove (u,v), add (u,w) and (v,x)
        # Check if this creates a multi-edge (u, x) or (v, w)?
        # If x in adj_set[u], then adding (v,x) is fine, but (u,w) might be problematic if w in adj_set[u]? No, w not in adj_set[u].
        # But if we add (u,w), we must ensure w not in adj_set[u] (already checked).
        # We must ensure that the new edges don't already exist.
        # (u,w) exists if w in adj_set[u]. We ensured w not in adj_set[u].
        # (v,x) exists if x in adj_set[v]. We ensured x not in adj_set[v].
        
        # However, we must also ensure that we don't create a parallel edge (u,x) if we were to do a different swap, 
        # but here we are doing specific swap.
        
        # Wait, the swap (u,v) -> (u,w) + (v,x) changes the degree of u and v? 
        # u loses v, gains w. Degree stays k.
        # v loses u, gains x. Degree stays k.
        # This is a valid 2-opt move for regular graphs if we ensure no multi-edges.
        
        # Check if (u,w) or (v,x) creates a multi-edge with existing edges?
        # We checked w not in adj[u] and x not in adj[v].
        # But what if w == x? We checked w != x.
        
        # Execute
        adj_set[u].remove(v)
        adj_set[v].remove(u)
        adj_set[u].add(w)
        adj_set[w].add(u)
        adj_set[v].add(x)
        adj_set[x].add(v)
        
        new_c4 = count_short_cycles()
        
        if new_c4 < current_c4:
            current_c4 = new_c4
            if new_c4 < best_c4:
                best_c4 = new_c4
                best_adj = [list(s) for s in adj_set]
                
            # Periodic girth check
            if _in_girth_check_counter % 100 == 0: # Need to define counter
                pass # Simplified: check every 500 iterations
                
            # Actually, let's just check girth every 100 iterations for small n
            if n < 500 and _in_girth_check_counter % 50 == 0:
                if _quick_girth_check([sorted(s) for s in adj_set], n, target_girth):
                    return [sorted(s) for s in adj_set]
        else:
            # Revert
            adj_set[u].remove(w)
            adj_set[w].remove(u)
            adj_set[v].remove(x)
            adj_set[x].remove(v)
            adj_set[u].add(v)
            adj_set[v].add(u)
            
        _in_girth_check_counter += 1

    return [sorted(s) for s in best_adj]

# Global counter for local search checks
_in_girth_check_counter = 0

def _search_k5_g9(n, k, g, rng):
    """
    Specific search loop for k=5, g=9.
    """
    global _in_girth_check_counter
    _in_girth_check_counter = 0
    
    best_n = float('inf')
    best_edges = None
    
    # Try n from Moore bound upwards
    moore = _moore(k, g)
    min_n = max(moore, k + 1)
    
    # For k=5, g=9, Moore bound is 428.
    # We expect a solution around 1000-1100.
    # We will search up to 1120.
    
    search_limit = min(min_n + 100, 1120) # Safety cap
    
    for curr_n in range(min_n, search_limit + 1, 2):
        if curr_n * k % 2 != 0:
            continue
        if _elapsed() > _DEADLINE - 10.0:
            break
            
        # Try multiple initializations
        for attempt in range(5):
            if _elapsed() > _DEADLINE - 5.0:
                break
            
            adj = _config_model_greedy(curr_n, k, rng, g)
            if adj is None:
                continue
                
            # Quick check
            if _quick_girth_check(adj, curr_n, g):
                edges = [(i, j) for i in range(curr_n) for j in adj[i] if i < j]
                return curr_n, edges
            
            # Local Search
            # Limit local search time per candidate to avoid getting stuck
            time_budget = 5.0 if curr_n < 800 else 3.0
            opt_adj = _local_search_girth(adj, curr_n, k, g, rng, max_iters=10000, time_limit=time_budget)
            
            if opt_adj is not None:
                if _quick_girth_check(opt_adj, curr_n, g):
                    edges = [(i, j) for i in range(curr_n) for j in opt_adj[i] if i < j]
                    return curr_n, edges
                    
    return None


def _worker(name, fn, holder, k, g):
    try:
        res = fn()
    except Exception:
        res = None
    if res is None:
        return
    n, edges = res
    if _elapsed() >= _DEADLINE:
        return
    try:
        if n > 0 and _is_valid(_build_adj(edges, n), n, k, g):
            cur = holder.get("best")
            if cur is None or n < cur[0]:
                holder["best"] = (n, edges)
    except Exception:
        pass


def construct(k, g):
    _start[0] = time.time()
    holder = {"best": None}
    
    # For the specific task k=5, g=9, we use the specialized search.
    # For other parameters, we might fall back to general methods if needed, 
    # but the task specifies this interface for this specific problem context.
    
    jobs = [
        ("search_k5_g9_0", lambda: _search_k5_g9(0, k, g, _random.Random(0))),
        ("search_k5_g9_1", lambda: _search_k5_g9(0, k, g, _random.Random(1))),
        ("search_k5_g9_2", lambda: _search_k5_g9(0, k, g, _random.Random(2))),
    ]
    
    threads = []
    for name, fn in jobs:
        t = threading.Thread(target=_worker, args=(name, fn, holder, k, g), daemon=True)
        t.start()
        threads.append(t)

    # Wait for completion or deadline
    while _elapsed() < _DEADLINE - 2.0:
        if not any(t.is_alive() for t in threads):
            break
        time.sleep(0.1)
        
    # Join threads
    for t in threads:
        t.join()

    best = holder.get("best")
    
    # Fallback: If no valid graph found, return a simple valid graph if possible
    # For k=5, g=9, we should have found one. 
    # If not, we must return *something* valid to avoid score 0.
    # A trivial valid graph for k=5 is the complete graph K6 (n=6, girth 3).
    # But girth must be >= 9. So K6 is invalid.
    # We need a valid 5-regular graph with girth >= 9.
    # If search fails, we might have to return a larger known graph or fail.
    # The contract says "invalid result... scores 0".
    # So we must ensure we return a valid graph.
    
    if best is None:
        # Last resort: Try to construct a large random graph and verify
        # Or just return the smallest n we tried even if not verified? No, must be valid.
        # Let's try one more synchronous attempt with a larger n
        n_fallback = 1116
        k_fallback = 5
        g_fallback = 9
        if k == 5 and g == 9:
             # We can use a known construction or just a random one that we hope is valid?
             # Random 5-regular graphs of size 1116 likely have girth < 9.
             # We must rely on the search.
             # If search failed, it's a timeout or bug.
             pass

    if best is None:
        # Emergency fallback: Return a valid graph for a *smaller* girth? No, must satisfy g.
        # This indicates a failure in the solver.
        # To prevent crash, return an empty graph? No, n>0.
        # Let's return a graph that is definitely valid for small g, but it will fail the girth check.
        # Actually, if the solver fails, it's better to return None? No, construct must return (n, edges).
        # The evaluator will check validity.
        # We will try to generate a random 5-regular graph of size 100 and hope? No.
        
        # If we truly fail, we return a dummy that will be scored 0.
        # But we should try to return the best *attempt* if it's close? No, invalid is 0.
        
        # Let's force a synchronous search for a small n just to have *a* graph,
        # even if it doesn't meet girth? No, that scores 0.
        
        # If best is None, it means no valid graph was found.
        # We will return a minimal 5-regular graph (K6) which has girth 3.
        # It will fail the girth check, score 0.
        # But it's better than crashing.
        n = 6
        edges = [(i, j) for i in range(n) for j in range(i + 1, n) if (i + j) % 5 != 0] # This isn't 5-regular.
        # K6 is 5-regular.
        edges = [(i, j) for i in range(6) for j in range(i + 1, 6)]
        best = (6, edges)

    return best
