"""Optimized program: Fast bitmask-based validity check and greedy/local search."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

N = 15
DEADLINE = 75.0  # Safety margin before 90s kill


class _GraphState:
    """Maintains graph state with fast incremental validity checking."""
    
    def __init__(self):
        self.adj = [0] * N  # adj[u] is a bitmask of neighbors of u
        self.deg = [0] * N
        self.edge_count = 0
        # For fast checking: for each vertex, list of neighbors
        self.neigh = [[] for _ in range(N)]
        # For each degree value, set of vertices with that degree
        self.deg_groups = defaultdict(set)
        for i in range(N):
            self.deg_groups[0].add(i)
        self._valid = True  # Assume valid initially
    
    def _update_deg_group(self, u, old_deg, new_deg):
        self.deg_groups[old_deg].discard(u)
        self.deg_groups[new_deg].add(u)
    
    def _add_edge_internal(self, u, v):
        """Add edge (u,v) without validity check. Updates state."""
        if self.adj[u] & (1 << v):
            return
        self.adj[u] |= (1 << v)
        self.adj[v] |= (1 << u)
        self.neigh[u].append(v)
        self.neigh[v].append(u)
        old_du = self.deg[u]
        old_dv = self.deg[v]
        self.deg[u] = old_du + 1
        self.deg[v] = old_dv + 1
        self._update_deg_group(u, old_du, self.deg[u])
        self._update_deg_group(v, old_dv, self.deg[v])
        self.edge_count += 1
    
    def _remove_edge_internal(self, u, v):
        """Remove edge (u,v) without validity check. Updates state."""
        if not (self.adj[u] & (1 << v)):
            return
        self.adj[u] &= ~(1 << v)
        self.adj[v] &= ~(1 << u)
        self.neigh[u].remove(v)
        self.neigh[v].remove(u)
        old_du = self.deg[u]
        old_dv = self.deg[v]
        self.deg[u] = old_du - 1
        self.deg[v] = old_dv - 1
        self._update_deg_group(u, old_du, self.deg[u])
        self._update_deg_group(v, old_dv, self.deg[v])
        self.edge_count -= 1
    
    def is_valid(self):
        """Full validity check. Returns True if no bad 3-edge path exists."""
        for u in range(N):
            for v in self.neigh[u]:
                if v <= u:
                    continue
                # For edge (u,v), check all s in N(u)\{v}, t in N(v)\{u}, s!=t
                # deg(s) != deg(t) must hold
                for s in self.neigh[u]:
                    if s == v:
                        continue
                    ds = self.deg[s]
                    for t in self.neigh[v]:
                        if t == u or t == s:
                            continue
                        if self.deg[t] == ds:
                            return False
        return True
    
    def check_edge_addition(self, u, v):
        """
        Check if adding edge (u,v) would keep graph valid.
        Only needs to check paths that involve the new edge (u,v).
        New paths: s-u-v-t where s in N(u), t in N(v), s!=v, t!=u, s!=t.
        Also s-v-u-t is the same path in reverse.
        """
        if u == v or (self.adj[u] & (1 << v)):
            return False
        # After adding (u,v), deg[u] increases by 1, deg[v] increases by 1
        # But for the path check, we use current degrees of s and t
        # Wait: the degrees in the constraint are the FULL graph degrees AFTER addition.
        # So deg'(s) = deg(s) for s not in {u,v}, and deg'(u) = deg(u)+1, deg'(v) = deg(v)+1.
        # For path s-u-v-t: deg'(s) and deg'(t) must differ.
        # s is in N(u) (current), t is in N(v) (current).
        # Note: s could be u? No, s is neighbor of u, so s != u. Similarly t != v.
        # But s could equal v? No, we exclude s==v.
        # t could equal u? No, we exclude t==u.
        # s and t could be u or v? s is neighbor of u so s != u. t is neighbor of v so t != v.
        # But s could be v? We exclude that. t could be u? We exclude that.
        # Can s == u? No. Can t == v? No.
        # So s and t are neither u nor v (since s in N(u) means s!=u, and we exclude s==v; 
        #    t in N(v) means t!=v, and we exclude t==u).
        # Therefore deg'(s) = deg(s) and deg'(t) = deg(t).
        
        for s in self.neigh[u]:
            if s == v:
                continue
            ds = self.deg[s]
            for t in self.neigh[v]:
                if t == u or t == s:
                    continue
                if self.deg[t] == ds:
                    return False
        return True
    
    def check_edge_removal(self, u, v):
        """
        Check if removing edge (u,v) would keep graph valid.
        Removing an edge can only create new bad paths if... actually no.
        Removing an edge reduces degrees and removes paths.
        The only way removal could create a violation is if it changes degrees such that
        some previously-valid path becomes invalid. But paths that existed before still exist
        (unless they used edge (u,v)). Paths that used (u,v) are gone.
        New paths? No new paths are created by removing an edge.
        However, degree changes: deg'(u) = deg(u)-1, deg'(v) = deg(v)-1.
        For any remaining path s-u'-v'-t (not using edge (u,v)), if s or t is u or v,
        their degree changed. So we need to re-check paths where s or t is u or v.
        
        Actually, let me think more carefully. After removing (u,v):
        - Any path that existed before and doesn't use (u,v) still exists.
        - For such a path s-a-b-t, if s != u,v and t != u,v, degrees unchanged, still valid.
        - If s == u or s == v or t == u or t == v, degrees changed.
        
        This is complex. For simplicity, let's just do a full check after removal.
        """
        pass
    
    def try_add(self, u, v):
        """Try to add edge (u,v). Returns True if successful."""
        if not self.check_edge_addition(u, v):
            return False
        self._add_edge_internal(u, v)
        return True
    
    def try_remove(self, u, v):
        """Try to remove edge (u,v). Returns True if successful."""
        if not (self.adj[u] & (1 << v)):
            return False
        self._remove_edge_internal(u, v)
        # After removal, need to verify validity
        if not self.is_valid():
            # Revert
            self._add_edge_internal(u, v)
            return False
        return True
    
    def try_swap(self, u, v, x, y):
        """
        Try to swap: remove (u,v) and add (x,y).
        Returns True if successful.
        """
        if not (self.adj[u] & (1 << v)):
            return False
        if x == y or (self.adj[x] & (1 << y)):
            return False
        
        # First remove (u,v)
        self._remove_edge_internal(u, v)
        
        # Check if adding (x,y) is valid
        if not self.check_edge_addition(x, y):
            # Revert
            self._add_edge_internal(u, v)
            return False
        
        self._add_edge_internal(x, y)
        
        # Full validity check
        if not self.is_valid():
            # Revert
            self._remove_edge_internal(x, y)
            self._add_edge_internal(u, v)
            return False
        
        return True
    
    def edges_list(self):
        """Return list of edges as [u,v] pairs."""
        result = []
        for u in range(N):
            for v in self.neigh[u]:
                if u < v:
                    result.append([u, v])
        return result


def _bipartite_start(a, b):
    """Create a bipartite graph K_{a,b} with vertices 0..a-1 in part A, a..a+b-1 in part B."""
    state = _GraphState()
    for i in range(a):
        for j in range(a, a + b):
            state._add_edge_internal(i, j)
    # Verify validity
    if state.is_valid():
        return state
    return None


def _smart_greedy(state, deadline):
    """
    Smart greedy: try to add edges in an order that maximizes density.
    """
    start_time = time.time()
    # Try all possible edges in random order
    all_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
    random.shuffle(all_edges)
    
    for u, v in all_edges:
        if time.time() - start_time > deadline:
            break
        state.try_add(u, v)
    
    return state


def _local_search(state, deadline):
    """
    Local search: try swaps and additions to improve edge count.
    """
    start_time = time.time()
    best_count = state.edge_count
    no_improve = 0
    
    while time.time() - start_time < deadline:
        improved = False
        
        # Try adding any missing edge
        for u in range(N):
            for v in range(u + 1, N):
                if time.time() - start_time > deadline:
                    break
                if not (state.adj[u] & (1 << v)):
                    if state.try_add(u, v):
                        improved = True
                        no_improve = 0
                        break
            if improved:
                break
        
        if improved:
            continue
        
        # Try swaps: remove one edge, add another
        current_edges = state.edges_list()
        random.shuffle(current_edges)
        
        for u, v in current_edges:
            if time.time() - start_time > deadline:
                break
            
            # Try to find an edge to add
            for x in range(N):
                for y in range(x + 1, N):
                    if time.time() - start_time > deadline:
                        break
                    if x == u or x == v or y == u or y == v:
                        continue
                    if not (state.adj[x] & (1 << y)):
                        if state.try_swap(u, v, x, y):
                            improved = True
                            no_improve = 0
                            break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            no_improve += 1
            if no_improve > 5:
                break
    
    return state


def construct():
    """
    Return a list of edges [u, v] forming a valid graph on vertices 0..14.
    """
    start_time = time.time()
    random.seed(42)
    
    best_state = None
    best_count = 0
    
    # Try different bipartite starts
    for a in range(1, N):
        b = N - a
        if a == b:
            continue  # K_{a,a} might be invalid
        if time.time() - start_time > DEADLINE:
            break
        
        state = _bipartite_start(a, b)
        if state is None:
            continue
        
        # Apply smart greedy
        state = _smart_greedy(state, DEADLINE - (time.time() - start_time))
        
        # Apply local search with remaining time
        remaining = DEADLINE - (time.time() - start_time)
        if remaining > 2:
            state = _local_search(state, remaining)
        
        if state.edge_count > best_count:
            best_count = state.edge_count
            best_state = state
    
    # If no bipartite start worked, try empty graph + greedy
    if best_state is None or best_count == 0:
        state = _GraphState()
        state = _smart_greedy(state, DEADLINE - (time.time() - start_time))
        if state.edge_count > best_count:
            best_count = state.edge_count
            best_state = state
    
    # Final local search on best state
    if best_state is not None:
        remaining = DEADLINE - (time.time() - start_time)
        if remaining > 1:
            best_state = _local_search(best_state, remaining)
    
    if best_state is None:
        # Fallback: star graph
        return [[0, v] for v in range(1, N)]
    
    # Final validation
    if not best_state.is_valid():
        # Fallback to star
        return [[0, v] for v in range(1, N)]
    
    # Ensure proper types
    result = []
    for e in best_state.edges_list():
        result.append([int(e[0]), int(e[1])])
    
    return result

# EVOLVE-BLOCK-END
