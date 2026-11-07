"""
Dense odd-order graphs avoiding equal-degree path endpoints.
Strategy: Greedy construction with careful degree management + local search polish.
The key insight is that in a valid graph, for every edge (u,v), no degree can appear
in both N(u)\{v} and N(v)\{u} (unless the same vertex is the only one with that degree on both sides).

This means: if u and v have the same set of neighbor-degrees (excluding each other),
we need to be careful. The constraint is really about degree distributions around edges.

Approach:
1. Start with a bipartite graph K_{7,8} which is valid.
2. Use a smart greedy that adds edges while maintaining validity, prioritizing
   edges that are "safe" (don't create conflicts).
3. Polish with local search (add, swap moves).
"""
import time
import random
from collections import defaultdict

N = 15
DEADLINE = 80.0  # Safety margin before 90s kill


class _GraphState:
    """Maintains incremental graph state for fast validity checks."""

    def __init__(self, adj=None):
        self.adj = [[0] * N for _ in range(N)]
        self.deg = [0] * N
        self.neighbors = [set() for _ in range(N)]
        if adj is not None:
            self._load(adj)

    def _load(self, adj):
        for u in range(N):
            for v in range(u + 1, N):
                if adj[u][v]:
                    self.add_edge(u, v)

    def add_edge(self, u, v):
        if u == v or self.adj[u][v]:
            return False
        self.adj[u][v] = 1
        self.adj[v][u] = 1
        self.deg[u] += 1
        self.deg[v] += 1
        self.neighbors[u].add(v)
        self.neighbors[v].add(u)
        return True

    def remove_edge(self, u, v):
        if u == v or not self.adj[u][v]:
            return False
        self.adj[u][v] = 0
        self.adj[v][u] = 0
        self.deg[u] -= 1
        self.deg[v] -= 1
        self.neighbors[u].discard(v)
        self.neighbors[v].discard(u)
        return True

    def edge_count(self):
        return sum(self.deg) // 2

    def copy(self):
        new_state = _GraphState()
        for u in range(N):
            new_state.adj[u] = self.adj[u][:]
            new_state.deg[u] = self.deg[u]
            new_state.neighbors[u] = set(self.neighbors[u])
        return new_state


def _check_edge_valid(state, u, v):
    """
    Check if adding edge (u,v) to the current graph would maintain validity.
    We only need to check paths that use the new edge (u,v).
    A path s-u-v-t is invalid if deg(s) == deg(t).
    Here s is in N(u)\{v} and t is in N(v)\{u}.
    """
    deg = state.deg
    neighbors_u = state.neighbors[u]
    neighbors_v = state.neighbors[v]

    # Build degree counts for neighbors of u (excluding v)
    deg_count_u = defaultdict(int)
    for s in neighbors_u:
        if s != v:
            deg_count_u[deg[s]] += 1

    # Check each t in neighbors of v (excluding u)
    for t in neighbors_v:
        if t == u:
            continue
        d = deg[t]
        if deg_count_u.get(d, 0) > 0:
            # There exists s in N(u)\{v} with deg(s) == deg(t)
            # But we need s != t. If s == t, that's a 2-cycle, not a simple path.
            # Actually s is in N(u) and t is in N(v). If s == t, then s is adjacent to both u and v,
            # which means we have triangle u-s-v-u. The path s-u-v-s is not simple (s==t endpoint).
            # So we need s != t.
            # Check if there's an s != t with deg(s) == d
            if deg_count_u[d] > 1:
                return False
            else:
                # deg_count_u[d] == 1, the only s with that degree is some specific vertex
                # Check if that vertex is t
                for s in neighbors_u:
                    if s != v and deg[s] == d:
                        if s != t:
                            return False
                        break
    return True


def _is_valid_state(state):
    """
    Full validity check: no simple 3-edge path s-u-v-t has deg(s) == deg(t).
    """
    deg = state.deg
    neighbors = state.neighbors

    for u in range(N):
        for v in neighbors[u]:
            if u >= v:
                continue

            # Build mapping: degree -> set of vertices in N(u)\{v}
            deg_map_u = defaultdict(set)
            for s in neighbors[u]:
                if s != v:
                    d = deg[s]
                    deg_map_u[d].add(s)

            # Check each t in N(v)\{u}
            for t in neighbors[v]:
                if t == u:
                    continue
                d = deg[t]
                if d in deg_map_u:
                    # Check if there exists s in deg_map_u[d] with s != t
                    if len(deg_map_u[d]) > 1 or (t not in deg_map_u[d]):
                        return False
    return True


def _try_add_state(state, u, v):
    if u == v or state.adj[u][v]:
        return False
    state.add_edge(u, v)
    if _is_valid_state(state):
        return True
    state.remove_edge(u, v)
    return False


def _try_swap_state(state, u1, v1, u2, v2):
    if state.adj[u1][v1] == 0 or state.adj[u2][v2] == 1:
        return False
    state.remove_edge(u1, v1)
    state.add_edge(u2, v2)
    if _is_valid_state(state):
        return True
    state.remove_edge(u2, v2)
    state.add_edge(u1, v1)
    return False


def _try_2swap_state(state, u1, v1, u2, v2, u3, v3, u4, v4):
    """Remove edges (u1,v1) and (u2,v2), add edges (u3,v3) and (u4,v4)."""
    if (state.adj[u1][v1] == 0 or state.adj[u2][v2] == 0 or
            state.adj[u3][v3] == 1 or state.adj[u4][v4] == 1):
        return False
    # Check no duplicate edges in the swap
    if (u1 == u3 and v1 == v3) or (u1 == u4 and v1 == v4) or \
            (u2 == u3 and v2 == v3) or (u2 == u4 and v2 == v4):
        return False

    state.remove_edge(u1, v1)
    state.remove_edge(u2, v2)
    state.add_edge(u3, v3)
    state.add_edge(u4, v4)
    if _is_valid_state(state):
        return True
    state.remove_edge(u4, v4)
    state.remove_edge(u3, v3)
    state.add_edge(u2, v2)
    state.add_edge(u1, v1)
    return False


def _local_search(state, start_time, seed, max_rounds=50):
    """Hill climbing with add, swap, and 2-swap moves."""
    random.seed(seed)
    current_count = state.edge_count()
    best_state = state.copy()
    best_count = current_count
    no_improve_rounds = 0

    while time.time() - start_time < DEADLINE and no_improve_rounds < max_rounds:
        improved = False

        # Try adding missing edges
        missing = [(u, v) for u in range(N) for v in range(u + 1, N) if not state.adj[u][v]]
        random.shuffle(missing)
        for u, v in missing:
            if time.time() - start_time > DEADLINE:
                break
            if _try_add_state(state, u, v):
                current_count += 1
                improved = True
                if current_count > best_count:
                    best_count = current_count
                    best_state = state.copy()

        if improved:
            no_improve_rounds = 0
            continue

        no_improve_rounds += 1

        # Try single swaps
        existing = [(u, v) for u in range(N) for v in range(u + 1, N) if state.adj[u][v]]
        missing = [(u, v) for u in range(N) for v in range(u + 1, N) if not state.adj[u][v]]
        random.shuffle(existing)
        random.shuffle(missing)

        found_swap = False
        for u1, v1 in existing:
            if time.time() - start_time > DEADLINE:
                break
            for u2, v2 in missing:
                if _try_swap_state(state, u1, v1, u2, v2):
                    current_count += 0  # swap keeps count same
                    found_swap = True
                    break
            if found_swap:
                break

        if found_swap:
            no_improve_rounds = 0
            continue

        # Try 2-swaps
        if no_improve_rounds > 2:
            existing = [(u, v) for u in range(N) for v in range(u + 1, N) if state.adj[u][v]]
            missing = [(u, v) for u in range(N) for v in range(u + 1, N) if not state.adj[u][v]]
            random.shuffle(existing)
            random.shuffle(missing)

            found_2swap = False
            max_existing = min(len(existing), 25)
            max_missing = min(len(missing), 25)
            for i in range(max_existing):
                if time.time() - start_time > DEADLINE:
                    break
                u1, v1 = existing[i]
                for j in range(i + 1, max_existing):
                    if time.time() - start_time > DEADLINE:
                        break
                    u2, v2 = existing[j]
                    for k in range(max_missing):
                        if time.time() - start_time > DEADLINE:
                            break
                        u3, v3 = missing[k]
                        for l in range(k + 1, max_missing):
                            if time.time() - start_time > DEADLINE:
                                break
                            u4, v4 = missing[l]
                            if _try_2swap_state(state, u1, v1, u2, v2, u3, v3, u4, v4):
                                found_2swap = True
                                break
                        if found_2swap:
                            break
                    if found_2swap:
                        break
                if found_2swap:
                    break

            if found_2swap:
                no_improve_rounds = 0
                continue

    return best_state, best_count


def _bipartite_start(a_size):
    """Create K_{a_size, N-a_size}."""
    adj = [[0] * N for _ in range(N)]
    for u in range(a_size):
        for v in range(a_size, N):
            adj[u][v] = 1
            adj[v][u] = 1
    return adj


def _smart_greedy(adj, start_time):
    """
    Smart greedy: try to add edges in order of 'safety'.
    An edge is safer if it connects vertices with different degree profiles.
    """
    state = _GraphState(adj)
    count = state.edge_count()

    # Sort missing edges by a heuristic: prefer edges that are more likely to be valid
    missing = [(u, v) for u in range(N) for v in range(u + 1, N) if not state.adj[u][v]]

    # Heuristic: edges between vertices with different current degrees are safer
    def edge_score(u, v):
        # Prefer edges where deg(u) != deg(v) (less likely to create equal-degree endpoints)
        score = abs(state.deg[u] - state.deg[v])
        # Also prefer edges that connect to vertices with fewer neighbors (less chance of conflict)
        score -= (len(state.neighbors[u]) + len(state.neighbors[v])) * 0.1
        return score

    missing.sort(key=lambda e: -edge_score(e[0], e[1]))

    for u, v in missing:
        if time.time() - start_time > DEADLINE:
            break
        if _try_add_state(state, u, v):
            count += 1

    return state, count


def construct():
    start_time = time.time()
    best_state = _GraphState()
    best_count = 0

    # Try different bipartite splits as starting points
    for a_size in [7, 8, 6, 9, 5, 10, 4, 11]:
        if time.time() - start_time > DEADLINE - 5:
            break
        adj = _bipartite_start(a_size)
        state = _GraphState(adj)
        count = state.edge_count()

        # Smart greedy to add more edges
        state_g, count_g = _smart_greedy(adj, start_time)
        if count_g > best_count:
            best_count = count_g
            best_state = state_g

        # Local search from this starting point
        state_ls, count_ls = _local_search(state_g, start_time, seed=a_size * 100)
        if count_ls > best_count:
            best_count = count_ls
            best_state = state_ls

    # Try a few random greedy starts
    for i in range(8):
        if time.time() - start_time > DEADLINE - 5:
            break
        random.seed(10000 + i)
        adj = [[0] * N for _ in range(N)]
        edges = [(u, v) for u in range(N) for v in range(u + 1, N)]
        random.shuffle(edges)
        state = _GraphState()
        for u, v in edges:
            if time.time() - start_time > DEADLINE - 3:
                break
            _try_add_state(state, u, v)
        count = state.edge_count()
        if count > best_count - 3:  # Only polish if reasonably close
            # Do local search from this point
            state_ls, count_ls = _local_search(state, start_time, seed=20000 + i)
            if count_ls > best_count:
                best_count = count_ls
                best_state = state_ls

    # Final validation
    if not _is_valid_state(best_state):
        # Fallback to K_{7,8}
        adj = _bipartite_start(7)
        best_state = _GraphState(adj)
        best_count = best_state.edge_count()

    result = []
    for u in range(N):
        for v in range(u + 1, N):
            if best_state.adj[u][v]:
                result.append([int(u), int(v)])

    return result
