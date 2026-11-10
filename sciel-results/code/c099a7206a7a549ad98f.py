import time
import itertools
import random
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    n = 16
    total_possible_edges = n * (n - 1) // 2  # 120

    # Q3 structure: 3-cube vertices 0..7, edges where hamming distance is 1
    Q3_EDGES = [
        (0, 1), (0, 2), (0, 4),
        (1, 3), (1, 5),
        (2, 3), (2, 6),
        (3, 7),
        (4, 5), (4, 6),
        (5, 7),
        (6, 7)
    ]

    # Precompute Q3 adjacency for each vertex
    q3_adj = [set() for _ in range(8)]
    for u, v in Q3_EDGES:
        q3_adj[u].add(v)
        q3_adj[v].add(u)
    q3_neighbors = [list(q3_adj[i]) for i in range(8)]

    # Try to compile a fast C++ Q3 checker
    fast_checker = None
    try:
        import subprocess
        import tempfile
        import os
        c_code = r"""
#include <cstdio>
#include <cstring>

int n = 16;
int adj[16]; // bitmask

// Q3 edges
int q3e[12][2] = {
    {0,1},{0,2},{0,4},{1,3},{1,5},{2,3},
    {3,7},{4,5},{4,6},{5,7},{6,7},{0,0} // dummy
};
int q3_adj[8];
int q3_adj_mask[8];

void init_q3() {
    memset(q3_adj, 0, sizeof(q3_adj));
    memset(q3_adj_mask, 0, sizeof(q3_adj_mask));
    for (int i = 0; i < 11; i++) {
        int u = q3e[i][0], v = q3e[i][1];
        q3_adj_mask[u] |= (1 << v);
        q3_adj_mask[v] |= (1 << u);
    }
}

// Check if S (8 vertices) contains a Q3
// S is array of 8 vertex IDs
// Returns 1 if Q3 found, 0 otherwise
int check_subset(int S[8]) {
    // Fix mapping[0] = S[0] (Q3 is vertex-transitive)
    int mapping[8];
    mapping[0] = S[0];

    // DFS
    int stack[8][8];
    int sp = 0;
    int used_mask = (1 << 0); // S[0] used

    // We'll use iterative DFS
    // State: which q3 vertex we're assigning (idx), and which S vertex to try
    int idx_stack[8];
    int try_stack[8];
    idx_stack[0] = 1;
    try_stack[0] = 1;
    sp = 1;

    while (sp > 0) {
        int idx = idx_stack[sp - 1];
        int try_idx = try_stack[sp - 1];

        if (idx == 8) {
            return 1; // Found
        }

        int found = 0;
        for (int t = try_idx; t < 8; t++) {
            int sbit = (1 << t);
            if (used_mask & sbit) continue;

            int v = S[t];

            // Check adjacency with already-assigned neighbors
            int ok = 1;
            for (int nb = 0; nb < 8; nb++) {
                if (nb >= idx) break;
                if (q3_adj_mask[idx] & (1 << nb)) {
                    int mapped_nb = mapping[nb];
                    if (!(adj[mapped_nb] & (1 << v))) {
                        ok = 0;
                        break;
                    }
                }
            }
            if (!ok) continue;

            // Also check non-adjacency with already-assigned non-neighbors
            // (For induced subgraph isomorphism, but we're looking for subgraph, so only need edges)
            // Actually for subgraph (not induced), we only need the 12 edges to exist.
            // Non-edges don't matter.

            mapping[idx] = v;
            used_mask |= sbit;

            if (idx + 1 < 8) {
                sp++;
                idx_stack[sp - 1] = idx + 1;
                try_stack[sp - 1] = 0;
            } else {
                return 1;
            }
            found = 1;
            break;
        }

        if (!found) {
            // Backtrack
            sp--;
            if (sp > 0) {
                int prev_idx = idx_stack[sp - 1];
                // Undo mapping
                used_mask &= ~(1 << /* which S index? */ 0); // This is tricky
                // Actually we need to track which S index was used
                // Let me redo this more carefully
            }
            mapping[idx] = -1;
            // Need to undo used_mask properly
            // This approach is getting complex. Let me use recursive instead.
        }
    }
    return 0;
}

// Simpler: recursive
int dfs(int idx, int S[8], int mapping[8], int used_mask) {
    if (idx == 8) return 1;
    for (int t = 0; t < 8; t++) {
        int sbit = (1 << t);
        if (used_mask & sbit) continue;
        int v = S[t];
        int ok = 1;
        for (int nb = 0; nb < idx; nb++) {
            if (q3_adj_mask[idx] & (1 << nb)) {
                if (!(adj[mapping[nb]] & (1 << v))) {
                    ok = 0;
                    break;
                }
            }
        }
        if (!ok) continue;
        mapping[idx] = v;
        if (dfs(idx + 1, S, mapping, used_mask | sbit)) return 1;
        mapping[idx] = -1;
    }
    return 0;
}

int check_subset2(int S[8]) {
    int mapping[8];
    memset(mapping, -1, sizeof(mapping));
    mapping[0] = S[0];
    return dfs(1, S, mapping, 1);
}

int main() {
    init_q3();
    // Read n edges from stdin? No, we'll pass adj as command line or stdin.
    // Let's read 16 integers (bitmasks) from stdin
    for (int i = 0; i < 16; i++) {
        scanf("%d", &adj[i]);
    }

    int S[8];
    // Iterate over all C(16,8) = 12870 subsets
    for (int a = 0; a < 16; a++)
    for (int b = a+1; b < 16; b++)
    for (int c = b+1; c < 16; c++)
    for (int d = c+1; d < 16; d++)
    for (int e = d+1; e < 16; e++)
    for (int f = e+1; f < 16; f++)
    for (int g = f+1; g < 16; g++)
    for (int h = g+1; h < 16; h++) {
        S[0] = a; S[1] = b; S[2] = c; S[3] = d;
        S[4] = e; S[5] = f; S[6] = g; S[7] = h;

        // Quick filter: min degree in induced subgraph >= 3
        int min_deg = 8;
        for (int i = 0; i < 8; i++) {
            int deg = 0;
            for (int j = 0; j < 8; j++) {
                if (i != j && (adj[S[i]] & (1 << S[j]))) deg++;
            }
            if (deg < min_deg) min_deg = deg;
        }
        if (min_deg < 3) continue;

        if (check_subset2(S)) {
            printf("1");
            return 0;
        }
    }
    printf("0");
    return 0;
}
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            c_path = os.path.join(tmpdir, 'q3check.c')
            exe_path = os.path.join(tmpdir, 'q3check')
            with open(c_path, 'w') as f:
                f.write(c_code)
            result = subprocess.run(
                ['gcc', '-O2', '-o', exe_path, c_path],
                capture_output=True, timeout=10
            )
            if result.returncode == 0:
                # Test it
                test_result = subprocess.run(
                    [exe_path],
                    input=b' '.join([str(0)] * 16),
                    capture_output=True, timeout=5
                )
                if test_result.returncode == 0 and test_result.stdout.strip() == b'0':
                    fast_checker = exe_path
    except Exception:
        fast_checker = None

    def find_q3_python(adj_mask):
        """Pure Python Q3 finder (fallback)."""
        for S in itertools.combinations(range(n), 8):
            S_list = list(S)

            # Quick filter: min degree >= 3
            min_deg = 8
            for i, v in enumerate(S_list):
                deg = 0
                for j, u in enumerate(S_list):
                    if i != j and (adj_mask[v] >> u) & 1:
                        deg += 1
                if deg < min_deg:
                    min_deg = deg
            if min_deg < 3:
                continue

            # Backtracking
            mapping = [-1] * 8
            mapping[0] = S_list[0]

            def dfs(idx):
                if idx == 8:
                    return True
                for t in range(idx):  # try unused S vertices
                    if mapping[t] == S_list[t]:  # already used as itself? No...
                        pass
                # Better: track used
                used_set = set(mapping[:idx])
                for v in S_list:
                    if v in used_set:
                        continue
                    ok = True
                    for nb in range(idx):
                        if nb in q3_adj[idx]:
                            if not (adj_mask[mapping[nb]] >> v) & 1:
                                ok = False
                                break
                    if not ok:
                        continue
                    mapping[idx] = v
                    if dfs(idx + 1):
                        return True
                    mapping[idx] = -1
                return False

            if dfs(1):
                return [(mapping[u], mapping[v]) for u, v in Q3_EDGES]
        return None

    def find_q3_fast(adj_mask):
        """Use C++ checker if available, return edge list or None."""
        if fast_checker is not None:
            try:
                input_str = ' '.join(str(adj_mask[i]) for i in range(n))
                result = subprocess.run(
                    [fast_checker],
                    input=input_str.encode(),
                    capture_output=True, timeout=5
                )
                if result.stdout.strip() == b'1':
                    # Q3 exists, but we need to find which edges
                    # For now, return a special marker
                    return "Q3_FOUND"
            except Exception:
                pass
        return None  # Fallback to Python if fast checker fails or not available

    def find_q3_full(adj_mask):
        """Find Q3 and return the 12 edges, using fast checker to verify existence first."""
        if fast_checker is not None:
            try:
                input_str = ' '.join(str(adj_mask[i]) for i in range(n))
                result = subprocess.run(
                    [fast_checker],
                    input=input_str.encode(),
                    capture_output=True, timeout=5
                )
                if result.stdout.strip() == b'0':
                    return None  # No Q3
            except Exception:
                pass
        # Use Python to find the actual edges
        return find_q3_python(adj_mask)

    def get_edge_list(adj_mask):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj_mask[u] >> v) & 1:
                    edges.append([u, v])
        return edges

    # Initial solution: K_{3,13} = 39 edges
    def seed_edges():
        return [[a, b] for a in range(3) for b in range(3, 16)]

    best_edges = seed_edges()
    best_m = len(best_edges)

    # Strategy: Start from K_{4,12} (48 edges) and greedily remove edges to eliminate Q3s
    if time.time() < deadline - 10.0:
        adj_mask = [0] * n
        for u in range(4):
            for v in range(4, 16):
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
        current_edges = 48

        # Try multiple random restarts with different edge removal orders
        for restart in range(20):
            if time.time() > deadline - 5.0:
                break

            # Reset to K_{4,12}
            adj_mask = [0] * n
            for u in range(4):
                for v in range(4, 16):
                    adj_mask[u] |= (1 << v)
                    adj_mask[v] |= (1 << u)
            current_edges = 48

            # Greedy removal
            while time.time() < deadline - 3.0:
                q3_edges = find_q3_full(adj_mask)

                if q3_edges is None:
                    if current_edges > best_m:
                        best_m = current_edges
                        best_edges = get_edge_list(adj_mask)
                    break

                # Remove a random edge from the Q3
                random.shuffle(q3_edges)
                u, v = q3_edges[0]
                adj_mask[u] &= ~(1 << v)
                adj_mask[v] &= ~(1 << u)
                current_edges -= 1

                if current_edges <= best_m:
                    break

    # Strategy 2: Try to add edges to best solution
    if time.time() < deadline - 10.0:
        adj_mask_best = [0] * n
        for u, v in best_edges:
            adj_mask_best[u] |= (1 << v)
            adj_mask_best[v] |= (1 << u)
        current_edges = best_m

        missing_edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj_mask_best[u] >> v) & 1:
                    missing_edges.append((u, v))

        random.shuffle(missing_edges)

        for u, v in missing_edges:
            if time.time() > deadline - 3.0:
                break

            adj_mask_best[u] |= (1 << v)
            adj_mask_best[v] |= (1 << u)

            if find_q3_full(adj_mask_best) is None:
                current_edges += 1
                if current_edges > best_m:
                    best_m = current_edges
                    best_edges = get_edge_list(adj_mask_best)
            else:
                adj_mask_best[u] &= ~(1 << v)
                adj_mask_best[v] &= ~(1 << u)

    # Strategy 3: Local search - try swapping edges
    if time.time() < deadline - 15.0:
        adj_mask_ls = [0] * n
        for u, v in best_edges:
            adj_mask_ls[u] |= (1 << v)
            adj_mask_ls[v] |= (1 << u)
        current_edges = best_m

        present_edges = [(u, v) for u in range(n) for v in range(u + 1, n) if (adj_mask_ls[u] >> v) & 1]
        missing_edges_ls = [(u, v) for u in range(n) for v in range(u + 1, n) if not (adj_mask_ls[u] >> v) & 1]

        random.shuffle(present_edges)
        random.shuffle(missing_edges_ls)

        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            for add_u, add_v in missing_edges_ls[:50]:  # Limit per iteration
                if time.time() > deadline - 5.0:
                    break
                for rem_u, rem_v in present_edges[:50]:
                    if time.time() > deadline - 5.0:
                        break
                    if (add_u, add_v) == (rem_u, rem_v):
                        continue
                    # Try swap: remove (rem_u, rem_v), add (add_u, add_v)
                    adj_mask_ls[rem_u] &= ~(1 << rem_v)
                    adj_mask_ls[rem_v] &= ~(1 << rem_u)
                    adj_mask_ls[add_u] |= (1 << add_v)
                    adj_mask_ls[add_v] |= (1 << add_u)

                    if find_q3_full(adj_mask_ls) is None:
                        # Keep the swap
                        present_edges.remove((rem_u, rem_v))
                        present_edges.append((add_u, add_v))
                        missing_edges_ls.remove((add_u, add_v))
                        missing_edges_ls.append((rem_u, rem_v))
                        improved = True
                    else:
                        # Revert
                        adj_mask_ls[rem_u] |= (1 << rem_v)
                        adj_mask_ls[rem_v] |= (1 << rem_u)
                        adj_mask_ls[add_u] &= ~(1 << add_v)
                        adj_mask_ls[add_v] &= ~(1 << add_u)

    return best_edges
