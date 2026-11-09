import time
import math
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    N = 256
    
    # Precompute 4-AP tuples
    four_ap_tuples = []
    for d in range(1, (N - 1) // 3 + 1):
        max_a = N - 3 * d
        for a in range(1, max_a + 1):
            four_ap_tuples.append((a, a + d, a + 2 * d, a + 3 * d))
    
    # Precompute 3-AP tuples
    three_ap_tuples = []
    for d in range(1, (N - 1) // 2 + 1):
        max_a = N - 2 * d
        for a in range(1, max_a + 1):
            three_ap_tuples.append((a, a + d, a + 2 * d))
    
    # --- Maximum Weight Closure Formulation ---
    # We want to maximize sum_{x in S} w_x subject to: for all 4-APs (a,b,c,d), not (a,b,c,d in S).
    # This is equivalent to: Maximize sum_{x in S} w_x - sum_{4-APs fully in S} M
    # where M is a large penalty.
    # This can be modeled as a Minimum Cut problem on a Directed Acyclic Graph (DAG).
    
    # Graph Nodes:
    # 1. Source (s)
    # 2. Sink (t)
    # 3. Element nodes: 1..N (representing inclusion in S)
    # 4. Constraint nodes: One for each 4-AP (representing the "violation" if all 4 elements are chosen)
    
    # Edges:
    # 1. s -> element_i: capacity = w_i (if w_i > 0)
    # 2. element_i -> t: capacity = -w_i (if w_i < 0)
    # 3. element_j -> constraint_k: capacity = infinity (if element_j is part of 4-AP_k)
    # 4. constraint_k -> t: capacity = M (penalty)
    
    # The min-cut will separate the "included" nodes (reachable from s) from the "excluded" nodes.
    # If an element is on the s-side, it's in S.
    # If a constraint node is on the s-side, it means all its elements are on the s-side (otherwise there would be an inf edge from s-side to t-side, which is impossible in min-cut).
    # Thus, if a constraint node is on the s-side, the 4-AP is fully included, and we pay the penalty M.
    # The weight of the closure is sum(w_i for i in S) - M * (number of 4-APs in S).
    # To maximize this, we set M to be larger than the sum of all positive weights, ensuring no 4-APs are included if possible.
    # However, we also want to maximize the number of 3-APs.
    # The number of 3-APs is NOT linear in the inclusion variables.
    # A 3-AP (a,b,c) exists if a,b,c in S.
    # This is a quadratic term: 1_{a in S} * 1_{b in S} * 1_{c in S}.
    # Max Weight Closure handles linear objectives.
    # 
    # Wait, the problem asks to MAXIMIZE the number of 3-APs, subject to NO 4-APs.
    # If we set w_i = 1 for all i, we maximize the size of S.
    # This does not necessarily maximize the number of 3-APs.
    # For example, a set of size 100 might have fewer 3-APs than a set of size 90 that is more "structured".
    # 
    # However, note the knowledge: "Measured: 1.148 -> 1.306" with Iterated Local Search.
    # The seed score is 1.0.
    # The inspirations suggest that density heuristics or SA work.
    # 
    # Can we encode 3-AP count in the weights?
    # No, because 3-AP count is a high-order interaction.
    # 
    # Alternative:
    # Maybe the "Maximize 3-APs" objective is correlated with "Maximize Size" in the context of "No 4-APs"?
    # Actually, for fixed size, the number of 3-APs varies.
    # 
    # Let's look at the constraint: No 4-APs.
    # If we maximize size, we get a set of maximum size with no 4-APs.
    # Does this set have the maximum number of 3-APs?
    # Not necessarily.
    # 
    # However, the prompt says: "Maximize the number of distinct three-term arithmetic progressions".
    # 
    # If Max Weight Closure only solves for linear objectives, it might not be the right tool for the 3-AP count objective directly.
    # 
    # BUT, we can use Max Weight Closure to find the **Maximum Size** set with no 4-APs.
    # Then, we can use that as a starting point for a local search that optimizes the 3-AP count?
    # Or, can we formulate the 3-AP count as a closure problem?
    # 
    # Actually, there is a known result: The problem of maximizing the number of k-APs in a set with no (k+1)-APs is complex.
    # 
    # Let's re-read the directive: "Model the problem as a maximum weight closure ... to guarantee a globally optimal set for the fixed 4-AP constraint structure".
    # 
    # If I set weights w_i = 1, I get the max size set.
    # If I set weights based on "potential" to form 3-APs, it's still linear approximation.
    # 
    # Let's try a hybrid:
    # 1. Use Max Flow to find the maximum size subset with no 4-APs.
    # 2. Use this subset as the initial state for a fast local search that optimizes the 3-AP count.
    # 
    # Why is this better?
    # The seed program uses greedy initialization which might get stuck in small sets.
    # Max Flow guarantees the largest possible set (or close to it, if M is tuned).
    # A larger set has more "room" for 3-APs.
    # 
    # Let's implement Max Flow for Max Size.
    
    # --- Max Flow Implementation (Dinic's Algorithm) ---
    
    class Dinic:
        def __init__(self, n):
            self.n = n
            self.graph = [[] for _ in range(n)]
            self.level = [0] * n
            self.it = [0] * n
            
        def add_edge(self, u, v, cap):
            # Add forward edge and backward edge
            self.graph[u].append([v, cap, len(self.graph[v])])
            self.graph[v].append([u, 0, len(self.graph[u]) - 1])
            
        def bfs(self, s, t):
            self.level = [-1] * self.n
            self.level[s] = 0
            queue = [s]
            for u in queue:
                for v, cap, rev in self.graph[u]:
                    if cap > 0 and self.level[v] < 0:
                        self.level[v] = self.level[u] + 1
                        queue.append(v)
            return self.level[t] >= 0
            
        def dfs(self, v, t, f):
            if v == t:
                return f
            for i in range(self.it[v], len(self.graph[v])):
                self.it[v] = i
                v2, cap, rev = self.graph[v][i]
                if cap > 0 and self.level[v] + 1 == self.level[v2]:
                    d = self.dfs(v2, t, min(f, cap))
                    if d > 0:
                        self.graph[v][i][1] -= d
                        self.graph[v2][rev][1] += d
                        return d
            return 0
            
        def max_flow(self, s, t):
            flow = 0
            INF = 10**18
            while self.bfs(s, t):
                self.it = [0] * self.n
                while True:
                    f = self.dfs(s, t, INF)
                    if f == 0:
                        break
                    flow += f
            return flow

    # Build Graph for Max Weight Closure (Max Size)
    # Nodes:
    # 0: Source
    # 1..N: Elements 1..N
    # N+1..N+K: Constraint nodes for 4-APs (K = len(four_ap_tuples))
    # N+K+1: Sink
    
    K = len(four_ap_tuples)
    num_nodes = N + K + 2
    source = 0
    sink = num_nodes - 1
    
    dinic = Dinic(num_nodes)
    
    # Weights w_i = 1 for all elements (to maximize size)
    # Edge s -> i: cap 1
    # Edge i -> t: cap 0
    for i in range(1, N + 1):
        dinic.add_edge(source, i, 1)
        
    # Constraint nodes
    # For each 4-AP (a,b,c,d), create a node node_id = N + idx + 1
    # Edges: a -> node_id, b -> node_id, c -> node_id, d -> node_id with INF capacity
    # Edge: node_id -> sink with capacity M (large penalty)
    
    INF_CAP = 10**18
    M = 1000 # Penalty must be > max possible weight sum (N=256). So M=1000 is safe.
    
    for idx, (a, b, c, d) in enumerate(four_ap_tuples):
        node_id = N + idx + 1
        dinic.add_edge(a, node_id, INF_CAP)
        dinic.add_edge(b, node_id, INF_CAP)
        dinic.add_edge(c, node_id, INF_CAP)
        dinic.add_edge(d, node_id, INF_CAP)
        dinic.add_edge(node_id, sink, M)
        
    # Solve Max Flow
    flow = dinic.max_flow(source, sink)
    
    # Extract Solution
    # Nodes reachable from source in the residual graph are in the "Source Set".
    # We need to find which nodes are reachable.
    visited = [False] * num_nodes
    stack = [source]
    visited[source] = True
    while stack:
        u = stack.pop()
        for v, cap, rev in dinic.graph[u]:
            if cap > 0 and not visited[v]:
                visited[v] = True
                stack.append(v)
                
    S_closure = set()
    for i in range(1, N + 1):
        if visited[i]:
            S_closure.add(i)
            
    # Verify S_closure has no 4-APs
    # It should, by construction, because if it had a 4-AP, the constraint node would be reachable,
    # and the edge constraint_node -> sink would be cut, incurring cost M.
    # Since M is large, the min-cut avoids this if possible.
    # However, if it's impossible to avoid all 4-APs (which is not the case here, empty set is valid),
    # it might include some. But we want NO 4-APs.
    # The max size set with NO 4-APs is the optimal solution to the "Max Size" problem.
    # The closure formulation with penalty M finds the set that maximizes Size - M * (#4-APs).
    # If M is large enough, it will find a set with 0 4-APs if one exists with max size.
    
    # Check if S_closure is valid
    def has_4ap(S):
        for a, b, c, d in four_ap_tuples:
            if a in S and b in S and c in S and d in S:
                return True
        return False
        
    if has_4ap(S_closure):
        # Fallback to greedy if closure failed (shouldn't happen)
        S_closure = set()
        for x in range(1, N + 1):
            if not has_4ap(S_closure | {x}):
                S_closure.add(x)
                
    # Now, we have a large valid set S_closure.
    # We need to maximize the number of 3-APs.
    # We will use Local Search on this set.
    
    # Precompute 3-AP partners for fast delta calculation
    three_ap_partners = {i: [] for i in range(1, N + 1)}
    for a, b, c in three_ap_tuples:
        three_ap_partners[a].append((b, c))
        three_ap_partners[b].append((a, c))
        three_ap_partners[c].append((a, b))
        
    def count_3aps(S):
        count = 0
        for a, b, c in three_ap_tuples:
            if a in S and b in S and c in S:
                count += 1
        return count
        
    def creates_4ap(S, x):
        for idx, (a, b, c, d) in enumerate(four_ap_tuples):
            if x == a:
                if b in S and c in S and d in S: return True
            elif x == b:
                if a in S and c in S and d in S: return True
            elif x == c:
                if a in S and b in S and d in S: return True
            elif x == d:
                if a in S and b in S and c in S: return True
        return False
        
    def delta_add(S, x):
        delta = 0
        for u, v in three_ap_partners[x]:
            if u in S and v in S:
                delta += 1
        return delta
        
    # Local Search to optimize 3-AP count
    S = set(S_closure)
    score = count_3aps(S)
    no_improve_count = 0
    max_no_improve = 5000
    
    while time.time() < deadline - 1.0 and no_improve_count < max_no_improve:
        improved = False
        
        # 1. Try additions
        best_add = None
        best_add_delta = 0
        for x in range(1, N + 1):
            if x not in S:
                if not creates_4ap(S, x):
                    d = delta_add(S, x)
                    if d > best_add_delta:
                        best_add_delta = d
                        best_add = x
                        
        if best_add is not None and best_add_delta > 0:
            S.add(best_add)
            score += best_add_delta
            improved = True
            no_improve_count = 0
            continue
            
        # 2. Try swaps if stuck
        # Find element with lowest "value" (contribution to 3-APs) and try to swap with high potential element
        # Or just random swaps
        
        if len(S) < 2:
            break
            
        # Random swap
        x = random.choice(list(S))
        y = random.choice([i for i in range(1, N + 1) if i not in S])
        
        # Check validity of S - {x} + {y}
        valid = True
        # We need to check if y creates a 4-AP with S - {x}
        # This is equivalent to: does y create a 4-AP with S, where one of the partners is x?
        # Or simply: iterate 4-APs containing y.
        for idx, (a, b, c, d) in enumerate(four_ap_tuples):
            if y not in (a, b, c, d):
                continue
            others = [v for v in (a, b, c, d) if v != y]
            # Check if all others are in S - {x}
            # i.e., others \ {x} subset of S
            ok = True
            for v in others:
                if v == x:
                    continue
                if v not in S:
                    ok = False
                    break
            if ok:
                valid = False
                break
                
        if valid:
            S_temp = set(S)
            S_temp.remove(x)
            S_temp.add(y)
            new_score = count_3aps(S_temp)
            if new_score > score:
                S = S_temp
                score = new_score
                improved = True
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1
            
        if not improved:
            no_improve_count += 1
            if no_improve_count > 200:
                break
                
    return sorted(S)

# Test
if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
    # Count 3-APs
    count = 0
    for d in range(1, 128):
        for a in range(1, 256 - 2*d + 1):
            if a in res and a+d in res and a+2*d in res:
                count += 1
    print(f"3-APs: {count}")
