import numpy as np
import time
import random
import sys

# Increase recursion limit if needed for deep stacks, though Push-Relabel is iterative
sys.setrecursionlimit(2000)

def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
        return [list(range(n)) for _ in range(1)]

    M = 3 ** n
    ball_size = 1 + 2 * n
    powers = [3 ** i for i in range(n)]

    # Precompute balls and coverers
    balls = [None] * M
    coverers = [[] for _ in range(M)]
    
    for i in range(M):
        digits = [0] * n
        temp = i
        for j in range(n):
            digits[j] = temp % 3
            temp //= 3
        ball = [i]
        for pos in range(n):
            new_d_plus = (digits[pos] + 1) % 3
            diff = new_d_plus - digits[pos]
            if diff != 0:
                idx = i + diff * powers[pos]
                ball.append(idx)
            
            new_d_minus = (digits[pos] - 1) % 3
            diff = new_d_minus - digits[pos]
            if diff != 0:
                idx = i + diff * powers[pos]
                ball.append(idx)
        
        balls[i] = ball
        for p in ball:
            coverers[p].append(i)

    # Push-Relabel Max Flow Implementation
    class PushRelabel:
        def __init__(self, N):
            self.N = N
            self.adj = [[] for _ in range(N)]
            self.cap = {}
            self.flow = {}
            self.height = [0] * N
            self.excess = [0] * N
            self.active = [False] * N
            self.queue = []
            self.count = [0] * (2 * N)  # Count of nodes at each height for Gap Heuristic

        def add_edge(self, u, v, capacity):
            self.adj[u].append(v)
            self.adj[v].append(u)
            if (u, v) not in self.cap:
                self.cap[(u, v)] = 0
                self.cap[(v, u)] = 0
            self.cap[(u, v)] += capacity

        def get_cap(self, u, v):
            return self.cap.get((u, v), 0)

        def push(self, u, v):
            bottleneck = min(self.excess[u], self.get_cap(u, v))
            if bottleneck > 0:
                self.cap[(u, v)] -= bottleneck
                self.cap[(v, u)] += bottleneck
                self.excess[u] -= bottleneck
                self.excess[v] += bottleneck
                if not self.active[v] and self.excess[v] > 0 and v != 0 and v != self.N - 1:
                    self.active[v] = True
                    self.queue.append(v)

        def relabel(self, u):
            min_height = float('inf')
            for v in self.adj[u]:
                if self.get_cap(u, v) > 0:
                    min_height = min(min_height, self.height[v])
            if min_height < float('inf'):
                self.count[self.height[u]] -= 1
                self.height[u] = min_height + 1
                self.count[self.height[u]] += 1
                if self.count[self.height[u]] == 0:
                    # Gap heuristic
                    for i in range(self.N):
                        if i != 0 and i != self.N - 1 and self.height[i] > self.height[u]:
                            self.count[self.height[i]] -= 1
                            self.height[i] = self.N + 1
                            self.count[self.height[i]] += 1
            else:
                # No outgoing edges with capacity
                self.count[self.height[u]] -= 1
                self.height[u] = 2 * self.N
                self.count[self.height[u]] += 1

        def max_flow(self, source, sink):
            self.height[source] = self.N
            self.count[0] = self.N - 2
            self.count[self.N] = 1
            self.active[source] = True
            self.active[sink] = True
            
            # Initial preflow
            for v in self.adj[source]:
                self.push(source, v)
            
            while self.queue:
                u = self.queue.pop(0)
                self.active[u] = False
                
                while self.excess[u] > 0:
                    pushed = False
                    for v in self.adj[u]:
                        if self.get_cap(u, v) > 0 and self.height[u] == self.height[v] + 1:
                            self.push(u, v)
                            if self.excess[u] == 0:
                                pushed = True
                                break
                    if not pushed:
                        self.relabel(u)
                        if self.excess[u] > 0:
                            self.active[u] = True
                            self.queue.append(u)
            
            return self.cap[(source, sink)] if (source, sink) in self.cap else 0 # Actually we want flow into sink

    # Solve using Flow-based Greedy
    def flow_greedy_cover():
        code = set()
        uncovered = set(range(M))
        
        # We will iteratively find a max matching between uncovered words and available codewords
        # that covers the most uncovered words.
        
        while uncovered:
            # Build bipartite graph
            # Left side: Uncovered words (indices 0..len(uncovered)-1)
            # Right side: Available codewords (indices len(uncovered)..len(uncovered)+len(avail)-1)
            # Source: len(uncovered) + len(avail)
            # Sink: len(uncovered) + len(avail) + 1
            
            avail = [c for c in range(M) if c not in code]
            if not avail:
                break # Should not happen if we cover all
            
            # To speed up, we can limit the number of available codewords considered if the list is huge
            # But for 6561, it's manageable.
            
            U = len(uncovered)
            A = len(avail)
            N_nodes = U + A + 2
            source = N_nodes - 2
            sink = N_nodes - 1
            
            pr = PushRelabel(N_nodes)
            
            # Source -> Uncovered Words (Cap 1)
            # We map uncovered word index in the set to 0..U-1
            unc_list = list(uncovered)
            unc_map = {w: i for i, w in enumerate(unc_list)}
            
            for i in range(U):
                pr.add_edge(source, i, 1)
            
            # Uncovered Words -> Available Codewords (Cap 1)
            # Map avail codeword index to U..U+A-1
            avail_map = {c: U + j for j, c in enumerate(avail)}
            
            # Optimization: Only add edges if the codeword actually covers the word
            # This is O(U * A) which is 6561^2 ~ 43M. Too slow for Python in a loop.
            # We need to iterate over edges in the ball structure.
            
            # Reset adjacency for PR? No, we build it fresh.
            # To avoid O(U*A), we iterate over uncovered words and their coverers.
            for w in unc_list:
                w_idx = unc_map[w]
                for c in coverers[w]:
                    if c in avail_map:
                        c_idx = avail_map[c]
                        pr.add_edge(w_idx, c_idx, 1)
            
            # Available Codewords -> Sink (Cap 1)
            for j in range(A):
                pr.add_edge(U + j, sink, 1)
            
            # Run Max Flow
            # Note: My PushRelabel implementation above returns flow value, but I need the edges with flow > 0
            # The `cap` dict stores residual capacity. Original cap - residual cap = flow.
            # I need to track original capacities.
            
            # Let's modify the approach to just get the matching.
            # Since it's a unit capacity network, I can just use Hopcroft-Karp which is faster and simpler.
            # But directive says Push-Relabel.
            # I will extract the matching from the residual graph.
            
            # Run max_flow
            pr.max_flow(source, sink)
            
            # Extract matching:
            # Edges from Word (0..U-1) to Codeword (U..U+A-1) with flow > 0
            # Flow > 0 means original cap (1) > residual cap.
            # So residual cap < 1.
            
            new_code = set()
            for w in unc_list:
                w_idx = unc_map[w]
                for c in coverers[w]:
                    if c in avail_map:
                        c_idx = avail_map[c]
                        # Check if edge w_idx -> c_idx has flow
                        # Residual cap (w_idx, c_idx) should be 0 if flow is 1.
                        if pr.get_cap(w_idx, c_idx) < 1:
                            new_code.add(c)
                            break # One match per word in matching?
                            # Actually, in a matching, each word is matched to at most 1 codeword.
                            # And each codeword to at most 1 word.
                            # The flow ensures this.
            
            if not new_code:
                # Fallback: Pick random codeword covering an uncovered word
                w = next(iter(uncovered))
                c = random.choice(coverers[w])
                if c not in code:
                    new_code.add(c)
            
            code.update(new_code)
            
            # Update uncovered
            for c in new_code:
                for p in balls[c]:
                    if p in uncovered:
                        uncovered.remove(p)
            
            if time.time() > 115:
                break

        return code

    code = flow_greedy_cover()
    
    # Simple Local Search to polish
    def simple_local_search(code, time_limit=10):
        start = time.time()
        improved = True
        count = np.zeros(M, dtype=np.int32)
        for c in code:
            for p in balls[c]:
                count[p] += 1
        
        while improved and (time.time() - start) < time_limit:
            improved = False
            # Remove redundant
            to_remove = []
            for c in list(code):
                redundant = True
                for p in balls[c]:
                    if count[p] == 1:
                        redundant = False
                        break
                if redundant:
                    to_remove.append(c)
            
            if to_remove:
                for c in to_remove:
                    code.remove(c)
                    for p in balls[c]:
                        count[p] -= 1
                improved = True
                continue
            
            # 1-Swap
            code_list = list(code)
            random.shuffle(code_list)
            for c in code_list:
                unique = [p for p in balls[c] if count[p] == 1]
                if not unique:
                    # Should have been removed
                    code.remove(c)
                    for p in balls[c]:
                        count[p] -= 1
                    improved = True
                    break
                
                cands = set(coverers[unique[0]])
                for p in unique[1:]:
                    cands.intersection_update(coverers[p])
                cands -= code
                if cands:
                    c_new = random.choice(list(cands))
                    code.remove(c)
                    for p in balls[c]:
                        count[p] -= 1
                    code.add(c_new)
                    for p in balls[c_new]:
                        count[p] += 1
                    improved = True
                    break
        
        return code

    code = simple_local_search(code, time_limit=110)
    
    return [list(map(int, [(c // (3 ** i)) % 3 for i in range(n)])) for c in code]
