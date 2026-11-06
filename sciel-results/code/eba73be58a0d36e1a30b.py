import time
import numpy as np
import random
import itertools

def synthesize(matrix, depth_limit):
    start_time = time.time()
    HARD_DEADLINE = start_time + 50.0  # Safety margin for 60s limit (actually 90s eval, but safe)

    def elapsed():
        return time.time() - start_time

    m = len(matrix)
    if m == 0:
        return {'gates': [], 'outputs': []}
    n = len(matrix[0])

    mat = np.array(matrix, dtype=np.uint8) % 2

    # Precompute rows as tuples of set bits
    rows = []
    for i in range(m):
        row = mat[i]
        cols = np.where(row == 1)[0].tolist()
        rows.append(frozenset(cols))

    # Helper to create a circuit
    class Circuit:
        def __init__(self):
            self.gates = []  # list of [wire_a, wire_b]
            self.next_wire = n + 1
            # map from frozenset to wire
            self.memo = {}
            for j in range(n):
                self.memo[frozenset([j])] = j
            self.memo[frozenset()] = n  # wire n is 0

        def add_gate(self, w1, w2):
            self.gates.append([w1, w2])
            w = self.next_wire
            self.next_wire += 1
            return w

        def ensure_set(self, s, deadline):
            """
            Ensure set 's' is represented by a wire. Returns wire id.
            Uses greedy memoization with balanced tree fallback.
            """
            if s in self.memo:
                return self.memo[s]

            if len(s) == 0:
                self.memo[s] = n
                return n

            if len(s) == 1:
                j = next(iter(s))
                self.memo[s] = j
                return j

            if elapsed() > deadline:
                # Fast fallback: balanced tree
                wires = [next(iter(x)) for x in s]
                # Actually need to get the wire for each singleton
                # singletons are just the indices
                wire_list = []
                for j in s:
                    wire_list.append(j)

                while len(wire_list) > 1:
                    new_list = []
                    for i in range(0, len(wire_list) - 1, 2):
                        w1, w2 = wire_list[i], wire_list[i + 1]
                        new_w = self.add_gate(w1, w2)
                        new_list.append(new_w)
                    if len(wire_list) % 2 == 1:
                        new_list.append(wire_list[-1])
                    wire_list = new_list
                w = wire_list[0]
                self.memo[s] = w
                return w

            # Try to find a split s = A ∪ B where A, B are in memo or cheap
            # We iterate over subsets of s that are in memo
            # To avoid duplicates A,B and B,A, we ensure min(size(A), size(B)) is handled
            # Simple approach: check all memo keys that are subsets of s
            
            best_w = None
            best_depth = float('inf')
            
            # Optimization: If we have a very large set, iterating all memo keys is slow.
            # But memo is smallish.
            
            for A, wA in self.memo.items():
                if not A.issubset(s):
                    continue
                # Avoid duplicate work: A must be smaller than or equal to B in some sense?
                # Actually, if A == s, B = empty. If A == empty, B = s.
                # We just need a valid split.
                B = s - A
                
                # If B is in memo, we have a 1-gate solution
                if B in self.memo:
                    wB = self.memo[B]
                    # We found a split. Use it.
                    # Is this optimal? Locally yes (1 gate).
                    # We can return immediately?
                    # Yes, a 1-gate solution is the best possible for a new set.
                    new_w = self.add_gate(wA, wB)
                    self.memo[s] = new_w
                    return new_w
                else:
                    # B not in memo.
                    # Estimate cost.
                    # Only consider if A is "large" enough to be helpful?
                    # Or just pick the best heuristic split?
                    # Let's pick the split that minimizes (len(B) + 1) roughly.
                    # Since we can't compute B cost exactly without recursion,
                    # and recursion might be expensive, we should be careful.
                    
                    # Heuristic: Prefer A such that B is small.
                    # If B is small, it's cheap to compute.
                    pass

            # If no 1-gate split found, use a greedy strategy:
            # Pick an element x in s. Let A = {x}, B = s - {x}.
            # This always works but might be deep.
            # Better: Pick a subset A of s that is large but "simple"?
            # For now, let's use the singleton strategy or a 2-element strategy to keep depth low.
            
            # Let's try a few random subsets to find a good split?
            # Or just balanced tree?
            # Balanced tree is safe but uses O(popcount) gates.
            # Greedy memoization often finds better.
            
            # Let's try to find ANY A in memo that is a proper subset of s and s-A is also "nice".
            # Since we didn't find 1-gate, we must recurse.
            # To avoid deep recursion, let's just do balanced tree if memo search fails?
            # No, let's do a recursive split.
            # Pick A = {some element}
            x = next(iter(s))
            A = frozenset([x])
            B = s - A
            
            # Ensure A is in memo (it is, singleton)
            # Recurse on B
            wB = self.ensure_set(B, deadline)
            wA = self.memo[A]
            
            new_w = self.add_gate(wA, wB)
            self.memo[s] = new_w
            return new_w

    circuit = Circuit()
    
    # Process rows
    # Sort by popcount?
    indices = list(range(m))
    # Random shuffle to avoid bias? Or sort by complexity.
    # Sorting by popcount ascending helps populate memo with small sets first.
    indices.sort(key=lambda i: len(rows[i]))
    
    # Maybe also try a different order?
    # Let's stick to popcount sort.
    
    final_outputs = [0] * m
    
    for i in indices:
        s = rows[i]
        w = circuit.ensure_set(s, HARD_DEADLINE)
        final_outputs[i] = w
        
    # Verify and refine?
    # With 50s, we might have time for improvement.
    # But the current method is already decent.
    # Let's check if we can do better by re-synthesizing with different random choices?
    
    # For now, return the result.
    return {'gates': circuit.gates, 'outputs': final_outputs}
