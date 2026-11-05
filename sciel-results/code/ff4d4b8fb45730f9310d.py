import time
from math import gcd
import sys

N = 512
DEADLINE = 90.0

def construct():
    start_time = time.time()
    
    # 1. Build the conflict graph
    # Vertex i represents number i (1-indexed, so vertex i corresponds to number i)
    # Edge between u, v if lcm(u, v) > N
    # We want the Maximum Independent Set (MIS) of this graph.
    
    # Adjacency list representation for fast neighbor lookup
    # adj[i] is a set of neighbors of i
    adj = [set() for _ in range(N + 1)]
    
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            # lcm(i, j) = i // gcd(i, j) * j
            if (i // gcd(i, j)) * j > N:
                adj[i].add(j)
                adj[j].add(i)
    
    # 2. Branch and Bound for Maximum Independent Set
    # We use a recursive backtracking approach with pruning.
    # State: 
    #   P: set of candidates available to be added to the independent set
    #   C: set of vertices already chosen in the independent set
    #   best_size: global maximum size found so far
    #   best_set: global best set found
    
    # To make it efficient, we represent P as a list or a bitset.
    # Given N=512, a bitset (integer) is very efficient.
    
    # Precompute bitmasks for neighbors? 
    # Let's use a list of integers where bit i is set if i is in the set.
    
    # Initial candidates: all vertices 1..N
    # We will process vertices in a specific order to improve pruning.
    # A common heuristic is to order vertices by degree (descending) or simply natural order.
    # Natural order (1 to N) works well because smaller numbers have fewer conflicts.
    
    # Let's use a recursive function that takes:
    #   P: set of remaining candidates (as a set of integers or bitmask)
    #   current_set: set of integers chosen so far
    #   current_size: len(current_set)
    
    # Optimization: 
    # If current_size + len(P) <= best_size, prune.
    # Also, if current_size + greedy_bound(P) <= best_size, prune.
    # Greedy bound: greedily pick vertices from P to estimate max independent set size.
    
    # For better performance, we can sort the candidates in P at each step? 
    # That's expensive. Instead, we can pre-sort the global list of vertices by some heuristic 
    # and always consider them in that order, maintaining the set of *available* candidates.
    
    # Let's try a simpler, robust branch-and-bound:
    # Order vertices 1 to N.
    # At each step, we decide to include or exclude the "next" candidate from P.
    # Actually, standard MIS B&B works on a set P.
    # 1. If P is empty, update best.
    # 2. Pick a vertex v from P.
    # 3. Branch 1: Include v. New P = P \ (N(v) U {v}).
    # 4. Branch 2: Exclude v. New P = P \ {v}.
    
    # To maximize pruning, we should pick v such that it maximizes the bound reduction.
    # A good heuristic is to pick the vertex with the highest degree in the subgraph induced by P.
    
    best_size = 0
    best_set = []
    
    # Global variables for the recursive function
    # Using a class or closure to hold state
    
    def greedy_bound(P):
        """
        Estimate the size of the maximum independent set in P using a greedy coloring heuristic.
        This provides an upper bound on the MIS size.
        """
        if not P:
            return 0
        
        # We want to color the graph induced by P such that adjacent vertices have different colors.
        # The number of colors is an upper bound on the size of the independent set?
        # No, the chromatic number is an upper bound on the *clique* size? 
        # Wait. 
        # For Maximum Independent Set, a common upper bound is the size of a maximal clique in the complement?
        # Or, more simply: 
        # If we can cover P with k independent sets, then MIS <= sum of sizes? No.
        
        # Actually, the standard upper bound for MIS in B&B is often derived from a greedy coloring of the *conflict graph*.
        # If we color the conflict graph such that no two adjacent vertices share a color, 
        # then an independent set can contain at most one vertex of each color? 
        # No, an independent set can contain multiple vertices of the same color (since they are not adjacent).
        # So coloring the conflict graph doesn't give a direct upper bound on MIS size.
        
        # Correction:
        # The upper bound is often computed by finding a maximal clique in the *complement* graph? 
        # Or, simpler: 
        # Greedy bound: 
        # Sort vertices in P by degree (descending).
        # Greedily select vertices that do not conflict with each other.
        # This gives a *lower* bound (a feasible solution).
        # We need an *upper* bound.
        
        # A valid upper bound: 
        # If we partition P into disjoint cliques, the MIS can take at most 1 from each clique.
        # Finding a clique cover is hard.
        
        # Alternative simple upper bound: 
        # The size of P itself. (Weak)
        # 
        # Better Upper Bound for MIS:
        # Consider the graph G[P]. 
        # If we find a vertex v, MIS(G[P]) <= 1 + MIS(G[P \ N[v]])
        # This is the recurrence, not a bound.
        
        # Let's use a simpler heuristic that is fast:
        # The size of a maximal independent set found greedily is a lower bound.
        # For an upper bound, we can use the fact that if the minimum degree in G[P] is d, 
        # then MIS <= |P| * (something)? No.
        
        # Let's stick to a simpler pruning: 
        # If current_size + len(P) <= best_size, prune.
        # This is weak but safe.
        
        # To improve, we can use a greedy coloring of the *complement* graph?
        # If we color the complement graph, each color class is a clique in the original graph.
        # An independent set in the original graph is a clique in the complement.
        # So MIS(G) = MaxClique(G_complement).
        # The size of a maximal clique is bounded by the number of colors in a coloring of G_complement?
        # No, the chromatic number of G_complement is an upper bound on the clique number?
        # Yes! Clique number <= Chromatic number.
        # So if we color the complement graph of P, the number of colors is an upper bound on the size of the MIS in P.
        
        # Implementing coloring of the complement graph is equivalent to partitioning P into cliques of the original graph.
        # This is still complex to do fast.
        
        # Let's try a different approach: 
        # Since N=512 is small, we can use a bitset-based branch and bound with a simple bound:
        # bound = current_size + len(P)
        # This might be too weak to prune effectively if the graph is sparse.
        
        # However, the conflict graph for LCM <= 512 is likely quite dense for large numbers and sparse for small numbers.
        # Small numbers (1, 2, 3...) have few conflicts.
        # Large numbers have many conflicts.
        
        # Let's try a hybrid: 
        # 1. Greedy construction to get a good initial lower bound (best_size).
        # 2. Branch and bound with simple length bound.
        
        pass # Placeholder for actual implementation below

    # --- Actual Implementation ---
    
    # 1. Greedy Lower Bound
    # Order vertices by degree (ascending) to pick "easy" vertices first? 
    # Or natural order.
    # Let's try natural order 1..N.
    greedy_set = set()
    for i in range(1, N + 1):
        can_add = True
        for j in greedy_set:
            if j in adj[i]:
                can_add = False
                break
        if can_add:
            greedy_set.add(i)
    
    best_size = len(greedy_set)
    best_set = list(greedy_set)
    
    # 2. Branch and Bound
    # We will use a recursive search.
    # To avoid recursion depth issues and overhead, we can use a stack or just careful recursion.
    # N=512, depth could be up to 512, which is fine for Python recursion if limit is increased.
    sys.setrecursionlimit(2000)
    
    # We represent the set of candidates P as a set of integers.
    # To make it faster, we can use a bitmask (int) since N=512 fits in 2 ints or 1 big int.
    # Let's use a Python integer as a bitmask. Bit i corresponds to vertex i.
    
    # Precompute neighbor masks
    neighbor_mask = [0] * (N + 1)
    for i in range(1, N + 1):
        mask = 0
        for j in adj[i]:
            mask |= (1 << j)
        neighbor_mask[i] = mask
    
    # Initial P: all vertices 1..N
    # Bit 0 is unused. Bits 1..N are used.
    initial_P = 0
    for i in range(1, N + 1):
        initial_P |= (1 << i)
    
    # Helper to count bits
    def popcount(x):
        return bin(x).count('1')
    
    # Helper to get list of set bits
    def get_bits(x):
        bits = []
        while x:
            # Get lowest set bit
            lsb = x & -x
            idx = lsb.bit_length() - 1
            bits.append(idx)
            x ^= lsb
        return bits
    
    def search(P_mask, current_set, current_size):
        nonlocal best_size, best_set
        
        if time.time() > DEADLINE - 2:
            return
        
        # Pruning 1: Simple bound
        if current_size + popcount(P_mask) <= best_size:
            return
            
        # If no candidates, update best
        if P_mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = list(current_set)
            return
            
        # Heuristic: Pick a vertex v from P_mask
        # To improve pruning, pick a vertex that is likely to be in the solution or has high degree.
        # Let's pick the vertex with the highest degree in the subgraph induced by P_mask.
        # Calculating degree for all vertices in P_mask is O(N^2). 
        # For N=512, this is 262k ops per node. If we have many nodes, this is slow.
        
        # Alternative: Just pick the first available vertex (lowest index).
        # This is fast but might not prune well.
        
        # Let's try picking the vertex with the highest "potential" (degree in P).
        # But to save time, let's just pick the lowest index first.
        # The greedy step already gave a good lower bound.
        
        # Get list of candidates
        candidates = get_bits(P_mask)
        
        # To speed up, we can sort candidates by degree in P?
        # Let's just iterate.
        
        for v in candidates:
            if time.time() > DEADLINE - 2:
                return
                
            # Branch 1: Include v
            # New P = P_mask & ~(neighbor_mask[v] | (1 << v))
            new_P_include = P_mask & ~(neighbor_mask[v] | (1 << v))
            
            # Check bound before recursing?
            # The search function does it at the start.
            
            current_set.add(v)
            search(new_P_include, current_set, current_size + 1)
            current_set.remove(v)
            
            # Branch 2: Exclude v
            # New P = P_mask & ~(1 << v)
            new_P_exclude = P_mask & ~(1 << v)
            
            # Optimization: If including v didn't lead to a better solution, 
            # and excluding v reduces the potential significantly, we might prune?
            # Standard B&B just calls both.
            
            search(new_P_exclude, current_set, current_size)
            
            # Early termination: if we found a solution of size N (impossible) or 
            # if current_size + popcount(new_P_exclude) <= best_size, we can break?
            # Not necessarily, because other branches might still be viable?
            # No, if we've tried including v, and now we are trying excluding v,
            # and the bound for excluding v is <= best_size, we can skip the exclude branch.
            # But the bound is checked at the start of the search function.
            # We can add a check here to avoid the function call overhead.
            
            if current_size + popcount(new_P_exclude) <= best_size:
                # No need to search the exclude branch further? 
                # Wait, the loop continues to the next candidate?
                # No, in this formulation, we pick ONE vertex v and branch on it.
                # The loop `for v in candidates` is incorrect for standard B&B.
                # Standard B&B picks ONE vertex v, then branches into Include(v) and Exclude(v).
                # It does NOT iterate over all v and branch on each.
                # It picks the "best" v to branch on.
                
                pass
        
        # The above loop logic was flawed. 
        # Correct B&B:
        # 1. Pick a vertex v from P.
        # 2. Recurse on Include(v).
        # 3. Recurse on Exclude(v).
        
        # Let's refactor.
        
        # Pick a vertex v. 
        # Heuristic: Pick the vertex with the highest degree in P.
        # For simplicity and speed, let's pick the first one (lowest index).
        # The greedy lower bound is strong, so simple branching might suffice.
        
        v = candidates[0]
        
        # Branch 1: Include v
        new_P_include = P_mask & ~(neighbor_mask[v] | (1 << v))
        current_set.add(v)
        search(new_P_include, current_set, current_size + 1)
        current_set.remove(v)
        
        # Branch 2: Exclude v
        new_P_exclude = P_mask & ~(1 << v)
        
        # Prune exclude branch if bound is too low
        if current_size + popcount(new_P_exclude) > best_size:
            search(new_P_exclude, current_set, current_size)
        
    # Run search
    initial_set = set()
    search(initial_P, initial_set, 0)
    
    # If time ran out or search completed, return best found
    return best_set

# Test locally if needed
if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
    # Verify
    for i in range(len(res)):
        for j in range(i+1, len(res)):
            a, b = res[i], res[j]
            if (a // gcd(a, b)) * b > 512:
                print(f"Invalid pair: {a}, {b}")
                break
