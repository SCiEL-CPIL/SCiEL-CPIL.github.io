import time
import random
import itertools
import sys

# Precomputation
N = 16
EDGES = []
EDGE_INDEX = {}
for i in range(N):
    for j in range(i+1, N):
        EDGE_INDEX[(i, j)] = len(EDGES)
        EDGES.append((i, j))

NUM_EDGES = len(EDGES) # 120

K4S = []
K4_INDEX = {}
for quad in itertools.combinations(range(N), 4):
    K4_INDEX[quad] = len(K4S)
    edges_in_k4 = []
    for u, v in itertools.combinations(quad, 2):
        edges_in_k4.append(EDGE_INDEX[(u, v)])
    K4S.append(edges_in_k4)

NUM_K4S = len(K4S) # 1820

# For each edge, list of K4s containing it
EDGE_TO_K4S = [[] for _ in range(NUM_EDGES)]
for k4_idx, edges in enumerate(K4S):
    for e in edges:
        EDGE_TO_K4S[e].append(k4_idx)

def count_distinct_colors(colors, k4_edges):
    """Count distinct colors in a K4 given the global color array."""
    # k4_edges is a list of 6 edge indices
    c_set = set()
    for e in k4_edges:
        c_set.add(colors[e])
    return len(c_set)

def calculate_violations(colors):
    """Calculate violation count for all K4s."""
    viol = 0
    for edges in K4S:
        d = count_distinct_colors(colors, edges)
        if d < 5:
            viol += (5 - d)
    return viol

def run_sa(K, time_limit, initial_state=None, seed=42):
    random.seed(seed)
    
    if initial_state is None:
        colors = [random.randint(0, K-1) for _ in range(NUM_EDGES)]
    else:
        colors = [c % K for c in initial_state]
        
    # Precompute initial violations per K4
    viol_per_k4 = [0] * NUM_K4S
    total_viol = 0
    for k4_idx, edges in enumerate(K4S):
        d = count_distinct_colors(colors, edges)
        if d < 5:
            viol_per_k4[k4_idx] = 5 - d
            total_viol += viol_per_k4[k4_idx]
            
    best_colors = colors[:]
    best_viol = total_viol
    
    start_time = time.time()
    T = 1.0
    cooling_rate = 0.999
    
    # We want to minimize total_viol.
    # If total_viol == 0, we have a valid solution.
    # We can stop early if we find a valid solution and want to minimize color count?
    # The task says "Minimize the number of colours used".
    # The evaluator measures the raw value (number of colours).
    # So we should return the solution with the minimum number of distinct colours among valid ones.
    # However, finding a valid one is the primary hurdle.
    # We can track the number of distinct colours in the best valid solution found.
    
    best_distinct = K # Max possible
    if best_viol == 0:
        best_distinct = len(set(colors))
        
    while time.time() - start_time < time_limit:
        # Pick random edge
        e = random.randint(0, NUM_EDGES - 1)
        old_c = colors[e]
        new_c = random.randint(0, K-1)
        
        if old_c == new_c:
            continue
            
        # Calculate delta cost
        delta = 0
        affected_k4s = EDGE_TO_K4S[e]
        old_viol_vals = []
        
        for k4_idx in affected_k4s:
            old_v = viol_per_k4[k4_idx]
            old_viol_vals.append((k4_idx, old_v))
            
            # Temporarily update color to calculate new distinct count
            # We can optimize this by just counting distincts in the 6 edges
            # with the new color for e.
            edges = K4S[k4_idx]
            # Get colors of other 5 edges
            other_colors = []
            for ee in edges:
                if ee != e:
                    other_colors.append(colors[ee])
            # Add new color for e
            other_colors.append(new_c)
            new_d = len(set(other_colors))
            new_v = max(0, 5 - new_d)
            delta += new_v - old_v
            
        # Acceptance criterion
        accept = False
        if delta <= 0:
            accept = True
        else:
            # Simulated annealing
            # T decreases
            if random.random() < pow(2.718281828, -delta / T):
                accept = True
                
        if accept:
            colors[e] = new_c
            # Update viol_per_k4
            for k4_idx, old_v in old_viol_vals:
                # Recalculate new_v for this k4 (we did it in delta loop, but need to store)
                # To avoid recalculation, we can store new_v in the loop.
                pass # We'll recalculate in the loop below for correctness/simplicity, it's fast enough (91 * 6)
            
            # Actually, let's just recalculate for affected k4s to update viol_per_k4
            for k4_idx in affected_k4s:
                edges = K4S[k4_idx]
                d = count_distinct_colors(colors, edges)
                viol_per_k4[k4_idx] = max(0, 5 - d)
            
            total_viol += delta
            
            if total_viol < best_viol:
                best_viol = total_viol
                best_colors = colors[:]
                if total_viol == 0:
                    # Check distinct colours
                    d_count = len(set(colors))
                    if d_count < best_distinct:
                        best_distinct = d_count
                        # We could stop here if we find a solution with K=13?
                        # But we are running with fixed K.
                        # If we are running K=13 and find valid, we can stop.
                        if K == 13:
                            return best_colors
                            
        T *= cooling_rate
        if T < 0.01:
            T = 0.01
            
    return best_colors

def construct():
    deadline = 110.0 # 120s limit, keep margin
    start = time.time()
    
    # Try K=13
    if time.time() - start < 5.0:
        res = run_sa(13, time_limit=4.0, seed=1)
        if res and calculate_violations(res) == 0:
            return res
            
    # Try K=14
    if time.time() - start < 15.0:
        # Use previous result as init if available? No, keep independent for now to avoid bias
        res = run_sa(14, time_limit=10.0, seed=2)
        if res and calculate_violations(res) == 0:
            return res
            
    # Try K=15
    if time.time() - start < 100.0:
        res = run_sa(15, time_limit=90.0, seed=3)
        if res and calculate_violations(res) == 0:
            return res
            
    # Fallback to seed
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]

if __name__ == "__main__":
    # Quick test
    c = construct()
    # Verify
    import itertools
    valid = True
    for quad in itertools.combinations(range(16), 4):
        cols = set()
        for u, v in itertools.combinations(quad, 2):
            idx = EDGE_INDEX[(u, v)]
            cols.add(c[idx])
        if len(cols) < 5:
            valid = False
            break
    if valid:
        print("Valid")
    else:
        print("Invalid")
    print("Colours used:", len(set(c)))
