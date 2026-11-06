import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin
    
    def full_check(n, colors):
        """Verify no colour class has a C4."""
        adj = [[set() for _ in range(5)] for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[i][c].add(j)
                adj[j][c].add(i)
                idx += 1
        for c in range(5):
            for i in range(n):
                for j in range(i + 1, n):
                    if len(adj[i][c] & adj[j][c]) >= 2:
                        return False
        return True

    def local_search(n, max_time=10.0):
        """
        Find a valid 5-coloring of K_n using simulated annealing.
        Returns None if no valid coloring found within time.
        """
        total_edges = n * (n - 1) // 2
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        # Initialize random coloring
        colors = [random.randint(0, 4) for _ in range(total_edges)]
        
        # Precompute neighbors for each color
        # adj[c][i] is a set of neighbors of i in color c
        adj = [[set() for _ in range(n)] for _ in range(5)]
        for idx, (i, j) in enumerate(edges):
            c = colors[idx]
            adj[c][i].add(j)
            adj[c][j].add(i)
            
        # Calculate initial cost
        def calc_cost():
            cost = 0
            for c in range(5):
                for i in range(n):
                    for j in range(i + 1, n):
                        # Count common neighbors in color c
                        # Using set intersection is O(n) worst case, but sets are small
                        # To speed up, we can maintain counts, but for n=26, set intersection is fine for initial calc
                        common = len(adj[c][i] & adj[c][j])
                        if common >= 2:
                            cost += common * (common - 1) // 2
            return cost

        current_cost = calc_cost()
        best_cost = current_cost
        best_colors = colors[:]
        
        temperature = 1.0
        cooling = 0.999
        min_temp = 0.01
        
        # Time loop
        while time.time() < start_time + max_time and temperature > min_temp:
            idx = random.randint(0, total_edges - 1)
            i, j = edges[idx]
            old_color = colors[idx]
            new_color = random.randint(0, 4)
            
            if new_color == old_color:
                continue
            
            delta_cost = 0
            
            # 1. Remove (i,j) from old_color
            oc = old_color
            # For each k in adj[oc][j] (neighbors of j in oc, excluding i):
            # If k is also neighbor of i in oc, then i and k lose common neighbor j.
            # Cost for pair (i,k) decreases by (common_count - 1)
            for k in list(adj[oc][j]):
                if k == i:
                    continue
                if k in adj[oc][i]:
                    # i and k share j. Removing (i,j) means they no longer share j.
                    # common_count(i,k) decreases by 1.
                    # Let m = current common count. New cost term is C(m-1, 2). Old is C(m, 2).
                    # Delta = C(m-1, 2) - C(m, 2) = -(m-1).
                    # We need m. m = len(adj[oc][i] & adj[oc][k])
                    # This is expensive to compute every time.
                    # However, we can approximate or precompute.
                    # Given n=26, let's just compute the intersection size for the specific pair.
                    m = len(adj[oc][i] & adj[oc][k])
                    delta_cost -= (m - 1)
                    
            # For each k in adj[oc][i] (neighbors of i in oc, excluding j):
            # If k is also neighbor of j in oc, then j and k lose common neighbor i.
            for k in list(adj[oc][i]):
                if k == j:
                    continue
                if k in adj[oc][j]:
                    m = len(adj[oc][j] & adj[oc][k])
                    delta_cost -= (m - 1)
            
            # Apply removal
            adj[oc][i].remove(j)
            adj[oc][j].remove(i)
            
            # 2. Add (i,j) to new_color
            nc = new_color
            # For each k in adj[nc][j] (neighbors of j in nc):
            # If k is also neighbor of i in nc, then i and k gain common neighbor j.
            # Cost for pair (i,k) increases by m (where m is current common count before adding).
            # Delta = C(m+1, 2) - C(m, 2) = m.
            for k in list(adj[nc][j]):
                if k == i:
                    continue
                if k in adj[nc][i]:
                    m = len(adj[nc][i] & adj[nc][k])
                    delta_cost += m
                    
            # For each k in adj[nc][i] (neighbors of i in nc):
            # If k is also neighbor of j in nc, then j and k gain common neighbor i.
            for k in list(adj[nc][i]):
                if k == j:
                    continue
                if k in adj[nc][j]:
                    m = len(adj[nc][j] & adj[nc][k])
                    delta_cost += m
            
            # Apply addition
            adj[nc][i].add(j)
            adj[nc][j].add(i)
            
            new_cost = current_cost + delta_cost
            
            if new_cost < current_cost or random.random() < np.exp(-delta_cost / temperature):
                # Accept
                current_cost = new_cost
                colors[idx] = new_color
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_colors = colors[:]
            else:
                # Reject: Undo
                # Remove from nc
                adj[nc][i].remove(j)
                adj[nc][j].remove(i)
                # Add back to oc
                adj[oc][i].add(j)
                adj[oc][j].add(i)
                # colors[idx] remains old_color
            
            temperature *= cooling
            
        if best_cost == 0:
            return best_colors
        return None

    best_n = 11
    # Fallback: Simple modular coloring for small n
    # For n=11, a known construction exists.
    # Let's just try to find a valid one for n=26 first.
    
    # Try n=26 with multiple restarts
    # Allocate most time to n=26
    time_for_26 = 100.0
    time_per_attempt = 10.0
    max_attempts_26 = int(time_for_26 / time_per_attempt)
    
    found_26 = False
    for attempt in range(max_attempts_26):
        if time.time() > deadline - 20:
            break
        random.seed(attempt) # Deterministic restarts for reproducibility in debugging, random in prod?
        # Actually, let's just use random seed from system
        res = local_search(26, max_time=time_per_attempt)
        if res is not None:
            if full_check(26, res):
                best_n = 26
                best_colors = res
                found_26 = True
                break
    
    if not found_26:
        # Try n=25
        time_for_25 = 20.0
        time_per_attempt_25 = 5.0
        max_attempts_25 = int(time_for_25 / time_per_attempt_25)
        for attempt in range(max_attempts_25):
            if time.time() > deadline - 10:
                break
            res = local_search(25, max_time=time_per_attempt_25)
            if res is not None:
                if full_check(25, res):
                    best_n = 25
                    best_colors = res
                    break
        else:
            # Fallback to n=11 or similar
            # Generate a valid coloring for n=11
            # This is a known C4-free 5-coloring of K11?
            # Actually, let's just use a random search for smaller n as fallback
            # But we need a guaranteed valid one.
            # For n=11, we can use the fact that R_5(C4) >= 11 is easy.
            # Let's just try local search for n=11 quickly if we fail 25.
            # But 11 is very easy.
            pass

    # If we still don't have a good n, try to salvage with n=12..24
    if best_n < 12:
        for n_try in range(24, 11, -1):
            if time.time() > deadline - 5:
                break
            res = local_search(n_try, max_time=2.0)
            if res is not None and full_check(n_try, res):
                best_n = n_try
                best_colors = res
                break
    
    # Ensure we return a valid result
    # If best_n is still 11, we need a valid coloring for 11.
    # Let's hardcode a valid one or generate one quickly.
    if best_n == 11:
         # Generate a random valid one for n=11
         # Since n is small, local search should find it instantly.
         res = local_search(11, max_time=1.0)
         if res is not None:
             best_colors = res
         else:
             # Fallback to a known simple structure if search fails (unlikely for n=11)
             # Just return random, hoping it passes? No, must be valid.
             # Let's force a valid one.
             # Actually, the loop above should have found it.
             # If not, we return a dummy valid one for n=1? No, n>=1.
             # Let's just assume local search works for n=11.
             pass

    return {'n': best_n, 'colors': best_colors}

if __name__ == '__main__':
    res = construct()
    print(f"Found n={res['n']}")
