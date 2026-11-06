import time
import random
import sys
from math import log as _log

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin (evaluator kills at 150s)
    
    # --- 1. Generate initial K32 coloring (Clebsch graph / GF(16)) ---
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r

    def get_initial_32():
        log, x = {}, 1
        for e in range(15):
            log[x] = e
            x = mul(x, 2)
        colors = []
        for i in range(32):
            for j in range(i + 1, 32):
                if i // 16 == j // 16:
                    colors.append(log[(i % 16) ^ (j % 16)] % 3)
                else:
                    colors.append(3)
        return colors

    current_n = 32
    current_colors = get_initial_32()
    best_n = 32
    best_colors = current_colors[:]
    
    # --- 2. Helper Functions ---
    
    def edge_idx(i, j, n):
        if i > j:
            i, j = j, i
        return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)

    def decode_edge(e, n):
        # i is the row, j is the column
        # Sum of lengths of rows 0..i-1 is i*(n-1) - i*(i-1)/2
        # We need i such that prefix[i] <= e < prefix[i+1]
        # prefix[i] = i * (n - 1 - i / 2) ... approx
        # Let's use binary search or direct formula
        # i*(n-1) - i*(i-1)/2 <= e
        # 2*i*(n-1) - i^2 + i <= 2e
        # -i^2 + (2n-1)i - 2e <= 0
        # i^2 - (2n-1)i + 2e >= 0
        # Roots: ((2n-1) - sqrt((2n-1)^2 - 8e)) / 2
        # Since we want the smaller root (start of the block), we take the minus sign?
        # Actually, the quadratic is upward opening. We want i such that the value is >= 0.
        # The roots are i1, i2. The interval [i1, i2] is where it's >= 0? No.
        # Let's just iterate or binary search. N is small (<=62), so linear search is fine but slow.
        # Binary search is better.
        lo, hi = 0, n - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            # prefix[mid] = mid * (n - 1) - mid * (mid - 1) // 2
            if mid * (n - 1) - mid * (mid - 1) // 2 <= e:
                lo = mid
            else:
                hi = mid - 1
        i = lo
        # j - i - 1 = e - prefix[i]
        # j = i + 1 + e - prefix[i]
        prefix_i = i * (n - 1) - i * (i - 1) // 2
        j = i + 1 + (e - prefix_i)
        return i, j

    def compute_threats(colors, n):
        threats = [0] * (n * (n - 1) // 2)
        # For each triangle (i, j, k), if all 3 edges have same color, increment threat for each edge
        # This is O(N^3). For N=50, 125k ops. Very fast.
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    c_ij = colors[edge_idx(i, j, n)]
                    c_ik = colors[edge_idx(i, k, n)]
                    c_jk = colors[edge_idx(j, k, n)]
                    if c_ij == c_ik == c_jk:
                        threats[edge_idx(i, j, n)] += 1
                        threats[edge_idx(i, k, n)] += 1
                        threats[edge_idx(j, k, n)] += 1
        return threats

    def get_delta(i, j, c_new, colors, n):
        """
        Calculate change in total threats if we change color of (i, j) to c_new.
        Total Threats = sum(threats[e]).
        A monochromatic triangle contributes 3 to the sum (1 for each edge).
        So delta = 3 * (new_mono_tris - old_mono_tris)
        """
        idx = edge_idx(i, j, n)
        c_old = colors[idx]
        if c_old == c_new:
            return 0
        
        old_mono = 0
        new_mono = 0
        for k in range(n):
            if k == i or k == j:
                continue
            ik_idx = edge_idx(i, k, n)
            jk_idx = edge_idx(j, k, n)
            c_ik = colors[ik_idx]
            c_jk = colors[jk_idx]
            
            if c_ik == c_old and c_jk == c_old:
                old_mono += 1
            if c_ik == c_new and c_jk == c_new:
                new_mono += 1
        
        return 3 * (new_mono - old_mono)

    # --- 3. Extension Strategy ---
    
    def try_extend(n, colors, threats, max_time=10.0):
        """
        Try to extend K_n to K_{n+1}.
        Returns (success, new_colors, new_threats)
        """
        new_n = n + 1
        new_colors = [0] * (new_n * (new_n - 1) // 2)
        
        # Copy old edges
        for i in range(n):
            for j in range(i + 1, n):
                old_idx = edge_idx(i, j, n)
                new_idx = edge_idx(i, j, new_n)
                new_colors[new_idx] = colors[old_idx]
        
        # Initialize new edges (k, n) for k < n randomly
        for k in range(n):
            new_idx = edge_idx(k, n, new_n)
            new_colors[new_idx] = random.randint(0, 3)
        
        new_threats = compute_threats(new_colors, new_n)
        current_energy = sum(new_threats)
        
        if current_energy == 0:
            return True, new_colors, new_threats
        
        # Simulated Annealing
        # We focus on minimizing total threats.
        # Time budget for this extension attempt.
        t_start = time.time()
        t_end = t_start + max_time
        
        T = 5.0
        T_min = 0.01
        cooling = 0.995
        
        # Precompute edge list for new_n? No, decode on fly or cache.
        # Since N is small, decoding is cheap enough if we do it carefully.
        # But we can also just iterate all edges in SA moves.
        
        # To speed up SA, we can restrict moves to edges involving the new vertex initially,
        # then allow all edges.
        # Let's try a mixed strategy: 80% new vertex edges, 20% old edges.
        new_edges = [edge_idx(k, n, new_n) for k in range(n)]
        
        while time.time() < t_end:
            if current_energy == 0:
                return True, new_colors, new_threats
            
            # Choose edge
            if random.random() < 0.7:
                e = random.choice(new_edges)
            else:
                e = random.randint(0, len(new_colors) - 1)
            
            i, j = decode_edge(e, new_n)
            c_old = new_colors[e]
            c_new = random.randint(0, 3)
            if c_old == c_new:
                continue
            
            delta = get_delta(i, j, c_new, new_colors, new_n)
            
            if delta < 0 or random.random() < pow(2.718, -delta / T):
                new_colors[e] = c_new
                current_energy += delta
                # Periodically recompute threats to ensure correctness
                # This is expensive, so we do it rarely or when energy is low?
                # Actually, get_delta is exact if colors are consistent.
                # We don't need to update the `new_threats` array for the SA logic,
                # only the `current_energy` sum is needed for acceptance.
                # But at the end, we need to verify.
                # However, if we want to use `new_threats` for something else, we'd need to update.
                # For now, we just track energy.
            
            T *= cooling
            if T < T_min:
                T = T_min
        
        # Final verification
        final_threats = compute_threats(new_colors, new_n)
        if sum(final_threats) == 0:
            return True, new_colors, final_threats
        
        return False, new_colors, final_threats

    # --- 4. Main Loop ---
    
    # We will try to extend from 32 to 62.
    # If we get stuck, we can try to "perturb" the current valid coloring to find a different one
    # that might be easier to extend.
    
    while current_n < 62 and time.time() < deadline - 10:
        # Try extension
        success, new_colors, new_threats = try_extend(current_n, current_colors, current_threats, max_time=5.0)
        
        if success:
            current_n += 1
            current_colors = new_colors
            current_threats = new_threats
            if current_n > best_n:
                best_n = current_n
                best_colors = current_colors[:]
        else:
            # Failed to extend.
            # Strategy: Perturb the current valid coloring to escape local minima in the extension landscape.
            # We run a short SA on the current K_n to find a different valid coloring (threats=0).
            # This might change the structure enough to allow extension.
            
            # We need to ensure we stay valid (threats=0) during perturbation?
            # Or allow temporary invalidity and come back?
            # Let's do a "valid walk": only accept moves that keep threats=0.
            # This is a Metropolis chain on the space of valid colorings.
            
            t_start = time.time()
            t_end = t_start + 5.0
            perturb_colors = current_colors[:]
            perturb_threats = current_threats[:] # Should be all 0
            
            # We need to be able to check if a move keeps threats=0.
            # This requires checking delta.
            # If delta == 0, the move is valid.
            
            while time.time() < t_end:
                # Pick a random edge
                e = random.randint(0, len(perturb_colors) - 1)
                i, j = decode_edge(e, current_n)
                c_old = perturb_colors[e]
                c_new = random.randint(0, 3)
                if c_old == c_new:
                    continue
                
                delta = get_delta(i, j, c_new, perturb_colors, current_n)
                
                if delta == 0:
                    # Valid move
                    perturb_colors[e] = c_new
                    # We don't strictly need to update perturb_threats if we only care about validity
                    # and we re-verify at the end. But for speed, we assume delta==0 means
                    # the total threats remain 0.
                    
            # After perturbation, check if we are still valid
            final_perturb_threats = compute_threats(perturb_colors, current_n)
            if sum(final_perturb_threats) == 0:
                # Try extending from this perturbed state
                success, new_colors, new_threats = try_extend(current_n, perturb_colors, final_perturb_threats, max_time=5.0)
                if success:
                    current_n += 1
                    current_colors = new_colors
                    current_threats = new_threats
                    if current_n > best_n:
                        best_n = current_n
                        best_colors = current_colors[:]
                else:
                    # Still stuck.
                    # If we've been stuck for a long time, break.
                    # For now, just break to avoid infinite loop of failed perturbations.
                    # A more advanced strategy would be to restart from scratch or use a different seed.
                    break
            else:
                # Perturbation broke validity. This shouldn't happen if delta==0 logic is correct.
                # If it does, revert.
                break

    return {'n': best_n, 'colors': best_colors}

# Test
# if __name__ == "__main__":
#     res = construct()
#     print(res['n'])
