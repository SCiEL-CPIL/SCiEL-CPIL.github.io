import time
import random
import math

def construct():
    """
    Attempts to find a 4-coloring of K_n with no monochromatic triangles for the largest possible n.
    Uses Simulated Annealing.
    """
    deadline = time.time() + 140.0
    
    # Fallback: The known 32-vertex solution from the seed
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r

    log, x = {}, 1
    for e in range(15):
        log[x] = e
        x = mul(x, 2)
    
    def get_seed_colors(n):
        # This generates the 32-vertex solution. 
        # If n > 32, we can't just use it directly without extension logic, 
        # but we can use it to initialize the first 32 vertices in a larger search?
        # For now, we will just return the 32-vertex solution if n==32.
        if n != 32:
            return None
        colors = []
        for i in range(32):
            for j in range(i + 1, 32):
                colors.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
        return colors

    def solve_n(n, time_limit):
        """
        Try to find a valid 4-coloring of K_n.
        Returns (n, colors) if found, else None.
        """
        if n < 2:
            return None
        
        # Initialize random coloring
        # colors is a list of lists, colors[i][j] for i < j
        # We use a 2D list for O(1) access.
        # colors[i][j] = c for i < j. colors[j][i] is not stored, we access via min/max or symmetric fill.
        # Symmetric fill is easier: colors[i][j] = colors[j][i].
        colors = [[0] * n for _ in range(n)]
        
        # Initialize randomly
        for i in range(n):
            for j in range(i + 1, n):
                c = random.randint(0, 3)
                colors[i][j] = c
                colors[j][i] = c
        
        # Count initial monochromatic triangles
        # A triangle (i, j, k) is monochromatic if colors[i][j] == colors[j][k] == colors[k][i]
        def count_mono_triangles():
            count = 0
            for i in range(n):
                for j in range(i + 1, n):
                    c_ij = colors[i][j]
                    for k in range(j + 1, n):
                        if c_ij == colors[j][k] and c_ij == colors[k][i]:
                            count += 1
            return count

        current_energy = count_mono_triangles()
        
        if current_energy == 0:
            # Flatten and return
            flat_colors = []
            for i in range(n):
                for j in range(i + 1, n):
                    flat_colors.append(colors[i][j])
            return (n, flat_colors)

        # Simulated Annealing
        # T starts high, cools down
        T_start = 1.0
        T_end = 0.01
        # Number of iterations to do within time_limit
        # We estimate iterations per second based on n
        # For n=50, one iteration is ~50 ops. ~10^6 iters/s.
        
        start_time = time.time()
        best_energy = current_energy
        best_colors = [row[:] for row in colors]
        
        # We will iterate until time runs out
        # To handle the "deadline" globally, we check time frequently.
        
        # Optimization: Pre-calculate neighbors? 
        # For each edge (i,j), the set of k such that (i,j,k) is a triangle.
        # This is just all k != i,j.
        
        # We use a simple loop.
        # To speed up, we can use local variables.
        
        # If we don't find 0, we return None.
        
        # Heuristic: If n is small, we can be more exhaustive.
        # If n is large, we rely on SA.
        
        while time.time() < start_time + time_limit:
            # Pick a random edge
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            
            old_c = colors[i][j]
            if old_c == 0:
                new_c = random.randint(1, 3)
            elif old_c == 1:
                new_c = random.choice([0, 2, 3])
            elif old_c == 2:
                new_c = random.choice([0, 1, 3])
            else:
                new_c = random.choice([0, 1, 2])
                
            # Calculate Delta E
            delta_e = 0
            for k in range(n):
                if k == i or k == j:
                    continue
                c_ik = colors[i][k]
                c_jk = colors[j][k]
                
                # Triangle (i, j, k)
                # Before: old_c, c_ik, c_jk
                # After: new_c, c_ik, c_jk
                
                # Check if it was mono
                was_mono = (old_c == c_ik and c_ik == c_jk)
                # Check if it is mono now
                is_mono = (new_c == c_ik and c_ik == c_jk)
                
                if was_mono and not is_mono:
                    delta_e -= 1
                elif not was_mono and is_mono:
                    delta_e += 1
            
            # Acceptance
            if delta_e <= 0:
                # Always accept
                colors[i][j] = new_c
                colors[j][i] = new_c
                current_energy += delta_e
            else:
                # Accept with probability exp(-delta_e / T)
                # We need to track T. 
                # Let's use a cooling schedule based on time or iterations.
                # Simpler: Just use a fixed T that decays?
                # Let's use iterations.
                # But we are in a time loop.
                # Let's just use a random T that decreases?
                # Actually, for this problem, "Hill Climbing" (greedy) often works if we do many random restarts.
                # SA is better.
                
                # Let's estimate progress
                # We don't have a total iteration count easily.
                # Let's just use a constant T that is small, so we mostly accept good moves.
                # Or, we can just do Greedy (Hill Climbing) and if stuck, restart.
                
                # Let's try a hybrid: 
                # If we are stuck (energy hasn't changed in X seconds), restart.
                
                # For now, let's just accept if random() < exp(-delta_e / 0.5)
                # This is a fixed temperature.
                if random.random() < math.exp(-delta_e / 0.5):
                    colors[i][j] = new_c
                    colors[j][i] = new_c
                    current_energy += delta_e
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_colors = [row[:] for row in colors]
                
            if best_energy == 0:
                flat_colors = []
                for i in range(n):
                    for j in range(i + 1, n):
                        flat_colors.append(best_colors[i][j])
                return (n, flat_colors)
                
        return None

    # Strategy:
    # 1. Try n=50 with a short time budget (e.g., 10s).
    # 2. If fail, try n=49, etc.
    # 3. If all fail, return n=32.
    
    # To maximize success, we should allocate time based on likelihood.
    # n=50 is the target.
    # If we can't find it in 10s, we probably won't in 100s with this simple SA.
    # But let's give it 20s.
    
    # Actually, a known 50-vertex coloring exists.
    # If we can't find it via search, we should hardcode it?
    # The prompt says "Write a new design from scratch".
    # Hardcoding is "from scratch" if we derive it or find it?
    # No, "algorithmic" is preferred.
    
    # Let's try to find it.
    # We will try n from 50 down to 33.
    # Time budget: 10s per n.
    # Total 18 * 10s = 180s. Too much.
    # We have 140s.
    # So 140 / 18 ~ 7s per n.
    
    # But n=50 is the most important.
    # Let's give n=50 30s.
    # n=49 20s.
    # n=48 10s.
    # ...
    
    # Actually, if we find a valid n=40, that's a score of 40/32 = 1.25.
    # n=50 is 1.56.
    
    # Let's run the search.
    
    best_result = None
    remaining_time = 140.0
    
    # Prioritize higher n
    for n in range(50, 32, -1):
        # Allocate time
        if n == 50:
            t_limit = 30.0
        elif n == 49:
            t_limit = 20.0
        elif n == 48:
            t_limit = 10.0
        else:
            t_limit = 5.0
            
        if remaining_time < t_limit:
            t_limit = remaining_time * 0.5
            if t_limit < 1.0:
                break
                
        result = solve_n(n, t_limit)
        remaining_time = 140.0 - (time.time() - (time.time() - 140.0)) # This logic is flawed, let's use absolute deadline
        
    # Correct time management:
    start_total = time.time()
    deadline_total = start_total + 140.0
    
    best_n = 32
    best_colors = get_seed_colors(32)
    
    for n in range(50, 32, -1):
        if time.time() > deadline_total - 5.0:
            break
            
        # Dynamic time allocation
        # If we haven't found anything better, give more time to higher n?
        # Let's just give a fixed chunk.
        # 140s total.
        # Try 50 for 20s.
        # Try 49 for 10s.
        # Try 48 for 5s.
        # Try 47 for 2s.
        # ...
        
        t_limit = 20.0 if n == 50 else (10.0 if n == 49 else (5.0 if n == 48 else 2.0))
        
        if time.time() + t_limit > deadline_total:
            t_limit = deadline_total - time.time()
            if t_limit < 1.0:
                break
                
        result = solve_n(n, t_limit)
        
        if result:
            best_n = result[0]
            best_colors = result[1]
            # If we found 50, we are done.
            if n == 50:
                break
            # If we found 49, we can try 50 again with remaining time?
            # No, 50 is harder.
            # We continue to try smaller n? No, we want max n.
            # So if we found 49, we don't need to try 48.
            # But we might find 50 later? Unlikely if we failed before.
            # So we break if we found a solution?
            # Wait, the loop goes 50, 49, 48...
            # If we fail 50, we try 49. If we succeed 49, we have a valid 49.
            # We should stop because 48 is worse.
            break
            
    if best_colors is None:
        best_colors = get_seed_colors(32)
        best_n = 32
        
    return {'n': best_n, 'colors': best_colors}
