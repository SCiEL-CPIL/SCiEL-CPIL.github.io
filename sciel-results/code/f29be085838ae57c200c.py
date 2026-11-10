import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 80  # safety margin before 90s kill
    
    def square_nonzero_count(coeffs):
        """Return number of nonzero coefficients in the exact square of coeffs."""
        n = len(coeffs)
        sq = [0] * (2 * n - 1)
        for i in range(n):
            if coeffs[i] == 0:
                continue
            for j in range(i, n):
                val = coeffs[i] * coeffs[j]
                if i == j:
                    sq[2 * i] += val
                else:
                    sq[2 * i] += val
                    sq[2 * j] += val
        return sum(1 for v in sq if v != 0)
    
    def is_valid(coeffs):
        """Check all constraints."""
        if len(coeffs) > 41:
            return False
        if len(coeffs) == 0 or coeffs[-1] == 0:
            return False
        for c in coeffs:
            if not isinstance(c, int) or c < -64 or c > 64:
                return False
        if sum(1 for c in coeffs if c != 0) != 12:
            return False
        return True
    
    def make_state(n_terms, degree, rng):
        """Create a random state with n_terms nonzero coefficients."""
        positions = sorted(rng.sample(range(degree + 1), n_terms))
        coeffs = [0] * (degree + 1)
        for p in positions:
            coeffs[p] = rng.choice([-1, 1]) * rng.randint(1, 64)
        # Ensure last position is nonzero
        if coeffs[-1] == 0:
            coeffs[-1] = rng.choice([-1, 1]) * rng.randint(1, 64)
        return coeffs
    
    def local_search(coeffs, rng, max_iters=5000, deadline=None):
        """Hill-climbing with random restarts within a fixed support."""
        n = len(coeffs)
        best = square_nonzero_count(coeffs)
        current = list(coeffs)
        
        for it in range(max_iters):
            if deadline and time.time() > deadline:
                break
            # Pick a random nonzero position to modify
            nonzero_pos = [i for i in range(n) if current[i] != 0]
            if not nonzero_pos:
                break
            idx = rng.choice(nonzero_pos)
            
            # Try a random new value
            old_val = current[idx]
            new_val = rng.choice([-1, 1]) * rng.randint(1, 64)
            while new_val == old_val:
                new_val = rng.choice([-1, 1]) * rng.randint(1, 64)
            
            current[idx] = new_val
            new_count = square_nonzero_count(current)
            
            if new_count < best:
                best = new_count
            else:
                current[idx] = old_val
        
        return current, best
    
    def try_support(positions, degree, rng, deadline):
        """For a fixed set of support positions, search for best coefficients."""
        n = len(positions)
        coeffs = [0] * (degree + 1)
        for p in positions:
            coeffs[p] = rng.choice([-1, 1]) * rng.randint(1, 64)
        if coeffs[-1] == 0:
            coeffs[-1] = rng.choice([-1, 1]) * rng.randint(1, 64)
        
        # Local search
        coeffs, count = local_search(coeffs, rng, max_iters=2000, deadline=deadline)
        return coeffs, count
    
    best_coeffs = None
    best_count = float('inf')
    
    rng = random.Random(42)
    
    # Strategy 1: Try various support patterns
    # Known good pattern: symmetric supports tend to create cancellation
    # For 12 terms, try degree up to 40
    
    # First, try a structured approach: pairs that cancel
    # If we have terms at positions that form arithmetic progressions,
    # the square has structure
    
    # Try random supports with local search
    for trial in range(200):
        if time.time() > deadline:
            break
        
        # Random degree between 15 and 40
        degree = rng.randint(15, 40)
        
        # Generate 12 random positions, ensuring last position is included
        positions = set()
        positions.add(degree)  # Must have nonzero final coefficient
        while len(positions) < 12:
            positions.add(rng.randint(0, degree))
        positions = sorted(positions)
        
        coeffs, count = try_support(positions, degree, rng, deadline)
        
        if count < best_count:
            best_count = count
            best_coeffs = coeffs
    
    # Strategy 2: Try structured supports (arithmetic progressions, symmetric)
    for trial in range(100):
        if time.time() > deadline:
            break
        
        degree = rng.randint(20, 40)
        
        # Symmetric pattern around degree/2
        center = degree // 2
        positions = set()
        positions.add(degree)
        
        # Add symmetric pairs
        while len(positions) < 12:
            offset = rng.randint(1, center)
            if center - offset >= 0:
                positions.add(center - offset)
            if center + offset <= degree:
                positions.add(center + offset)
        
        if len(positions) < 12:
            while len(positions) < 12:
                positions.add(rng.randint(0, degree))
        
        positions = sorted(positions)
        
        coeffs, count = try_support(positions, degree, rng, deadline)
        
        if count < best_count:
            best_count = count
            best_coeffs = coeffs
    
    # Strategy 3: Try to find patterns with high cancellation
    # Use pairs: if we have a_i * a_j and a_k * a_l with i+j = k+l,
    # we can cancel if a_i*a_j = -a_k*a_l
    
    for trial in range(100):
        if time.time() > deadline:
            break
        
        degree = rng.randint(20, 40)
        
        # Create support with deliberate pairing for cancellation
        positions = [degree]  # Must include last
        remaining = 11
        
        # Add pairs that can create cancellation
        used_positions = set(positions)
        while remaining > 0 and len(used_positions) < 12:
            # Pick two positions that sum to some target
            target = rng.randint(0, 2 * degree)
            for i in range(0, min(degree, target) + 1):
                j = target - i
                if 0 <= i <= degree and 0 <= j <= degree and i != j:
                    if i not in used_positions and j not in used_positions:
                        positions.append(i)
                        positions.append(j)
                        used_positions.add(i)
                        used_positions.add(j)
                        remaining -= 2
                        break
        
        # Fill remaining
        while len(positions) < 12:
            p = rng.randint(0, degree)
            if p not in used_positions:
                positions.append(p)
                used_positions.add(p)
        
        positions = sorted(set(positions))
        if len(positions) > 12:
            positions = positions[:12]
        if degree not in positions:
            positions = sorted(set(positions + [degree]))
            positions = positions[:12]
        
        coeffs, count = try_support(positions, degree, rng, deadline)
        
        if count < best_count:
            best_count = count
            best_coeffs = coeffs
    
    # Final: try a few more targeted searches with lower degree
    for degree in [12, 14, 16, 18, 20, 22, 24]:
        if time.time() > deadline:
            break
        
        for trial in range(30):
            if time.time() > deadline:
                break
            
            positions = set()
            positions.add(degree)
            while len(positions) < 12:
                positions.add(rng.randint(0, degree))
            positions = sorted(positions)
            
            coeffs, count = try_support(positions, degree, rng, deadline)
            
            if count < best_count:
                best_count = count
                best_coeffs = coeffs
    
    if best_coeffs is None or not is_valid(best_coeffs):
        # Fallback: return the seed
        return [1] * 12
    
    return best_coeffs
