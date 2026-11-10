import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s

    N = 51975

    # Generate all divisors of N greater than 1
    divisors = []
    for i in range(1, int(np.sqrt(N)) + 1):
        if N % i == 0:
            if i > 1:
                divisors.append(i)
            other = N // i
            if other != i and other > 1:
                divisors.append(other)
    divisors.sort()
    num_divs = len(divisors)

    def greedy_init(moduli_list, rng):
        """Greedy: for each modulus in order, pick residue covering most uncovered."""
        covered = np.zeros(N, dtype=np.uint8)
        classes = []
        for m in moduli_list:
            best_r = 0
            best_count = -1
            for r in range(m):
                cnt = int(np.sum(covered[r::m] == 0))
                if cnt > best_count:
                    best_count = cnt
                    best_r = r
            classes.append([m, best_r])
            covered[best_r::m] = 1
        uncovered = int(np.sum(covered == 0))
        return classes, uncovered

    def compute_uncovered(cover_count):
        return int(np.sum(cover_count == 0))

    def local_search(classes, time_limit, rng):
        """
        Fast local search with incremental score updates.
        For each class i with modulus m and residue r, trying new residue r':
          - Positions that lose coverage: those where cover_count becomes 0
          - Positions that gain coverage: those where cover_count was 0
        We track delta = new_uncovered - old_uncovered
        """
        classes = [c[:] for c in classes]
        
        cover_count = np.zeros(N, dtype=np.int32)
        for m, r in classes:
            cover_count[r::m] += 1
            
        uncovered_count = compute_uncovered(cover_count)
        
        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            # Randomize the order of classes to escape local optima
            order = list(range(len(classes)))
            rng.shuffle(order)
            for i in order:
                if (time.time() - start_time) > time_limit:
                    break
                m, r = classes[i]
                best_r = r
                best_score = uncovered_count
                
                for new_r in range(m):
                    if new_r == r:
                        continue
                    
                    # Compute delta efficiently
                    # Positions affected: those in r::m and new_r::m
                    # But we need to be careful about overlap
                    
                    # Simple but correct: temporarily apply, compute, revert
                    # For speed, use the fact that only positions in the union of the two slices change
                    # But for correctness and simplicity with numpy, do the full update
                    
                    cover_count[r::m] -= 1
                    cover_count[new_r::m] += 1
                    new_score = compute_uncovered(cover_count)
                    cover_count[new_r::m] -= 1
                    cover_count[r::m] += 1
                    
                    if new_score < best_score:
                        best_score = new_score
                        best_r = new_r

                if best_r != r:
                    # Apply the best change permanently
                    cover_count[r::m] -= 1
                    cover_count[best_r::m] += 1
                    classes[i] = [m, best_r]
                    uncovered_count = best_score
                    improved = True

        return classes, uncovered_count

    def fast_local_search(classes, time_limit, rng):
        """
        Optimized local search using precomputed slice indices.
        Key insight: for a given modulus m, the slices r::m for different r are disjoint
        in terms of which positions they cover within each residue class mod m.
        """
        classes = [c[:] for c in classes]
        
        cover_count = np.zeros(N, dtype=np.int32)
        for m, r in classes:
            cover_count[r::m] += 1
            
        uncovered_count = compute_uncovered(cover_count)
        
        # Precompute slice info for each modulus
        # For modulus m, slice r::m covers positions r, r+m, r+2m, ...
        
        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            order = list(range(len(classes)))
            rng.shuffle(order)
            for i in order:
                if (time.time() - start_time) > time_limit:
                    break
                m, r = classes[i]
                best_r = r
                best_score = uncovered_count
                
                # For this modulus, try all residues
                # Optimization: only need to check residues that could improve
                # But for now, check all with fast delta computation
                
                for new_r in range(m):
                    if new_r == r:
                        continue
                    
                    # The positions that change are:
                    # - Positions in r::m that were only covered by this class (lose coverage)
                    # - Positions in new_r::m that were not covered (gain coverage)
                    # But positions in both r::m and new_r::m (if any overlap) need care
                    
                    # Since r != new_r and both are mod m, the slices r::m and new_r::m
                    # are disjoint (no position x satisfies x ≡ r (mod m) and x ≡ new_r (mod m)
                    # when r ≠ new_r and 0 <= r, new_r < m)
                    
                    # So:
                    # lose: positions in r::m where cover_count == 1 (they become 0)
                    # gain: positions in new_r::m where cover_count == 0 (they become 1)
                    
                    # Count how many positions in r::m have cover_count == 1
                    old_slice = cover_count[r::m]
                    lose_count = int(np.sum(old_slice == 1))
                    
                    # Count how many positions in new_r::m have cover_count == 0
                    new_slice = cover_count[new_r::m]
                    gain_count = int(np.sum(new_slice == 0))
                    
                    delta = lose_count - gain_count
                    new_score = uncovered_count + delta
                    
                    if new_score < best_score:
                        best_score = new_score
                        best_r = new_r

                if best_r != r:
                    # Apply the best change permanently
                    cover_count[r::m] -= 1
                    cover_count[best_r::m] += 1
                    classes[i] = [m, best_r]
                    uncovered_count = best_score
                    improved = True

        return classes, uncovered_count

    # Strategy 1: Greedy with all divisors in increasing order
    rng1 = random.Random(42)
    best_classes, best_score = greedy_init(divisors, rng1)

    # Run fast local search
    if (time.time() - start_time) < deadline - 10:
        best_classes, best_score = fast_local_search(best_classes, deadline - 5, rng1)

    # Strategy 2: Try different orderings
    remaining = deadline - (time.time() - start_time)

    if remaining > 15:
        # Try greedy with shuffled order
        shuffled = divisors[:]
        random.Random(123).shuffle(shuffled)
        classes2, score2 = greedy_init(shuffled, random.Random(123))
        if score2 < best_score:
            best_classes, best_score = classes2, score2
            if (time.time() - start_time) < deadline - 5:
                best_classes, best_score = fast_local_search(best_classes, deadline - 3, random.Random(123))

    if (time.time() - start_time) < deadline - 15:
        # Try with larger moduli first
        divs_desc = sorted(divisors, reverse=True)
        classes3, score3 = greedy_init(divs_desc, random.Random(456))
        if score3 < best_score:
            best_classes, best_score = classes3, score3
            if (time.time() - start_time) < deadline - 5:
                best_classes, best_score = fast_local_search(best_classes, deadline - 3, random.Random(456))

    # Strategy 3: More diverse greedy initializations
    if (time.time() - start_time) < deadline - 20:
        # Try: sort by number of divisors (more constrained first)
        def divisor_count(n):
            cnt = 0
            for d in divisors:
                if n % d == 0:
                    cnt += 1
            return cnt
        
        divs_by_dc = sorted(divisors, key=lambda x: divisor_count(x))
        classes4, score4 = greedy_init(divs_by_dc, random.Random(789))
        if score4 < best_score:
            best_classes, best_score = classes4, score4
            if (time.time() - start_time) < deadline - 5:
                best_classes, best_score = fast_local_search(best_classes, deadline - 3, random.Random(789))

    # Strategy 4: Simulated Annealing
    if (time.time() - start_time) < deadline - 10:
        current_classes = [c[:] for c in best_classes]
        current_score = best_score
        
        cover_count = np.zeros(N, dtype=np.int32)
        for m, r in current_classes:
            cover_count[r::m] += 1
            
        temperature = max(1, best_score // 3)
        sa_rng = random.Random(999)
        
        max_iter = 2000
        for iteration in range(max_iter):
            if (time.time() - start_time) > deadline - 5:
                break

            # Perturb: change one residue
            idx = sa_rng.randint(0, len(current_classes) - 1)
            m, r = current_classes[idx]
            old_r = r
            new_r = sa_rng.randint(0, m - 1)
            
            if new_r == old_r:
                continue

            # Fast delta evaluation
            old_slice = cover_count[old_r::m]
            lose_count = int(np.sum(old_slice == 1))
            new_slice = cover_count[new_r::m]
            gain_count = int(np.sum(new_slice == 0))
            
            delta = lose_count - gain_count
            new_score = current_score + delta
            
            if delta < 0 or sa_rng.random() < np.exp(-delta / max(0.1, temperature)):
                # Accept
                cover_count[old_r::m] -= 1
                cover_count[new_r::m] += 1
                current_classes[idx] = [m, new_r]
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_classes = [c[:] for c in current_classes]
            else:
                # Reject (no change needed since we didn't modify cover_count)
                pass

            # Cool down
            temperature = max(0.1, temperature * 0.97)

    # Final polish: multiple rounds of local search
    if (time.time() - start_time) < deadline - 3:
        best_classes, best_score = fast_local_search(best_classes, deadline - 2, random.Random(1234))
    
    if (time.time() - start_time) < deadline - 1:
        best_classes, best_score = fast_local_search(best_classes, deadline - 0.5, random.Random(5678))

    return best_classes
