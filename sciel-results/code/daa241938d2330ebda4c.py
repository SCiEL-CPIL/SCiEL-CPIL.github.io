import time
import random
import itertools

def construct(q, n, radius):
    # Parameters for the specific task: q=3, n=8, radius=1
    # The evaluator passes this task's alphabet size q, length n and covering radius.
    # We tune the algorithm for q=3, n=8, radius=1 but add a fallback for other cases.
    
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin before the 120s limit
    
    total_words = q ** n
    
    # Helper functions
    def idx_to_tuple(idx):
        t = [0] * n
        for i in range(n):
            t[n - 1 - i] = idx % q
            idx //= q
        return tuple(t)
    
    def tuple_to_idx(t):
        idx = 0
        for i in range(n):
            idx = idx * q + t[i]
        return idx
    
    # Precompute balls (sets of indices within Hamming distance <= radius from each index)
    # For radius 1, a ball has 1 + n*(q-1) elements
    balls = []
    for i in range(total_words):
        t = idx_to_tuple(i)
        neighbors = [i]
        for pos in range(n):
            orig_val = t[pos]
            for new_val in range(q):
                if new_val != orig_val:
                    t[pos] = new_val
                    neighbors.append(tuple_to_idx(t))
            t[pos] = orig_val
        balls.append(set(neighbors))
    
    # For the specific case q=3, n=8, radius=1, use a smarter initialization
    # The direct sum of the tetracode gives 486 words.
    # We can do better with a greedy approach starting from a good seed.
    
    if q == 3 and n == 8 and radius == 1:
        # Tetracode-based initialization: 9 * 81 = 729 words
        # But we want to start from a smaller, good initial covering set.
        # Let's use a greedy construction to get a good initial covering.
        
        # Greedy set cover: iteratively pick the center that covers the most uncovered words
        uncovered = set(range(total_words))
        centers = set()
        coverage = [0] * total_words
        
        while uncovered:
            if time.time() > deadline:
                break
            
            best_c = -1
            best_count = -1
            
            # Random sampling of candidates to speed up
            # Try a random subset of uncovered words and their balls
            sample_size = min(len(uncovered), 200)
            sample = random.sample(list(uncovered), sample_size)
            
            scores = {}
            for w in sample:
                for c in balls[w]:
                    scores[c] = scores.get(c, 0) + 1
            
            if scores:
                for c, s in scores.items():
                    # Count how many currently uncovered words c covers
                    count = 0
                    for u in balls[c]:
                        if u in uncovered:
                            count += 1
                    if count > best_count:
                        best_count = count
                        best_c = c
            
            if best_c == -1:
                break
            
            centers.add(best_c)
            for w in balls[best_c]:
                coverage[w] += 1
                if w in uncovered:
                    uncovered.discard(w)
            
            if time.time() > deadline:
                break
        
        # Now do local search to improve
        best_size = len(centers)
        best_centers = set(centers)
        
        # Local search with kicks
        while time.time() < deadline:
            # Deletion phase: remove redundant centers
            removed_any = False
            current_centers_list = list(centers)
            random.shuffle(current_centers_list)
            
            for c in current_centers_list:
                if c not in centers:
                    continue
                redundant = True
                for w in balls[c]:
                    if coverage[w] <= 1:
                        redundant = False
                        break
                if redundant:
                    centers.remove(c)
                    for w in balls[c]:
                        coverage[w] -= 1
                    removed_any = True
            
            if removed_any:
                if len(centers) < best_size:
                    best_size = len(centers)
                    best_centers = set(centers)
                continue
            
            # Kick: remove a random center and repair
            if not centers:
                break
            
            kick_c = random.choice(list(centers))
            centers.remove(kick_c)
            for w in balls[kick_c]:
                coverage[w] -= 1
            
            # Repair: cover uncovered words
            uncovered = [w for w in range(total_words) if coverage[w] == 0]
            
            while uncovered and time.time() < deadline:
                # Greedy: pick center covering most uncovered
                scores = {}
                # Sample uncovered words for efficiency
                sample_size = min(len(uncovered), 300)
                sample = random.sample(uncovered, sample_size)
                for w in sample:
                    for c in balls[w]:
                        scores[c] = scores.get(c, 0) + 1
                
                if not scores:
                    break
                
                best_c = max(scores, key=scores.get)
                # Verify by counting actual uncovered coverage
                actual_count = sum(1 for u in balls[best_c] if coverage[u] == 0)
                
                # Pick best among top candidates
                top_candidates = sorted(scores.items(), key=lambda x: -x[1])[:10]
                best_actual = -1
                best_c_final = best_c
                for c, _ in top_candidates:
                    cnt = sum(1 for u in balls[c] if coverage[u] == 0)
                    if cnt > best_actual:
                        best_actual = cnt
                        best_c_final = c
                
                centers.add(best_c_final)
                for w in balls[best_c_final]:
                    coverage[w] += 1
                
                uncovered = [w for w in uncovered if coverage[w] > 0]
            
            if len(centers) < best_size:
                best_size = len(centers)
                best_centers = set(centers)
        
        # Convert to list of lists
        result = []
        for c in best_centers:
            t = idx_to_tuple(c)
            result.append(list(t))
        return result
    else:
        # Generic fallback: greedy set cover
        uncovered = set(range(total_words))
        centers = set()
        
        while uncovered and time.time() < deadline:
            best_c = -1
            best_count = -1
            sample_size = min(len(uncovered), 100)
            sample = random.sample(list(uncovered), sample_size)
            scores = {}
            for w in sample:
                for c in balls[w]:
                    scores[c] = scores.get(c, 0) + 1
            for c, s in scores.items():
                count = sum(1 for u in balls[c] if u in uncovered)
                if count > best_count:
                    best_count = count
                    best_c = c
            if best_c == -1:
                break
            centers.add(best_c)
            for w in balls[best_c]:
                if w in uncovered:
                    uncovered.discard(w)
        
        result = []
        for c in centers:
            t = idx_to_tuple(c)
            result.append(list(t))
        return result
