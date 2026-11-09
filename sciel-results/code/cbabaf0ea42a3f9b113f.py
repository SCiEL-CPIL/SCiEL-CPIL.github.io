import time
import itertools
import numpy as np
from collections import defaultdict

_SEED = 20240612

def _rng(seed):
    return np.random.RandomState(seed)

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill

    rng = _rng(_SEED)

    # Precompute all triples and map to indices
    all_triples = list(itertools.combinations(range(v), t))
    n_all = len(all_triples)
    triple_to_idx = {tr: i for i, tr in enumerate(all_triples)}

    # Precompute direct pair->(third_point)->triple_idx mapping for O(1) lookup
    # pair_third_to_triple[(a,b,c)] = triple_idx where a<b, c is third point
    pair_third_to_triple = {}
    for tr_idx, tr in enumerate(all_triples):
        for i in range(t):
            for j in range(i+1, t):
                for p in tr:
                    if p != tr[i] and p != tr[j]:
                        c = p
                        break
                a, b = tr[i], tr[j]
                if a > b:
                    a, b = b, a
                pair_third_to_triple[(a, b, c)] = tr_idx

    # For each point, precompute which pairs it forms with other points
    # point_pairs[p] = list of (other_point, pair_key)
    point_pairs = defaultdict(list)
    for a in range(v):
        for b in range(a+1, v):
            point_pairs[a].append((b, (a, b)))
            point_pairs[b].append((a, (a, b)))

    # Count of how many times each triple is covered
    count = np.zeros(n_all, dtype=np.int32)
    blocks = []
    block_set = set()

    def add_block(block):
        if block in block_set:
            return False
        block_set.add(block)
        blocks.append(block)
        for tr in itertools.combinations(block, t):
            count[triple_to_idx[tr]] += 1
        return True

    def remove_block(block):
        for tr in itertools.combinations(block, t):
            count[triple_to_idx[tr]] -= 1
        block_set.discard(block)

    def get_uncovered_indices():
        return np.where(count == 0)[0]

    def build_block_greedy(start_triple, count_arr, rng_obj):
        """Build a block greedily starting from a given triple. Optimized version."""
        block = list(start_triple)
        remaining = set(range(v))
        for p in start_triple:
            remaining.discard(p)

        for _ in range(k - t):
            best_cand = None
            best_s = -1
            # Get pairs from current block (t-1=2 for t=3)
            pairs = list(itertools.combinations(block, t-1))
            
            # Iterate candidates
            for cand in remaining:
                s = 0
                for p1, p2 in pairs:
                    # Use precomputed mapping for O(1) lookup
                    a, b = (p1, p2) if p1 < p2 else (p2, p1)
                    tr_idx = pair_third_to_triple.get((a, b, cand))
                    if tr_idx is not None and count_arr[tr_idx] == 0:
                        s += 1
                if s > best_s:
                    best_s = s
                    best_cand = cand
            
            if best_cand is None:
                best_cand = next(iter(remaining))
            
            block.append(best_cand)
            remaining.discard(best_cand)
        
        return tuple(sorted(block))

    def compute_gain(block, count_arr):
        """Compute how many uncovered triples this block covers."""
        gain = 0
        for tr in itertools.combinations(block, t):
            if count_arr[triple_to_idx[tr]] == 0:
                gain += 1
        return gain

    def compute_overlap_score(block, count_arr):
        """Compute a weighted score: gain + small bonus for low-redundancy triples."""
        gain = 0
        redundancy_penalty = 0.0
        for tr in itertools.combinations(block, t):
            idx = triple_to_idx[tr]
            if count_arr[idx] == 0:
                gain += 1
            else:
                # Penalize adding blocks that cover already-covered triples
                redundancy_penalty += 0.1 * count_arr[idx]
        return gain - redundancy_penalty

    def build_block_probabilistic(start_triple, count_arr, rng_obj, alpha=0.3):
        """Build a block using probabilistic selection inspired by Johnson scheme.
        
        Uses a soft-max over candidate scores with temperature controlled by alpha.
        This allows occasional non-greedy choices that may lead to better global structure.
        """
        block = list(start_triple)
        remaining = set(range(v))
        for p in start_triple:
            remaining.discard(p)

        for _ in range(k - t):
            if not remaining:
                break
            
            pairs = list(itertools.combinations(block, t-1))
            
            # Compute scores for all candidates
            cand_scores = {}
            for cand in remaining:
                s = 0
                for p1, p2 in pairs:
                    a, b = (p1, p2) if p1 < p2 else (p2, p1)
                    tr_idx = pair_third_to_triple.get((a, b, cand))
                    if tr_idx is not None and count_arr[tr_idx] == 0:
                        s += 1
                cand_scores[cand] = s
            
            if not cand_scores:
                break
            
            # Use softmax with temperature for probabilistic selection
            scores = np.array(list(cand_scores.values()), dtype=np.float64)
            candidates = list(cand_scores.keys())
            
            if alpha > 0 and len(scores) > 1:
                # Softmax: exp(score / alpha) / sum(exp(score / alpha))
                max_score = np.max(scores)
                exp_scores = np.exp((scores - max_score) / alpha)
                probs = exp_scores / np.sum(exp_scores)
                best_cand = candidates[rng_obj.choice(len(candidates), p=probs)]
            else:
                # Greedy: pick highest score
                best_idx = np.argmax(scores)
                best_cand = candidates[best_idx]
            
            block.append(best_cand)
            remaining.discard(best_cand)
        
        return tuple(sorted(block))

    def find_replacement_block(critical_triples, count_arr, rng_obj):
        """Find a replacement block that covers all critical triples.
        
        Uses a targeted local search that optimizes overlap with critical triples
        rather than simple gain.
        """
        if not critical_triples:
            return None
        
        best_block = None
        best_score = -1e9
        
        # Strategy 1: Try building from each critical triple with different alpha values
        alphas = [0.0, 0.1, 0.2, 0.3, 0.5]
        for start_tr in critical_triples[:10]:  # Limit to 10 for speed
            for alpha in alphas:
                block = build_block_probabilistic(start_tr, count_arr, rng_obj, alpha)
                
                # Check if this block covers all critical_triples
                block_triples = set(itertools.combinations(block, t))
                covered_count = sum(1 for tr in critical_triples if tr in block_triples)
                
                if covered_count == len(critical_triples):
                    # Score: maximize gain, minimize redundancy
                    gain = compute_gain(block, count_arr)
                    score = gain - 0.05 * len(block)  # Slight penalty for large blocks
                    if score > best_score:
                        best_score = score
                        best_block = block
        
        # Strategy 2: Find points that appear in multiple critical triples
        point_freq = defaultdict(int)
        for tr in critical_triples:
            for p in tr:
                point_freq[p] += 1
        
        # Sort points by frequency
        sorted_points = sorted(point_freq.keys(), key=lambda p: -point_freq[p])
        
        # Try building from top frequent points
        for p in sorted_points[:5]:
            # Find critical triples containing p
            relevant_triples = [tr for tr in critical_triples if p in tr]
            if not relevant_triples:
                continue
            
            # Start with a triple containing p
            start_tr = relevant_triples[0]
            for alpha in [0.0, 0.2, 0.4]:
                block = build_block_probabilistic(start_tr, count_arr, rng_obj, alpha)
                
                block_triples = set(itertools.combinations(block, t))
                covered_count = sum(1 for tr in critical_triples if tr in block_triples)
                
                if covered_count == len(critical_triples):
                    gain = compute_gain(block, count_arr)
                    score = gain - 0.05 * len(block)
                    if score > best_score:
                        best_score = score
                        best_block = block
        
        return best_block

    # Phase 1: Probabilistic greedy construction
    for _ in range(300):
        if time.time() > deadline - 10:
            break
        
        uncovered_indices = get_uncovered_indices()
        if len(uncovered_indices) == 0:
            break

        # Sample uncovered triples to find best starting point
        sample_size = min(30, len(uncovered_indices))
        sample_idx = rng.choice(uncovered_indices, size=sample_size, replace=False)

        best_block = None
        best_score = -1e9

        for idx in sample_idx:
            tr = all_triples[idx]
            # Use probabilistic construction with varying alpha
            for alpha in [0.0, 0.2, 0.4]:
                block = build_block_probabilistic(tr, count, rng, alpha)
                score = compute_overlap_score(block, count)
                if score > best_score:
                    best_score = score
                    best_block = block

        if best_block is None:
            break

        add_block(best_block)

    # Phase 2: Local improvement - try removing or replacing blocks
    for _ in range(150):
        if time.time() > deadline - 5:
            break
        improved = False
        for bi in range(len(blocks)):
            if time.time() > deadline - 5:
                break
            
            removed_block = blocks[bi]
            # Identify which triples will become uncovered if we remove this block
            critical_triples = []
            for tr in itertools.combinations(removed_block, t):
                if count[triple_to_idx[tr]] == 1:
                    critical_triples.append(tr)
            
            if not critical_triples:
                # Can remove this block safely
                remove_block(removed_block)
                blocks.pop(bi)
                improved = True
                break
            
            # Try to find a replacement block that covers all critical triples
            best_repl = find_replacement_block(critical_triples, count, rng)
            
            if best_repl is not None:
                remove_block(removed_block)
                blocks[bi] = best_repl
                add_block(best_repl)
                improved = True
                break

        if not improved:
            break

    # Phase 3: Ensure full coverage
    while time.time() < deadline - 2:
        uncovered_indices = get_uncovered_indices()
        if len(uncovered_indices) == 0:
            break
        tr = all_triples[uncovered_indices[0]]
        block = build_block_probabilistic(tr, count, rng, 0.0)  # Greedy for final coverage
        add_block(block)

    # Phase 4: Final polish - try to remove redundant blocks
    for _ in range(50):
        if time.time() > deadline - 1:
            break
        improved = False
        for bi in range(len(blocks)):
            if time.time() > deadline - 1:
                break
            
            removed_block = blocks[bi]
            can_remove = True
            for tr in itertools.combinations(removed_block, t):
                if count[triple_to_idx[tr]] == 1:
                    can_remove = False
                    break
            
            if can_remove:
                remove_block(removed_block)
                blocks.pop(bi)
                improved = True
                break
        
        if not improved:
            break

    # Phase 5: 2-for-1 replacement - try replacing two blocks with one better block
    for _ in range(80):
        if time.time() > deadline - 1:
            break
        improved = False
        # Sample pairs of blocks to try merging
        if len(blocks) < 2:
            break
        n_blocks = len(blocks)
        for _attempt in range(20):
            if time.time() > deadline - 1:
                break
            i, j = rng.choice(n_blocks, size=2, replace=False)
            if i > j:
                i, j = j, i
            
            block_i = blocks[i]
            block_j = blocks[j]
            
            # Find critical triples: those covered only by block_i or only by block_j
            # (i.e., count==1 and in that block)
            critical_triples = []
            for tr in itertools.combinations(block_i, t):
                if count[triple_to_idx[tr]] == 1:
                    critical_triples.append(tr)
            for tr in itertools.combinations(block_j, t):
                if count[triple_to_idx[tr]] == 1:
                    critical_triples.append(tr)
            
            if not critical_triples:
                # Both blocks can be removed safely
                remove_block(block_i)
                remove_block(block_j)
                blocks.pop(j)
                blocks.pop(i)
                improved = True
                break
            
            # Try to build a single block that covers all critical triples
            # Use the most frequent point in critical triples as anchor
            point_freq = defaultdict(int)
            for tr in critical_triples:
                for p in tr:
                    point_freq[p] += 1
            sorted_points = sorted(point_freq.keys(), key=lambda p: -point_freq[p])
            
            best_block = None
            best_gain = -1
            
            for p in sorted_points[:3]:
                # Find a critical triple containing p
                relevant = [tr for tr in critical_triples if p in tr]
                if not relevant:
                    continue
                start_tr = relevant[0]
                for alpha in [0.0, 0.2, 0.4]:
                    block = build_block_probabilistic(start_tr, count, rng, alpha)
                    # Check coverage of all critical triples
                    block_triple_set = set(itertools.combinations(block, t))
                    if all(tr in block_triple_set for tr in critical_triples):
                        gain = compute_gain(block, count)
                        if gain > best_gain:
                            best_gain = gain
                            best_block = block
            
            if best_block is not None:
                remove_block(block_i)
                remove_block(block_j)
                blocks.pop(j)
                blocks.pop(i)
                add_block(best_block)
                improved = True
                break
        
        if not improved:
            break

    # Final deduplication (should be empty due to block_set, but safety check)
    seen = set()
    unique_blocks = []
    for b in blocks:
        if b not in seen:
            seen.add(b)
            unique_blocks.append(b)

    return [list(b) for b in unique_blocks]
