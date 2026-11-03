"""Constructs a large set of integer vectors in R^11 with min pairwise distance >= max norm."""
import numpy as np
import time
import random
import itertools
from collections import defaultdict

D = 11


def construct():
    start_time = time.time()
    time_limit = 110.0  # Leave 10s buffer for 120s limit

    def check_validity(vectors):
        """Check if all pairwise distances >= max norm."""
        if len(vectors) == 0:
            return True
        norms2 = [np.dot(v, v) for v in vectors]
        max_norm2 = max(norms2)
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                diff = vectors[i] - vectors[j]
                dist2 = np.dot(diff, diff)
                if dist2 < max_norm2:
                    return False
        return True

    def greedy_add(vectors, candidate, max_norm2):
        """Try to add candidate to vectors if it maintains the constraint."""
        for v in vectors:
            diff = candidate - v
            if np.dot(diff, diff) < max(max_norm2, np.dot(candidate, candidate)):
                return False
        return True

    # Start with all weight-2 vectors (±1, ±1 on two coordinates)
    vectors = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i], v[j] = si, sj
                vectors.append(v)

    # Convert to numpy array for efficiency
    vectors = np.array(vectors, dtype=np.int64)
    n = len(vectors)

    # Compute max norm squared
    norms2 = np.sum(vectors ** 2, axis=1)
    max_norm2 = int(np.max(norms2))

    # Try to add weight-3 and weight-4 vectors
    # Generate candidates
    candidates = []

    # Weight 3: three coordinates with ±1
    for i, j, k in itertools.combinations(range(D), 3):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i], v[j], v[k] = si, sj, sk
                    candidates.append(v)

    # Weight 4: four coordinates with ±1
    for i, j, k, l in itertools.combinations(range(D), 4):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    for sl in (1, -1):
                        v = np.zeros(D, dtype=np.int64)
                        v[i], v[j], v[k], v[l] = si, sj, sk, sl
                        candidates.append(v)

    random.shuffle(candidates)

    added = 0
    for cand in candidates:
        if time.time() - start_time > time_limit:
            break
        cand_norm2 = int(np.dot(cand, cand))
        new_max_norm2 = max(max_norm2, cand_norm2)
        # Check against all existing vectors
        valid = True
        for v in vectors:
            diff = cand - v
            if int(np.dot(diff, diff)) < new_max_norm2:
                valid = False
                break
        if valid:
            vectors = np.vstack([vectors, cand])
            max_norm2 = new_max_norm2
            added += 1

    return vectors
