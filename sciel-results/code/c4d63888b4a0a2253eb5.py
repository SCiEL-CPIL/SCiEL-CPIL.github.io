"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 90s limit
    
    # 1. Generate 14 distinct positive integers <= 10000 forming a Sidon set (B_2 sequence).
    # A Sidon set ensures that all pairwise sums a_i + a_j (i < j) are distinct.
    # This guarantees that the number of distinct sums for any set of edges is exactly the number of edges (35).
    # We use a known construction or generate one to be safe.
    # Simple Sidon set generation:
    # Start with small numbers and ensure no sum collisions.
    
    def generate_sidon_set(n, max_val):
        # Greedy construction of a Sidon set
        S = []
        used_sums = set()
        # Start with 1, 2, 4... powers of 2 are a Sidon set, but might grow too fast?
        # 2^13 = 8192. 2^0..2^13 are 14 numbers. Max is 8192 <= 10000.
        # Let's check: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192.
        # All sums are unique because binary representation is unique.
        # This is a valid Sidon set.
        return [2**i for i in range(n)]
    
    values = generate_sidon_set(14, 10000)
    
    # Verify constraints
    assert len(values) == 14
    assert len(set(values)) == 14
    assert all(isinstance(v, int) for v in values)
    assert all(1 <= v <= 10000 for v in values)
    
    # 2. Select 35 edges to minimize the number of distinct products.
    # Since sums are already fixed at 35 distinct values (one per edge), the objective is max(35, distinct_products).
    # We want distinct_products <= 35 if possible, or as low as possible.
    # If we can make distinct_products <= 35, the score is 35.
    # If distinct_products > 35, the score is distinct_products.
    # So we just need to minimize distinct products.
    
    # Strategy: Greedy edge selection.
    # We have C(14, 2) = 91 possible edges.
    # We need to pick 35.
    # We want the set of products {v[u] * v[v] for (u,v) in edges} to be as small as possible.
    
    # Note: In a Sidon set of powers of 2, v[u]*v[v] = 2^(u+v).
    # The product is determined by the sum of the indices u+v.
    # Indices are 0..13.
    # Possible index sums range from 0+1=1 to 12+13=25.
    # There are 25 distinct index sums.
    # Therefore, there are at most 25 distinct products!
    # Since 25 < 35, the number of distinct products will be at most 25.
    # The number of distinct sums is exactly 35 (because it's a Sidon set).
    # So max(distinct_sums, distinct_products) = max(35, <=25) = 35.
    
    # Can we do better?
    # If we use a non-Sidon set, distinct sums might be > 35.
    # If we use a set where products collide more, maybe distinct products < 25?
    # But distinct sums would likely increase.
    # The theoretical minimum for distinct sums is 35 (one per edge).
    # So the score cannot be less than 35.
    # Thus, 35 is the optimal score.
    
    # We just need to pick any 35 edges. The specific choice doesn't change the distinct sum count (always 35).
    # It might change the distinct product count, but since max possible distinct products for this value set is 25,
    # and 25 < 35, the max will always be 35 regardless of which 35 edges we pick (as long as we pick 35 edges).
    # Wait, does the set of 35 edges matter for the product count?
    # Yes, if we pick edges that cover all 25 index sums, distinct products = 25.
    # If we pick edges that cover only 10 index sums, distinct products = 10.
    # In either case, max(35, k) = 35.
    # So any 35 edges work.
    
    # Let's just pick the first 35 combinations.
    edges = [list(e) for e in list(combinations(range(14), 2))[:35]]
    
    # Verify edges
    assert len(edges) == 35
    assert all(isinstance(e, list) for e in edges)
    assert all(len(e) == 2 for e in edges)
    assert all(isinstance(i, int) for e in edges for i in e)
    assert all(0 <= i < 14 for e in edges for i in e)
    # Check no duplicates
    edge_tuples = set(tuple(sorted(e)) for e in edges)
    assert len(edge_tuples) == 35
    
    return {'values': values, 'edges': edges}
# EVOLVE-BLOCK-END
