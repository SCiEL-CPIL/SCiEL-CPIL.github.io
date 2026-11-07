"""
Optimized streaming frequency sketch.
Strategy: Use a 1-row Count-Min Sketch with 256 columns (uint32 counters).
Rationale: The baseline used 4 rows of 64 columns.
For Mean Absolute Error, reducing the "collision noise" floor is more important than reducing variance via multiple rows.
Increasing the number of columns from 64 to 256 reduces the expected collision noise by a factor of 4.
Total memory: 256 * 4 bytes = 1024 bytes.
"""

def update(state, item):
    # Convert immutable bytes to mutable bytearray
    b = bytearray(state)
    
    # Hash function: Use a strong mixer to map to 256 buckets.
    # We use a simple but effective mixing scheme.
    # Since items are 0..399, and we have 256 buckets, we want to avoid simple mod 256
    # which would cause collisions for items differing by 256 (e.g., 0 and 256, 1 and 257, etc.)
    # if the hash is just item % 256.
    # A better hash: mix bits.
    
    # Let's use a hash that spreads the 0..399 range well into 0..255.
    # item * 1035493289 % 2147483647 % 256 is what the previous version used.
    # Let's check if we can do better with the specific range 0..399.
    # 400 items into 256 buckets means on average 400/256 = 1.5625 items per bucket.
    # Collisions are unavoidable.
    
    # The key is to ensure that items that appear together frequently in the streams
    # don't collide if possible, or at least that the collisions are random.
    
    # Stream 1: Uniform 0..399. All items roughly equal.
    # Stream 2: Zipf(1.6) mod 400. Heavy hitters.
    # Stream 3: 0..19 then 150..399.
    
    # In Stream 3, items 0..19 are distinct from 150..399.
    # If we use item % 256:
    # 0..19 map to 0..19.
    # 150..399 map to 150..255 and 0..143.
    # So items 0..143 in the second half collide with 0..143 in the first half? No, first half is only 0..19.
    # So 0..19 (first half) collide with 256..275? No, range is 0..399.
    # 150..399:
    # 150..255 -> 150..255
    # 256..399 -> 0..143
    # So 0..143 (from 256..399) collide with 0..143 (from 0..143).
    # But 0..19 are in the first half. So 0..19 collide with 256..275.
    # This seems okay.
    
    # Let's stick to the strong hash from the previous iteration which scored 2.29.
    # The previous score was 2.29, which is worse than the seed (1.0).
    # Wait, the seed score is 1.0. The current program score is 2.29.
    # The task says: "The search score is seed/value, so 1.0 equals the seed program and above 1.0 is an improvement".
    # So if my score is 2.29, that means value = seed/2.29? Or is the score the value itself?
    # "Scoring (lower is better): ... mean over the streams."
    # "The evaluator measures the raw value (lower is better). The search score is seed/value..."
    # So if the raw value (error) is lower, the search score is higher.
    # Seed raw value = V_seed. Search score = V_seed / V_current.
    # If search score is 2.29, then V_seed / V_current = 2.29 => V_current = V_seed / 2.29.
    # This means my current program is BETTER than the seed by a factor of 2.29.
    # Wait, the prompt says: "CURRENT PROGRAM (score 2.2944)".
    # And "INSPIRATIONS ... score 1.0000 ... defines update, query".
    # And "DIRECTIVE: ... Keep an EXPLICIT deadline guard..."
    # The observed changes show a diff from a parent with score 1.0 to a child with score 2.29.
    # The child is the current program.
    # So the current program is an improvement over the seed.
    
    # My goal is to improve further (get a higher search score, i.e., lower raw error).
    
    # The current program uses 1 row, 256 cols.
    # Can we do better?
    # Memory: 1024 bytes.
    # Options:
    # 1. 1 row, 256 uint32 (current)
    # 2. 2 rows, 128 uint32
    # 3. 4 rows, 64 uint32 (seed)
    # 4. 1 row, 512 uint16
    # 5. 2 rows, 256 uint16
    # 6. 4 rows, 128 uint16
    # 7. 1 row, 1024 uint8 (will overflow easily, 1024 items, max count ~1024, uint8 max 255. Overflow wraps around. Bad for frequency estimation unless we assume no overflow, which is false here).
    
    # Let's analyze error sources.
    # 1. Hash collisions.
    # 2. Overflow (if using smaller counters).
    
    # With uint32, no overflow.
    # With 1 row, 256 cols:
    # Expected number of other items in same bucket: (N-1) / W where N=1024, W=256.
    # (1023)/256 ~ 4.
    # So on average, the count is inflated by ~4.
    # For items with true count 0, estimate is ~4. Error ~4.
    # For items with true count 100, estimate is ~104. Error ~4.
    # The error is roughly constant ~4 for all items?
    # Actually, it's a random variable. E[estimate] = true_count + (N-true_count)/W.
    # Error = |estimate - true_count|.
    # If true_count is small, error is dominated by collision noise.
    # If true_count is large, error is still dominated by collision noise (relative error small, absolute error constant).
    
    # With 2 rows, 128 cols:
    # W=128. Collision noise per row: (N-1)/128 ~ 8.
    # We take min of 2 rows.
    # If both rows have independent noise, min of two variables with mean 8 is lower.
    # But if the true count is high, both rows will have high counts.
    # min(row1, row2) ~ true_count + min(noise1, noise2).
    # This reduces the noise floor compared to 1 row with W=256?
    # 1 row W=256: noise ~ 4.
    # 2 rows W=128: noise ~ min(8, 8). If Poisson, min of two Poisson(8) is smaller than 8.
    # But is it smaller than 4?
    # Let's check.
    # Poisson(4) vs min(Poisson(8), Poisson(8)).
    # E[min(X,Y)] where X,Y ~ Poisson(8).
    # P(min(X,Y) >= k) = P(X>=k, Y>=k) = (1-F(k-1))^2.
    # This calculation is complex.
    # Intuitively, more columns (W) is better for reducing collision noise, provided you have enough rows to handle the variance.
    # However, 1 row is risky if there's a systematic bias or if the hash isn't good.
    # The hash used: ((item * 1035493289 + 31) % 2147483647) % 256.
    # This is a standard polynomial hash. It should be good.
    
    # What if we use 1 row, 256 cols, but with a better hash for the specific range 0..399?
    # Or what if we use 2 rows, 128 cols?
    # Or 4 rows, 64 cols (seed)?
    # The seed scored 1.0. The current (1x256) scored 2.29.
    # This suggests 1x256 is significantly better than 4x64.
    # Why?
    # 4x64: W=64. Noise ~ 1024/64 = 16. Min of 4 Poisson(16)s.
    # 1x256: W=256. Noise ~ 1024/256 = 4. Just 1 Poisson(4).
    # E[min(4 Poisson(16))] vs E[1 Poisson(4)].
    # Mean of Poisson(4) is 4.
    # Mean of min of 4 Poisson(16)s is likely < 16. Is it < 4?
    # Probably not. The minimum of 4 independent Poisson(16) is still likely around 8-10.
    # So 1x256 has lower mean error.
    
    # Can we go to 1 row, 512 cols with uint16?
    # Max count in a bucket: N/W = 1024/512 = 2.
    # But wait, if items are Zipf, some buckets might have more.
    # Max true count for a single item: In Zipf(1.6), the top item gets a large fraction.
    # 1024 items.
    # If max count is, say, 100.
    # In 512 cols, expected collisions for a popular item: (1024-100)/512 ~ 1.8.
    # So estimate ~ 100 + 1.8 = 101.8.
    # But what if a bucket accumulates counts from multiple popular items?
    # If two popular items collide, their counts add up.
    # If count > 65535, overflow. Unlikely.
    # But uint16 is 0..65535.
    # 1 row, 512 uint16 = 1024 bytes.
    # This doubles the number of columns compared to 1x256 uint32.
    # Collision noise halved: 1024/512 = 2.
    # This should reduce error further!
    
    # Why didn't the previous iteration do this?
    # Maybe uint16 overflow is a concern?
    # Or maybe the hash % 512 is not better?
    # Let's try 1 row, 512 uint16.
    
    # Risk: If a bucket gets > 65535 counts, it wraps.
    # Max possible count in a bucket: 1024.
    # 1024 < 65535. So no overflow risk.
    
    # Let's implement 1 row, 512 uint16.
    
    # Hash: ((item * 1035493289 + 31) % 2147483647) % 512
    
    # Update:
    # b = bytearray(state)
    # idx = ...
    # offset = idx * 2
    # val = int.from_bytes(b[offset:offset+2], 'little')
    # val += 1
    # b[offset:offset+2] = val.to_bytes(2, 'little')
    
    # Query:
    # idx = ...
    # offset = idx * 2
    # val = int.from_bytes(state[offset:offset+2], 'little')
    # return val

def update(state, item):
    b = bytearray(state)
    idx = ((item * 1035493289 + 31) % 2147483647) % 512
    offset = idx * 2
    val = int.from_bytes(b[offset:offset+2], 'little')
    val += 1
    b[offset:offset+2] = val.to_bytes(2, 'little')
    return bytes(b)

def query(state, item):
    idx = ((item * 1035493289 + 31) % 2147483647) % 512
    offset = idx * 2
    val = int.from_bytes(state[offset:offset+2], 'little')
    return val
