gain = 0.0
for idx in block_subs[b]:
    if counts[idx] == 0:
        gain += weights[idx]
