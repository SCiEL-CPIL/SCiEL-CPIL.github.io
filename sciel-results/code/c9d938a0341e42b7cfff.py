    mask4_np = np.array([mask4[x] for x in range(1, n+1)], dtype=np.int64) if any(mask4[x] for x in range(1,n+1)) else None
