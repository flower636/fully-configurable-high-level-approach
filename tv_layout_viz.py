import math
import cutlass.cute as cute            
import cutlass

def visualize_tv_layout(
    tiler_mn: tuple[int, int],
    tv_layout,                       # (((thr_shape),(val_shape)),
                                    #  ((thr_stride),(val_stride)))
    *,
    font_size: int = 10,
    cell_px: int = 70,
    grid_lw: float = 1.5,
    color_fn=None,                  # optional (tid,vid) -> colour
):
    """Draw a T/V checkerboard for an arbitrary TV layout."""
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors    

    # -----------------------------------------------------------------
    # 1)  Build a real CuTe layout from the tuple the user passed
    # -----------------------------------------------------------------
    shape, stride = tv_layout

    if isinstance(shape[0], int):
        n_thr = shape[0]
    else:
        n_thr = math.prod(shape[0])
    if isinstance(shape[1], int):
        n_val = shape[1]
    else:
        n_val = math.prod(shape[1])
    M, N  = tiler_mn

    thr_ids = np.full((M, N), -1, dtype=int)
    val_ids = np.full((M, N), -1, dtype=int)
    filled  = np.zeros((M, N), dtype=bool)

    # -----------------------------------------------------------------
    # 2)  Query CuTe for every (tid, vid) → (m,n)
    # -----------------------------------------------------------------

    @cute.jit
    def g():
        tv_layout  = cute.make_layout(shape, stride=stride)
        tid_vals = []
        for tid in cutlass.range_constexpr(n_thr):
            vid_vals = []
            for vid in cutlass.range_constexpr(n_val):
                vid_vals.append(tv_layout((tid, vid)))
            tid_vals.append(vid_vals)
        return tid_vals
    vals = g()
    for tid in range(n_thr):
        for vid in range(n_val):
            pos = vals[tid][vid]
            n = pos // M
            m = pos % M
            if filled[m, n]:
                continue
            thr_ids[m, n] = tid
            val_ids[m, n] = vid
            filled[m, n]  = True

    # -----------------------------------------------------------------
    # 3)  Colours (default: pastel per-thread)
    # -----------------------------------------------------------------
    if color_fn is None:
        pastel = plt.cm.Set3.colors
        cmap   = (pastel * ((n_thr // 12) + 1))[:n_thr]
        color_fn = lambda t, v: cmap[t % len(cmap)]

    bg_rgb = np.zeros((M, N, 3))
    for m in range(M):
        for n in range(N):
            tid = thr_ids[m, n]
            if tid >= 0:
                bg_rgb[m, n] = mcolors.to_rgb(color_fn(tid, val_ids[m, n]))

    # -----------------------------------------------------------------
    # 4)  Draw
    # -----------------------------------------------------------------
    fig_w, fig_h = N * cell_px / 100, M * cell_px / 100
    fig, ax      = plt.subplots(figsize=(fig_w, fig_h), dpi=100)
    ax.imshow(bg_rgb, interpolation="none")

    for m in range(M):
        for n in range(N):
            if thr_ids[m, n] >= 0:
                ax.text(
                    n, m, f"T{thr_ids[m,n]}\nV{val_ids[m,n]}",
                    ha="center", va="center",
                    fontsize=font_size, weight="bold"
                )

    ax.set_xticks(np.arange(N + 1) - 0.5)
    ax.set_yticks(np.arange(M + 1) - 0.5)
    ax.set_xticklabels([str(i) for i in range(N + 1)])  # Show x tick labels
    ax.set_yticklabels([str(i) for i in range(M + 1)])  # Show y tick labels
    ax.tick_params(axis='both', which='both', length=6, width=1)  # Make ticks more visible
    ax.tick_params(axis='x', which='both', top=True, bottom=False, labeltop=True, labelbottom=False)  # Show ticks and labels on top
    ax.tick_params(axis='y', which='both', left=True, right=False)  # Show ticks on left
    ax.grid(which="major", color="black", linewidth=grid_lw)
    ax.set_xlim(-.5, N -.5); ax.set_ylim(M -.5, -.5)

    ax.set_title(f"tv_layout = {tv_layout}", fontsize=font_size + 2, pad=12)
    plt.tight_layout()
    plt.savefig("tv_layout.svg")


tiler_mn = (8, 8)
tv = (
    ((2, 2, 2), (2, 2, 2)),  # thr_shape / val_shape
    ((1, 16, 4), (8, 2, 32)),  # thr_stride / val_stride
)
visualize_tv_layout(tiler_mn, tv)