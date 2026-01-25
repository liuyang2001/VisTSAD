# plotting.py
from pathlib import Path
from typing import List
import numpy as np
import matplotlib.pyplot as plt

# 全局字体设置（支持中文）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def draw_with_labels(img, var_names: List[str], title: str, save_path: Path):

    H0, W0, _ = img.shape
    n = len(var_names)
    assert n == H0, "The number of variable names must be equal to the height of `img`."


    scale = 32
    img_up = np.repeat(np.repeat(img, scale, axis=0),
                       scale, axis=1)
    H, W, _ = img_up.shape      # H = H0*32, W = W0*32

    dpi      = 100
    label_w  = 140   
    title_h  = 60    
    time_h   = 40    

    fig_w_px = W + label_w                  
    fig_h_px = H + title_h + time_h        

    fig = plt.figure(figsize=(fig_w_px / dpi, fig_h_px / dpi), dpi=dpi)

    ax_title = fig.add_axes([
        0,
        (time_h + H) / fig_h_px,
        1,
        title_h / fig_h_px
    ])
    ax_title.axis("off")
    ax_title.text(0.5, 0.5, title,
                  ha="center", va="center",
                  fontsize=20, fontweight="bold")

    ax_label = fig.add_axes([
        0,
        time_h / fig_h_px,
        label_w / fig_w_px,
        H / fig_h_px
    ])
    ax_label.set_xlim(0, 1)
    ax_label.set_ylim(n, 0)
    ax_label.set_xticks([])
    ax_label.set_yticks([])
    for spine in ax_label.spines.values():
        spine.set_visible(False)

    ax_img = fig.add_axes([
        label_w / fig_w_px,
        time_h / fig_h_px,
        W / fig_w_px,
        H / fig_h_px
    ])
    ax_img.imshow(
        img_up,
        interpolation="nearest",
        aspect="auto",
        extent=[0, W0, n, 0]   # x:0..W0, y:0..n
    )
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    ax_img.set_ylim(n, 0)
    ax_img.set_zorder(1)

    for i, name in enumerate(var_names):
        y = i + 0.5
        ax_label.text(
            0.98, y, name,
            ha="right", va="center",
            fontsize=14,
            fontweight="bold"
        )

    ax_time = fig.add_axes([
        label_w / fig_w_px,
        0,
        W / fig_w_px,
        time_h / fig_h_px
    ])
    ax_time.set_xlim(0, W0)
    ax_time.set_ylim(0, 1)
    ax_time.set_xticks([])
    ax_time.set_yticks([])
    for spine in ax_time.spines.values():
        spine.set_visible(False)

    for j in range(W0):
        x = j + 0.5
        ax_time.text(
            x, 0.5, str(j + 1),
            ha="center", va="center",
            fontsize=14,
            fontweight="bold"
        )

    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)


def save_corr_heatmap(corr: np.ndarray, save_path: Path,
                      var_names: List[str] = None, title: str = None):

    n = corr.shape[0]
    if var_names is not None:
        assert len(var_names) == n, "The length of `var_names` must be equal to the size of the correlation matrix."

    scale = 32
    H = n * scale
    W = n * scale

    dpi      = 100
    label_w  = 140
    title_h  = 60
    bottom_h = 140

    fig_w_px = W + label_w
    fig_h_px = H + title_h + bottom_h

    fig = plt.figure(figsize=(fig_w_px / dpi, fig_h_px / dpi), dpi=dpi)

    ax_title = fig.add_axes([
        0,
        (bottom_h + H) / fig_h_px,
        1,
        title_h / fig_h_px
    ])
    ax_title.axis("off")
    if title is not None:
        ax_title.text(
            0.5, 0.5, title,
            ha="center", va="center",
            fontsize=20,
            fontweight="bold"
        )

    ax_left = fig.add_axes([
        0,
        bottom_h / fig_h_px,
        label_w / fig_w_px,
        H / fig_h_px
    ])
    ax_left.set_xlim(0, 1)
    ax_left.set_ylim(n, 0)
    ax_left.set_xticks([])
    ax_left.set_yticks([])
    for spine in ax_left.spines.values():
        spine.set_visible(False)

    if var_names is not None:
        for i, name in enumerate(var_names):
            y = i + 0.5
            ax_left.text(
                0.98, y, name,
                ha="right", va="center",
                fontsize=14,
                fontweight="bold"
            )

    ax_img = fig.add_axes([
        label_w / fig_w_px,
        bottom_h / fig_h_px,
        W / fig_w_px,
        H / fig_h_px
    ])
    ax_img.imshow(
        corr,
        vmin=-1, vmax=1,
        cmap="coolwarm",
        interpolation="nearest",
        aspect="auto",
        extent=[0, n, n, 0]
    )
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    ax_img.set_ylim(n, 0)
    ax_img.set_zorder(1)

    ax_bottom = fig.add_axes([
        label_w / fig_w_px,
        0,
        W / fig_w_px,
        bottom_h / fig_h_px
    ])
    ax_bottom.set_xlim(0, n)
    ax_bottom.set_ylim(0, 1)
    ax_bottom.set_xticks([])
    ax_bottom.set_yticks([])
    for spine in ax_bottom.spines.values():
        spine.set_visible(False)

    if var_names is not None:
        for j, name in enumerate(var_names):
            x = j + 0.5
            ax_bottom.text(
                x, 0.5, name,
                ha="center", va="center",
                rotation=90,
                fontsize=14,
                fontweight="bold"
            )

    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)

    if var_names is not None:
        print(f"[INFO] Heatmap saved to {save_path.name}, variable order:")
        print(var_names)

def save_ts_heatmap(
    M: np.ndarray,              # (D, W)
    save_path: Path,
    var_names: List[str],
    title: str = None,
    vmin: float = -1.0,
    vmax: float = 1.0,
    cmap: str = "coolwarm",
):

    assert M.ndim == 2, f"M must be 2D (D,W), got shape={M.shape}"
    D, W0 = M.shape
    assert len(var_names) == D, "The number of variable names must be equal to the number of rows in `M`."

    scale = 32

    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cm = plt.get_cmap(cmap)
    rgba = cm(norm(M))[:, :, :3]               
    img = (rgba * 255).astype(np.uint8)        

    img_up = np.repeat(np.repeat(img, scale, axis=0), scale, axis=1)
    H, Wpx, _ = img_up.shape                   # H = D*scale, Wpx = W0*scale


    dpi     = 100
    label_w = 140
    title_h = 60
    time_h  = 40

    fig_w_px = Wpx + label_w
    fig_h_px = H + title_h + time_h

    fig = plt.figure(figsize=(fig_w_px / dpi, fig_h_px / dpi), dpi=dpi)


    ax_title = fig.add_axes([
        0,
        (time_h + H) / fig_h_px,
        1,
        title_h / fig_h_px
    ])
    ax_title.axis("off")
    if title is not None:
        ax_title.text(
            0.5, 0.5, title,
            ha="center", va="center",
            fontsize=20, fontweight="bold"
        )


    ax_label = fig.add_axes([
        0,
        time_h / fig_h_px,
        label_w / fig_w_px,
        H / fig_h_px
    ])
    ax_label.set_xlim(0, 1)
    ax_label.set_ylim(D, 0)
    ax_label.set_xticks([])
    ax_label.set_yticks([])
    for spine in ax_label.spines.values():
        spine.set_visible(False)

    for i, name in enumerate(var_names):
        y = i + 0.5
        ax_label.text(
            0.98, y, name,
            ha="right", va="center",
            fontsize=14,
            fontweight="bold"
        )


    ax_img = fig.add_axes([
        label_w / fig_w_px,
        time_h / fig_h_px,
        Wpx / fig_w_px,
        H / fig_h_px
    ])
    ax_img.imshow(
        img_up,
        interpolation="nearest",
        aspect="auto",
        extent=[0, W0, D, 0]     # x:0..W0, y:0..D
    )
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    ax_img.set_ylim(D, 0)
    ax_img.set_zorder(1)


    ax_time = fig.add_axes([
        label_w / fig_w_px,
        0,
        Wpx / fig_w_px,
        time_h / fig_h_px
    ])
    ax_time.set_xlim(0, W0)
    ax_time.set_ylim(0, 1)
    ax_time.set_xticks([])
    ax_time.set_yticks([])
    for spine in ax_time.spines.values():
        spine.set_visible(False)

    for j in range(W0):
        x = j + 0.5
        ax_time.text(
            x, 0.5, str(j + 1),
            ha="center", va="center",
            fontsize=14,
            fontweight="bold"
        )

    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)
