# plotting.py
from pathlib import Path
from typing import List
import numpy as np
import matplotlib.pyplot as plt

# 全局字体设置（支持中文）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def draw_with_labels(img, var_names: List[str], title: str, save_path: Path):
    """
    img: (H0, W0, 3)，H0 = 变量数，W0 = 时间步数
    """
    H0, W0, _ = img.shape
    n = len(var_names)
    assert n == H0, "变量名数量必须等于 img 的高度"

    # ==== 1. 放大倍数：32 ====
    scale = 32
    img_up = np.repeat(np.repeat(img, scale, axis=0),
                       scale, axis=1)
    H, W, _ = img_up.shape      # H = H0*32, W = W0*32

    # ==== 2. 像素级布局参数 ====
    dpi      = 100
    label_w  = 140   # 左侧标签宽度（像素）
    title_h  = 60    # 顶部标题高度（像素）
    time_h   = 40    # 底部时间轴高度（像素）

    fig_w_px = W + label_w                  # 总宽度 = 标签 + 图像
    fig_h_px = H + title_h + time_h         # 总高度 = 时间轴 + 图像 + 标题

    fig = plt.figure(figsize=(fig_w_px / dpi, fig_h_px / dpi), dpi=dpi)

    # ==== 3. 顶部标题轴 ====
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

    # ==== 4. 左侧标签轴 ====
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

    # ==== 5. 右侧图像轴 ====
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

    # ==== 6. 左侧变量名 ====
    for i, name in enumerate(var_names):
        y = i + 0.5
        ax_label.text(
            0.98, y, name,
            ha="right", va="center",
            fontsize=14,
            fontweight="bold"
        )

    # ==== 7. 底部“时间步”文本轴 ====
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
    """
    相关性矩阵热力图：
      - 使用和 draw_with_labels 类似的布局
      - 矩阵本身按 32 倍像素放大
      - 左侧、下侧都标变量名（与 corr 行列顺序一致）
    """
    n = corr.shape[0]
    if var_names is not None:
        assert len(var_names) == n, "var_names 长度必须等于相关矩阵大小"

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

    # 标题
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

    # 左侧变量名
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

    # 中间矩阵
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

    # 底部变量名
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
    """
    按 draw_with_labels 的像素级布局逻辑绘制 “变量×时间” 的热力图：
      - 左侧变量名
      - 顶部标题
      - 底部时间步（1..W）
      - 右侧图像区域用热力图 (cmap/vmin/vmax 与 save_corr_heatmap 风格一致)
    不绘制 colorbar。

    M: (D, W)  float/float32，建议范围 [-1,1]（soft-threshold 后）
    """
    assert M.ndim == 2, f"M must be 2D (D,W), got shape={M.shape}"
    D, W0 = M.shape
    assert len(var_names) == D, "变量名数量必须等于 M 的行数"

    # ==== 1. 放大倍数：32 ====
    scale = 32

    # 把矩阵映射成 RGBA（热力图颜色），再做像素放大
    # 注意：matplotlib 的 colormap 会输出 float RGBA (H,W,4)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cm = plt.get_cmap(cmap)
    rgba = cm(norm(M))[:, :, :3]               # (D, W, 3) 取 RGB
    img = (rgba * 255).astype(np.uint8)        # (D, W, 3) uint8 便于 repeat

    img_up = np.repeat(np.repeat(img, scale, axis=0), scale, axis=1)
    H, Wpx, _ = img_up.shape                   # H = D*scale, Wpx = W0*scale

    # ==== 2. 像素级布局参数（保持与你 draw_with_labels 一致） ====
    dpi     = 100
    label_w = 140
    title_h = 60
    time_h  = 40

    fig_w_px = Wpx + label_w
    fig_h_px = H + title_h + time_h

    fig = plt.figure(figsize=(fig_w_px / dpi, fig_h_px / dpi), dpi=dpi)

    # ==== 3. 顶部标题轴 ====
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

    # ==== 4. 左侧标签轴 ====
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

    # ==== 5. 右侧图像轴 ====
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

    # ==== 6. 底部时间步文本轴 ====
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
