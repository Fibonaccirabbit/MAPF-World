"""MAPF-World observation, dream and coordinated trajectory visualization."""

import glob
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib.colors import ListedColormap
from PIL import Image
from pogema import GridConfig
from pogema.svg_animation.animation_drawer import (
    AnimationConfig,
    AnimationDrawer,
    GridHolder,
    SvgSettings,
)
from pogema.wrappers.persistence import AgentState

# Adapted from MAPF-World world/visualize.py.


def visualize_decoded_obs(
    obs: Dict,
    pred_actions: Sequence[int],
    *,
    greedy_priority: Tuple[str, ...] = ("u", "d", "l", "r"),
    agent_colors: Optional[List[str]] = None,
    show_grid: bool = True,
    save_path: Optional[str] = None,
):
    """Render an observation and predicted actions."""

    cost_raw = np.array(obs["cost2go"], dtype=float)

    if cost_raw.shape == (121,):
        cost = cost_raw.reshape(11, 11)
    elif cost_raw.shape == (11, 11):
        cost = cost_raw
    else:
        raise ValueError(
            f"Unsupported cost2go shape: {cost_raw.shape}; expected (121,) or (11, 11)"
        )

    obstacle = (cost < -20) | (cost > 20)

    agents_raw = obs.get("agents", [])

    def _is_num(x):
        return isinstance(x, (int, float)) and np.isfinite(x)

    def _to_int(x):
        return int(x) if _is_num(x) else None

    agents = []
    for i, a in enumerate(agents_raw):
        rx, ry = a.get("relative_pos", (None, None))
        gx, gy = a.get("relative_goal", (None, None))

        if not (_is_num(rx) and _is_num(ry)):
            continue

        sx, sy = float(rx), float(ry)

        array_x = int(rx + 5)
        array_y = int(ry + 5)

        if not (0 <= array_x < 11 and 0 <= array_y < 11):
            print(f"Warning: agent {i} start ({rx}, {ry}) is outside the cost map; skipped")
            continue

        if obstacle[array_y, array_x]:
            print(f"Warning: agent {i} start ({rx}, {ry}) is on an obstacle; skipped")
            continue

        goal_ok = False
        gx_plot, gy_plot = None, None
        if _is_num(gx) and _is_num(gy):
            gx_plot, gy_plot = float(gx), float(gy)
            goal_array_x = int(gx + 5)
            goal_array_y = int(gy + 5)

            if 0 <= goal_array_x < 11 and 0 <= goal_array_y < 11:
                goal_ok = True
                if obstacle[goal_array_y, goal_array_x]:
                    print(f"Warning: agent {i} goal ({gx}, {gy}) is on an obstacle")
            else:
                print(f"Warning: agent {i} goal ({gx}, {gy}) is outside the cost map")

        hist = a.get("previous_actions", ["n"] * 5)
        hist = [
            (str(h).lower() if str(h).lower() in ("u", "d", "l", "r") else "n") for h in hist[:5]
        ]
        mask = str(a.get("next_action", "0000"))
        mask = "".join(ch if ch in "01" else "0" for ch in mask)
        mask = (mask + "0000")[:4]  # pad to 4

        order = {"u": 0, "d": 1, "l": 2, "r": 3}
        greedy = "n"
        for g in greedy_priority:
            if mask[order[g]] == "1":
                greedy = g
                break

        agents.append(
            {
                "start": (sx, sy),
                "goal": (gx_plot, gy_plot) if goal_ok else None,
                "history": hist,  # 5
                "greedy": greedy,  # 1
                "goal_valid": goal_ok,
            }
        )

    action_map = {0: "u", 1: "d", 2: "l", 3: "r", 4: "w"}  # w for wait
    pa = [action_map.get(x, "w") for x in pred_actions]
    if len(pa) < len(agents):
        pa += ["w"] * (len(agents) - len(pa))
    pa = pa[: len(agents)]

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 11,
            "axes.linewidth": 1.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.alpha": 0.3,
            "figure.dpi": 300,
        }
    )

    nA = len(agents)
    max_agents = 13

    if nA == 0:
        fig_height = 8.0
        height_ratios = [4, 1]
        hspace = 0.1
    else:
        table_height_needed = 3.5 + max_agents * 0.25
        map_height = 6.0
        fig_height = map_height + table_height_needed + 0.5
        height_ratios = [map_height, table_height_needed]
        hspace = 0.03

    fig = plt.figure(figsize=(8.0, fig_height), layout="constrained")
    gs = fig.add_gridspec(2, 1, height_ratios=height_ratios, hspace=hspace)
    ax_map = fig.add_subplot(gs[0, 0])
    ax_seq = fig.add_subplot(gs[1, 0])

    cmap = ListedColormap(["#f8f9fa", "#2c3e50"])
    ax_map.imshow(
        obstacle.astype(int), cmap=cmap, origin="upper", extent=[-5.5, 5.5, 5.5, -5.5], alpha=0.9
    )

    if show_grid:
        ax_map.set_xticks(range(-5, 6))
        ax_map.set_yticks(range(-5, 6))
        ax_map.grid(True, linewidth=0.8, alpha=0.25, color="#7f8c8d")

    ax_map.set_xlim(-5.5, 5.5)
    ax_map.set_ylim(5.5, -5.5)
    ax_map.set_xlabel("Relative Position X", fontsize=12, fontweight="bold")
    ax_map.set_ylabel("Relative Position Y", fontsize=12, fontweight="bold")
    ax_map.set_title("Multi-Agent Path Finding Environment", fontsize=14, fontweight="bold", pad=20)

    for spine in ax_map.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color("#34495e")

    if agent_colors is None:
        academic_colors = [
            "#e74c3c",
            "#3498db",
            "#2ecc71",
            "#f39c12",
            "#9b59b6",
            "#1abc9c",
            "#e67e22",
            "#34495e",
            "#e91e63",
            "#795548",
        ]
        agent_colors = [
            academic_colors[i % len(academic_colors)] for i in range(max(1, len(agents)))
        ]

    for i, ag in enumerate(agents):
        c = agent_colors[i % len(agent_colors)]
        sx, sy = ag["start"]

        ax_map.scatter(
            [sx],
            [sy],
            s=120,
            marker="o",
            edgecolor="white",
            linewidths=2.5,
            c=c,
            zorder=4,
            alpha=0.9,
        )

        if ag["goal"] is not None and ag.get("goal_valid", True):
            gx, gy = ag["goal"]

            ax_map.scatter(
                [gx],
                [gy],
                s=150,
                marker="*",
                edgecolor="white",
                linewidths=2.5,
                c=c,
                zorder=4,
                alpha=0.9,
            )
        elif ag["goal"] is not None:
            gx, gy = ag["goal"]

            ax_map.scatter(
                [gx],
                [gy],
                s=120,
                marker="x",
                edgecolor="red",
                linewidths=3,
                c=c,
                zorder=4,
                alpha=0.7,
            )

        label_text = f"A{i}"
        if ag["goal"] is not None and not ag.get("goal_valid", True):
            label_text += "!"

        ax_map.text(
            sx + 0.25,
            sy - 0.25,
            label_text,
            fontsize=10,
            color="white",
            fontweight="bold",
            ha="center",
            va="center",
            bbox=dict(facecolor=c, edgecolor="white", alpha=0.9, pad=0.3, boxstyle="round,pad=0.3"),
        )

    ax_seq.set_axis_off()
    if nA == 0:
        ax_seq.text(
            0.5,
            0.5,
            "No Valid Agents",
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color="#7f8c8d",
        )
    else:
        action_display = {"u": "↑", "d": "↓", "l": "←", "r": "→", "w": "W", "n": "-"}
        action_names = {"u": "Up", "d": "Down", "l": "Left", "r": "Right", "w": "Wait", "n": "None"}

        labels = ["H1", "H2", "H3", "H4", "H5", "Greedy", "Predicted"]
        nC = len(labels)

        table_margin = 0.02
        table_left = table_margin
        table_right = 1.0 - table_margin
        table_top = 0.96
        table_bottom = 0.04

        table_width = table_right - table_left
        table_height = table_top - table_bottom

        agent_col_width = 0.08
        data_col_width = (table_width - agent_col_width) / nC

        total_rows = max_agents + 1  # +1 for header
        row_height = table_height / (total_rows + 0.5)  # +0.5 for title space

        ax_seq.set_xlim(0, 1)
        ax_seq.set_ylim(0, 1)

        table_rect = patches.Rectangle(
            (table_left, table_bottom),
            table_width,
            table_height,
            linewidth=2,
            edgecolor="#34495e",
            facecolor="none",
        )
        ax_seq.add_patch(table_rect)

        header_rect = patches.Rectangle(
            (table_left, table_top - row_height),
            table_width,
            row_height,
            facecolor="#ecf0f1",
            edgecolor="#34495e",
            linewidth=1.5,
            alpha=0.8,
        )
        ax_seq.add_patch(header_rect)

        for j in range(nC + 1):
            x = table_left + agent_col_width + j * data_col_width
            ax_seq.plot(
                [x, x], [table_bottom, table_top], color="#34495e", linewidth=1.2, alpha=0.7
            )

        ax_seq.plot(
            [table_left + agent_col_width, table_left + agent_col_width],
            [table_bottom, table_top],
            color="#34495e",
            linewidth=2,
        )

        for i in range(nA + 1):
            y = table_top - (i + 1) * row_height
            ax_seq.plot(
                [table_left, table_right], [y, y], color="#34495e", linewidth=1.2, alpha=0.7
            )

        header_fontsize = 9
        ax_seq.text(
            table_left + agent_col_width / 2,
            table_top - row_height / 2,
            "Agent",
            ha="center",
            va="center",
            fontsize=header_fontsize,
            fontweight="bold",
            color="#2c3e50",
        )

        for j, label in enumerate(labels):
            x = table_left + agent_col_width + (j + 0.5) * data_col_width
            y = table_top - row_height / 2
            ax_seq.text(
                x,
                y,
                label,
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                color="#2c3e50",
            )

        for i, ag in enumerate(agents):
            y = table_top - (i + 2) * row_height + row_height / 2
            color = agent_colors[i % len(agent_colors)]

            if i % 2 == 1:
                row_rect = patches.Rectangle(
                    (table_left, table_top - (i + 2) * row_height),
                    table_width,
                    row_height,
                    facecolor="#f8f9fa",
                    edgecolor="none",
                    alpha=0.5,
                )
                ax_seq.add_patch(row_rect)

            agent_fontsize = 9
            ax_seq.text(
                table_left + agent_col_width / 2,
                y,
                f"A{i}",
                ha="center",
                va="center",
                fontsize=agent_fontsize,
                fontweight="bold",
                color=color,
            )

            seq = ag["history"] + [ag["greedy"], pa[i]]
            for j, a_char in enumerate(seq):
                x = table_left + agent_col_width + (j + 0.5) * data_col_width

                if j == len(seq) - 1:
                    highlight_rect = patches.Rectangle(
                        (
                            table_left + agent_col_width + j * data_col_width,
                            table_top - (i + 2) * row_height,
                        ),
                        data_col_width,
                        row_height,
                        facecolor=color,
                        alpha=0.15,
                        edgecolor=color,
                        linewidth=1.5,
                    )
                    ax_seq.add_patch(highlight_rect)

                symbol = action_display.get(str(a_char).lower(), "-")
                name = action_names.get(str(a_char).lower(), "Unknown")

                symbol_fontsize = 11
                name_fontsize = 7

                symbol_offset = row_height * 0.12
                name_offset = row_height * 0.18

                ax_seq.text(
                    x,
                    y + symbol_offset,
                    symbol,
                    ha="center",
                    va="center",
                    fontsize=symbol_fontsize,
                    fontweight="bold",
                    color=color,
                )

                ax_seq.text(
                    x,
                    y - name_offset,
                    name,
                    ha="center",
                    va="center",
                    fontsize=name_fontsize,
                    color="#7f8c8d",
                    alpha=0.8,
                )

        title_y = 0.985
        ax_seq.text(
            0.5,
            title_y,
            "Action Sequence Table",
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color="#2c3e50",
        )

    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, (ax_map, ax_seq)


def create_gif_from_png_folder(
    folder_path: str,
    output_path: str,
    duration: int = 500,
    loop: int = 0,
    sort_by_name: bool = True,
    max_images: Optional[int] = None,
) -> None:
    """Create a GIF from PNG frames in a directory."""

    if not os.path.exists(folder_path):
        raise ValueError(f"Directory does not exist: {folder_path}")

    if not os.path.isdir(folder_path):
        raise ValueError(f"Path is not a directory: {folder_path}")

    png_pattern = os.path.join(folder_path, "*.png")
    png_files = glob.glob(png_pattern)

    if not png_files:
        raise ValueError(f"No PNG files found in: {folder_path}")

    if sort_by_name:

        def natural_sort_key(filename):

            numbers = re.findall(r"\d+", os.path.basename(filename))
            return [int(num) for num in numbers] if numbers else [0]

        png_files.sort(key=natural_sort_key)
    else:
        png_files.sort()

    if max_images is not None and max_images > 0:
        png_files = png_files[:max_images]
        print(
            f"Found {len(glob.glob(png_pattern))} PNG files; using the first {len(png_files)} files"
        )
    else:
        print(f"Found {len(png_files)} PNG files")

    print(
        f"File order: {[os.path.basename(f) for f in png_files[:5]]}{'...' if len(png_files) > 5 else ''}"
    )

    images = []
    for i, png_file in enumerate(png_files):
        try:
            img = Image.open(png_file)

            if img.mode != "RGB":
                img = img.convert("RGB")
            images.append(img)
            print(f"Loaded image {i + 1}/{len(png_files)}: {os.path.basename(png_file)}")
        except Exception as e:
            print(f"Warning: cannot read image {png_file}: {e}")
            continue

    if not images:
        raise IOError("No images loaded")

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    try:
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=loop,
            optimize=True,
        )
        print(f"GIF saved to: {output_path}")
        print(
            f"Frame count: {len(images)}, frame duration: {duration}ms, loops: {'infinite' if loop == 0 else loop} times"
        )
    except Exception as e:
        raise IOError(f"GIF save failed: {e}")


def create_gif_from_png_list(
    png_files: List[str], output_path: str, duration: int = 500, loop: int = 0
) -> None:
    """Create a GIF from an ordered list of PNG frames."""
    if not png_files:
        raise ValueError("PNG file list cannot be empty")

    print(f"Processing {len(png_files)} PNG files")

    images = []
    for i, png_file in enumerate(png_files):
        if not os.path.exists(png_file):
            print(f"Warning: missing file {png_file}")
            continue

        try:
            img = Image.open(png_file)

            if img.mode != "RGB":
                img = img.convert("RGB")
            images.append(img)
            print(f"Loaded image {i + 1}/{len(png_files)}: {os.path.basename(png_file)}")
        except Exception as e:
            print(f"Warning: cannot read image {png_file}: {e}")
            continue

    if not images:
        raise IOError("No images loaded")

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    try:
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=loop,
            optimize=True,
        )
        print(f"GIF saved to: {output_path}")
        print(
            f"Frame count: {len(images)}, frame duration: {duration}ms, loops: {'infinite' if loop == 0 else loop} times"
        )
    except Exception as e:
        raise IOError(f"GIF save failed: {e}")


# Adapted from MAPF-World world/visualize_coordination.py.


class CoordinatedVisualizer:
    def __init__(
        self, global_map, history, dreams, target_agent_idx, goals=None, radius=5, save_dir="dreams"
    ):
        self.global_map = global_map
        self.history = history  # [agent_idx][step] -> (x, y)
        self.dreams = dreams  # {step: [decoded_obs]}
        self.target_agent_idx = target_agent_idx
        self.goals = goals  # [agent_idx] -> (x, y)
        self.radius = radius
        self.save_dir = save_dir

        self.svg_settings = SvgSettings()
        self.scale = self.svg_settings.scale_size

        self.colors = [
            "#c1433c",
            "#2e6f9e",
            "#6e81af",
            "#00b9c8",
            "#72D5C8",
            "#0ea08c",
            "#8F7B66",
            "#F1948A",
            "#85C1E9",
            "#BB8FCE",
            "#F7DC6F",
            "#E59866",
            "#82E0AA",
            "#D7BDE2",
        ]

        self.global_h, self.global_w = global_map.shape
        self.dream_size = 2 * radius + 1

        self.margin = 50
        self.global_view_width = self.global_w * self.scale
        self.global_view_height = self.global_h * self.scale

        self.dream_view_width = self.dream_size * self.scale
        self.dream_view_height = self.dream_size * self.scale

        self.total_width = self.global_view_width + self.margin + self.dream_view_width
        self.total_height = max(self.global_view_height, self.dream_view_height)

        self.timeline_events = []  # List of (type, duration, data)
        self._build_timeline()

    def _build_timeline(self):
        num_steps = len(self.history[0])

        for t in range(num_steps):
            if t in self.dreams:
                dream_frames = self.dreams[t]
                duration = len(dream_frames) * 0.25  # 0.25s per frame
                self.timeline_events.append(
                    {"type": "THINK", "duration": duration, "step": t, "dream_frames": dream_frames}
                )

            if t < num_steps - 1:
                self.timeline_events.append(
                    {
                        "type": "ACT",
                        "duration": 0.5,  # 0.5s for move
                        "step": t,
                        "next_step": t + 1,
                    }
                )

        self.timeline_events.append(
            {
                "type": "HOLD",
                "duration": 2.0,  # Hold for 2 seconds
                "step": num_steps - 1,
            }
        )

        self.total_duration = sum(e["duration"] for e in self.timeline_events)

    def render(self, filename="coordinated.svg"):
        os.makedirs(self.save_dir, exist_ok=True)

        svg_content = []

        svg_content.append(
            f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{self.total_width}" height="{self.total_height}" viewBox="0 0 {self.total_width} {self.total_height}">'
        )

        svg_content.append(self._generate_defs())

        svg_content.append('<g id="global_map">')
        svg_content.append(self._render_global_background())
        svg_content.append("</g>")

        if self.goals:
            svg_content.append('<g id="global_targets">')
            svg_content.append(self._render_global_targets())
            svg_content.append("</g>")

        svg_content.append('<g id="global_agents">')
        svg_content.append(self._render_global_agents())
        svg_content.append("</g>")

        svg_content.append(
            f'<g id="dream_view" transform="translate({self.global_view_width + self.margin}, 0)">'
        )
        svg_content.append(self._render_dream_view())
        svg_content.append("</g>")

        svg_content.append('<g id="connector">')
        svg_content.append(self._render_connector())
        svg_content.append("</g>")

        svg_content.append("</svg>")

        with open(os.path.join(self.save_dir, filename), "w") as f:
            f.write("\n".join(svg_content))

        return os.path.join(self.save_dir, filename)

    def _generate_defs(self):
        return f'''
        <defs>
            <rect id="obstacle" width="{self.svg_settings.r * 2}" height="{self.svg_settings.r * 2}" fill="{self.svg_settings.obstacle_color}" rx="{self.svg_settings.rx}"/>
            <style>
                .line {{stroke: {self.svg_settings.obstacle_color}; stroke-width: {self.svg_settings.stroke_width};}}
                .grid-line {{stroke: #eee; stroke-width: 2;}}
                .agent {{r: {self.svg_settings.r};}}
                .target {{fill: none; stroke-width: {self.svg_settings.stroke_width}; r: {self.svg_settings.r};}}
                .connector {{stroke: {self.svg_settings.ego_color}; stroke-width: 5; stroke-dasharray: 10;}}
            </style>
        </defs>
        '''

    def _render_global_targets(self):
        elements = []
        for i, goal in enumerate(self.goals):
            gx, gy = goal

            x = gy * self.scale + self.scale / 2
            y = gx * self.scale + self.scale / 2

            color = self.colors[i % len(self.colors)]
            elements.append(f'<circle cx="{x}" cy="{y}" stroke="{color}" class="target"/>')

        return "\n".join(elements)

    def _render_global_background(self):
        elements = []
        for i in range(self.global_h + 1):
            y = i * self.scale
            elements.append(
                f'<line x1="0" y1="{y}" x2="{self.global_view_width}" y2="{y}" class="grid-line"/>'
            )
        for i in range(self.global_w + 1):
            x = i * self.scale
            elements.append(
                f'<line x1="{x}" y1="0" x2="{x}" y2="{self.global_view_height}" class="grid-line"/>'
            )

        for r in range(self.global_h):
            for c in range(self.global_w):
                if self.global_map[r, c] == 1:
                    x = c * self.scale + self.scale / 2 - self.svg_settings.r
                    y = r * self.scale + self.scale / 2 - self.svg_settings.r
                    elements.append(f'<use xlink:href="#obstacle" x="{x}" y="{y}"/>')
        return "\n".join(elements)

    def _render_global_agents(self):
        elements = []

        agent_animations = {
            i: {"cx": [], "cy": [], "keyTimes": []} for i in range(len(self.history))
        }

        def add_checkpoint(time_point):
            time_point / self.total_duration if self.total_duration > 0 else 0
            for i in range(len(self.history)):
                pass

        current_time = 0

        for i in range(len(self.history)):
            r, c = self.history[i][0]
            x = c * self.scale + self.scale / 2
            y = r * self.scale + self.scale / 2
            agent_animations[i]["cx"].append(x)
            agent_animations[i]["cy"].append(y)
            agent_animations[i]["keyTimes"].append(0)

        for event in self.timeline_events:
            end_time = current_time + event["duration"]
            norm_end = end_time / self.total_duration if self.total_duration > 0 else 1.0

            if event["type"] == "THINK" or event["type"] == "HOLD":
                step = event["step"]
                for i in range(len(self.history)):
                    r, c = self.history[i][step]
                    x = c * self.scale + self.scale / 2
                    y = r * self.scale + self.scale / 2

                    agent_animations[i]["cx"].append(x)
                    agent_animations[i]["cy"].append(y)
                    agent_animations[i]["keyTimes"].append(norm_end)

            elif event["type"] == "ACT":
                step = event["step"]
                next_step = event["next_step"]
                for i in range(len(self.history)):
                    r, c = self.history[i][next_step]
                    x = c * self.scale + self.scale / 2
                    y = r * self.scale + self.scale / 2

                    agent_animations[i]["cx"].append(x)
                    agent_animations[i]["cy"].append(y)
                    agent_animations[i]["keyTimes"].append(norm_end)

            current_time = end_time

        for i in range(len(self.history)):
            cx_vals = ";".join(map(str, agent_animations[i]["cx"]))
            cy_vals = ";".join(map(str, agent_animations[i]["cy"]))
            keyTimes = ";".join(map(str, agent_animations[i]["keyTimes"]))

            color = self.colors[i % len(self.colors)]

            anim_cx = f'<animate attributeName="cx" values="{cx_vals}" keyTimes="{keyTimes}" dur="{self.total_duration}s" repeatCount="indefinite"/>'
            anim_cy = f'<animate attributeName="cy" values="{cy_vals}" keyTimes="{keyTimes}" dur="{self.total_duration}s" repeatCount="indefinite"/>'

            start_cx = agent_animations[i]["cx"][0]
            start_cy = agent_animations[i]["cy"][0]

            elements.append(
                f'<circle cx="{start_cx}" cy="{start_cy}" fill="{color}" class="agent">{anim_cx}{anim_cy}</circle>'
            )

        return "\n".join(elements)

    def _render_dream_view(self):

        elements = []

        current_time = 0

        elements.append(
            f'<rect width="{self.dream_view_width}" height="{self.dream_view_height}" fill="none" stroke="#333" stroke-width="2"/>'
        )

        for event in self.timeline_events:
            start_time = current_time
            end_time = current_time + event["duration"]

            if event["type"] == "THINK":
                dream_group = self._generate_dream_sequence(
                    event["dream_frames"], start_time, event["duration"]
                )
                elements.append(dream_group)

            current_time = end_time

        return "\n".join(elements)

    def _generate_dream_sequence(self, dream_frames, start_time, duration):

        norm_start = start_time / self.total_duration if self.total_duration > 0 else 0
        norm_end = (start_time + duration) / self.total_duration if self.total_duration > 0 else 1

        # KeyTimes must be strictly increasing.

        vals = []
        keys = []

        if norm_start > 0:
            vals.append("none")
            keys.append("0")

        vals.append("inline")
        keys.append(str(norm_start))

        if norm_end < 1:
            vals.append("none")
            keys.append(str(norm_end))
            vals.append("none")  # To fill to 1
            keys.append("1")
        else:
            pass

        if len(vals) == 1:
            vals.append("inline")
            keys.append("1")

        outer_anim = f'<animate attributeName="display" values="{";".join(vals)}" keyTimes="{";".join(keys)}" dur="{self.total_duration}s" repeatCount="indefinite" calcMode="discrete"/>'

        inner_elements = []

        frame_duration = 0.25
        len(dream_frames)

        for i, frame in enumerate(dream_frames):
            frame_start = start_time + i * frame_duration
            frame_end = frame_start + frame_duration

            f_norm_start = frame_start / self.total_duration
            f_norm_end = frame_end / self.total_duration

            f_vals = []
            f_keys = []

            if f_norm_start > 0:
                f_vals.append("none")
                f_keys.append("0")

            f_vals.append("inline")
            f_keys.append(str(f_norm_start))

            if f_norm_end < 1:
                f_vals.append("none")
                f_keys.append(str(f_norm_end))
                f_vals.append("none")
                f_keys.append("1")

            if len(f_vals) == 1:
                f_vals.append("inline")
                f_keys.append("1")

            f_anim = f'<animate attributeName="display" values="{";".join(f_vals)}" keyTimes="{";".join(f_keys)}" dur="{self.total_duration}s" repeatCount="indefinite" calcMode="discrete"/>'

            content = self._render_single_dream_frame_content(frame)
            inner_elements.append(f'<g display="none">{f_anim}{content}</g>')

        return f'<g display="none">{outer_anim}{"".join(inner_elements)}</g>'

    def _render_single_dream_frame_content(self, decoded):

        grid_size = self.dream_size
        radius = self.radius

        elements = []

        if "cost2go" in decoded:
            cost_raw = np.array(decoded["cost2go"], dtype=float)
            if cost_raw.shape == (grid_size * grid_size,):
                cost_grid = cost_raw.reshape(grid_size, grid_size)
            elif cost_raw.shape == (grid_size, grid_size):
                cost_grid = cost_raw
            else:
                cost_grid = np.zeros((grid_size, grid_size))

            obstacles_mask = (cost_grid < -20) | (cost_grid > 20)

            for r in range(grid_size):
                for c in range(grid_size):
                    if obstacles_mask[r, c]:
                        x = c * self.scale + self.scale / 2 - self.svg_settings.r
                        y = r * self.scale + self.scale / 2 - self.svg_settings.r
                        elements.append(f'<use xlink:href="#obstacle" x="{x}" y="{y}"/>')

        frame_agents = []
        frame_agents.append((radius, radius, self.svg_settings.ego_color))

        if "agents" in decoded:
            for agent_info in decoded["agents"]:
                rel_x, rel_y = agent_info.get("relative_pos", (None, None))
                if rel_x is None or rel_y is None or rel_x == "!" or rel_y == "!":
                    continue
                try:
                    rx, ry = float(rel_x), float(rel_y)
                    if rx == 0 and ry == 0:
                        continue
                    abs_x, abs_y = radius + rx, radius + ry
                    if 0 <= abs_x < grid_size and 0 <= abs_y < grid_size:
                        frame_agents.append(
                            (int(abs_x), int(abs_y), self.svg_settings.ego_other_color)
                        )
                except Exception:
                    continue

        for ax, ay, color in frame_agents:
            cx = ay * self.scale + self.scale / 2

            cx = ay * self.scale + self.scale / 2
            cy = ax * self.scale + self.scale / 2
            elements.append(
                f'<circle cx="{cx}" cy="{cy}" r="{self.svg_settings.r}" fill="{color}"/>'
            )

        return "\n".join(elements)

    def _render_connector(self):

        elements = []
        current_time = 0

        dream_center_x = self.global_view_width + self.margin  # Left edge of dream box
        dream_center_y = self.dream_view_height / 2

        for event in self.timeline_events:
            start_time = current_time
            end_time = current_time + event["duration"]

            if event["type"] == "THINK":
                step = event["step"]
                r, c = self.history[self.target_agent_idx][step]

                gx = c * self.scale + self.scale / 2
                gy = r * self.scale + self.scale / 2

                norm_start = start_time / self.total_duration
                norm_end = end_time / self.total_duration

                color = self.colors[self.target_agent_idx % len(self.colors)]

                vals = []
                keys = []
                if norm_start > 0:
                    vals.append("none")
                    keys.append("0")
                vals.append("inline")
                keys.append(str(norm_start))
                if norm_end < 1:
                    vals.append("none")
                    keys.append(str(norm_end))
                    vals.append("none")
                    keys.append("1")
                if len(vals) == 1:
                    vals.append("inline")
                    keys.append("1")

                anim = f'<animate attributeName="display" values="{";".join(vals)}" keyTimes="{";".join(keys)}" dur="{self.total_duration}s" repeatCount="indefinite" calcMode="discrete"/>'

                elements.append(
                    f'<line x1="{gx}" y1="{gy}" x2="{dream_center_x}" y2="{dream_center_y}" class="connector" stroke="{color}" display="none">{anim}</line>'
                )

            current_time = end_time

        return "\n".join(elements)


def save_coordinated_animation(
    global_map, history, dreams, target_agent_idx, save_dir, goals=None, radius=5
):
    viz = CoordinatedVisualizer(
        global_map, history, dreams, target_agent_idx, goals=goals, radius=radius, save_dir=save_dir
    )
    return viz.render()


# Adapted from MAPF-World world/visualize_dream.py.


def save_dream_animation(decoded_dreams, env_step, target_agent_idx, radius=5, save_dir="dreams"):
    """
    Saves the dream sequence as individual SVG frames to visualize dynamic changes in egocentric view.

    Args:
        decoded_dreams: List of decoded observations for the sequence.
        env_step: Current environment step.
        target_agent_idx: Index of the agent being visualized.
        radius: Observation radius.
        save_dir: Base directory.
    """
    dream_dir = os.path.join(save_dir, f"step_{env_step}")
    os.makedirs(dream_dir, exist_ok=True)

    grid_size = 2 * radius + 1

    frames_content = []
    header_defs = None

    for t, decoded in enumerate(decoded_dreams):
        obstacles_np = np.zeros((grid_size, grid_size), dtype=np.int32)
        if "cost2go" in decoded:
            cost_raw = np.array(decoded["cost2go"], dtype=float)
            if cost_raw.shape == (grid_size * grid_size,):
                cost_grid = cost_raw.reshape(grid_size, grid_size)
            elif cost_raw.shape == (grid_size, grid_size):
                cost_grid = cost_raw
            else:
                cost_grid = np.zeros((grid_size, grid_size))

            obstacles_mask = (cost_grid < -20) | (cost_grid > 20)
            obstacles_np[obstacles_mask] = 1

        frame_agents_state = []  # List of AgentState for THIS frame

        frame_agents_state.append(AgentState(radius, radius, radius, radius, 0, True))

        if "agents" in decoded:
            for i, agent_info in enumerate(decoded["agents"]):
                rel_x, rel_y = agent_info.get("relative_pos", (None, None))

                if rel_x is None or rel_y is None or rel_x == "!" or rel_y == "!":
                    continue
                try:
                    rel_x = float(rel_x)
                    rel_y = float(rel_y)
                except (ValueError, TypeError):
                    continue

                if rel_x == 0 and rel_y == 0:
                    continue  # Skip Ego (already added)

                abs_x, abs_y = radius + rel_x, radius + rel_y
                if 0 <= abs_x < grid_size and 0 <= abs_y < grid_size:
                    frame_agents_state.append(
                        AgentState(int(abs_x), int(abs_y), int(abs_x), int(abs_y), 0, True)
                    )

        svg_settings = SvgSettings()
        colors = {
            i: svg_settings.ego_color if i == 0 else svg_settings.ego_other_color
            for i in range(len(frame_agents_state))
        }

        history = [[state] for state in frame_agents_state]
        cfg = GridConfig(num_agents=len(frame_agents_state), obs_radius=radius, map_name="dream")

        anim_cfg = AnimationConfig(directory=dream_dir, static=True, frame_idx=0, show_agents=True)

        gh = GridHolder(
            obstacles=obstacles_np,
            episode_length=1,
            height=grid_size,
            width=grid_size,
            colors=colors,
            history=history,
            obs_radius=radius,
            grid_config=cfg,
            config=anim_cfg,
            svg_settings=svg_settings,
        )

        drawer = AnimationDrawer()
        drawing = drawer.create_animation(gh)
        full_svg = drawing.render()

        if t == 0:
            defs_end_tag = "</defs>"
            split_idx = full_svg.find(defs_end_tag)
            if split_idx != -1:
                header_defs = full_svg[: split_idx + len(defs_end_tag)]
            else:
                header_defs = full_svg.split(">", 1)[0] + ">"

        defs_end_tag = "</defs>"
        start_idx = full_svg.find(defs_end_tag)
        if start_idx != -1:
            start_idx += len(defs_end_tag)
        else:
            start_idx = full_svg.find(">") + 1

        end_idx = full_svg.rfind("</svg>")
        body = full_svg[start_idx:end_idx].strip()
        frames_content.append(body)

    if not frames_content:
        return dream_dir

    num_frames = len(frames_content)
    time_per_frame = 0.25
    total_duration = num_frames * time_per_frame

    final_svg_parts = []
    if header_defs:
        final_svg_parts.append(header_defs)
    else:
        final_svg_parts.append('<svg xmlns="http://www.w3.org/2000/svg">')

    for t, content in enumerate(frames_content):
        values = ["none"] * num_frames
        values[t] = "inline"
        values_str = ";".join(values)

        # With discrete, equal intervals are used if keyTimes is omitted.
        anim_tag = f'<animate attributeName="display" values="{values_str}" dur="{total_duration}s" repeatCount="indefinite" calcMode="discrete"/>'

        initial_display = "inline" if t == 0 else "none"

        group = f'<g display="{initial_display}">{anim_tag}{content}</g>'
        final_svg_parts.append(group)

    final_svg_parts.append("</svg>")

    output_path = os.path.join(dream_dir, "dream_animated.svg")
    with open(output_path, "w") as f:
        f.write("\n".join(final_svg_parts))

    return dream_dir
