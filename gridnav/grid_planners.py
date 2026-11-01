"""
Animated simulation of the Wave-front and A* path planners on occupancy grids
(Matplotlib). Follows T. Facchinetti, "Robotics - Robot Navigation (2)",
University of Pavia, 2023.

Wave-front (slides "Wave-front algorithm")
    1. mark the cell containing the goal with i = 1
    2. mark with i + 1 every adjacent cell to the one marked with i
    3. repeat step 2 until the cell containing the start is marked
       (the current wave is completed) or all cells have been marked
    4. gradient descent from the start cell to the goal cell
    Labels are assigned with a breadth-first visit (one animation step = one
    cell taken from the FIFO queue, whose free neighbours get its label + 1).
    Default: 4-point connectivity, as in the slide example.

A* (slide "The A* algorithm: pseudo-code")
    add V_start to O
    while O is not empty:
        select V_best in O with minimum f = g + h; move it from O to C
        if V_best = goal: path found (cost g(goal));
                          move from O to C all nodes with f >= g(goal)
        for V in Neigh(V_best), V not in C:
            if V not in O: add V to O (parent V_best)
            elif g(V_best) + c(V_best, V) < g(V): connect V to V_best
    Default: 8-point connectivity, step cost 1 / 1.4 and heuristic computed
    with the same 1 / 1.4 steps, as in the slide example. Ties in f are
    broken in FIFO order. Nodes moved to C after the goal has been found
    are drawn with a red cross, parent pointers as yellow arrows.

Usage
-----
    python grid_planners.py                          # slide wave-front map
    python grid_planners.py --map astar_slide        # slide A* map
    python grid_planners.py --map rooms --fps 8      # bigger map
    python grid_planners.py --wf-conn 8 --astar-conn 4
    python grid_planners.py --heuristic zero         # A* becomes Dijkstra
    python grid_planners.py --save wf.gif            # or .mp4 (ffmpeg)
"""
import argparse
import heapq
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon, Rectangle

# ---------------------------------------------------------------------------
# Maps ('#' obstacle, 'S' start, 'G' goal, '.' free); row 1 is the top row
# ---------------------------------------------------------------------------
MAPS = {
    # slide "Wave-front algorithm: example" (grey cells covered by the polygon)
    "wf_slide": dict(rows=[
        "............",
        "............",
        "....#....G..",
        "...###......",
        "...#####....",
        "...######...",
        "...#######..",
        "....#####...",
        "....####....",
        "..S..##.....",
        "............",
        "............"],
        polygon=[(4.87, 3.27), (4.19, 4.02), (4.11, 6.81), (6.35, 10.33), (9.93, 7.21)]),
    # slide "Example of application of A*" (8-point connectivity)
    "astar_slide": dict(rows=[
        "......#.",
        ".G..#.#.",
        "###.#...",
        "..##...#",
        "...#...#",
        "...##.#.",
        "..S..#..",
        "........"]),
    "rooms": dict(rows=[
        "........#...........#...",
        "..S.....#...........#...",
        "........#.....#.....#...",
        "........#.....#.........",
        "..............#.....#...",
        "........#.....#.....#...",
        "#####.###.....#######.##",
        "..............#.........",
        "..............#.........",
        "...######.....#.........",
        "........#..........#####",
        "........#...............",
        "........#.....#......G..",
        "..............#........."]),
    "u_trap": dict(rows=[
        "....................",
        "....................",
        "....................",
        "......#########.....",
        "..............#.....",
        "..............#.....",
        "........S.....#...G.",
        "..............#.....",
        "..............#.....",
        "......#########.....",
        "....................",
        "...................."]),
}


def load_map(name):
    m = MAPS[name]
    rows = m["rows"]
    assert len({len(r) for r in rows}) == 1, "ragged map"
    occ = np.array([[ch == "#" for ch in r] for r in rows])
    find = lambda ch: next((i, r.index(ch)) for i, r in enumerate(rows) if ch in r)
    return occ, find("S"), find("G"), m.get("polygon")


# neighbour order: W, S, E, N (+ diagonals); rows grow downward
N4 = [(0, -1), (1, 0), (0, 1), (-1, 0)]
N8 = N4 + [(1, -1), (1, 1), (-1, 1), (-1, -1)]


def neighbours(occ, cell, conn, cut_corners=True):
    """Free adjacent cells and step cost in tenths (10 = 1, 14 = 1.4)."""
    R, C = occ.shape
    for dr, dc in (N4 if conn == 4 else N8):
        r, c = cell[0] + dr, cell[1] + dc
        if not (0 <= r < R and 0 <= c < C) or occ[r, c]:
            continue
        diag = dr != 0 and dc != 0
        if diag and not cut_corners and (occ[cell[0] + dr, cell[1]] or occ[cell[0], cell[1] + dc]):
            continue
        yield (r, c), (14 if diag else 10)


# ---------------------------------------------------------------------------
# Wave-front
# ---------------------------------------------------------------------------
def wavefront(occ, start, goal, conn=4, full=False):
    lab = np.zeros(occ.shape, int)          # 0 = not marked yet
    lab[goal] = 1
    q = deque([goal])
    snaps = [dict(lab=lab.copy(), cur=None, new=[], qlen=1)]
    stop = None
    while q:
        cell = q.popleft()
        if stop is not None and lab[cell] >= stop:
            break                            # start marked and its wave completed
        new = []
        for nb, _ in neighbours(occ, cell, conn):
            if lab[nb] == 0:
                lab[nb] = lab[cell] + 1
                q.append(nb)
                new.append(nb)
                if nb == start and not full:
                    stop = lab[nb]
        snaps.append(dict(lab=lab.copy(), cur=cell, new=new, qlen=len(q)))

    res = dict(name=f"Wave-front ({conn}-point connectivity)", kind="wf",
               snaps=snaps, labels=lab, path=[], cost=None)
    if lab[start] == 0:
        res["status"] = "goal NOT reachable"
        return res
    # gradient descent: move to the adjacent cell with the lowest label,
    # keeping the current direction when there is a tie
    path, cur, d = [start], start, None
    while lab[cur] != 1:
        cands = [nb for nb, _ in neighbours(occ, cur, conn) if 0 < lab[nb] < lab[cur]]
        lo = min(lab[nb] for nb in cands)
        opts = [nb for nb in cands if lab[nb] == lo]
        straight = None if d is None else (cur[0] + d[0], cur[1] + d[1])
        nxt = straight if straight in opts else opts[0]
        d = (nxt[0] - cur[0], nxt[1] - cur[1])
        path.append(nxt)
        cur = nxt
    res.update(path=path, status="goal reached",
               cost=sum(1.4 if abs(a[0] - b[0]) + abs(a[1] - b[1]) == 2 else 1.0
                        for a, b in zip(path, path[1:])))
    return res


# ---------------------------------------------------------------------------
# A*
# ---------------------------------------------------------------------------
def heuristic_fn(kind, conn, goal):
    def h(cell):
        dr, dc = abs(cell[0] - goal[0]), abs(cell[1] - goal[1])
        if kind == "zero":
            return 0
        if kind == "euclid":
            return 10 * float(np.hypot(dr, dc))
        if kind == "manhattan" or (kind == "auto" and conn == 4):
            return 10 * (dr + dc)
        return 14 * min(dr, dc) + 10 * (max(dr, dc) - min(dr, dc))   # 1 / 1.4 steps
    return h


OPEN, CLOSED, PRUNED = 1, 2, 3


def astar(occ, start, goal, conn=8, heuristic="auto", cut_corners=True):
    h = heuristic_fn(heuristic, conn, goal)
    R, C = occ.shape
    g = np.full((R, C), np.inf)
    H = np.array([[h((r, c)) for c in range(C)] for r in range(R)], float)
    par = np.full((R, C, 2), -1)
    st = np.zeros((R, C), int)
    g[start] = 0
    st[start] = OPEN
    cnt = 0
    heap = [(H[start], cnt, start)]
    g_goal = np.inf

    def snap(best, pruned):
        opn = [(g[c] + H[c], k, c) for f, k, c in heap if st[c] == OPEN and f == g[c] + H[c]]
        return dict(st=st.copy(), g=g.copy(), par=par.copy(), best=best, pruned=pruned,
                    g_goal=g_goal, top=sorted(set(opn))[:4],
                    nO=int((st == OPEN).sum()), nC=int(((st == CLOSED) | (st == PRUNED)).sum()))

    snaps = [snap(None, [])]
    while True:
        while heap and (st[heap[0][2]] != OPEN or heap[0][0] != g[heap[0][2]] + H[heap[0][2]]):
            heapq.heappop(heap)                       # stale entries
        if not heap:
            break
        _, _, best = heapq.heappop(heap)              # min f, FIFO among ties
        st[best] = CLOSED
        pruned = []
        if best == goal:
            g_goal = g[goal]
            for r, c in zip(*np.where(st == OPEN)):   # move from O to C: f >= g(goal)
                if g[r, c] + H[r, c] >= g_goal:
                    st[r, c] = PRUNED
                    pruned.append((r, c))
        else:
            for nb, cost in neighbours(occ, best, conn, cut_corners):
                if st[nb] in (CLOSED, PRUNED):
                    continue
                ng = g[best] + cost
                if st[nb] == 0 or ng < g[nb]:
                    g[nb] = ng
                    par[nb] = best
                    st[nb] = OPEN
                    cnt += 1
                    heapq.heappush(heap, (ng + H[nb], cnt, nb))
                    if ng + H[nb] >= g_goal:          # cannot improve the path found
                        st[nb] = PRUNED
                        pruned.append(nb)
        snaps.append(snap(best, pruned))

    hname = {"auto": "1/1.4 steps" if conn == 8 else "Manhattan", "zero": "h = 0 (Dijkstra)",
             "euclid": "Euclidean", "manhattan": "Manhattan", "octile": "1/1.4 steps"}[heuristic]
    res = dict(name=f"A* ({conn}-point connectivity, h: {hname})", kind="astar",
               snaps=snaps, H=H, path=[], cost=None)
    if not np.isfinite(g_goal):
        res["status"] = "goal NOT reachable"
        return res
    path, cur = [goal], goal                          # backward path goal -> start
    while cur != start:
        cur = tuple(par[cur])
        path.append(cur)
    res.update(path=path[::-1], cost=g_goal / 10, status="goal reached")
    return res


# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------
COL = dict(free="#ffffff", obst="#9a9a9a", poly="#c8a02c",
           wf_lab="#d6e9fb", wf_new="#ffcf73",
           open="#9ccbf7", closed="#3d93f0", pruned="#cfe3f7", start="#4a4af5",
           cur="#ff7f0e", wf_path="#1f3fff", a_path="#00c000", goalbox="#e00000")


def fmt(v):
    v = v / 10
    return f"{v:.1f}".rstrip("0").rstrip(".")


class GridPanel:
    def __init__(self, ax, occ, start, goal, res, polygon, cell_pt):
        self.res, self.occ, self.start, self.goal = res, occ, start, goal
        R, C = occ.shape
        self.R, self.C = R, C
        self.kind = res["kind"]
        self.snaps = res["snaps"]
        self.nsearch = len(self.snaps) - 1
        self.path = res["path"]

        self.base = np.empty((R, C, 4))
        self.base[:] = to_rgba(COL["free"])
        self.base[occ] = to_rgba(COL["obst"])
        self.img = ax.imshow(self.base, extent=(0.5, C + 0.5, R + 0.5, 0.5),
                             interpolation="nearest", zorder=0)
        segs = [[(x + 0.5, 0.5), (x + 0.5, R + 0.5)] for x in range(C + 1)] + \
               [[(0.5, y + 0.5), (C + 0.5, y + 0.5)] for y in range(R + 1)]
        ax.add_collection(LineCollection(segs, colors="#b0b0b0", lw=0.8, zorder=1))
        if polygon is not None:
            ax.add_patch(Polygon(polygon, closed=True, fc=COL["poly"], ec="k",
                                 lw=1, alpha=0.85, zorder=2))
        ax.add_patch(Rectangle((goal[1] + 0.5, goal[0] + 0.5), 1, 1, fill=False,
                               ec=COL["goalbox"], lw=2.2, zorder=6))
        self.cur_box = Rectangle((0, 0), 1, 1, fill=False, ec=COL["cur"], lw=2.5,
                                 zorder=7, visible=False)
        ax.add_patch(self.cur_box)
        ax.plot(start[1] + 1, start[0] + 1, "o", color=COL["wf_path"], ms=7, zorder=9)

        # cell texts (labels for wave-front, h/g/f for A*)
        self.texts = {}
        fs = cell_pt * (0.34 if self.kind == "wf" else 0.165)
        self.show_text = fs >= (5 if self.kind == "wf" else 4.3)
        if self.show_text:
            for r in range(R):
                for c in range(C):
                    if not occ[r, c]:
                        self.texts[(r, c)] = ax.text(
                            c + 1 if self.kind == "wf" else c + 0.6, r + 1, "",
                            ha="center" if self.kind == "wf" else "left", va="center",
                            fontsize=min(fs, 11),
                            family="DejaVu Sans" if self.kind == "wf" else "monospace",
                            zorder=5, linespacing=1.15)

        if self.kind == "astar":
            self.quiv = None
            self.cross = LineCollection([], colors="red", lw=1.6, zorder=8)
            ax.add_collection(self.cross)

        pc = COL["wf_path"] if self.kind == "wf" else COL["a_path"]
        self.path_line, = ax.plot([], [], "-", color=pc, lw=3, zorder=10,
                                  solid_capstyle="round")
        self.robot, = ax.plot([], [], "o", color="k", ms=9, zorder=11)
        self.info = ax.text(0.0, -0.015, "", transform=ax.transAxes, va="top", ha="left",
                            fontsize=8.5, family="monospace",
                            bbox=dict(boxstyle="round", fc="white", alpha=0.9))
        ax.set_xlim(0.5, C + 0.5)
        ax.set_ylim(R + 0.5, 0.5)
        ax.set_aspect("equal")
        ax.set_xticks(range(1, C + 1))
        ax.set_yticks(range(1, R + 1))
        ax.tick_params(labelsize=7, length=0, labeltop=True, labelbottom=False)
        ax.set_title(res["name"], fontsize=11, pad=16)

    def frames_needed(self, spf):
        return int(np.ceil(self.nsearch / spf)) + max(len(self.path) - 1, 0)

    # -- one frame ---------------------------------------------------------
    def update(self, f, spf):
        sf = int(np.ceil(self.nsearch / spf))
        t = min(f * spf, self.nsearch)
        S = self.snaps[t]
        img = self.base.copy()
        txt = {}
        lines = []
        searching = f < sf

        if self.kind == "wf":
            lab = S["lab"]
            img[lab > 0] = to_rgba(COL["wf_lab"])
            if searching and lab.max() > 0:
                img[lab == lab.max()] = to_rgba(COL["wf_new"])    # current wave
            for (r, c), tx in self.texts.items():
                txt[(r, c)] = str(lab[r, c]) if lab[r, c] > 0 else ""
            phase = "wave expansion" if searching else "gradient descent"
            lines += [f"phase    : {phase}",
                      f"expanded : {t} cells (FIFO queue: {S['qlen']})",
                      f"marked   : {(lab > 0).sum()} cells, max label {lab.max()}"]
            if S["cur"] is not None and searching:
                lines.append(f"current  : ({S['cur'][0] + 1},{S['cur'][1] + 1}) "
                             f"label {lab[S['cur']]} -> marks {len(S['new'])} cells")
        else:
            st, g, H = S["st"], S["g"], self.res["H"]
            img[st == OPEN] = to_rgba(COL["open"])
            img[st == CLOSED] = to_rgba(COL["closed"])
            img[st == PRUNED] = to_rgba(COL["pruned"])
            img[self.start] = to_rgba(COL["start"])
            for (r, c), tx in self.texts.items():
                if st[r, c]:
                    txt[(r, c)] = f"h={fmt(H[r, c])}\ng={fmt(g[r, c])}\nf={fmt(g[r, c] + H[r, c])}"
                else:
                    txt[(r, c)] = ""
            # parent pointers: short arrow across the cell border, toward the parent
            seen = st > 0
            seen[self.start] = False
            par = S["par"]
            rr, cc = np.where(seen & (par[..., 0] >= 0))
            u = (par[rr, cc, 1] - cc).astype(float)
            v = (par[rr, cc, 0] - rr).astype(float)
            n = np.hypot(u, v)
            n[n == 0] = 1
            ax = self.img.axes
            if self.quiv is not None:
                self.quiv.remove()
            self.quiv = ax.quiver(cc + 1 + 0.36 * u / n, rr + 1 + 0.36 * v / n,
                                  0.2 * u / n, 0.2 * v / n, color="#ffc000",
                                  edgecolor="#7a5c00", linewidth=0.4, angles="xy",
                                  scale_units="xy", scale=1, width=0.006, headwidth=4,
                                  headlength=4, headaxislength=3.5, zorder=8)
            pr, pc = np.where(st == PRUNED)
            self.cross.set_segments(
                [s for r, c in zip(pr, pc) for s in
                 ([(c + 0.62, r + 0.62), (c + 1.38, r + 1.38)],
                  [(c + 0.62, r + 1.38), (c + 1.38, r + 0.62)])])
            phase = "search" if searching else "path following"
            lines.append(f"phase    : {phase}")
            if S["best"] is not None and searching:
                b = S["best"]
                lines.append(f"V_best   : ({b[0] + 1},{b[1] + 1})  "
                             f"f={fmt(g[b] + H[b])}")
            lines.append(f"|O| = {S['nO']:3d}   |C| = {S['nC']:3d}   expanded: {t}")
            top = "  ".join(f"({c[0] + 1},{c[1] + 1}){fmt(fv)}" for fv, _, c in S["top"])
            lines.append(f"O (best) : {top if top else '-'}")
            if np.isfinite(S["g_goal"]):
                lines.append(f"g(goal)  : {fmt(S['g_goal'])}  (nodes with f >= g(goal) -> C)")

        for cell, tx in self.texts.items():
            tx.set_text(txt.get(cell, ""))
        self.img.set_data(img)

        if searching and S.get("cur" if self.kind == "wf" else "best") is not None:
            r, c = S["cur"] if self.kind == "wf" else S["best"]
            self.cur_box.set_xy((c + 0.5, r + 0.5))
            self.cur_box.set_visible(True)
        else:
            self.cur_box.set_visible(False)

        # robot along the path after the search
        k = f - sf
        if not searching and self.path:
            k = min(k, len(self.path) - 1)
            P = np.array(self.path[: k + 1]) + 1
            self.path_line.set_data(P[:, 1], P[:, 0])
            self.robot.set_data([P[-1, 1]], [P[-1, 0]])
            n = len(self.path) - 1
            lines.append(f"path     : {k}/{n} moves, cost "
                         f"{self.res['cost']:.1f}".rstrip("0").rstrip("."))
            if k == n:
                lines.append(f"result   : {self.res['status']}")
        else:
            self.path_line.set_data([], [])
            s = np.array(self.start) + 1
            self.robot.set_data([s[1]], [s[0]])
            if not searching:
                lines.append(f"result   : {self.res['status']}")
        self.info.set_text("\n".join(lines))
        arts = [self.img, self.cur_box, self.path_line, self.robot, self.info,
                *self.texts.values()]
        if self.kind == "astar":
            arts += [self.quiv, self.cross]
        return arts


def animate(map_name="wf_slide", wf_conn=4, astar_conn=8, heuristic="auto",
            cut_corners=True, fps=4, max_search_frames=160, save=None, dpi=100):
    occ, start, goal, polygon = load_map(map_name)
    res = [wavefront(occ, start, goal, wf_conn),
           astar(occ, start, goal, astar_conn, heuristic, cut_corners)]
    for r in res:
        cost = f"{r['cost']:.1f}" if r["cost"] is not None else "-"
        print(f"[{map_name}] {r['name']:52s} {r['status']:18s} "
              f"expansions = {len(r['snaps']) - 1:4d}  path cost = {cost}")

    R, C = occ.shape
    panel_w = 6.6
    H = panel_w * R / C
    fig = plt.figure(figsize=(2 * panel_w + 0.6, H + 2.9))
    FH = H + 2.9
    axes = [fig.add_axes([0.03 + 0.495 * i, 1.75 / FH, 0.455, H / FH]) for i in range(2)]
    # cell size in points -> font sizes
    bbox = axes[0].get_position()
    cell_pt = min(bbox.width * fig.get_figwidth() / C, bbox.height * fig.get_figheight() / R) * 72
    panels = [GridPanel(ax, occ, start, goal, r, polygon, cell_pt) for ax, r in zip(axes, res)]

    fig.suptitle(f"Grid path planning - map '{map_name}'   "
                 "(one step = one expanded cell)", fontsize=12)
    handles = [Patch(fc=COL["obst"], ec="k", label="occupied cell"),
               Patch(fc=COL["wf_lab"], ec="k", label="wave-front: marked"),
               Patch(fc=COL["wf_new"], ec="k", label="wave-front: current wave"),
               Patch(fc=COL["open"], ec="k", label="A*: open set O"),
               Patch(fc=COL["closed"], ec="k", label="A*: closed set C (expanded)"),
               Patch(fc=COL["pruned"], ec="red", label="A*: moved to C, f >= g(goal)"),
               Patch(fill=False, ec=COL["cur"], lw=2, label="cell being expanded"),
               Patch(fill=False, ec=COL["goalbox"], lw=2, label="goal"),
               Line2D([], [], marker="o", ls="", color=COL["wf_path"], label="start"),
               Line2D([], [], color=COL["wf_path"], lw=3, label="wave-front path"),
               Line2D([], [], color=COL["a_path"], lw=3, label="A* path"),
               Line2D([], [], marker=r"$\leftarrow$", ls="", color="#ffc000", ms=10,
                      label="A*: parent pointer")]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=8, frameon=False,
               handlelength=1.6, columnspacing=1.2)

    spf = max(1, int(np.ceil(max(p.nsearch for p in panels) / max_search_frames)))
    total = max(p.frames_needed(spf) for p in panels) + 3 * int(np.ceil(fps))

    def update(f):
        return [a for p in panels for a in p.update(f, spf)]

    anim = FuncAnimation(fig, update, frames=total, interval=1000 / fps,
                         blit=False, repeat=False)
    if save:
        writer = FFMpegWriter(fps=fps) if save.endswith(".mp4") else PillowWriter(fps=fps)
        anim.save(save, writer=writer, dpi=dpi)
        print(f"saved {save}")
    else:
        plt.show()
    return anim


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", default="wf_slide", choices=list(MAPS))
    ap.add_argument("--wf-conn", type=int, default=4, choices=[4, 8])
    ap.add_argument("--astar-conn", type=int, default=8, choices=[4, 8])
    ap.add_argument("--heuristic", default="auto",
                    choices=["auto", "octile", "manhattan", "euclid", "zero"],
                    help="auto: 1/1.4 steps with 8-conn, Manhattan with 4-conn")
    ap.add_argument("--no-corner-cutting", action="store_true",
                    help="forbid diagonal moves between two occupied cells")
    ap.add_argument("--fps", type=float, default=4, help="playback frame rate")
    ap.add_argument("--save", default=None, help="output .gif or .mp4")
    a = ap.parse_args()
    animate(a.map, a.wf_conn, a.astar_conn, a.heuristic, not a.no_corner_cutting,
            a.fps, save=a.save)
