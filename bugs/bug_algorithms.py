"""
Animated simulation of the Bug 1, Bug 2 and Tangent Bug path-planning
algorithms (point robot, polygonal obstacles, Matplotlib).

Follows T. Facchinetti, "Robotics - Robot Navigation (1)", Univ. of Pavia, 2023.

Bug 1  motion-to-goal until an obstacle is hit (hit point H); COMPLETE
       circumnavigation of the obstacle (back to H) while remembering the
       boundary point closest to the goal (leave point L); return to L along
       the shorter way round; motion-to-goal again. If moving toward the goal
       from L enters the obstacle, the goal is not reachable.

Bug 2  motion-to-goal along the m-line r (start -> goal); on a hit, follow
       the boundary until r is crossed at m != H with d(m,goal) < d(H,goal)
       and the robot can head to the goal from m; if H is reached again the
       goal is not reachable.

Tangent Bug (range sensor with range R; R = 0 tactile, R = inf ideal, or any
       finite R > 0)
       motion-to-goal: if the goal direction is free within the sensing range
       go toward the goal; otherwise move toward the discontinuity point O_i
       of the range function rho_R, on the obstacle that blocks the goal
       direction, that minimises d(x,O_i) + d(O_i,goal).
       With R = 0 the only "sensed" points are the contact points, so the
       robot slides along the obstacle border while d(x,goal) decreases.
       A local minimum is detected when the best direction would increase
       d(x,goal) -> boundary-following, in the same direction as the last
       motion. During boundary-following
         d_followed = min distance to the goal of the boundary sensed so far
         d_reach    = min distance to the goal of the currently visible part
                      of the followed obstacle (for R = 0: d(x, goal))
       and the robot goes back to motion-to-goal as soon as
       d_reach < d_followed. A complete loop around the obstacle means the
       goal is not reachable.
       Simplification: in boundary-following the robot walks along the
       border (it does not cut across concavities using the sensor).

Usage
-----
    python bug_algorithms.py                              # all 4 panels, slide map
    python bug_algorithms.py --scenario blobs             # polygons|blobs|trap|spiral
    python bug_algorithms.py --algos bug1 bug2            # choose panels
    python bug_algorithms.py --algos tb0 tb2.5 tbinf      # Tangent Bug, R = 0, 2.5, inf
    python bug_algorithms.py --fps 6                      # slower playback
    python bug_algorithms.py --save out.gif               # or out.mp4 (ffmpeg)
"""
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, PathPatch
from matplotlib.path import Path

INF = float("inf")


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def unit(v):
    n = np.hypot(*v)
    return v / n if n > 0 else v


def point_in_ring(p, ring):
    x, y = p
    xs, ys = ring[:, 0], ring[:, 1]
    xs2, ys2 = np.roll(xs, -1), np.roll(ys, -1)
    cond = (ys > y) != (ys2 > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        xint = xs + (y - ys) * (xs2 - xs) / (ys2 - ys)
    return int(np.count_nonzero(cond & (x < xint)) % 2)


def signed_area(ring):
    x, y = ring[:, 0], ring[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


class World:
    """Closed boundary rings (obstacle outlines; a nested ring is a hole)."""

    def __init__(self, rings):
        self.rings = [np.asarray(r, float) for r in rings]
        self.cum = []
        for r in self.rings:
            seg = np.roll(r, -1, axis=0) - r
            self.cum.append(np.concatenate([[0.0], np.cumsum(np.hypot(*seg.T))]))
        self.A = np.vstack(self.rings)                                   # edge starts = vertices
        self.B = np.vstack([np.roll(r, -1, axis=0) for r in self.rings])  # edge ends / next vertex
        self.Vprev = np.vstack([np.roll(r, 1, axis=0) for r in self.rings])
        self.E = self.B - self.A
        self.EE = np.einsum("ij,ij->i", self.E, self.E)
        self.rid = np.concatenate([np.full(len(r), k) for k, r in enumerate(self.rings)])
        self.lidx = np.concatenate([np.arange(len(r)) for r in self.rings])
        self.scale = float(np.ptp(self.A, axis=0).max())
        self.tol = 1e-7 * self.scale

    # ---- basic queries ----------------------------------------------------
    def perimeter(self, k):
        return self.cum[k][-1]

    def inside_many(self, P):
        P = np.atleast_2d(P)
        ax, ay, bx, by = self.A[:, 0], self.A[:, 1], self.B[:, 0], self.B[:, 1]
        px, py = P[:, :1], P[:, 1:]
        cond = (ay > py) != (by > py)
        with np.errstate(divide="ignore", invalid="ignore"):
            xint = ax + (py - ay) * (bx - ax) / (by - ay)
        return np.count_nonzero(cond & (px < xint), axis=1) % 2 == 1

    def inside(self, p):
        return bool(self.inside_many(p)[0])

    def dist_to_boundary_many(self, P):
        W = P[:, None, :] - self.A[None]
        t = np.clip(np.einsum("nek,ek->ne", W, self.E) / self.EE, 0, 1)
        proj = self.A[None] + t[..., None] * self.E[None]
        return np.hypot(*(P[:, None, :] - proj).transpose(2, 0, 1)).min(axis=1)

    def locate(self, x, tol=None):
        """(ring, arc length) if x lies on a boundary, else None."""
        tol = 1e-6 * self.scale if tol is None else tol
        w = x - self.A
        t = np.clip(np.einsum("ek,ek->e", w, self.E) / self.EE, 0, 1)
        d = np.hypot(*(x - (self.A + t[:, None] * self.E)).T)
        j = int(np.argmin(d))
        if d[j] > tol:
            return None
        k, i = self.rid[j], self.lidx[j]
        return k, self.cum[k][i] + t[j] * np.sqrt(self.EE[j])

    def point_at(self, k, s):
        r, cum = self.rings[k], self.cum[k]
        s = s % cum[-1]
        i = min(np.searchsorted(cum, s, "right") - 1, len(r) - 1)
        a, b = r[i], r[(i + 1) % len(r)]
        return a + (s - cum[i]) / (cum[i + 1] - cum[i]) * (b - a)

    def tangents(self, k, s):
        """Unit directions along the ring for increasing (fwd) / decreasing (bwd) s."""
        e = 1e-6 * self.scale
        p = self.point_at(k, s)
        return unit(self.point_at(k, s + e) - p), unit(self.point_at(k, s - e) - p)

    def vertex_dist(self, k, s, d):
        """Arc length from s to the next vertex in direction d."""
        cum, L = self.cum[k], self.cum[k][-1]
        s = s % L
        if d > 0:
            j = np.searchsorted(cum, s + self.tol, "right")
            return cum[min(j, len(cum) - 1)] - s
        j = np.searchsorted(cum, s - self.tol, "left") - 1
        return s - cum[j] if j >= 0 else s + L - cum[-2]

    def first_hit(self, p, q):
        """First crossing of segment p->q with a boundary (t > 0): (point, ring, s)."""
        d = q - p
        dl = np.hypot(*d)
        if dl == 0:
            return None
        denom = cross(np.broadcast_to(d, self.E.shape), self.E)
        ok = np.abs(denom) > 1e-14
        w = self.A - p
        with np.errstate(divide="ignore", invalid="ignore"):
            t = cross(w, self.E) / denom
            u = cross(w, np.broadcast_to(d, self.E.shape)) / denom
        ok &= (t > 1e-9 * self.scale / dl) & (t <= 1 + 1e-12) & (u >= -1e-12) & (u <= 1 + 1e-12)
        if not ok.any():
            return None
        idx = np.where(ok)[0]
        j = idx[np.argmin(t[idx])]
        k, i = self.rid[j], self.lidx[j]
        s = self.cum[k][i] + np.clip(u[j], 0, 1) * np.sqrt(self.EE[j])
        return p + t[j] * d, k, s

    def entering_hit(self, p, q):
        """First point of p->q where the segment really enters an obstacle."""
        while True:
            h = self.first_hit(p, q)
            if h is None:
                return None
            if self.inside(h[0] + 1e-5 * self.scale * unit(q - p)):
                return h
            p = h[0]          # grazing a vertex: keep going

    def free_towards(self, p, goal):
        return not self.inside(p + 1e-4 * self.scale * unit(goal - p))

    def boundary_path(self, k, s0, direction, length):
        """Exact polyline walking `length` along ring k from s0."""
        L = self.perimeter(k)
        s0 = s0 % L
        verts = np.concatenate([self.cum[k][:-1] + m * L for m in (-2, -1, 0, 1, 2)])
        if direction > 0:
            vs = np.sort(verts[(verts > s0) & (verts < s0 + length)])
            end = s0 + length
        else:
            vs = np.sort(verts[(verts < s0) & (verts > s0 - length)])[::-1]
            end = s0 - length
        return [self.point_at(k, s) for s in [s0, *vs, end]]

    def closest_on_ring(self, k, x):
        r, cum = self.rings[k], self.cum[k]
        e = np.roll(r, -1, axis=0) - r
        t = np.clip(np.einsum("ij,ij->i", x - r, e) / np.einsum("ij,ij->i", e, e), 0, 1)
        proj = r + t[:, None] * e
        dist = np.hypot(*(proj - x).T)
        j = int(np.argmin(dist))
        return cum[j] + t[j] * (cum[j + 1] - cum[j]), proj[j], dist[j]

    # ---- range sensor -----------------------------------------------------
    def visible_many(self, x, P):
        """Is the straight segment x -> P[j] collision-free?"""
        P = np.atleast_2d(P)
        n = len(P)
        if n == 0:
            return np.zeros(0, bool)
        D = P - x
        W = self.A - x
        denom = D[:, None, 0] * self.E[None, :, 1] - D[:, None, 1] * self.E[None, :, 0]
        num_t = cross(W, self.E)[None, :]
        num_u = W[None, :, 0] * D[:, None, 1] - W[None, :, 1] * D[:, None, 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            t, u = num_t / denom, num_u / denom
        e = 1e-9
        proper = (np.abs(denom) > 1e-14) & (t > e) & (t < 1 - e) & (u > e) & (u < 1 - e)
        blocked = proper.any(axis=1)
        S = (x + np.array([0.15, 0.5, 0.85])[None, :, None] * D[:, None, :]).reshape(-1, 2)
        ins = self.inside_many(S) & (self.dist_to_boundary_many(S) > 10 * self.tol)
        return ~blocked & ~ins.reshape(n, 3).any(axis=1)

    def circle_hits(self, x, R, mask=None, return_ids=False):
        A, E = (self.A, self.E) if mask is None else (self.A[mask], self.E[mask])
        rid = self.rid if mask is None else self.rid[mask]
        a = np.einsum("ij,ij->i", E, E)
        b = 2 * np.einsum("ij,ij->i", E, A - x)
        c = np.einsum("ij,ij->i", A - x, A - x) - R * R
        disc = b * b - 4 * a * c
        pts, ids = [], []
        for sgn in (-1, 1):
            with np.errstate(invalid="ignore"):
                tau = (-b + sgn * np.sqrt(disc)) / (2 * a)
            ok = (disc >= 0) & (tau >= 0) & (tau <= 1)
            pts.append(A[ok] + tau[ok, None] * E[ok])
            ids.append(rid[ok])
        return (np.vstack(pts), np.concatenate(ids)) if return_ids else np.vstack(pts)

    def discontinuities(self, x, R):
        """Endpoints O_i of the continuity intervals of rho_R seen from x.
        Returns (points, unit directions of the tangent vertices)."""
        D = self.A - x
        dist = np.hypot(*D.T)
        m = (dist > 10 * self.tol) & (dist <= R)
        with np.errstate(invalid="ignore", divide="ignore"):
            u = D / dist[:, None]
        e = 1e-10 * self.scale
        s1 = cross(u, self.Vprev - self.A)
        s2 = cross(u, self.B - self.A)
        opposite = ((s1 < -e) & (s2 > e)) | ((s1 > e) & (s2 < -e))
        both0 = (np.abs(s1) <= e) & (np.abs(s2) <= e)
        idx = np.where(m & ~opposite & ~both0)[0]
        idx = idx[~self.inside_many(self.A[idx] + 1e-6 * self.scale * u[idx])]
        idx = idx[self.visible_many(x, self.A[idx])]
        pts, dirs, ids = self.A[idx], u[idx], self.rid[idx]
        if np.isfinite(R) and R > 0:      # saturation points on the sensing circle
            C, cid = self.circle_hits(x, R, return_ids=True)
            if len(C):
                vis = self.visible_many(x, C)
                pts, ids = np.vstack([pts, C[vis]]), np.concatenate([ids, cid[vis]])
        return pts, dirs, ids

    def shadows(self, x, tang, dirs, k=None):
        """Points where the rays grazing the tangent vertices land behind them
        (other endpoint of the same discontinuity); optionally only on ring k."""
        out = []
        for v, u in zip(tang, dirs):
            h = self.first_hit(v + 1e-6 * self.scale * u, v + 4 * self.scale * u)
            if h is not None and (k is None or h[1] == k):
                out.append(h[0])
        return np.array(out).reshape(-1, 2)

    def d_reach(self, x, k, R, goal):
        """Min distance to the goal over the part of ring k visible from x."""
        if R == 0:
            return np.hypot(*(goal - x))
        mask = self.rid == k
        A, E, EE = self.A[mask], self.E[mask], self.EE[mask]
        t = np.clip(np.einsum("ij,ij->i", goal - A, E) / EE, 0, 1)
        cands = [A, A + t[:, None] * E]
        O, dirs, _ = self.discontinuities(x, R)
        cands.append(self.shadows(x, O[: len(dirs)], dirs, k))   # behind tangent vertices
        if np.isfinite(R):
            cands.append(self.circle_hits(x, R, mask))
        C = np.vstack(cands)
        C = C[np.hypot(*(C - x).T) <= R + self.tol]
        best = np.hypot(*(goal - x))
        if len(C):
            C = C[self.visible_many(x, C)]
            if len(C):
                best = min(best, np.hypot(*(C - goal).T).min())
        return best


# ---------------------------------------------------------------------------
# Path recording
# ---------------------------------------------------------------------------
class Recorder:
    def __init__(self, start, name):
        self.name = name
        self.P = [np.asarray(start, float)]
        self.modes = []          # mode of segment P[i] -> P[i+1]
        self.events = []         # (arc length, label, point)
        self.loops = []          # (arc length at start of boundary-following, perimeter)
        self.length = 0.0
        self.status = ""

    def add(self, mode, pts):
        for q in pts[1:]:
            q = np.asarray(q, float)
            step = np.hypot(*(q - self.P[-1]))
            if step < 1e-12:
                continue
            self.P.append(q)
            self.modes.append(mode)
            self.length += step

    def event(self, label, p):
        self.events.append((self.length, label, np.asarray(p, float)))


def motion_to_goal(world, p, goal):
    """Straight line toward the goal until an obstacle is entered."""
    h = world.entering_hit(p, goal)
    if h is None:
        return [p, goal], None
    q, k, s = h
    return [p, q], (k, s, q)


def follow_direction(world, k, s, heading, turn):
    """+1/-1 along ring k so that the robot turns left/right w.r.t. heading."""
    fwd, _ = world.tangents(k, s)
    d = 1 if cross(heading, fwd) > 0 else -1
    return d if turn == "left" else -d


# ---------------------------------------------------------------------------
# Bug 1
# ---------------------------------------------------------------------------
def bug1(world, start, goal, turn="left", max_obstacles=50):
    rec = Recorder(start, "Bug 1")
    p = np.asarray(start, float)
    for i in range(1, max_obstacles + 1):
        pts, hit = motion_to_goal(world, p, goal)
        rec.add("goal", pts)
        if hit is None:
            rec.status = "goal reached"
            return rec
        k, s_hit, p_hit = hit
        rec.event(f"H{i}", p_hit)
        d = follow_direction(world, k, s_hit, goal - pts[0], turn)
        L = world.perimeter(k)
        # (1) COMPLETE circumnavigation: walk the whole perimeter, back to H
        rec.loops.append((rec.length, L))
        rec.add("follow", world.boundary_path(k, s_hit, d, L))
        # (2) leave point = boundary point closest to the goal
        s_leave, p_leave, _ = world.closest_on_ring(k, goal)
        # (3) go back to it along the shorter way round
        fwd = ((s_leave - s_hit) * d) % L
        back = (world.boundary_path(k, s_hit, d, fwd) if fwd <= L / 2
                else world.boundary_path(k, s_hit, -d, L - fwd))
        rec.add("return", back)
        rec.event(f"L{i}", p_leave)
        p = p_leave
        if not world.free_towards(p, goal):
            rec.status = "goal NOT reachable"
            return rec
    rec.status = "aborted"
    return rec


# ---------------------------------------------------------------------------
# Bug 2
# ---------------------------------------------------------------------------
def bug2(world, start, goal, turn="left", max_hits=200):
    rec = Recorder(start, "Bug 2")
    rec.mline = True
    start, goal = np.asarray(start, float), np.asarray(goal, float)
    r_vec = goal - start
    tol = world.tol
    p = start
    for i in range(1, max_hits + 1):
        pts, hit = motion_to_goal(world, p, goal)
        rec.add("goal", pts)
        if hit is None:
            rec.status = "goal reached"
            return rec
        k, s_hit, p_hit = hit
        rec.event(f"H{i}", p_hit)
        d_hit = np.hypot(*(goal - p_hit))
        d = follow_direction(world, k, s_hit, goal - pts[0], turn)
        loop = world.boundary_path(k, s_hit, d, world.perimeter(k))
        leave = None
        for j in range(len(loop) - 1):
            a, b = loop[j], loop[j + 1]
            ca, cb = cross(r_vec, a - start), cross(r_vec, b - start)
            if ca * cb < 0:
                m = a + (b - a) * ca / (ca - cb)
            elif abs(cb) < tol * np.hypot(*r_vec):
                m = b
            else:
                continue
            lam = np.dot(m - start, r_vec) / np.dot(r_vec, r_vec)
            if (-1e-9 <= lam <= 1 + 1e-9 and np.hypot(*(m - p_hit)) > tol
                    and np.hypot(*(goal - m)) < d_hit - tol
                    and world.free_towards(m, goal)):
                leave = (j, m)
                break
        if leave is None:
            rec.add("follow", loop)
            rec.status = "goal NOT reachable"
            return rec
        j, m = leave
        rec.add("follow", loop[: j + 1] + [m])
        rec.event(f"L{i}", m)
        p = m
    rec.status = "aborted"
    return rec


# ---------------------------------------------------------------------------
# Tangent Bug
# ---------------------------------------------------------------------------
def tangent_bug(world, start, goal, R=INF, turn="left", step=None, max_steps=60000):
    name = "Tangent Bug, R = " + ("∞" if np.isinf(R) else f"{R:g}")
    rec = Recorder(start, name)
    rec.R = R
    step = step or world.scale / 300
    goal = np.asarray(goal, float)
    x = np.asarray(start, float).copy()
    on = world.locate(x)
    mode, last_dir, nmin, passed, hist = "goal", np.zeros(2), 0, [], []
    k = s = d = L = walked = d_followed = None

    for _ in range(max_steps):
        to_goal = goal - x
        dg = np.hypot(*to_goal)
        if dg < 1e-6 * world.scale:
            rec.status = "goal reached"
            return rec
        u = to_goal / dg

        if mode == "goal":
            local_min, best_dir = False, None
            contact = on is not None and world.inside(x + 1e-5 * world.scale * u)
            if R == 0:                                   # ---- tactile sensor
                if not contact:                          # free: straight to goal
                    tgt = x + min(step, dg) * u
                    h = world.entering_hit(x, tgt)
                    if h is not None:
                        rec.add("goal", [x, h[0]])
                        x, on = h[0], (h[1], h[2])
                    else:
                        rec.add("goal", [x, tgt])
                        x, on = tgt, world.locate(tgt)
                    last_dir = u
                    continue
                kk, ss = on                              # touching: slide on the border
                fwd, bwd = world.tangents(kk, ss)
                df, db = np.dot(fwd, u), np.dot(bwd, u)
                if max(df, db) <= 1e-7:
                    local_min = True                     # every move increases d(x,goal)
                else:
                    dd = 1 if df >= db else -1
                    tv = fwd if dd > 0 else bwd
                    # stop at the next vertex or where d(x,goal) stops decreasing
                    lim = min(world.vertex_dist(kk, ss, dd), np.dot(to_goal, tv))
                    w = min(step, lim)
                    pts = world.boundary_path(kk, ss, dd, w)
                    rec.add("goal", pts)
                    x = pts[-1]
                    on = (kk, (ss + dd * w) % world.perimeter(kk))
                    last_dir = tv
                    continue
            else:                                        # ---- range sensor R > 0
                if contact:
                    blocked, kb = True, on[0]
                else:
                    h = world.entering_hit(x, goal)
                    blocked = h is not None and np.hypot(*(h[0] - x)) <= R
                    kb = h[1] if blocked else None
                T = goal
                if blocked:
                    # endpoints O_i of the continuity interval of the BLOCKING obstacle kb:
                    # its tangent vertices / saturation points, plus the points where
                    # rays grazing other obstacles land on it (occlusion boundaries)
                    Oall, dirs, ids = world.discontinuities(x, R)
                    sh = world.shadows(x, Oall[: len(dirs)], dirs, kb)
                    sh = sh[np.hypot(*(sh - x).T) <= R] if len(sh) else sh
                    O = np.vstack([Oall[ids == kb], sh])
                    # ignore tangent vertices already rounded (stepping artefact:
                    # one step past a vertex it is still a discontinuity behind us)
                    for q in passed:
                        O = O[np.hypot(*(O - q).T) > 1e-6 * world.scale] if len(O) else O
                    if len(O) == 0:
                        T = None
                    else:
                        hcost = np.hypot(*(O - x).T) + np.hypot(*(O - goal).T)
                        T = O[int(np.argmin(hcost))]
                        if np.dot(T - x, to_goal) <= 1e-9 * world.scale ** 2:
                            T = None                     # best direction moves away from goal
                if T is not None:
                    dT = np.hypot(*(T - x))
                    # never step past the point where d(x,goal) stops decreasing
                    proj = np.dot((T - x) / dT, to_goal)
                    if proj <= 1e-6 * world.scale or (proj < 1e-3 * step and proj < dT):
                        T = None                         # no progress possible: local minimum
                if T is None:
                    local_min = True
                    best_dir = unit(O[int(np.argmin(hcost))] - x) if blocked and len(O) else None
                else:
                    if min(step, proj) < dT:
                        xn = x + min(step, proj) * (T - x) / dT
                    else:
                        xn = T.copy()
                        if T is not goal:
                            passed.append(T.copy())
                    rec.add("goal", [x, xn])
                    last_dir = unit(xn - x)
                    x, on = xn, world.locate(xn)
                    hist.append(np.hypot(*(goal - x)))
                    if len(hist) > 20 and hist[-21] - hist[-1] < 0.01 * step:
                        hist.clear()                     # stalled (zig-zag): local minimum
                        local_min = True
                    else:
                        continue

            # ---- local minimum: switch to boundary-following
            nmin += 1
            rec.event(f"M{nmin}", x)
            if on is None:                               # not touching: reach the blocking obstacle
                h = world.entering_hit(x, goal)
                if h is None:
                    rec.status = "aborted"
                    return rec
                rec.add("goal", [x, h[0]])
                x, on = h[0], (h[1], h[2])
            k, s = on
            fwd, bwd = world.tangents(k, s)
            if best_dir is not None:                    # go round toward the best O_i
                d = 1 if np.dot(best_dir, fwd) >= np.dot(best_dir, bwd) else -1
            elif np.hypot(*last_dir) > 0:               # keep the direction of the last motion
                d = 1 if np.dot(last_dir, fwd) >= np.dot(last_dir, bwd) else -1
            else:
                d = follow_direction(world, k, s, u, turn)
            L, walked = world.perimeter(k), 0.0
            d_followed = world.d_reach(x, k, R, goal)
            rec.loops.append((rec.length, L))
            mode = "follow"
            continue

        # ---- boundary-following
        d_reach = world.d_reach(x, k, R, goal)
        if d_reach < d_followed - 1e-6 * world.scale:
            rec.event(f"L{nmin}", x)
            mode, passed, hist = "goal", [], []
            continue
        d_followed = min(d_followed, d_reach)
        if walked >= L - 1e-9:
            rec.status = "goal NOT reachable"
            return rec
        w = min(step, L - walked)
        pts = world.boundary_path(k, s, d, w)
        rec.add("follow", pts)
        last_dir = unit(pts[-1] - pts[-2])
        s = (s + d * w) % L
        x, on, walked = pts[-1], (k, s), walked + w
    rec.status = "aborted"
    return rec


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------
def blob(center, R, a1=0.18, a2=0.08, phase=0.0, n=160):
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = R * (1 + a1 * np.sin(3 * th + phase) + a2 * np.cos(5 * th + 2 * phase))
    return np.c_[center[0] + r * np.cos(th), center[1] + r * np.sin(th)]


def thick_polyline(c, w):
    """Polygon of width w around an open polyline (square caps, miter joints)."""
    c = np.asarray(c, float)
    h = w / 2
    d = np.array([unit(c[i + 1] - c[i]) for i in range(len(c) - 1)])
    n = np.c_[-d[:, 1], d[:, 0]]
    c = c.copy()
    c[0] -= h * d[0]
    c[-1] += h * d[-1]
    miters = [n[0]] + [(n[i - 1] + n[i]) / (1 + np.dot(n[i - 1], n[i]))
                       for i in range(1, len(c) - 1)] + [n[-1]]
    miters = np.array(miters)
    return np.vstack([c + h * miters, (c - h * miters)[::-1]])


def scenario(name):
    if name == "polygons":   # slides "Tangent Bug with R = 0 / R = inf"
        px = lambda pts: [(x / 100, -y / 100) for x, y in pts]   # slide pixels -> units
        rings = [px([(280, 258), (283, 405), (375, 575), (355, 315)]),
                 px([(523, 218), (580, 408), (507, 578), (658, 447)]),
                 px([(790, 337), (790, 392), (925, 390), (940, 562), (843, 562),
                     (828, 618), (1000, 618), (982, 352)])]
        return World(rings), np.array([1.55, -3.43]), np.array([10.95, -4.90])
    if name == "blobs":      # slides "Bug 1" / "Bug 2"
        rings = [blob((4.0, 0.5), 1.8, phase=0.3),
                 blob((10.0, 1.1), 2.1, a1=0.22, phase=1.7)]
        return World(rings), np.array([0.0, 0.0]), np.array([14.0, 1.5])
    if name == "trap":       # slide "no path to goal"
        outer = [(0, 0), (10, 0), (10, 5), (6.5, 5), (6.5, 7.5), (0, 7.5)]
        inner = [(0.6, 0.6), (9.4, 0.6), (9.4, 4.4), (5.9, 4.4), (5.9, 6.9), (0.6, 6.9)]
        return World([outer, inner]), np.array([2.0, 2.0]), np.array([8.3, 6.6])
    if name == "spiral":     # slide "Bug 2: example of bad case"
        spiral = [(0.9, -0.9), (-0.9, -0.9), (-0.9, 0.9), (1.8, 0.9), (1.8, -1.8),
                  (-1.8, -1.8), (-1.8, 1.8), (2.7, 1.8), (2.7, -2.7), (-1.0, -2.7)]
        hook = [(-3.6, -1.2), (-3.6, 3.0), (-1.0, 3.0)]
        rings = [thick_polyline(spiral, 0.3), thick_polyline(hook, 0.3)]
        return World(rings), np.array([-6.0, 0.0]), np.array([0.0, 0.0])
    raise ValueError(f"unknown scenario {name!r}")


def run_algo(spec, world, start, goal, turn):
    if spec == "bug1":
        return bug1(world, start, goal, turn)
    if spec == "bug2":
        return bug2(world, start, goal, turn)
    if spec.startswith("tb"):
        R = INF if spec[2:] in ("inf", "∞") else float(spec[2:])
        return tangent_bug(world, start, goal, R, turn)
    raise ValueError(f"unknown algorithm {spec!r} (bug1, bug2, tb0, tbinf, tb<R>)")


# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------
COLORS = {"goal": "#d62728", "follow": "#1f77b4", "return": "#9467bd"}
WIDTHS = {"goal": 2.0, "follow": 4.5, "return": 1.6}   # wide loop under thin return
LABELS = {"goal": "motion-to-goal", "follow": "boundary-following",
          "return": "Bug 1: return to leave point"}
EV_COLORS = {"H": "k", "L": "#2ca02c", "M": "#ff7f0e"}


def obstacle_patch(world):
    verts, codes = [], []
    for k, r in enumerate(world.rings):
        depth = sum(point_in_ring(r[0] + 1e-9, o) for j, o in enumerate(world.rings) if j != k)
        if (depth % 2 == 0) != (signed_area(r) > 0):
            r = r[::-1]
        verts += [*r, r[0]]
        codes += [Path.MOVETO] + [Path.LINETO] * (len(r) - 1) + [Path.CLOSEPOLY]
    return PathPatch(Path(verts, codes), facecolor="#ffd700", edgecolor="k", lw=1.2, zorder=1)


class Panel:
    def __init__(self, ax, world, start, goal, rec):
        self.rec, self.world, self.goal = rec, world, goal
        self.P = np.array(rec.P)
        self.modes = np.array(rec.modes)
        self.C = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(self.P, axis=0).T))])
        self.R = getattr(rec, "R", None)
        ax.add_patch(obstacle_patch(world))
        if getattr(rec, "mline", False):
            ax.plot(*np.c_[start, goal], "--", color="gray", lw=1)
        for p, lab, off in ((start, "start", (-10, -14)), (goal, "goal", (6, 6))):
            ax.plot(*p, "o", mfc="white", mec="k", ms=8, zorder=7)
            ax.annotate(lab, p, xytext=off, textcoords="offset points", fontsize=9)
        self.lines = {m: ax.plot([], [], color=COLORS[m], lw=WIDTHS[m], zorder=3,
                                 solid_capstyle="round")[0]
                      for m in ("follow", "goal", "return")}
        self.sense_lines = LineCollection([], colors="0.45", lw=0.6, zorder=2)
        ax.add_collection(self.sense_lines)
        self.sense_pts, = ax.plot([], [], "o", ms=4, mfc="white", mec="0.3", zorder=4)
        self.circle = None
        if self.R is not None and np.isfinite(self.R) and self.R > 0:
            self.circle = Circle(start, self.R, fill=False, ls=":", ec="0.4", zorder=2)
            ax.add_patch(self.circle)
        self.robot, = ax.plot([], [], "o", color="k", ms=7, zorder=8)
        self.ev_art = []
        for _, lab, p in rec.events:
            dot, = ax.plot(*p, "o", ms=6, zorder=7, visible=False, color=EV_COLORS[lab[0]])
            txt = ax.annotate(lab, p, xytext=(4, 4), textcoords="offset points",
                              fontsize=8, visible=False, zorder=9)
            self.ev_art.append((dot, txt))
        self.info = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", zorder=10,
                            fontsize=8.5, family="monospace",
                            bbox=dict(boxstyle="round", fc="white", alpha=0.85))
        ax.set_title(rec.name)
        ax.set_aspect("equal")

    def update(self, s):
        s = min(s, self.C[-1])
        idx = int(np.searchsorted(self.C, s, "right"))
        if idx >= len(self.P):
            pts, segm = self.P, self.modes
        else:
            i = idx - 1
            t = (s - self.C[i]) / (self.C[i + 1] - self.C[i])
            pts = np.vstack([self.P[:idx], self.P[i] + t * (self.P[i + 1] - self.P[i])])
            segm = self.modes[:idx]
        for m, line in self.lines.items():
            keep = np.zeros(len(pts), bool)
            sel = segm == m
            keep[:-1] |= sel
            keep[1:] |= sel
            xy = np.where(keep[:, None], pts, np.nan)
            line.set_data(xy[:, 0], xy[:, 1])
        x = pts[-1]
        self.robot.set_data([x[0]], [x[1]])
        done = s >= self.C[-1] - 1e-9

        # range sensor: discontinuity points O_i seen from the robot
        if self.R is not None and self.R > 0 and not done:
            O, _, _ = self.world.discontinuities(x, self.R)
            self.sense_lines.set_segments([[x, o] for o in O])
            self.sense_pts.set_data(O[:, 0] if len(O) else [], O[:, 1] if len(O) else [])
        else:
            self.sense_lines.set_segments([])
            self.sense_pts.set_data([], [])
        if self.circle is not None:
            self.circle.center = tuple(x)

        for (dot, txt), (se, _, _) in zip(self.ev_art, self.rec.events):
            dot.set_visible(se <= s + 1e-9)
            txt.set_visible(se <= s + 1e-9)

        mode = "finished" if done else (LABELS[segm[-1]] if len(segm) else "-")
        lines = [f"mode   : {mode}", f"length : {s:6.2f}"]
        for s0, L in self.rec.loops:          # progress of the current boundary walk
            if s0 <= s and not done and len(segm) and segm[-1] == "follow" \
                    and s <= s0 + L + 1e-9:
                lines.append(f"loop   : {100 * (s - s0) / L:5.1f}% of perimeter")
        if done:
            lines.append(f"result : {self.rec.status}")
        self.info.set_text("\n".join(lines))
        return [*self.lines.values(), self.robot, self.info, self.sense_lines,
                self.sense_pts, *[a for pair in self.ev_art for a in pair]] + \
               ([self.circle] if self.circle is not None else [])


def animate(scn="polygons", algos=("bug1", "bug2", "tb0", "tbinf"), turn="left",
            frames=300, fps=10, save=None):
    world, start, goal = scenario(scn)
    recs = [run_algo(a, world, start, goal, turn) for a in algos]
    for r in recs:
        print(f"[{scn}] {r.name:22s} {r.status:18s} path length = {r.length:7.2f}")

    n = len(recs)
    nrows, ncols = {1: (1, 1), 2: (1, 2), 3: (2, 2), 4: (2, 2)}.get(n, (2, (n + 1) // 2))
    allpts = np.vstack([world.A, start, goal])
    lo, hi = allpts.min(0), allpts.max(0)
    pad = 0.08 * (hi - lo).max()
    aspect = (hi[1] - lo[1] + 2 * pad) / (hi[0] - lo[0] + 2 * pad)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.4 * ncols, nrows * (6.4 * aspect + 0.6) + 0.9),
                             squeeze=False)
    axes = axes.ravel()
    for ax in axes[n:]:
        ax.set_visible(False)
    for ax in axes[:n]:
        ax.set_xlim(lo[0] - pad, hi[0] + pad)
        ax.set_ylim(lo[1] - pad, hi[1] + pad)
    panels = [Panel(ax, world, start, goal, r) for ax, r in zip(axes, recs)]

    fig.suptitle(f"scenario '{scn}' (turn {turn}) - all robots move at the same speed")
    handles = [plt.Line2D([], [], color=COLORS[m], lw=2.5, label=LABELS[m]) for m in COLORS]
    handles += [plt.Line2D([], [], ls="--", color="gray", label="m-line r"),
                plt.Line2D([], [], ls="", marker="o", color="k", label="hit H"),
                plt.Line2D([], [], ls="", marker="o", color=EV_COLORS["L"], label="leave L"),
                plt.Line2D([], [], ls="", marker="o", color=EV_COLORS["M"], label="local min M"),
                plt.Line2D([], [], ls="-", marker="o", mfc="white", color="0.45", lw=0.6,
                           label="sensed O_i")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8.5, frameon=False)
    fig.tight_layout(rect=(0, 0.07 if nrows == 1 else 0.05, 1, 1))

    speed = max(r.length for r in recs) / frames
    hold = 2 * fps

    def update(f):
        return [a for p in panels for a in p.update(f * speed)]

    anim = FuncAnimation(fig, update, frames=frames + hold + 1,
                         interval=1000 / fps, blit=True, repeat=False)
    if save:
        writer = FFMpegWriter(fps=fps) if save.endswith(".mp4") else PillowWriter(fps=fps)
        anim.save(save, writer=writer, dpi=72)
        print(f"saved {save}")
    else:
        plt.show()
    return anim


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="polygons",
                    choices=["polygons", "blobs", "trap", "spiral"])
    ap.add_argument("--algos", nargs="+", default=["bug1", "bug2", "tb0", "tbinf"],
                    help="bug1 bug2 tb0 tbinf tb<R> (e.g. tb2.5)")
    ap.add_argument("--turn", default="left", choices=["left", "right"])
    ap.add_argument("--frames", type=int, default=300, help="frames of motion")
    ap.add_argument("--fps", type=float, default=10, help="playback frame rate")
    ap.add_argument("--save", default=None, help="output .gif or .mp4")
    a = ap.parse_args()
    animate(a.scenario, a.algos, a.turn, a.frames, a.fps, a.save)
