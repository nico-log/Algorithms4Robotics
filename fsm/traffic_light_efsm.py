"""
Extended Finite State Machine of the traffic-light controller at a pedestrian
crosswalk, its animated state diagram, and a synchronized 3D animation of the
crosswalk (Matplotlib).

Follows T. Facchinetti, "Finite State Machines", Univ. of Pavia, slides
"Extended state machine example: traffic light":

    inputs   : btn : {true, false}
    outputs  : light(x), x in {R, G, Y}
    variables: cnt : {0, ..., 60}

    initial          -> RED        / cnt:=0
    RED     -> RED                   cnt:=cnt+1
    RED     -> GREEN   cnt==60       / light(G)  cnt:=0
    GREEN   -> GREEN   cnt < 60      / cnt:=cnt+1
    GREEN   -> PENDING btn && cnt<60 / cnt:=cnt+1
    GREEN   -> YELLOW  btn && cnt==60/ light(Y)  cnt:=0
    PENDING -> PENDING                 cnt:=cnt+1
    PENDING -> YELLOW  cnt==60       / light(Y)  cnt:=0
    YELLOW  -> YELLOW                  cnt:=cnt+1
    YELLOW  -> RED     cnt>=5        / light(R)  cnt:=0

The machine is time triggered: it reacts once per second (t = 1, 2, 3, ...).
The self-loops are *default* transitions: they are taken only when no other
outgoing transition is enabled. This removes the overlap between
"cnt < 60" and "btn && cnt < 60" in GREEN (the slide abbreviates the
self-loop guard). When nothing is enabled (GREEN, cnt == 60, btn false) the
machine stutters: it stays in GREEN and cnt stays 60.

The 3D scene is driven by the same simulation. Pedestrians press the button
(the btn input, a pulse sampled at the next reaction), cars follow the car
light produced by light(x). The pedestrian WALK signal is not an output of
the slide's EFSM: it is derived from the extended state (WALK in RED while
cnt < 60 - clearance, then flashing DON'T WALK, DON'T WALK otherwise).

Usage
-----
    python traffic_light_efsm.py                       # 3D crosswalk (left) + FSM (right)
    python traffic_light_efsm.py --fsm-only            # 2D FSM animation only
    python traffic_light_efsm.py --check               # reachable states + reaction trace
    python traffic_light_efsm.py --fps 8 --substeps 2  # playback speed
    python traffic_light_efsm.py --t-green 20 --t-yellow 3 --t-end 120   # short demo
    python traffic_light_efsm.py --save crosswalk.gif  # or .mp4 (ffmpeg)
"""
import argparse
from collections import deque
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, FancyArrowPatch, Patch
from matplotlib.path import Path
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# ===========================================================================
# 1. The extended finite state machine
# ===========================================================================
@dataclass(frozen=True)
class Transition:
    src: str
    dst: str
    guard: object          # callable(btn, cnt) -> bool, or None (no guard)
    guard_txt: str
    outputs: tuple         # e.g. ("light(G)",)
    action: object         # callable(cnt) -> new cnt   (set action)
    action_txt: str
    default: bool = False  # taken only if no other outgoing transition is enabled

    def enabled(self, btn, cnt):
        return self.guard is None or bool(self.guard(btn, cnt))

    def label(self):
        """Label in the slide's format: guard / outputs set-actions."""
        if not self.guard_txt and not self.outputs:
            return self.action_txt
        return "\n".join([self.guard_txt, "/", *self.outputs, self.action_txt])


@dataclass(frozen=True)
class Reaction:
    k: int                 # reaction index (time t = k seconds)
    btn: bool
    src: str
    dst: str
    cnt_before: int
    cnt_after: int
    tr: int                # index of the fired transition, -1 = stutter
    outputs: tuple
    light: str             # car light after the reaction


class TrafficLightEFSM:
    STATES = ("RED", "GREEN", "PENDING", "YELLOW")

    def __init__(self, t_green=60, t_yellow=5):
        T, TY = int(t_green), int(t_yellow)
        self.T, self.TY = T, TY
        inc = lambda c: c + 1
        zero = lambda c: 0
        self.transitions = [
            Transition("RED", "GREEN", lambda b, c: c == T, f"cnt=={T}",
                       ("light(G)",), zero, "cnt:=0"),
            Transition("RED", "RED", None, "", (), inc, "cnt:=cnt+1", default=True),
            Transition("GREEN", "YELLOW", lambda b, c: b and c == T, f"btn && cnt=={T}",
                       ("light(Y)",), zero, "cnt:=0"),
            Transition("GREEN", "PENDING", lambda b, c: b and c < T, f"btn && cnt < {T}",
                       (), inc, "cnt:=cnt+1"),
            Transition("GREEN", "GREEN", lambda b, c: c < T, f"cnt < {T}",
                       (), inc, "cnt:=cnt+1", default=True),
            Transition("PENDING", "YELLOW", lambda b, c: c == T, f"cnt=={T}",
                       ("light(Y)",), zero, "cnt:=0"),
            Transition("PENDING", "PENDING", None, "", (), inc, "cnt:=cnt+1", default=True),
            Transition("YELLOW", "RED", lambda b, c: c >= TY, f"cnt>={TY}",
                       ("light(R)",), zero, "cnt:=0"),
            Transition("YELLOW", "YELLOW", None, "", (), inc, "cnt:=cnt+1", default=True),
        ]
        self.reset()

    def reset(self):
        """Initial transition: cnt:=0, state RED. The car light is assumed red."""
        self.state, self.cnt, self.light, self.k = "RED", 0, "R", 0

    def enabled(self, state, btn, cnt):
        """Index of the transition that fires from (state, cnt) with input btn."""
        out = [i for i, tr in enumerate(self.transitions)
               if tr.src == state and not tr.default and tr.enabled(btn, cnt)]
        if len(out) > 1:
            raise RuntimeError(f"non-deterministic EFSM in {state} (cnt={cnt}, btn={btn})")
        if not out:
            out = [i for i, tr in enumerate(self.transitions)
                   if tr.src == state and tr.default and tr.enabled(btn, cnt)]
        return out[0] if out else -1

    def react(self, btn):
        """One reaction (called once per second)."""
        self.k += 1
        src, c0 = self.state, self.cnt
        i = self.enabled(src, btn, c0)
        outputs = ()
        if i >= 0:
            tr = self.transitions[i]
            outputs = tr.outputs                       # output actions first ...
            for o in outputs:
                self.light = o[o.index("(") + 1]
            self.cnt = tr.action(c0)                   # ... then the set action
            self.state = tr.dst
        return Reaction(self.k, bool(btn), src, self.state, c0, self.cnt, i, outputs, self.light)

    def reachable_states(self):
        """Breadth-first exploration of (state, cnt) over every input sequence."""
        start = ("RED", 0)
        seen, q = {start}, deque([start])
        while q:
            s, c = q.popleft()
            for btn in (False, True):
                i = self.enabled(s, btn, c)
                nxt = (s, c) if i < 0 else (self.transitions[i].dst,
                                            self.transitions[i].action(c))
                if nxt not in seen:
                    seen.add(nxt)
                    q.append(nxt)
        return seen


def ped_signal(state, cnt, T):
    """Pedestrian signal derived from the extended state (not an EFSM output)."""
    if state != "RED":
        return "DONT"
    return "WALK" if cnt < T - clearance(T) else "FLASH"


def clearance(T):
    return max(3, round(T / 6))


# ===========================================================================
# 2. Crosswalk world (pedestrians press btn, cars obey the car light)
# ===========================================================================
XL = 19.0                     # visible half length of the street [m]
ROAD_END = 85.0               # cars are simulated on [-85, 85]
CURB, SIDE_Y, WAIT_Y, SIDE_H = 3.5, 5.0, 3.85, 0.15
CROSS_X = 2.5                 # crosswalk spans x in [-2.5, 2.5]
STOP_S = ROAD_END - 4.0       # stop lines at x = -4 (eastbound) and +4 (westbound)
CAR_L, CAR_W = 4.4, 1.8
PED_POLE = {-1: np.array([-3.15, -4.1]), +1: np.array([3.15, 4.1])}
BUTTON_PT = {-1: np.array([-2.7, -4.5]), +1: np.array([2.7, 4.5])}
SLOTS_X = {-1: [-2.0, -1.0, 0.0, 1.0, 2.0], +1: [2.0, 1.0, 0.0, -1.0, -2.0]}
CAR_COLORS = ["#1f4e9c", "#c0392b", "#2e8b57", "#6c3483", "#e8e8e8", "#2b2b2b",
              "#d35400", "#7f8c8d", "#16a085", "#b7950b"]
SHIRTS = ["#e74c3c", "#3498db", "#9b59b6", "#f39c12", "#1abc9c", "#e84393",
          "#2c3e50", "#27ae60", "#d35400", "#8e44ad"]


def ground_z(y):
    return SIDE_H if abs(y) > CURB else 0.0


class Pedestrian:
    def __init__(self, idx, t_spawn, side, x0, exit_dir, speed, cross=True):
        self.idx, self.t_spawn, self.side, self.exit_dir = idx, t_spawn, side, exit_dir
        self.speed, self.cross = speed, cross
        self.color = SHIRTS[idx % len(SHIRTS)]
        self.pos = np.array([x0, side * SIDE_Y], float)
        self.heading = np.array([np.sign(-x0) or 1.0, 0.0])
        self.mode, self.slot, self.phase = "hidden", None, 0.0
        self.moving, self.press_until, self.react_t = False, -1.0, None
        self.delay = 0.4 + 0.25 * (idx % 4)          # reaction time to WALK
        self.cross_x = None

    def move_to(self, target, dt):
        d = target - self.pos
        n = np.hypot(*d)
        if n < 1e-6:
            self.moving = False
            return True
        step = self.speed * dt
        self.heading = d / n
        self.moving = True
        self.phase += 2 * np.pi * step / 1.4        # one gait cycle every 1.4 m
        if step >= n:
            self.pos = target.astype(float).copy()
            return True
        self.pos = self.pos + d / n * step
        return False

    def step(self, t, dt, w):
        s = self.side
        if self.mode == "hidden":
            if t < self.t_spawn:
                return
            self.mode = "approach" if self.cross else "stroll"
        if self.mode == "stroll":
            if self.move_to(np.array([self.exit_dir * (XL + 2), s * SIDE_Y]), dt):
                self.mode = "gone"
        elif self.mode == "approach":
            # walk along the sidewalk, then step to the push button
            tgt = BUTTON_PT[s] if abs(self.pos[0] - BUTTON_PT[s][0]) < 1.2 else \
                np.array([BUTTON_PT[s][0] - np.sign(BUTTON_PT[s][0] - self.pos[0]) * 1.0, s * SIDE_Y])
            if self.move_to(tgt, dt) and np.allclose(tgt, BUTTON_PT[s]):
                if w.signal == "WALK":
                    self.mode = "to_slot"
                else:
                    self.press(t, w)
                    self.mode = "press"
        elif self.mode == "press":
            self.moving = False
            self.heading = np.array([0.0, 1.0]) * 0 + (PED_POLE[s] - self.pos) / \
                max(1e-6, np.hypot(*(PED_POLE[s] - self.pos)))
            if t >= self.press_until:
                self.mode = "to_slot"
        elif self.mode == "to_slot":
            if self.slot is None:
                self.slot = w.take_slot(s, self)
            tgt = np.array([SLOTS_X[s][self.slot], s * WAIT_Y])
            if self.move_to(tgt, dt):
                self.mode = "wait"
        elif self.mode == "wait":
            self.moving = False
            self.heading = np.array([0.0, -s])
            if w.signal == "WALK":
                if self.react_t is None:
                    self.react_t = t + self.delay
                if t >= self.react_t:
                    self.cross_x = SLOTS_X[s][self.slot]
                    w.free_slot(s, self)
                    self.mode = "cross"
            else:
                self.react_t = None
                # nobody pressed for a long time and the cars still have green:
                # the pedestrian next to the pole presses again
                if (self.slot == 0 and w.fsm.state == "GREEN"
                        and t - w.last_press[s] > 30 and t >= self.press_until):
                    self.press(t, w)
        elif self.mode == "cross":
            if self.move_to(np.array([self.cross_x, -s * WAIT_Y]), dt):
                self.mode = "exit"
        elif self.mode == "exit":
            if self.move_to(np.array([self.cross_x, -s * SIDE_Y]), dt):
                self.mode = "leave"
        elif self.mode == "leave":
            if self.move_to(np.array([self.exit_dir * (XL + 2), -s * SIDE_Y]), dt):
                self.mode = "gone"

    def press(self, t, w):
        self.press_until = t + 0.9
        w.press(self.side, t)

    def snapshot(self, t):
        if self.mode in ("hidden", "gone"):
            return None
        pressing = t < self.press_until
        return dict(pos=self.pos.copy(), heading=self.heading.copy(), phase=self.phase,
                    moving=self.moving, color=self.color, side=self.side,
                    press=PED_POLE[self.side] if pressing else None, mode=self.mode)


class Car:
    def __init__(self, d, s, v, color):
        self.d, self.s, self.v, self.a, self.color = d, s, v, 0.0, color
        self.decision = None          # 'stop' / 'go' while the light is yellow or red

    def x_front(self):
        return self.d * (self.s - ROAD_END)


def idm(v, gap, dv, s0, v0=13.9, a=1.4, b=2.0, T=1.2):
    s_star = s0 + max(0.0, v * T + v * dv / (2 * np.sqrt(a * b)))
    return a * (1 - (v / v0) ** 4 - (s_star / max(gap, 0.05)) ** 2)


class Crosswalk:
    def __init__(self, fsm, pedestrians, seed=7, car_headway=(2.5, 7.0)):
        self.fsm = fsm
        self.peds = pedestrians
        self.cars = {+1: [], -1: []}
        self.rng = np.random.default_rng(seed)
        self.headway = car_headway
        self.next_spawn = {+1: 0.0, -1: 1.5}
        self.slots = {-1: [None] * 5, +1: [None] * 5}
        self.last_press = {-1: -1e9, +1: -1e9}
        self.press_flag = False
        self.signal = "WALK"

    # ---- push buttons and waiting slots -----------------------------------
    def press(self, side, t):
        self.last_press[side] = t
        self.press_flag = True

    def take_slot(self, side, ped):
        sl = self.slots[side]
        i = sl.index(None) if None in sl else len(sl) - 1
        sl[i] = ped
        return i

    def free_slot(self, side, ped):
        sl = self.slots[side]
        for i, p in enumerate(sl):
            if p is ped:
                sl[i] = None

    # ---- cars -----------------------------------------------------------------
    def must_stop(self, car, light, ped_on_crossing):
        if car.s > STOP_S + 0.2:                       # already past the stop line
            return False
        if light == "G":
            car.decision = None
            return ped_on_crossing                     # yield to pedestrians
        gap = STOP_S - car.s
        if car.decision is None:
            if light == "Y":                           # stop only if comfortably possible
                car.decision = "stop" if gap >= car.v ** 2 / (2 * 3.0) else "go"
            else:
                car.decision = "stop"
        return car.decision == "stop"

    def step_cars(self, t, dt, light):
        crossing = any(p.mode == "cross" for p in self.peds)
        for d, lane in self.cars.items():
            lane.sort(key=lambda c: -c.s)
            for i, car in enumerate(lane):
                acc = idm(car.v, 1e9, 0.0, 2.0)
                if i > 0:
                    ld = lane[i - 1]
                    acc = min(acc, idm(car.v, ld.s - CAR_L - car.s, car.v - ld.v, 2.0))
                if self.must_stop(car, light, crossing):
                    acc = min(acc, idm(car.v, STOP_S - car.s, car.v, 0.6))
                car.a = max(acc, -8.0)
            for car in lane:
                car.v = max(0.0, car.v + car.a * dt)
                car.s += car.v * dt
            lane[:] = [c for c in lane if c.s < 2 * ROAD_END + 5]
            if t >= self.next_spawn[d]:
                last = min(lane, key=lambda c: c.s) if lane else None
                if last is None or last.s > CAR_L + 8:
                    v = 13.9 if last is None or last.s > 40 else min(13.9, last.v)
                    lane.append(Car(d, 0.0, v, CAR_COLORS[self.rng.integers(len(CAR_COLORS))]))
                    self.next_spawn[d] = t + self.rng.uniform(*self.headway)

    def warmup(self, seconds, dt=0.05):
        for i in range(int(seconds / dt)):
            self.step_cars(-seconds + i * dt, dt, "R")
        self.next_spawn = {d: v - seconds if v > 0 else 0.0 for d, v in self.next_spawn.items()}
        self.next_spawn = {+1: 1.0, -1: 3.0}

    # ---- one time step ------------------------------------------------------
    def step(self, t, dt):
        """Advance the world by dt; returns True if the button was pressed."""
        self.press_flag = False
        self.signal = ped_signal(self.fsm.state, self.fsm.cnt, self.fsm.T)
        for p in self.peds:
            p.step(t, dt, self)
        self.step_cars(t, dt, self.fsm.light)
        return self.press_flag

    def snapshot(self, t):
        f = self.fsm
        return dict(
            t=t, state=f.state, cnt=f.cnt, light=f.light,
            signal=ped_signal(f.state, f.cnt, f.T),
            cars=[(c.d, c.s, c.v, c.a, c.color) for lane in self.cars.values() for c in lane],
            peds=[s for s in (p.snapshot(t) for p in self.peds) if s is not None],
            button={sd: t - self.last_press[sd] < 1.0 for sd in (-1, 1)})


def default_pedestrians(T=60, TY=5):
    """Scenario: shows every transition of the EFSM.
       times are when the pedestrian reaches the push button."""
    g1 = T + 1                              # first GREEN starts at t = T + 1
    p_press = g1 + round(0.35 * T)          # GREEN, cnt < T  -> PENDING
    y1 = g1 + 1 + T                         # PENDING -> YELLOW
    r2 = y1 + TY + 1                        # YELLOW  -> RED
    g2 = r2 + T + 1                         # RED     -> GREEN
    late = g2 + T + 7                       # GREEN, cnt == T (stutter) -> YELLOW
    spec = [  # (arrival at button, side, start x, exit dir, speed, crosses)
        (6, -1, -20, +1, 1.35, True),                    # crosses on WALK
        (p_press, -1, -24, +1, 1.30, True),              # presses: GREEN -> PENDING
        (p_press + round(0.25 * T), +1, 22, -1, 1.40, True),   # press ignored (PENDING)
        (p_press + round(0.5 * T), -1, 23, -1, 1.25, True),    # press ignored (PENDING)
        (r2 + round(0.4 * T), +1, -21, +1, 1.45, True),  # arrives on WALK, crosses
        (late, -1, -22, +1, 1.30, True),                 # presses: GREEN -> YELLOW
        (20, +1, -24, +1, 1.40, False),                  # passers-by
        (g1 + 20, -1, 24, -1, 1.30, False),
        (r2 + 10, +1, 24, -1, 1.35, False),
    ]
    peds = []
    for i, (ta, side, x0, ex, v, cross) in enumerate(spec):
        dist = abs(BUTTON_PT[side][0] - x0) if cross else 0.0
        peds.append(Pedestrian(i, ta - dist / v, side, x0, ex, v, cross))
    return peds, late + TY + 22


def simulate(t_green=60, t_yellow=5, t_end=None, substeps=2, seed=7, hz=20):
    """Joint simulation: the EFSM reacts at t = 1, 2, ... (1 Hz); between two
    reactions the world advances with dt = 1/hz and the button presses are
    collected as the btn input of the next reaction."""
    assert hz % substeps == 0
    fsm = TrafficLightEFSM(t_green, t_yellow)
    peds, t_default = default_pedestrians(fsm.T, fsm.TY)
    t_end = t_end or t_default
    world = Crosswalk(fsm, peds, seed)
    world.warmup(30.0)
    frames, reactions, press = [], [], False
    every = hz // substeps
    for i in range(int(round(t_end * hz)) + 1):
        t = i / hz
        if i > 0 and i % hz == 0:
            reactions.append(fsm.react(press))
            press = False
        if i % every == 0:
            frames.append(world.snapshot(t))
        press |= world.step(t, 1.0 / hz)
    return dict(fsm=fsm, reactions=reactions, frames=frames, substeps=substeps, t_end=t_end)


# ===========================================================================
# 3. Animated state diagram (right panel / standalone)
# ===========================================================================
STATE_COL = {"RED": "#ff4d4d", "GREEN": "#22d14a", "PENDING": "#6fe3ea", "YELLOW": "#ffe11a"}
LIGHT_COL = {"R": "#e53935", "G": "#2ecc40", "Y": "#ffd400"}
SIG_COL = {"WALK": "#2ecc40", "FLASH": "#ff9800", "DONT": "#e53935"}
SIG_TXT = {"WALK": "WALK", "FLASH": "DON'T WALK (flashing)", "DONT": "DON'T WALK"}
HL = "#ff7f0e"                               # fired transition


def blend(c, w, a):
    c, w = np.array(to_rgb(c)), np.array(to_rgb(w))
    return tuple(c + (w - c) * a)


class FSMPanel:
    """State diagram + info box + timing diagram, laid out in a figure rect."""
    POS = {"RED": (2.0, 2.75), "GREEN": (5.4, 4.63), "PENDING": (8.8, 2.75),
           "YELLOW": (5.4, 1.05)}
    A, B = 0.93, 0.56
    # geometry of each transition (same order as TrafficLightEFSM.transitions)
    #   edge : (start angle on src, end angle on dst, arc rad, label xy)
    #   loop : ("loop", angle1, angle2, size, label xy)
    GEOM = [
        (80, 180, -0.30, (2.55, 5.05)),             # RED -> GREEN
        ("loop", 220, 140, 1.15, (0.68, 3.75)),     # RED self
        (270, 90, 0.0, (4.4, 2.9)),                 # GREEN -> YELLOW
        (2, 95, -0.30, (8.55, 4.85)),               # GREEN -> PENDING
        ("loop", 60, 120, 1.15, (6.95, 5.95)),      # GREEN self
        (265, 2, -0.25, (8.45, 0.75)),              # PENDING -> YELLOW
        ("loop", -40, 40, 1.15, (10.15, 3.75)),     # PENDING self
        (178, 275, -0.30, (2.65, 0.65)),            # YELLOW -> RED
        ("loop", 300, 240, 1.15, (6.95, -0.05)),    # YELLOW self
    ]

    def __init__(self, fig, rect, sim, title=True):
        self.sim = sim
        self.fsm = sim["fsm"]
        self.R = sim["reactions"]
        x0, y0, w, h = rect
        self.ax = fig.add_axes([x0, y0 + 0.40 * h, w, 0.60 * h])
        self.ax_t = fig.add_axes([x0 + 0.13 * w, y0, 0.84 * w, 0.19 * h])
        self.info = fig.text(x0 + 0.01 * w, y0 + 0.37 * h, "", va="top", ha="left",
                             fontsize=8.5, family="monospace",
                             bbox=dict(boxstyle="round", fc="white", alpha=0.9))
        self._draw_diagram(title)
        self._draw_timeline()

    # ---- static drawing ---------------------------------------------------
    def pt(self, state, ang):
        cx, cy = self.POS[state]
        a = np.radians(ang)
        return np.array([cx + self.A * np.cos(a), cy + self.B * np.sin(a)])

    def _draw_diagram(self, title):
        ax = self.ax
        ax.set_xlim(-0.4, 11.3)
        ax.set_ylim(-0.75, 6.75)
        ax.set_aspect("equal")
        ax.axis("off")
        if title:
            ax.set_title("Extended FSM - traffic light controller", fontsize=11)
        T = self.fsm.T
        ax.text(-0.35, 6.7, f"inputs: btn : {{true, false}}\n"
                f"outputs: light(x), x $\\in$ {{R, G, Y}}\n"
                f"variables: cnt : {{0, ..., {T}}}", va="top", fontsize=8.5)
        self.ell, self.glow, self.names = {}, {}, {}
        for s, (cx, cy) in self.POS.items():
            self.glow[s] = Ellipse((cx, cy), 2 * self.A + 0.35, 2 * self.B + 0.35,
                                   fc=STATE_COL[s], ec="none", alpha=0.0, zorder=1)
            self.ell[s] = Ellipse((cx, cy), 2 * self.A, 2 * self.B, fc="white",
                                  ec="k", lw=1.5, zorder=3)
            ax.add_patch(self.glow[s])
            ax.add_patch(self.ell[s])
            self.names[s] = ax.text(cx, cy, s, ha="center", va="center", fontsize=10,
                                    zorder=5)
        self.arrows, self.labels = [], []
        for tr, g in zip(self.fsm.transitions, self.GEOM):
            if g[0] == "loop":
                _, a1, a2, L, lxy = g
                p1, p2 = self.pt(tr.src, a1), self.pt(tr.dst, a2)
                c = np.array(self.POS[tr.src])
                u1, u2 = unit(p1 - c), unit(p2 - c)
                path = Path([p1, p1 + L * u1 + 0.25 * L * unit(p1 - p2),
                             p2 + L * u2 + 0.25 * L * unit(p2 - p1), p2],
                            [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4])
                arr = FancyArrowPatch(path=path, arrowstyle="-|>", mutation_scale=14,
                                      lw=1.6, color="k", zorder=2)
            else:
                a1, a2, rad, lxy = g
                arr = FancyArrowPatch(self.pt(tr.src, a1), self.pt(tr.dst, a2),
                                      connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>",
                                      mutation_scale=14, lw=1.6, color="k",
                                      shrinkA=0, shrinkB=0, zorder=2)
            ax.add_patch(arr)
            self.arrows.append(arr)
            self.labels.append(ax.text(*lxy, tr.label(), ha="center", va="center",
                                       fontsize=8, linespacing=1.05, zorder=6))
        # initial transition
        self.init_arrow = FancyArrowPatch((1.1, 4.65), self.pt("RED", 105),
                                          connectionstyle="arc3,rad=-0.15", arrowstyle="-|>",
                                          mutation_scale=14, lw=1.6, color="k", zorder=2)
        ax.add_patch(self.init_arrow)
        self.init_label = ax.text(0.9, 4.9, "cnt:=0", ha="center", fontsize=8)

    def _segments(self, values):
        """[(start, width, value)] of piecewise-constant per-second values."""
        out, start = [], 0
        for k in range(1, len(values) + 1):
            if k == len(values) or values[k] != values[start]:
                out.append((start, k - start, values[start]))
                start = k
        return out

    def _draw_timeline(self):
        ax = self.ax_t
        T, t_end = self.fsm.T, self.sim["t_end"]
        st = ["RED"] + [r.dst for r in self.R]
        cn = [0] + [r.cnt_after for r in self.R]
        lt = ["R"] + [r.light for r in self.R]
        sg = [ped_signal(s, c, T) for s, c in zip(st, cn)]
        rows = [(3, st, STATE_COL), (2, lt, LIGHT_COL), (1, sg, SIG_COL)]
        for y, vals, cmap in rows:
            for s0, wd, v in self._segments(vals):
                ax.broken_barh([(s0, wd)], (y - 0.38, 0.76), fc=cmap[v], ec="none")
        for r in self.R:
            if r.btn:
                eff = r.tr >= 0 and "btn" in self.fsm.transitions[r.tr].guard_txt
                ax.vlines(r.k, -0.35, 0.35, color=HL if eff else "0.55",
                          lw=2.5 if eff else 1.5)
        ax.set_xlim(0, t_end)
        ax.set_ylim(-0.6, 3.6)
        ax.set_yticks([0, 1, 2, 3])
        ax.set_yticklabels(["btn", "ped. signal*", "car light", "state"], fontsize=7.5)
        ax.tick_params(axis="x", labelsize=7.5)
        ax.set_xlabel("t [s]", fontsize=8, labelpad=1)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        self.cursor = ax.axvline(0, color="k", lw=1.5)

    # ---- one frame ---------------------------------------------------------
    def update(self, t):
        k = int(np.floor(t + 1e-9))
        k = min(k, len(self.R))
        r = self.R[k - 1] if k >= 1 else None
        state = r.dst if r else "RED"
        cnt = r.cnt_after if r else 0
        light = r.light if r else "R"
        frac = t - k                                  # time since the last reaction

        for s in self.POS:
            act = s == state
            self.ell[s].set_facecolor(STATE_COL[s] if act else blend(STATE_COL[s], "w", 0.72))
            self.ell[s].set_linewidth(3.0 if act else 1.2)
            self.ell[s].set_edgecolor("k" if act else "0.45")
            self.glow[s].set_alpha(0.35 if act else 0.0)
            self.names[s].set_text(f"{s}\ncnt = {cnt}" if act else s)
            self.names[s].set_fontweight("bold" if act else "normal")
            self.names[s].set_color("k" if act else "0.35")

        fired = r.tr if r else None
        for i, (arr, lab) in enumerate(zip(self.arrows, self.labels)):
            on = fired == i and frac < 1.0
            arr.set_color(HL if on else "k")
            arr.set_linewidth(3.2 if on else 1.4)
            lab.set_color(HL if on else "k")
            lab.set_fontweight("bold" if on else "normal")
        init_on = r is None
        self.init_arrow.set_color(HL if init_on else "k")
        self.init_arrow.set_linewidth(3.2 if init_on else 1.4)
        self.init_label.set_color(HL if init_on else "k")

        sig = ped_signal(state, cnt, self.fsm.T)
        lines = [f"t = {t:6.1f} s    reaction k = {k}   (1 Hz)"]
        if r is None:
            lines += ["fired  : initial transition / cnt:=0",
                      "input  : -", "output : -  (car light assumed R)"]
        else:
            lines.append(f"input  : btn = {'true' if r.btn else 'false'}")
            if r.tr < 0:
                lines.append(f"fired  : none -> stutter in {r.src} (cnt = {r.cnt_before})")
            else:
                tr = self.fsm.transitions[r.tr]
                g = tr.guard_txt or ("default" if tr.default else "true")
                lines.append(f"fired  : {r.src} -> {r.dst}   [{g}]")
            lines.append(f"output : {', '.join(r.outputs) if r.outputs else '-'}")
        lines.append(f"state  : {state:8s} cnt = {cnt:2d}   car light = {light}")
        lines.append(f"ped. signal* : {SIG_TXT[sig]}")
        self.info.set_text("\n".join(lines))
        self.cursor.set_xdata([t, t])
        return [*self.ell.values(), *self.glow.values(), *self.names.values(),
                *self.arrows, *self.labels, self.init_arrow, self.init_label,
                self.info, self.cursor]


def unit(v):
    n = np.hypot(*v)
    return v / n if n > 0 else v


FSM_LEGEND = [
    Patch(fc="white", ec="k", lw=3, label="active state"),
    Line2D([], [], color=HL, lw=3, label="transition fired at the last reaction"),
    Line2D([], [], color=HL, lw=2.5, label="btn = true (used by a guard)"),
    Line2D([], [], color="0.55", lw=1.5, label="btn = true (ignored)"),
    Patch(fc=SIG_COL["WALK"], label="WALK*"),
    Patch(fc=SIG_COL["FLASH"], label="DON'T WALK flashing*"),
    Patch(fc=SIG_COL["DONT"], label="DON'T WALK*"),
]
FOOTNOTE = ("* pedestrian signal derived from the extended state "
            "(not an output of the EFSM)")


def _save_or_show(fig, anim, save, fps, dpi):
    if save:
        writer = FFMpegWriter(fps=fps, bitrate=2400) if save.endswith(".mp4") \
            else PillowWriter(fps=fps)
        anim.save(save, writer=writer, dpi=dpi)
        print(f"saved {save}")
    else:
        plt.show()


def animate_fsm(sim, fps=8, save=None, dpi=80):
    """2D animation of the EFSM alone (state diagram + timing diagram)."""
    fig = plt.figure(figsize=(9.2, 9.0))
    panel = FSMPanel(fig, (0.03, 0.11, 0.94, 0.84), sim, title=False)
    fig.suptitle("Extended FSM - traffic light at a pedestrian crosswalk "
                 "(time triggered, 1 reaction per second)", fontsize=12)
    fig.legend(handles=FSM_LEGEND, loc="lower center", ncol=4, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, 0.015))
    fig.text(0.5, 0.005, FOOTNOTE, ha="center", fontsize=7.5, color="0.3")
    n = len(sim["frames"])

    def update(f):
        return panel.update(sim["frames"][min(f, n - 1)]["t"])

    anim = FuncAnimation(fig, update, frames=n + 2 * int(fps), interval=1000 / fps,
                         blit=False, repeat=False)
    _save_or_show(fig, anim, save, fps, dpi)
    return anim


# ===========================================================================
# 4. 3D crosswalk (left) synchronized with the FSM (right)
# ===========================================================================
def box(x0, x1, y0, y1, z0, z1):
    """Faces of an axis-aligned box: bottom, top, -x, +x, -y, +y."""
    return [
        [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)],
        [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],
        [(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)],
        [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)],
        [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],
        [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)],
    ]


def flat(x0, x1, y0, y1, z):
    return [(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)]


def disc(center, normal, r, n=14):
    nx, ny = normal
    tvec = np.array([-ny, nx, 0.0])
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    c = np.asarray(center, float)
    return [tuple(c + r * (np.cos(a) * tvec + np.sin(a) * np.array([0, 0, 1.0]))) for a in ang]


class Crosswalk3D:
    def __init__(self, ax, sim, panel_rect):
        """panel_rect: figure rect (x0, y0, w, h) of the visible panel; the 3D
        axes itself is larger and square (3D axes are always fitted in a square)."""
        self.ax, self.sim, self.panel_rect = ax, sim, panel_rect
        self.T = sim["fsm"].T
        ax.set_xlim(-XL, XL)
        ax.set_ylim(-7.5, 7.5)
        ax.set_zlim(0, 4.5)
        self.elev, self.azim = 27, -70
        ax.set_box_aspect((2 * XL, 15, 4.5), zoom=1.33)
        ax.view_init(elev=self.elev, azim=self.azim)
        ax.set_axis_off()
        fig = ax.figure
        bb = self.panel_rect
        fig.text(bb[0] + bb[2] / 2, bb[1] + bb[3], "Pedestrian crosswalk (3D)",
                 ha="center", va="top", fontsize=11)
        # drawing order follows the depth seen from the camera (south side):
        #   ground < markings < north objects < cars < south objects
        self._static()
        self.wheels = Poly3DCollection([], lw=0.0, facecolor="#151515", zorder=3.9)
        self.cars = Poly3DCollection([], lw=0.25, edgecolor="#222222", zorder=4)
        ax.add_collection3d(self.wheels)
        ax.add_collection3d(self.cars)
        self.lamps = {+1: Poly3DCollection([], lw=0.0, zorder=3.3),
                      -1: Poly3DCollection([], lw=0.0, zorder=5.3)}
        for c in self.lamps.values():
            ax.add_collection3d(c)
        self._lamp_geometry()
        self.cd_pos = [(b[0] + n[0] * 0.25, b[1] + n[1] * 0.25, 3.15)
                       for b, n in self.ped_heads.values()]
        self.legs, self.bodies, self.heads = [], [], []
        for _ in range(12):
            self.legs.append(ax.plot([], [], [], color="#2b2b2b", lw=2.6, zorder=7,
                                     solid_capstyle="round")[0])
            self.bodies.append(ax.plot([], [], [], lw=3.2, zorder=7,
                                       solid_capstyle="round")[0])
            self.heads.append(ax.plot([], [], [], "o", ms=6.5, color="#e0ac7e",
                                      mec="#8a5a3a", mew=0.5, zorder=8)[0])
        self.countdown = [ax.text(*p, "", color="#ffb000", fontsize=7, fontweight="bold",
                                  ha="center", va="center", zorder=9)
                          for p in self.cd_pos]
        self.info = fig.text(bb[0] + 0.01, bb[1] + 0.01, "", fontsize=8.5,
                             family="monospace", va="bottom",
                             bbox=dict(boxstyle="round", fc="white", alpha=0.9))

    # ---- static scene ------------------------------------------------------
    def _static(self):
        ax = self.ax
        layers = {}            # zorder -> (polys, colors)
        cur = [0.0]

        def add(fs, c):
            P, C = layers.setdefault(cur[0], ([], []))
            P.extend(fs)
            C.extend([c] * len(fs))

        add([flat(-XL, XL, -CURB, CURB, 0)], "#45484d")                       # asphalt
        for s in (-1, 1):                                                      # sidewalks
            y0, y1 = sorted((s * CURB, s * 8.0))
            b = box(-XL, XL, y0, y1, 0, SIDE_H)
            add([b[1]], "#cdc8bd")
            add([b[5] if s < 0 else b[4]], "#a8a397")                         # curb face
        cur[0] = 0.5                                                           # markings
        for y in np.arange(-3.25, 3.3, 1.0):                                   # zebra stripes
            add([flat(-CROSS_X, CROSS_X, y, y + 0.5, 0.01)], "#f4f4f4")
        add([flat(-4.4, -4.0, -CURB + 0.1, -0.1, 0.01)], "#f4f4f4")           # stop lines
        add([flat(4.0, 4.4, 0.1, CURB - 0.1, 0.01)], "#f4f4f4")
        for x in np.arange(-XL, XL, 4.0):                                      # lane markings
            if x + 2.2 < -5 or x > 5:
                add([flat(x, x + 2.2, -0.06, 0.06, 0.01)], "#f4f4f4")
        add([flat(-11, -4.5, -0.06, 0.06, 0.01), flat(4.5, 11, -0.06, 0.06, 0.01)],
            "#f4f4f4")
        for y in (-3.3, 3.3):
            add([flat(-XL, XL, y - 0.05, y + 0.05, 0.01)], "#f4f4f4")

        # signal heads: car signals (one per direction) and pedestrian signals
        self.car_heads = {+1: (np.array([-5.6, -4.4]), np.array([-1.0, 0.0])),
                          -1: (np.array([5.6, 4.4]), np.array([1.0, 0.0]))}
        self.ped_heads = {-1: (PED_POLE[-1], np.array([0.0, 1.0])),   # seen from north
                          +1: (PED_POLE[+1], np.array([0.0, -1.0]))}  # seen from south
        for base, _ in self.car_heads.values():
            z = 3.0 if base[1] > 0 else 5.0
            cur[0] = z
            ax.plot([base[0]] * 2, [base[1]] * 2, [SIDE_H, 3.7], color="#5d6166", lw=3,
                    zorder=z - 0.1)
            add(box(base[0] - 0.22, base[0] + 0.22, base[1] - 0.26, base[1] + 0.26, 2.45, 3.75),
                "#1d1f22")
        for base, _ in self.ped_heads.values():
            z = 3.0 if base[1] > 0 else 5.0
            cur[0] = z
            ax.plot([base[0]] * 2, [base[1]] * 2, [SIDE_H, 2.95], color="#5d6166", lw=3,
                    zorder=z - 0.1)
            add(box(base[0] - 0.2, base[0] + 0.2, base[1] - 0.2, base[1] + 0.2, 2.05, 2.95),
                "#1d1f22")
            add(box(base[0] - 0.1, base[0] + 0.1, base[1] - 0.1, base[1] + 0.1, 1.05, 1.38),
                "#f1c40f")                                                     # button box
        for z, (P, C) in sorted(layers.items()):
            ax.add_collection3d(Poly3DCollection(P, facecolors=C, edgecolors="none",
                                                 zorder=z, linewidths=0))
        self.cd_pos = [(b[0] + n[0] * 0.25, b[1] + n[1] * 0.25, 3.15)
                       for b, n in self.ped_heads.values()]

    def _lamp_geometry(self):
        self.lamp_polys = {+1: [], -1: []}
        self.lamp_keys = {+1: [], -1: []}

        def add(base, poly, key):
            sd = 1 if base[1] > 0 else -1
            self.lamp_polys[sd].append(poly)
            self.lamp_keys[sd].append(key)

        # signal heads are double-sided: the lamps are drawn on the face that
        # looks toward the (fixed) camera
        e, a = np.radians(self.elev), np.radians(self.azim)
        cam = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a)])
        vis = lambda n: n if np.dot(n, cam) > 0 else -n
        self.car_heads = {d: (b, vis(n)) for d, (b, n) in self.car_heads.items()}
        self.ped_heads = {d: (b, vis(n)) for d, (b, n) in self.ped_heads.items()}

        for d, (base, n) in self.car_heads.items():
            c0 = np.array([base[0] + n[0] * 0.23, base[1] + n[1] * 0.27])
            for col, z in (("R", 3.42), ("Y", 3.1), ("G", 2.78)):
                add(base, disc((c0[0], c0[1], z), n, 0.15), ("car", col))
        for side, (base, n) in self.ped_heads.items():
            c0 = np.array([base[0] + n[0] * 0.21, base[1] + n[1] * 0.21])
            add(base, disc((c0[0], c0[1], 2.73), n, 0.15), ("ped", "R"))
            add(base, disc((c0[0], c0[1], 2.29), n, 0.15), ("ped", "G"))
            # push-button lamp, facing the crosswalk
            bn = vis(np.array([1.0, 0.0]) if side < 0 else np.array([-1.0, 0.0]))
            add(base, disc((base[0] + bn[0] * 0.11, base[1], 1.22), bn, 0.08, 10), ("btn", side))
        for sd, c in self.lamps.items():
            c.set_verts(self.lamp_polys[sd])

    # ---- dynamic objects ---------------------------------------------------
    def _car_faces(self, d, s, a, color):
        xf = d * (s - ROAD_END)
        xr = xf - d * CAR_L
        x0, x1 = sorted((xf, xr))
        if x1 < -XL or x0 > XL:
            return [], [], []
        y = -1.75 * d
        cx0, cx1 = max(x0, -XL), min(x1, XL)
        faces, cols = [], []
        body = box(cx0, cx1, y - CAR_W / 2, y + CAR_W / 2, 0.3, 1.0)
        brake = a < -0.6
        base = np.array(to_rgb(color))
        shade = [base * 0.6, np.clip(base + 0.25, 0, 1), base * 0.75, base * 0.75,
                 base * 0.85, base * 0.85]
        rear_idx = 2 if d > 0 else 3
        for i, f in enumerate(body):
            faces.append(f)
            if i == rear_idx and x0 >= -XL and x1 <= XL:
                cols.append("#ff1a1a" if brake else "#8b1a1a")
            else:
                cols.append(tuple(shade[i]))
        # cabin
        rc = xr + d * 0.9
        fc_ = xf - d * 1.3
        c0, c1 = sorted((rc, fc_))
        c0, c1 = max(c0, -XL), min(c1, XL)
        if c1 > c0:
            cab = box(c0, c1, y - 0.75, y + 0.75, 1.0, 1.5)
            for i, f in enumerate(cab[1:], 1):
                faces.append(f)
                cols.append(tuple(np.clip(base + 0.2, 0, 1)) if i == 1 else "#a9c3d6")
        wheels = []                         # drawn under the body (separate collection)
        for xa in (xr + d * 0.85, xf - d * 0.85):
            if -XL < xa < XL:
                for yw in (y - CAR_W / 2 + 0.08, y + CAR_W / 2 - 0.08):
                    wheels += box(xa - 0.33, xa + 0.33, yw - 0.12, yw + 0.12, 0.0, 0.6)[2:]
        return faces, cols, wheels

    def _ped_lines(self, p, t):
        x, y = p["pos"]
        z0 = ground_z(y)
        f = np.array([*p["heading"], 0.0])
        lat = np.array([-f[1], f[0], 0.0])
        up = np.array([0, 0, 1.0])
        P = np.array([x, y, z0])
        sw = np.sin(p["phase"]) if p["moving"] else 0.0
        hip = P + up * 0.95
        legs = []
        for sgn in (+1, -1):
            foot = P + f * 0.32 * sw * sgn + lat * 0.11 * sgn
            knee = (hip + foot) / 2 + f * 0.06 + lat * 0.02 * sgn
            legs += [foot, knee, hip + lat * 0.1 * sgn, [np.nan] * 3]
        neck = P + up * 1.5
        body = [hip, neck, [np.nan] * 3]
        for sgn in (+1, -1):
            sh = neck + lat * 0.21 * sgn - up * 0.05
            if p["press"] is not None and sgn == +1:
                hand = np.array([p["press"][0], p["press"][1], 1.2]) + 0.12 * \
                    np.array([*unit(np.array([x, y]) - p["press"]), 0.0])
                elbow = (sh + hand) / 2 - up * 0.08
            else:
                hand = sh - up * 0.62 - f * 0.22 * sw * sgn
                elbow = (sh + hand) / 2 + lat * 0.04 * sgn
            body += [sh, elbow, hand, [np.nan] * 3]
        head = P + up * 1.68
        return np.array(legs), np.array(body), head

    def update(self, fr):
        t = fr["t"]
        faces, cols, wheels = [], [], []
        for d, s, v, a, c in fr["cars"]:
            fs, cs, ws = self._car_faces(d, s, a, c)
            faces += fs
            cols += cs
            wheels += ws
        self.wheels.set_verts(wheels)
        self.cars.set_verts(faces)
        self.cars.set_facecolor(cols if cols else "none")

        light, sig = fr["light"], fr["signal"]
        blink_on = (t % 1.0) < 0.5
        for sd, keys in self.lamp_keys.items():
            lc = []
            for kind, key in keys:
                if kind == "car":
                    on = key == light
                    lc.append(LIGHT_COL[key] if on else blend(LIGHT_COL[key], "#111111", 0.75))
                elif kind == "ped":
                    if key == "G":
                        lc.append("#36ff5a" if sig == "WALK" else "#0f3a18")
                    else:
                        on = sig == "DONT" or (sig == "FLASH" and blink_on)
                        lc.append("#ff3b1f" if on else "#3a120c")
                else:
                    lc.append("#fff35c" if fr["button"][key] else "#7a6a10")
            self.lamps[sd].set_facecolor(lc)

        cd = str(self.T - fr["cnt"]) if fr["state"] == "RED" else ""
        for tx in self.countdown:
            tx.set_text(cd)

        peds = fr["peds"]
        for i in range(len(self.legs)):
            if i < len(peds):
                L, Bd, H = self._ped_lines(peds[i], t)
                self.legs[i].set_data_3d(L[:, 0], L[:, 1], L[:, 2])
                self.bodies[i].set_data_3d(Bd[:, 0], Bd[:, 1], Bd[:, 2])
                self.bodies[i].set_color(peds[i]["color"])
                self.heads[i].set_data_3d([H[0]], [H[1]], [H[2]])
                behind = peds[i]["pos"][1] > 2.7          # north of the cars: drawn behind
                for j, art in enumerate((self.legs[i], self.bodies[i], self.heads[i])):
                    art.set_visible(True)
                    art.set_zorder((3.5 if behind else 6.0) + 0.1 * j)
            else:
                for art in (self.legs[i], self.bodies[i], self.heads[i]):
                    art.set_visible(False)

        visible = [c for c in fr["cars"] if abs(c[0] * (c[1] - ROAD_END)) < XL + CAR_L]
        stopped = sum(1 for c in visible if c[2] < 0.3)
        waiting = sum(1 for p in peds if p["mode"] in ("wait", "press", "to_slot"))
        crossing = sum(1 for p in peds if p["mode"] == "cross")
        btn = any(fr["button"].values())
        self.info.set_text(
            f"car light   : {dict(R='RED', G='GREEN', Y='YELLOW')[light]}\n"
            f"ped. signal*: {SIG_TXT[sig]}\n"
            f"cars        : {len(visible):2d} in view, {stopped} stopped\n"
            f"pedestrians : {waiting} waiting, {crossing} crossing\n"
            f"push button : {'PRESSED -> btn = true' if btn else '-'}")
        return [self.cars, self.wheels, *self.lamps.values(), self.info, *self.countdown,
                *self.legs, *self.bodies, *self.heads]


def animate_crosswalk(sim, fps=8, save=None, dpi=80):
    """3D crosswalk (left) and EFSM (right), driven by the same simulation."""
    fig = plt.figure(figsize=(16, 8.4))
    rect = (0.0, 0.085, 0.565, 0.83)                  # visible 3D panel
    side = rect[2] * fig.get_figwidth()               # square 3D axes, centred on it
    h = side / fig.get_figheight()
    ax3d = fig.add_axes([rect[0], rect[1] + rect[3] / 2 - h / 2, rect[2], h],
                        projection="3d", computed_zorder=False)
    ax3d.patch.set_alpha(0)
    scene = Crosswalk3D(ax3d, sim, rect)
    panel = FSMPanel(fig, (0.57, 0.095, 0.42, 0.815), sim)
    fig.suptitle("Traffic light at a pedestrian crosswalk - 3D scene and extended FSM "
                 "(synchronized, 1 reaction per second)", fontsize=13, y=0.985)
    handles = [Patch(fc=LIGHT_COL["R"], label="car light (light(x) output)"),
               Patch(fc="#fff35c", ec="#7a6a10", label="push button pressed (btn)"),
               *FSM_LEGEND]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, 0.012))
    fig.text(0.5, 0.004, FOOTNOTE, ha="center", fontsize=7.5, color="0.3")
    frames = sim["frames"]
    n = len(frames)

    def update(f):
        fr = frames[min(f, n - 1)]
        return scene.update(fr) + panel.update(fr["t"])

    anim = FuncAnimation(fig, update, frames=n + 2 * int(fps), interval=1000 / fps,
                         blit=False, repeat=False)
    _save_or_show(fig, anim, save, fps, dpi)
    return anim


# ===========================================================================
# CLI
# ===========================================================================
def print_check(sim):
    fsm = sim["fsm"]
    reach = fsm.reachable_states()
    per = {s: sorted(c for st, c in reach if st == s) for s in fsm.STATES}
    print(f"reachable states: {len(reach)}")
    for s, cs in per.items():
        print(f"  {s:8s} {len(cs):3d} values of cnt: {cs[0]}..{cs[-1]}")
    print("reaction trace (only transitions between different states and btn presses):")
    for r in sim["reactions"]:
        if r.src != r.dst or r.btn or r.tr < 0 and (r.k == 1 or sim["reactions"][r.k - 2].tr >= 0):
            tr = "stutter" if r.tr < 0 else f"{r.src} -> {r.dst}"
            print(f"  k={r.k:4d}  btn={int(r.btn)}  cnt {r.cnt_before:2d} -> {r.cnt_after:2d}"
                  f"  {tr:20s} {' '.join(r.outputs)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fsm-only", action="store_true", help="2D FSM animation only")
    ap.add_argument("--check", action="store_true", help="print reachable states and trace")
    ap.add_argument("--t-green", type=int, default=60, help="the 60 s of the slides")
    ap.add_argument("--t-yellow", type=int, default=5, help="the 5 s of the slides")
    ap.add_argument("--t-end", type=float, default=None, help="simulated time [s]")
    ap.add_argument("--substeps", type=int, default=2, choices=[1, 2, 4, 5, 10],
                    help="animation frames per simulated second")
    ap.add_argument("--fps", type=float, default=8, help="playback frame rate")
    ap.add_argument("--seed", type=int, default=7, help="random traffic seed")
    ap.add_argument("--dpi", type=int, default=80)
    ap.add_argument("--save", default=None, help="output .gif or .mp4")
    a = ap.parse_args()
    sim = simulate(a.t_green, a.t_yellow, a.t_end, a.substeps, a.seed)
    if a.check:
        print_check(sim)
    elif a.fsm_only:
        animate_fsm(sim, a.fps, a.save, a.dpi)
    else:
        animate_crosswalk(sim, a.fps, a.save, a.dpi)
