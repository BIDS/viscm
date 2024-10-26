"""BézierBuilder, an interactive Bézier curve explorer.

Copyright (c) 2013, Juan Luis Cano Rodríguez <juanlu001@gmail.com>
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

   * Redistributions of source code must retain the above copyright notice,
     this list of conditions and the following disclaimer.
   * Redistributions in binary form must reproduce the above copyright notice,
     this list of conditions and the following disclaimer in the documentation
     and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER
OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import numpy as np
from matplotlib.backends.qt_compat import QtCore  # type: ignore [attr-defined]
from matplotlib.lines import Line2D

from viscm.bezierbuilder.curve import curve_method
from viscm.minimvc import Trigger

Qt = QtCore.Qt


class ControlPointModel:
    def __init__(self, xp, yp, fixed=None):
        # fixed is either None (if no point is fixed) or and index of a fixed
        # point
        self._xp = list(xp)
        self._yp = list(yp)
        self._fixed = fixed
        self.trigger = Trigger()

    def get_control_points(self):
        return list(self._xp), list(self._yp), self._fixed

    def add_point(self, i, new_x, new_y):
        self._xp.insert(i, new_x)
        self._yp.insert(i, new_y)
        if self._fixed is not None and i <= self._fixed:
            self._fixed += 1
        self.trigger.fire()

    def remove_point(self, i):
        if i == self._fixed:
            return
        del self._xp[i]
        del self._yp[i]
        if self._fixed is not None and i < self._fixed:
            self._fixed -= 1
        self.trigger.fire()

    def move_point(self, i, new_x, new_y):
        if i == self._fixed:
            return
        self._xp[i] = new_x
        self._yp[i] = new_y
        self.trigger.fire()

    def set_control_points(self, xp, yp, fixed=None):
        self._xp = list(xp)
        self._yp = list(yp)
        self._fixed = fixed
        self.trigger.fire()


class ControlPointBuilder:
    def __init__(self, ax, control_point_model):
        self.ax = ax
        self.control_point_model = control_point_model

        self.canvas = self.ax.figure.canvas
        xp, yp, _ = self.control_point_model.get_control_points()
        self.control_polygon = Line2D(
            xp, yp, ls="--", c="#666666", marker="x", mew=2, mec="#204a87"
        )
        self.ax.add_line(self.control_polygon)

        # Event handler for mouse clicking
        self.canvas.mpl_connect("button_press_event", self.on_button_press)
        self.canvas.mpl_connect("button_release_event", self.on_button_release)
        self.canvas.mpl_connect("motion_notify_event", self.on_motion_notify)

        self._index = None  # Active vertex

        self.control_point_model.trigger.add_callback(self._refresh)
        self.mode = "move"
        self._refresh()

    def on_button_press(self, event):
        modkey = event.guiEvent.modifiers()
        # Ignore clicks outside axes
        if event.inaxes != self.ax:
            return
        res, ind = self.control_polygon.contains(event)
        if res and modkey == Qt.KeyboardModifier.NoModifier:
            self._index = ind["ind"][0]
        if res and (
            modkey == Qt.KeyboardModifier.ControlModifier or self.mode == "remove"
        ):
            # Control-click deletes
            self.control_point_model.remove_point(ind["ind"][0])
        if modkey == Qt.KeyboardModifier.ShiftModifier or self.mode == "add":
            # Adding a new point. Find the two closest points and insert it in
            # between them.
            total_squared_dists = []
            xp, yp, _ = self.control_point_model.get_control_points()
            for i in range(len(xp) - 1):
                dist = (event.xdata - xp[i]) ** 2
                dist += (event.ydata - yp[i]) ** 2
                dist += (event.xdata - xp[i + 1]) ** 2
                dist += (event.ydata - yp[i + 1]) ** 2
                total_squared_dists.append(dist)
            best = np.argmin(total_squared_dists)

            self.control_point_model.add_point(best + 1, event.xdata, event.ydata)

    def on_button_release(self, event):
        if event.button != 1:
            return
        self._index = None

    def on_motion_notify(self, event):
        if event.inaxes != self.ax:
            return
        if self._index is None:
            return
        x, y = event.xdata, event.ydata

        self.control_point_model.move_point(self._index, x, y)

    def _refresh(self):
        xp, yp, _ = self.control_point_model.get_control_points()
        self.control_polygon.set_data(xp, yp)

        self.canvas.draw()


################################################################


def compute_bezier_points(xp, yp, at, method, grid=256):
    at = np.asarray(at)
    # The Bezier curve is parameterized by a value t which ranges from 0
    # to 1. However, there is a nonlinear relationship between this value
    # and arclength. We want to parameterize by t', which measures
    # normalized arclength. To do this, we have to calculate the function
    # arclength(t), and then invert it.
    t = np.linspace(0, 1, grid)

    arclength = compute_arc_length(xp, yp, method, t=t)
    arclength /= arclength[-1]
    # Now (t, arclength) is a lookup table describing the t -> arclength
    # mapping. Invert it to get at -> t
    at_t = np.interp(at, arclength, t)
    # And finally look up at the Bezier values at at_t
    # (Might be quicker to np.interp againts x and y, but eh, doesn't
    # really matter.)

    return method(list(zip(xp, yp)), at_t).T


def compute_arc_length(xp, yp, method, t=None, grid=256):
    if t is None:
        t = np.linspace(0, 1, grid)
    x, y = method(list(zip(xp, yp)), t).T
    x_deltas = np.diff(x)
    y_deltas = np.diff(y)
    arclength_deltas = np.empty(len(x))
    if t.size == 0:
        return np.asarray([0])
    arclength_deltas[0] = 0
    np.hypot(x_deltas, y_deltas, out=arclength_deltas[1:])
    return np.cumsum(arclength_deltas)


class SingleBezierCurveModel:
    def __init__(self, control_point_model, method="CatmulClark"):
        self.method = curve_method[method]
        self.control_point_model = control_point_model
        x, y = self.get_bezier_points()
        self.bezier_curve = Line2D(x, y)
        self.trigger = self.control_point_model.trigger
        self.trigger.add_callback(self._refresh)

    def get_bezier_points(self, num=200):
        return self.get_bezier_points_at(np.linspace(0, 1, num))

    def get_bezier_points_at(self, at, grid=1000):
        xp, yp, _ = self.control_point_model.get_control_points()
        return compute_bezier_points(xp, yp, at, self.method, grid=grid)

    def _refresh(self):
        x, y = self.get_bezier_points()
        self.bezier_curve.set_data(x, y)
        # self.canvas.draw()


class TwoBezierCurveModel:
    def __init__(self, control_point_model, method="CatmulClark"):
        self.method = curve_method[method]
        self.control_point_model = control_point_model
        x, y = self.get_bezier_points()
        self.bezier_curve = Line2D(x, y)
        self.trigger = self.control_point_model.trigger
        self.trigger.add_callback(self._refresh)

    def get_bezier_points(self, num=200):
        return self.get_bezier_points_at(np.linspace(0, 1, num))

    def get_bezier_points_at(self, at, grid=256):
        at = np.asarray(at)
        if at.ndim == 0:
            at = np.array([at])

        low_mask = at < 0.5
        high_mask = at >= 0.5

        xp, yp, fixed = self.control_point_model.get_control_points()
        assert fixed is not None

        low_xp = xp[: fixed + 1]
        low_yp = yp[: fixed + 1]
        high_xp = xp[fixed:]
        high_yp = yp[fixed:]

        low_al = compute_arc_length(low_xp, low_yp, self.method).max()
        high_al = compute_arc_length(high_xp, high_yp, self.method).max()

        sf = min(low_al, high_al) / max(low_al, high_al)

        high_at = at[high_mask]
        low_at = at[low_mask]
        if high_al < low_al:
            high_at = high_at * 2 - 1
            low_at = (0.5 - (0.5 - low_at) * sf) * 2
        else:
            high_at = (0.5 + (high_at - 0.5) * sf) * 2 - 1
            low_at = low_at * 2

        low_points = compute_bezier_points(
            low_xp, low_yp, low_at, self.method, grid=grid
        )
        high_points = compute_bezier_points(
            high_xp, high_yp, high_at, self.method, grid=grid
        )
        out = np.concatenate([low_points, high_points], 1)
        return out

    def _refresh(self):
        x, y = self.get_bezier_points()
        self.bezier_curve.set_data(x, y)


class BezierCurveView:
    def __init__(self, ax, bezier_curve_model):
        self.ax = ax
        self.bezier_curve_model = bezier_curve_model

        self.canvas = self.ax.figure.canvas
        x, y = self.bezier_model.get_bezier_points()
        self.bezier_curve = Line2D(x, y)
        self.ax.add_line(self.bezier_curve)

        self.bezier_curve_model.trigger.add_callback(self._refresh)
        self._refresh()

    def _refresh(self):
        x, y = self.bezier_curve_model.get_bezier_points()
        self.bezier_curve.set_data(x, y)
        self.canvas.draw()
