# This file is part of viscm
# Copyright (C) 2015 Nathaniel Smith <njs@pobox.com>
# Copyright (C) 2015 Stefan van der Walt <stefanv@berkeley.edu>
# See file LICENSE.txt for license information.

# Simple script using CIECAM02 and CAM02-UCS to visualize properties of a
# matplotlib colormap

import json
import os.path
import sys

import matplotlib
import matplotlib.colors
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
import numpy as np
from colorspacious import (
    CIECAM02Space,
    CIECAM02Surround,
    cspace_convert,
    cspace_converter,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

# matplotlib.rcParams['backend'] = "QtAgg"
# Do this first before any other matplotlib imports, to force matplotlib to
# use a Qt backend
from matplotlib.backends.qt_compat import (  # type: ignore [attr-defined]
    QtCore,
    QtGui,
    QtWidgets,
)
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec

from viscm.minimvc import Trigger

Qt = QtCore.Qt


# The correct L_A value for the standard sRGB viewing conditions is:
#   (64 / np.pi) / 5
# Due to an error in our color conversion code, the matplotlib colormaps were
# designed using the assumption that they would be viewed with an L_A value of
#   (64 / np.pi) * 5
# (i.e., 125x brighter ambient illumination than appropriate). It turns out
# that when all is said and done this has negligible effect on the uniformity
# of the resulting colormaps (phew), BUT fixing the bug has the effect of
# somewhat shrinking the sRGB color solid as projected into CAM02-UCS
# space. This means that the bezier points for existing colormaps (like the
# matplotlib ones) are in the wrong place. We can reproduce the original
# colormaps from these points by using this buggy_CAM02UCS space as our
# uniform space:
buggy_sRGB_viewing_conditions = CIECAM02Space(
    XYZ100_w="D65",
    Y_b=20,
    L_A=(64 / np.pi) * 5,  # bug: should be / 5
    surround=CIECAM02Surround.AVERAGE,
)
buggy_CAM02UCS = {
    "name": "CAM02-UCS",
    "ciecam02_space": buggy_sRGB_viewing_conditions,
}

GREYSCALE_CONVERSION_SPACE = "JCh"

_sRGB1_to_JCh = cspace_converter("sRGB1", GREYSCALE_CONVERSION_SPACE)
_JCh_to_sRGB1 = cspace_converter(GREYSCALE_CONVERSION_SPACE, "sRGB1")


def to_greyscale(sRGB1):
    JCh = _sRGB1_to_JCh(sRGB1)
    JCh[..., 1] = 0
    return np.clip(_JCh_to_sRGB1(JCh), 0, 1)


_deuter50_space = {"name": "sRGB1+CVD", "cvd_type": "deuteranomaly", "severity": 50}
_deuter50_to_sRGB1 = cspace_converter(_deuter50_space, "sRGB1")
_deuter100_space = {"name": "sRGB1+CVD", "cvd_type": "deuteranomaly", "severity": 100}
_deuter100_to_sRGB1 = cspace_converter(_deuter100_space, "sRGB1")
_prot50_space = {"name": "sRGB1+CVD", "cvd_type": "protanomaly", "severity": 50}
_prot50_to_sRGB1 = cspace_converter(_prot50_space, "sRGB1")
_prot100_space = {"name": "sRGB1+CVD", "cvd_type": "protanomaly", "severity": 100}
_prot100_to_sRGB1 = cspace_converter(_prot100_space, "sRGB1")


def _show_cmap(ax, rgb):
    ax.imshow(rgb[np.newaxis, ...], aspect="auto")


def _apply_rgb_mat(mat, rgb):
    return np.clip(np.einsum("...ij,...j->...i", mat, rgb), 0, 1)


# sRGB corners: a' goes from -37.4 to 45
AP_LIM = (-38, 46)
# b' goes from -46.5 to 42
BP_LIM = (-47, 43)
# J'/K goes from 0 to 100
JP_LIM = (-1, 101)


def _setup_Jpapbp_axis(ax):
    ax.set_xlabel("a' (green -> red)")
    ax.set_ylabel("b' (blue -> yellow)")
    ax.set_zlabel("J'/K (white -> black)")
    ax.set_xlim(*AP_LIM)
    ax.set_ylim(*BP_LIM)
    ax.set_zlim(*JP_LIM)


# Adapt a matplotlib colormap to a linearly transformed version -- useful for
# visualizing how colormaps look given color deficiency.
# Kinda a hack, b/c we inherit from Colormap (this is required), but then
# ignore its implementation entirely. This results in errors at runtime:
#       File "/<env>/site-packages/matplotlib/artist.py", line 1343, in format_cursor_data  # noqa: E501
#         n = self.cmap.N
#             ^^^^^^^^^^^
#     AttributeError: 'TransformedCMap' object has no attribute 'N'
class TransformedCMap(matplotlib.colors.Colormap):
    def __init__(self, transform, base_cmap):
        self.transform = transform
        self.base_cmap = base_cmap

    def __call__(self, *args, **kwargs):
        bts = kwargs.pop("bytes", False)
        fx = self.base_cmap(*args, bytes=False, **kwargs)
        tfx = self.transform(fx)
        if bts:
            return (tfx * 255).astype("uint8")
        return tfx

    def set_bad(self, *args, **kwargs):
        self.base_cmap.set_bad(*args, **kwargs)

    def set_under(self, *args, **kwargs):
        self.base_cmap.set_under(*args, **kwargs)

    def set_over(self, *args, **kwargs):
        self.base_cmap.set_over(*args, **kwargs)

    def is_gray(self):
        return False


def _vis_axes(fig):
    grid = GridSpec(
        10,
        4,
        left=0.02,
        right=0.98,
        bottom=0.02,
        width_ratios=[1] * 4,
        height_ratios=[1] * 10,
    )
    axes = {
        "cmap": grid[0, 0],
        "deltas": grid[1:4, 0],
        "cmap-greyscale": grid[0, 1],
        "lightness-deltas": grid[1:4, 1],
        "deuteranomaly": grid[4, 0],
        "deuteranopia": grid[5, 0],
        "protanomaly": grid[4, 1],
        "protanopia": grid[5, 1],
        # 'lightness': grid[4:6, 1],
        # 'colourfulness': grid[4:6, 2],
        # 'hue': grid[4:6, 3],
        "image0": grid[0:3, 2],
        "image0-cb": grid[0:3, 3],
        "image1": grid[3:6, 2],
        "image1-cb": grid[3:6, 3],
        "image2": grid[6:8, 2:],
        "image2-cb": grid[8:, 2:],
    }

    axes = {key: fig.add_subplot(value) for (key, value) in axes.items()}
    axes["gamut"] = fig.add_subplot(grid[6:, :2], projection="3d")
    return axes


def lookup_colormap_by_name(name):
    try:
        return plt.get_cmap(name)
    except ValueError:
        pass
    # Try expanding a setuptools-style entrypoint:
    #   foo.bar:baz.quux
    #   -> import foo.bar; return foo.bar.baz.quux
    if ":" in name:
        module_name, object_name = name.split(":", 1)
        object_path = object_name.split(".")
        import importlib

        cm = importlib.import_module(module_name)
        for entry in object_path:
            cm = getattr(cm, entry)
        return cm
    raise ValueError(f"Can't find colormap {name!r}")


class viscm:
    def __init__(
        self,
        cm,
        figure=None,
        uniform_space="CAM02-UCS",
        name=None,
        N=256,
        N_dots=50,
        show_gamut=False,
    ):
        if isinstance(cm, str):
            cm = lookup_colormap_by_name(cm)
        if name is None:
            name = cm.name
        if figure is None:
            figure = plt.figure()
        self._sRGB1_to_uniform = cspace_converter("sRGB1", uniform_space)

        self.figure = figure
        self.figure.suptitle(f"Colormap evaluation: {name}", fontsize=24)

        axes = _vis_axes(self.figure)

        # ListedColormap is used for many matplotlib builtin colormaps
        # (e.g. viridis) and also what we use in the editor. It's the most
        # efficient way to work with arbitrary smooth colormaps -- pick enough
        # points that it looks smooth, and then don't waste time interpolating
        # between them. But then it creates weird issues in the analyzer if
        # our N doesn't match their N, especially when we try to compute the
        # derivative. (Specifically the derivative oscillates between 0 and a
        # large value depending on where our sample points happen to fall
        # relative to the cutoffs between the ListedColormap samples.) So if
        # this is a smooth (large N) ListedColormap, then just use its samples
        # directly:
        if isinstance(cm, ListedColormap) and cm.N >= 100:
            RGB = np.asarray(cm.colors)[:, :3]
            N = RGB.shape[0]
            x = np.linspace(0, 1, N)
        else:
            x = np.linspace(0, 1, N)
            RGB = cm(x)[:, :3]
        x_dots = np.linspace(0, 1, N_dots)
        RGB_dots = cm(x_dots)[:, :3]

        ax = axes["cmap"]
        _show_cmap(ax, RGB)
        ax.set_title("The colormap in its glory")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        def label(ax, s):
            ax.text(
                0.95,
                0.05,
                s,
                horizontalalignment="right",
                verticalalignment="bottom",
                transform=ax.transAxes,
            )

        def title(ax, s):
            ax.text(
                0.98,
                0.98,
                s,
                horizontalalignment="right",
                verticalalignment="top",
                transform=ax.transAxes,
            )

        Jpapbp = self._sRGB1_to_uniform(RGB)

        def delta_ymax(values):
            return max(np.max(values) * 1.1, 0)

        ax = axes["deltas"]
        local_deltas = np.sqrt(np.sum((Jpapbp[:-1, :] - Jpapbp[1:, :]) ** 2, axis=-1))
        local_derivs = N * local_deltas
        ax.plot(x[1:], local_derivs)
        arclength = np.sum(local_deltas)
        rmse = np.std(local_derivs)
        title(ax, "Perceptual derivative")
        label(
            ax,
            f"Length: {arclength:0.1f}\nRMS deviation from flat: "
            f"{rmse:0.1f} ({100 * rmse / arclength:0.1f}%)",
        )
        ax.set_ylim(-delta_ymax(-local_derivs), delta_ymax(local_derivs))
        ax.get_xaxis().set_visible(False)

        ax = axes["cmap-greyscale"]
        _show_cmap(ax, to_greyscale(RGB))
        ax.set_title("Black-and-white printed")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = axes["lightness-deltas"]
        ax.axhline(0, linestyle="--", color="grey")
        lightness_deltas = np.diff(Jpapbp[:, 0])
        lightness_derivs = N * lightness_deltas

        ax.plot(x[1:], lightness_derivs)
        title(ax, "Perceptual lightness derivative")
        lightness_arclength = np.sum(np.abs(lightness_deltas))
        lightness_rmse = np.std(lightness_derivs)
        label(
            ax,
            f"Length: {lightness_arclength:0.1f}\nRMS deviation from flat: "
            f"{lightness_rmse:0.1f} "
            f"({100 * lightness_rmse / lightness_arclength:0.1f}%)",
        )

        ax.set_ylim(-delta_ymax(-lightness_derivs), delta_ymax(lightness_derivs))
        ax.get_xaxis().set_visible(False)

        # ax = axes['lightness']
        # ax.plot(x, ciecam02.J)
        # label(ax, "Lightness (J)")
        # ax.set_ylim(0, 105)

        # ax = axes['colourfulness']
        # ax.plot(x, ciecam02.M)
        # label(ax, "Colourfulness (M)")

        # ax = axes['hue']
        # ax.plot(x, ciecam02.h)
        # label(ax, "Hue angle (h)")
        # ax.set_ylim(0, 360)

        def anom(ax, converter, name):
            _show_cmap(ax, np.clip(converter(RGB), 0, 1))
            label(ax, name)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        anom(axes["deuteranomaly"], _deuter50_to_sRGB1, "Moderate deuteranomaly")
        anom(axes["deuteranopia"], _deuter100_to_sRGB1, "Complete deuteranopia")

        anom(axes["protanomaly"], _prot50_to_sRGB1, "Moderate protanomaly")
        anom(axes["protanopia"], _prot100_to_sRGB1, "Complete protanopia")

        ax = axes["gamut"]
        ax.plot(Jpapbp[:, 1], Jpapbp[:, 2], Jpapbp[:, 0])
        Jpapbp_dots = self._sRGB1_to_uniform(RGB_dots)
        ax.scatter(
            Jpapbp_dots[:, 1],
            Jpapbp_dots[:, 2],
            Jpapbp_dots[:, 0],
            c=RGB_dots[:, :],
            s=80,
        )

        # Draw a wireframe indicating the sRGB gamut
        self.gamut_patch = sRGB_gamut_patch(uniform_space)
        # That function returns a patch where each face is colored to match
        # the represented colors. For present purposes we want something
        # less... colorful.
        self.gamut_patch.set_facecolor([0.5, 0.5, 0.5, 0.1])
        self.gamut_patch.set_edgecolor([0.2, 0.2, 0.2, 0.1])
        ax.add_collection3d(self.gamut_patch)
        self.gamut_patch.set_visible(show_gamut)

        ax.view_init(elev=75, azim=-75)

        _setup_Jpapbp_axis(ax)

        images = []
        image_args = []
        example_dir = os.path.join(os.path.dirname(__file__), "examples")

        images.append(
            np.loadtxt(os.path.join(example_dir, "st-helens_before-modified.txt.gz")).T
        )
        image_args.append({})

        # Adapted from
        #   http://matplotlib.org/mpl_examples/images_contours_and_fields/pcolormesh_levels.py
        dx = dy = 0.05
        y, x = np.mgrid[-5 : 5 + dy : dy, -5 : 10 + dx : dx]
        z = np.sin(x) ** 10 + np.cos(10 + y * x) + np.cos(x) + 0.2 * y + 0.1 * x
        images.append(z)
        image_args.append({})

        # Peter Kovesi's colormap test image at
        #   http://peterkovesi.com/projects/colourmaps/colourmaptest.tif

        images.append(np.load(os.path.join(example_dir, "colourmaptest.npy")))

        image_args.append({})

        def _deuter_transform(RGBA):
            # clipping, alpha handling
            RGB = RGBA[..., :3]
            RGB = np.clip(_deuter50_to_sRGB1(RGB), 0, 1)
            return np.concatenate((RGB, RGBA[..., 3:]), axis=-1)

        deuter_cm = TransformedCMap(_deuter_transform, cm)

        for i, (image, args) in enumerate(zip(images, image_args)):
            ax = axes["image%i" % (i,)]
            ax.imshow(image, cmap=cm, **args)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            ax_cb = axes["image%i-cb" % (i,)]
            ax_cb.imshow(image, cmap=deuter_cm, **args)
            ax_cb.get_xaxis().set_visible(False)
            ax_cb.get_yaxis().set_visible(False)

        axes["image0"].set_title("Sample images")
        axes["image0-cb"].set_title("Moderate deuter.")
        self.axes = axes

    def toggle_gamut(self):
        self.gamut_patch.set_visible(not self.gamut_patch.get_visible())

    def save_figure(self, path):
        self.figure.savefig(path)


def sRGB_gamut_patch(uniform_space, resolution=20):
    step = 1.0 / resolution
    sRGB_quads = []
    sRGB_values = []
    # each entry in 'quads' is a 4x3 array where each row contains the
    # coordinates of a corner point
    for fixed in 0, 1:
        for i in range(resolution):
            for j in range(resolution):
                # R quad
                sRGB_quads.append(
                    [
                        [fixed, i * step, j * step],
                        [fixed, (i + 1) * step, j * step],
                        [fixed, (i + 1) * step, (j + 1) * step],
                        [fixed, i * step, (j + 1) * step],
                    ]
                )
                sRGB_values.append((fixed, (i + 0.5) * step, (j + 0.5) * step, 1))
                # G quad
                sRGB_quads.append(
                    [
                        [i * step, fixed, j * step],
                        [(i + 1) * step, fixed, j * step],
                        [(i + 1) * step, fixed, (j + 1) * step],
                        [i * step, fixed, (j + 1) * step],
                    ]
                )
                sRGB_values.append(((i + 0.5) * step, fixed, (j + 0.5) * step, 1))
                # B quad
                sRGB_quads.append(
                    [
                        [i * step, j * step, fixed],
                        [(i + 1) * step, j * step, fixed],
                        [(i + 1) * step, (j + 1) * step, fixed],
                        [i * step, (j + 1) * step, fixed],
                    ]
                )
                sRGB_values.append(((i + 0.5) * step, (j + 0.5) * step, fixed, 1))
    sRGB_quads = np.asarray(sRGB_quads)
    # work around colorspace transform bugginess in handling high-dim
    # arrays
    sRGB_quads_2d = sRGB_quads.reshape((-1, 3))
    Jpapbp_quads_2d = cspace_convert(sRGB_quads_2d, "sRGB1", uniform_space)
    Jpapbp_quads = Jpapbp_quads_2d.reshape((-1, 4, 3))
    gamut_patch = mpl_toolkits.mplot3d.art3d.Poly3DCollection(
        Jpapbp_quads[:, :, [1, 2, 0]]
    )
    gamut_patch.set_facecolor(sRGB_values)
    gamut_patch.set_edgecolor(sRGB_values)
    return gamut_patch


def sRGB_gamut_Jp_slice(
    Jp, uniform_space, ap_lim=(-50, 50), bp_lim=(-50, 50), resolution=200
):
    bp_grid, ap_grid = np.mgrid[
        bp_lim[0] : bp_lim[1] : resolution * 1j, ap_lim[0] : ap_lim[1] : resolution * 1j
    ]
    Jp_grid = Jp * np.ones((resolution, resolution))
    Jpapbp = np.concatenate(
        (
            Jp_grid[:, :, np.newaxis],
            ap_grid[:, :, np.newaxis],
            bp_grid[:, :, np.newaxis],
        ),
        axis=2,
    )
    sRGB = cspace_convert(Jpapbp, uniform_space, "sRGB1")
    sRGBA = np.concatenate((sRGB, np.ones(sRGB.shape[:2] + (1,))), axis=2)
    sRGBA[np.any((sRGB < 0) | (sRGB > 1), axis=-1)] = [0, 0, 0, 0]
    return sRGBA


def draw_pure_hue_angles(ax):
    # Pure hue angles from CIECAM-02
    for color, angle in [("r", 20.14), ("y", 90.00), ("g", 164.25), ("b", 237.53)]:
        x = np.cos(np.deg2rad(angle))
        y = np.sin(np.deg2rad(angle))
        ax.plot([0, x * 1000], [0, y * 1000], color + "--")


def draw_sRGB_gamut_Jp_slice(
    ax, Jp, uniform_space, ap_lim=(-50, 50), bp_lim=(-50, 50), **kwargs
):
    sRGB = sRGB_gamut_Jp_slice(
        Jp, uniform_space, ap_lim=ap_lim, bp_lim=bp_lim, **kwargs
    )
    im = ax.imshow(sRGB, aspect="equal", extent=ap_lim + bp_lim, origin="lower")
    draw_pure_hue_angles(ax)
    ax.set_xlim(ap_lim)
    ax.set_ylim(bp_lim)
    return im


# def sRGB_gamut_J_slice(J,
#                        ap_lim=(-50, 50), bp_lim=(-50, 50), resolution=200):
#     a_grid, b_grid = np.mgrid[ap_lim[0] : ap_lim[1] : resolution * 1j,
#                               bp_lim[0] : bp_lim[1] : resolution * 1j]
#     J_grid = J * np.ones((resolution, resolution))
#     h = np.rad2deg(np.arctan2(b_grid, a_grid))
#     M = np.hypot(a_grid, b_grid)
#     XYZ = ViewingConditions.sRGB.CIECAM02_to_XYZ(J=J_grid, M=M, h=h)
#     sRGB = XYZ_to_sRGB(XYZ)
#     sRGB[np.any((sRGB < 0) | (sRGB > 1), axis=-1)] = np.nan
#     return sRGB


def _viscm_editor_axes(fig):
    grid = GridSpec(1, 2, width_ratios=[9, 1], height_ratios=[50])
    axes = {"bezier": grid[0, 0], "cm": grid[0, 1]}

    axes = {key: fig.add_subplot(value) for (key, value) in axes.items()}
    return axes


class viscm_editor:
    def __init__(
        self,
        figure=None,
        uniform_space="CAM02-UCS",
        min_Jp=15,
        max_Jp=95,
        xp=None,
        yp=None,
        cmtype="linear",
        filter_k=100,
        fixed=-1,
        name="new cm",
        method="CatmulClark",
    ):
        from .bezierbuilder import (
            ControlPointBuilder,
            ControlPointModel,
            SingleBezierCurveModel,
            TwoBezierCurveModel,
        )

        if figure is None:
            figure = plt.figure()
        self.cmtype = cmtype
        self.method = method
        self._uniform_space = uniform_space
        self.name = name
        self.figure = figure
        axes = _viscm_editor_axes(self.figure)
        self.min_Jp = min_Jp
        self.max_Jp = max_Jp
        self.fixed = fixed
        if self.cmtype in ["diverging", "diverging-continuous"] and xp is None:
            self.fixed = 4
        if xp is None or yp is None:
            if method == "Bezier":
                xp = {
                    "linear": [
                        -2.0591553836234482,
                        59.377014829142524,
                        43.552546744036135,
                        4.7670857511283202,
                        -9.5059638942617539,
                    ],
                    "diverging": [-9, -15, 43, 30, 0, -20, -30, 20, 1],
                    "diverging-continuous": [-9, -15, 43, 30, 0, -20, -30, 20, 1],
                }[cmtype]
                yp = {
                    "linear": [
                        -25.664893617021221,
                        -21.941489361702082,
                        38.874113475177353,
                        20.567375886524871,
                        32.047872340425585,
                    ],
                    "diverging": [-5, 20, 20, -21, 0, 21, -38, -20, -5],
                    "diverging-continuous": [-5, 20, 20, -21, 0, 21, -38, -20, -5],
                }[cmtype]
            if method == "CatmulClark":
                xp = {
                    "linear": [-2, 20, 23, 5, -9],
                    "diverging": [-9, -15, -10, 0, 0, 5, 10, 15, 2],
                    "diverging-continuous": [-9, -5, -1, 0, 0, 5, 10, 15, 2],
                }[cmtype]
                yp = {
                    "linear": [-25, -21, 18, 10, 12],
                    "diverging": [-5, -8, -20, -10, 0, 2, 8, 15, 5],
                    "diverging-continuous": [-5, -8, -20, -10, 0, 2, 8, 15, 5],
                }[cmtype]
        xy_lim = {"Bezier": (-100, 100), "CatmulClark": (-50, 50)}[self.method]

        BezierModel, startJp = {
            "linear": (SingleBezierCurveModel, 0.5),
            "diverging": (TwoBezierCurveModel, 0.75),
            "diverging-continuous": (TwoBezierCurveModel, 0.5),
        }[cmtype]

        self.control_point_model = ControlPointModel(xp, yp, fixed=self.fixed)
        self.bezier_model = BezierModel(self.control_point_model, self.method)
        axes["bezier"].add_line(self.bezier_model.bezier_curve)
        self.cmap_model = BezierCMapModel(
            self.bezier_model,
            self.min_Jp,
            self.max_Jp,
            uniform_space,
            cmtype=cmtype,
            filter_k=filter_k,
        )

        self.highlight_point_model = HighlightPointModel(self.cmap_model, startJp)
        self.highlight_point_model1 = None

        self.bezier_builder = ControlPointBuilder(
            axes["bezier"], self.control_point_model
        )

        self.bezier_gamut_viewer = GamutViewer2D(
            axes["bezier"],
            self.highlight_point_model,
            uniform_space,
        )

        self.bezier_highlight_point_view = HighlightPoint2DView(
            axes["bezier"], self.highlight_point_model
        )
        if cmtype == "diverging":
            self.highlight_point_model1 = HighlightPointModel(
                self.cmap_model, 1 - startJp
            )
            self.bezier_highlight_point_view1 = HighlightPoint2DView(
                axes["bezier"], self.highlight_point_model1
            )

        # draw_pure_hue_angles(axes['bezier'])
        axes["bezier"].set_xlim(*xy_lim)
        axes["bezier"].set_ylim(*xy_lim)

        self.cmap_view = CMapView(axes["cm"], self.cmap_model)
        self.cmap_highlighter = HighlightPointBuilder(
            axes["cm"], self.highlight_point_model, self.highlight_point_model1
        )
        self.axes = axes

    def save_colormap(self, filepath):
        with open(filepath, "w") as f:
            xp, yp, fixed = self.control_point_model.get_control_points()
            rgb, _ = self.cmap_model.get_sRGB(num=256)
            hex_blob = ""
            for color in rgb:
                for component in color:
                    hex_blob += f"{int(round(component * 255)):02x}"
            usage_hints = ["red-green-colorblind-safe", "greyscale-safe"]
            if self.cmtype == "diverging":
                usage_hints.append("diverging")
            elif self.cmtype == "linear":
                usage_hints.append("sequential")
            xp, yp, fixed = self.control_point_model.get_control_points()
            extensions = {
                "min_Jp": self.min_Jp,
                "max_Jp": self.max_Jp,
                "xp": xp,
                "yp": yp,
                "fixed": fixed,
                "filter_k": self.cmap_model.filter_k,
                "cmtype": self.cmtype,
                "uniform_colorspace": self._uniform_space,
                "spline_method": self.method,
            }
            json.dump(
                {
                    "content-type": "application/vnd.matplotlib.colormap-v1+json",
                    "name": self.name,
                    "license": "http://creativecommons.org/publicdomain/zero/1.0/",
                    "usage-hints": usage_hints,
                    "colorspace": "sRGB",
                    "domain": "continuous",
                    "colors": hex_blob,
                    "extensions": {"https://matplotlib.org/viscm": extensions},
                },
                f,
                indent=4,
            )
        print("Saved")

    def export_py(self, filepath):
        import textwrap

        template = textwrap.dedent(
            """
        from matplotlib.colors import ListedColormap

        cm_type = "{type}"

        cm_data = {array_list}
        test_cm = ListedColormap(cm_data, name="{name}")
        """
        )
        rgb, _ = self.cmap_model.get_sRGB(num=256)
        array_list = np.array2string(
            rgb, max_line_width=78, prefix="cm_data = ", separator=","
        )
        with open(filepath, "w") as f:
            f.write(
                template.format(
                    **{"array_list": array_list, "type": self.cmtype, "name": self.name}
                )
            )

    def show_viscm(self):
        cm = ListedColormap(self.cmap_model.get_sRGB(num=256)[0], name=self.name)

        return cm

    def _jp_update(self, minval, maxval):
        if minval >= 0 and minval <= 100 and maxval >= 0 and maxval <= 100:
            self.min_Jp = minval
            self.max_Jp = maxval
            self.cmap_model.set_Jp_minmax(self.min_Jp, self.max_Jp)

    def _filter_k_update(self, filter_k):
        self.cmap_model.set_filter_k(filter_k)


class BezierCMapModel:
    def __init__(
        self, bezier_model, min_Jp, max_Jp, uniform_space, filter_k=100, cmtype="linear"
    ):
        self.bezier_model = bezier_model
        self.min_Jp = min_Jp
        self.max_Jp = max_Jp
        self.filter_k = filter_k
        self.cmtype = cmtype
        self.trigger = Trigger()
        self.Jp_minmax_trigger = Trigger()
        self.Jp_minmax_trigger.add_callback(self.trigger.fire)
        self.filter_k_trigger = Trigger()
        self.filter_k_trigger.add_callback(self.trigger.fire)
        self.uniform_to_sRGB1 = cspace_converter(uniform_space, "sRGB1")
        self.bezier_model.trigger.add_callback(self.trigger.fire)

    def set_Jp_minmax(self, min_Jp, max_Jp):
        self.min_Jp = min_Jp
        self.max_Jp = max_Jp
        self.Jp_minmax_trigger.fire()

    def set_filter_k(self, filter_k):
        self.filter_k = filter_k
        self.filter_k_trigger.fire()

    def get_Jpapbp_at_point(self, point):
        from scipy.interpolate import interp1d

        Jp, ap, bp = self.get_Jpapbp()
        Jp, ap, bp = interp1d(np.linspace(0, 1, Jp.size), np.array([Jp, ap, bp]))(point)
        return Jp, ap, bp

    def get_Jpapbp(self, num=200):
        ap, bp = self.bezier_model.get_bezier_points_at(np.linspace(0, 1, num))
        at = np.linspace(0, 1, num)
        if self.cmtype == "diverging":
            from scipy.special import erf

            at = 1 + 2 * np.cumsum(erf(self.filter_k * (at - 0.5))) / num
        Jp = (self.max_Jp - self.min_Jp) * at + self.min_Jp
        return Jp, ap, bp

    def get_sRGB(self, num=200):
        # Return sRGB and out-of-gamut mask
        Jp, ap, bp = self.get_Jpapbp(num=num)
        sRGB = self.uniform_to_sRGB1(np.column_stack((Jp, ap, bp)))
        oog = np.any((sRGB > 1) | (sRGB < 0), axis=-1)
        sRGB[oog, :] = np.nan
        return sRGB, oog


class CMapView:
    def __init__(self, ax, cmap_model):
        self.ax = ax
        self.cmap_model = cmap_model

        rgb_display, oog_display = self._drawable_arrays()
        self.image = self.ax.imshow(rgb_display, extent=(0, 0.2, 0, 1), origin="lower")
        self.gamut_alert_image = self.ax.imshow(
            oog_display, extent=(0.05, 0.15, 0, 1), origin="lower"
        )
        self.ax.set_xlim(0, 0.2)
        self.ax.set_ylim(0, 1)
        self.ax.get_xaxis().set_visible(False)

        self.cmap_model.trigger.add_callback(self._refresh)

    def _drawable_arrays(self):
        rgb, oog = self.cmap_model.get_sRGB()
        rgb_display = rgb[:, np.newaxis, :]
        oog_display = np.empty((rgb.shape[0], 1, 4))
        oog_display[...] = [0, 0, 0, 0]
        oog_display[oog, :, :] = [0, 1, 1, 1]
        return rgb_display, oog_display

    def _refresh(self):
        rgb_display, oog_display = self._drawable_arrays()
        self.image.set_data(rgb_display)
        self.gamut_alert_image.set_data(oog_display)


class HighlightPointModel:
    def __init__(self, cmap_model, point):
        self._cmap_model = cmap_model
        self._point = point
        self.trigger = Trigger()

        self._cmap_model.trigger.add_callback(self.trigger.fire)

    def get_point(self):
        return self._point

    def set_point(self, point):
        self._point = point
        self.trigger.fire()

    def get_Jpapbp(self):
        return self._cmap_model.get_Jpapbp_at_point(self._point)


class HighlightPointBuilder:
    def __init__(self, ax, highlight_point_model_a, highlight_point_model_b):
        self.ax = ax
        self.highlight_point_model_b = highlight_point_model_b
        self.highlight_point_model_a = highlight_point_model_a

        self.canvas = self.ax.figure.canvas
        self._in_drag = False

        self.marker_line_a = self.ax.axhline(
            highlight_point_model_a.get_point(), linewidth=3, color="r"
        )
        if self.highlight_point_model_b:
            self.marker_line_b = self.ax.axhline(
                highlight_point_model_b.get_point(), linewidth=3, color="r"
            )

        self.canvas.mpl_connect("button_press_event", self._on_button_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("button_release_event", self._on_button_release)

        self.highlight_point_model_a.trigger.add_callback(self._refresh)
        if highlight_point_model_b:
            self.highlight_point_model_a.trigger.add_callback(self._refresh)

    def _on_button_press(self, event):
        if event.inaxes != self.ax:
            return
        if event.button != 1:
            return
        self._in_drag = True
        self.highlight_point_model_a.set_point(event.ydata)
        if self.highlight_point_model_b:
            self.highlight_point_model_b.set_point(1 - event.ydata)

    def _on_motion(self, event):
        if self._in_drag and event.ydata is not None:
            self.highlight_point_model_a.set_point(event.ydata)
            if self.highlight_point_model_b:
                self.highlight_point_model_b.set_point(1 - event.ydata)

    def _on_button_release(self, event):
        if event.button != 1:
            return
        self._in_drag = False

    def _refresh(self):
        point = self.highlight_point_model_a.get_point()
        self.marker_line_a.set_data([0, 1], [point, point])
        if self.highlight_point_model_b:
            self.marker_line_b.set_data([0, 1], [1 - point, 1 - point])
        self.canvas.draw()


class GamutViewer2D:
    def __init__(
        self,
        ax,
        highlight_point_model,
        uniform_space,
        ap_lim=(-50, 50),
        bp_lim=(-50, 50),
    ):
        self.ax = ax
        self.highlight_point_model = highlight_point_model
        self.ap_lim = ap_lim
        self.bp_lim = bp_lim
        self.uniform_space = uniform_space

        self.bgcolors = {"light": (0.9, 0.9, 0.9), "dark": (0.1, 0.1, 0.1)}
        # We want some hysteresis, so that there's no point where wiggling the
        # line back and forth causes background flickering.
        self.bgcolor_ranges = {"light": (0, 60), "dark": (40, 100)}
        self.bg_opposites = {"light": "dark", "dark": "light"}
        self.bg = "light"
        self.ax.set_facecolor(self.bgcolors[self.bg])

        self.image = self.ax.imshow(
            [[[0, 0, 0]]], aspect="equal", extent=ap_lim + bp_lim, origin="lower"
        )

        self.highlight_point_model.trigger.add_callback(self._refresh)

    def _refresh(self):
        Jp, _, _ = self.highlight_point_model.get_Jpapbp()
        low, high = self.bgcolor_ranges[self.bg]
        if not (low <= Jp <= high):
            self.bg = self.bg_opposites[self.bg]
            self.ax.set_facecolor(self.bgcolors[self.bg])
        sRGB = sRGB_gamut_Jp_slice(Jp, self.uniform_space, self.ap_lim, self.bp_lim)
        self.image.set_data(sRGB)


class HighlightPoint2DView:
    def __init__(self, ax, highlight_point_model):
        self.ax = ax
        self.highlight_point_model = highlight_point_model

        _, ap, bp = self.highlight_point_model.get_Jpapbp()
        self.marker = self.ax.plot([ap], [bp], "y.", mew=3)[0]

        self.highlight_point_model.trigger.add_callback(self._refresh)

    def _refresh(self):
        _, ap, bp = self.highlight_point_model.get_Jpapbp()
        self.marker.set_data([ap], [bp])
        self.ax.figure.canvas.draw()


def loadpyfile(path):
    is_native = True
    cmtype = "linear"
    method = "Bezier"
    ns = {
        "__name__": "",
        "__file__": os.path.basename(path),
    }

    # FIXME: `args.colormap` should be `path`?
    with open(args.colormap) as f:  # noqa: F821
        code = compile(
            f.read(),
            os.path.basename(args.colormap),  # noqa: F821
            "exec",
        )
        exec(code, globals(), ns)

    params = ns.get("parameters", {})
    if "min_JK" in params:
        params["min_Jp"] = params.pop("min_JK")
        params["max_Jp"] = params.pop("max_JK")
    cmap = ns.get("test_cm", None)
    return params, cmtype, cmap.name, cmap, is_native, method


class Colormap:
    def __init__(self, cmtype, method, uniform_space):
        self.can_edit = True
        self.params = {}
        self.cmtype = cmtype
        self.method = method
        self.name = None
        self.cmap = None
        self.uniform_space = uniform_space
        if self.uniform_space == "buggy-CAM02-UCS":
            self.uniform_space = buggy_CAM02UCS

    def load(self, path):
        self.path = path
        if os.path.isfile(path):
            _, extension = os.path.splitext(path)
            if extension == ".py":
                self.can_edit = True
                self.cmtype = "linear"
                self.method = "Bezier"
                ns = {
                    "__name__": "",
                    "__file__": os.path.basename(self.path),
                }
                with open(self.path) as f:
                    code = compile(f.read(), os.path.basename(self.path), "exec")
                    exec(code, globals(), ns)
                self.params = ns.get("parameters", {})
                if not self.params:
                    self.can_edit = False
                if "min_JK" in self.params:
                    self.params["min_Jp"] = self.params.pop("min_JK")
                    self.params["max_Jp"] = self.params.pop("max_JK")
                self.cmap = ns.get("test_cm", None)
                self.name = self.cmap.name
            elif extension == ".jscm":
                self.can_edit = False
                with open(self.path) as f:
                    data = json.loads(f.read())
                    self.name = data["name"]
                    colors = data["colors"]
                    colors = [colors[i : i + 6] for i in range(0, len(colors), 6)]
                    colors = [
                        [int(c[2 * i : 2 * i + 2], 16) / 255 for i in range(3)]
                        for c in colors
                    ]
                    self.cmap = matplotlib.colors.ListedColormap(colors, self.name)
                    if (
                        "extensions" in data
                        and "https://matplotlib.org/viscm" in data["extensions"]
                    ):
                        self.can_edit = True
                        self.params = {
                            k: v
                            for k, v in data["extensions"][
                                "https://matplotlib.org/viscm"
                            ].items()
                            if k
                            in {
                                "xp",
                                "yp",
                                "min_Jp",
                                "max_Jp",
                                "fixed",
                                "filter_k",
                                "uniform_space",
                            }
                        }
                        self.params["name"] = self.name
                        self.cmtype = data["extensions"][
                            "https://matplotlib.org/viscm"
                        ]["cmtype"]
                        self.method = data["extensions"][
                            "https://matplotlib.org/viscm"
                        ]["spline_method"]
                        self.uniform_space = data["extensions"][
                            "https://matplotlib.org/viscm"
                        ]["uniform_colorspace"]
            else:
                sys.exit("Unsupported filetype")
        else:
            self.can_edit = False
            self.cmap = lookup_colormap_by_name(path)
            self.name = path


def about():
    QtWidgets.QMessageBox.about(
        None,
        "VISCM",
        "Copyright (C) 2015-2016 Nathaniel Smith\n"
        + "Copyright (C) 2015-2016 Stéfan van der Walt\n"
        "Copyright (C) 2016 Hankun Zhao",
    )


class ViewerWindow(QtWidgets.QMainWindow):
    def __init__(self, figurecanvas, viscm, cmapname, parent=None):
        QtWidgets.QMainWindow.__init__(self, parent)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.main_widget = QtWidgets.QWidget(self)
        self.cmapname = cmapname

        file_menu = QtWidgets.QMenu("&File", self)
        file_menu.addAction("&Save", self.save).setShortcut("Ctrl+S")
        file_menu.addAction("&Quit", self.fileQuit).setShortcut("Ctrl+Q")

        options_menu = QtWidgets.QMenu("&Options", self)
        options_menu.addAction("&Toggle Gamut", self.toggle_gamut).setShortcut("Ctrl+G")

        help_menu = QtWidgets.QMenu("&Help", self)
        help_menu.addAction("&About", about)

        self.menuBar().addMenu(file_menu)
        self.menuBar().addMenu(options_menu)
        self.menuBar().addMenu(help_menu)
        self.setWindowTitle("VISCM Editing : " + cmapname)

        self.viscm = viscm
        self.figurecanvas = figurecanvas

        v = QtWidgets.QVBoxLayout(self.main_widget)
        v.addWidget(figurecanvas)

        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)

    def toggle_gamut(self):
        self.viscm.toggle_gamut()
        self.figurecanvas.draw()

    def fileQuit(self):
        self.close()

    def closeEvent(self, ce):
        self.fileQuit()

    def save(self):
        fileName, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            caption="Save file",
            directory=self.cmapname + ".png",
            filter="Image Files (*.png *.jpg *.bmp)",
        )
        if fileName:
            self.viscm.save_figure(fileName)


class EditorWindow(QtWidgets.QMainWindow):
    def __init__(self, figurecanvas, viscm_editor, parent=None):
        QtWidgets.QMainWindow.__init__(self, parent)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.viscm_editor = viscm_editor

        file_menu = QtWidgets.QMenu("&File", self)
        file_menu.addAction("&Save", self.save).setShortcut("Ctrl+S")
        file_menu.addAction("&Export .py", self.export)
        file_menu.addAction("&Quit", self.fileQuit).setShortcut("Ctrl+Q")

        options_menu = QtWidgets.QMenu("&Options", self)
        options_menu.addAction("&Load in Viewer", self.loadviewer).setShortcut("Ctrl+V")

        help_menu = QtWidgets.QMenu("&Help", self)
        help_menu.addAction("&About", about)

        self.menuBar().addMenu(file_menu)
        self.menuBar().addMenu(options_menu)
        self.menuBar().addMenu(help_menu)
        self.setWindowTitle("VISCM Editing : " + viscm_editor.name)

        self.main_widget = QtWidgets.QWidget(self)

        self.max_slider = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.max_slider.setMinimum(0)
        self.max_slider.setMaximum(100)
        self.max_slider.setValue(int(viscm_editor.max_Jp))
        self.max_slider.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
        self.max_slider.setTickInterval(10)
        self.max_slider.valueChanged.connect(self.updatejp)
        self.max_slider_num = QtWidgets.QLabel(str(viscm_editor.max_Jp))
        self.max_slider_num.setFixedWidth(30)

        self.min_slider = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.min_slider.setMinimum(0)
        self.min_slider.setMaximum(100)
        self.min_slider.setValue(int(viscm_editor.min_Jp))
        self.min_slider.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
        self.min_slider.setTickInterval(10)
        self.min_slider.valueChanged.connect(self.updatejp)
        self.min_slider_num = QtWidgets.QLabel(str(viscm_editor.min_Jp))
        self.min_slider_num.setFixedWidth(30)

        max_slider_layout = QtWidgets.QHBoxLayout()
        max_slider_layout.addWidget(QtWidgets.QLabel("Jp 0"))
        max_slider_layout.addWidget(self.max_slider)
        max_slider_layout.addWidget(self.max_slider_num)
        min_slider_layout = QtWidgets.QHBoxLayout()
        min_slider_layout.addWidget(QtWidgets.QLabel("Jp 1"))
        min_slider_layout.addWidget(self.min_slider)
        min_slider_layout.addWidget(self.min_slider_num)

        figure_layout = QtWidgets.QHBoxLayout()
        figure_layout.addWidget(figurecanvas)

        mainlayout = QtWidgets.QVBoxLayout(self.main_widget)
        mainlayout.addLayout(figure_layout)
        mainlayout.addLayout(max_slider_layout)
        mainlayout.addLayout(min_slider_layout)

        if viscm_editor.cmtype == "diverging":
            smoothness_slider = QtWidgets.QSlider(Qt.Horizontal)
            # We want the slider to vary filter_k exponentially between 5 and
            # 1000. So it should go from [log10(5), log10(1000)]
            # which is about [0.699, 3.0]
            # But QSlider is integer-only, so we also scale up by 1000.
            # smoothness_slider_moved and update_smoothness_slider also have
            # to deal with this.
            smoothness_slider.setMinimum(699)
            smoothness_slider.setMaximum(3000)
            smoothness_slider.setTickPosition(QtWidgets.QSlider.NoTicks)
            smoothness_slider.valueChanged.connect(self.smoothness_slider_moved)

            # maximum width
            smoothness_slider_num = QtWidgets.QLabel("1000.00")
            metrics = QtGui.QFontMetrics(smoothness_slider_num.font())
            max_width = metrics.width("1000.00")
            smoothness_slider_num.setFixedWidth(max_width)
            smoothness_slider_num.setAlignment(Qt.AlignRight)
            self.smoothness_slider_num = smoothness_slider_num

            smoothness_slider_layout = QtWidgets.QHBoxLayout()
            smoothness_slider_layout.addWidget(QtWidgets.QLabel("Transition sharpness"))
            smoothness_slider_layout.addWidget(smoothness_slider)
            smoothness_slider_layout.addWidget(smoothness_slider_num)

            mainlayout.addLayout(smoothness_slider_layout)
            self.smoothness_slider = smoothness_slider

            viscm_editor.cmap_model.filter_k_trigger.add_callback(
                self.update_smoothness_slider
            )

        self.toolbar = self.addToolBar("Tools")
        self.moveAction = self.toolbar.addAction("Drag points")
        self.moveAction.triggered.connect(self.set_move_mode)
        self.moveAction.setCheckable(True)
        self.addAction = self.toolbar.addAction("Add points")
        self.addAction.triggered.connect(self.set_add_mode)
        self.addAction.setCheckable(True)
        self.removeAction = self.toolbar.addAction("Remove points")
        self.removeAction.triggered.connect(self.set_remove_mode)
        self.removeAction.setCheckable(True)

        self.toolbar.addSeparator()
        self.swapAction = self.toolbar.addAction("Flip brightness")
        self.swapAction.triggered.connect(self.swapjp)

        self.toolbar.addSeparator()
        renameAction = self.toolbar.addAction("Rename colormap")
        renameAction.triggered.connect(self.rename)
        saveAction = self.toolbar.addAction("Save")
        saveAction.triggered.connect(self.save)

        self.moveAction.setChecked(True)

        self.main_widget.setFocus()
        figurecanvas.setFocus()
        figurecanvas.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setCentralWidget(self.main_widget)

    def rename(self):
        name, ok = QtWidgets.QInputDialog.getText(
            self, "Rename your colormap", "Enter a name", text=self.viscm_editor.name
        )
        self.viscm_editor.name = name
        self.setWindowTitle("VISCM Editing : " + self.viscm_editor.name)

    def smoothness_slider_moved(self):
        num = 10 ** (self.smoothness_slider.value() / 1000)
        self.smoothness_slider_num.setText(str(num))
        self.viscm_editor.cmap_model.set_filter_k(num)

    def update_smoothness_slider(self):
        filter_k = self.viscm_editor.cmap_model.filter_k
        self.smoothness_slider_num.setText(f"{filter_k:0.2f}")
        self.smoothness_slider.setValue(int(round(np.log10(filter_k) * 1000)))

    def swapjp(self):
        jp1, jp2 = self.min_slider.value(), self.max_slider.value()
        self.min_slider.setValue(int(jp2))
        self.max_slider.setValue(int(jp1))
        self.updatejp()

    def updatejp(self):
        minval = self.min_slider.value()
        maxval = self.max_slider.value()
        self.viscm_editor._jp_update(minval, maxval)
        self.min_slider_num.setText(str(minval))
        self.max_slider_num.setText(str(maxval))

    def set_move_mode(self):
        self.addAction.setChecked(False)
        self.removeAction.setChecked(False)
        self.viscm_editor.bezier_builder.mode = "move"

    def set_add_mode(self):
        self.moveAction.setChecked(False)
        self.removeAction.setChecked(False)
        self.viscm_editor.bezier_builder.mode = "add"

    def set_remove_mode(self):
        self.addAction.setChecked(False)
        self.moveAction.setChecked(False)
        self.viscm_editor.bezier_builder.mode = "remove"

    def export(self):
        fileName, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            caption="Export file",
            directory=self.viscm_editor.name + ".py",
            filter=".py (*.py)",
        )
        if fileName:
            self.viscm_editor.export_py(fileName)

    def fileQuit(self):
        self.close()

    def closeEvent(self, ce):
        self.fileQuit()

    def save(self):
        fileName, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            caption="Save file",
            directory=self.viscm_editor.name + ".jscm",
            filter="JSCM Files (*.jscm)",
        )
        if fileName:
            self.viscm_editor.save_colormap(fileName)

    def loadviewer(self):
        newfig = plt.figure()
        newcanvas = FigureCanvas(newfig)
        cm = self.viscm_editor.show_viscm()
        v = viscm(cm, name=self.viscm_editor.name, figure=newfig)

        newcanvas.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        newcanvas.updateGeometry()

        newwindow = ViewerWindow(newcanvas, v, self.viscm_editor.name, parent=self)
        newwindow.resize(800, 600)

        newwindow.show()
