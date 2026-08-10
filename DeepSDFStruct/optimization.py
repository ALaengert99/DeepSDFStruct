"""
Structural Optimization Utilities
==================================

This module provides tools for gradient-based optimization of SDF-based geometries,
with a focus on structural design problems. It integrates with TorchFEM for
finite element analysis and provides optimization algorithms suitable for
constrained design problems.

Key Features
------------

MMA Optimizer
    Implementation of the Method of Moving Asymptotes (MMA), a gradient-based
    algorithm well-suited for structural optimization with nonlinear constraints.
    MMA is particularly effective for:
    - Topology optimization
    - Shape optimization with constraints
    - Problems with expensive objective evaluations
    - Highly nonlinear design spaces

Finite Element Integration
    - Conversion between TorchFEM and PyVista mesh formats
    - Support for tetrahedral and hexahedral elements
    - Linear and quadratic element types
    - Integration with gradient computation

Mesh Quality Utilities
    - Signed volume computation for tetrahedra
    - Mesh quality metrics
    - Degeneracy detection

The module is designed to work seamlessly with differentiable SDF representations,
enabling gradient-based optimization of complex 3D structures.
"""

import torchfem.materials
import torchfem.solid
from torchfem.elements import Hexa1, Hexa2, Tetra1, Tetra2
import torch
import numpy as np
from mmapy import mmasub, gcmmasub, asymp, raaupdate
import pyvista
import logging
import DeepSDFStruct

from abc import ABC, abstractmethod
from DeepSDFStruct.SDF import SDFfromDeepSDF, SDFfromMesh, DifferenceSDF, UnionSDF, NegatedCallable, TransformedSDF
from DeepSDFStruct.local_shapes import LocalShapesSDF
import trimesh
import gustaf as gus
from DeepSDFStruct.mesh import (
    create_3D_mesh,
    export_sdf_grid_vtk,
    torchVolumeMesh,
)
from torchfem import Solid
from torchfem.materials import IsotropicElasticity3D
from types import SimpleNamespace
import splinepy
from DeepSDFStruct.parametrization import SplineParametrization
from hashlib import sha256
from pathlib import Path
from DeepSDFStruct.sampling import sample_mesh_surface, random_sample_sdf
from DeepSDFStruct.deep_sdf.reconstruction import reconstruct_from_samples
import matplotlib.pyplot as plt
import pyvista as pv
from sklearn.decomposition import PCA
from DeepSDFStruct.pretrained_models import PretrainedModels
from DeepSDFStruct.deep_sdf.models import DeepSDFModel
from DeepSDFStruct.geom_reconstruction import LocalShapesReconstructor
from DeepSDFStruct.SDF import SDFBase






logger = logging.getLogger(DeepSDFStruct.__name__)


def get_mesh_from_torchfem(Solid: torchfem.Solid) -> pyvista.UnstructuredGrid:
    """Convert a TorchFEM Solid mesh to PyVista UnstructuredGrid.

    This function enables visualization and export of TorchFEM finite element
    meshes using PyVista. It supports both tetrahedral and hexahedral elements
    with linear and quadratic shape functions.

    Parameters
    ----------
    Solid : torchfem.Solid
        TorchFEM solid mesh object containing nodes, elements, and element type.

    Returns
    -------
    pyvista.UnstructuredGrid
        PyVista mesh representation suitable for visualization and I/O.

    Raises
    ------
    NotImplementedError
        If input is not a torchfem.Solid object.

    Notes
    -----
    Supported element types:
    - Tetra1: 4-node linear tetrahedron
    - Tetra2: 10-node quadratic tetrahedron
    - Hexa1: 8-node linear hexahedron
    - Hexa2: 20-node quadratic hexahedron

    Examples
    --------
    >>> from DeepSDFStruct.optimization import get_mesh_from_torchfem
    >>> import torchfem
    >>>
    >>> # Assume we have a TorchFEM solid mesh
    >>> # solid = torchfem.Solid(...)
    >>>
    >>> # Convert to PyVista for visualization
    >>> pv_mesh = get_mesh_from_torchfem(solid)
    >>> pv_mesh.plot()
    """
    if not isinstance(Solid, torchfem.Solid):
        raise NotImplementedError("Currently only solid mesh is supported.")
    # VTK cell types
    etype = Solid.etype

    if etype is Tetra1 or isinstance(etype, Tetra1):
        cell_types = [pyvista.CellType.TETRA] * Solid.n_elem
    elif etype is Tetra2 or isinstance(etype, Tetra2):
        cell_types = [pyvista.CellType.QUADRATIC_TETRA] * Solid.n_elem
    elif etype is Hexa1 or isinstance(etype, Hexa1):
        cell_types = [pyvista.CellType.HEXAHEDRON] * Solid.n_elem
    elif etype is Hexa2 or isinstance(etype, Hexa2):
        cell_types = [pyvista.CellType.QUADRATIC_HEXAHEDRON] * Solid.n_elem
    else:
        raise TypeError(f"Unsupported element type: {etype} ({type(etype)})")

    # VTK element list
    el = len(Solid.elements[0]) * torch.ones(Solid.n_elem, dtype=Solid.elements.dtype)
    elements = torch.cat([el[:, None], Solid.elements], dim=1).view(-1).tolist()

    # Deformed node positions
    pos = Solid.nodes

    # Create unstructured mesh
    mesh = pyvista.UnstructuredGrid(elements, cell_types, pos.tolist())
    return mesh


def tet_signed_vol(vertices, tets):
    """Compute signed volumes of tetrahedral elements.

    Calculates the signed volume of each tetrahedron, which is positive for
    correctly oriented elements and negative for inverted elements. This is
    useful for detecting mesh degeneracies and enforcing mesh quality constraints.

    Parameters
    ----------
    vertices : torch.Tensor
        Vertex coordinates of shape (N, 3).
    tets : torch.Tensor
        Tetrahedral connectivity of shape (M, 4), where each row contains
        vertex indices [v0, v1, v2, v3].

    Returns
    -------
    torch.Tensor
        Signed volumes of shape (M,), one per tetrahedron. Positive volumes
        indicate correctly oriented elements.

    Notes
    -----
    The signed volume is computed as:
        V = (1/6) * ((v1-v0) × (v2-v0)) · (v3-v0)

    Examples
    --------
    >>> import torch
    >>> from DeepSDFStruct.optimization import tet_signed_vol
    >>>
    >>> # Define a simple tetrahedron
    >>> vertices = torch.tensor([
    ...     [0.0, 0.0, 0.0],
    ...     [1.0, 0.0, 0.0],
    ...     [0.0, 1.0, 0.0],
    ...     [0.0, 0.0, 1.0]
    ... ])
    >>> tets = torch.tensor([[0, 1, 2, 3]])
    >>> volumes = tet_signed_vol(vertices, tets)
    >>> print(f"Volume: {volumes[0]:.3f}")  # Should be 1/6 ≈ 0.167
    """
    v0 = vertices[tets[:, 0]]
    v1 = vertices[tets[:, 1]]
    v2 = vertices[tets[:, 2]]
    v3 = vertices[tets[:, 3]]
    vols = torch.einsum("ij,ij->i", torch.cross(v1 - v0, v2 - v0, dim=1), v3 - v0) / 6.0
    return vols


class MMA:
    """Method of Moving Asymptotes (MMA) optimizer for constrained problems.

    MMA is a gradient-based optimization algorithm designed for nonlinear
    constrained problems. It constructs convex subproblems using moving
    asymptotes and is particularly effective for structural optimization.

    The optimizer handles a single objective function and a single constraint,
    with box bounds on design variables. It automatically normalizes the
    objective by its initial value for better numerical behavior.

    Parameters
    ----------
    parameters : torch.Tensor
        Initial design variables (will be optimized in-place).
    bounds : array-like of shape (n, 2)
        Box constraints [[lower_1, upper_1], ..., [lower_n, upper_n]]
        for each design variable.
    max_step : float, default 0.1
        Maximum allowed change in design variables per iteration,
        as a fraction of the bound range.

    Attributes
    ----------
    parameters : torch.Tensor
        Current design variables (updated in-place each iteration).
    loop : int
        Current iteration number.
    x : ndarray
        Current design variables in numpy format.
    xold1, xold2 : ndarray
        Design variables from previous two iterations (for MMA history).

    Methods
    -------
    step(F, dF, G, dG)
        Perform one MMA optimization step given objective, constraint,
        and their gradients.

    Notes
    -----
    MMA was developed by Krister Svanberg and is widely used in topology
    optimization. It is particularly effective for problems where:
    - The objective and constraints are expensive to evaluate
    - Gradients are available (via automatic differentiation)
    - The design space is high-dimensional
    - Strong nonlinearity is present

    The implementation uses the mmapy package for the core MMA algorithm.

    Examples
    --------
    >>> import torch
    >>> from DeepSDFStruct.optimization import MMA
    >>>
    >>> # Define design variables
    >>> params = torch.ones(10, requires_grad=True)
    >>> bounds = [[0.0, 2.0]] * 10
    >>>
    >>> # Create optimizer
    >>> optimizer = MMA(params, bounds, max_step=0.1)
    >>>
    >>> # Optimization loop
    >>> for i in range(100):
    ...     # Compute objective and constraint
    ...     objective = (params ** 2).sum()
    ...     constraint = params.sum() - 5.0
    ...
    ...     # Compute gradients
    ...     dF = torch.autograd.grad(objective, params, create_graph=True)[0]
    ...     dG = torch.autograd.grad(constraint, params, create_graph=True)[0]
    ...
    ...     # MMA step
    ...     optimizer.step(objective, dF, constraint, dG)
    ...
    ...     if optimizer.ch < 1e-3:
    ...         break

    References
    ----------
    .. [1] Svanberg, K. (1987). "The method of moving asymptotes—a new method
           for structural optimization." International Journal for Numerical
           Methods in Engineering, 24(2), 359-373.
    .. [2] mmapy: Python implementation of MMA
           https://github.com/arjendeetman/mmapy
    """

    def __init__(self, parameters, bounds, max_step=0.1, n_constraints=1):
        self.max_step = max_step
        self.bounds = np.asarray(bounds, dtype=float)
        self.parameters = parameters

        self.m = n_constraints
        self.n = parameters.numel()

        self.x = parameters.detach().cpu().numpy().reshape(-1, 1)
        self.xold1 = self.x.copy()
        self.xold2 = self.x.copy()

        self.low = np.zeros((self.n, 1))
        self.upp = np.zeros((self.n, 1))

        self.a0_MMA = 1.0
        self.a_MMA = np.zeros((self.m, 1))
        self.c_MMA = 10000 * np.ones((self.m, 1))
        self.d_MMA = np.zeros((self.m, 1))

        self.loop = 0
        self.ch = 1.0
        self.F0 = None

    def _restore_feasibility(self, x, restore_eval, tol, max_steps, step_limit):
        """Project an accepted candidate back onto the cheap-constraint feasible set.

        Minimum-norm Gauss-Newton on the violated rows: solve ``J dx = -g`` for the
        smallest ``dx`` (so the objective is disturbed as little as possible to first
        order), capped at ``step_limit`` per pass, clipped to the box bounds, with
        backtracking on the TRUE values. Locked design variables never move because the
        supplied row gradients are already masked to zero there. Stops when every row
        reads ``g <= tol``, when ``max_steps`` passes are exhausted, or when backtracking
        cannot reduce the worst violation (evaluation-noise floor) -- the residual is
        logged either way.
        """
        lim = float(step_limit) if step_limit is not None else float(self.max_step)
        lo, hi = self.bounds[:, 0:1], self.bounds[:, 1:2]
        g, J = restore_eval(x)
        g = np.asarray(g, dtype=float).reshape(-1)
        v0 = float(g.max())
        if v0 <= tol:
            return x
        steps_used = 0
        for _ in range(max(1, int(max_steps))):
            viol = g > tol
            if not viol.any():
                break
            Jv = np.asarray(J, dtype=float).reshape(g.size, -1)[viol]
            gv = g[viol]
            # Least-norm correction onto g = 0 (strictly inside the tol-acceptance):
            # dx = Jv^T (Jv Jv^T)^-1 (-gv), tiny Tikhonov guard for degenerate rows.
            A = Jv @ Jv.T
            A += (
                1e-10 * max(float(np.trace(A)) / max(gv.size, 1), 0.0) + 1e-30
            ) * np.eye(gv.size)
            dx = (Jv.T @ np.linalg.solve(A, -gv)).reshape(-1, 1)
            nrm = float(np.abs(dx).max())
            if nrm <= 0.0:
                logger.warning(
                    "  feasibility restoration: zero correction direction (all row "
                    f"gradients masked/vanishing); residual max g = {g.max():+.3e}"
                )
                break
            if nrm > lim:
                dx *= lim / nrm
            improved = False
            for _bt in range(4):
                x_try = np.clip(x + dx, lo, hi)
                g_try, J_try = restore_eval(x_try)
                g_try = np.asarray(g_try, dtype=float).reshape(-1)
                if g_try.max() < g.max() - 1e-12:
                    x, g, J = x_try, g_try, J_try
                    improved = True
                    steps_used += 1
                    break
                dx *= 0.5
            if not improved:
                break
        if float(g.max()) > tol:
            logger.warning(
                f"  feasibility restoration: residual violation max g = {g.max():+.3e} "
                f"> tol {tol:.1e} after {steps_used} step(s) (started at {v0:+.3e}) -- "
                f"likely an evaluation-noise floor or a step_limit cap."
            )
        else:
            logger.info(
                f"  feasibility restoration: max g {v0:+.3e} -> {g.max():+.3e} "
                f"in {steps_used} step(s)"
            )
        return x

    def step(
        self,
        F,
        dF,
        G,
        dG,
        geom_eval=None,
        geom_rows=None,
        max_inner=1,
        feas_tol=0.05,
        restore_eval=None,
        restore_tol=5e-3,
        restore_max_steps=8,
        restore_step_limit=None,
    ):
        """Perform one MMA optimization step.

        Updates design variables by solving a convex subproblem constructed
        from the objective, constraint, and their gradients.

        Parameters
        ----------
        F : torch.Tensor or float
            Objective function value at current design.
        dF : torch.Tensor
            Gradient of objective w.r.t. design variables, shape (n,).
        G : torch.Tensor or float
            Constraint values at current design (≤ 0 is feasible), reshaped to
            (m, 1) for the ``m = n_constraints`` rows this instance was built
            with. A scalar is accepted when ``m == 1``.
        dG : torch.Tensor
            Gradient of the constraints w.r.t. design variables, reshaped to
            (m, n). For ``m == 1`` a flat (n,) tensor is accepted.
        geom_eval : callable, optional
            Cheap geometry-only re-evaluation ``x_np -> np.ndarray``. Given a
            candidate design vector it returns the *true* (nonlinear) constraint
            values ``g = value - target`` for the rows listed in ``geom_rows``,
            without running the expensive (CFD) objective/constraints. Enables the
            hybrid-GCMMA conservativeness loop; when ``None`` this is a plain MMA
            step.
        geom_rows : sequence of int, optional
            Row indices into ``G`` for the geometry-only constraints that
            ``geom_eval`` returns, in the same order. Required with ``geom_eval``.
        max_inner : int, optional
            Maximum GCMMA inner (conservativeness) iterations per step when
            ``geom_eval`` is supplied. Each rejected candidate raises the offending
            rows' curvature parameter rho (Svanberg 2002 ``raaupdate``) and re-solves;
            the move limit stays FIXED. Default 1 (no inner loop).
        feas_tol : float, optional
            TRUE-violation level below which a candidate is accepted outright, in the
            units of the constraint rows (pass normalized, "fraction over budget" rows
            and the default 0.05 reads "5% over budget is tolerated transiently").
            Must be well above 0: at an ACTIVE constraint the true value always reads
            slightly above the approximation (residual curvature, evaluation noise) --
            with a ~0 tolerance the inner loop fires on every boundary-riding step and
            burns max_inner solves per iteration for nothing.
        restore_eval : callable, optional
            Enables POST-STEP FEASIBILITY RESTORATION on the cheap geometry rows:
            ``x_np -> (g, J)`` returning the true values (k,) AND their gradients
            (k, n) -- masked for locked variables, in the same normalized units as the
            constraint rows. After the step is accepted (through whichever gate), the
            candidate is projected back onto the geometry-feasible set with
            minimum-norm Gauss-Newton passes, so geometry violations beyond
            ``restore_tol`` cannot survive an iteration. Independent of the GCMMA
            inner loop (works with or without ``geom_eval``). No CFD is invoked.
        restore_tol : float, optional
            Restoration target/trigger: rows with ``g <= restore_tol`` are left alone.
            Keep it above the geometry-evaluation noise floor (default 5e-3).
        restore_max_steps : int, optional
            Maximum Gauss-Newton passes per outer iteration (default 8; typically 1-2
            are used). Each pass costs one geometry evaluation plus up to 4 backtracks.
        restore_step_limit : float, optional
            Per-pass infinity-norm cap on the correction; defaults to ``max_step``.

        Notes
        -----
        The method automatically:
        - Normalizes the objective by its initial value
        - Enforces move limits based on max_step
        - Updates MMA history (xold1, xold2)
        - Computes and logs convergence metric (ch)
        - Updates self.parameters in-place

        The convergence metric ch is the relative change in design variables.
        """
        F_np = np.asarray(F.detach().cpu().numpy(), dtype=float).reshape(1, 1)
        dFdx_np = np.asarray(dF.detach().cpu().numpy(), dtype=float).reshape(self.n, 1)

        G_np = np.asarray(G.detach().cpu().numpy(), dtype=float).reshape(self.m, 1)
        dGdx_np = np.asarray(dG.detach().cpu().numpy(), dtype=float).reshape(
            self.m, self.n
        )

        if self.loop == 0:
            # Normalize by the MAGNITUDE of the initial objective. Dividing by a
            # signed F0 flips the sign of both F and dF whenever F(x0) < 0, which
            # turns the minimization into a maximization: the same problem with a
            # constant added to the objective (which cannot move the optimum) then
            # converges to a different point. An exactly-zero F(x0) would divide by
            # zero, so fall back to 1.0 and leave the objective unscaled.
            f0_mag = float(np.abs(F_np[0, 0]))
            self.F0 = np.full((1, 1), f0_mag if f0_mag > 0.0 else 1.0)

        F_np = F_np / self.F0
        dFdx_np = dFdx_np / self.F0

        # Hybrid GCMMA conservativeness loop (Svanberg 2002, CCSA). Solve the subproblem,
        # then -- when a cheap geometry-only re-evaluation callback is supplied -- check
        # whether any geometry row reads worse at the candidate than its own conservative
        # approximation predicted. If so, RAISE that row's curvature parameter rho
        # (raaupdate) and re-solve with the move limit FIXED: the subproblem then *sees*
        # the nonlinearity (e.g. KS-margin softmax curvature) and picks a genuinely
        # different direction, instead of re-scaling the same bad step. (The previous
        # move-limit-halving back-off converged to a null step of the SAME direction:
        # g_true -> g_now + noise floor as dx -> 0, so the loop burned max_inner halvings
        # every iteration and then accepted a candidate that ratcheted the violation up
        # ~1.5e-3/iter with the objective long converged.) Only the geometry rows are
        # re-evaluated: the (expensive, CFD-based) objective and remaining constraints
        # stay frozen at their approximation (f0valnew = f0app, fvalnew = fapp), so
        # raaupdate can never touch them and no extra primal/adjoint solves happen.
        do_inner = (
            geom_eval is not None and geom_rows is not None and len(geom_rows) > 0
        )
        pred_slack = 1e-6  # slack on "worse than the model" (rows are O(1) normalized)

        self.loop += 1
        xmin = np.maximum(self.x - float(self.max_step), self.bounds[:, 0:1])
        xmax = np.minimum(self.x + float(self.max_step), self.bounds[:, 1:2])

        if not do_inner:
            xmma, ymma, zmma, lam, xsi, eta, muMMA, zet, s, low, upp = mmasub(
                self.m,
                self.n,
                self.loop,
                self.x,
                xmin,
                xmax,
                self.xold1,
                self.xold2,
                F_np,
                dFdx_np,
                G_np,
                dGdx_np,
                self.low,
                self.upp,
                self.a0_MMA,
                self.a_MMA,
                self.c_MMA,
                self.d_MMA,
            )
        else:
            # Standard GCMMA constants (Svanberg's reference driver): epsimin is the
            # conservativeness slack, raa0eps/raaeps floor the curvature parameters.
            epsimin = 1e-7
            raa0eps = 1e-6
            raaeps = 1e-6 * np.ones((self.m, 1))
            rows = list(geom_rows)
            n_inner = max(1, int(max_inner))
            # asymp: asymptote update (same rule as mmasub) + fresh per-outer-iteration
            # initialization of raa0 (objective) / raa (constraint rows) from the
            # current gradients. The raa0/raa inputs are overwritten, so pass dummies.
            low, upp, raa0, raa = asymp(
                self.loop,
                self.n,
                self.x,
                self.xold1,
                self.xold2,
                xmin,
                xmax,
                self.low,
                self.upp,
                raa0eps,
                np.full((self.m, 1), raaeps[0, 0]),
                raa0eps,
                raaeps,
                dFdx_np,
                dGdx_np,
            )
            g_now = G_np[rows, 0]
            best_x, best_score = None, np.inf
            for inner in range(n_inner):
                xmma, ymma, zmma, lam, xsi, eta, muMMA, zet, s, f0app, fapp = gcmmasub(
                    self.m,
                    self.n,
                    self.loop,
                    epsimin,
                    self.x,
                    xmin,
                    xmax,
                    low,
                    upp,
                    raa0,
                    raa,
                    F_np,
                    dFdx_np,
                    G_np,
                    dGdx_np,
                    self.a0_MMA,
                    self.a_MMA,
                    self.c_MMA,
                    self.d_MMA,
                )
                # True (nonlinear) geometry values at the candidate (g = value - target,
                # > 0 infeasible) vs the conservative approximation at the same point
                # (fapp includes the current rho curvature, unlike a bare linearization).
                g_true = np.asarray(geom_eval(xmma), dtype=float).reshape(-1)
                g_app = np.asarray(fapp, dtype=float).reshape(-1)[rows]
                # Accept when, for every geometry row, at least one holds: the true
                # violation is small (<= feas_tol -- boundary riding always reads a bit
                # above the model), the approximation was conservative (g_true <= g_app),
                # or the candidate does not worsen the row vs the CURRENT point (progress
                # toward feasibility must never be rejected).
                overshoot = (
                    (g_true > feas_tol)
                    & (g_true > g_app + pred_slack)
                    & (g_true > g_now + pred_slack)
                )
                # Track the least-worsening candidate for the exhaustion fallback.
                score = float(np.max(g_true - g_now))
                if score < best_score:
                    best_score, best_x = score, xmma.copy()
                if not overshoot.any():
                    best_x = xmma
                    break
                if inner == n_inner - 1:
                    logger.warning(
                        f"  GCMMA inner loop exhausted ({n_inner} solves): accepting the "
                        f"least-worsening candidate (max geom-row increase "
                        f"{best_score:+.3e} vs current); rho escalation could not make "
                        f"the model conservative -- likely an evaluation noise floor."
                    )
                    break
                logger.info(
                    f"  GCMMA rho update (attempt {inner + 1}/{n_inner}): geom overshoot "
                    f"g_true={g_true[overshoot].tolist()} > g_app="
                    f"{g_app[overshoot].tolist()}; raa[geom]="
                    f"{np.asarray(raa).reshape(-1)[rows].tolist()}"
                )
                # Raise rho on every geometry row that read worse than its model
                # (Svanberg raaupdate: raa <- min(1.1*(raa + delta), 10*raa)). Objective
                # and CFD rows are frozen at their approximation, so only geometry rows
                # can be updated.
                fvalnew = np.asarray(fapp, dtype=float).reshape(self.m, 1).copy()
                fvalnew[rows, 0] = g_true
                raa0, raa = raaupdate(
                    xmma,
                    self.x,
                    xmin,
                    xmax,
                    low,
                    upp,
                    np.asarray(f0app, dtype=float).reshape(1, 1),
                    fvalnew,
                    np.asarray(f0app, dtype=float).reshape(1, 1),
                    np.asarray(fapp, dtype=float).reshape(self.m, 1),
                    raa0,
                    raa,
                    raa0eps,
                    raaeps,
                    epsimin,
                )
            xmma = best_x

        # Optional post-step feasibility restoration: whatever gate accepted the
        # candidate (feas_tol shortcut, conservativeness, improves-vs-current,
        # exhaustion fallback), project it back onto the cheap-constraint feasible
        # set before committing it as the new design.
        if restore_eval is not None:
            xmma = self._restore_feasibility(
                xmma,
                restore_eval,
                float(restore_tol),
                restore_max_steps,
                restore_step_limit,
            )

        self.xold2 = self.xold1.copy()
        self.xold1 = self.x.copy()
        self.x = xmma
        self.low = low
        self.upp = upp

        self.ch = np.abs(np.mean(self.x.T - self.xold1.T) / np.mean(self.x.T))

        with torch.no_grad():
            self.parameters.copy_(
                torch.tensor(
                    xmma.reshape(self.parameters.shape),
                    dtype=self.parameters.dtype,
                    device=self.parameters.device,
                )
            )

        logger.info(
            f"It.: {self.loop:4d} | J.: {F_np[0,0]:1.3e} | "
            f"G: {[float(g) for g in G_np[:, 0]]} | ch.: {self.ch:1.3e}"
        )




class Region(ABC):
    """
       Abstraction class for geometric Regions to be used in Optimization problems.
       As Input, DeepSDFs, SDFs, stl meshes or formulas can be used.
       Uniform scaling is handled by this class.
    """
    def __init__(self):
        super().__init__()
        pass

    @classmethod
    def create(cls, geometry, threshold=None):
        logger.info(f"Factory method for Region called with region type {type(geometry)}")
        if isinstance(geometry, SDFBase):
            return SDFRegion(geometry)   # Okay to use SDF base class?
        elif isinstance(geometry, trimesh.Trimesh):
            return MeshRegion(geometry, threshold)
        elif isinstance(geometry, gus.Faces):
            # Convenience wrapper
            return MeshRegion(trimesh.Trimesh(geometry.vertices, geometry.faces), threshold)
        elif isinstance(geometry, callable):
            return CallableRegion(geometry)
        else:
            raise TypeError(f"Geometry of type {type(geometry)} not supported")

    @classmethod
    def create_parametrized(cls,
            geometry,
            tiling: list[int] = [8, 8, 8],
            save_dir: str = './'
        ):
        """
            Convenience wrapper for directly creating a parametrized region
        """
        region = cls.create(geometry)
        return region.parametrize(tiling, save_dir)
        
    def parametrize(self,
            tiling: list[int] = [8, 8, 8],
            save_dir: str = './'
        ):
        if isinstance(self, CallableRegion):
            raise Exception("A Formula Region can't be parametrized")

        if self.mesh is None:
            raise Exception("The given region doesn't expose a mesh")

        recon = LocalShapesReconstructor(output_dir=save_dir, device='cpu')
    
        filename = f"{self._hash}.pt"
        base_dir = Path(save_dir).resolve()
        f = Path(base_dir / filename)
        if f.is_file():   # TODO: check if other params like tiling & base_sdf are the same
            built = recon.build_struct(
                self.mesh,
                tiling,
                spline_degree=(1, 1, 0)
            )
            struct = built.struct
            scaling = built.scaling
            recon_param = torch.load(
                str(f), weights_only=True, map_location='cpu'  # TODO: change cpu to parameter
            )
            struct.parametrization.load_state_dict(recon_param)
            logger.warning("Using found reconstruction parameters")
        else:
            logger.info(f"fitting lattice to provided mesh..")
            struct, scaling, _, _ = recon.fit_mesh(
                mesh=self.mesh,
                tiling=tiling,
                spline_degree=(1, 1, 0)
            )
            torch.save(
                struct.parametrization.state_dict(),
                str(f),
            )
            logger.info(f"Saved parameter dict to {str(f)}")
    
        export_sdf_grid_vtk(struct, save_dir / "fitted_latice.vtk")
    
    
        lattice = ParametrizedLattice(
            geometry=struct,
            scaling=scaling,
        )
        
        export_sdf_grid_vtk(lattice.sdf, save_dir / "scaled_latice.vtk")
    
        return lattice


    @property
    @abstractmethod
    def sdf(self):
        """
            sdf that is scaled to the original physical space
        """
        pass

    @property
    @abstractmethod
    def mesh(self):
        """
            Trimesh
        """

    @property
    @abstractmethod
    def _hash(self):
        """
            Return a hash of the underlying region & geometry for caching purposes
        """
        pass

    @abstractmethod
    def contains(self, vertices: torch.Tensor):
        """
            Computes if the specified vertices lie inside the region.

            Parameters
            ----------
            vertices : torch.Tensor
                Vertex coordinates of shape (N, 3).
            Returns
            -------
            torch.Tensor
                Boolean array of shape (N, ). True for each vertice that lies inside the Region, False otherwise.

        """
        pass



class SDFRegion(Region):
    def __init__(self, geometry):
        super().__init__()
        self._mesh = None
        self._sdf = geometry

    def contains(self, vertices):
        return self.sdf(vertices) <= 0

    @property
    def sdf(self):
        return self._sdf

    @property
    def mesh(self):
        if self._mesh is None:
            torch_mesh, _ = create_3D_mesh(
                self.sdf,
                64,  # TODO: Not hardcoded
                mesh_type="surface",
                differentiate=False,
                device='cpu'   # TODO: Not hardcoded
            )
            self._mesh = torch_mesh.to_trimesh()
        return self._mesh

    @property
    def _hash(self):
        return sha256(self.mesh.vertices).hexdigest()   # Base hash on the mesh for now



class MeshRegion(Region):
    def __init__(self, geometry, threshold=None):
        super().__init__()
        self._mesh = geometry
        self._sdf = None
        self.threshold = threshold
        logger.info(f"Instantiated {self.__class__}")

    @property
    def sdf(self):
        if self._sdf is None:
            self._sdf = SDFfromMesh(
                self.mesh,
                scale=False,
                **({'threshold': self.threshold} if self.threshold else {})
            )
        return self._sdf

    @property
    def mesh(self):
        return self._mesh

    @property
    def _hash(self):
        return sha256(self.mesh.vertices).hexdigest()

    def contains(self, vertices):
        return np.squeeze(self.sdf(vertices) <= 0)   # TODO: Investigate accuracy offset


class CallableRegion(Region):
    def __init__(self, func: callable):
        super().__init__()
        self.func = func

    @property
    def sdf(self):
        raise Exception(f"The sdf representation of a Formula Region was requested")

    @property
    def mesh(self):
        raise Exception(f"The mesh representation of a Formula Region was requested")

    def contains(self, vertices):
        return self.func(vertices)

    @property
    def _hash(self):
        return sha256(self.func).hexdigest()


class ParametrizedRegion(ABC):
    def __init__(self):
        super().__init__()

    @property
    @abstractmethod
    def sdf(self):
        """
            Return sdf that is scaled to the original physical space
        """
        pass

    @property
    @abstractmethod
    def parametrization(self):
        """
            Return the parametrization of this region to be uses in an optimization.
        """
        pass

    # TODO: Is that needed?
    # @abstractmethod
    # def contains(self, vertices: torch.Tensor):
    #     pass


class ParametrizedLattice(ParametrizedRegion):
    def __init__(self, geometry, scaling = None):
        super().__init__()
        self.lattice = geometry
        self.scaling = scaling

    @property
    def parametrization(self):
        return next(self.lattice.parametrization.parameters())

    @property
    def sdf(self):
        if self.scaling is None:
            return self.lattice
        else:
            return TransformedSDF(
                sdf=self.lattice,
                translation=self.scaling.translation / self.scaling.scale_factors,    # Potential bug in TransformedSDF? Need to scale the translation by the scale factor
                scaleFactor=self.scaling.scale_factors
            )


class Condition(ABC):
    """
    Abstraction class for conditions for the FEM simulation.
    Once instantiated, they must implement the method apply to apply the
    BC on a given torchfem solid simulation
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def apply(self, model: torchfem.Solid):
        pass


class Force(Condition):
    def __init__(self, region: Region, magnitude):
        super().__init__()
        self.region = region
        self.magnitude = magnitude

    def apply(self, model: torchfem.Solid):
        mask = self.region.contains(model.nodes)
        indices = torch.nonzero(mask, as_tuple=True)[0]
        num_nodes = len(indices)
        if num_nodes == 0:
            # TODO: How do we want to handle this?
            logger.warning("No nodes found to apply force")
        else:
            for i in range(len(self.magnitude)):
                model.forces[indices, i] += self.magnitude[i] / num_nodes


class Moment(Condition):
    def __init__(self, region: Region, center: list, moment: list):
        super().__init__()
        self.region = region
        self.center = torch.tensor(center, dtype=torch.float64)
        self.moment = torch.tensor(moment, dtype=torch.float64)

    def apply(self, model: torchfem.Solid):
        mask = self.region.contains(model.nodes)
        indices = torch.nonzero(mask, as_tuple=True)[0]
        num_nodes = len(indices)
        if num_nodes == 0:
            # TODO: How do we want to handle this?
            logger.warning("No nodes found to apply moment")
        else:
            r = model.nodes[mask] - self.center
            c = torch.cross(self.moment[None, :], r, dim=1)
            f = c / torch.square(torch.linalg.norm(r, dim=1))[:, None]
            model.forces[indices] += f / num_nodes


class SPC(Condition):
    def __init__(self, region: Region, dofs):
        super().__init__()
        self.region = region
        self.dofs = dofs

    def apply(self, model):
        mask = self.region.contains(model.nodes)
        indices = torch.nonzero(mask, as_tuple=True)[0]
        if len(indices) == 0:
            # TODO: How do we want to handle this?
            logger.warning("No nodes found to apply SPC")
        else:
            for dof in self.dofs:
                model.constraints[indices, dof] = True


class Displacement(Condition):
    def __init__(self, region: Region, disp):
        super().__init__()
        self.region = region
        self.disp = disp

    def apply(self, model):
        raise NotImplementedError("Displacement constraint not implemented yet")
       


class DesignResponse(ABC, torch.nn.Module):
    """
    Abstraction class for Design Responses. Can be used as Optimization Objectives or Constraints.
    Calculates the Response based on the FEM simulation result
    """
    def __init__(self):
        super().__init__()
        pass

    @abstractmethod
    def forward(self, fe_results: SimpleNamespace):
        """
            Calculates the Design Response from the given SimulationResults
        """
        pass

    def _filter_elements(self, nodes, elements, region):
        """
            Return a boolean mask specifying for each element if the centroid (average nodal) position lies inside the specified region

            Parameters
            ----------
            nodes : torch.Tensor
                Vertex coordinates of shape (N, 3).
            elements: torch.Tensor
                Tensor of shape (N, o) specifying the node indices per element. For thetrahedral elements, o=4
            region: Region
                Region for which the contained elements should be filtered
            Returns
            -------
            torch.Tensor
                Boolean array of shape (N, ). True for each element which centroid lies inside the Region, False otherwise.
        """
        # mask_nodes = region.contains(nodes)
        # mask_elements = mask_nodes[elements].all(dim=1)     # Filter elements so all nodes are inside the region
        # mask_elements = mask[elements].any(dim=1)   # Alternative implementation where at least one node has to be inside
        centroids = torch.mean(nodes[elements], dim=1)  # Calculate the centroid position of each element by taking the average along each coordinate axis
        mask_elements = region.contains(centroids)
        return mask_elements
        

        

class VolumeResponse(DesignResponse, torch.nn.Module):
    def __init__(self, region: Region | None = None):
        """
            Calculates the volume of the mesh inside the specified region.
            If no region is specified, the total volume is returned.
        """
        super().__init__()
        self.region = region

    def forward(self, fe_results):
        if self.region is None:
            relevant = fe_results.model.elements
        else:
            relevant = fe_results.model.elements[self._filter_elements(fe_results.model.nodes, fe_results.model.elements, self.region)]
        
        volume = tet_signed_vol(fe_results.model.nodes, relevant).sum()
        logger.info(f"Volume Response current: {volume:.2f}")
        
        return volume


class ComplianceResponse(DesignResponse, torch.nn.Module):
    def __init__(self, region: Region | None = None):
        super().__init__()
        self.region = region

    def forward(self, fe_results):
        if self.region is None:
            mask = torch.full((fe_results.model.nodes.shape[0],), True)
        else:
            mask = self.region.contains(fe_results.model.nodes).reshape(-1)
        compliance = torch.inner(fe_results.f[mask].ravel(), fe_results.u[mask].ravel())
        logger.info(f"Compliance Response: {compliance:.2f}")

        # def hook(grad):
        #     logger.info(f"Gradient is being computed for ComplianceResponse: {grad}")
        # compliance.register_hook(hook)

        return compliance


class MisesStressResponse(DesignResponse, torch.nn.Module):
    def __init__(self, region: Region | None = None):
        super().__init__()
        self.region = region

    def forward(self, fe_results):

        # if self.region is None:
        #     relevant = fe_results.model.elements
        # else:
        #     relevant = self._filter_elements(fe_results.model.nodes, fe_results.model.elements, self.region)
        # TODO: validate, add filtering
        cauchy = fe_results.s
        I = torch.eye(3, device=fe_results.s.device, dtype=fe_results.s.dtype)
        mean_stress = torch.diagonal(cauchy, dim1=1, dim2=2).sum(1)[:, None, None] / 3.
        deviatoric_stress = cauchy - mean_stress * I
        mises = torch.sqrt(1.5 * torch.sum(torch.square(deviatoric_stress), dim=(1, 2)))
        # mises = torch.sqrt(0.5 * (
        #     (cauchy[:, 0, 0] - cauchy[:, 1, 2]) ** 2
        #     +(cauchy[:, 1, 1] - cauchy[:, 2, 2]) ** 2
        #     +(cauchy[:, 2, 2] - cauchy[:, 0, 0]) ** 2
        #     +6*(cauchy[:, 0, 1] ** 2 + cauchy[:, 0, 2] ** 2 + cauchy[:, 1, 2] ** 2)
        # ))
        logger.info(f"Stress response, maximum: {torch.max(mises)}")
        return mises


class DisplacementResponse(DesignResponse, torch.nn.Module):
    # TODO: Add way to specify relevant displacement component
    def __init__(self, region: Region | None = None):
        super().__init__()
        self.region = region

    def forward(self, fe_results):
        if self.region is None:
            mask = torch.full((fe_results.model.nodes.shape[0],), True)
        else:
            mask = self.region.contains(fe_results.model.nodes).reshape(-1)
        disp = torch.linalg.norm(fe_results.u[mask, :])
        return disp
        




class Analysis():
    """
        Class to define an Analysis with the associated loads and boundary conditions
    """
    def __init__(self,
            conditions: list[Condition] | Condition,
            responses: list[DesignResponse] | DesignResponse,
            func   # TODO: Better name
        ):
        self.conditions = conditions if isinstance(conditions, list) else [conditions]
        self.responses = responses if isinstance(responses, list) else [responses]
        self.func = func  # TODO: evaluate function


class Topo():
    """
        Main class to run a topology optimization
        TODO: de-spaghettify
    """
    def __init__(self,
            design_domain: list[Region] | Region,
            parametrized_domain: list[Region] | Region,
            frozen_domain: list[Region] | Region,
            analyses: list[Analysis] | Analysis
        ):
        self.design_domain = design_domain if isinstance(design_domain, list) else [design_domain]
        self.parametrized_domain = parametrized_domain if isinstance(parametrized_domain, list) else [parametrized_domain]
        self.frozen_domain = frozen_domain if isinstance(frozen_domain, list) else [frozen_domain]
        self.analyses = analyses if isinstance(analyses, list) else [analyses]

    def _geometry_creation(self):
        if len(self.parametrized_domain) == 1:
            parametrized_domain_complete = self.parametrized_domain[0].sdf
        else:
            parametrized_domain_complete = UnionSDF(*[g.sdf for g in self.parametrized_domain])

        if len(self.design_domain) == 1:
            design_domain_complete = self.design_domain[0].sdf
        else:
            design_domain_complete = UnionSDF(*[g.sdf for g in self.design_domain])

        if len(self.frozen_domain) == 0:
            frozen_domain_complete = None
        elif len(self.frozen_domain) == 1:
            frozen_domain_complete = self.frozen_domain[0].sdf
        else:
            frozen_domain_complete = UnionSDF(*[g.sdf for g in self.frozen_domain])

        intersection = DifferenceSDF(
            parametrized_domain_complete,
            NegatedCallable(design_domain_complete)
        )
        if frozen_domain_complete is None:
            return intersection
        else:
            return UnionSDF(intersection, frozen_domain_complete)

    def _plot_graph(self, output_dir, history_objective, history_constraint, name):
        fig, ax1 = plt.subplots()
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Objective", color='tab:blue')
        ax1.plot(range(1, len(history_objective) + 1), history_objective, color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.set_yscale('log')

        ax2 = ax1.twinx()
        ax2.set_ylabel("Constraint", color='tab:red')
        ax2.plot(range(1, len(history_constraint) + 1), history_constraint, color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')


        fig.tight_layout()
        plt.savefig(output_dir / (name + ".png"))
        plt.close(fig)

    def _plot_pca(self, output_dir, history_params, dF, dG):
        if len(history_params) < 3:
            return

        dF = dF / np.linalg.norm(dF)
        dG = dG / np.linalg.norm(dG)

        pca = PCA(n_components=2)
        X = np.stack(history_params)
        Y = pca.fit_transform(X)

        fig, ax = plt.subplots(figsize=(10, 8))
        plt.plot(Y[:, 0], Y[:, 1], c='gray')
        plt.scatter(Y[:-1, 0], Y[:-1, 1], c='gray', marker="o", linewidths=1.5)
        plt.scatter(Y[-1, 0], Y[-1, 1], c='purple', marker="*", linewidths=4)

        origin = Y[-1]
        # dF:
        dF2d = pca.transform((history_params[-1] + dF)[None])[0]
        ax.annotate(
            "",
            xytext=origin,
            xy=dF2d,
            arrowprops=dict(arrowstyle='->', color='blue')
        )

        # dG:
        dG2d = pca.transform((history_params[-1] + dG)[None])[0]
        ax.annotate(
            "",
            xytext=origin,
            xy=dG2d,
            arrowprops=dict(arrowstyle='->', color='red')
        )
        
        plt.savefig(output_dir / "PCA.png")
        plt.close(fig)

        return False

    def _plot_mesh(self, output_dir, mesh, title):
        plotter = pv.Plotter(off_screen=True)
        plotter.add_mesh(
            mesh,
            scalars=mesh.cell_data['mises'],
            show_edges=True,
            edge_opacity=0.5,
            clim=[0, np.percentile(mesh.cell_data['mises'], 99)],  # Limit to 99% percentile since there usually are some outliers
            cmap='turbo',
            scalar_bar_args={
                'title': "Von Mises Stress",
            },
        )
        plotter.screenshot(str(output_dir / title) + "_mises.png")
        plotter.close()
        del plotter

        # Instantiate new plotter instead of reusing it, since some states get carried over otherwise
        plotter = pv.Plotter(off_screen=True)
        plotter.add_mesh(
            mesh,
            scalars=np.linalg.norm(
                mesh.point_data["displacement"],
                axis=1
            ),
            show_edges=True,
            edge_opacity=0.5,
            cmap='turbo',
            scalar_bar_args={
                "title": "Displacement",
            },
        )
        plotter.screenshot(str(output_dir / title) + "_displacement.png")
        plotter.close()
        del plotter

    def _get_pyvista_mesh(self, fe_results):
        mesh = get_mesh_from_torchfem(fe_results.model)

        mesh.point_data["displacement"] = fe_results.u.detach().cpu().numpy()

        mises = MisesStressResponse().forward(fe_results).detach().cpu().numpy()  # Uses StressResponse to calculate mises stress
        mesh.cell_data["mises"] = mises

        mesh.cell_data["stress"] = fe_results.s.detach().cpu().numpy()

        return mesh


    def run(self,
            output_dir='./',
            plot_graph=False,
            override_graph=True,
            plot_mesh=False,
            override_mesh_plot=True,
            export_mesh=False,
            override_mesh_export=True
        ):
        # TODO: support multiple desing domains, right now quick & dirty for only one
        params = self.parametrized_domain[0].parametrization
        for _, region in enumerate(self.parametrized_domain[1:]):
            if region.parametrization is not None:
                torch.stack([params, region.parametrization])  # TODO: research if legal

        if params is None:
            raise ValueError("No region with parametrization given")
        param_bounds = np.zeros(params.reshape(-1, 1).shape) + np.array([-1.0, 1.0])
        optimizer = MMA(params.reshape(-1, 1), param_bounds, max_step=0.005)    # TODO: make parameter for max step

        history_objective = []
        history_constraint = []
        history_params = []

        for i in range(1, 801):
            logger.info(
                f"Starting iteration with parameters: "
                f"shape={tuple(params.shape)}, mean={params.mean().item():.4f}, std={params.std().item():.4f}"
            )
    
            # Show first 10 values (flattened)
            logger.debug(
                f"First 10 parameter values: {[round(x, 4) for x in params.flatten()[:10].tolist()]}"
            )

            torch.set_default_dtype(torch.float32)
            geom = self._geometry_creation()

            export_sdf_grid_vtk(geom, str(output_dir / "current_complete_sdf.vtk"))

            mesh, _ = create_3D_mesh(
                geom,
                64,  # TODO: Not hardcoded
                mesh_type="volume",
                differentiate=False,
                device='cpu'   # TODO: as optional parameter
            )
            # mesh.remove_disconnected_regions(clear_unused=True)

            gus.io.meshio.export(str( output_dir / "current_complete_mesh.vtk"), mesh.to_gus())

            torch.set_default_dtype(torch.float64)
            verts = mesh.vertices.double()   # Convert to double for torchfem pardiso solver
            tets = mesh.volumes
            material = IsotropicElasticity3D(E=210000.0, nu=0.342)
            model = Solid(verts, tets, material)


            objectives = []
            constraints = []
            for j, analysis in enumerate(self.analyses):
                # Reset BC / Loads:
                model.constraints[:] = False
                model.forces[:] = 0.
                model.displacements[:] = 0.

                # Apply new BC / Loads:
                for condition in analysis.conditions:
                    condition.apply(model)

                logger.info(f"Solving FE simulation..")
                # TODO: Sometimes after many optimization steps, this breaks because of "Matrix A is singular because it contains empty rows" -> disconnected load introduction?
                u, f, s, F, a = model.solve(rtol=0.001, device="cpu", method="pardiso", verbose=True)
                fe_results = SimpleNamespace(u=u, f=f, s=s, F=F, a=a, model=model)

                responses = []
                for response in analysis.responses:
                    responses.append(response.forward(fe_results))  # Evaluate Design Responses
                objective, constraint = analysis.func(responses, i)    # Calculate objective & constraint
                if objective is not None: objectives.append(objective)
                if constraint is not None: constraints.append(constraint)


                if plot_mesh or export_mesh:
                    mesh = self._get_pyvista_mesh(fe_results)
                    if plot_mesh:
                        name = f"current_mesh_analysis_{j}"
                        if not override_mesh_plot:
                            name += f"_iteration_{i:03}"
                        self._plot_mesh(output_dir, mesh, name)
                    if export_mesh:
                        name = f"current_mesh_analysis_{j}"
                        if not override_mesh_export:
                            name += f"_iteration_{(i):03}"
                        mesh.save(str(output_dir / name) + ".vtk")
                

            logger.info(f"objectives: {objectives}")
            logger.info(f"constraints: {constraints}")

            if not objectives:
                raise ValueError("No objective given")
            if not constraints:
                raise ValueError("No constraints given")

            # For now, simply sum over all objectives / constraints
            F = torch.stack(objectives).sum()
            G = torch.stack(constraints).sum()
            dF = torch.autograd.grad(F, params, retain_graph=True)[0]
            dG = torch.autograd.grad(G, params, retain_graph=True)[0]
            optimizer.step(F, dF, G, dG)


            logger.info(f"========STEP INFO========")
            logger.info(f"dF norm: {torch.norm(dF)}, variance: {torch.var(dF)}")
            logger.info(f"dG norm: {torch.norm(dG)}, variance: {torch.var(dG)}")
            logger.info(f"objective & constraint consistency: {torch.dot(dF.reshape(-1), dG.reshape(-1)) / (torch.norm(dF) * torch.norm(dG))}")


            history_params.append(params.reshape(-1).detach().cpu().numpy())
            # self._plot_pca(output_dir, history_params, dF.reshape(-1).detach().cpu().numpy(), dG.reshape(-1).detach().cpu().numpy())

            history_objective.append(float(F.detach().cpu()))
            history_constraint.append(float(G.detach().cpu()))

            if plot_graph:
                name = f"optimization_history"
                if not override_graph:
                    name += f"_iteration_{(i):03}"
                self._plot_graph(output_dir, history_objective, history_constraint, name)

            
            logger.info(f"iteration {i} finished. Objective: {F}, gradient norm: {dF.norm()}, Constraint: {G}, gradient norm: {dG.norm()}")
