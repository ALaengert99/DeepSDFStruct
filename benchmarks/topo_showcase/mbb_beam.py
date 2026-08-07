import sys
from pathlib import Path
import gustaf as gus
import trimesh
import numpy as np
import torch


# Add parent directory to import DeepSDFStruct
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from DeepSDFStruct.optimization import Region, SPC, Force, Moment, Analysis, VolumeResponse, ComplianceResponse, Topo, MisesStressResponse, DisplacementResponse
from DeepSDFStruct.sdf_primitives import BoxSDF

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.set_default_dtype(torch.float32)


base_dir = Path(__file__).resolve().parent
tmp_dir = base_dir / 'mbb_output'

beam = Region.create(BoxSDF(
        center=[30., 10., 0.],
        extents=[60., 20., 10.]
))
parametrized = beam.parametrize(
    tiling=[6, 2, 1],
    save_dir=tmp_dir
)

roller_region = Region.create(BoxSDF(
    center=[60. - 5. / 2, 0., 0.],
    extents=[5., 2., 10.]
))
symmetry_region = Region.create(BoxSDF(
    center=[0., 10., 0.],
    extents=[2., 20., 10.]
))
load_region = Region.create(BoxSDF(
    center=[0. + 3. / 2, 18., 0.],
    extents=[3., 2., 10.]
))

bc_roller = SPC(roller_region, [1, 2])
bc_symmetry = SPC(symmetry_region, [0, 2])
load = Force(load_region, [0., -30000., 0.])

vol0 = beam.mesh.volume

analysis = Analysis([bc_roller, bc_symmetry, load], [DisplacementResponse(load_region), VolumeResponse()], lambda res: (res[0].sum(), res[1] / (vol0 * 0.3) - 1))

optimization = Topo(
    design_domain=beam,
    parametrized_domain=parametrized,
    frozen_domain=[load_region],
    analyses=analysis
)
optimization.run(
    output_dir=tmp_dir,
    plot_graph=True,
    plot_mesh=True,
    export_mesh=True
)