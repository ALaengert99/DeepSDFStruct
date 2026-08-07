import sys
from pathlib import Path
import gustaf as gus
import trimesh
import numpy as np
import torch


# Add parent directory to import DeepSDFStruct
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from DeepSDFStruct.optimization import Region, SPC, Force, Moment, Analysis, VolumeResponse, ComplianceResponse, Topo, parametrize_region, MisesStressResponse
from DeepSDFStruct.pretrained_models import get_model, PretrainedModels
from DeepSDFStruct.geom_reconstruction import LocalShapesReconstructor
import pymeshfix

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.set_default_dtype(torch.float32)


base_dir = Path(__file__).resolve().parent
tmp_dir = base_dir / 'tmp'

# recon = LocalShapesReconstructor(output_dir=tmp_dir, device='cpu')
# mesh_gus = gus.io.meshio.load(base_dir / 'ge_engine_hypermesh.stl')
# mesh_fix = pymeshfix.MeshFix(mesh_gus.vertices, mesh_gus.faces)
# # mesh_fix.repair()   # Make the mesh watertight using pymeshfix
# mesh = trimesh.Trimesh(mesh_fix.points, mesh_fix.faces)

# trimesh.exchange.export.export_mesh(mesh, str( tmp_dir / "before_fit.stl"), 'stl')

# # assert mesh.is_watertight, "Mesh is not watertight!"
# struct, scaling, gt_sdf, params = recon.fit_mesh(
#     mesh=mesh,
#     tiling=[8, 8, 8]
# )
# # recon.export(struct, scaling)

mesh = gus.io.meshio.load(base_dir / 'constraints_only.stl')
bolt_region = Region.create(trimesh.Trimesh(mesh.vertices, mesh.faces))
load_region = Region.create(gus.io.meshio.load(base_dir / 'load_points_only.stl'), threshold=0.1)
design_domain = Region.create(gus.io.meshio.load(base_dir / 'ge_engine_hypermesh.stl'))
starting_geometry = Region.create(gus.io.meshio.load(base_dir / 'inclusions.stl'))
parametrized_domain = parametrize_region(
    starting_geometry,
    save_dir=tmp_dir
)


bolt = SPC(bolt_region, [0, 1, 2])
force1 = Force(load_region, [0, 0, 8000])
force2 = Force(load_region, [0, -8500, 0])
force3 = Force(load_region, [0, -9500 * np.sin(np.deg2rad(42)), 9500 * np.cos(np.deg2rad(42))])
force4 = Moment(load_region, [0, 0, 0], [0, 5000, 0])

vol0 = design_domain.mesh.volume
vol_target = vol0 * 0.5
volume_response = VolumeResponse()
compliance_response = ComplianceResponse()
stress_response = MisesStressResponse()
  

analysis1 = Analysis([bolt, force1], [compliance_response, volume_response], lambda res: (res[0], res[1] / vol_target - 1))
analysis2 = Analysis([bolt, force2], compliance_response, lambda res: (res[0], None))
analysis3 = Analysis([bolt, force3], compliance_response, lambda res: (res[0], None))
# analysis4 = Analysis([bolt, force4], [compliance_response, stress_response], lambda res: (res[0], torch.linalg.norm(res[1], ord=8) / 1e5 - 1))
analysis4 = Analysis([bolt, force4], [compliance_response, stress_response], lambda res: (res[0], None))    



optimization = Topo(
    design_domain=design_domain,
    parametrized_domain=parametrized_domain,
    frozen_domain=[bolt_region, load_region],
    analyses=[analysis1, analysis2, analysis3, analysis4],
)
optimization.run(
    output_dir=tmp_dir,
    plot_graph=True,
    plot_mesh=True,
    export_mesh=True
)