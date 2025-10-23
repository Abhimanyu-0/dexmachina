# Bug Report: ValueError when loading URDF with links missing inertial properties

## Environment
- **OS**: Linux 6.8.0-85-generic
- **Genesis Version**: 0.3.3
- **Python Version**: 3.10.12

## Description
When loading certain URDF files (particularly hand models like Allegro Hand), Genesis crashes with a `ValueError` due to improper handling of `None` values in link inertial properties.

## Error Message
```
ValueError: setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions. The detected shape was (29,) + inhomogeneous part.
```

## Error Traceback
```
File "genesis/engine/solvers/rigid/rigid_solver_decomp.py", line 639
    links_inertial_pos=np.array([link.inertial_pos for link in links], dtype=gs.np_float),
File "genesis/engine/solvers/rigid/rigid_solver_decomp.py", line 640
    links_inertial_quat=np.array([link.inertial_quat for link in links], dtype=gs.np_float),
File "genesis/engine/solvers/rigid/rigid_solver_decomp.py", line 641
    links_inertial_i=np.array([link.inertial_i for link in links], dtype=gs.np_float),
File "genesis/engine/solvers/rigid/rigid_solver_decomp.py", line 642
    links_inertial_mass=np.array([link.inertial_mass for link in links], dtype=gs.np_float),
```

## Root Cause
In `genesis/engine/solvers/rigid/rigid_solver_decomp.py` (lines 639-642), the code creates numpy arrays directly from link properties without checking for `None` values:

```python
links_inertial_pos=np.array([link.inertial_pos for link in links], dtype=gs.np_float),
links_inertial_quat=np.array([link.inertial_quat for link in links], dtype=gs.np_float),
links_inertial_i=np.array([link.inertial_i for link in links], dtype=gs.np_float),
links_inertial_mass=np.array([link.inertial_mass for link in links], dtype=gs.np_float),
```

Some URDF links may have `None` values for these properties (particularly for links without explicitly defined `<inertial>` tags), which causes numpy to fail when creating homogeneous arrays.

## Steps to Reproduce

```python
import genesis as gs
import numpy as np

# Initialize Genesis
gs.init(backend=gs.gpu)

# Create a scene with rigid body physics
scene = gs.Scene(
    sim_options=gs.options.SimOptions(
        dt=1/60,
        substeps=2,
        gravity=(0, 0, 0),
    ),
    rigid_options=gs.options.RigidOptions(
        enable_self_collision=True,
        enable_joint_limit=True,
    ),
    show_viewer=True,
)

# Load a URDF with missing inertial properties (e.g., Allegro Hand)
# Replace with path to your URDF file that has links without <inertial> tags
urdf_path = "path/to/allegro_hand_left.urdf"

hand = scene.add_entity(
    gs.morphs.URDF(
        file=urdf_path,
        pos=(0.0, 0.0, 1.0),
        quat=(1, 0, 0, 0),
        fixed=True,
        convexify=True,
        merge_fixed_links=False,
        recompute_inertia=True,
    )
)

# This will trigger the error
scene.build(n_envs=1)
```

## Expected Behavior
Genesis should handle links with missing inertial properties gracefully by using sensible defaults.

## Proposed Fix
Add None-handling with appropriate defaults in `genesis/engine/solvers/rigid/rigid_solver_decomp.py` (lines 639-642):

- `inertial_pos`: Default to `np.zeros(3)` (zero position)
- `inertial_quat`: Default to `np.array([1.0, 0.0, 0.0, 0.0])` (identity quaternion, w-first convention)
- `inertial_i`: Default to `np.eye(3)` (identity inertia matrix)
- `inertial_mass`: Default to `0.0` (zero mass)

```python
# Before:
links_inertial_pos=np.array([link.inertial_pos for link in links], dtype=gs.np_float),
links_inertial_quat=np.array([link.inertial_quat for link in links], dtype=gs.np_float),
links_inertial_i=np.array([link.inertial_i for link in links], dtype=gs.np_float),
links_inertial_mass=np.array([link.inertial_mass for link in links], dtype=gs.np_float),

# After:
links_inertial_pos=np.array([link.inertial_pos if link.inertial_pos is not None else np.zeros(3) for link in links], dtype=gs.np_float),
links_inertial_quat=np.array([link.inertial_quat if link.inertial_quat is not None else np.array([1.0, 0.0, 0.0, 0.0]) for link in links], dtype=gs.np_float),
links_inertial_i=np.array([link.inertial_i if link.inertial_i is not None else np.eye(3) for link in links], dtype=gs.np_float),
links_inertial_mass=np.array([link.inertial_mass if link.inertial_mass is not None else 0.0 for link in links], dtype=gs.np_float),
```

## Additional Context
This issue affects URDF files where some links (particularly cosmetic or visualization-only links) do not define complete inertial properties. The fix ensures backward compatibility while allowing Genesis to handle a wider variety of URDF files.

Common URDF files affected:
- Allegro Hand
- Shadow Hand
- Custom hand models with simplified link definitions
