# Changes to DexMachina

## Bug Fixes

### Removed unsupported Genesis 0.3.3 RigidOptions fields (2025-10-03)

**Files:**
- `dexmachina/envs/base_env.py:212-213`
- `dexmachina/retargeting/parallel_retarget.py:69-72`
- `dexmachina/hand_proc/inspect_raw_urdf.py:300-319`

**Issue:** Running `bash examples/train_rl.sh` resulted in:
```
ValueError: "RigidOptions" object has no field "self_collision_group_filter"
```

**Root Cause:** Genesis 0.3.3 doesn't support `self_collision_group_filter` and `link_group_mapping` fields in `RigidOptions`. These fields were being set in the code, causing the ValueError.

**Fix:** Commented out the unsupported fields and kept only `enable_self_collision = True`, which is supported in Genesis 0.3.3.

**Changed lines:**
```python
# Before:
scene_cfg['rigid_options'].self_collision_group_filter = True
scene_cfg['rigid_options'].link_group_mapping = collision_groups

# After:
# Note: self_collision_group_filter and link_group_mapping not available in Genesis 0.3.3
scene_cfg['rigid_options'].enable_self_collision = True
```

### Fixed ValueError in Genesis rigid solver (2025-10-03)

**File:** `Genesis/genesis/engine/solvers/rigid/rigid_solver_decomp.py:639-642`

**Issue:** Running `python examples/inspect_hand.py --hand allegro_hand --vis` resulted in:
```
ValueError: setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions. The detected shape was (29,) + inhomogeneous part.
```

**Root Cause:** Some links had `None` values for `inertial_pos`, `inertial_quat`, `inertial_i`, and `inertial_mass` properties, which caused numpy to fail when creating arrays.

**Fix:** Added None-handling with appropriate defaults:
- `inertial_pos`: Default to `np.zeros(3)` (zero position)
- `inertial_quat`: Default to `np.array([0., 0., 0., 1.])` (identity quaternion)
- `inertial_i`: Default to `np.eye(3)` (identity inertia matrix)
- `inertial_mass`: Default to `0.0` (zero mass)

**Changed lines:**
```python
# Before:
links_inertial_pos=np.array([link.inertial_pos for link in links], dtype=gs.np_float),
links_inertial_quat=np.array([link.inertial_quat for link in links], dtype=gs.np_float),
links_inertial_i=np.array([link.inertial_i for link in links], dtype=gs.np_float),
links_inertial_mass=np.array([link.inertial_mass for link in links], dtype=gs.np_float),

# After:
links_inertial_pos=np.array([link.inertial_pos if link.inertial_pos is not None else np.zeros(3) for link in links], dtype=gs.np_float),
links_inertial_quat=np.array([link.inertial_quat if link.inertial_quat is not None else np.array([0., 0., 0., 1.]) for link in links], dtype=gs.np_float),
links_inertial_i=np.array([link.inertial_i if link.inertial_i is not None else np.eye(3) for link in links], dtype=gs.np_float),
links_inertial_mass=np.array([link.inertial_mass if link.inertial_mass is not None else 0.0 for link in links], dtype=gs.np_float),
```

### Fixed Genesis 0.3.3 contact API compatibility (2025-10-03)

**File:** `dexmachina/envs/contacts.py:179-242`

**Issue:** Training failed with:
```
AttributeError: 'Collider' object has no attribute 'contact_data'
```

**Root Cause:** Genesis 0.3.3 changed the contact data API:
- Old API: `solver.collider.contact_data` (direct attribute access with nested `.to_torch()` calls)
- New API: `solver.collider.get_contacts(as_tensor=True, to_torch=True)` (method returning dict)
- Also, tensor shapes changed from `(n_contacts_max, n_envs, ...)` to `(n_envs, n_contacts_max, ...)`

**Fix:** Updated `get_filtered_contacts()` function to use the new API:

1. **Contact data access:**
   ```python
   # Before:
   contact_data = entity_a._solver.collider.contact_data
   force = contact_data.force.to_torch(device=device)
   geom_a = contact_data.geom_a.to_torch(device=device)
   contact_pos = contact_data.pos.to_torch(device=device)

   # After:
   contact_data = entity_a._solver.collider.get_contacts(as_tensor=True, to_torch=True)
   force = contact_data['force'].transpose(0, 1)
   geom_a = contact_data['geom_a'].transpose(0, 1)
   contact_pos = contact_data['position'].transpose(0, 1)
   ```

2. **n_contacts access:**
   ```python
   # Before:
   n_contacts = entity_a._solver.collider.n_contacts.to_torch(device=device)

   # After:
   n_contacts_raw = entity_a._solver.collider._collider_state.n_contacts.to_numpy()
   n_contacts = torch.tensor(n_contacts_raw, dtype=torch.int32, device=device)
   ```

3. **Key changes:**
   - Changed dict key `'pos'` → `'position'`
   - Added `.transpose(0, 1)` to all contact tensors to convert from `(n_envs, n_contacts_max, ...)` to `(n_contacts_max, n_envs, ...)`
   - Contact data now returns as dict with keys: `'link_a'`, `'link_b'`, `'geom_a'`, `'geom_b'`, `'penetration'`, `'position'`, `'normal'`, `'force'`

### Fixed Genesis DOF gains tensor shape incompatibility (2025-10-06)

**File:** `dexmachina/envs/robot.py:410-421`

**Issue:** Running `python retargeting/parallel_retarget.py --clip $CLIP --hand ${HAND} --control_steps 2000 --save_name para --save -ow` resulted in:
```
genesis.GenesisException: Expecting 1D output tensor.
```

**Error trace:**
```
File "envs/robot.py", line 457, in post_scene_build_setup
    self.set_dof_gains_by_group(self.cfg["actuators"])
File "envs/robot.py", line 483, in set_dof_gains_by_group
    self.set_joint_gains(kp, kv, fr, joint_idxs)
File "envs/robot.py", line 421, in set_joint_gains
    self.entity.set_dofs_kp(batched_kp, joint_idxs)
File "Genesis/genesis/engine/solvers/rigid/rigid_solver_decomp.py", line 1568
    gs.raise_exception("Expecting 1D output tensor.")
```

**Root Cause:** Genesis expects **different tensor shapes based on the number of environments**, not just the `batch_dofs_info` setting:
- `num_envs == 1` (retargeting): expects **1D tensors** with shape `(num_joints,)`
- `num_envs > 1` (RL training): expects **2D tensors** with shape `(num_envs, num_joints)` for batched operations

The original code was creating 2D tensors `(n_envs, n_joints)` which worked for multi-env scenarios but broke for single-env retargeting.

**First Fix (for retargeting only):** Changed to 1D tensors - this fixed retargeting but broke RL training

**Final Fix (for both):** Added conditional logic to handle both cases:

```python
# Before (broke retargeting):
batched_kp = torch.tensor(
    [[kp]*num_joints]*self.num_envs, dtype=torch.float32, device=self.device
)  # Shape: (num_envs, num_joints) - 2D always
batched_kv = torch.tensor(
    [[kv]*num_joints]*self.num_envs, dtype=torch.float32, device=self.device
)
fr = torch.tensor(
    [[fr]*num_joints]*self.num_envs, dtype=torch.float32, device=self.device
)

self.entity.set_dofs_kp(batched_kp, joint_idxs)
self.entity.set_dofs_kv(batched_kv, joint_idxs)
self.entity.set_dofs_force_range(-1.0 * fr, fr, joint_idxs)

# After (works for both retargeting and RL):
# Genesis expects different shapes based on number of environments:
# - num_envs > 1: 2D tensor (num_envs, num_joints) for batched operations
# - num_envs == 1: 1D tensor (num_joints,) for single environment
if self.num_envs > 1:
    gains_kp = torch.full((self.num_envs, num_joints), kp, dtype=torch.float32, device=self.device)
    gains_kv = torch.full((self.num_envs, num_joints), kv, dtype=torch.float32, device=self.device)
    gains_fr = torch.full((self.num_envs, num_joints), fr, dtype=torch.float32, device=self.device)
else:
    gains_kp = torch.full((num_joints,), kp, dtype=torch.float32, device=self.device)
    gains_kv = torch.full((num_joints,), kv, dtype=torch.float32, device=self.device)
    gains_fr = torch.full((num_joints,), fr, dtype=torch.float32, device=self.device)

self.entity.set_dofs_kp(gains_kp, joint_idxs)
self.entity.set_dofs_kv(gains_kv, joint_idxs)
self.entity.set_dofs_force_range(-1.0 * gains_fr, gains_fr, joint_idxs)
```

**Why it broke twice:**
1. **First break (retargeting)**: Original 2D code failed for single-env scenarios (`num_envs=1`)
2. **Second break (RL training)**: Fixed to 1D-only, which broke multi-env scenarios (`num_envs>1`)
3. **Final solution**: Conditional logic handles both single-env and multi-env cases

**Why the change was needed:**
- Genesis API has an undocumented quirk where tensor shape requirements change based on environment count
- Single-env retargeting needs 1D tensors; multi-env RL training needs 2D tensors
- The tensor shape validation in `_sanitize_1D_io_variables()` (rigid_solver_decomp.py:1546-1568) strictly enforces this

### Fixed segmentation fault with laptop retargeting (2025-10-06)

**File:** `dexmachina/retargeting/parallel_retarget.py:60`

**Issue:** Running `python retargeting/parallel_retarget.py --clip laptop-30-230-s01-u01 --hand xhand --control_steps 2000 --save_name para --save -ow` resulted in:
```
Segmentation fault (core dumped)
```

**Error warning before crash:**
```
[Genesis] [18:35:55] [WARNING] max_collision_pairs 100 is smaller than the theoretical maximal possible pairs 461, it uses less memory but might lead to missing some collision pairs if there are too many collision pairs
```

**Root Cause:** Genesis pre-allocates a fixed-size GPU memory buffer for collision detection results. With `max_collision_pairs=100`:
- Genesis allocated memory for only 100 collision pairs
- The laptop demo (articulated object with 2 parts + 2 hands × multiple links) creates ~461 potential collision pairs
- When Genesis tried to write collision data beyond the 100th pair, it wrote **outside the allocated buffer** → **segmentation fault** (illegal memory access)

**Why box worked but laptop didn't:**
- Box: Simple geometry, few collision pairs (< 100) ✓
- Laptop: Articulated object (2 parts: top/bottom), complex decomposed collision meshes, 461+ collision pairs ✗

**Fix:** Increased the collision pair buffer size:

```python
# Before:
max_collision_pairs=100, # default was 100

# After:
max_collision_pairs=512, # increased from 100 to handle laptop's complex collision geometry
```

**Why the change was needed:**
- 512 > 461, so all collision pairs fit within the allocated buffer
- No memory overflow → no segfault
- Trade-off: Uses slightly more GPU memory, but prevents crashes with complex articulated objects
- The warning message was helpful - it indicated exactly how many pairs were needed (461)
