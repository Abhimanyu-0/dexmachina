# [BUG FIX] Handle None values in link inertial properties

## Description
Fixes #<ISSUE_NUMBER> - ValueError when loading URDF files with links missing inertial properties.

## Changes
Modified `genesis/engine/solvers/rigid/rigid_solver_decomp.py` (lines 639-642) to handle `None` values in link inertial properties by providing sensible defaults:

- `inertial_pos`: Defaults to `np.zeros(3)` (zero position)
- `inertial_quat`: Defaults to `np.array([1.0, 0.0, 0.0, 0.0])` (identity quaternion, w-first convention)
- `inertial_i`: Defaults to `np.eye(3)` (identity inertia matrix)
- `inertial_mass`: Defaults to `0.0` (zero mass)

## Testing
Tested with URDF files that have links without `<inertial>` tags:
- Allegro Hand
- Shadow Hand
- Custom hand models

### Test Command
```bash
python test_urdf_load.py --urdf path/to/allegro_hand_left.urdf
```

### Before Fix
```
ValueError: setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions.
```

### After Fix
✅ URDF loads successfully with default inertial values for links missing explicit inertial properties.

## Backward Compatibility
This change is backward compatible - URDFs with complete inertial properties continue to work as before, while URDFs with missing properties now work correctly instead of crashing.
