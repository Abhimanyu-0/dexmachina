import os
from pathlib import Path
try:
    # Python 3.9+
    from importlib.resources import files
except ImportError:
    # Python < 3.9
    from importlib_resources import files

def get_asset_path(asset_name):
    """
    Get the full path to an asset file.
    
    Args:
        asset_name (str): Relative path from the assets directory
                         e.g., 'inspire_hand/left.urdf'
    
    Returns:
        Path: Full path to the asset file
    """
    try:
        # Modern approach (Python 3.9+)
        assets_dir = files('dexmachina') / 'assets'
        return assets_dir / asset_name
    except:
        # Fallback for older Python or if importlib.resources fails
        import pkg_resources
        return Path(pkg_resources.resource_filename('dexmachina', f'assets/{asset_name}'))

def get_urdf_path(urdf_name):
    """Convenience function specifically for URDF files."""
    return get_asset_path(urdf_name)
