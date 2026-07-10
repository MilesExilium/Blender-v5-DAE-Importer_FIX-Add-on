# Blender v5 DAE / COLLADA Importer & Exporter

A Blender 5-compatible add-on that restores practical `.dae` / COLLADA import and export support after Blender removed the old native COLLADA tools.

Originally created by `/u/varyingopinions` on Reddit and `ekztal` on GitHub. Extended and reworked by the community for newer Blender versions.

## Download

Use this file:

`DAE_Importer_Exporter_Ver42.py`

Do not use the older `DAE_Importer_Ver34.py` or `DAE_Importer_Ver35.py` files unless you specifically need the old importer-only version.

## Installation

1. Download `DAE_Importer_Exporter_Ver42.py`.
2. Open Blender.
3. Go to `Edit > Preferences > Add-ons`.
4. Click `Install...`.
5. Select `DAE_Importer_Exporter_Ver42.py`.
6. Enable the add-on.

Import:

`File > Import > DAE / COLLADA`

Export:

`File > Export > DAE / COLLADA`

## Features

- Import and export `.dae` / COLLADA files
- Mesh geometry import/export
- Materials and texture references
- Multiple UV maps
- Vertex colors
- Custom normals
- Object hierarchy and transforms
- Unit and axis conversion
- Cameras and lights
- Armatures, bones, skin controllers, and vertex weights
- Shape keys / morph controller support
- Object and bone animation support
- Batch import of multiple DAE files
- Optional collection creation per imported file
- Safer batch import error handling, so one broken file does not stop the whole batch
- Support for common game/exporter primitive types, including `triangles`, `polylist`, `polygons`, `tristrips`, and `trifans`
- Fallback geometry import for DAE files with no visual scene
- Faster large mesh import using Blender bulk data APIs

## What changed in v4.2.2

This version expands the original importer into a fuller importer/exporter with stronger compatibility for Blender 5, Blender 4.5-style COLLADA files, game-exported DAE files, and 3ds Max round-trip workflows.

Main fixes include:

- Fixed add-on registration issues
- Fixed missing or broken `bl_info` metadata problems
- Added DAE export support
- Added multi-file batch import
- Added multiple UV map support
- Improved transform, unit, and axis handling
- Improved armature, bone, skin-weight, and bind-pose handling
- Added shape key / morph controller import and export
- Added animation import/export support
- Added camera and light support
- Added fallback handling for files that contain geometry but no visual scene
- Added support for triangle strips and triangle fans
- Improved texture path handling, including Windows-style paths exported by 3ds Max
- Improved performance on large meshes
- Improved error reporting for partial or malformed DAE files
- Changed imported armatures to use a less intrusive stick display by default, so long helper/root bones do not visually cover game characters on first import
- Fixed Blender 2.7x / BOTW-style zero-translation bind-shape matrices that could transpose incorrectly and split upper-body meshes away from the rest of the character

## Contributors

- `/u/varyingopinions` / `ekztal` - original Simple COLLADA importer foundation
- MilesExilium - extended and maintained the Blender 5 importer fork
- RebeccaNod1 - importer fixes and compatibility work
- Zack Wilde / `ZackWilde27` - multi-file import contribution in the earlier importer fork
- XDM-Inc

## Compatibility

Tested with:

- Blender 5
- Blender 4.5 native COLLADA compatibility checks
- Autodesk 3ds Max 2024 COLLADA import/export smoke test

Validated areas include:

- Add-on registration
- Mesh import/export
- Materials and textures
- Multiple UV maps
- Cameras and lights
- Armatures and skin weights
- Shape keys / morph animation
- Batch import
- Large mesh import performance
- Blender-to-3ds-Max-to-Blender DAE round trip

## Known limitations

COLLADA is a broad format and different tools export it in different ways. This add-on handles many common Blender, 3ds Max, and game-exporter patterns, but unusual vendor-specific DAE files may still need sample-based fixes.

Autodesk 3ds Max 2024 did not preserve standard COLLADA morph targets during the tested Max round trip. Skin, UVs, hierarchy, cameras, lights, and mesh data were preserved, but Max did not recreate a Morpher modifier from the COLLADA morph controller.

If you find a DAE file that fails, please open an issue and include:

- The `.dae` file, if possible
- Blender version
- Exporting tool, if known
- Console error output
- Whether the issue happens on import, export, or round trip

## Issue categories addressed

This release addresses the main reported problem areas:

- Add-on not showing in Blender
- Missing commas or invalid add-on metadata
- Missing multi-UV import
- Incorrect scale, object position, or object center behavior
- Rig not importing
- Files importing only the armature without the mesh
- Slow imports on larger files
- No multi-file import support
- Game-style DAE primitive handling
- Generic or unclear import errors

## License

Use the same license terms as the original project unless the repository owner updates the license.
