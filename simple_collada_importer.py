bl_info = {
    "name": "Simple COLLADA (.dae) Importer (Positions + Normals + Colors + UVs + Textures + Rig)",
    "author": "ekztal",
    "additional help": "MilesExilium",
    "version": (0, 9, 6),
    "blender": (4, 0, 0),
    "location": "File > Import > Simple COLLADA (.dae)",
    "description": "Imports COLLADA meshes with textures, armature, and skin weights.",
    "category": "Import-Export",
    "support": "COMMUNITY",
}

import os
import math
import bpy
from bpy_extras.io_utils import ImportHelper
from bpy.types import Operator
from bpy.props import StringProperty, BoolProperty, FloatProperty
from mathutils import Vector, Matrix
import xml.etree.ElementTree as ET


# ---------------------- XML/NAMESPACE HELPERS ----------------------

def get_collada_ns(root):
    """Return COLLADA namespace prefix '{...}' or empty."""
    if root.tag.startswith("{"):
        return root.tag.split("}")[0] + "}"
    return ""


def q(ns, tag):
    """Qualify XML tag with namespace."""
    return f"{ns}{tag}"


def parse_source_float_array(source_elem, ns):
    """
    Parse <source><float_array>...</float_array></source>
    Handles stride from <accessor>.
    Returns list of tuples (length = stride).
    """
    float_array = source_elem.find(q(ns, "float_array"))
    if float_array is None or float_array.text is None:
        return []
    raw_vals = float_array.text.strip().split()
    try:
        floats = [float(v) for v in raw_vals]
    except ValueError:
        return []
    accessor = source_elem.find(f"{q(ns,'technique_common')}/{q(ns,'accessor')}")
    stride = int(accessor.attrib.get("stride", "3")) if accessor is not None else 3
    out = []
    for i in range(0, len(floats), stride):
        chunk = floats[i:i+stride]
        if len(chunk) < stride:
            break
        out.append(tuple(chunk))
    return out


def parse_matrix(text):
    """Parse a 16-float COLLADA row-major matrix string into a Blender Matrix."""
    vals = [float(v) for v in text.strip().split()]
    if len(vals) != 16:
        return Matrix.Identity(4)
    return Matrix([vals[0:4], vals[4:8], vals[8:12], vals[12:16]])


def get_up_axis_matrix(root, ns):
    """
    Return a 4x4 correction Matrix to bring the DAE coordinate system into
    Blender's Z-up right-handed space.
      Z_UP: identity  (already correct)
      Y_UP: rotate +90° around X  (most exporters)
      X_UP: rotate -90° around Y
    If no <up_axis> tag is present, default to Z_UP (no correction) since
    most game-rip exporters that omit the tag are already in Z-up space.
    """
    asset = root.find(q(ns, "asset"))
    up    = asset.find(q(ns, "up_axis")) if asset is not None else None
    axis  = up.text.strip().upper() if (up is not None and up.text) else "Z_UP"
    if axis == "Z_UP":
        return Matrix.Identity(4)
    elif axis == "X_UP":
        return Matrix.Rotation(-math.pi / 2.0, 4, 'Y')
    else:   # Y_UP
        return Matrix.Rotation(math.pi / 2.0, 4, 'X')


# ---------------------- MATERIAL / TEXTURE HELPERS ----------------------

def extract_material_texture_map(root, ns):
    """
    Returns dict: material_id -> {"diffuse": path, "normal": path, "ao": path, "specular": path}
    Reads library_images -> library_effects (sampler/surface chain) -> library_materials.
    Handles both standard <diffuse> and FCOLLADA <extra><bump> for normal maps.
    """

    # 1. image_id -> file path
    image_path_for_id = {}
    for img in root.findall(f".//{q(ns,'image')}"):
        img_id = img.attrib.get("id")
        if not img_id:
            continue
        init_from = img.find(q(ns, "init_from"))
        if init_from is not None and init_from.text:
            image_path_for_id[img_id] = init_from.text.strip()

    # 2. effect_id -> {channel: file_path}
    channels_for_effect = {}
    for eff in root.findall(f".//{q(ns,'effect')}"):
        eff_id = eff.attrib.get("id")
        if not eff_id:
            continue

        # Build sampler sid -> image path lookup for THIS effect
        # (must be built as a local dict, not a closure over a loop variable)
        sid_to_image   = {}   # surface sid  -> image_id
        sid_to_surface = {}   # sampler sid  -> surface sid
        for newparam in eff.findall(f".//{q(ns,'newparam')}"):
            sid     = newparam.attrib.get("sid", "")
            surface = newparam.find(q(ns, "surface"))
            if surface is not None:
                inf = surface.find(q(ns, "init_from"))
                if inf is not None and inf.text:
                    sid_to_image[sid] = inf.text.strip()
            sampler = newparam.find(q(ns, "sampler2D"))
            if sampler is not None:
                src = sampler.find(q(ns, "source"))
                if src is not None and src.text:
                    sid_to_surface[sid] = src.text.strip()

        def resolve(tex_ref, s2surf=sid_to_surface, s2img=sid_to_image):
            """Resolve texture/@texture ref -> file path, using captured dicts."""
            if tex_ref in s2surf:
                image_id = s2img.get(s2surf[tex_ref], "")
            elif tex_ref in s2img:
                image_id = s2img[tex_ref]
            else:
                image_id = tex_ref
            return image_path_for_id.get(image_id)

        channels = {}
        shininess   = 10.0   # default
        spec_color  = None

        # --- Standard phong/lambert profile_COMMON technique ---
        profile = eff.find(q(ns, "profile_COMMON"))
        if profile is not None:
            technique = profile.find(q(ns, "technique"))
            if technique is not None:
                for shader in technique:
                    shader_tag = shader.tag.replace(ns, "")
                    if shader_tag not in ("phong","lambert","blinn","constant"):
                        continue
                    for chan in shader:
                        chan_name = chan.tag.replace(ns, "")
                        tex = chan.find(q(ns, "texture"))
                        if tex is not None:
                            path = resolve(tex.attrib.get("texture", ""))
                            if path:
                                if chan_name == "diffuse":
                                    channels["diffuse"] = path
                                elif chan_name in ("bump", "normal"):
                                    channels["normal"] = path
                                elif chan_name == "transparent":
                                    channels["alpha"] = path
                                elif chan_name == "specular":
                                    channels["specular"] = path
                        # Read shininess float
                        if chan_name == "shininess":
                            fval = chan.find(q(ns, "float"))
                            if fval is not None and fval.text:
                                try: shininess = float(fval.text.strip())
                                except: pass
                        # Read specular color if no specular texture
                        if chan_name == "specular" and tex is None:
                            cval = chan.find(q(ns, "color"))
                            if cval is not None and cval.text:
                                try:
                                    rgba = [float(x) for x in cval.text.strip().split()]
                                    spec_color = rgba[:3]
                                except: pass

        # Convert phong shininess to PBR roughness.
        # shininess=1 (matte) -> roughness=0.9, shininess=100 (shiny) -> roughness=0.3
        roughness = max(0.2, min(0.95, 1.0 - (shininess / 128.0) ** 0.5))
        channels["_roughness"]  = roughness
        channels["_spec_color"] = spec_color

        # --- Extra technique blocks: FCOLLADA and OpenCOLLADA3dsMax ---
        # Both store bump/normal maps here with no namespace prefix on tags.
        # We search the whole effect tree for any <technique> with known profiles.
        for tech in eff.findall(f".//{q(ns,'technique')}") + eff.findall(".//technique"):
            profile_name = tech.attrib.get("profile", "")
            if profile_name in ("FCOLLADA", "OpenCOLLADA3dsMax", "MAX3D"):
                # <bump> -> normal map
                bump = tech.find("bump")
                if bump is not None:
                    tex = bump.find("texture")
                    if tex is not None:
                        path = resolve(tex.attrib.get("texture", ""))
                        if path:
                            channels.setdefault("normal", path)
                # <specularLevel> -> specular texture
                spec_lvl = tech.find("specularLevel")
                if spec_lvl is not None:
                    tex = spec_lvl.find("texture")
                    if tex is not None:
                        path = resolve(tex.attrib.get("texture", ""))
                        if path:
                            channels.setdefault("specular", path)

        # --- Filename-hint fallback for any textures not yet categorised ---
        all_tex_refs = [t.attrib.get("texture","") for t in eff.findall(f".//{q(ns,'texture')}")]
        all_paths    = [resolve(ref) for ref in all_tex_refs]
        all_paths    = [p for p in all_paths if p]

        for path in all_paths:
            base = os.path.basename(path).lower()
            if any(h in base for h in ("_nrm","_normal","_norm","normal_map","_nor")):
                channels.setdefault("normal", path)
            elif any(h in base for h in ("_ao","_ambient_occlusion","_occlusion")):
                channels.setdefault("ao", path)
            elif any(h in base for h in ("_alb","_albedo","_diffuse","_color","_col","_base")):
                channels.setdefault("diffuse", path)
            elif any(h in base for h in ("_spm","_spec","_specular","_roughness","_rgh")):
                channels.setdefault("specular", path)

        # Absolute last resort: first resolved texture = diffuse
        if "diffuse" not in channels and all_paths:
            channels["diffuse"] = all_paths[0]

        # Sanity-check: if diffuse is an AO/normal/specular map (bad DAE export),
        # try to substitute the _alb variant from the same directory.
        diff = channels.get("diffuse", "")
        diff_base = os.path.basename(diff).lower()
        non_albedo_hints = ("_ao", "_nrm", "_normal", "_spm", "_spec", "_bump")
        if any(h in diff_base for h in non_albedo_hints):
            for suffix in non_albedo_hints:
                if suffix in diff_base:
                    alb_name = diff_base.replace(suffix, "_alb")
                    alb_path = os.path.join(os.path.dirname(diff), alb_name)
                    if os.path.isfile(alb_path):
                        channels["diffuse"] = alb_path
                    break

        if channels:
            channels_for_effect[eff_id] = channels

    # 3. material_id -> effect_id
    material_to_effect = {}
    for mat in root.findall(f".//{q(ns,'material')}"):
        mat_id = mat.attrib.get("id")
        if not mat_id:
            continue
        inst = mat.find(f"./{q(ns,'instance_effect')}")
        if inst is not None:
            eff_url = inst.attrib.get("url", "")[1:]
            material_to_effect[mat_id] = eff_url

    # 4. final map: mat_id -> channel dict
    mat_to_textures = {}
    for mat_id, eff_id in material_to_effect.items():
        if eff_id in channels_for_effect:
            mat_to_textures[mat_id] = channels_for_effect[eff_id]

    return mat_to_textures


# ---------------------- ARMATURE BUILDER ----------------------

def build_armature(root, ns, collection, model_name="Armature", correction_mat=None):
    """
    Parse joint hierarchy from <library_visual_scenes> and create a Blender Armature.

    Bone world positions are derived from the INV_BIND matrices in library_controllers.
    This is the authoritative approach: inv_bind[i] = inverse of the bone's world
    transform in the skeleton's bind pose. Inverting it gives exact bone positions,
    completely immune to armature node matrix confusion or exporter quirks.

    The armature object stays at identity. Mesh vertices are transformed by BSM only.
    Returns (armature_object, bsm_per_geom_dict) or (None, {}).
    """
    vs = root.find(f".//{q(ns,'visual_scene')}")
    if vs is None:
        return None, {}

    # --- Collect inv_bind matrices from all skin controllers ---
    # joint_id -> 4x4 Matrix (bind-pose world transform = inv of inv_bind)
    joint_bind_world = {}   # joint_id -> world Matrix in bind pose
    joint_bsm        = {}   # geom_id  -> bind_shape_matrix

    ctrl_lib = root.find(f".//{q(ns,'library_controllers')}")
    if ctrl_lib is not None:
        for ctrl in ctrl_lib.findall(q(ns, "controller")):
            skin = ctrl.find(q(ns, "skin"))
            if skin is None:
                continue
            geom_id = skin.attrib.get("source", "")[1:]

            # bind_shape_matrix for this skin
            bsm_elem = skin.find(q(ns, "bind_shape_matrix"))
            bsm = parse_matrix(bsm_elem.text) if (bsm_elem is not None and bsm_elem.text) else Matrix.Identity(4)
            joint_bsm[geom_id] = bsm

            # Find joint names and inv_bind sources
            joints_elem = skin.find(q(ns, "joints"))
            if joints_elem is None:
                continue
            jnames_src = ibm_src = None
            for inp in joints_elem.findall(q(ns, "input")):
                sem = inp.attrib.get("semantic", "")
                src = inp.attrib.get("source", "")[1:]
                if sem == "JOINT":           jnames_src = src
                elif sem == "INV_BIND_MATRIX": ibm_src  = src

            sources = {}
            for src in skin.findall(q(ns, "source")):
                sid = src.attrib.get("id", "")
                na  = src.find(q(ns, "Name_array"))
                fa  = src.find(q(ns, "float_array"))
                if na is not None and na.text:   sources[sid] = na.text.strip().split()
                elif fa is not None and fa.text: sources[sid] = [float(x) for x in fa.text.strip().split()]

            jnames     = sources.get(jnames_src, [])
            ibm_floats = sources.get(ibm_src, [])
            for i, jname in enumerate(jnames):
                if jname in joint_bind_world:
                    continue  # already have it from another controller
                start = i * 16
                if start + 16 > len(ibm_floats):
                    continue
                inv_bind = Matrix([ ibm_floats[start:start+4],
                                    ibm_floats[start+4:start+8],
                                    ibm_floats[start+8:start+12],
                                    ibm_floats[start+12:start+16] ])
                # bind_world = inverse of inv_bind = bone's world transform at bind pose
                try:
                    joint_bind_world[jname] = inv_bind.inverted()
                except Exception:
                    joint_bind_world[jname] = Matrix.Identity(4)

    if not joint_bind_world:
        return None, {}

    # --- Walk visual scene to get bone hierarchy and names ---
    # bone_info keyed by node_id; we also build a name->id reverse map
    bone_info    = {}   # joint_id -> {name, parent_id}
    name_to_id   = {}   # joint name attribute -> joint_id

    def walk_joints(node, parent_id):
        node_id   = node.attrib.get("id",   "")
        node_name = node.attrib.get("name", node_id)
        node_type = node.attrib.get("type", "")
        if node_type == "JOINT" and node_id:
            # Normalise name: replace spaces with underscores so it matches skin references
            node_name_normalised = node_name.replace(" ", "_")
            bone_info[node_id] = {"name": node_name_normalised, "parent_id": parent_id}
            name_to_id[node_name] = node_id
            name_to_id[node_name_normalised] = node_id
            for child in node.findall(q(ns, "node")):
                walk_joints(child, node_id)
        else:
            for child in node.findall(q(ns, "node")):
                walk_joints(child, parent_id)

    for node in vs.findall(q(ns, "node")):
        walk_joints(node, None)

    # --- Resolve joint_bind_world keys to node IDs ---
    # Different exporters use different conventions for joint references in the skin:
    #   Strategy 1: skin ref == node id exactly              (some exporters)
    #   Strategy 2: skin ref == node name exactly            (Dio-style)
    #   Strategy 3: skin ref == node name with spaces→_      (Jagi-style)
    #   Strategy 4: skin ref == node id suffix after prefix  (Link/Armor-style, e.g.
    #               node id "_0010201_model_JNT_Hips", skin ref "JNT_Hips")
    # We build lookup tables for all strategies and remap everything to node_id keys.

    # Lookup tables
    id_to_id     = {jid: jid for jid in bone_info}
    name_to_id   = {}
    norm_to_id   = {}   # name with spaces→underscores
    suffix_to_id = {}   # last component after splitting on _ or space

    for jid, info in bone_info.items():
        raw_name  = next((n.attrib.get("name","") for n in
                          vs.findall(f".//{q(ns,'node')}[@id='{jid}']")), info["name"])
        norm_name = info["name"]   # already normalised with underscores
        name_to_id[raw_name]   = jid
        name_to_id[norm_name]  = jid
        norm_to_id[norm_name]  = jid
        # Suffix: try progressively shorter suffixes of the node id
        parts = jid.replace("-","_").split("_")
        for i in range(len(parts)):
            suffix = "_".join(parts[i:])
            if suffix and suffix not in suffix_to_id:
                suffix_to_id[suffix] = jid

    def resolve_skin_ref(ref):
        """Map a skin Name_array entry to a node_id using all strategies."""
        ref_norm = ref.replace(" ", "_")
        return (id_to_id.get(ref)
             or name_to_id.get(ref)
             or norm_to_id.get(ref_norm)
             or suffix_to_id.get(ref)
             or suffix_to_id.get(ref_norm)
             or None)

    # Remap joint_bind_world so all keys are node_ids
    remapped = {}
    for skin_ref, world_mat in joint_bind_world.items():
        jid = resolve_skin_ref(skin_ref)
        if jid:
            remapped[jid] = world_mat
        else:
            remapped[skin_ref] = world_mat   # keep as-is, may already be a node_id
    joint_bind_world = remapped

    # Also remap parse_controllers joint_names to bone display names for vertex groups
    # Store the resolve function result in a module-level accessible way via closure
    _resolve_skin_ref = resolve_skin_ref

    # Create Armature object at identity
    arm_data = bpy.data.armatures.new(model_name)
    arm_data.display_type = 'OCTAHEDRAL'
    arm_obj  = bpy.data.objects.new(model_name, arm_data)
    collection.objects.link(arm_obj)

    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode='EDIT')

    edit_bones = arm_data.edit_bones
    created    = {}   # joint_id -> EditBone

    for bid, info in bone_info.items():
        # Use inv_bind world if available, otherwise skip (no position data)
        if bid not in joint_bind_world:
            continue
        world      = joint_bind_world[bid]
        head_world = world.to_translation()

        eb       = edit_bones.new(info["name"])
        eb.head  = head_world

        # Tail: average of child heads, or fallback to world Y axis of this bone
        children_with_pos = [c for c, ci in bone_info.items()
                             if ci["parent_id"] == bid and c in joint_bind_world]
        if children_with_pos:
            child_heads = [joint_bind_world[c].to_translation() for c in children_with_pos]
            avg_child   = sum(child_heads, Vector()) / len(child_heads)
            tail_vec    = avg_child - head_world
            length      = tail_vec.length
            eb.tail     = (head_world + tail_vec.normalized() * max(length, 0.02)
                           if length > 1e-4 else head_world + Vector((0, 0, 0.05)))
        else:
            y_axis  = world.to_3x3() @ Vector((0, 1, 0))
            y_axis  = y_axis.normalized() if y_axis.length > 1e-6 else Vector((0, 0, 1))
            eb.tail = head_world + y_axis * 0.05

        if (eb.tail - eb.head).length < 1e-5:
            eb.tail = eb.head + Vector((0, 0, 0.05))

        created[bid] = eb

    # Parent bones
    for bid, info in bone_info.items():
        if bid not in created:
            continue
        pid = info["parent_id"]
        if pid and pid in created:
            created[bid].parent = created[pid]

    bpy.ops.object.mode_set(mode='OBJECT')
    print(f"Armature '{model_name}' created with {len(created)} bones.")
    return arm_obj, joint_bsm


# ---------------------- SKIN WEIGHT PARSER ----------------------

def parse_controllers(root, ns):
    """
    Parse <library_controllers> and return dict:
      controller_id -> {
        skin_source: str,
        joint_names: [str],
        vertex_weights: {vert_idx: [(joint_idx, weight)]},
      }
    """
    result   = {}
    ctrl_lib = root.find(f".//{q(ns,'library_controllers')}")
    if ctrl_lib is None:
        return result

    for ctrl in ctrl_lib.findall(q(ns, "controller")):
        ctrl_id = ctrl.attrib.get("id", "")
        skin    = ctrl.find(q(ns, "skin"))
        if skin is None:
            continue

        skin_source = skin.attrib.get("source", "")[1:]

        # bind_shape_matrix: transforms mesh vertices into skeleton bind-pose space
        bsm_elem = skin.find(q(ns, "bind_shape_matrix"))
        bind_shape_matrix = parse_matrix(bsm_elem.text) if (bsm_elem is not None and bsm_elem.text) else Matrix.Identity(4)

        # Parse all <source> blocks
        sources = {}
        for src in skin.findall(q(ns, "source")):
            src_id   = src.attrib.get("id", "")
            name_arr = src.find(q(ns, "Name_array"))
            if name_arr is not None and name_arr.text:
                sources[src_id] = name_arr.text.strip().split()
                continue
            float_arr = src.find(q(ns, "float_array"))
            if float_arr is not None and float_arr.text:
                try:
                    sources[src_id] = [float(v) for v in float_arr.text.strip().split()]
                except ValueError:
                    sources[src_id] = []

        # <joints>: find joint-names source and inv-bind-matrix source
        joints_elem     = skin.find(q(ns, "joints"))
        joint_names_src = None
        if joints_elem is not None:
            for inp in joints_elem.findall(q(ns, "input")):
                if inp.attrib.get("semantic") == "JOINT":
                    joint_names_src = inp.attrib.get("source", "")[1:]

        joint_names = sources.get(joint_names_src, []) if joint_names_src else []

        # <vertex_weights>
        # Skip controllers where every joint is a placeholder (no real bone data at all)
        real_bones = [n for n in joint_names if not n.lower().startswith("notabone")]
        if not real_bones:
            print(f"Skipping placeholder-only skin controller '{ctrl_id}'")
            continue

        vw             = skin.find(q(ns, "vertex_weights"))
        vertex_weights = {}
        if vw is not None:
            joint_offset  = 0
            weight_offset = 1
            weight_src_id = None
            for inp in vw.findall(q(ns, "input")):
                sem = inp.attrib.get("semantic", "")
                off = int(inp.attrib.get("offset", "0"))
                src = inp.attrib.get("source", "")[1:]
                if sem == "JOINT":
                    joint_offset  = off
                elif sem == "WEIGHT":
                    weight_offset = off
                    weight_src_id = src

            weight_values = sources.get(weight_src_id, []) if weight_src_id else []
            vcount_elem   = vw.find(q(ns, "vcount"))
            v_elem        = vw.find(q(ns, "v"))

            if vcount_elem is not None and v_elem is not None and vcount_elem.text and v_elem.text:
                vcounts    = [int(x) for x in vcount_elem.text.strip().split()]
                v_data     = [int(x) for x in v_elem.text.strip().split()]
                # Guard: if all vcounts are zero, no weights to parse
                if any(c > 0 for c in vcounts):
                    num_inputs = max(joint_offset, weight_offset) + 1
                    cursor     = 0
                    for vert_idx, count in enumerate(vcounts):
                        pairs = []
                        for _ in range(count):
                            j_idx = v_data[cursor + joint_offset]
                            w_idx = v_data[cursor + weight_offset]
                            w_val = weight_values[w_idx] if 0 <= w_idx < len(weight_values) else 0.0
                            pairs.append((j_idx, w_val))
                            cursor += num_inputs
                        vertex_weights[vert_idx] = pairs

        result[ctrl_id] = {
            "skin_source":        skin_source,
            # Normalise spaces to underscores so vertex group names match bone names
            # regardless of whether the exporter used spaces or underscores
            "joint_names":        [n.replace(" ", "_") for n in joint_names],
            "vertex_weights":     vertex_weights,
            "bind_shape_matrix":  bind_shape_matrix,
        }

    return result


def build_ctrl_mat_map(root, ns, controllers):
    """
    Returns dict: geometry_id -> {material_symbol: material_target_id}
    by matching instance_controller urls to controllers.
    """
    geom_to_mat_override = {}
    for ic in root.findall(f".//{q(ns,'instance_controller')}"):
        ctrl_url = ic.attrib.get("url", "")[1:]
        if ctrl_url not in controllers:
            continue
        geom_id  = controllers[ctrl_url]["skin_source"]
        mat_map  = {}
        for im in ic.findall(f".//{q(ns,'instance_material')}"):
            symbol = im.attrib.get("symbol", "")
            target = im.attrib.get("target", "")[1:]
            mat_map[symbol] = target
        geom_to_mat_override[geom_id] = mat_map
    return geom_to_mat_override


# ---------------------- GEOMETRY IMPORTER ----------------------

def build_mesh_from_geometry(geom_elem, ns, collection, material_texture_map,
                              arm_obj, controllers, ctrl_mat_override, dae_filepath,
                              armature_node_mat=None,
                              import_uvs=True, import_normals=True,
                              import_vertex_colors=True, merge_vertices=False,
                              merge_threshold=0.0001):
    """
    Convert <geometry> -> Blender mesh with positions, normals, colors, UVs,
    materials, textures, and optionally skin weights linked to arm_obj.
    """
    mesh_elem = geom_elem.find(q(ns, "mesh"))
    if mesh_elem is None:
        print("Skipping geometry (no <mesh>):", geom_elem.attrib.get("id"))
        return None

    geom_id   = geom_elem.attrib.get("id", "")
    geom_name = geom_elem.attrib.get("name") or geom_id or "DAE_Mesh"

    # --- Parse <source> blocks ---
    sources = {}
    for src in mesh_elem.findall(q(ns, "source")):
        src_id = src.attrib.get("id")
        if not src_id:
            continue
        sources[src_id] = parse_source_float_array(src, ns)

    # --- Parse <vertices> mapping ---
    # <vertices> can contain POSITION, NORMAL, TEXCOORD, COLOR all sharing the same index.
    # We map the vertices block id to all its semantic sources.
    vertices_map       = {}   # vertices_id -> position source id (for VERTEX lookup)
    vertices_normals   = {}   # vertices_id -> normal source id
    vertices_texcoords = {}   # vertices_id -> texcoord source id
    vertices_colors    = {}   # vertices_id -> color source id
    for verts in mesh_elem.findall(q(ns, "vertices")):
        v_id = verts.attrib.get("id")
        if not v_id:
            continue
        for inp in verts.findall(q(ns, "input")):
            sem = inp.attrib.get("semantic","")
            src = inp.attrib.get("source", "")[1:]
            if sem == "POSITION":
                vertices_map[v_id] = src
            elif sem == "NORMAL":
                vertices_normals[v_id] = src
            elif sem == "TEXCOORD":
                vertices_texcoords[v_id] = src
            elif sem == "COLOR":
                vertices_colors[v_id] = src

    # --- Accumulators ---
    positions    = None
    faces        = []
    face_mat_ids = []
    corner_uvs   = []
    corner_cols  = []
    corner_norms = []

    # --- Process <triangles> and <polylist> blocks ---
    # Both formats use the same index layout; polylist just needs vcount to know
    # how many vertices each polygon has (we triangulate fans on the fly).
    prim_blocks = (
        [(tri, None) for tri in mesh_elem.findall(q(ns, "triangles"))] +
        [(pl,  pl.find(q(ns, "vcount"))) for pl in mesh_elem.findall(q(ns, "polylist"))]
    )

    for prim, vcount_elem in prim_blocks:
        count  = int(prim.attrib.get("count", "0"))
        p_elem = prim.find(q(ns, "p"))
        if p_elem is None or not p_elem.text:
            continue

        # Resolve material symbol -> actual material id
        tri_mat_symbol = prim.attrib.get("material")
        tri_mat_id     = ctrl_mat_override.get(tri_mat_symbol, tri_mat_symbol)

        input_by_offset = {}
        max_offset      = 0
        for inp in prim.findall(q(ns, "input")):
            sem   = inp.attrib.get("semantic")
            src   = inp.attrib.get("source", "")[1:]
            off   = int(inp.attrib.get("offset", "0"))
            set_i = inp.attrib.get("set")
            input_by_offset[off] = (sem, src, set_i)
            max_offset = max(max_offset, off)

        num_inputs = max_offset + 1

        vertex_offset = pos_source_id = None
        vertex_src_id = None  # the vertices block id (for fallback lookups)
        for off, (sem, src, _) in input_by_offset.items():
            if sem == "VERTEX":
                vertex_offset = off
                vertex_src_id = src
                pos_source_id = vertices_map.get(src)
                break

        if vertex_offset is None or pos_source_id is None:
            print("Missing POSITION source in:", geom_name)
            return None

        positions = sources.get(pos_source_id)
        if not positions:
            print("Position source missing:", pos_source_id)
            return None

        normal_offset = uv_offset = color_offset = None
        normal_source = uv_source = color_source = None
        for off, (sem, src, set_idx) in input_by_offset.items():
            if sem == "NORMAL":
                normal_offset = off;  normal_source = sources.get(src)
            elif sem == "COLOR":
                color_offset  = off;  color_source  = sources.get(src)
            elif sem == "TEXCOORD":
                if uv_source is None or set_idx == "0":
                    uv_offset = off;  uv_source = sources.get(src)

        # Fallback: if NORMAL/TEXCOORD/COLOR were declared in <vertices> block
        # they share the VERTEX offset and index, so we use vertex_offset for them
        if normal_source is None and vertex_src_id in vertices_normals:
            normal_offset = vertex_offset
            normal_source = sources.get(vertices_normals[vertex_src_id])
        if uv_source is None and vertex_src_id in vertices_texcoords:
            uv_offset = vertex_offset
            uv_source = sources.get(vertices_texcoords[vertex_src_id])
        if color_source is None and vertex_src_id in vertices_colors:
            color_offset = vertex_offset
            color_source = sources.get(vertices_colors[vertex_src_id])

        raw_idx = [int(x) for x in p_elem.text.strip().split()]

        # Build per-polygon vertex counts
        # <triangles>: every polygon is exactly 3 verts
        # <polylist>:  read from <vcount>
        if vcount_elem is not None and vcount_elem.text:
            vcounts = [int(x) for x in vcount_elem.text.strip().split()]
        else:
            vcounts = [3] * count

        # Walk the flat index stream polygon by polygon
        cursor = 0
        for poly_vcount in vcounts:
            # Collect all corners of this polygon
            poly_vi   = []
            poly_uv   = []
            poly_col  = []
            poly_norm = []

            for v in range(poly_vcount):
                b  = cursor + v * num_inputs
                vi = raw_idx[b + vertex_offset]
                poly_vi.append(vi)

                if normal_offset is not None and normal_source:
                    ni = raw_idx[b + normal_offset]
                    poly_norm.append(Vector(normal_source[ni]) if 0 <= ni < len(normal_source) else Vector((0, 0, 1)))

                if color_offset is not None and color_source:
                    ci = raw_idx[b + color_offset]
                    if 0 <= ci < len(color_source):
                        c = color_source[ci]
                        poly_col.append((c[0], c[1], c[2], c[3] if len(c) == 4 else 1.0))
                    else:
                        poly_col.append((1, 1, 1, 1))

                if uv_offset is not None and uv_source:
                    ti = raw_idx[b + uv_offset]
                    uv = uv_source[ti] if 0 <= ti < len(uv_source) else (0, 0)
                    poly_uv.append((uv[0], uv[1]))

            cursor += poly_vcount * num_inputs

            # Triangulate as a fan from vertex 0: (0,1,2), (0,2,3), (0,3,4) ...
            for i in range(1, poly_vcount - 1):
                tri_vi = [poly_vi[0],   poly_vi[i],   poly_vi[i+1]]
                if len(set(tri_vi)) < 3:
                    continue
                faces.append(tuple(tri_vi))
                face_mat_ids.append(tri_mat_id)

                if poly_norm:
                    corner_norms.extend([poly_norm[0], poly_norm[i], poly_norm[i+1]])
                if poly_col:
                    corner_cols.extend([poly_col[0], poly_col[i], poly_col[i+1]])
                if poly_uv:
                    corner_uvs.extend([poly_uv[0], poly_uv[i], poly_uv[i+1]])

    if not positions or not faces:
        print("No valid geometry in:", geom_name)
        return None

    # ---------------------- CREATE MESH ----------------------
    # Mesh vertices only need the bind_shape_matrix applied.
    # Bones are computed as armature_node_mat @ joint_chain.
    # BSM brings vertices into the same space bones expect — do NOT also apply armature_node_mat.
    skin_ctrl = next((c for c in controllers.values() if c["skin_source"] == geom_id), None)
    if skin_ctrl is not None:
        bsm = skin_ctrl.get("bind_shape_matrix", Matrix.Identity(4))
        if bsm != Matrix.Identity(4):
            bsm3 = bsm.to_3x3()
            bsm_t = bsm.to_translation()
            positions = [tuple(bsm3 @ Vector(p) + bsm_t) for p in positions]

    mesh = bpy.data.meshes.new(geom_name)
    mesh.from_pydata([Vector(p) for p in positions], [], faces)
    mesh.update(calc_edges=True)

    obj = bpy.data.objects.new(geom_name, mesh)
    collection.objects.link(obj)

    # ---------------------- MATERIALS ----------------------
    dae_dir = os.path.dirname(bpy.path.abspath(dae_filepath))

    def _resolve_tex(raw_path):
        if not raw_path:
            return None
        fname = os.path.basename(raw_path)
        # Build candidate list: raw path, next to DAE, then search parent dirs
        candidates = [
            raw_path,
            os.path.join(dae_dir, raw_path),
            os.path.join(dae_dir, fname),
        ]
        # Also search up to 2 parent directories (handles nested folder structures)
        parent = os.path.dirname(dae_dir)
        for _ in range(2):
            candidates.append(os.path.join(parent, fname))
            # And any immediate subdirectory of the parent
            try:
                for sub in os.listdir(parent):
                    subpath = os.path.join(parent, sub, fname)
                    candidates.append(subpath)
            except OSError:
                pass
            parent = os.path.dirname(parent)

        for candidate in candidates:
            candidate = os.path.normpath(candidate)
            if os.path.isfile(candidate):
                return candidate
        return None

    def _load_img(raw_path, colorspace="sRGB"):
        resolved = _resolve_tex(raw_path)
        if not resolved:
            return None
        try:
            img = bpy.data.images.load(resolved, check_existing=True)
            img.colorspace_settings.name = colorspace
            return img
        except Exception as e:
            print(f"Failed to load texture '{resolved}': {e}")
            return None

    def _mat_diffuse_path(m):
        """Return the filepath of the diffuse TexImage node, or None."""
        if not m.use_nodes:
            return None
        for n in m.node_tree.nodes:
            if n.type == 'TEX_IMAGE' and n.image and n.label == "diffuse":
                return os.path.normpath(bpy.path.abspath(n.image.filepath))
        return None

    def _build_mat_nodes(m, channels, has_second_uv=False):
        """
        Build PBR node setup from channels dict.
        Normal map only wired when has_second_uv=True.
        Roughness derived from phong shininess so model isn't oily.
        """
        m.use_nodes = True
        nodes = m.node_tree.nodes
        links = m.node_tree.links
        nodes.clear()

        out_n  = nodes.new("ShaderNodeOutputMaterial"); out_n.location  = (700, 0)
        bsdf_n = nodes.new("ShaderNodeBsdfPrincipled"); bsdf_n.location = (300, 0)
        links.new(bsdf_n.outputs["BSDF"], out_n.inputs["Surface"])

        # Roughness from phong shininess (shininess=50 → roughness~0.65)
        roughness = channels.get("_roughness", 0.8)
        bsdf_n.inputs["Roughness"].default_value = roughness

        # Specular: read from DAE specular color; default very low for skin/cloth
        spec_color = channels.get("_spec_color")
        if spec_color is not None:
            spec_intensity = (spec_color[0] + spec_color[1] + spec_color[2]) / 3.0
        else:
            spec_intensity = 0.05   # non-metallic default — not oily
        for inp_name in ("Specular IOR Level", "Specular"):
            if inp_name in bsdf_n.inputs:
                bsdf_n.inputs[inp_name].default_value = min(1.0, spec_intensity)
                break

        x = -400

        # Diffuse / albedo
        diff_path = channels.get("diffuse")
        if diff_path:
            img = _load_img(diff_path, "sRGB")
            if img:
                n = nodes.new("ShaderNodeTexImage")
                n.image = img; n.label = "diffuse"; n.location = (x, 200)
                links.new(n.outputs["Color"], bsdf_n.inputs["Base Color"])
                links.new(n.outputs["Alpha"], bsdf_n.inputs["Alpha"])
                m.blend_method = 'CLIP'

        # Normal map — load it always; wire it if the mesh has UVs to drive it.
        # Without the correct UV channel it should NOT be connected or it turns the face pink.
        nrm_path = channels.get("normal")
        if nrm_path:
            img = _load_img(nrm_path, "Non-Color")
            if img:
                img_n = nodes.new("ShaderNodeTexImage"); img_n.location = (x - 300, -200)
                img_n.image = img; img_n.label = "normal"
                if has_second_uv:
                    nrm_n = nodes.new("ShaderNodeNormalMap"); nrm_n.location = (x, -200)
                    links.new(img_n.outputs["Color"], nrm_n.inputs["Color"])
                    links.new(nrm_n.outputs["Normal"], bsdf_n.inputs["Normal"])

        # AO — multiply into base color
        ao_path = channels.get("ao")
        if ao_path and diff_path:
            img = _load_img(ao_path, "Non-Color")
            if img:
                ao_n  = nodes.new("ShaderNodeTexImage"); ao_n.location  = (x - 300, 450)
                mix_n = nodes.new("ShaderNodeMixRGB");   mix_n.location = (x, 450)
                ao_n.image = img; ao_n.label = "ao"
                mix_n.blend_type = 'MULTIPLY'
                mix_n.inputs[0].default_value = 1.0
                diff_node = next((n for n in nodes if n.type == 'TEX_IMAGE' and n.label == "diffuse"), None)
                if diff_node:
                    links.new(diff_node.outputs["Color"], mix_n.inputs[1])
                    links.new(ao_n.outputs["Color"],      mix_n.inputs[2])
                    for lnk in list(links):
                        if lnk.to_socket == bsdf_n.inputs["Base Color"]:
                            links.remove(lnk)
                    links.new(mix_n.outputs["Color"], bsdf_n.inputs["Base Color"])

        # Specular texture (_spm) — load it into the node tree for manual use
        # Do NOT wire it automatically; incorrect wiring causes the shiny/black look
        spec_path = channels.get("specular")
        if spec_path:
            img = _load_img(spec_path, "Non-Color")
            if img:
                n = nodes.new("ShaderNodeTexImage"); n.location = (x, -450)
                n.image = img; n.label = "specular"
                # Left unconnected — user can wire to Specular IOR Level or Roughness

    # Detect whether this mesh has a second UV channel (set="1")
    has_second_uv = any(
        inp.attrib.get("semantic") == "TEXCOORD" and inp.attrib.get("set","0") == "1"
        for prim in mesh_elem
        for inp in prim.findall(q(ns, "input"))
    )

    unique_mat_ids = sorted({m for m in face_mat_ids if m is not None})
    mat_index_map  = {}
    obj.data.materials.clear()

    for idx, mat_id in enumerate(unique_mat_ids):
        channels     = material_texture_map.get(mat_id, {})
        diff_path    = _resolve_tex(channels.get("diffuse"))

        # Use diffuse filename as material name, fall back to mat_id
        tex_base = os.path.splitext(os.path.basename(diff_path))[0] if diff_path else mat_id

        # Reuse existing material only if it already has a diffuse texture
        # that matches what we want. Otherwise always rebuild it.
        existing   = bpy.data.materials.get(tex_base)
        want_path  = os.path.normpath(diff_path) if diff_path else None
        exist_path = _mat_diffuse_path(existing) if existing is not None else None
        if existing is not None and exist_path is not None and exist_path == want_path:
            mat = existing
        else:
            mat = bpy.data.materials.new(tex_base) if existing is None else existing
            _build_mat_nodes(mat, {k: v for k, v in channels.items()}, has_second_uv)
            if diff_path:
                print(f"Material built: '{mat.name}' diffuse='{os.path.basename(diff_path)}'")
            else:
                print(f"Material built: '{mat.name}' (no diffuse)")

        obj.data.materials.append(mat)
        mat_index_map[mat_id] = idx

    for poly, mat_id in zip(mesh.polygons, face_mat_ids):
        if mat_id and mat_id in mat_index_map:
            poly.material_index = mat_index_map[mat_id]

    # ---------------------- UVs ----------------------
    if import_uvs and corner_uvs and len(corner_uvs) == len(mesh.loops):
        uv_layer = mesh.uv_layers.new(name="UVMap")
        for li, uv in enumerate(corner_uvs):
            uv_layer.data[li].uv = uv

    # ---------------------- COLORS ----------------------
    if import_vertex_colors and corner_cols and len(corner_cols) == len(mesh.loops):
        col_attr = mesh.color_attributes.new(name="Col", type="FLOAT_COLOR", domain="CORNER")
        for li, col in enumerate(corner_cols):
            col_attr.data[li].color = col

    # ---------------------- NORMALS ----------------------
    if import_normals and corner_norms and len(corner_norms) == len(mesh.loops):
        mesh.normals_split_custom_set(corner_norms)

    # ---------------------- MERGE VERTICES ----------------------
    if merge_vertices:
        import bmesh
        bm = bmesh.new()
        bm.from_mesh(mesh)
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=merge_threshold)
        bm.to_mesh(mesh)
        bm.free()
        mesh.update()

    # ---------------------- SKIN WEIGHTS ----------------------
    if arm_obj is not None and skin_ctrl is not None and skin_ctrl["vertex_weights"]:
        joint_names    = skin_ctrl["joint_names"]
        vertex_weights = skin_ctrl["vertex_weights"]

        # Create vertex groups only for real bones (skip NotABone placeholders)
        vgroups = {}
        for jname in joint_names:
            if jname.lower().startswith("notabone"):
                vgroups[jname] = None  # placeholder — no group created
            else:
                vgroups[jname] = obj.vertex_groups.new(name=jname)

        # Assign weights vertex by vertex, skipping NotABone slots
        for vert_idx, pairs in vertex_weights.items():
            for j_idx, weight in pairs:
                if j_idx < 0 or j_idx >= len(joint_names) or weight <= 0.0:
                    continue
                vg = vgroups.get(joint_names[j_idx])
                if vg is not None:
                    vg.add([vert_idx], weight, 'ADD')

        # Parent mesh to armature with Armature modifier
        obj.parent = arm_obj
        mod = obj.modifiers.new(name="Armature", type='ARMATURE')
        mod.object            = arm_obj
        mod.use_vertex_groups = True
        print(f"Skin weights applied to '{geom_name}' ({len(vgroups)} bone groups).")

    elif arm_obj is not None:
        # Mesh has no skin weights (e.g. rigid accessory exported with empty skin stub).
        # Parent it to the armature object so it moves with it, without any bone offset.
        obj.parent      = arm_obj
        obj.parent_type = 'OBJECT'
        print(f"'{geom_name}' has no skin weights — parented to armature object.")

    return obj


# ---------------------- IMPORT OPERATOR ----------------------

class IMPORT_OT_simple_collada_full(Operator, ImportHelper):
    """Import a COLLADA (.dae) mesh with full features"""
    bl_idname    = "import_scene.simple_collada_full"
    bl_label     = "Import Simple COLLADA (.dae)"
    filename_ext = ".dae"
    filter_glob: StringProperty(default="*.dae", options={'HIDDEN'})

    import_rig: BoolProperty(
        name="Import Rig",
        description="Import armature and skin weights if present",
        default=True,
    )
    import_materials: BoolProperty(
        name="Import Materials",
        description="Load textures and build material node graphs",
        default=True,
    )
    import_normals: BoolProperty(
        name="Import Normals",
        description="Use custom split normals from the DAE file",
        default=True,
    )
    import_uvs: BoolProperty(
        name="Import UVs",
        description="Import texture coordinate data",
        default=True,
    )
    import_vertex_colors: BoolProperty(
        name="Import Vertex Colors",
        description="Import vertex color data if present",
        default=True,
    )
    merge_vertices: BoolProperty(
        name="Merge Vertices",
        description="Remove duplicate vertices by distance after import",
        default=False,
    )
    merge_threshold: FloatProperty(
        name="Merge Distance",
        description="Maximum distance between vertices to merge",
        default=0.0001,
        min=0.0,
        max=0.1,
        precision=5,
    )

    def execute(self, context):
        if not os.path.isfile(self.filepath):
            self.report({'ERROR'}, f"File not found: {self.filepath}")
            return {'CANCELLED'}

        try:
            tree = ET.parse(self.filepath)
            root = tree.getroot()
        except ET.ParseError as e:
            # Some exporters (e.g. Bricklink Studio) write namespace prefixes like
            # xsi:type without declaring the xmlns:xsi binding, which causes
            # "unbound prefix" errors. Strip all namespace declarations and retry.
            try:
                import re
                with open(self.filepath, 'r', encoding='utf-8', errors='replace') as f:
                    raw = f.read()
                # Remove undeclared namespace prefixes from tags and attributes
                raw = re.sub(r'<(\w+):(\w)', r'<\2', raw)
                raw = re.sub(r'</(\w+):(\w)', r'</\2', raw)
                raw = re.sub(r'\s+\w+:\w+\s*=\s*"[^"]*"', '', raw)
                raw = re.sub(r"\s+\w+:\w+\s*=\s*'[^']*'", '', raw)
                root = ET.fromstring(raw)
            except Exception as e2:
                self.report({'ERROR'}, f"Failed to parse DAE: {e} / {e2}")
                return {'CANCELLED'}
        except Exception as e:
            self.report({'ERROR'}, f"Failed to parse DAE: {e}")
            return {'CANCELLED'}

        ns  = get_collada_ns(root)
        dae = self.filepath

        if context.view_layer.active_layer_collection:
            collection = context.view_layer.active_layer_collection.collection
        else:
            collection = context.scene.collection

        material_texture_map = extract_material_texture_map(root, ns) if self.import_materials else {}

        model_name    = os.path.splitext(os.path.basename(dae))[0]
        correction_mat = get_up_axis_matrix(root, ns)

        arm_obj           = None
        armature_node_mat = Matrix.Identity(4)
        controllers       = {}
        if self.import_rig:
            arm_obj, armature_node_mat = build_armature(root, ns, collection, model_name, correction_mat)
            controllers = parse_controllers(root, ns)

        geom_mat_override = build_ctrl_mat_map(root, ns, controllers)

        geometries = root.findall(f".//{q(ns,'geometry')}")
        if not geometries:
            self.report({'ERROR'}, "No <geometry> found in DAE")
            return {'CANCELLED'}

        imported = 0
        for geom in geometries:
            geom_id      = geom.attrib.get("id", "")
            mat_override = geom_mat_override.get(geom_id, {})
            obj = build_mesh_from_geometry(
                geom, ns, collection, material_texture_map,
                arm_obj, controllers, mat_override, dae, armature_node_mat,
                import_uvs=self.import_uvs,
                import_normals=self.import_normals,
                import_vertex_colors=self.import_vertex_colors,
                merge_vertices=self.merge_vertices,
                merge_threshold=self.merge_threshold,
            )
            if obj:
                imported += 1

        if imported == 0:
            self.report({'ERROR'}, "No objects created. Check console.")
            return {'CANCELLED'}

        rig_msg = f" + armature ({arm_obj.name})" if arm_obj else ""
        self.report({'INFO'}, f"Imported {imported} object(s){rig_msg}.")
        return {'FINISHED'}

    def draw(self, context):
        layout = self.layout
        layout.label(text="Mesh")
        layout.prop(self, "import_uvs")
        layout.prop(self, "import_normals")
        layout.prop(self, "import_vertex_colors")
        layout.prop(self, "merge_vertices")
        if self.merge_vertices:
            layout.prop(self, "merge_threshold")
        layout.separator()
        layout.label(text="Materials")
        layout.prop(self, "import_materials")
        layout.separator()
        layout.label(text="Rig")
        layout.prop(self, "import_rig")


# ---------------------- TEXTURE ASSIGN OPERATOR ----------------------

class OBJECT_OT_assign_textures_by_name(Operator):
    """Assign textures based on material names matching image file names"""
    bl_idname  = "object.assign_textures_by_name"
    bl_label   = "Assign Textures by Name"
    bl_options = {'REGISTER', 'UNDO'}

    directory: StringProperty(
        name="Texture Folder", description="Folder containing texture images", subtype='DIR_PATH'
    )

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        folder = bpy.path.abspath(self.directory)
        if not os.path.isdir(folder):
            self.report({'ERROR'}, f"Not a directory: {folder}")
            return {'CANCELLED'}

        exts   = {".png", ".jpg", ".jpeg", ".tga", ".bmp", ".tif", ".tiff", ".dds"}
        images = {}
        for f in os.listdir(folder):
            name, ext = os.path.splitext(f)
            if ext.lower() in exts:
                full = os.path.join(folder, f)
                try:
                    img = bpy.data.images.load(full, check_existing=True)
                    images[name] = img
                except:
                    pass

        assigned = 0
        for obj in context.selected_objects:
            if not hasattr(obj.data, "materials"):
                continue
            for mat in obj.data.materials:
                if not mat or str(mat.name).strip() not in images:
                    continue
                img = images[str(mat.name).strip()]
                mat.use_nodes = True
                nodes = mat.node_tree.nodes
                links = mat.node_tree.links
                while nodes:
                    nodes.remove(nodes[0])
                out_n  = nodes.new("ShaderNodeOutputMaterial"); out_n.location  = ( 300, 0)
                bsdf_n = nodes.new("ShaderNodeBsdfPrincipled"); bsdf_n.location = (   0, 0)
                img_n  = nodes.new("ShaderNodeTexImage");       img_n.location  = (-300, 0)
                img_n.image = img
                links.new(img_n.outputs["Color"], bsdf_n.inputs["Base Color"])
                links.new(bsdf_n.outputs["BSDF"], out_n.inputs["Surface"])
                assigned += 1

        self.report({'INFO'}, f"Assigned textures to {assigned} materials.")
        return {'FINISHED'}


# ---------------------- MENUS & REGISTER ----------------------

def menu_func_import(self, context):
    self.layout.operator(IMPORT_OT_simple_collada_full.bl_idname, text="Simple COLLADA (.dae)")


def menu_func_assign_textures(self, context):
    self.layout.operator(OBJECT_OT_assign_textures_by_name.bl_idname, text="Assign Textures by Name")


def register():
    bpy.utils.register_class(IMPORT_OT_simple_collada_full)
    bpy.utils.register_class(OBJECT_OT_assign_textures_by_name)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)
    bpy.types.VIEW3D_MT_object.append(menu_func_assign_textures)


def unregister():
    bpy.types.VIEW3D_MT_object.remove(menu_func_assign_textures)
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)
    bpy.utils.unregister_class(OBJECT_OT_assign_textures_by_name)
    bpy.utils.unregister_class(IMPORT_OT_simple_collada_full)


if __name__ == "__main__":
    register()
