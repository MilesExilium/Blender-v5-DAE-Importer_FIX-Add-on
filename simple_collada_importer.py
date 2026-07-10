bl_info = {
    "name": "COLLADA (.dae) Importer & Exporter",
    "author": "ekztal / MilesExilium / RebeccaNod1 / XDM-Inc / ZackWilde27 ",
    "version": (4, 2, 2),
    "blender": (5, 0, 0),
    "location": "File > Import/Export > COLLADA (.dae)",
    "description": "Fast batch COLLADA import/export for full Blender 5 scenes.",
    "category": "Import-Export",
    "support": "COMMUNITY",
    "doc_url": "https://github.com/ekztal/Blender-v5-DAE-Importer-Add-on",
    "tracker_url": "https://github.com/ekztal/Blender-v5-DAE-Importer-Add-on/issues",
}

import os
import math
import datetime
import time
import traceback
import urllib.parse
import re
import copy
import bpy
from bpy_extras.io_utils import ImportHelper, ExportHelper
from bpy.types import Operator, OperatorFileListElement
from bpy.props import (
    StringProperty, BoolProperty, FloatProperty, EnumProperty,
    CollectionProperty,
)
from mathutils import Vector, Matrix
import xml.etree.ElementTree as ET


# ── XML / NAMESPACE HELPERS ─────────────────────────────────────────────────

def get_collada_ns(root):
    if root.tag.startswith("{"):
        return root.tag.split("}")[0] + "}"
    return ""

def q(ns, tag):
    return f"{ns}{tag}"

def strip_url(value):
    value = (value or "").strip()
    return value[1:] if value.startswith("#") else value

def xml_local_name(tag):
    return tag.rsplit("}", 1)[-1].split(":", 1)[-1]

def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

def parse_source_float_array(source_elem, ns):
    fa = source_elem.find(q(ns, "float_array"))
    if fa is None or fa.text is None:
        return []
    try:
        floats = [float(v) for v in fa.text.strip().split()]
    except ValueError:
        return []
    acc = source_elem.find(f"{q(ns,'technique_common')}/{q(ns,'accessor')}")
    stride = int(acc.attrib.get("stride", "1")) if acc is not None else 1
    offset = int(acc.attrib.get("offset", "0")) if acc is not None else 0
    count = int(acc.attrib.get("count", "0")) if acc is not None else 0
    if stride <= 0:
        stride = 1
    if count <= 0:
        count = max(0, (len(floats) - offset) // stride)
    out = []
    for item_index in range(count):
        i = offset + item_index * stride
        chunk = floats[i:i+stride]
        if len(chunk) < stride:
            break
        out.append(tuple(chunk))
    return out

def parse_source_name_array(source_elem, ns):
    arr = source_elem.find(q(ns, "Name_array"))
    if arr is None:
        arr = source_elem.find(q(ns, "IDREF_array"))
    if arr is None:
        arr = source_elem.find(q(ns, "bool_array"))
    if arr is None or not arr.text:
        return []
    values = arr.text.strip().split()
    acc = source_elem.find(f"{q(ns,'technique_common')}/{q(ns,'accessor')}")
    if acc is None:
        return values
    offset = int(acc.attrib.get("offset", "0"))
    stride = max(1, int(acc.attrib.get("stride", "1")))
    count = int(acc.attrib.get("count", "0")) or max(0, (len(values) - offset) // stride)
    return [values[offset + i * stride] for i in range(count)
            if offset + i * stride < len(values)]

def matrix_from_collada_values(vals):
    """Read a COLLADA matrix, tolerating row/column-major vendor files.

    Blender/OpenCOLLADA-style files store the translation in the last value
    of each of the first three rows.  Some vendor files instead store it in
    the final row.  When a matrix has no translation at all, both layouts are
    ambiguous; prefer the row-major interpretation because Blender's own
    COLLADA exporter writes bind-shape and joint matrices this way.  This is
    important for BOTW-era Blender 2.7x exports where an upper-body
    bind_shape_matrix has zero translation but still must not be transposed.
    """
    if len(vals) != 16:
        return Matrix.Identity(4)
    raw = Matrix([vals[0:4], vals[4:8], vals[8:12], vals[12:16]])

    # COLLADA specifies column-major storage. A few older exporters write
    # row-major matrices; detect those when only the row-major translation
    # slots are populated.
    row_translation = abs(vals[3]) + abs(vals[7]) + abs(vals[11])
    col_translation = abs(vals[12]) + abs(vals[13]) + abs(vals[14])
    if row_translation <= 1.0e-9 and col_translation <= 1.0e-9:
        return raw
    if row_translation > 1.0e-9 and col_translation <= 1.0e-9:
        return raw
    return raw.transposed()

def parse_matrix(text):
    if not text:
        return Matrix.Identity(4)
    try:
        vals = [float(v) for v in text.strip().split()]
    except ValueError:
        return Matrix.Identity(4)
    return matrix_from_collada_values(vals)

def parse_node_transform(node, ns):
    """
    Parse all COLLADA transform tags on a node in order and return
    their combined 4x4 Matrix. Handles <matrix>, <translate>,
    <rotate>, and <scale> so that DAEs using individual transforms
    (instead of a single <matrix>) are placed correctly.
    """
    combined = Matrix.Identity(4)
    for child in node:
        tag = xml_local_name(child.tag)
        if not child.text:
            continue
        if tag == "matrix":
            combined @= parse_matrix(child.text)
        elif tag == "translate":
            v = [float(x) for x in child.text.split()]
            if len(v) == 3:
                combined @= Matrix.Translation(Vector(v))
        elif tag == "rotate":
            vals = [float(x) for x in child.text.split()]
            if len(vals) == 4:
                axis  = Vector(vals[:3])
                angle = math.radians(vals[3])
                if axis.length > 1e-6:
                    combined @= Matrix.Rotation(angle, 4, axis)
        elif tag == "scale":
            s = [float(x) for x in child.text.split()]
            if len(s) == 3:
                combined @= Matrix.Diagonal(Vector((s[0], s[1], s[2], 1.0)))
    return combined


def get_up_axis_matrix(root, ns):
    asset = root.find(q(ns, "asset"))
    up    = asset.find(q(ns, "up_axis")) if asset is not None else None
    axis  = up.text.strip().upper() if (up is not None and up.text) else "Z_UP"
    if axis == "Z_UP":
        return Matrix.Identity(4)
    elif axis == "X_UP":
        return Matrix.Rotation(-math.pi / 2.0, 4, 'Y')
    else:
        return Matrix.Rotation(math.pi / 2.0, 4, 'X')

def get_unit_scale(root, ns):
    asset = root.find(q(ns, "asset"))
    unit = asset.find(q(ns, "unit")) if asset is not None else None
    scale = safe_float(unit.attrib.get("meter"), 1.0) if unit is not None else 1.0
    return scale if scale > 0.0 else 1.0

def get_scene_correction_matrix(root, ns, use_units=True):
    axis = get_up_axis_matrix(root, ns)
    if not use_units:
        return axis
    return Matrix.Scale(get_unit_scale(root, ns), 4) @ axis

def find_active_visual_scene(root, ns):
    scene = root.find(q(ns, "scene"))
    instance = scene.find(q(ns, "instance_visual_scene")) if scene is not None else None
    visual_scene_id = strip_url(instance.attrib.get("url")) if instance is not None else ""
    library = root.find(q(ns, "library_visual_scenes"))
    if library is None:
        return root.find(f".//{q(ns,'visual_scene')}")
    if visual_scene_id:
        for visual_scene in library.findall(q(ns, "visual_scene")):
            if visual_scene.attrib.get("id") == visual_scene_id:
                return visual_scene
    return library.find(q(ns, "visual_scene"))

def analyse_dae(root, ns):
    asset      = root.find(q(ns, "asset"))
    up_elem    = asset.find(q(ns, "up_axis")) if asset is not None else None
    up_axis    = up_elem.text.strip().upper() if (up_elem is not None and up_elem.text) else "Z_UP"
    unit_elem  = asset.find(q(ns, "unit")) if asset is not None else None
    unit_meter = float(unit_elem.attrib.get("meter", 1.0)) if unit_elem is not None else 1.0
    joint_nodes  = root.findall(f".//{q(ns,'node')}[@type='JOINT']")
    ctrl_lib     = root.find(q(ns, "library_controllers"))
    ctrl_list    = list(ctrl_lib) if ctrl_lib is not None else []
    skin_ctrls   = [c for c in ctrl_list if c.find(q(ns,"skin")) is not None]
    is_rigged    = len(joint_nodes) > 0 and len(skin_ctrls) > 0
    has_lib_nodes  = root.find(q(ns, "library_nodes")) is not None
    has_inst_nodes = bool(root.findall(f".//{q(ns,'instance_node')}"))
    is_assembly    = has_inst_nodes or (has_lib_nodes and not is_rigged)
    anim_lib  = root.find(q(ns, "library_animations"))
    has_anims = anim_lib is not None and len(list(anim_lib)) > 0
    profile = {
        "is_rigged": is_rigged, "is_assembly": is_assembly,
        "up_axis": up_axis, "unit_meter": unit_meter,
        "joint_count": len(joint_nodes), "controller_count": len(skin_ctrls),
        "has_lib_nodes": has_lib_nodes, "has_inst_nodes": has_inst_nodes,
        "has_anims": has_anims,
    }
    print(f"[DAE Profile] rigged={is_rigged} assembly={is_assembly} "
          f"up={up_axis} unit={unit_meter} "
          f"joints={len(joint_nodes)} ctrls={len(skin_ctrls)} anims={has_anims}")
    return profile


# ── MATERIAL / TEXTURE HELPERS ───────────────────────────────────────────────

def extract_material_texture_map(root, ns):
    image_path_for_id = {}
    for img in root.findall(f".//{q(ns,'image')}"):
        img_id = img.attrib.get("id")
        if not img_id:
            continue
        init_from = img.find(q(ns, "init_from"))
        if init_from is not None and init_from.text:
            image_path_for_id[img_id] = init_from.text.strip()

    channels_for_effect = {}
    for eff in root.findall(f".//{q(ns,'effect')}"):
        eff_id = eff.attrib.get("id")
        if not eff_id:
            continue
        sid_to_image   = {}
        sid_to_surface = {}
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
            if tex_ref in s2surf:
                image_id = s2img.get(s2surf[tex_ref], "")
            elif tex_ref in s2img:
                image_id = s2img[tex_ref]
            else:
                image_id = tex_ref
            return image_path_for_id.get(image_id)

        channels = {}
        prof = eff.find(q(ns, "profile_COMMON"))
        if prof is not None:
            technique = prof.find(q(ns, "technique"))
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
                                elif chan_name == "specular":
                                    channels["specular"] = path
                                elif chan_name in ("transparent", "transparency"):
                                    channels["alpha"] = path
                        color = chan.find(q(ns, "color"))
                        if color is not None and color.text:
                            values = [safe_float(v) for v in color.text.split()]
                            while len(values) < 4:
                                values.append(1.0)
                            if chan_name == "diffuse":
                                channels["_diffuse_color"] = tuple(values[:4])
                            elif chan_name in ("emission", "ambient"):
                                channels["_emission_color"] = tuple(values[:4])
                            elif chan_name == "specular":
                                channels["_specular_color"] = tuple(values[:4])
                            elif chan_name == "transparent":
                                channels["_transparent_color"] = tuple(values[:4])
                        scalar = chan.find(q(ns, "float"))
                        if scalar is not None and scalar.text:
                            value = safe_float(scalar.text, 1.0)
                            if chan_name == "transparency":
                                channels["_transparency"] = value
                            elif chan_name == "shininess":
                                channels["_roughness"] = max(
                                    0.0, min(1.0, 1.0 - math.sqrt(max(value, 0.0) / 128.0))
                                )

        for tech in eff.findall(f".//{q(ns,'technique')}") + eff.findall(".//technique"):
            profile_name = tech.attrib.get("profile", "")
            if profile_name in ("FCOLLADA", "OpenCOLLADA3dsMax", "MAX3D"):
                bump = tech.find("bump")
                if bump is not None:
                    tex = bump.find("texture")
                    if tex is not None:
                        path = resolve(tex.attrib.get("texture", ""))
                        if path:
                            channels.setdefault("normal", path)

        all_tex_refs = [t.attrib.get("texture","") for t in eff.findall(f".//{q(ns,'texture')}")]
        all_paths    = [resolve(ref) for ref in all_tex_refs if resolve(ref)]
        if "diffuse" not in channels and all_paths:
            channels["diffuse"] = all_paths[0]

        if channels:
            channels_for_effect[eff_id] = channels

    material_to_effect = {}
    for mat in root.findall(f".//{q(ns,'material')}"):
        mat_id = mat.attrib.get("id")
        if not mat_id:
            continue
        inst = mat.find(f"./{q(ns,'instance_effect')}")
        if inst is not None:
            eff_url = inst.attrib.get("url", "")[1:]
            material_to_effect[mat_id] = eff_url

    mat_to_textures = {}
    for mat_id, eff_id in material_to_effect.items():
        if eff_id in channels_for_effect:
            mat_to_textures[mat_id] = channels_for_effect[eff_id]
    return mat_to_textures


# ── ARMATURE BUILDER ─────────────────────────────────────────────────────────

def build_armature(root, ns, collection, model_name="Armature", correction_mat=None):
    vs = find_active_visual_scene(root, ns)
    if vs is None:
        return None

    joint_bind_world = {}
    joint_bsm        = {}
    scene_joint_world = {}
    scene_joint_names = {}

    def collect_scene_joints(node, parent_matrix):
        local_matrix = parse_node_transform(node, ns)
        world_matrix = parent_matrix @ local_matrix
        if node.attrib.get("type", "").upper() == "JOINT":
            identifiers = {
                node.attrib.get("id"),
                node.attrib.get("sid"),
                node.attrib.get("name"),
            }
            for identifier in identifiers:
                if identifier:
                    scene_joint_world[identifier] = world_matrix.copy()
            node_id = node.attrib.get("id") or node.attrib.get("sid")
            if node_id:
                scene_joint_names[node_id] = (
                    node.attrib.get("name")
                    or node.attrib.get("sid")
                    or node_id
                )
        for child in node.findall(q(ns, "node")):
            collect_scene_joints(child, world_matrix)

    for scene_node in vs.findall(q(ns, "node")):
        collect_scene_joints(scene_node, Matrix.Identity(4))

    ctrl_lib = root.find(f".//{q(ns,'library_controllers')}")
    if ctrl_lib is not None:
        for ctrl in ctrl_lib.findall(q(ns, "controller")):
            skin = ctrl.find(q(ns, "skin"))
            if skin is None:
                continue
            geom_id  = skin.attrib.get("source", "")[1:]
            bsm_elem = skin.find(q(ns, "bind_shape_matrix"))
            bsm = parse_matrix(bsm_elem.text) if (bsm_elem is not None and bsm_elem.text) else Matrix.Identity(4)
            joint_bsm[geom_id] = bsm

            joints_elem = skin.find(q(ns, "joints"))
            if joints_elem is None:
                continue
            jnames_src = ibm_src = None
            for inp in joints_elem.findall(q(ns, "input")):
                sem = inp.attrib.get("semantic", "")
                src = inp.attrib.get("source", "")[1:]
                if sem == "JOINT":             jnames_src = src
                elif sem == "INV_BIND_MATRIX": ibm_src    = src

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
                    continue
                start = i * 16
                if start + 16 > len(ibm_floats):
                    continue
                m = ibm_floats[start:start+16]
                inv_bind = matrix_from_collada_values(m)
                try:
                    joint_bind_world[jname] = inv_bind.inverted()
                except Exception:
                    joint_bind_world[jname] = Matrix.Identity(4)

    bone_info  = {}
    name_to_id = {}

    def walk_joints(node, parent_id):
        node_id   = node.attrib.get("id",   "")
        node_name = (
            node.attrib.get("name")
            or node.attrib.get("sid")
            or node_id
        )
        node_type = node.attrib.get("type", "").upper()
        if node_type == "JOINT" and node_id:
            node_name_norm = node_name.replace(" ", "_")
            bone_info[node_id] = {"name": node_name_norm, "parent_id": parent_id}
            name_to_id[node_name] = node_id
            name_to_id[node_name_norm] = node_id
            for child in node.findall(q(ns, "node")):
                walk_joints(child, node_id)
        else:
            for child in node.findall(q(ns, "node")):
                walk_joints(child, parent_id)

    for node in vs.findall(q(ns, "node")):
        walk_joints(node, None)

    if not bone_info:
        return None

    id_to_id     = {jid: jid for jid in bone_info}
    name_to_id   = {}
    norm_to_id   = {}
    suffix_to_id = {}

    for jid, info in bone_info.items():
        raw_name  = scene_joint_names.get(jid, info["name"])
        norm_name = info["name"]
        name_to_id[raw_name]  = jid
        name_to_id[norm_name] = jid
        norm_to_id[norm_name] = jid
        parts = jid.replace("-","_").split("_")
        for i in range(len(parts)):
            suffix = "_".join(parts[i:])
            if suffix and suffix not in suffix_to_id:
                suffix_to_id[suffix] = jid

    def resolve_skin_ref(ref):
        ref_norm = ref.replace(" ", "_")
        return (id_to_id.get(ref) or name_to_id.get(ref) or
                norm_to_id.get(ref_norm) or suffix_to_id.get(ref) or
                suffix_to_id.get(ref_norm) or None)

    remapped = {}
    for skin_ref, world_mat in joint_bind_world.items():
        jid = resolve_skin_ref(skin_ref)
        remapped[jid if jid else skin_ref] = world_mat
    joint_bind_world = remapped

    # Some game and older 3ds Max/OpenCOLLADA files omit inverse-bind
    # matrices or identify joints differently.  Their visual-scene joint
    # transforms are still a valid rest-pose fallback.
    for joint_id, info in bone_info.items():
        if joint_id in joint_bind_world:
            continue
        candidates = (
            joint_id,
            info["name"],
            info["name"].replace("_", " "),
        )
        fallback = next(
            (
                scene_joint_world[candidate]
                for candidate in candidates
                if candidate in scene_joint_world
            ),
            None,
        )
        if fallback is not None:
            joint_bind_world[joint_id] = fallback

    if not joint_bind_world:
        return None

    arm_data = bpy.data.armatures.new(model_name)
    # Game-exported rigs often contain long helper/root bones.  Blender's
    # solid octahedral display can make those bones look like broken mesh
    # geometry on first import, so default to a less intrusive viewport style.
    arm_data.display_type = 'STICK'
    arm_obj  = bpy.data.objects.new(model_name, arm_data)
    collection.objects.link(arm_obj)
    arm_obj.matrix_world = correction_mat or Matrix.Identity(4)
    arm_obj.show_in_front = False

    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode='EDIT')
    edit_bones = arm_data.edit_bones
    created    = {}

    children_by_parent = {}
    for child_id, child_info in bone_info.items():
        children_by_parent.setdefault(
            child_info["parent_id"], []
        ).append(child_id)

    for bid, info in bone_info.items():
        if bid not in joint_bind_world:
            continue
        world      = joint_bind_world[bid]
        head_world = world.to_translation()
        eb         = edit_bones.new(info["name"])
        eb.head    = head_world
        children_with_pos = [
            child_id for child_id in children_by_parent.get(bid, [])
            if child_id in joint_bind_world
        ]
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

    for bid, info in bone_info.items():
        if bid not in created:
            continue
        pid = info["parent_id"]
        if pid and pid in created:
            created[bid].parent = created[pid]

    bpy.ops.object.mode_set(mode='OBJECT')
    print(f"Armature '{model_name}' created with {len(created)} bones.")
    return arm_obj


# ── SKIN WEIGHT PARSER ───────────────────────────────────────────────────────

def parse_controllers(root, ns):
    result   = {}
    ctrl_lib = root.find(f".//{q(ns,'library_controllers')}")
    if ctrl_lib is None:
        return result

    controller_sources = {}
    for ctrl in ctrl_lib.findall(q(ns, "controller")):
        ctrl_id = ctrl.attrib.get("id", "")
        skin = ctrl.find(q(ns, "skin"))
        morph = ctrl.find(q(ns, "morph"))
        if skin is not None:
            controller_sources[ctrl_id] = strip_url(skin.attrib.get("source"))
        elif morph is not None:
            controller_sources[ctrl_id] = strip_url(morph.attrib.get("source"))

    def resolve_geometry(source_id, stack=frozenset()):
        if source_id in stack or source_id not in controller_sources:
            return source_id
        return resolve_geometry(
            controller_sources[source_id], stack | {source_id}
        )

    for ctrl in ctrl_lib.findall(q(ns, "controller")):
        ctrl_id = ctrl.attrib.get("id", "")
        skin    = ctrl.find(q(ns, "skin"))
        if skin is None:
            continue
        skin_source = resolve_geometry(strip_url(skin.attrib.get("source")))
        bsm_elem    = skin.find(q(ns, "bind_shape_matrix"))
        bind_shape_matrix = parse_matrix(bsm_elem.text) if (bsm_elem is not None and bsm_elem.text) else Matrix.Identity(4)

        sources = {}
        for src in skin.findall(q(ns, "source")):
            src_id   = src.attrib.get("id", "")
            name_values = parse_source_name_array(src, ns)
            if name_values:
                sources[src_id] = name_values
                continue
            tuple_values = parse_source_float_array(src, ns)
            sources[src_id] = [v for item in tuple_values for v in item]

        joints_elem     = skin.find(q(ns, "joints"))
        joint_names_src = None
        if joints_elem is not None:
            for inp in joints_elem.findall(q(ns, "input")):
                if inp.attrib.get("semantic") == "JOINT":
                    joint_names_src = inp.attrib.get("source", "")[1:]

        joint_names = sources.get(joint_names_src, []) if joint_names_src else []
        real_bones  = [n for n in joint_names if not n.lower().startswith("notabone")]
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
            "joint_names":        [n.replace(" ", "_") for n in joint_names],
            "vertex_weights":     vertex_weights,
            "bind_shape_matrix":  bind_shape_matrix,
            "inv_bind_col_scale": 1.0,
            "inv_bind_R":         None,
        }
    return result


def parse_morph_controllers(root, ns):
    """Return morph targets grouped by their resolved base geometry."""
    controllers = {}
    controller_elements = root.findall(f".//{q(ns,'controller')}")
    for controller in controller_elements:
        controller_id = controller.attrib.get("id", "")
        skin = controller.find(q(ns, "skin"))
        morph = controller.find(q(ns, "morph"))
        if skin is not None:
            controllers[controller_id] = ("skin", strip_url(skin.attrib.get("source")))
        elif morph is not None:
            controllers[controller_id] = ("morph", strip_url(morph.attrib.get("source")))

    def resolve_geometry(source_id, stack=frozenset()):
        if source_id in stack:
            return source_id
        entry = controllers.get(source_id)
        if not entry:
            return source_id
        return resolve_geometry(entry[1], stack | {source_id})

    result = {}
    for controller in controller_elements:
        morph = controller.find(q(ns, "morph"))
        if morph is None:
            continue
        base_geometry = resolve_geometry(strip_url(morph.attrib.get("source")))
        sources = {}
        for source in morph.findall(q(ns, "source")):
            source_id = source.attrib.get("id", "")
            names = parse_source_name_array(source, ns)
            sources[source_id] = names if names else parse_source_float_array(source, ns)
        targets = morph.find(q(ns, "targets"))
        if targets is None:
            continue
        target_source = weight_source = None
        for inp in targets.findall(q(ns, "input")):
            semantic = inp.attrib.get("semantic")
            source_id = strip_url(inp.attrib.get("source"))
            if semantic == "MORPH_TARGET":
                target_source = source_id
            elif semantic == "MORPH_WEIGHT":
                weight_source = source_id
        target_ids = sources.get(target_source, [])
        raw_weights = sources.get(weight_source, [])
        weights = [
            item[0] if isinstance(item, tuple) and item else safe_float(item)
            for item in raw_weights
        ]
        if target_ids:
            result.setdefault(base_geometry, []).append({
                "controller_id": controller.attrib.get("id", ""),
                "method": morph.attrib.get("method", "NORMALIZED"),
                "targets": list(target_ids),
                "weights": weights,
            })
    return result


def geometry_position_array(geometry, ns):
    mesh = geometry.find(q(ns, "mesh")) if geometry is not None else None
    if mesh is None:
        return []
    sources = {
        source.attrib.get("id", ""): parse_source_float_array(source, ns)
        for source in mesh.findall(q(ns, "source"))
    }
    for vertices in mesh.findall(q(ns, "vertices")):
        for inp in vertices.findall(q(ns, "input")):
            if inp.attrib.get("semantic") == "POSITION":
                return sources.get(strip_url(inp.attrib.get("source")), [])
    for source_id, values in sources.items():
        if "position" in source_id.lower():
            return values
    return []


def apply_morph_targets(obj, base_geometry_id, morph_controllers, geometry_map, ns):
    morph_sets = morph_controllers.get(base_geometry_id, [])
    if not morph_sets or obj.type != 'MESH':
        return 0
    base_positions = geometry_position_array(geometry_map.get(base_geometry_id), ns)
    if not base_positions or len(base_positions) != len(obj.data.vertices):
        return 0
    if obj.data.shape_keys is None:
        obj.shape_key_add(name="Basis")
    imported = 0
    used_names = set(obj.data.shape_keys.key_blocks.keys())
    for morph_set in morph_sets:
        for target_index, target_geometry_id in enumerate(morph_set["targets"]):
            target_positions = geometry_position_array(
                geometry_map.get(target_geometry_id), ns
            )
            if len(target_positions) != len(base_positions):
                print(
                    f"Skipping morph '{target_geometry_id}': vertex count differs "
                    f"from '{base_geometry_id}'"
                )
                continue
            target_geometry = geometry_map.get(target_geometry_id)
            base_name = (
                target_geometry.attrib.get("name")
                if target_geometry is not None else target_geometry_id
            ) or target_geometry_id
            name = base_name
            suffix = 1
            while name in used_names:
                suffix += 1
                name = f"{base_name}.{suffix:03d}"
            used_names.add(name)
            key = obj.shape_key_add(name=name)
            coordinates = []
            for vertex, base, target in zip(
                obj.data.vertices, base_positions, target_positions
            ):
                delta = Vector(target[:3]) - Vector(base[:3])
                coordinates.extend(vertex.co + delta)
            key.data.foreach_set("co", coordinates)
            if target_index < len(morph_set["weights"]):
                key.value = morph_set["weights"][target_index]
            imported += 1
    if imported:
        print(f"Imported {imported} shape key(s) on '{obj.name}'.")
    return imported


def build_ctrl_mat_map(root, ns, controllers):
    geom_to_mat_override = {}
    for ic in root.findall(f".//{q(ns,'instance_controller')}"):
        ctrl_url = ic.attrib.get("url", "")[1:]
        if ctrl_url not in controllers:
            continue
        geom_id = controllers[ctrl_url]["skin_source"]
        mat_map = {}
        for im in ic.findall(f".//{q(ns,'instance_material')}"):
            mat_map[im.attrib.get("symbol","")] = im.attrib.get("target","")[1:]
        geom_to_mat_override[geom_id] = mat_map
    for ig in root.findall(f".//{q(ns,'instance_geometry')}"):
        geom_id = ig.attrib.get("url", "")[1:]
        mat_map = {}
        for im in ig.findall(f".//{q(ns,'instance_material')}"):
            mat_map[im.attrib.get("symbol","")] = im.attrib.get("target","")[1:]
        if mat_map:
            geom_to_mat_override[geom_id] = mat_map
    return geom_to_mat_override


# ── GEOMETRY IMPORTER ────────────────────────────────────────────────────────

def build_mesh_from_geometry(geom_elem, ns, collection, material_texture_map,
                              arm_obj, controllers, ctrl_mat_override, dae_filepath,
                              import_uvs=True, import_normals=True,
                              import_vertex_colors=True, merge_vertices=False,
                              merge_threshold=0.0001, runtime_cache=None):
    mesh_elem = geom_elem.find(q(ns, "mesh"))
    if mesh_elem is None:
        return None

    runtime_cache = runtime_cache if runtime_cache is not None else {}
    texture_path_cache = runtime_cache.setdefault("texture_paths", {})
    directory_cache = runtime_cache.setdefault("directory_entries", {})
    image_cache = runtime_cache.setdefault("images", {})
    material_cache = runtime_cache.setdefault("materials", {})
    skin_by_geometry = runtime_cache.setdefault(
        "skin_by_geometry",
        {
            controller["skin_source"]: controller
            for controller in controllers.values()
        },
    )

    geom_id   = geom_elem.attrib.get("id", "")
    geom_name = geom_elem.attrib.get("name") or geom_id or "DAE_Mesh"

    sources = {}
    for src in mesh_elem.findall(q(ns, "source")):
        src_id = src.attrib.get("id")
        if src_id:
            sources[src_id] = parse_source_float_array(src, ns)

    vertices_map       = {}
    vertices_normals   = {}
    vertices_texcoords = {}
    vertices_colors    = {}
    for verts in mesh_elem.findall(q(ns, "vertices")):
        v_id = verts.attrib.get("id")
        if not v_id:
            continue
        for inp in verts.findall(q(ns, "input")):
            sem = inp.attrib.get("semantic","")
            src = inp.attrib.get("source", "")[1:]
            if sem == "POSITION":  vertices_map[v_id]       = src
            elif sem == "NORMAL":  vertices_normals[v_id]   = src
            elif sem == "TEXCOORD": vertices_texcoords[v_id] = src
            elif sem == "COLOR":   vertices_colors[v_id]    = src

    positions    = None
    faces        = []
    face_mat_ids = []
    corner_uv_sets = {}
    corner_cols  = []
    corner_norms = []

    prim_blocks = (
        [
            (triangle, None, "TRIANGLES")
            for triangle in mesh_elem.findall(q(ns, "triangles"))
        ]
        + [
            (polylist, polylist.find(q(ns, "vcount")), "POLYLIST")
            for polylist in mesh_elem.findall(q(ns, "polylist"))
        ]
        + [
            (polygon, None, "POLYGONS")
            for polygon in mesh_elem.findall(q(ns, "polygons"))
        ]
        + [
            (fan, None, "TRIFANS")
            for fan in mesh_elem.findall(q(ns, "trifans"))
        ]
        + [
            (strip, None, "TRISTRIPS")
            for strip in mesh_elem.findall(q(ns, "tristrips"))
        ]
    )

    for prim, vcount_elem, primitive_kind in prim_blocks:
        count  = int(prim.attrib.get("count", "0"))
        p_elems = (
            prim.findall(q(ns, "p"))
            if primitive_kind in {"POLYGONS", "TRIFANS", "TRISTRIPS"}
            else [prim.find(q(ns, "p"))]
        )
        p_elems = [p for p in p_elems if p is not None and p.text]
        if not p_elems:
            continue

        tri_mat_symbol = prim.attrib.get("material")
        tri_mat_id     = ctrl_mat_override.get(tri_mat_symbol, tri_mat_symbol)

        all_inputs = []
        max_offset = 0
        for inp in prim.findall(q(ns, "input")):
            sem   = inp.attrib.get("semantic")
            src_val = inp.attrib.get("source", "")
            src   = src_val[1:] if src_val.startswith("#") else src_val
            off   = int(inp.attrib.get("offset", "0"))
            set_i = inp.attrib.get("set")
            all_inputs.append((sem, src, off, set_i))
            max_offset = max(max_offset, off)

        num_inputs = max_offset + 1

        input_by_offset = {}
        for sem, src, off, set_i in all_inputs:
            if off not in input_by_offset or sem == "VERTEX":
                input_by_offset[off] = (sem, src, set_i)

        vertex_offset = pos_source_id = None
        vertex_src_id = None
        for off, (sem, src, _) in input_by_offset.items():
            if sem == "VERTEX":
                vertex_offset = off
                vertex_src_id = src
                pos_source_id = vertices_map.get(src)
                break

        # Fallback 1: some exporters use POSITION directly instead of VERTEX
        if pos_source_id is None:
            for off, (sem, src, _) in input_by_offset.items():
                if sem == "POSITION":
                    vertex_offset = off
                    pos_source_id = src
                    break

        # Fallback 2: match by source ID containing "position" (Second Life etc.)
        if pos_source_id is None:
            for src_id in sources.keys():
                if "position" in src_id.lower():
                    pos_source_id = src_id
                    for off, (s_sem, s_src, _) in input_by_offset.items():
                        if s_src == pos_source_id:
                            vertex_offset = off
                            break
                    break

        # Last resort: any source ID containing "pos"
        if pos_source_id is None:
            for src_id in sources.keys():
                if "pos" in src_id.lower():
                    pos_source_id = src_id
                    for off, (s_sem, s_src, _) in input_by_offset.items():
                        if s_src == pos_source_id:
                            vertex_offset = off
                            break
                    break

        if pos_source_id is None:
            print("Missing POSITION source in:", geom_name)
            return None

        positions = sources.get(pos_source_id)
        if not positions:
            print("Position source missing:", pos_source_id)
            return None

        normal_offset = color_offset = None
        normal_source = color_source = None
        uv_inputs = {}

        # Scan ALL inputs (not just input_by_offset) so we catch TEXCOORD and COLOR
        # even when they share offset=0 with VERTEX (common in OoT/MM 3DS exports).
        for sem, src, off, set_idx in all_inputs:
            if sem == "NORMAL":
                if normal_source is None:
                    normal_offset = off;  normal_source = sources.get(src)
            elif sem == "COLOR":
                if color_source is None:
                    color_offset  = off;  color_source  = sources.get(src)
            elif sem == "TEXCOORD":
                uv_set = str(set_idx if set_idx is not None else len(uv_inputs))
                uv_inputs.setdefault(uv_set, (off, sources.get(src)))

        # Fallback: check if NORMAL/TEXCOORD/COLOR were declared inside <vertices>
        if normal_source is None and vertex_src_id in vertices_normals:
            normal_offset = vertex_offset
            normal_source = sources.get(vertices_normals[vertex_src_id])
        if not uv_inputs and vertex_src_id in vertices_texcoords:
            uv_inputs["0"] = (vertex_offset, sources.get(vertices_texcoords[vertex_src_id]))
        if color_source is None and vertex_src_id in vertices_colors:
            color_offset = vertex_offset
            color_source = sources.get(vertices_colors[vertex_src_id])

        raw_chunks = [[int(x) for x in p.text.strip().split()] for p in p_elems]
        raw_idx = [index for chunk in raw_chunks for index in chunk]

        if primitive_kind in {"POLYGONS", "TRIFANS", "TRISTRIPS"}:
            vcounts = [len(chunk) // num_inputs for chunk in raw_chunks]
        elif vcount_elem is not None and vcount_elem.text:
            vcounts = [int(x) for x in vcount_elem.text.strip().split()]
        else:
            real_count = len(raw_idx) // (3 * num_inputs) if num_inputs > 0 else count
            if real_count != count and real_count > 0:
                print(f"  [{geom_name}] count={count} but p has {len(raw_idx)} values "
                      f"-> correcting to {real_count} triangles")
                count = real_count
            vcounts = [3] * count

        cursor = 0
        for poly_vcount in vcounts:
            poly_vi   = []
            poly_uv_sets = {uv_set: [] for uv_set in uv_inputs}
            poly_col  = []
            poly_norm = []

            for v in range(poly_vcount):
                b  = cursor + v * num_inputs
                if vertex_offset is None or b + vertex_offset >= len(raw_idx):
                    continue
                vi = raw_idx[b + vertex_offset]
                poly_vi.append(vi)

                if normal_offset is not None and normal_source:
                    ni = raw_idx[b + normal_offset]
                    poly_norm.append(Vector(normal_source[ni]) if 0 <= ni < len(normal_source) else Vector((0,0,1)))

                if color_offset is not None and color_source:
                    ci = raw_idx[b + color_offset]
                    if 0 <= ci < len(color_source):
                        c = color_source[ci]
                        poly_col.append((c[0], c[1], c[2], c[3] if len(c) == 4 else 1.0))
                    else:
                        poly_col.append((1,1,1,1))

                for uv_set, (uv_offset, uv_source) in uv_inputs.items():
                    if uv_source and b + uv_offset < len(raw_idx):
                        ti = raw_idx[b + uv_offset]
                        uv = uv_source[ti] if 0 <= ti < len(uv_source) else (0, 0)
                    else:
                        uv = (0, 0)
                    poly_uv_sets[uv_set].append((uv[0], uv[1]))

            cursor += poly_vcount * num_inputs
            if len(poly_vi) != poly_vcount:
                continue

            if primitive_kind == "TRISTRIPS":
                corner_triplets = [
                    (
                        (index, index + 1, index + 2)
                        if index % 2 == 0
                        else (index + 1, index, index + 2)
                    )
                    for index in range(poly_vcount - 2)
                ]
            else:
                corner_triplets = [
                    (0, index, index + 1)
                    for index in range(1, poly_vcount - 1)
                ]

            for corner_a, corner_b, corner_c in corner_triplets:
                tri_vi = [
                    poly_vi[corner_a],
                    poly_vi[corner_b],
                    poly_vi[corner_c],
                ]
                if len(set(tri_vi)) < 3:
                    continue
                faces.append(tuple(tri_vi))
                face_mat_ids.append(tri_mat_id)
                if poly_norm:
                    corner_norms.extend(
                        [
                            poly_norm[corner_a],
                            poly_norm[corner_b],
                            poly_norm[corner_c],
                        ]
                    )
                if poly_col:
                    corner_cols.extend(
                        [
                            poly_col[corner_a],
                            poly_col[corner_b],
                            poly_col[corner_c],
                        ]
                    )
                for uv_set, poly_uvs in poly_uv_sets.items():
                    if len(poly_uvs) == poly_vcount:
                        corner_uv_sets.setdefault(uv_set, []).extend(
                            [
                                poly_uvs[corner_a],
                                poly_uvs[corner_b],
                                poly_uvs[corner_c],
                            ]
                        )

    if not positions or not faces:
        print("No valid geometry in:", geom_name)
        return None

    # ── CREATE MESH ──────────────────────────────────────────────────────────
    skin_ctrl = skin_by_geometry.get(geom_id)

    if skin_ctrl is not None:
        bsm = skin_ctrl.get("bind_shape_matrix", Matrix.Identity(4))
        if bsm != Matrix.Identity(4):
            bsm3 = bsm.to_3x3()
            bsm_t = bsm.to_translation()
            positions = [tuple(bsm3 @ Vector(p) + bsm_t) for p in positions]

        ibm_col_scale = skin_ctrl.get("inv_bind_col_scale", 1.0)
        if abs(ibm_col_scale - 1.0) > 0.001:
            s     = ibm_col_scale
            ibm_R = skin_ctrl.get("inv_bind_R", None)
            if ibm_R is not None:
                positions = [tuple(ibm_R @ (Vector(p) * (1.0/s))) for p in positions]
            else:
                positions = [tuple(x / s for x in p) for p in positions]

    mesh = bpy.data.meshes.new(geom_name)
    mesh.vertices.add(len(positions))
    mesh.vertices.foreach_set(
        "co",
        [
            component
            for position in positions
            for component in position[:3]
        ],
    )
    loop_vertex_indices = [
        vertex_index for face in faces for vertex_index in face
    ]
    mesh.loops.add(len(loop_vertex_indices))
    mesh.loops.foreach_set("vertex_index", loop_vertex_indices)
    mesh.polygons.add(len(faces))
    mesh.polygons.foreach_set(
        "loop_start", range(0, len(loop_vertex_indices), 3)
    )
    mesh.polygons.foreach_set("loop_total", [3] * len(faces))
    mesh.update(calc_edges=True)

    obj = bpy.data.objects.new(geom_name, mesh)
    collection.objects.link(obj)

    # ── MATERIALS ────────────────────────────────────────────────────────────
    dae_dir = os.path.dirname(bpy.path.abspath(dae_filepath))

    def _resolve_tex(raw_path):
        if not raw_path:
            return None
        cache_key = (dae_dir, str(raw_path))
        if cache_key in texture_path_cache:
            return texture_path_cache[cache_key]
        raw_path = urllib.parse.unquote(str(raw_path).strip())
        if raw_path.lower().startswith("file://"):
            parsed_url = urllib.parse.urlparse(raw_path)
            if parsed_url.path:
                if parsed_url.netloc:
                    raw_path = (
                        f"//{parsed_url.netloc}{parsed_url.path}"
                    )
                else:
                    raw_path = parsed_url.path
            elif parsed_url.netloc:
                # 3ds Max commonly emits file://C:\folder\texture.png,
                # which urllib interprets entirely as the URL authority.
                raw_path = parsed_url.netloc
            else:
                raw_path = raw_path[7:]
            if raw_path.startswith("/") and len(raw_path) > 2 and raw_path[2] == ":":
                raw_path = raw_path[1:]
            raw_path = urllib.parse.unquote(raw_path)
        fname = os.path.basename(raw_path)
        candidates = [
            raw_path,
            os.path.join(dae_dir, raw_path),
            os.path.join(dae_dir, fname),
        ]
        parent = os.path.dirname(dae_dir)
        for _ in range(2):
            candidates.append(os.path.join(parent, fname))
            if parent not in directory_cache:
                try:
                    directory_cache[parent] = [
                        entry.path
                        for entry in os.scandir(parent)
                        if entry.is_dir()
                    ]
                except Exception:
                    directory_cache[parent] = []
            candidates.extend(
                os.path.join(subdirectory, fname)
                for subdirectory in directory_cache[parent]
            )
            parent = os.path.dirname(parent)
        for c in candidates:
            c = os.path.normpath(c)
            if os.path.isfile(c):
                texture_path_cache[cache_key] = c
                return c
        texture_path_cache[cache_key] = None
        return None

    def _load_img(raw_path, colorspace="sRGB"):
        resolved = _resolve_tex(raw_path)
        if not resolved:
            return None
        cache_key = (os.path.normcase(resolved), colorspace)
        if cache_key in image_cache:
            return image_cache[cache_key]
        try:
            img = bpy.data.images.load(resolved, check_existing=True)
            try:
                img.colorspace_settings.name = colorspace
            except Exception:
                pass
            image_cache[cache_key] = img
            return img
        except Exception as e:
            print(f"Failed to load texture '{resolved}': {e}")
            return None

    def _mat_diffuse_path(m):
        if not m.use_nodes:
            return None
        for n in m.node_tree.nodes:
            if n.type == 'TEX_IMAGE' and n.image and n.label == "diffuse":
                return os.path.normpath(bpy.path.abspath(n.image.filepath))
        return None

    def _build_mat_nodes(m, channels):
        m.use_nodes = True
        nodes = m.node_tree.nodes
        links = m.node_tree.links
        nodes.clear()

        out_n  = nodes.new("ShaderNodeOutputMaterial")
        out_n.location = (600, 0)

        bsdf_n = nodes.new("ShaderNodeBsdfPrincipled")
        bsdf_n.location = (200, 0)
        links.new(bsdf_n.outputs["BSDF"], out_n.inputs["Surface"])
        diffuse_color = channels.get("_diffuse_color", (0.6, 0.6, 0.6, 1.0))
        bsdf_n.inputs["Base Color"].default_value = diffuse_color
        if "Roughness" in bsdf_n.inputs:
            bsdf_n.inputs["Roughness"].default_value = channels.get("_roughness", 0.65)
        specular_color = channels.get("_specular_color")
        specular_level = (
            sum(specular_color[:3]) / 3.0 if specular_color else 0.25
        )
        for spec_input in ("Specular IOR Level", "Specular"):
            if spec_input in bsdf_n.inputs:
                bsdf_n.inputs[spec_input].default_value = max(0.0, min(1.0, specular_level))
                break

        emission_color = channels.get("_emission_color")
        if emission_color:
            emission_input = (
                bsdf_n.inputs.get("Emission Color") or bsdf_n.inputs.get("Emission")
            )
            if emission_input:
                emission_input.default_value = emission_color
                if "Emission Strength" in bsdf_n.inputs:
                    bsdf_n.inputs["Emission Strength"].default_value = 1.0

        diff_path = channels.get("diffuse")
        tex_n = None
        if diff_path:
            img = _load_img(diff_path, "sRGB")
            if not img:
                bsdf_n.inputs["Base Color"].default_value = (0.8, 0.5, 0.2, 1.0)
                print(f"  Missing texture: '{diff_path}'")
            else:
                tex_n          = nodes.new("ShaderNodeTexImage")
                tex_n.image    = img
                tex_n.label    = "diffuse"
                tex_n.location = (-300, 0)
                links.new(tex_n.outputs["Color"], bsdf_n.inputs["Base Color"])

        transparency = max(0.0, min(1.0, channels.get("_transparency", diffuse_color[3])))
        if transparency < 0.999 or channels.get("alpha"):
            alpha_path = channels.get("alpha")
            alpha_tex = tex_n
            if alpha_path and alpha_path != diff_path:
                alpha_img = _load_img(alpha_path, "Non-Color")
                if alpha_img:
                    alpha_tex = nodes.new("ShaderNodeTexImage")
                    alpha_tex.image = alpha_img
                    alpha_tex.label = "alpha"
                    alpha_tex.location = (-300, -500)
            if alpha_tex is not None:
                links.new(alpha_tex.outputs["Alpha"], bsdf_n.inputs["Alpha"])
            bsdf_n.inputs["Alpha"].default_value = transparency
            try:
                if hasattr(m, "surface_render_method"):
                    m.surface_render_method = 'DITHERED'
                elif hasattr(m, "blend_method"):
                    m.blend_method = 'BLEND'
            except Exception:
                pass

        normal_path = channels.get("normal")
        if normal_path:
            normal_img = _load_img(normal_path, "Non-Color")
            if normal_img:
                normal_tex = nodes.new("ShaderNodeTexImage")
                normal_tex.image = normal_img
                normal_tex.label = "normal"
                normal_tex.location = (-500, -200)
                normal_map = nodes.new("ShaderNodeNormalMap")
                normal_map.location = (-100, -200)
                links.new(normal_tex.outputs["Color"], normal_map.inputs["Color"])
                links.new(normal_map.outputs["Normal"], bsdf_n.inputs["Normal"])

        specular_path = channels.get("specular")
        if specular_path:
            spec_img = _load_img(specular_path, "Non-Color")
            if spec_img:
                spec_tex = nodes.new("ShaderNodeTexImage")
                spec_tex.image = spec_img
                spec_tex.label = "specular"
                spec_tex.location = (-500, -400)
                for spec_input in ("Specular IOR Level", "Specular"):
                    if spec_input in bsdf_n.inputs:
                        links.new(spec_tex.outputs["Color"], bsdf_n.inputs[spec_input])
                        break

    unique_mat_ids = sorted({m for m in face_mat_ids if m is not None})
    mat_index_map  = {}
    obj.data.materials.clear()

    for idx, mat_id in enumerate(unique_mat_ids):
        channels  = material_texture_map.get(mat_id, {})
        diff_path = _resolve_tex(channels.get("diffuse"))
        tex_base  = os.path.splitext(os.path.basename(diff_path))[0] if diff_path else mat_id
        material_cache_key = (
            mat_id,
            os.path.normcase(diff_path) if diff_path else None,
        )

        if material_cache_key in material_cache:
            mat = material_cache[material_cache_key]
        else:
            existing   = bpy.data.materials.get(tex_base)
            want_path  = os.path.normpath(diff_path) if diff_path else None
            exist_path = (
                _mat_diffuse_path(existing)
                if existing is not None else None
            )
            if (
                existing is not None
                and (
                    (want_path is None and exist_path is None)
                    or exist_path == want_path
                )
            ):
                mat = existing
            else:
                mat = (
                    bpy.data.materials.new(tex_base)
                    if existing is None else existing
                )
                _build_mat_nodes(mat, channels)
                if diff_path:
                    print(
                        f"Material: '{mat.name}' <- "
                        f"'{os.path.basename(diff_path)}'"
                    )
                else:
                    print(f"Material: '{mat.name}' (no diffuse)")
            material_cache[material_cache_key] = mat

        obj.data.materials.append(mat)
        mat_index_map[mat_id] = idx

    mesh.polygons.foreach_set(
        "material_index",
        [
            mat_index_map.get(material_id, 0)
            for material_id in face_mat_ids
        ],
    )

    # ── UVs ──────────────────────────────────────────────────────────────────
    if import_uvs:
        def _uv_sort_key(item):
            name = item[0]
            return (0, int(name)) if name.isdigit() else (1, name)

        for layer_index, (set_name, uvs) in enumerate(sorted(corner_uv_sets.items(), key=_uv_sort_key)):
            if not uvs or len(uvs) != len(mesh.loops):
                print(f"Skipping incomplete UV set {set_name!r} on '{geom_name}'")
                continue
            layer_name = "UVMap" if layer_index == 0 else f"UVMap.{layer_index:03d}"
            uv_layer = mesh.uv_layers.new(name=layer_name)
            uv_layer.data.foreach_set(
                "uv", [component for uv in uvs for component in uv[:2]]
            )

    # ── VERTEX COLORS ────────────────────────────────────────────────────────
    if import_vertex_colors and corner_cols and len(corner_cols) == len(mesh.loops):
        col_attr = mesh.color_attributes.new(name="Col", type="FLOAT_COLOR", domain="CORNER")
        col_attr.data.foreach_set(
            "color",
            [
                max(0.0, min(1.0, component))
                for color in corner_cols
                for component in color
            ],
        )

    # ── NORMALS ──────────────────────────────────────────────────────────────
    if import_normals and corner_norms and len(corner_norms) == len(mesh.loops):
        try:
            mesh.normals_split_custom_set(corner_norms)
        except Exception:
            pass

    # ── REMOVE STRAY VERTICES ────────────────────────────────────────────────
    referenced = set(v for f in faces for v in f)
    if skin_ctrl is None and len(referenced) < len(positions):
        import bmesh as _bm
        bm2 = _bm.new()
        bm2.from_mesh(mesh)
        stray = [v for v in bm2.verts if not v.link_edges]
        if stray:
            _bm.ops.delete(bm2, geom=stray, context='VERTS')
            bm2.to_mesh(mesh)
            mesh.update()
        bm2.free()

    # ── MERGE VERTICES ───────────────────────────────────────────────────────
    if merge_vertices and skin_ctrl is None:
        import bmesh
        bm = bmesh.new()
        bm.from_mesh(mesh)
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=merge_threshold)
        bm.to_mesh(mesh)
        bm.free()
        mesh.update()

    # ── SKIN WEIGHTS ─────────────────────────────────────────────────────────
    if arm_obj is not None and skin_ctrl is not None and skin_ctrl["vertex_weights"]:
        joint_names    = skin_ctrl["joint_names"]
        vertex_weights = skin_ctrl["vertex_weights"]
        vgroups = {}
        for jname in joint_names:
            if jname.lower().startswith("notabone"):
                vgroups[jname] = None
            else:
                vgroups[jname] = obj.vertex_groups.new(name=jname)
        for vert_idx, pairs in vertex_weights.items():
            valid_pairs = [
                (j_idx, weight) for j_idx, weight in pairs
                if 0 <= j_idx < len(joint_names) and weight > 0.0
            ]
            total_weight = sum(weight for _, weight in valid_pairs)
            if total_weight <= 0.0 or vert_idx >= len(obj.data.vertices):
                continue
            for j_idx, weight in valid_pairs:
                vg = vgroups.get(joint_names[j_idx])
                if vg is not None:
                    vg.add([vert_idx], weight / total_weight, 'REPLACE')
        obj.parent = arm_obj
        mod = obj.modifiers.new(name="Armature", type='ARMATURE')
        mod.object            = arm_obj
        mod.use_vertex_groups = True
        print(f"Skin weights applied to '{geom_name}' ({len(vgroups)} groups).")
    return obj


def parse_camera_library(root, ns):
    result = {}
    for camera in root.findall(f".//{q(ns,'camera')}"):
        camera_id = camera.attrib.get("id")
        optics = camera.find(f"{q(ns,'optics')}/{q(ns,'technique_common')}")
        if not camera_id or optics is None:
            continue
        projection = next(iter(optics), None)
        if projection is None:
            continue
        values = {}
        for child in projection:
            if child.text:
                values[xml_local_name(child.tag)] = safe_float(child.text)
        result[camera_id] = {
            "name": camera.attrib.get("name") or camera_id,
            "projection": xml_local_name(projection.tag),
            "values": values,
        }
    return result


def create_camera_object(camera_id, camera_defs, collection, name=None):
    definition = camera_defs.get(camera_id)
    if not definition:
        return None
    camera_data = bpy.data.cameras.new(name or definition["name"])
    values = definition["values"]
    if definition["projection"] == "orthographic":
        camera_data.type = 'ORTHO'
        camera_data.ortho_scale = max(
            values.get("xmag", 0.0) * 2.0,
            values.get("ymag", 0.0) * 2.0,
            0.001,
        )
    else:
        camera_data.type = 'PERSP'
        angle = values.get("xfov") or values.get("yfov")
        if angle:
            camera_data.angle = math.radians(angle)
    camera_data.clip_start = max(values.get("znear", 0.1), 0.0001)
    camera_data.clip_end = max(values.get("zfar", 1000.0), camera_data.clip_start + 0.001)
    obj = bpy.data.objects.new(name or definition["name"], camera_data)
    collection.objects.link(obj)
    return obj


def parse_light_library(root, ns):
    result = {}
    for light in root.findall(f".//{q(ns,'light')}"):
        light_id = light.attrib.get("id")
        technique = light.find(q(ns, "technique_common"))
        if not light_id or technique is None:
            continue
        light_type = next(iter(technique), None)
        if light_type is None:
            continue
        values = {}
        for child in light_type:
            key = xml_local_name(child.tag)
            if key == "color" and child.text:
                values[key] = tuple(safe_float(v) for v in child.text.split()[:3])
            elif child.text:
                values[key] = safe_float(child.text)
        result[light_id] = {
            "name": light.attrib.get("name") or light_id,
            "type": xml_local_name(light_type.tag),
            "values": values,
        }
    return result


def create_light_object(light_id, light_defs, collection, name=None):
    definition = light_defs.get(light_id)
    if not definition:
        return None
    blender_type = {
        "directional": "SUN",
        "point": "POINT",
        "spot": "SPOT",
        "ambient": "AREA",
    }.get(definition["type"], "POINT")
    light_data = bpy.data.lights.new(name or definition["name"], type=blender_type)
    values = definition["values"]
    light_data.color = tuple(max(0.0, value) for value in values.get("color", (1.0, 1.0, 1.0)))
    light_data.energy = 1000.0 if blender_type != "SUN" else 3.0
    if blender_type == "SPOT":
        light_data.spot_size = math.radians(max(1.0, values.get("falloff_angle", 45.0) * 2.0))
        exponent = values.get("falloff_exponent", 0.0)
        light_data.spot_blend = max(0.0, min(1.0, 1.0 - exponent / 128.0))
    obj = bpy.data.objects.new(name or definition["name"], light_data)
    collection.objects.link(obj)
    return obj


def import_collada_animations(root, ns, node_targets, arm_obj, scene,
                              root_correction=Matrix.Identity(4)):
    animation_library = root.find(q(ns, "library_animations"))
    if animation_library is None:
        return 0

    sources = {}
    for source in animation_library.findall(f".//{q(ns,'source')}"):
        source_id = source.attrib.get("id")
        if not source_id:
            continue
        names = parse_source_name_array(source, ns)
        sources[source_id] = names if names else parse_source_float_array(source, ns)

    samplers = {}
    for sampler in animation_library.findall(f".//{q(ns,'sampler')}"):
        sampler_id = sampler.attrib.get("id")
        if not sampler_id:
            continue
        samplers[sampler_id] = {
            inp.attrib.get("semantic"): strip_url(inp.attrib.get("source"))
            for inp in sampler.findall(q(ns, "input"))
        }

    joint_nodes = {}
    visual_scene = find_active_visual_scene(root, ns)
    if arm_obj is not None and visual_scene is not None:
        for node in visual_scene.findall(f".//{q(ns,'node')}[@type='JOINT']"):
            node_id = node.attrib.get("id", "")
            node_name = (node.attrib.get("name") or node_id).replace(" ", "_")
            bone = arm_obj.pose.bones.get(node_name)
            if bone:
                joint_nodes[node_id] = (bone, parse_node_transform(node, ns))
                joint_nodes[node.attrib.get("sid", "")] = (bone, parse_node_transform(node, ns))

    fps = scene.render.fps / max(scene.render.fps_base, 1.0e-8)
    keyed = 0
    max_frame = scene.frame_start

    def timeline_frame_index(frame):
        nearest = round(frame)
        if abs(frame - nearest) < 1.0e-4:
            return int(nearest)
        return int(math.ceil(frame))

    def set_transform(target, matrix, frame):
        nonlocal keyed, max_frame
        if isinstance(target, tuple):
            if target and target[0] == "MORPH":
                return
            pose_bone, rest_local = target
            try:
                pose_bone.matrix_basis = rest_local.inverted_safe() @ matrix
            except Exception:
                pose_bone.matrix_basis = matrix
            pose_bone.rotation_mode = 'QUATERNION'
            pose_bone.keyframe_insert("location", frame=frame, group=pose_bone.name)
            pose_bone.keyframe_insert("rotation_quaternion", frame=frame, group=pose_bone.name)
            pose_bone.keyframe_insert("scale", frame=frame, group=pose_bone.name)
        else:
            if target.parent is None:
                matrix = root_correction @ matrix
            location, rotation, scale = matrix.decompose()
            target.location = location
            target.rotation_mode = 'QUATERNION'
            target.rotation_quaternion = rotation
            target.scale = scale
            target.keyframe_insert("location", frame=frame)
            target.keyframe_insert("rotation_quaternion", frame=frame)
            target.keyframe_insert("scale", frame=frame)
        keyed += 1
        max_frame = max(max_frame, timeline_frame_index(frame))

    for channel in animation_library.findall(f".//{q(ns,'channel')}"):
        sampler = samplers.get(strip_url(channel.attrib.get("source")))
        target_path = channel.attrib.get("target", "")
        if not sampler or "/" not in target_path:
            continue
        target_name, transform_path = target_path.split("/", 1)
        targets = list(node_targets.get(target_name, []))
        if target_name in joint_nodes:
            targets.append(joint_nodes[target_name])
        if not targets:
            continue

        input_values = sources.get(sampler.get("INPUT"), [])
        output_values = sources.get(sampler.get("OUTPUT"), [])
        if not input_values or not output_values:
            continue
        times = [value[0] if isinstance(value, tuple) else safe_float(value) for value in input_values]
        channel_frame_offset = (
            scene.frame_start - min(times) * fps if times else scene.frame_start
        )

        transform_lower = transform_path.lower()
        morph_targets = [
            target[1] for target in targets
            if isinstance(target, tuple) and target and target[0] == "MORPH"
        ]
        if morph_targets and "weight" in transform_lower:
            match = re.search(r"\((\d+)\)|\[(\d+)\]", transform_lower)
            weight_index = int(next(
                group for group in match.groups() if group is not None
            )) if match else 0
            for time_value, output in zip(times, output_values):
                values = output if isinstance(output, tuple) else (safe_float(output),)
                frame = channel_frame_offset + time_value * fps
                for target in morph_targets:
                    if (
                        target.data.shape_keys and
                        weight_index + 1 < len(target.data.shape_keys.key_blocks)
                    ):
                        key = target.data.shape_keys.key_blocks[weight_index + 1]
                        key.value = values[0]
                        key.keyframe_insert("value", frame=frame)
                        keyed += 1
                        max_frame = max(
                            max_frame, timeline_frame_index(frame)
                        )
            continue

        if ("matrix" in transform_lower or "transform" in transform_lower):
            for time_value, output in zip(times, output_values):
                if not isinstance(output, tuple) or len(output) != 16:
                    continue
                matrix = matrix_from_collada_values(list(output))
                frame = channel_frame_offset + time_value * fps
                for target in targets:
                    set_transform(target, matrix, frame)
            continue

        # Common vector/component animation channels used by Maya, Max, and
        # OpenCOLLADA. These are applied to object nodes; matrix channels above
        # are used for armature joints.
        object_targets = [target for target in targets if not isinstance(target, tuple)]
        if not object_targets:
            continue
        for time_value, output in zip(times, output_values):
            values = output if isinstance(output, tuple) else (safe_float(output),)
            frame = channel_frame_offset + time_value * fps
            for target in object_targets:
                if "translate" in transform_lower:
                    if len(values) >= 3:
                        target.location = values[:3]
                        target.keyframe_insert("location", frame=frame)
                    else:
                        axis = next((i for i, axis_name in enumerate("xyz")
                                     if axis_name in transform_lower), 0)
                        target.location[axis] = values[0]
                        target.keyframe_insert("location", index=axis, frame=frame)
                elif "scale" in transform_lower:
                    if len(values) >= 3:
                        target.scale = values[:3]
                        target.keyframe_insert("scale", frame=frame)
                    else:
                        axis = next((i for i, axis_name in enumerate("xyz")
                                     if axis_name in transform_lower), 0)
                        target.scale[axis] = values[0]
                        target.keyframe_insert("scale", index=axis, frame=frame)
                elif "rotate" in transform_lower or "rotation" in transform_lower:
                    target.rotation_mode = 'XYZ'
                    axis = next((i for i, axis_name in enumerate("xyz")
                                 if axis_name in transform_lower), 2)
                    angle = values[3] if len(values) >= 4 else values[0]
                    target.rotation_euler[axis] = math.radians(angle)
                    target.keyframe_insert("rotation_euler", index=axis, frame=frame)
                keyed += 1
                max_frame = max(max_frame, timeline_frame_index(frame))

    if keyed:
        scene.frame_end = max_frame
    return keyed


# ── IMPORT OPERATOR ──────────────────────────────────────────────────────────

class IMPORT_OT_simple_collada_full(Operator, ImportHelper):
    """Import one or more COLLADA (.dae) files"""
    bl_idname    = "import_scene.simple_collada_full"
    bl_label     = "Import COLLADA (.dae)"
    filename_ext = ".dae"
    filter_glob: StringProperty(default="*.dae", options={'HIDDEN'})
    files: CollectionProperty(
        name="DAE Files",
        type=OperatorFileListElement,
        options={'HIDDEN', 'SKIP_SAVE'},
    )
    directory: StringProperty(
        name="Directory",
        subtype='DIR_PATH',
        options={'HIDDEN', 'SKIP_SAVE'},
    )
    batch_create_collections: BoolProperty(
        name="Collection per File",
        description=(
            "When importing multiple files, place each DAE in its own "
            "collection"
        ),
        default=True,
    )

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
    import_units: BoolProperty(
        name="Convert File Units",
        description="Convert the COLLADA unit scale to Blender meters",
        default=True,
    )
    import_hierarchy: BoolProperty(
        name="Preserve Node Hierarchy",
        description="Create empties for COLLADA nodes and preserve parent-child transforms",
        default=True,
    )
    import_cameras: BoolProperty(
        name="Import Cameras",
        default=True,
    )
    import_lights: BoolProperty(
        name="Import Lights",
        default=True,
    )
    import_animation: BoolProperty(
        name="Import Animation",
        description="Import object and matrix-based armature animation",
        default=True,
    )
    import_shape_keys: BoolProperty(
        name="Import Shape Keys",
        description="Import COLLADA morph controllers as Blender shape keys",
        default=True,
    )
    merge_vertices: BoolProperty(
        name="Merge Vertices",
        description="Remove duplicate vertices by distance after import",
        default=False,
    )
    merge_threshold: FloatProperty(
        name="Merge Distance",
        default=0.0001, min=0.0, max=0.1, precision=5,
    )

    def _selected_filepaths(self):
        if self.files:
            base_directory = (
                self.directory
                or os.path.dirname(self.filepath)
            )
            return [
                os.path.normpath(os.path.join(base_directory, item.name))
                for item in self.files
                if item.name.lower().endswith(".dae")
            ]
        return [os.path.normpath(self.filepath)] if self.filepath else []

    def _notify(self, levels, message):
        if 'ERROR' in levels:
            self._last_import_error = message
        elif 'INFO' in levels:
            self._last_import_summary = message

    def execute(self, context):
        filepaths = self._selected_filepaths()
        if not filepaths:
            self.report({'ERROR'}, "No COLLADA files selected.")
            return {'CANCELLED'}

        original_filepath = self.filepath
        self._batch_mode = len(filepaths) > 1
        successes = []
        failures = []
        parent_collection = (
            context.view_layer.active_layer_collection.collection
            if context.view_layer.active_layer_collection
            else context.scene.collection
        )

        try:
            for filepath in filepaths:
                self.filepath = filepath
                self._cached_root = None
                self._cached_profile = None
                self._last_import_error = None
                self._last_import_summary = None
                self._import_stage = "starting"
                self._target_collection = None

                if self._batch_mode and self.batch_create_collections:
                    collection_name = os.path.splitext(
                        os.path.basename(filepath)
                    )[0]
                    target_collection = bpy.data.collections.new(
                        collection_name
                    )
                    parent_collection.children.link(target_collection)
                    self._target_collection = target_collection

                started = time.perf_counter()
                try:
                    result = self._execute_single(context)
                except Exception as error:
                    traceback.print_exc()
                    result = {'CANCELLED'}
                    self._last_import_error = (
                        f"{type(error).__name__} during "
                        f"{self._import_stage}: {error}"
                    )

                elapsed = time.perf_counter() - started
                filename = os.path.basename(filepath)
                if result == {'FINISHED'}:
                    successes.append((filename, elapsed))
                else:
                    failures.append(
                        (
                            filename,
                            self._last_import_error
                            or f"Import stopped during {self._import_stage}",
                        )
                    )
                    target = self._target_collection
                    if target is not None and not target.objects:
                        parent_collection.children.unlink(target)
                        bpy.data.collections.remove(target)
        finally:
            self.filepath = original_filepath
            self._batch_mode = False
            self._target_collection = None

        if len(filepaths) == 1:
            if failures:
                filename, reason = failures[0]
                self.report({'ERROR'}, f"{filename}: {reason}")
                return {'CANCELLED'}
            filename, elapsed = successes[0]
            summary = self._last_import_summary or f"Imported {filename}"
            self.report({'INFO'}, f"{summary} ({elapsed:.2f}s)")
            return {'FINISHED'}

        if failures:
            print("[DAE Batch Import] Failures:")
            for filename, reason in failures:
                print(f"  {filename}: {reason}")
        message = (
            f"Batch import: {len(successes)} succeeded, "
            f"{len(failures)} failed."
        )
        self.report(
            {'WARNING'} if failures else {'INFO'},
            message + (
                " See the console for per-file errors."
                if failures else ""
            ),
        )
        return {'FINISHED'} if successes else {'CANCELLED'}

    def _execute_single(self, context):
        self._prescan()

        if not os.path.isfile(self.filepath):
            self._notify({'ERROR'}, f"File not found: {self.filepath}")
            return {'CANCELLED'}

        self._import_stage = "parsing XML"
        # Reuse the tree already parsed by _prescan if available (avoids double parse)
        if hasattr(self, '_cached_root') and self._cached_root is not None:
            root = self._cached_root
            self._cached_root = None  # clear cache after use
        else:
            try:
                tree = ET.parse(self.filepath)
                root = tree.getroot()
            except ET.ParseError as e:
                try:
                    import re
                    with open(self.filepath, 'r', encoding='utf-8', errors='replace') as f:
                        raw = f.read()
                    raw = re.sub(r'<\w+:\w+[^>]*/>', '', raw)
                    raw = re.sub(r'<(\w+:\w+)[^>]*>.*?</\1>', '', raw, flags=re.DOTALL)
                    raw = re.sub(r'<(\w+):(\w+)', r'<\2', raw)
                    raw = re.sub(r'</(\w+):(\w+)', r'</\2', raw)
                    raw = re.sub(r'\s+\w+:\w+\s*=\s*"[^"]*"', '', raw)
                    raw = re.sub(r"\s+\w+:\w+\s*=\s*'[^']*'", '', raw)
                    root = ET.fromstring(raw)
                except Exception as e2:
                    self._notify(
                        {'ERROR'}, f"Failed to parse DAE: {e} / {e2}"
                    )
                    return {'CANCELLED'}
            except Exception as e:
                self._notify({'ERROR'}, f"Failed to parse DAE: {e}")
                return {'CANCELLED'}

        ns  = get_collada_ns(root)
        dae = self.filepath

        if getattr(self, "_target_collection", None) is not None:
            collection = self._target_collection
        elif context.view_layer.active_layer_collection:
            collection = context.view_layer.active_layer_collection.collection
        else:
            collection = context.scene.collection

        self._import_stage = "analysing scene libraries"
        profile = getattr(self, "_cached_profile", None)
        if profile is None:
            profile = analyse_dae(root, ns)
        self._cached_profile = None
        is_rigged      = profile["is_rigged"]
        is_assembly    = profile["is_assembly"]
        correction_mat = get_scene_correction_matrix(root, ns, self.import_units)

        material_texture_map = extract_material_texture_map(root, ns) if self.import_materials else {}
        camera_defs = parse_camera_library(root, ns) if self.import_cameras else {}
        light_defs = parse_light_library(root, ns) if self.import_lights else {}
        model_name           = os.path.splitext(os.path.basename(dae))[0]

        arm_obj     = None
        controllers = {}
        morph_controllers = parse_morph_controllers(root, ns)
        morph_by_controller = {
            morph_set["controller_id"]: (base_geometry, morph_set)
            for base_geometry, morph_sets in morph_controllers.items()
            for morph_set in morph_sets
        }
        if self.import_rig and is_rigged:
            self._import_stage = "building armature"
            arm_obj     = build_armature(root, ns, collection, model_name, correction_mat)
            controllers = parse_controllers(root, ns)
        elif self.import_rig and not is_rigged:
            print("[DAE] No rig found — skipping armature import.")

        geom_mat_override = build_ctrl_mat_map(root, ns, controllers)
        import_cache = {
            "skin_by_geometry": {
                controller["skin_source"]: controller
                for controller in controllers.values()
            }
        }

        # Build fast geometry lookup by id
        geom_map = {g.attrib.get("id"): g for g in root.findall(f".//{q(ns,'geometry')}")}
        if not geom_map and not camera_defs and not light_defs:
            self._notify(
                {'ERROR'},
                "No supported geometry, cameras, or lights found in DAE",
            )
            return {'CANCELLED'}

        imported = 0
        imported_meshes = 0
        imported_mesh_objects = []
        node_targets = {}
        static_mesh_templates = {}
        skinned_geometry_ids = {
            controller["skin_source"] for controller in controllers.values()
        }

        def build_static_geometry(geom_id, material_override):
            cache_key = (
                geom_id,
                tuple(sorted(material_override.items())),
                self.import_uvs,
                self.import_normals,
                self.import_vertex_colors,
                self.merge_vertices,
                round(self.merge_threshold, 9),
            )
            template = static_mesh_templates.get(cache_key)
            if template is not None and geom_id not in skinned_geometry_ids:
                obj = bpy.data.objects.new(template.name, template.data)
                collection.objects.link(obj)
                return obj, True
            obj = build_mesh_from_geometry(
                geom_map[geom_id], ns, collection, material_texture_map,
                arm_obj, controllers, material_override, dae,
                import_uvs=self.import_uvs,
                import_normals=self.import_normals,
                import_vertex_colors=self.import_vertex_colors,
                merge_vertices=self.merge_vertices,
                merge_threshold=self.merge_threshold,
                runtime_cache=import_cache,
            )
            if obj is not None and geom_id not in skinned_geometry_ids:
                static_mesh_templates[cache_key] = obj
            return obj, False

        def walk_scene(node, parent_mat, parent_obj=None, library_stack=frozenset()):
            """
            Recursively walk the visual scene, accumulating transforms.
            Uses parse_node_transform so <translate>/<rotate>/<scale> nodes
            are handled correctly, not just <matrix>.
            correction_mat is applied at the root level so the whole scene
            is rotated into Blender's Z-up space as object transforms,
            rather than baking the rotation into every vertex.
            """
            nonlocal imported, imported_meshes
            local_mat = parse_node_transform(node, ns)
            world_mat = parent_mat @ local_mat
            node_name = node.attrib.get("name") or node.attrib.get("id") or "DAE_Node"
            node_obj = parent_obj
            if self.import_hierarchy and node.attrib.get("type", "NODE") != "JOINT":
                node_obj = bpy.data.objects.new(node_name, None)
                node_obj.empty_display_type = 'PLAIN_AXES'
                collection.objects.link(node_obj)
                if parent_obj is not None:
                    node_obj.parent = parent_obj
                    node_obj.matrix_local = local_mat
                else:
                    node_obj.matrix_world = world_mat
                for key in (node.attrib.get("id"), node.attrib.get("sid"), node.attrib.get("name")):
                    if key:
                        node_targets.setdefault(key, []).append(node_obj)

            def register_node_target(obj):
                if node_obj is not None:
                    return
                for key in (node.attrib.get("id"), node.attrib.get("sid"), node.attrib.get("name")):
                    if key:
                        node_targets.setdefault(key, []).append(obj)

            # Instance geometry in this node (static meshes)
            for ig in node.findall(q(ns, "instance_geometry")):
                geom_url_val = ig.attrib.get("url", "")
                geom_id = geom_url_val[1:] if geom_url_val.startswith("#") else geom_url_val
                if geom_id in geom_map:
                    mat_override = geom_mat_override.get(geom_id, {})
                    obj, reused_data = build_static_geometry(
                        geom_id, mat_override
                    )
                    if obj:
                        if self.import_shape_keys and not reused_data:
                            apply_morph_targets(
                                obj, geom_id, morph_controllers, geom_map, ns
                            )
                        for morph_set in morph_controllers.get(geom_id, []):
                            node_targets.setdefault(
                                morph_set["controller_id"], []
                            ).append(("MORPH", obj))
                        if node_obj is not None:
                            obj.parent = node_obj
                            obj.matrix_local = Matrix.Identity(4)
                        else:
                            obj.matrix_world = world_mat
                        register_node_target(obj)
                        imported += 1
                        imported_meshes += 1
                        imported_mesh_objects.append(obj)

            # Instance controller in this node (rigged/skinned meshes)
            for ic in node.findall(q(ns, "instance_controller")):
                ctrl_url_val = ic.attrib.get("url", "")
                ctrl_id = ctrl_url_val[1:] if ctrl_url_val.startswith("#") else ctrl_url_val
                # Resolve controller -> skin_source geometry
                geom_id = controllers.get(ctrl_id, {}).get("skin_source")
                morph_entry = morph_by_controller.get(ctrl_id)
                if geom_id is None and morph_entry is not None:
                    geom_id = morph_entry[0]
                if geom_id and geom_id in geom_map:
                    mat_override = geom_mat_override.get(geom_id, {})
                    if ctrl_id in controllers:
                        obj = build_mesh_from_geometry(
                            geom_map[geom_id], ns, collection,
                            material_texture_map, arm_obj, controllers,
                            mat_override, dae,
                            import_uvs=self.import_uvs,
                            import_normals=self.import_normals,
                            import_vertex_colors=self.import_vertex_colors,
                            merge_vertices=self.merge_vertices,
                            merge_threshold=self.merge_threshold,
                            runtime_cache=import_cache,
                        )
                    else:
                        obj, _reused_data = build_static_geometry(
                            geom_id, mat_override
                        )
                    if obj:
                        if self.import_shape_keys:
                            apply_morph_targets(
                                obj, geom_id, morph_controllers, geom_map, ns
                            )
                        for morph_set in morph_controllers.get(geom_id, []):
                            node_targets.setdefault(
                                morph_set["controller_id"], []
                            ).append(("MORPH", obj))
                        obj.matrix_world = world_mat
                        register_node_target(obj)
                        imported += 1
                        imported_meshes += 1
                        imported_mesh_objects.append(obj)

            for instance_camera in node.findall(q(ns, "instance_camera")):
                camera_id = strip_url(instance_camera.attrib.get("url"))
                obj = create_camera_object(camera_id, camera_defs, collection, node_name)
                if obj:
                    if node_obj is not None:
                        obj.parent = node_obj
                        obj.matrix_local = Matrix.Identity(4)
                    else:
                        obj.matrix_world = world_mat
                    register_node_target(obj)
                    imported += 1

            for instance_light in node.findall(q(ns, "instance_light")):
                light_id = strip_url(instance_light.attrib.get("url"))
                obj = create_light_object(light_id, light_defs, collection, node_name)
                if obj:
                    if node_obj is not None:
                        obj.parent = node_obj
                        obj.matrix_local = Matrix.Identity(4)
                    else:
                        obj.matrix_world = world_mat
                    register_node_target(obj)
                    imported += 1

            # Instance node (library_nodes assembly)
            for inn in node.findall(q(ns, "instance_node")):
                nid_val = inn.attrib.get("url", "")
                nid = nid_val.lstrip("#")
                if nid in library_stack:
                    continue   # prevent infinite recursion
                lib = root.find(q(ns, "library_nodes"))
                if lib is not None:
                    tgt = lib.find(f".//{q(ns,'node')}[@id='{nid}']")
                    if tgt is not None:
                        walk_scene(tgt, world_mat, node_obj, library_stack | {nid})

            # Recurse into children
            for child in node.findall(q(ns, "node")):
                walk_scene(child, world_mat, node_obj, library_stack)

        vs = find_active_visual_scene(root, ns)
        self._import_stage = "building scene objects"
        if vs is not None:
            # Apply correction_mat at the root so the whole scene rotates into
            # Blender Z-up space as object-level transforms.
            for node in vs.findall(q(ns, "node")):
                walk_scene(node, correction_mat)
        else:
            print(
                "[DAE] No visual_scene found; importing geometry libraries "
                "directly."
            )

        animation_keys = 0
        if (
            self.import_animation
            and profile["has_anims"]
            and vs is not None
        ):
            self._import_stage = "importing animation"
            animation_keys = import_collada_animations(
                root, ns, node_targets, arm_obj, context.scene, correction_mat
            )

        # Fallback: import any geometry that walk_scene never reached.
        # This handles DAEs where geometries exist in library_geometries
        # but are not referenced from the visual_scene (common in simple
        # game exports and some SL/Firestorm files).
        if imported_meshes == 0 and geom_map:
            print(
                "[DAE] No referenced meshes were created; falling back to "
                "direct geometry import."
            )
            self._import_stage = "importing unreferenced geometry"
            morph_target_ids = {
                target_id
                for morph_sets in morph_controllers.values()
                for morph_set in morph_sets
                for target_id in morph_set["targets"]
            }
            for geom_id, geom in geom_map.items():
                if geom_id in morph_target_ids:
                    continue
                mat_override = geom_mat_override.get(geom_id, {})
                obj = build_mesh_from_geometry(
                    geom, ns, collection, material_texture_map,
                    arm_obj, controllers, mat_override, dae,
                    import_uvs=self.import_uvs,
                    import_normals=self.import_normals,
                    import_vertex_colors=self.import_vertex_colors,
                    merge_vertices=self.merge_vertices,
                    merge_threshold=self.merge_threshold,
                    runtime_cache=import_cache,
                )
                if obj:
                    if self.import_shape_keys:
                        apply_morph_targets(
                            obj, geom_id, morph_controllers, geom_map, ns
                        )
                    for morph_set in morph_controllers.get(geom_id, []):
                        node_targets.setdefault(
                            morph_set["controller_id"], []
                        ).append(("MORPH", obj))
                    obj.matrix_world = correction_mat
                    imported += 1
                    imported_meshes += 1
                    imported_mesh_objects.append(obj)

        if imported == 0:
            self._notify(
                {'ERROR'},
                (
                    "No objects created. "
                    f"Found {len(geom_map)} geometries, "
                    f"{len(controllers)} skin controllers, "
                    f"{len(camera_defs)} cameras and "
                    f"{len(light_defs)} lights."
                ),
            )
            return {'CANCELLED'}

        rig_msg = f" + armature ({arm_obj.name})" if arm_obj else ""
        animation_msg = f" + {animation_keys} animation keys" if animation_keys else ""
        if imported_mesh_objects:
            for scene_obj in context.scene.objects:
                scene_obj.select_set(False)
            for obj in imported_mesh_objects:
                obj.select_set(True)
            context.view_layer.objects.active = imported_mesh_objects[0]
        elif arm_obj is not None:
            for scene_obj in context.scene.objects:
                scene_obj.select_set(False)
            arm_obj.select_set(True)
            context.view_layer.objects.active = arm_obj
        self._notify(
            {'INFO'},
            f"Imported {imported} object(s){rig_msg}{animation_msg}.",
        )
        return {'FINISHED'}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def _prescan(self):
        if not self.filepath or not os.path.isfile(self.filepath):
            return
        try:
            import re as _re
            try:
                _tree = ET.parse(self.filepath)
                _root = _tree.getroot()
                self._cached_root = _root   # cache so execute() doesn't re-parse
            except ET.ParseError:
                with open(self.filepath, 'r', encoding='utf-8', errors='replace') as f:
                    _raw = f.read()
                _raw = _re.sub(r'<(\w+:\w+)[^>]*>.*?</\1>', '', _raw, flags=_re.DOTALL)
                _raw = _re.sub(r'<(\w+):(\w+)', r'<\2', _raw)
                _raw = _re.sub(r'</(\w+):(\w+)', r'</\2', _raw)
                _root = ET.fromstring(_raw)
            _ns = get_collada_ns(_root)
            _profile = analyse_dae(_root, _ns)
            self._cached_profile = _profile
            self.import_rig = _profile["is_rigged"]
            _img_lib = _root.find(f"{_ns}library_images" if _ns else "library_images")
            self.import_materials = _img_lib is not None and len(list(_img_lib)) > 0
            self._profile_summary = (
                f"Joints: {_profile['joint_count']}  "
                f"Controllers: {_profile['controller_count']}  "
                f"Up: {_profile['up_axis']}  "
                f"Unit: {_profile['unit_meter']}m  "
                f"Assembly: {_profile['is_assembly']}  "
                f"Anims: {_profile['has_anims']}"
            )
        except Exception as e:
            print(f"[DAE pre-scan failed: {e}]")

    def draw(self, context):
        layout = self.layout
        if hasattr(self, '_profile_summary'):
            box = layout.box()
            box.label(text="Detected:", icon='INFO')
            parts = self._profile_summary.split("  ")
            box.label(text="  ".join(parts[:3]))
            box.label(text="  ".join(parts[3:]))
            layout.separator()
        layout.label(text="Batch Import")
        layout.prop(self, "batch_create_collections")
        layout.separator()
        layout.label(text="Mesh")
        layout.prop(self, "import_uvs")
        layout.prop(self, "import_normals")
        layout.prop(self, "import_vertex_colors")
        layout.prop(self, "merge_vertices")
        if self.merge_vertices:
            layout.prop(self, "merge_threshold")
        layout.separator()
        layout.label(text="Scene")
        layout.prop(self, "import_units")
        layout.prop(self, "import_hierarchy")
        layout.prop(self, "import_cameras")
        layout.prop(self, "import_lights")
        layout.prop(self, "import_animation")
        layout.prop(self, "import_shape_keys")
        layout.separator()
        layout.label(text="Materials")
        layout.prop(self, "import_materials")
        layout.separator()
        layout.label(text="Rig")
        layout.prop(self, "import_rig")


# ── TEXTURE ASSIGN OPERATOR ──────────────────────────────────────────────────

def dae_safe_id(value, fallback="item"):
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or fallback))
    return value if value and re.match(r"[A-Za-z_]", value[0]) else f"id_{value}"


def collada_matrix_text(matrix):
    return " ".join(f"{value:.9g}" for row in matrix.transposed() for value in row)


def add_float_source(parent, source_id, values, params):
    source = ET.SubElement(parent, "source", id=source_id)
    flat = [component for value in values for component in value]
    array_id = f"{source_id}-array"
    array = ET.SubElement(source, "float_array", id=array_id, count=str(len(flat)))
    array.text = " ".join(f"{value:.9g}" for value in flat)
    technique = ET.SubElement(source, "technique_common")
    accessor = ET.SubElement(
        technique, "accessor", source=f"#{array_id}",
        count=str(len(values)), stride=str(len(params)),
    )
    for param in params:
        ET.SubElement(accessor, "param", name=param, type="float")


def add_matrix_source(parent, source_id, matrices):
    source = ET.SubElement(parent, "source", id=source_id)
    flat = [component for matrix in matrices for component in matrix]
    array_id = f"{source_id}-array"
    array = ET.SubElement(source, "float_array", id=array_id, count=str(len(flat)))
    array.text = " ".join(f"{value:.9g}" for value in flat)
    technique = ET.SubElement(source, "technique_common")
    accessor = ET.SubElement(
        technique, "accessor", source=f"#{array_id}",
        count=str(len(matrices)), stride="16",
    )
    ET.SubElement(accessor, "param", name="TRANSFORM", type="float4x4")


def material_image_for_socket(material, socket_names):
    if not material or not material.use_nodes or not material.node_tree:
        return None
    for node in material.node_tree.nodes:
        if node.type != 'BSDF_PRINCIPLED':
            continue
        for socket_name in socket_names:
            socket = node.inputs.get(socket_name)
            if not socket:
                continue
            for link in socket.links:
                source = link.from_node
                if source.type == 'TEX_IMAGE' and source.image:
                    return source.image
                if source.type == 'NORMAL_MAP':
                    color = source.inputs.get("Color")
                    if color:
                        for color_link in color.links:
                            tex = color_link.from_node
                            if tex.type == 'TEX_IMAGE' and tex.image:
                                return tex.image
    return None


def export_collada_scene(context, filepath, selected_only=False, apply_modifiers=True,
                         export_uvs=True, export_normals=True,
                         export_materials=True, export_cameras=True,
                         export_lights=True, export_animation=True,
                         export_rig=True, export_shape_keys=True,
                         deform_bones_only=False, preserve_hierarchy=True,
                         preserve_instances=True,
                         texture_mode="RELATIVE", global_scale=1.0):
    scene = context.scene
    depsgraph = context.evaluated_depsgraph_get()
    source_objects = list(context.selected_objects) if selected_only else list(scene.objects)
    objects = [
        obj for obj in source_objects
        if obj.type in {'MESH', 'EMPTY'}
        or (obj.type == 'ARMATURE' and export_rig)
        or (obj.type == 'CAMERA' and export_cameras)
        or (obj.type == 'LIGHT' and export_lights)
    ]
    if export_rig:
        for obj in list(objects):
            if obj.type != 'MESH':
                continue
            armature = next(
                (
                    modifier.object for modifier in obj.modifiers
                    if modifier.type == 'ARMATURE' and modifier.object
                ),
                obj.parent if obj.parent and obj.parent.type == 'ARMATURE' else None,
            )
            if armature is not None and armature not in objects:
                objects.append(armature)

    ET.register_namespace("", "http://www.collada.org/2005/11/COLLADASchema")
    root = ET.Element(
        "COLLADA", xmlns="http://www.collada.org/2005/11/COLLADASchema",
        version="1.4.1",
    )
    asset = ET.SubElement(root, "asset")
    contributor = ET.SubElement(asset, "contributor")
    ET.SubElement(contributor, "authoring_tool").text = (
        f"Blender {bpy.app.version_string} COLLADA Python Add-on"
    )
    now = datetime.datetime.now(datetime.timezone.utc).replace(
        microsecond=0
    ).isoformat().replace("+00:00", "Z")
    ET.SubElement(asset, "created").text = now
    ET.SubElement(asset, "modified").text = now
    ET.SubElement(asset, "unit", name="meter", meter="1")
    ET.SubElement(asset, "up_axis").text = "Z_UP"

    libraries = {
        name: ET.SubElement(root, name) for name in (
            "library_images", "library_effects", "library_materials",
            "library_geometries", "library_cameras", "library_lights",
            "library_controllers", "library_animations",
            "library_visual_scenes",
        )
    }
    visual_scene = ET.SubElement(
        libraries["library_visual_scenes"], "visual_scene",
        id="Scene", name=scene.name,
    )
    used_ids = set()

    def unique_id(value):
        base = dae_safe_id(value)
        candidate = base
        index = 1
        while candidate in used_ids:
            index += 1
            candidate = f"{base}_{index}"
        used_ids.add(candidate)
        return candidate

    object_ids = {obj: unique_id(obj.name) for obj in objects}
    material_ids = {}
    image_ids = {}

    def texture_path(image):
        absolute = bpy.path.abspath(image.filepath)
        if texture_mode == "FILENAME":
            return os.path.basename(absolute)
        if texture_mode == "RELATIVE":
            try:
                return os.path.relpath(
                    absolute, os.path.dirname(filepath)
                ).replace("\\", "/")
            except ValueError:
                pass
        return urllib.parse.quote(absolute.replace("\\", "/"))

    def ensure_image(image):
        if image in image_ids:
            return image_ids[image]
        image_id = unique_id(f"{image.name}-image")
        image_ids[image] = image_id
        element = ET.SubElement(
            libraries["library_images"], "image",
            id=image_id, name=image.name,
        )
        ET.SubElement(element, "init_from").text = texture_path(image)
        return image_id

    def ensure_material(material):
        if material in material_ids:
            return material_ids[material]
        material_id = unique_id(f"{material.name}-material")
        effect_id = unique_id(f"{material.name}-effect")
        material_ids[material] = material_id
        effect = ET.SubElement(
            libraries["library_effects"], "effect",
            id=effect_id, name=material.name,
        )
        profile = ET.SubElement(effect, "profile_COMMON")
        diffuse_image = material_image_for_socket(material, ("Base Color",))
        sampler_sid = None
        if diffuse_image:
            image_id = ensure_image(diffuse_image)
            surface_sid = f"{material_id}-surface"
            sampler_sid = f"{material_id}-sampler"
            surface_param = ET.SubElement(profile, "newparam", sid=surface_sid)
            surface = ET.SubElement(surface_param, "surface", type="2D")
            ET.SubElement(surface, "init_from").text = image_id
            sampler_param = ET.SubElement(profile, "newparam", sid=sampler_sid)
            sampler = ET.SubElement(sampler_param, "sampler2D")
            ET.SubElement(sampler, "source").text = surface_sid
        technique = ET.SubElement(profile, "technique", sid="common")
        phong = ET.SubElement(technique, "phong")
        ET.SubElement(ET.SubElement(phong, "emission"), "color").text = "0 0 0 1"
        diffuse = ET.SubElement(phong, "diffuse")
        if sampler_sid:
            ET.SubElement(
                diffuse, "texture", texture=sampler_sid, texcoord="UVSET0"
            )
        else:
            ET.SubElement(diffuse, "color").text = " ".join(
                f"{value:.9g}" for value in material.diffuse_color
            )
        ET.SubElement(ET.SubElement(phong, "specular"), "color").text = "0.2 0.2 0.2 1"
        ET.SubElement(ET.SubElement(phong, "shininess"), "float").text = "32"
        ET.SubElement(
            ET.SubElement(phong, "index_of_refraction"), "float"
        ).text = "1.45"
        ET.SubElement(ET.SubElement(phong, "transparency"), "float").text = (
            f"{material.diffuse_color[3]:.9g}"
        )
        material_element = ET.SubElement(
            libraries["library_materials"], "material",
            id=material_id, name=material.name,
        )
        ET.SubElement(material_element, "instance_effect", url=f"#{effect_id}")
        return material_id

    geometry_ids = {}
    exported_meshes = {}
    morph_controller_ids = {}
    morph_target_ids = {}
    mesh_armatures = {}
    skin_controller_ids = {}
    armature_joint_ids = {}
    geometry_cache = {}
    morph_cache = {}
    cleanup = []

    def find_mesh_armature(obj):
        for modifier in obj.modifiers:
            if modifier.type == 'ARMATURE' and modifier.object:
                return modifier.object
        if obj.parent and obj.parent.type == 'ARMATURE':
            return obj.parent
        return None

    for obj in objects:
        if obj.type != 'MESH':
            continue
        armature = find_mesh_armature(obj) if export_rig else None
        if armature is not None:
            mesh_armatures[obj] = armature
        has_shapes = bool(
            export_shape_keys and obj.data.shape_keys and
            len(obj.data.shape_keys.key_blocks) > 1
        )
        use_original = bool(armature is not None or has_shapes)
        evaluated = (
            obj.evaluated_get(depsgraph)
            if apply_modifiers and not use_original else obj
        )
        mesh = (
            evaluated.to_mesh()
            if apply_modifiers and not use_original else obj.data
        )
        if apply_modifiers and not use_original:
            cleanup.append(evaluated)
        mesh.calc_loop_triangles()
        exported_meshes[obj] = mesh

        cache_key = (
            obj.data if preserve_instances and (not apply_modifiers or use_original)
            else None
        )
        if cache_key is not None and cache_key in geometry_cache:
            geometry_ids[obj] = geometry_cache[cache_key]
            if cache_key in morph_cache:
                morph_controller_ids[obj] = morph_cache[cache_key]
            continue

        geometry_id = unique_id(f"{obj.name}-mesh")
        geometry_ids[obj] = geometry_id
        if cache_key is not None:
            geometry_cache[cache_key] = geometry_id
        geometry = ET.SubElement(
            libraries["library_geometries"], "geometry",
            id=geometry_id, name=obj.name,
        )
        mesh_element = ET.SubElement(geometry, "mesh")
        position_id = f"{geometry_id}-positions"
        add_float_source(
            mesh_element, position_id,
            [tuple(vertex.co) for vertex in mesh.vertices],
            ("X", "Y", "Z"),
        )
        normal_id = None
        if export_normals:
            normal_id = f"{geometry_id}-normals"
            add_float_source(
                mesh_element, normal_id,
                [tuple(loop.normal) for loop in mesh.loops],
                ("X", "Y", "Z"),
            )
        uv_ids = []
        if export_uvs:
            for layer_index, layer in enumerate(mesh.uv_layers):
                uv_id = f"{geometry_id}-uv-{layer_index}"
                add_float_source(
                    mesh_element, uv_id,
                    [tuple(item.uv) for item in layer.data],
                    ("S", "T"),
                )
                uv_ids.append(uv_id)
        vertices_id = f"{geometry_id}-vertices"
        vertices = ET.SubElement(mesh_element, "vertices", id=vertices_id)
        ET.SubElement(
            vertices, "input", semantic="POSITION", source=f"#{position_id}"
        )
        groups = {}
        for triangle in mesh.loop_triangles:
            groups.setdefault(triangle.material_index, []).append(triangle)
        for material_index, triangles in groups.items():
            attributes = {"count": str(len(triangles))}
            material = (
                mesh.materials[material_index]
                if 0 <= material_index < len(mesh.materials) else None
            )
            if export_materials and material:
                attributes["material"] = ensure_material(material)
            triangle_element = ET.SubElement(
                mesh_element, "triangles", attributes
            )
            ET.SubElement(
                triangle_element, "input", semantic="VERTEX",
                source=f"#{vertices_id}", offset="0",
            )
            offset = 1
            if normal_id:
                ET.SubElement(
                    triangle_element, "input", semantic="NORMAL",
                    source=f"#{normal_id}", offset=str(offset),
                )
                offset += 1
            for layer_index, uv_id in enumerate(uv_ids):
                ET.SubElement(
                    triangle_element, "input", semantic="TEXCOORD",
                    source=f"#{uv_id}", offset=str(offset),
                    set=str(layer_index),
                )
                offset += 1
            indices = []
            for triangle in triangles:
                for corner, vertex_index in enumerate(triangle.vertices):
                    loop_index = triangle.loops[corner]
                    indices.append(str(vertex_index))
                    if normal_id:
                        indices.append(str(loop_index))
                    indices.extend(str(loop_index) for _ in uv_ids)
            ET.SubElement(triangle_element, "p").text = " ".join(indices)

        if has_shapes:
            target_ids = []
            shape_keys = list(obj.data.shape_keys.key_blocks)[1:]
            for shape_key in shape_keys:
                target_id = unique_id(
                    f"{obj.name}-morph-{shape_key.name}-mesh"
                )
                target_ids.append(target_id)
                target_geometry = copy.deepcopy(geometry)
                target_geometry.attrib["id"] = target_id
                target_geometry.attrib["name"] = shape_key.name
                for element in target_geometry.iter():
                    for attribute in ("id", "source", "url"):
                        value = element.attrib.get(attribute)
                        if value:
                            element.attrib[attribute] = value.replace(
                                geometry_id, target_id
                            )
                target_position_source = target_geometry.find(
                    f"./mesh/source[@id='{target_id}-positions']"
                )
                if target_position_source is not None:
                    target_array = target_position_source.find("float_array")
                    if target_array is not None:
                        target_array.text = " ".join(
                            f"{component:.9g}"
                            for point in shape_key.data
                            for component in point.co
                        )
                libraries["library_geometries"].append(target_geometry)

            controller_id = unique_id(f"{obj.name}-morph")
            morph_controller_ids[obj] = controller_id
            morph_target_ids[obj] = target_ids
            if cache_key is not None:
                morph_cache[cache_key] = controller_id
            controller = ET.SubElement(
                libraries["library_controllers"], "controller",
                id=controller_id, name=f"{obj.name}-morph",
            )
            morph = ET.SubElement(
                controller, "morph", source=f"#{geometry_id}",
                method="NORMALIZED",
            )
            targets_source_id = f"{controller_id}-targets"
            targets_source = ET.SubElement(
                morph, "source", id=targets_source_id
            )
            target_array_id = f"{targets_source_id}-array"
            target_array = ET.SubElement(
                targets_source, "IDREF_array",
                id=target_array_id, count=str(len(target_ids)),
            )
            target_array.text = " ".join(target_ids)
            target_technique = ET.SubElement(
                targets_source, "technique_common"
            )
            target_accessor = ET.SubElement(
                target_technique, "accessor",
                source=f"#{target_array_id}", count=str(len(target_ids)),
                stride="1",
            )
            ET.SubElement(
                target_accessor, "param", name="IDREF", type="IDREF"
            )
            weights_source_id = f"{controller_id}-weights"
            add_float_source(
                morph, weights_source_id,
                [(shape_key.value,) for shape_key in shape_keys],
                ("MORPH_WEIGHT",),
            )
            targets = ET.SubElement(morph, "targets")
            ET.SubElement(
                targets, "input", semantic="MORPH_TARGET",
                source=f"#{targets_source_id}",
            )
            ET.SubElement(
                targets, "input", semantic="MORPH_WEIGHT",
                source=f"#{weights_source_id}",
            )

    for armature in set(mesh_armatures.values()):
        bones = [
            bone for bone in armature.data.bones
            if not deform_bones_only or bone.use_deform
        ]
        armature_joint_ids[armature] = {
            bone.name: unique_id(f"{object_ids[armature]}_{bone.name}")
            for bone in bones
        }

    for obj, armature in mesh_armatures.items():
        joint_id_map = armature_joint_ids.get(armature, {})
        bones = [
            bone for bone in armature.data.bones
            if bone.name in joint_id_map
        ]
        if not bones:
            continue
        controller_id = unique_id(f"{armature.name}_{obj.name}-skin")
        skin_controller_ids[obj] = controller_id
        controller = ET.SubElement(
            libraries["library_controllers"], "controller",
            id=controller_id, name=armature.name,
        )
        skin = ET.SubElement(
            controller, "skin", source=f"#{geometry_ids[obj]}"
        )
        relative_bind = armature.matrix_world.inverted_safe() @ obj.matrix_world
        ET.SubElement(skin, "bind_shape_matrix").text = (
            collada_matrix_text(relative_bind)
        )

        joints_source_id = f"{controller_id}-joints"
        joints_source = ET.SubElement(skin, "source", id=joints_source_id)
        joint_array_id = f"{joints_source_id}-array"
        joint_array = ET.SubElement(
            joints_source, "Name_array",
            id=joint_array_id, count=str(len(bones)),
        )
        joint_array.text = " ".join(bone.name for bone in bones)
        joint_technique = ET.SubElement(
            joints_source, "technique_common"
        )
        joint_accessor = ET.SubElement(
            joint_technique, "accessor",
            source=f"#{joint_array_id}", count=str(len(bones)),
            stride="1",
        )
        ET.SubElement(
            joint_accessor, "param", name="JOINT", type="Name"
        )

        bind_source_id = f"{controller_id}-bind-poses"
        inverse_bind_matrices = []
        for bone in bones:
            inverse_bind = bone.matrix_local.inverted_safe()
            inverse_bind_matrices.append(tuple(
                value for row in inverse_bind.transposed() for value in row
            ))
        add_matrix_source(skin, bind_source_id, inverse_bind_matrices)

        bone_index = {bone.name: index for index, bone in enumerate(bones)}
        group_to_bone = {
            group.index: bone_index[group.name]
            for group in obj.vertex_groups
            if group.name in bone_index
        }
        weight_values = []
        vcounts = []
        vdata = []
        for vertex in obj.data.vertices:
            influences = []
            for membership in vertex.groups:
                joint_index = group_to_bone.get(membership.group)
                if joint_index is not None and membership.weight > 0.0:
                    influences.append((joint_index, membership.weight))
            total = sum(weight for _, weight in influences)
            if total > 0.0:
                influences = [
                    (joint_index, weight / total)
                    for joint_index, weight in influences
                ]
            vcounts.append(len(influences))
            for joint_index, weight in influences:
                weight_index = len(weight_values)
                weight_values.append(weight)
                vdata.extend((joint_index, weight_index))
        weights_source_id = f"{controller_id}-weights"
        add_float_source(
            skin, weights_source_id,
            [(weight,) for weight in weight_values],
            ("WEIGHT",),
        )
        joints_element = ET.SubElement(skin, "joints")
        ET.SubElement(
            joints_element, "input", semantic="JOINT",
            source=f"#{joints_source_id}",
        )
        ET.SubElement(
            joints_element, "input", semantic="INV_BIND_MATRIX",
            source=f"#{bind_source_id}",
        )
        vertex_weights = ET.SubElement(
            skin, "vertex_weights", count=str(len(obj.data.vertices))
        )
        ET.SubElement(
            vertex_weights, "input", semantic="JOINT",
            source=f"#{joints_source_id}", offset="0",
        )
        ET.SubElement(
            vertex_weights, "input", semantic="WEIGHT",
            source=f"#{weights_source_id}", offset="1",
        )
        ET.SubElement(vertex_weights, "vcount").text = " ".join(
            str(count) for count in vcounts
        )
        ET.SubElement(vertex_weights, "v").text = " ".join(
            str(value) for value in vdata
        )

    camera_ids = {}
    light_ids = {}
    for obj in objects:
        if obj.type == 'CAMERA':
            camera_id = unique_id(f"{obj.name}-camera")
            camera_ids[obj] = camera_id
            camera = ET.SubElement(
                libraries["library_cameras"], "camera",
                id=camera_id, name=obj.name,
            )
            common = ET.SubElement(ET.SubElement(camera, "optics"), "technique_common")
            if obj.data.type == 'ORTHO':
                projection = ET.SubElement(common, "orthographic")
                ET.SubElement(projection, "ymag").text = f"{obj.data.ortho_scale * 0.5:.9g}"
            else:
                projection = ET.SubElement(common, "perspective")
                ET.SubElement(projection, "yfov").text = f"{math.degrees(obj.data.angle_y):.9g}"
            ET.SubElement(projection, "znear").text = f"{obj.data.clip_start:.9g}"
            ET.SubElement(projection, "zfar").text = f"{obj.data.clip_end:.9g}"
        elif obj.type == 'LIGHT':
            light_id = unique_id(f"{obj.name}-light")
            light_ids[obj] = light_id
            light = ET.SubElement(
                libraries["library_lights"], "light",
                id=light_id, name=obj.name,
            )
            common = ET.SubElement(light, "technique_common")
            tag = {'SUN': 'directional', 'SPOT': 'spot', 'AREA': 'ambient'}.get(
                obj.data.type, 'point'
            )
            light_type = ET.SubElement(common, tag)
            ET.SubElement(light_type, "color").text = " ".join(
                f"{value:.9g}" for value in obj.data.color
            )
            if obj.data.type == 'SPOT':
                ET.SubElement(light_type, "falloff_angle").text = (
                    f"{math.degrees(obj.data.spot_size) * 0.5:.9g}"
                )

    scale_matrix = Matrix.Scale(global_scale, 4)
    object_set = set(objects)

    def add_material_bind(instance, mesh):
        if not export_materials or not mesh.materials:
            return
        common = ET.SubElement(
            ET.SubElement(instance, "bind_material"),
            "technique_common",
        )
        for material in mesh.materials:
            if not material:
                continue
            material_id = ensure_material(material)
            instance_material = ET.SubElement(
                common, "instance_material",
                symbol=material_id, target=f"#{material_id}",
            )
            for layer_index, layer in enumerate(mesh.uv_layers):
                ET.SubElement(
                    instance_material, "bind_vertex_input",
                    semantic=layer.name or f"UVSET{layer_index}",
                    input_semantic="TEXCOORD",
                    input_set=str(layer_index),
                )

    def add_armature_joints(armature, armature_node):
        joint_ids = armature_joint_ids.get(armature, {})
        if not joint_ids:
            return []
        exported_bones = {
            bone.name: bone for bone in armature.data.bones
            if bone.name in joint_ids
        }
        root_joint_ids = []

        def add_bone(bone, parent_element):
            joint = ET.SubElement(
                parent_element, "node",
                id=joint_ids[bone.name], sid=bone.name,
                name=bone.name, type="JOINT",
            )
            if bone.parent and bone.parent.name in exported_bones:
                local_matrix = (
                    bone.parent.matrix_local.inverted_safe() @
                    bone.matrix_local
                )
            else:
                local_matrix = bone.matrix_local
                root_joint_ids.append(joint_ids[bone.name])
            ET.SubElement(
                joint, "matrix", sid="transform"
            ).text = collada_matrix_text(local_matrix)
            for child in bone.children:
                if child.name in exported_bones:
                    add_bone(child, joint)

        for bone in exported_bones.values():
            if not bone.parent or bone.parent.name not in exported_bones:
                add_bone(bone, armature_node)
        return root_joint_ids

    children = {obj: [] for obj in objects}
    roots = []
    for obj in objects:
        if preserve_hierarchy and obj.parent in object_set:
            children[obj.parent].append(obj)
        else:
            roots.append(obj)

    def add_object_node(obj, parent_element, is_root=False):
        node = ET.SubElement(
            parent_element, "node",
            id=object_ids[obj], name=obj.name, type="NODE",
        )
        transform = (
            obj.matrix_local
            if preserve_hierarchy and obj.parent in object_set
            else obj.matrix_world
        )
        if is_root:
            transform = scale_matrix @ transform
        ET.SubElement(
            node, "matrix", sid="transform"
        ).text = collada_matrix_text(transform)

        root_joint_ids = []
        if obj.type == 'ARMATURE':
            root_joint_ids = add_armature_joints(obj, node)
        elif obj in geometry_ids:
            if obj in skin_controller_ids:
                instance = ET.SubElement(
                    node, "instance_controller",
                    url=f"#{skin_controller_ids[obj]}",
                )
                armature = mesh_armatures[obj]
                joint_roots = [
                    joint_id for bone_name, joint_id
                    in armature_joint_ids.get(armature, {}).items()
                    if (
                        armature.data.bones[bone_name].parent is None or
                        armature.data.bones[bone_name].parent.name not in
                        armature_joint_ids.get(armature, {})
                    )
                ]
                for joint_id in joint_roots:
                    ET.SubElement(instance, "skeleton").text = f"#{joint_id}"
            elif obj in morph_controller_ids:
                instance = ET.SubElement(
                    node, "instance_controller",
                    url=f"#{morph_controller_ids[obj]}",
                )
            else:
                instance = ET.SubElement(
                    node, "instance_geometry", url=f"#{geometry_ids[obj]}"
                )
            add_material_bind(instance, exported_meshes[obj])
        elif obj in camera_ids:
            ET.SubElement(node, "instance_camera", url=f"#{camera_ids[obj]}")
        elif obj in light_ids:
            ET.SubElement(node, "instance_light", url=f"#{light_ids[obj]}")

        for child in children.get(obj, []):
            add_object_node(child, node, False)
        return node

    for root_object in roots:
        add_object_node(root_object, visual_scene, True)

    if export_animation:
        current_frame = scene.frame_current
        fps = scene.render.fps / max(scene.render.fps_base, 1.0e-8)
        for obj in objects:
            if not obj.animation_data or not obj.animation_data.action:
                continue
            animation_id = unique_id(f"{obj.name}-animation")
            animation = ET.SubElement(
                libraries["library_animations"], "animation",
                id=animation_id,
            )
            times, matrices = [], []
            for frame in range(scene.frame_start, scene.frame_end + 1):
                scene.frame_set(frame)
                times.append((frame / fps,))
                matrix = scale_matrix @ obj.matrix_world
                matrices.append(tuple(
                    value for row in matrix.transposed() for value in row
                ))
            input_id = f"{animation_id}-input"
            output_id = f"{animation_id}-output"
            add_float_source(animation, input_id, times, ("TIME",))
            add_matrix_source(animation, output_id, matrices)
            sampler_id = f"{animation_id}-sampler"
            sampler = ET.SubElement(animation, "sampler", id=sampler_id)
            ET.SubElement(
                sampler, "input", semantic="INPUT", source=f"#{input_id}"
            )
            ET.SubElement(
                sampler, "input", semantic="OUTPUT", source=f"#{output_id}"
            )
            ET.SubElement(
                animation, "channel", source=f"#{sampler_id}",
                target=f"{object_ids[obj]}/transform",
            )

        for armature, joint_ids in armature_joint_ids.items():
            if not armature.animation_data or not armature.animation_data.action:
                continue
            for bone_name, joint_id in joint_ids.items():
                pose_bone = armature.pose.bones.get(bone_name)
                if pose_bone is None:
                    continue
                animation_id = unique_id(
                    f"{armature.name}_{bone_name}-pose-animation"
                )
                animation = ET.SubElement(
                    libraries["library_animations"], "animation",
                    id=animation_id, name=armature.name,
                )
                times, matrices = [], []
                for frame in range(scene.frame_start, scene.frame_end + 1):
                    scene.frame_set(frame)
                    times.append((frame / fps,))
                    if (
                        pose_bone.parent and
                        pose_bone.parent.name in joint_ids
                    ):
                        matrix = (
                            pose_bone.parent.matrix.inverted_safe() @
                            pose_bone.matrix
                        )
                    else:
                        matrix = pose_bone.matrix
                    matrices.append(tuple(
                        value for row in matrix.transposed()
                        for value in row
                    ))
                input_id = f"{animation_id}-input"
                output_id = f"{animation_id}-output"
                add_float_source(animation, input_id, times, ("TIME",))
                add_matrix_source(animation, output_id, matrices)
                sampler_id = f"{animation_id}-sampler"
                sampler = ET.SubElement(
                    animation, "sampler", id=sampler_id
                )
                ET.SubElement(
                    sampler, "input", semantic="INPUT",
                    source=f"#{input_id}",
                )
                ET.SubElement(
                    sampler, "input", semantic="OUTPUT",
                    source=f"#{output_id}",
                )
                ET.SubElement(
                    animation, "channel", source=f"#{sampler_id}",
                    target=f"{joint_id}/transform",
                )

        for obj, controller_id in morph_controller_ids.items():
            shape_keys = obj.data.shape_keys
            if (
                shape_keys is None or not shape_keys.animation_data or
                not shape_keys.animation_data.action
            ):
                continue
            for shape_index, shape_key in enumerate(
                list(shape_keys.key_blocks)[1:]
            ):
                animation_id = unique_id(
                    f"{obj.name}_{shape_key.name}-morph-animation"
                )
                animation = ET.SubElement(
                    libraries["library_animations"], "animation",
                    id=animation_id, name=obj.name,
                )
                times, values = [], []
                for frame in range(scene.frame_start, scene.frame_end + 1):
                    scene.frame_set(frame)
                    times.append((frame / fps,))
                    values.append((shape_key.value,))
                input_id = f"{animation_id}-input"
                output_id = f"{animation_id}-output"
                add_float_source(animation, input_id, times, ("TIME",))
                add_float_source(
                    animation, output_id, values, ("MORPH_WEIGHT",)
                )
                sampler_id = f"{animation_id}-sampler"
                sampler = ET.SubElement(
                    animation, "sampler", id=sampler_id
                )
                ET.SubElement(
                    sampler, "input", semantic="INPUT",
                    source=f"#{input_id}",
                )
                ET.SubElement(
                    sampler, "input", semantic="OUTPUT",
                    source=f"#{output_id}",
                )
                ET.SubElement(
                    animation, "channel", source=f"#{sampler_id}",
                    target=f"{controller_id}/weights({shape_index})",
                )
        scene.frame_set(current_frame)

    ET.SubElement(
        ET.SubElement(root, "scene"),
        "instance_visual_scene", url="#Scene",
    )
    for name, library in list(libraries.items()):
        if name != "library_visual_scenes" and len(library) == 0:
            root.remove(library)
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(
        filepath, encoding="utf-8", xml_declaration=True
    )
    for evaluated in cleanup:
        evaluated.to_mesh_clear()
    return len(objects)


class EXPORT_OT_simple_collada_full(Operator, ExportHelper):
    """Export a COLLADA (.dae) scene"""
    bl_idname = "export_scene.simple_collada_full"
    bl_label = "Export COLLADA (.dae)"
    filename_ext = ".dae"
    filter_glob: StringProperty(default="*.dae", options={'HIDDEN'})

    selected_only: BoolProperty(name="Selected Objects", default=False)
    apply_modifiers: BoolProperty(name="Apply Modifiers", default=True)
    export_uvs: BoolProperty(name="UV Maps", default=True)
    export_normals: BoolProperty(name="Normals", default=True)
    export_materials: BoolProperty(name="Materials and Textures", default=True)
    export_cameras: BoolProperty(name="Cameras", default=True)
    export_lights: BoolProperty(name="Lights", default=True)
    export_rig: BoolProperty(
        name="Armatures and Skin Weights", default=True
    )
    deform_bones_only: BoolProperty(
        name="Deform Bones Only", default=False
    )
    export_shape_keys: BoolProperty(name="Shape Keys", default=True)
    export_animation: BoolProperty(
        name="Object, Bone and Shape Animation", default=True
    )
    preserve_hierarchy: BoolProperty(name="Preserve Hierarchy", default=True)
    preserve_instances: BoolProperty(name="Preserve Mesh Instances", default=True)
    texture_mode: EnumProperty(
        name="Texture Paths",
        items=(
            ("RELATIVE", "Relative", "Paths relative to the DAE"),
            ("ABSOLUTE", "Absolute", "Absolute texture paths"),
            ("FILENAME", "Filename Only", "Only write image filenames"),
        ),
        default="RELATIVE",
    )
    global_scale: FloatProperty(
        name="Global Scale", default=1.0,
        min=0.000001, max=1000000.0,
    )

    def execute(self, context):
        try:
            count = export_collada_scene(
                context, self.filepath,
                selected_only=self.selected_only,
                apply_modifiers=self.apply_modifiers,
                export_uvs=self.export_uvs,
                export_normals=self.export_normals,
                export_materials=self.export_materials,
                export_cameras=self.export_cameras,
                export_lights=self.export_lights,
                export_animation=self.export_animation,
                export_rig=self.export_rig,
                export_shape_keys=self.export_shape_keys,
                deform_bones_only=self.deform_bones_only,
                preserve_hierarchy=self.preserve_hierarchy,
                preserve_instances=self.preserve_instances,
                texture_mode=self.texture_mode,
                global_scale=self.global_scale,
            )
        except Exception as error:
            import traceback
            traceback.print_exc()
            self.report({'ERROR'}, f"COLLADA export failed: {error}")
            return {'CANCELLED'}
        self.report({'INFO'}, f"Exported {count} object(s).")
        return {'FINISHED'}

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "selected_only")
        layout.prop(self, "apply_modifiers")
        layout.prop(self, "global_scale")
        layout.separator()
        layout.prop(self, "export_uvs")
        layout.prop(self, "export_normals")
        layout.prop(self, "preserve_hierarchy")
        layout.prop(self, "preserve_instances")
        layout.prop(self, "export_materials")
        if self.export_materials:
            layout.prop(self, "texture_mode")
        layout.prop(self, "export_cameras")
        layout.prop(self, "export_lights")
        layout.prop(self, "export_rig")
        if self.export_rig:
            layout.prop(self, "deform_bones_only")
        layout.prop(self, "export_shape_keys")
        layout.prop(self, "export_animation")


class OBJECT_OT_assign_textures_by_name(Operator):
    """Assign textures by matching material names to image filenames"""
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
                try:
                    img = bpy.data.images.load(os.path.join(folder, f), check_existing=True)
                    images[name] = img
                except Exception:
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
                out_n  = nodes.new("ShaderNodeOutputMaterial"); out_n.location  = (600, 0)
                bsdf_n = nodes.new("ShaderNodeBsdfPrincipled"); bsdf_n.location = (200, 0)
                tex_n  = nodes.new("ShaderNodeTexImage");       tex_n.location  = (-300, 0)
                tex_n.image = img
                links.new(tex_n.outputs["Color"], bsdf_n.inputs["Base Color"])
                links.new(bsdf_n.outputs["BSDF"],  out_n.inputs["Surface"])
                assigned += 1

        self.report({'INFO'}, f"Assigned textures to {assigned} materials.")
        return {'FINISHED'}


# ── MENUS & REGISTER ─────────────────────────────────────────────────────────

def menu_func_import(self, context):
    self.layout.operator(IMPORT_OT_simple_collada_full.bl_idname, text="COLLADA (.dae)")

def menu_func_export(self, context):
    self.layout.operator(EXPORT_OT_simple_collada_full.bl_idname, text="COLLADA (.dae)")

def menu_func_assign_textures(self, context):
    self.layout.operator(OBJECT_OT_assign_textures_by_name.bl_idname, text="Assign Textures by Name")

def register():
    bpy.utils.register_class(IMPORT_OT_simple_collada_full)
    bpy.utils.register_class(EXPORT_OT_simple_collada_full)
    bpy.utils.register_class(OBJECT_OT_assign_textures_by_name)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)
    bpy.types.TOPBAR_MT_file_export.append(menu_func_export)
    bpy.types.VIEW3D_MT_object.append(menu_func_assign_textures)

def unregister():
    bpy.types.VIEW3D_MT_object.remove(menu_func_assign_textures)
    bpy.types.TOPBAR_MT_file_export.remove(menu_func_export)
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)
    bpy.utils.unregister_class(OBJECT_OT_assign_textures_by_name)
    bpy.utils.unregister_class(EXPORT_OT_simple_collada_full)
    bpy.utils.unregister_class(IMPORT_OT_simple_collada_full)

if __name__ == "__main__":
    register()
