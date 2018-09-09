#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Script to convert models to gltf format
# using Blender as a python module

import io
import os
import sys
import argparse
import subprocess
from datetime import datetime
from contextlib import redirect_stdout

# Use bpy blender module, when running from inside blender
if sys.argv[0] == 'blender':
    import bpy

def import_scene(input_file):
    ext = os.path.splitext(input_file)[1]
    if ext.lower() == ".fbx":
        bpy.ops.import_scene.fbx(
            filepath=input_file,
            axis_forward='-Z',
            axis_up='Y',
            directory="",
            filter_glob="*.fbx",
            ui_tab='MAIN',
            use_manual_orientation=False,
            global_scale=1,
            bake_space_transform=False,
            use_custom_normals=True,
            use_image_search=True,
            use_alpha_decals=False,
            decal_offset=0,
            use_anim=True,
            anim_offset=1,
            use_custom_props=True,
            use_custom_props_enum_as_string=True,
            ignore_leaf_bones=False,
            force_connect_children=False,
            automatic_bone_orientation=False,
            primary_bone_axis='Y',
            secondary_bone_axis='X',
            use_prepost_rot=True)
    elif ext.lower() == ".obj":
        bpy.ops.import_scene.obj(
            filepath=input_file,
            axis_forward='-Z',
            axis_up='Y',
            filter_glob="*.obj;*.mtl",
            use_edges=True,
            use_smooth_groups=True,
            use_split_objects=True,
            use_split_groups=True,
            use_groups_as_vgroups=False,
            use_image_search=True,
            split_mode='ON',
            global_clamp_size=0)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Model file to process")
    if "--" not in sys.argv:
        args = sys.argv[1:]
    else:
        args = sys.argv[sys.argv.index("--") + 1:]
    return parser.parse_args(args)

def main():
    # Handle arguments
    args = parse_args()
    if not os.path.isfile(args.input):
        print("Input file does not exist!")
        return
    input_file = args.input

    # Select all
    bpy.ops.object.select_all(action='SELECT')
    # Remove selected
    bpy.ops.object.delete()

    # Import model
    print("[+] Importing asset...")
    import_start_time = datetime.now()
    stdout = io.StringIO()
    with redirect_stdout(stdout):
        import_scene(input_file)
    import_delta_time = datetime.now() - import_start_time
    print("[+] Total import time: {} secs".format(import_delta_time.total_seconds()))

    # Export gltf
    print("[+] Exporting asset...")
    output_file = os.path.splitext(os.path.abspath(input_file))[0] + ".gltf"
    print(output_file)
    bpy.ops.export_scene.gltf(
        filepath=output_file,
        check_existing=True,
        export_copyright="",
        export_embed_buffers=False,
        export_embed_images=False,
        export_strip=False,
        export_indices='UNSIGNED_INT',
        export_force_indices=False,
        export_texcoords=True,
        export_normals=True,
        export_tangents=True,
        export_materials=True,
        export_colors=True,
        export_cameras=False,
        export_camera_infinite=False,
        export_selected=False,
        export_layers=True,
        export_extras=False,
        export_yup=False,
        export_apply=False,
        export_animations=True,
        export_frame_range=True,
        export_frame_step=1,
        export_move_keyframes=True,
        export_force_sampling=False,
        export_current_frame=True,
        export_skins=True,
        export_bake_skins=False,
        export_morph=True,
        export_morph_normal=True,
        export_morph_tangent=True,
        export_lights=False,
        export_displacement=False,
        will_save_settings=False,
        filter_glob="*.gltf")

    print("[+] Done.")

# Entrypoint
if __name__ == '__main__':
    if sys.argv[0] == 'blender':
        time_point = datetime.now()
        main()
        ellapsed = datetime.now() - time_point
        print("[+] Total time: {} secs".format(ellapsed.total_seconds()))
    else:
        argv = ['blender', '--background', '--python', sys.argv[0]]
        argv.append('--')
        argv.extend(sys.argv[1:])
        subprocess.call(argv)
