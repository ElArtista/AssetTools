#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Script to convert models to mdl/anm binaries
# using Blender as a python module

import os
import io
import argparse
import bpy
import bmesh
import mathutils

from datetime import datetime
from contextlib import redirect_stdout
from cffi import FFI
from bpy_extras.io_utils import axis_conversion

ffi = FFI()
ffi.cdef("""
typedef uint8_t  u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
typedef int8_t   i8;
typedef int16_t  i16;
typedef int32_t  i32;
typedef int64_t  i64;
typedef float    f32;
typedef double   f64;
typedef uint8_t byte;

typedef struct {
    u32 offset;
    u32 size;
} data_chunk;

struct mdl_header {
    byte id[4];
    struct {
        u16 maj;
        u16 min;
    } ver;
    struct {
        int rigged : 1;
        int unused : 31;
    } flags;
    u32 num_vertices;
    u32 num_indices;
    u16 num_mesh_descs;
    u16 num_joints;
    u32 num_strings;
    data_chunk mesh_descs;
    data_chunk verts;
    data_chunk weights;
    data_chunk indices;
    data_chunk joints;
    data_chunk joint_name_ofs;
    data_chunk strings;
};

struct mdl_mesh_desc {
    u32 ofs_name;
    u32 num_vertices;
    u32 num_indices;
    u32 ofs_verts;
    u32 ofs_weights;
    u32 ofs_indices;
    u16 mat_idx;
};

struct mdl_vertex {
    f32 position[3];
    f32 normal[3];
    f32 uv[2];
};

struct mdl_vertex_weight {
    u16 blend_ids[4];
    f32 blend_weights[4];
};

struct mdl_joint {
    u32 ref_parent;
    f32 position[3];
    f32 rotation[4];
    f32 scaling[3];
};""")

MDL_INVALID_OFFSET = 0xFFFFFFFF

class Vertex:
    def __init__(self, pos, nm, uv):
        self.pos = pos
        self.nm  = nm
        self.uv  = uv
    def __eq__(self, other):
        return self.pos == other.pos \
           and self.nm  == other.nm \
           and self.uv  == other.uv
    def __hash__(self):
        return hash(self.pos[0]) \
             ^ hash(self.pos[1]) \
             ^ hash(self.pos[2]) \
             ^ hash(self.nm[0])  \
             ^ hash(self.nm[1])  \
             ^ hash(self.nm[2])  \
             ^ hash(self.uv[0])  \
             ^ hash(self.uv[1])

class Joint:
    def __init__(self, name, pos, rot, scl, par_idx):
        self.name = name
        self.pos = pos
        self.rot = rot
        self.scl = scl
        self.par_idx = par_idx
    def __eq__(self, other):
        return self.name == other.name \
           and self.pos == other.pos \
           and self.rot == other.rot \
           and self.scl == other.scl \
           and self.par_idx == other.par_idx

class Mesh:
    def __init__(self, verts, inds, mat_idx):
        self._verts = verts
        self._inds  = inds
        self._mat_index = mat_idx
    @property
    def vertices(self):
        return self._verts
    @property
    def indices(self):
        return self._inds
    @property
    def mat_index(self):
        return self._mat_index
    @mat_index.setter
    def mat_index(self, mat_index):
        self._mat_index = mat_index
    @staticmethod
    def meshes_from_bmesh(bm):
        meshes = []
        verts = []; inds = []; vertex_db = {}
        mat_idx = 0
        uv_lay = bm.loops.layers.uv.active
        facelist = sorted([f for f in bm.faces], key=lambda f: f.material_index)
        for f in facelist:
            if f.material_index != mat_idx:
                # Flush mesh
                meshes.append(Mesh(verts, inds, mat_idx))
                mat_idx = f.material_index
                verts = []; inds = []; vertex_db = {}
            for loop in f.loops:
                indice = loop.vert.index
                v  = loop.vert
                nv = Vertex(v.co[:], v.normal[:], loop[uv_lay].uv[:])
                indice = vertex_db.get(nv, None)
                if indice is not None:
                    inds.append(indice)
                else:
                    indice = len(verts)
                    inds.append(indice)
                    verts.append(nv)
                    vertex_db[nv] = indice
        meshes.append(Mesh(verts, inds, mat_idx))
        return meshes

def model_to_mdlfile(meshes, joints):
    tot_verts = 0
    tot_inds  = 0
    for m in meshes:
        tot_verts += len(m.vertices)
        tot_inds  += len(m.indices)

    strings = ""
    string_ofs = []
    for j in joints:
        string_ofs.append(len(strings))
        strings += j.name + '\0'

    h = ffi.new("struct mdl_header*")
    h.id                    = [0x4D, 0x44, 0x4C, 0x00]
    h.ver                   = [0, 1]
    h.flags                 = [1 if len(joints) != 0 else 0, 0]
    h.num_mesh_descs        = len(meshes)
    h.num_vertices          = tot_verts
    h.num_indices           = tot_inds
    h.num_joints            = len(joints)
    h.num_strings           = 0
    h.mesh_descs.offset     = ffi.sizeof("struct mdl_header")
    h.mesh_descs.size       = h.num_mesh_descs * ffi.sizeof("struct mdl_mesh_desc")
    h.verts.offset          = h.mesh_descs.offset + h.mesh_descs.size;
    h.verts.size            = h.num_vertices * ffi.sizeof("struct mdl_vertex")
    h.weights.offset        = h.verts.offset + h.verts.size;
    h.weights.size          = h.num_vertices * ffi.sizeof("struct mdl_vertex_weight") if h.flags.rigged else 0
    h.indices.offset        = h.weights.offset + h.weights.size
    h.indices.size          = h.num_indices * ffi.sizeof("u32")
    h.joints.offset         = h.indices.offset + h.indices.size
    h.joints.size           = h.num_joints * ffi.sizeof("struct mdl_joint")
    h.joint_name_ofs.offset = h.joints.offset + h.joints.size
    h.joint_name_ofs.size   = h.num_joints * ffi.sizeof("u32")
    h.strings.offset        = h.joint_name_ofs.offset + h.joint_name_ofs.size
    h.strings.size          = len(strings)

    sz = h.strings.offset + h.strings.size
    buf = ffi.new("byte[]", sz)
    ffi.memmove(buf, h, ffi.sizeof("struct mdl_header"))

    vofs = 0; iofs = 0
    for i in range(len(meshes)):
        mesh = meshes[i]
        md = ffi.cast("struct mdl_mesh_desc*", ffi.cast("size_t", (buf + h.mesh_descs.offset))) + i
        md.ofs_name     = MDL_INVALID_OFFSET
        md.num_vertices = len(mesh.vertices)
        md.num_indices  = len(mesh.indices)
        md.ofs_verts    = vofs * ffi.sizeof("struct mdl_vertex")
        md.ofs_weights  = vofs * ffi.sizeof("struct mdl_vertex_weight") if h.flags.rigged else MDL_INVALID_OFFSET
        md.ofs_indices  = iofs * ffi.sizeof("u32")
        md.mat_idx = mesh.mat_index
        for j in range(len(mesh.vertices)):
            mv = ffi.cast("struct mdl_vertex*", (buf + h.verts.offset + md.ofs_verts)) + j
            v = mesh.vertices[j]
            mv.position = v.pos
            mv.normal   = v.nm
            mv.uv       = v.uv

        idxbuf = ffi.new("u32[]", mesh.indices)
        ffi.memmove(buf + h.indices.offset + md.ofs_indices, idxbuf, md.num_indices * ffi.sizeof("u32"))
        vofs += md.num_vertices
        iofs += md.num_indices

    if h.flags.rigged:
        for i in range(h.num_joints):
            jnt = joints[i]
            mj = ffi.cast("struct mdl_joint*", (buf + h.joints.offset)) + i
            mj.position = jnt.pos
            mj.rotation = jnt.rot
            mj.scaling  = jnt.scl
            mj.ref_parent = jnt.par_idx if jnt.par_idx != -1 else MDL_INVALID_OFFSET

    if h.strings.size != 0:
        jname_buf = ffi.new("u32[]", string_ofs)
        ffi.memmove(buf + h.joint_name_ofs.offset, jname_buf, len(string_ofs) * ffi.sizeof("u32"))
        strings_buf = ffi.new("char[]", strings.encode('utf-8'))
        ffi.memmove(buf + h.strings.offset, strings_buf, ffi.sizeof(strings_buf))

    return ffi.buffer(buf)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Model file to process")
    return parser.parse_args()

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
    ext = os.path.splitext(input_file)[1]
    if ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=input_file, axis_up='-Y', use_manual_orientation=True, use_anim=True, use_prepost_rot=False)
    elif ext == ".obj":
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            bpy.ops.import_scene.obj(filepath=input_file)
    import_delta_time = datetime.now() - import_start_time
    print("[+] Total import time: {} secs".format(import_delta_time.total_seconds()))

    # Processing
    meshes = []
    joints = []
    mat_db = {}
    for o in bpy.context.selected_objects:
        if o.type == 'MESH':
            print("[+] Processing {}".format(o.name))
            # Create bmesh repr
            bm = bmesh.new()
            bm.from_mesh(o.data)
            # Transform
            #tmat = axis_conversion(to_forward='-Z', to_up='Y').to_4x4() * o.matrix_local
            tmat = o.matrix_local
            bm.transform(tmat)
            if tmat.determinant() < 0.0:
                bm.flip_normals() # If negative scaling, we have to invert the normals...
            # Triangulate
            bmesh.ops.triangulate(bm, faces=bm.faces[:], quad_method=0, ngon_method=0)
            # Gather mesh data
            mlist = Mesh.meshes_from_bmesh(bm)
            # Reassign mat indexes
            for i in range(len(mlist)):
                mesh = mlist[i]
                mat  = o.material_slots[i].name
                if mat in mat_db:
                    mesh.mat_index = mat_db[mat]
                else:
                    nidx = len(mat_db)
                    mat_db[mat] = nidx
                    mesh.mat_index = nidx
            meshes.extend(mlist)
            # Free bmesh repr
            bm.free()
        elif o.type == 'ARMATURE':
            print("[+] Joint group {} with {} bones".format(o.name, len(o.pose.bones)))
            quat_fmt = lambda q: [q.x, q.y, q.z, q.w]
            joints.append(Joint(o.name, [0.0, 0.0, 0.0], quat_fmt(mathutils.Quaternion()), [1.0, 1.0, 1.0], -1))
            bones = list(o.data.bones)
            for b in bones:
                par_name = b.parent.name if b.parent else o.name
                par_idx = next(i for i,j in enumerate(joints) if j.name == par_name)
                bon_mat = b.matrix_local
                par_mat = mathutils.Matrix.Identity(4) if not b.parent else b.parent.matrix_local
                mat = par_mat.inverted() * bon_mat
                pos, rot, scl = mat.decompose()
                j = Joint(b.name, pos[:], quat_fmt(rot), scl[:], par_idx)
                joints.append(j)

    # Write output file
    fdata = model_to_mdlfile(meshes, joints)
    output_file = input_file + ".mdl"
    with open(output_file, "wb") as f:
        f.write(fdata)

# Entrypoint
if __name__ == '__main__':
    time_point = datetime.now()
    main()
    ellapsed = datetime.now() - time_point
    print("[+] Total time: {} secs".format(ellapsed.total_seconds()))
