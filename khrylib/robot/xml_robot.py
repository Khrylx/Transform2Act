import numpy as np
import math
from copy import deepcopy
from collections import defaultdict
from lxml.etree import XMLParser, parse, ElementTree, Element, SubElement
from lxml import etree
from io import BytesIO


def parse_vec(string):
    return np.fromstring(string, sep=' ')

def parse_fromto(string):
    fromto = np.fromstring(string, sep=' ')
    return fromto[:3], fromto[3:]

def normalize_range(value, lb, ub):
    return (value - lb) / (ub - lb) * 2 - 1

def denormalize_range(value, lb, ub):
    return (value + 1) * 0.5 * (ub - lb) + lb

def vec_to_polar(v):
    phi = math.atan2(v[1], v[0])
    theta = math.acos(v[2])
    return np.array([theta, phi])

def polar_to_vec(p):
    v = np.zeros(3)
    v[0] = math.sin(p[0]) * math.cos(p[1])
    v[1] = math.sin(p[0]) * math.sin(p[1])
    v[2] = math.cos(p[0])
    return v


class Joint:

    def __init__(self, node, body):
        self.node = node
        self.body = body
        self.cfg = body.cfg
        self.local_coord = body.local_coord
        self.name = node.attrib['name']
        self.type = node.attrib['type']
        if self.type == 'hinge':
            self.range = np.deg2rad(parse_vec(node.attrib.get('range', "-360 360")))
        actu_node = body.tree.getroot().find("actuator").find(f'motor[@joint="{self.name}"]')
        if actu_node is not None:
            self.actuator = Actuator(actu_node, self)
        else:
            self.actuator = None
        self.parse_param_specs()
        self.param_inited = False
        # tunable parameters
        self.pos = parse_vec(node.attrib['pos'])
        if self.type == 'hinge':
            self.axis = vec_to_polar(parse_vec(node.attrib['axis']))
        if self.local_coord:
            self.pos += body.pos
        assert(np.all(self.pos == body.pos))
    
    def __repr__(self):
        return 'joint_' + self.name

    def parse_param_specs(self):
        self.param_specs =  deepcopy(self.cfg['joint_params'])
        for name, specs in self.param_specs.items():
            if 'lb' in specs and isinstance(specs['lb'], list):
                specs['lb'] = np.array(specs['lb'])
            if 'ub' in specs and isinstance(specs['ub'], list):
                specs['ub'] = np.array(specs['ub'])

    def sync_node(self):
        pos = self.pos - self.body.pos if self.local_coord else self.pos
        self.name = self.body.name + '_joint'
        self.node.attrib['name'] = self.name
        self.node.attrib['pos'] = ' '.join([f'{x:.6f}'.rstrip('0').rstrip('.') for x in pos])
        if self.type == 'hinge':
            axis_vec = polar_to_vec(self.axis)
            self.node.attrib['axis'] = ' '.join([f'{x:.6f}'.rstrip('0').rstrip('.') for x in axis_vec])
        if self.actuator is not None:
            self.actuator.sync_node()

    def get_params(self, param_list, get_name=False, pad_zeros=False):
        if 'axis' in self.param_specs:
            if self.type == 'hinge':
                if get_name:
                    param_list += ['axis_theta', 'axis_phi']
                else:
                    axis = normalize_range(self.axis, np.array([0, -2 * np.pi]), np.array([np.pi, 2 * np.pi]))
                    param_list.append(axis)
            elif pad_zeros:
                param_list.append(np.zeros(2))

        if self.actuator is not None:
            self.actuator.get_params(param_list, get_name)
        elif pad_zeros:
            param_list.append(np.zeros(1))


        if not get_name:
            self.param_inited = True

    def set_params(self, params, pad_zeros=False):
        if 'axis' in self.param_specs:
            if self.type == 'hinge':
                self.axis = denormalize_range(params[:2], np.array([0, -2 * np.pi]), np.array([np.pi, 2 * np.pi]))
                params = params[2:]
            elif pad_zeros:
                params = params[2:]

        if self.actuator is not None:
            params = self.actuator.set_params(params)
        elif pad_zeros:
            params = params[1:]
        return params


class Geom:

    def __init__(self, node, body):
        self.node = node
        self.body = body
        self.cfg = body.cfg
        self.local_coord = body.local_coord
        self.name = node.attrib.get('name', '')
        self.type = node.attrib['type']
        self.parse_param_specs()
        self.param_inited = False
        # tunable parameters
        self.size = parse_vec(node.attrib['size'])
        if self.type == 'capsule':
            self.start, self.end = parse_fromto(node.attrib['fromto'])
            if self.local_coord:
                self.start += body.pos
                self.end += body.pos
            if body.bone_start is None:
                self.bone_start = self.start.copy()
                body.bone_start = self.bone_start.copy()
            else:
                self.bone_start = body.bone_start.copy()
            self.ext_start = np.linalg.norm(self.bone_start - self.start)
    
    def __repr__(self):
        return 'geom_' + self.name

    def parse_param_specs(self):
        self.param_specs = deepcopy(self.cfg['geom_params'])
        for name, specs in self.param_specs.items():
            if 'lb' in specs and isinstance(specs['lb'], list):
                specs['lb'] = np.array(specs['lb'])
            if 'ub' in specs and isinstance(specs['ub'], list):
                specs['ub'] = np.array(specs['ub'])

    def update_start(self):
        if self.type == 'capsule':
            vec = self.bone_start - self.end
            self.start = self.bone_start + vec * (self.ext_start / np.linalg.norm(vec))

    def sync_node(self):
        # self.node.attrib['name'] = self.name
        self.node.attrib.pop('name', None)
        self.node.attrib['size'] = ' '.join([f'{x:.6f}'.rstrip('0').rstrip('.') for x in self.size])
        if self.type == 'capsule':
            start = self.start - self.body.pos if self.local_coord else self.start
            end = self.end - self.body.pos if self.local_coord else self.end
            self.node.attrib['fromto'] = ' '.join([f'{x:.6f}'.rstrip('0').rstrip('.') for x in np.concatenate([start, end])])

    def get_params(self, param_list, get_name=False, pad_zeros=False):
        if 'size' in self.param_specs:
            if get_name:
                param_list.append('size')
            else:
                if self.type == 'capsule':
                    if not self.param_inited and self.param_specs['size'].get('rel', False):
                        self.param_specs['size']['lb'] += self.size
                        self.param_specs['size']['ub'] += self.size
                        self.param_specs['size']['lb'] = max(self.param_specs['size']['lb'], self.param_specs['size'].get('min', -np.inf))
                        self.param_specs['size']['ub'] = min(self.param_specs['size']['ub'], self.param_specs['size'].get('max', np.inf))
                    size = normalize_range(self.size, self.param_specs['size']['lb'], self.param_specs['size']['ub'])
                    param_list.append(size.flatten())
                elif pad_zeros:
                    param_list.append(np.zeros(1))
        if 'ext_start' in self.param_specs:
            if get_name:
                param_list.append('ext_start')
            else:
                if self.type == 'capsule':
                    if not self.param_inited and self.param_specs['ext_start'].get('rel', False):
                        self.param_specs['ext_start']['lb'] += self.ext_start
                        self.param_specs['ext_start']['ub'] += self.ext_start
                        self.param_specs['ext_start']['lb'] = max(self.param_specs['ext_start']['lb'], self.param_specs['ext_start'].get('min', -np.inf))
                        self.param_specs['ext_start']['ub'] = min(self.param_specs['ext_start']['ub'], self.param_specs['ext_start'].get('max', np.inf))
                    ext_start = normalize_range(self.ext_start, self.param_specs['ext_start']['lb'], self.param_specs['ext_start']['ub'])
                    param_list.append(ext_start.flatten())
                elif pad_zeros:
                    param_list.append(np.zeros(1))

        if not get_name:
            self.param_inited = True

    def set_params(self, params, pad_zeros=False):
        if 'size' in self.param_specs:
            if self.type == 'capsule':
                self.size = denormalize_range(params[[0]], self.param_specs['size']['lb'], self.param_specs['size']['ub'])
                params = params[1:]
            elif pad_zeros:
                params = params[1:]
        if 'ext_start' in self.param_specs:
            if self.type == 'capsule':
                self.ext_start = denormalize_range(params[[0]], self.param_specs['ext_start']['lb'], self.param_specs['ext_start']['ub'])
                params = params[1:]
            elif pad_zeros:
                params = params[1:]
        return params


class Actuator:

    def __init__(self, node, joint):
        self.node = node
        self.joint = joint
        self.cfg = joint.cfg
        self.joint_name = node.attrib['joint']
        self.name = self.joint_name
        self.parse_param_specs()
        self.param_inited = False
        # tunable parameters
        self.gear = float(node.attrib['gear'])

    def parse_param_specs(self):
        self.param_specs =  deepcopy(self.cfg['actuator_params'])
        for name, specs in self.param_specs.items():
            if 'lb' in specs and isinstance(specs['lb'], list):
                specs['lb'] = np.array(specs['lb'])
            if 'ub' in specs and isinstance(specs['ub'], list):
                specs['ub'] = np.array(specs['ub'])

    def sync_node(self):
        self.node.attrib['gear'] = f'{self.gear:.6f}'.rstrip('0').rstrip('.')
        self.name = self.joint.name
        self.node.attrib['name'] = self.name
        self.node.attrib['joint'] = self.joint.name

    def get_params(self, param_list, get_name=False):
        if 'gear' in self.param_specs:
            if get_name:
                param_list.append('gear')
            else:
                if not self.param_inited and self.param_specs['gear'].get('rel', False):
                    self.param_specs['gear']['lb'] += self.gear
                    self.param_specs['gear']['ub'] += self.gear
                    self.param_specs['gear']['lb'] = max(self.param_specs['gear']['lb'], self.param_specs['gear'].get('min', -np.inf))
                    self.param_specs['gear']['ub'] = min(self.param_specs['gear']['ub'], self.param_specs['gear'].get('max', np.inf))
                gear = normalize_range(self.gear, self.param_specs['gear']['lb'], self.param_specs['gear']['ub'])
                param_list.append(np.array([gear]))

        if not get_name:
            self.param_inited = True

    def set_params(self, params):
        if 'gear' in self.param_specs:
            self.gear = denormalize_range(params[0].item(), self.param_specs['gear']['lb'], self.param_specs['gear']['ub'])
            params = params[1:]
        return params


class Body:

    def __init__(self, node, parent_body, robot, cfg):
        self.node = node
        self.parent = parent_body
        if parent_body is not None:
            parent_body.child.append(self)
            parent_body.cind += 1
            self.depth = parent_body.depth + 1
        else:
            self.depth = 0
        self.robot = robot
        self.cfg = cfg
        self.tree = robot.tree
        self.local_coord = robot.local_coord
        self.name = node.attrib['name'] if 'name' in node.attrib else self.parent.name + f'_child{len(self.parent.child)}'
        self.child = []
        self.cind = 0
        self.pos = parse_vec(node.attrib['pos'])
        if self.local_coord and parent_body is not None:
            self.pos += parent_body.pos
        if cfg.get('init_root_from_geom', False):
            self.bone_start = None if parent_body is None else self.pos.copy()
        else:
            self.bone_start = self.pos.copy()
        self.joints = [Joint(x, self) for x in node.findall('joint[@type="hinge"]')] + [Joint(x, self) for x in node.findall('joint[@type="free"]')]
        self.geoms = [Geom(x, self) for x in node.findall('geom[@type="capsule"]')] + [Geom(x, self) for x in node.findall('geom[@type="sphere"]')]
        self.parse_param_specs()
        self.param_inited = False
        # parameters
        self.bone_end = None
        self.bone_offset = None

    def __repr__(self):
        return 'body_' + self.name

    def parse_param_specs(self):
        self.param_specs = deepcopy(self.cfg['body_params'])
        for name, specs in self.param_specs.items():
            if 'lb' in specs and isinstance(specs['lb'], list):
                specs['lb'] = np.array(specs['lb'])
            if 'ub' in specs and isinstance(specs['ub'], list):
                specs['ub'] = np.array(specs['ub'])
            if name == 'bone_ang':
                specs['lb'] = np.deg2rad(specs['lb'])
                specs['ub'] = np.deg2rad(specs['ub'])

    def reindex(self):
        if self.parent is None:
            self.name = '0'
        else:
            ind = self.parent.child.index(self) + 1
            pname = '' if self.parent.name == '0' else self.parent.name
            self.name = str(ind) + pname

    def init(self):
        if len(self.child) > 0:
            bone_ends = [x.bone_start for x in self.child]
        else:
            bone_ends = [x.end for x in self.geoms]
        if len(bone_ends) > 0:
            self.bone_end = np.mean(np.stack(bone_ends), axis=0)
            self.bone_offset = self.bone_end - self.bone_start

    def get_actuator_name(self):
        for joint in self.joints:
            if joint.actuator is not None:
                return joint.actuator.name
        return None

    def get_joint_range(self):
        assert len(self.joints) == 1
        return self.joints[0].range

    def sync_node(self):
        pos = self.pos - self.parent.pos if self.local_coord and self.parent is not None else self.pos
        self.node.attrib['name'] = self.name
        self.node.attrib['pos'] = ' '.join([f'{x:.6f}'.rstrip('0').rstrip('.') for x in pos])
        for joint in self.joints:
            joint.sync_node()
        for geom in self.geoms:
            geom.sync_node()

    def sync_geom(self):
        for geom in self.geoms:
            geom.bone_start = self.bone_start.copy()
            geom.end = self.bone_end.copy()
            geom.update_start()

    def sync_joint(self):
        if self.parent is not None:
            for joint in self.joints:
                joint.pos = self.pos.copy()

    def rebuild(self):
        if self.parent is not None:
            self.bone_start = self.parent.bone_end.copy()
            self.pos = self.bone_start.copy()
        if self.bone_offset is not None:
            self.bone_end = self.bone_start + self.bone_offset
        if self.parent is None and self.cfg.get('no_root_offset', False):
            self.bone_end = self.bone_start
        self.sync_geom()
        self.sync_joint()

    def get_params(self, param_list, get_name=False, pad_zeros=False, demap_params=False):
        if self.bone_offset is not None and 'offset' in self.param_specs:
            if get_name:
                if self.param_specs['offset']['type'] == 'xz':
                    param_list += ['offset_x', 'offset_z']
                elif self.param_specs['offset']['type'] == 'xy':
                    param_list += ['offset_x', 'offset_y']
                else:
                    param_list += ['offset_x', 'offset_y', 'offset_z']
            else:
                if self.param_specs['offset']['type'] == 'xz':
                    offset = self.bone_offset[[0, 2]]
                elif self.param_specs['offset']['type'] == 'xy':
                    offset = self.bone_offset[[0, 1]]
                else:
                    offset = self.bone_offset
                if not self.param_inited and self.param_specs['offset'].get('rel', False):
                    self.param_specs['offset']['lb'] += offset
                    self.param_specs['offset']['ub'] += offset
                    self.param_specs['offset']['lb'] = np.maximum(self.param_specs['offset']['lb'], self.param_specs['offset'].get('min', np.full_like(offset, -np.inf)))
                    self.param_specs['offset']['ub'] = np.minimum(self.param_specs['offset']['ub'], self.param_specs['offset'].get('max', np.full_like(offset, np.inf)))
                offset = normalize_range(offset, self.param_specs['offset']['lb'], self.param_specs['offset']['ub'])
                param_list.append(offset.flatten())

        if self.bone_offset is not None and 'bone_len' in self.param_specs:
            if get_name:
                param_list += ['bone_len']
            else:
                bone_len = np.linalg.norm(self.bone_offset)
                if not self.param_inited and self.param_specs['bone_len'].get('rel', False):
                    self.param_specs['bone_len']['lb'] += bone_len
                    self.param_specs['bone_len']['ub'] += bone_len
                    self.param_specs['bone_len']['lb'] = max(self.param_specs['bone_len']['lb'], self.param_specs['bone_len'].get('min',-np.inf))
                    self.param_specs['bone_len']['ub'] = min(self.param_specs['bone_len']['ub'], self.param_specs['bone_len'].get('max', np.inf))
                bone_len = normalize_range(bone_len, self.param_specs['bone_len']['lb'], self.param_specs['bone_len']['ub'])
                param_list.append(np.array([bone_len]))

        if self.bone_offset is not None and 'bone_ang' in self.param_specs:
            if get_name:
                param_list += ['bone_ang']
            else:
                bone_ang = math.atan2(self.bone_offset[2], self.bone_offset[0])
                if not self.param_inited and self.param_specs['bone_ang'].get('rel', False):
                    self.param_specs['bone_ang']['lb'] += bone_ang
                    self.param_specs['bone_ang']['ub'] += bone_ang
                    self.param_specs['bone_ang']['lb'] = max(self.param_specs['bone_ang']['lb'], self.param_specs['bone_ang'].get('min',-np.inf))
                    self.param_specs['bone_ang']['ub'] = min(self.param_specs['bone_ang']['ub'], self.param_specs['bone_ang'].get('max', np.inf))
                bone_ang = normalize_range(bone_ang, self.param_specs['bone_ang']['lb'], self.param_specs['bone_ang']['ub'])
                param_list.append(np.array([bone_ang]))

        for joint in self.joints:
            joint.get_params(param_list, get_name, pad_zeros)
        if pad_zeros and len(self.joints) == 0:
            param_list.append(np.zeros(1))

        for geom in self.geoms:
            geom.get_params(param_list, get_name, pad_zeros)
        if not get_name:
            self.param_inited = True

        if demap_params and not get_name:
            params = self.robot.demap_params(np.concatenate(param_list))
            return params

    def set_params(self, params, pad_zeros=False, map_params=False):
        if map_params:
            params = self.robot.map_params(params)

        if self.bone_offset is not None and 'offset' in self.param_specs:
            if self.param_specs['offset']['type'] in {'xz', 'xy'}:
                offset = denormalize_range(params[:2], self.param_specs['offset']['lb'], self.param_specs['offset']['ub'])
                if np.all(offset == 0.0):
                    offset[0] += 1e-8
                if self.param_specs['offset']['type'] == 'xz':
                    self.bone_offset[[0, 2]] = offset
                elif self.param_specs['offset']['type'] == 'xy':
                    self.bone_offset[[0, 1]] = offset
                params = params[2:]
            else:
                offset = denormalize_range(params[:3], self.param_specs['offset']['lb'], self.param_specs['offset']['ub'])
                if np.all(offset == 0.0):
                    offset[0] += 1e-8
                self.bone_offset[:] = offset
                params = params[3:]

        if self.bone_offset is not None and 'bone_len' in self.param_specs:
            bone_len = denormalize_range(params[0].item(), self.param_specs['bone_len']['lb'], self.param_specs['bone_len']['ub'])
            bone_len = max(bone_len, 1e-4)
            params = params[1:]
        else:
            bone_len = np.linalg.norm(self.bone_offset)

        if self.bone_offset is not None and 'bone_ang' in self.param_specs:
            bone_ang = denormalize_range(params[0].item(), self.param_specs['bone_ang']['lb'], self.param_specs['bone_ang']['ub'])
            params = params[1:]
        else:
            bone_ang = math.atan2(self.bone_offset[2], self.bone_offset[0])

        if 'bone_len' in self.param_specs or 'bone_ang' in self.param_specs:
            self.bone_offset = np.array([bone_len * math.cos(bone_ang), 0, bone_len * math.sin(bone_ang)])

        for joint in self.joints:
            params = joint.set_params(params, pad_zeros)
        for geom in self.geoms:
            params = geom.set_params(params, pad_zeros)
        # rebuild bone, geom, joint
        self.rebuild()
        return params


class Robot:

    def __init__(self, cfg, xml, is_xml_str=False):
        self.bodies = []
        self.cfg = cfg
        self.param_mapping = cfg.get('param_mapping', 'clip')
        self.tree = None    # xml tree
        self.load_from_xml(xml, is_xml_str)
        self.init_bodies()
        self.param_names = self.get_params(get_name=True)
        self.init_params = self.get_params()

    def load_from_xml(self, xml, is_xml_str=False):
        parser = XMLParser(remove_blank_text=True)
        self.tree = parse(BytesIO(xml) if is_xml_str else xml, parser=parser)
        self.local_coord = self.tree.getroot().find('.//compiler').attrib['coordinate'] == 'local'
        root = self.tree.getroot().find('worldbody').find('body')
        self.add_body(root, None)

    def add_body(self, body_node, parent_body):
        body = Body(body_node, parent_body, self, self.cfg)
        self.bodies.append(body)

        for body_node_c in body_node.findall('body'):
            self.add_body(body_node_c, body)

    def init_bodies(self):
        for body in self.bodies:
            body.init()
        self.sync_node()

    def sync_node(self):
        for body in self.bodies:
            body.reindex()
            body.sync_node()

    def add_child_to_body(self, body):
        if body == self.bodies[0]:
            body2clone = body.child[0]
        else:
            body2clone = body
        child_node = deepcopy(body2clone.node)
        for bnode in child_node.findall('body'):
            child_node.remove(bnode)
        child_body = Body(child_node, body, self, self.cfg)
        actu_node = body.tree.getroot().find("actuator")
        for joint in child_body.joints:
            new_actu_node = deepcopy(actu_node.find(f'motor[@joint="{joint.name}"]'))
            actu_node.append(new_actu_node)
            joint.actuator = Actuator(new_actu_node, joint)
        child_body.bone_offset = body.bone_offset.copy()
        child_body.param_specs = deepcopy(body.param_specs)
        child_body.param_inited = True
        child_body.rebuild()
        child_body.sync_node()
        body.node.append(child_node)
        self.bodies.append(child_body)
        self.sync_node()
        
    def remove_body(self, body):
        body.node.getparent().remove(body.node)
        body.parent.child.remove(body)
        self.bodies.remove(body)
        actu_node = body.tree.getroot().find("actuator")
        for joint in body.joints:
            actu_node.remove(joint.actuator.node)
        del body
        self.sync_node()

    def write_xml(self, fname):
        self.tree.write(fname, pretty_print=True)

    def export_xml_string(self):
        return etree.tostring(self.tree, pretty_print=True)

    def demap_params(self, params):
        if not np.all((params <= 1.0) & (params >= -1.0)):
            print(f'param out of bounds: {params}')
        params = np.clip(params, -1.0, 1.0)
        if self.param_mapping == 'sin':
            params = np.arcsin(params) / (0.5 * np.pi)
        return params

    def get_params(self, get_name=False):
        param_list = []
        for body in self.bodies:
            body.get_params(param_list, get_name)

        if not get_name:
            params = np.concatenate(param_list)
            params = self.demap_params(params)
        else:
            params = np.array(param_list)
        return params

    def map_params(self, params):
        if self.param_mapping == 'clip':
            params = np.clip(params, -1.0, 1.0)
        elif self.param_mapping == 'sin':
            params = np.sin(params * (0.5 * np.pi))
        return params

    def set_params(self, params):
        # clip params to range
        params = self.map_params(params)
        for body in self.bodies:
            params = body.set_params(params)
        assert(len(params) == 0)    # all parameters need to be consumed!
        self.sync_node()

    def rebuild(self):
        for body in self.bodies:
            body.rebuild()
            body.sync_node()

    def get_gnn_edges(self):
        edges = []
        for i, body in enumerate(self.bodies):
            if body.parent is not None:
                j = self.bodies.index(body.parent)
                edges.append([i, j])
                edges.append([j, i])
        edges = np.stack(edges, axis=1)
        return edges


if __name__ == "__main__":
    import os
    import sys
    import time
    import yaml
    sys.path.append(os.getcwd())
    from mujoco_py import load_model_from_path, MjSim, MjViewer

    model_name = 'ant'
    cfg_path = f'khrylib/assets/ant.yml'
    # model = load_model_from_path(f'assets/mujoco_envs/{model_name}.xml')
    yml = yaml.safe_load(open(cfg_path, 'r'))
    cfg = yml['robot']
    xml_robot = Robot(cfg, xml=f'assets/mujoco_envs/{model_name}.xml')
    params_names = xml_robot.get_params(get_name=True)

    params = xml_robot.get_params()
    print(params_names, params)
    new_params = params # + 0.1
    xml_robot.set_params(new_params)
    params_new = xml_robot.get_params()

    os.makedirs('out', exist_ok=True)
    xml_robot.write_xml(f'out/{model_name}_test.xml')

    model = load_model_from_path(f'out/{model_name}_test.xml')
    sim = MjSim(model)
    viewer = MjViewer(sim)
    viewer.cam.distance = 10

    while True:
        sim.data.qpos[2] = 1.0
        sim.data.qpos[7:] = np.pi / 6
        sim.forward()
        viewer.render()