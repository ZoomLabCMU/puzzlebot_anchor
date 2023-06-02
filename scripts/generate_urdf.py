import os
import sys
import numpy as np
import xml.etree.ElementTree as ET
from xml.dom import minidom

param = {'body_length': 0.05, 'wall_thickness': 0.002,
        'base_height': 0.02, 'knob_radius': 0.0045,
        'knob_height': 0.01, 'top_height': 0.01,

        # anchor hole specifications
        'anchor_hole_width': 0.012, 
        'anchor_holder_width': 0.005, 'anchor_holder_height': 0.008,
        'anchor_holder_length': 0.008, 'anchor_door_width': 0.001,

        # anchor specifications
        'anchor_length': 0.011, 'anchor_height': 0.0095,
        'anchor_head_length': 0.0055, 'anchor_side_length': 0.01,
        'anchor_head_angle': 0.3, 'anchor_width': 0.002,

        # wheels
        'wheel_radius': 0.01, 'wheel_length': 0.003,
        'wheel_loc_y': 0.02, 'wheel_loc_z': 0.002,
        'ball_radius': 0.005, 'ball_loc_z': -0.00397,
        'eth': 0.001,
        
        # battery
        'battery_width': 0.02, 'battery_length': 0.026,
        'battery_height': 0.02
        }

def init_root(rname="puzzlebot"):
    robot = ET.Element('robot')
    robot.set("name", rname) 
    return robot

def add_material(root, material_name, rgba):
    '''
    rgba: array of size 4
    '''
    material = ET.SubElement(root, 'material')
    material.set('name', material_name)
    color = ET.SubElement(material, 'color')
    color.set('rgba', " ".join(format(i, ".2f") for i in rgba))

def add_origin(root, coord=[0,0,0,0,0,0]):
    origin = ET.SubElement(root, 'origin')
    origin.set('xyz', "%.4f %.4f %.4f" % tuple(coord[0:3]))
    origin.set('rpy', "%.4f %.4f %.4f" % tuple(coord[3:6]))

def add_base(root, m=0.1):
    base = ET.SubElement(root, 'link')
    base.set('name', "base_link")
    inertial = ET.SubElement(base, 'inertial')
    add_origin(inertial, np.zeros(6))
    mass = ET.SubElement(inertial, 'mass')
    mass.set('value', str(m))
    add_inertial(inertial, "manual", [0.01, 0.01, 0.01])

def add_inertial(root, otype, data):
    inertia = ET.SubElement(root, 'inertia')
    if otype == "manual":
        assert(len(data) == 3), "Manual inertia requires 3 values."
        set_inertia(inertia, data)
    elif otype == "box":
        assert(len(data) == 4), "Box inertia requires 4 values."
        m, x, y, z = data
        ixx = m * (y*y + z*z) / 12
        iyy = m * (x*x + z*z) / 12
        izz = m * (x*x + y*y) / 12
        set_inertia(inertia, [ixx, iyy, izz])
    elif otype == "cylinder":
        assert(len(data) == 3), "Cylinder inertia requires 3 values."
        m, r, l = data
        ixx = m * (3*r*r + l*l) / 12
        iyy = ixx
        izz = m * r*r / 2
        set_inertia(inertia, [ixx, iyy, izz])
    elif otype == "sphere":
        assert(len(data) == 2), "Sphere inertia requires 2 values."
        m, r = data
        ixx = m * r*r *2/5
        iyy = ixx
        izz = ixx
        set_inertia(inertia, [ixx, iyy, izz])

def add_component(root, cname, pname, xyz, rpy, shape, shape_name,
                    m=0.01, material="blue", 
                    joint_type="fixed", joint_param=None,
                    origin=np.zeros(6)):
    assert(shape_name in ['box', 'cylinder', 'sphere'])
    base = ET.SubElement(root, 'link')
    base.set('name', cname)
    for dname in ['visual', 'collision']:
        desc = ET.SubElement(base, dname)
        add_origin(desc, origin)
        geometry = ET.SubElement(desc, 'geometry')
        geom_child = ET.SubElement(geometry, shape_name)
        if shape_name == "box":
            geom_child.set('size', " ".join(
                                format(i, ".4f") for i in shape))
        elif shape_name == "cylinder":
            geom_child.set('radius', str(shape[0]))
            geom_child.set('length', str(shape[1]))
        elif shape_name == "sphere":
            geom_child.set('radius', str(shape[0]))
        if material and dname == 'visual':
            ET.SubElement(desc, 'material').set('name', material) 
    
    # add inertial components
    inertial = ET.SubElement(base, 'inertial')
    add_origin(inertial, origin)
    ET.SubElement(inertial, 'mass').set('value', str(m))
    add_inertial(inertial, shape_name, [m] + shape)
    
    add_joint(root, cname, pname, xyz, rpy, 
            joint_type, joint_param=joint_param)

def add_joint(root, cname, pname, xyz, rpy, jtype, 
            joint_param=None):

    if jtype == "floating": 
        add_floating_joint(root, cname, pname, xyz, rpy)
        return

    # add joint transform
    joint = ET.SubElement(root, 'joint')
    joint.set('name', cname + "_joint")
    joint.set('type', jtype)
    add_origin(joint, np.array(
                        [xyz[0], xyz[1], xyz[2], 
                        rpy[0], rpy[1], rpy[2]]))
    ET.SubElement(joint, 'parent').set('link', pname)
    ET.SubElement(joint, 'child').set('link', cname)

    if jtype == "fixed": return
    if not joint_param: return

    joint_param_dict = {}
    for attr in joint_param.keys():
        key = ET.SubElement(joint, attr)
        vs = joint_param[attr]
        for i in range(int(len(vs)/2)):
            key.set(vs[2*i], vs[2*i+1])

def add_floating_joint(root, cname, pname, xyz, rpy):
    # add joint transform
    virtual_link = cname + "_virtual"
    virtual_base = ET.SubElement(root, 'link')
    virtual_base.set('name', virtual_link)
    inertial = ET.SubElement(virtual_base, 'inertial')
    s = 0.0001
    ET.SubElement(inertial, 'mass').set('value', str(s))
    add_inertial(inertial, "box", [s, s, s, s])
    add_joint(root, virtual_link, pname, 
            xyz, rpy, "prismatic", 
            joint_param={'axis': ('xyz', "1 0 0"),
                        'limit': ('lower', str(-0.01), 
                                'upper', str(0.01),
                                'effort', str(5),
                                'velocity', str(1)),
                        'dynamics': ('friction', str(0.0))
                        })
    add_joint(root, cname, virtual_link, 
            xyz, rpy, "revolute", 
            joint_param={'axis': ('xyz', "0 0 1"),
                        'limit': ('lower', str(-np.pi), 
                                'upper', str(np.pi),
                                'effort', str(5),
                                'velocity', str(np.pi)),
                        'dynamics': ('friction', str(0.0))
                        })

def add_side(root, prefix, xyz, rpy):
    add_component(root, prefix + '_base', 'base_link',
                xyz, rpy,
                [param['wall_thickness'], L, param['base_height']],
                "box"
                )

    # add center box
    eth = param['eth']
    actual_knob_length = param['knob_radius']*2 + eth
    center_width = (param['body_length']/2 
                    - 2*actual_knob_length - 2*eth)*2
    knob_z_origin = param['base_height']/2 + param['knob_height']/2
    add_component(root, prefix + '_center', prefix + '_base',
                [0, 0, knob_z_origin],
                [0, 0, 0],
                [param['wall_thickness'], center_width, param['knob_height']],
                "box")

    # add knobs
    add_component(root, prefix + '_c0', prefix + '_base',
                [0, -1.5*actual_knob_length - center_width/2, knob_z_origin],
                [0, 0, 0],
                [param['knob_radius'], param['knob_height'] - 2*eth],
                "cylinder")
    add_component(root, prefix + '_c1', prefix + '_base',
                [0, 0.5*actual_knob_length + center_width/2, knob_z_origin],
                [0, 0, 0],
                [param['knob_radius'], param['knob_height'] - 2*eth],
                "cylinder")

    # add box next to knobs
    add_component(root, prefix + '_c_left', prefix + '_base',
                [0, param['body_length']/2 - eth, knob_z_origin],
                [0, 0, 0],
                [param['wall_thickness'], 2*eth, param['knob_height']],
                "box")
    add_component(root, prefix + '_c_right', prefix + '_base',
                [0, -param['body_length']/2 + eth, knob_z_origin],
                [0, 0, 0],
                [param['wall_thickness'], 2*eth, param['knob_height']],
                "box")

    # add top boxrobot
    add_component(root, prefix + '_top', prefix + '_base',
                [0, 0, param['base_height']/2 + param['knob_height'] + param['top_height']/2],
                [0, 0, 0],
                [param['wall_thickness'], param['body_length'], param['top_height']],
                "box")

def add_side_wheels(root, prefix, pname, xyz, rpy):
    add_component(root, prefix + '_wheel', pname,
                xyz, rpy,
                [param['wheel_radius'], param['wheel_length']],
                "cylinder", material=None,
                m=0.1,
                joint_type="continuous",
                joint_param={'axis': ('xyz', "0 0 1"), 
                        'dynamics': ('friction', "1.0")})

def add_ball_wheels(root, prefix, xyz, rpy):
    add_component(root, prefix + '_wheel', 'base_link',
                xyz, rpy,
                [param['ball_radius']],
                "sphere", material=None,
                m=0.1,
                joint_type="fixed")

def add_transmission(root, prefix):
    base = ET.SubElement(root, 'transmission')
    base.set('name', prefix + '_transmission')
    base.set('type', "SimpleTransmission")
    ET.SubElement(base, 'type').text = "transmission_interface/SimpleTransmission"
    joint_el = ET.SubElement(base, 'joint')
    joint_el.set('name', prefix + '_joint')
    ET.SubElement(joint_el, 'hardwareInterface').text = "EffortJointInterface"
    actuator_el = ET.SubElement(base, 'actuator')
    actuator_el.set('name', prefix + '_motor')
    ET.SubElement(actuator_el, 'hardwareInterface').text = "EffortJointInterface"
    ET.SubElement(actuator_el, 'mechanicalReduction').text = "1"

def add_anchor_side(root, prefix, xyz, rpy):
    eth = param['eth']
    kh = param['knob_height']
    add_component(root, prefix + '_base', 'base_link',
                xyz, rpy,
                [param['wall_thickness'], L, param['base_height']],
                "box"
                )
    # add top box
    add_component(root, prefix + '_top', prefix + '_base',
                [0, 0, param['base_height']/2 + kh + param['top_height']/2],
                [0, 0, 0],
                [param['wall_thickness'], param['body_length'], param['top_height']],
                "box")
    # add middle box
    add_component(root, prefix + '_middle', prefix + '_base',
                [0, 0, param['base_height']/2 + kh/2],
                [0, 0, 0],
                [param['wall_thickness'], param['body_length'], param['knob_height']],
                "box")

    # add anchor holder
    l = param['anchor_holder_length']
    w = param['anchor_holder_width']
    wd = param['anchor_door_width']
    h = param['anchor_holder_height']
    add_component(root, prefix + '_holder_bottom', prefix + '_middle',
                [l/2, 0, -kh/2 + 0.5*eth],
                [0, 0, 0],
                [l, w, eth],
                "box", m=0.001)
    add_component(root, prefix + '_holder_top', prefix + '_middle',
                [l/2, 0, kh/2 - 0.5*eth],
                [0, 0, 0],
                [l, w, eth],
                "box", m=0.001)
    add_component(root, prefix + '_front_left', prefix + '_middle',
                [l-eth/2, w/2-eth/2, 0],
                [0, 0, 0],
                [wd, wd, h],
                "box", m=0.001)
    add_component(root, prefix + '_front_right', prefix + '_middle',
                [l-eth/2, -w/2+eth/2, 0],
                [0, 0, 0],
                [wd, wd, h],
                "box", m=0.001)

def add_anchor(root, prefix, bname, xyz, rpy, is_ind=False):
    eth = param['eth']
    # add anchor
    l = param['anchor_length']
    w = param['anchor_holder_width']
    h = param['anchor_height']
    aw = param['anchor_width']
    hl = param['anchor_head_length']
    sl = param['anchor_side_length']
    ag = param['anchor_head_angle']
    #  base_joint_type = "floating" # bullet can't handle floating joint
    side_joint_type = "revolute"
    base_joint_type = "fixed"
    add_component(root, prefix + '_anchor_leg', bname,
                [eth*2, 0, 0],
                [0, 0, 0],
                [aw, w, h-1*eth], 
                "box", material="gray", m=0.001,
                joint_type=base_joint_type
                )
    add_component(root, prefix + '_anchor_base', prefix + '_anchor_leg',
                [l/2, 0, 0],
                [0, 0, 0],
                [l, aw, h-1*eth],
                "box", m=0.001, material="gray"
                )
    add_component(root, prefix + '_anchor_head', prefix + '_anchor_base',
                [l/2, 0, 0],
                [0, 0, 0],
                [eth, hl, h], 
                "box", material="gray", m=0.001)
    add_component(root, prefix + '_anchor_left', prefix + '_anchor_head',
                [0, hl/2, 0],
                [0, 0, ag],
                [eth, sl, h],
                "box", m=0.001, material="gray",
                joint_type=side_joint_type,
                joint_param={'axis': ('xyz', "0 0 1"),
                            'limit': ('lower', str(-ag), 
                                    'upper', str(np.pi/2 - ag),
                                    'effort', str(5),
                                    'velocity', str(2))
                            },
                origin=np.array([0, sl/2, 0, 0, 0, 0]))
    add_component(root, prefix + '_anchor_right', prefix + '_anchor_head',
                [0, -hl/2, 0],
                [0, 0, -ag],
                [eth, sl, h],
                "box", m=0.001, material="gray",
                joint_type=side_joint_type,
                joint_param={'axis': ('xyz', "0 0 1"),
                            'limit': ('lower', str(-np.pi/2 + ag), 
                                    'upper', str(ag),
                                    'effort', str(5),
                                    'velocity', str(2))
                            },
                origin=np.array([0, -sl/2, 0, 0, 0, 0]))

    # add virtual transmission for the anchor
    add_transmission(root, prefix + '_anchor_left')
    add_transmission(root, prefix + '_anchor_right')

def add_hole_side(root, prefix, xyz, rpy):
    eth = param['eth']
    L = param['body_length']
    side_width = (L - param['anchor_hole_width'])/2
    h = param['anchor_holder_height'] + 2*eth

    add_component(root, prefix + '_base', 'base_link',
                xyz, rpy,
                [param['wall_thickness'], L, param['base_height']],
                "box"
                )
    add_component(root, prefix + '_hole_left', prefix + '_base',
                [0, L/2 - side_width/2, h/2 + param['base_height']/2],
                [0, 0, 0],
                [param['wall_thickness'], side_width, h],
                "box"
                )
    add_component(root, prefix + '_hole_right', prefix + '_base',
                [0, -L/2 + side_width/2, h/2 + param['base_height']/2],
                [0, 0, 0],
                [param['wall_thickness'], side_width, h],
                "box"
                )

    # add top box
    add_component(root, prefix + '_top', prefix + '_base',
                [0, 0, param['base_height']/2 + h + param['top_height']/2],
                [0, 0, 0],
                [param['wall_thickness'], param['body_length'], param['top_height']],
                "box"
                )

def set_inertia(inertia, data):
    assert(len(data) == 3), "set_inertia requires 3 values"
    inertia.set('ixx', str(data[0]))
    inertia.set('ixy', "0")
    inertia.set('ixz', "0")
    inertia.set('iyy', str(data[1]))
    inertia.set('iyz', "0")
    inertia.set('izz', str(data[2]))
    return inertia

def write_file(data, rname="puzzlebot"):
    mydata = minidom.parseString(ET.tostring(data)).toprettyxml(indent="  ")
    dir_path = os.path.dirname(os.path.realpath(__file__))
    myfile = open(dir_path + "/../urdf/{}.urdf".format(rname), "w")
    myfile.write(mydata)

if __name__ == "__main__":
    L = param['body_length']
    eth = param['eth']

    robot = init_root()
    add_material(robot, "blue", [0.4, 0.6, 0.8, 0.99])
    add_material(robot, "gray", [0.5, 0.5, 0.5, 0.99])
    add_base(robot)
    add_side(robot, 'left', 
                [0, (L - param['wall_thickness'])/2, param['base_height']/2],
                [0, 0, np.pi/2])
    add_side(robot, 'right', 
                [0, -(L - param['wall_thickness'])/2, param['base_height']/2],
                [0, 0, -np.pi/2])

    # add wheels
    add_side_wheels(robot, 'left', 'base_link',
                [0, param['wheel_loc_y'], param['wheel_loc_z']],
                [-np.pi/2, 0, 0])
    add_side_wheels(robot, 'right', 'base_link',
                [0, -param['wheel_loc_y'], param['wheel_loc_z']],
                [-np.pi/2, 0, 0])
    add_transmission(robot, 'left_wheel')
    add_transmission(robot, 'right_wheel')
    add_ball_wheels(robot, 'front',
                [param['wheel_loc_y'] - eth, 0, param['ball_loc_z']],
                [0, 0, 0])
    add_ball_wheels(robot, 'back',
                [-param['wheel_loc_y'] + eth, 0, param['ball_loc_z']],
                [0, 0, 0])

    print(sys.argv)
    if sys.argv[1] == '0':
        add_side(robot, 'front', 
                    [(L - param['wall_thickness'])/2, 0, param['base_height']/2],
                    [0, 0, 0])
        add_side(robot, 'back', 
                    [-(L - param['wall_thickness'])/2, 0, param['base_height']/2],
                    [0, 0, np.pi])
        write_file(robot)
    elif sys.argv[1] == '1':
        add_hole_side(robot, 'front', 
                    [(L - param['wall_thickness'])/2, 0, param['base_height']/2],
                    [0, 0, 0])
        add_anchor_side(robot, 'back', 
                    [-(L - param['wall_thickness'])/2, 0, param['base_height']/2],
                    [0, 0, np.pi])
        add_anchor(robot, 'back', 'back_middle',
                    [-(L - param['wall_thickness'])/2, 0, param['base_height']/2],
                    [0, 0, np.pi])
        # add battery
        h = param['base_height'] + param['knob_height'] + param['top_height'] - param['battery_height']/2
        add_component(robot, 'battery', 'base_link',
                    [0, 0, h],
                    [0, 0, 0],
                    [param['battery_length'], param['battery_width'], param['battery_height']],
                    "box", m=0.1
                    )
        write_file(robot)
    elif sys.argv[1] == '2':
        add_hole_side(robot, 'front', 
                    [(L - param['wall_thickness'])/2, 0, param['base_height']/2],
                    [0, 0, 0])
        add_anchor_side(robot, 'back', 
                    [-(L - param['wall_thickness'])/2, 0, param['base_height']/2],
                    [0, 0, np.pi])
        # add battery
        h = param['base_height'] + param['knob_height'] + param['top_height'] - param['battery_height']/2
        add_component(robot, 'battery', 'base_link',
                    [0, 0, h],
                    [0, 0, 0],
                    [param['battery_length'], param['battery_width'], param['battery_height']],
                    "box", m=0.1
                    )

        write_file(robot)

        anchor_body = init_root("puz_anchor")
        add_base(anchor_body)
        add_material(anchor_body, "gray", [0.5, 0.5, 0.5, 0.99])
        add_anchor(anchor_body, 'back', 'base_link',
                    [0, 0, 0],
                    [0, 0, 0], is_ind=True)

        write_file(anchor_body, "puz_anchor")

