from enum import Enum

class Spraying(Enum):
        Single = 1
        Multiple = 2

class FlatField(Enum):
        No = 1
        Constant = 2
        Adaptive = 3
        Mean = 4

class FileInputMode(Enum):
        Single = 1
        DatasetNameParam = 2
        DatasetParam = 3
        NameParam = 4

exp_2018_09_ersf_mi1516 = {'dataset_name': 'ESRF Sep 2018 mi1516', 
        'root_path': u'/mnt/LSDF/projects/pn-reduction/2018_09_esrf_me1516/Phantom/',
        'datasets': ['17_3_18_1', '17_3_23_1', '17_3_5_1', '17_3_7_3'],
        'params': ['0', '2.5', '5', '7.5', '10', '12.5', '15', '17.5', '20'],
        
        'param_name': '_Tile_d',
        'file_input_mode': FileInputMode.DatasetNameParam,
        'single_filename': '------------',

        'image_size': (0, 0, 1024, 512),         # Region of intetest used for processing (x0, y0, width, height)
        'flip_image': False,

        'spray_mode': Spraying.Multiple,
        'num_frames': 3348,                     # Number of frames for reading and processing. num_frames <= 0 -> All 
        'spray_mode_params': {
                'shot_events' : [0,1,2,3,4,5,6,7,8,9,10,11,12,13],
                'start_events_offsets': [0,2,4,6,8,10,12,14,16],
                'start_index': 66,
                'end_index': 3300,
                'spray_duration': 94,           # number of consecutive frames with spray
                'spray_events_separation': 224, # number of frames without a spray
                'batch_size': 50,               # number of images used for processing in ech spraying event
                'use_every_nth': 1
                },
        
        'flat_mode': FlatField.Adaptive,              
        'adaptive_flat':  {
                'sigma': 15.0,          # sigma for low-pass filtering
                'flat_num': 20,         # number of used flats prior to each shot 
                'flat_offset': 20,      # offset from the start of spraying                        
                },
        'flat_field_value' : 1.0,

        'clean_intermediate_results': True


        }


exp_2018_03_ersf_mi1315 = {'dataset_name': 'ESRF March 2018 mi1315', 
        'root_path': u'/mnt/LSDF/projects/pn-reduction/2018_03_esrf_mi1325/Phantom/Glasduese/Nachtschicht 10.3 auf 11.3/',
        'datasets': ['007_1', '018_1'],
        'params': ['Z0Y0', 'Z2.5Y0', 'Z5Y0'],
        
        'param_name': '_Tile_d',
        'file_input_mode': FileInputMode.Single,
        'single_filename': 'OP_1bar_25C_100bar_25C.tif',

        'image_size': (0, 0, 1024, 512),         # Region of intetest used for processing (x0, y0, width, height)
        'flip_image': False,

        'spray_mode': Spraying.Multiple,
        'num_frames': 2882,                     # Number of frames for reading and processing. num_frames <= 0 -> All 
        'spray_mode_params': {
                'shot_events' : [0,1,2,3,4,5,6,7,8,9],
                'start_events_offsets': [0,2,4,6,8, 10,12,14,16],
                'start_index': 23,
                'end_index': 2800,
                'spray_duration': 90,           # number of consecutive frames with spray
                'spray_events_separation': 280, # number of frames without a spray
                'batch_size': 50,               # number of images used for processing in ech spraying event
                'use_every_nth': 1
                },
        
        'flat_mode': FlatField.Adaptive,              
        'adaptive_flat':  {
                'sigma': 15.0,          # sigma for low-pass filtering
                'flat_num': 20,         # number of used flats prior to each shot 
                'flat_offset': 20,      # offset from the start of spraying                        
                },
        'flat_field_value' : 1.0,

        'clean_intermediate_results': True


        }


sim= {'dataset_name': 'simulations', 
        'root_path':  u'/mnt/LSDF/projects/pn-reduction/ershov/',
        'datasets': ['test'],
        'params': ['1', '2'],

        'param_name': 'dataset',
        'file_input_mode': FileInputMode.NameParam,
        'single_filename': '------------',

        'image_size': (0, 0, 251, 251),         # Region of intetest used for processing (x0, y0, width, height)
        'flip_image': False,

        'spray_mode': Spraying.Single,
        'num_frames': 10,                     # Number of frames for reading and processing. num_frames <= 0 -> All 
        'spray_mode_params': {
                'start_events_offsets': [],
                'shot_events' : [0],
                'start_index': 0,
                'spray_duration': 0,           # number of consecutive frames with spray
                'end_index': 0,
                'spray_events_separation': 0,  # number of frames without a spray
                'batch_size': 0,               # number of images used for processing in ech spraying event
                'use_every_nth': 1
                },
        
        'flat_mode': FlatField.Constant,              
        'adaptive_flat':  {
                'sigma': 15.0,          # sigma for low-pass filtering
                'flat_num': 20,         # number of used flats prior to each shot 
                'flat_offset': 20,      # offset from the start of spraying                        
                },
        'flat_field_value' : 1.0,

        'clean_intermediate_results': True


        }

