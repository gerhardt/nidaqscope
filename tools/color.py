# define color schemes as simple dictionaries

from chaco.api import hot, jet
from chaco.color_mapper import ColorMapper

def creeper(range, **traits):
    """ Generator function for my 'creeper' colormap. """
    
    _data = { 'red'   : [ (0.0, 0.0625, 0.0625), # (threshold, color value below, color value above)
                          (0.5, 0.7656, 0.7656), #    => linear interpolation between thresholds
                          (1.0, 1.0000, 1.0000),
                        ],
              'green' : [ (0.0, 0.3750, 0.3750),
                          (0.5, 0.8398, 0.8398),
                          (1.0, 1.0000, 1.0000)
                        ],
              'blue'  : [ (0.0, 0.0000, 0.0000),
                          (0.5, 0.7500, 0.7500),
                          (1.0, 1.0000, 1.0000)
                        ],
              'alpha' : [ (0.0, 1.0000, 1.0000),
                          (1.0, 1.0000, 1.0000)
                        ]
            }
    return ColorMapper.from_segment_map(_data, range=range, **traits)

def hive(range, **traits):
    """ Generator function for my 'hive' colormap. """

    _data = { 'red'   : [ (0.0, 0.0000, 0.0000),
                          (0.1, 0.1796, 0.1796),
                          (0.4, 0.4438, 0.4438),
                          #(0.9, 0.8047, 0.8047),
                          (0.9, 0.8945, 0.8945),
                          (1.0, 1.0000, 1.0000),
                        ],
              'green' : [ (0.0, 0.0000, 0.0000),
                          (0.1, 0.1796, 0.1796),
                          (0.4, 0.2500, 0.2500),
                          #(0.9, 0.5352, 0.5352),
                          (0.9, 0.6367, 0.6367),
                          (1.0, 1.0000, 1.0000),
                        ],
              'blue'  : [ (0.0, 0.0000, 0.0000),
                          (0.1, 0.1796, 0.1796),
                          (0.4, 0.4438, 0.4438),
                          #(0.9, 0.3047, 0.3047),
                          (0.9, 0.3398, 0.3398),
                          (1.0, 1.0000, 1.0000),
                        ],
              'alpha' : [ (0.0, 1.0000, 1.0000),
                          (1.0, 1.0000, 1.0000)
                        ]
            }
    return ColorMapper.from_segment_map(_data, range=range, **traits)

def orc(range, **traits):
    """ Generator function for my 'orc' colormap. """

    _data = { 'red'   : [ (0.0, 0.0625, 0.0625),
                          (0.4, 0.5104, 0.5104),
                          (0.7, 1.0000, 1.0000),
                          (1.0, 1.0000, 1.0000),
                        ],
              'green' : [ (0.0, 0.3750, 0.3750),
                          (0.4, 0.5599, 0.5599),
                          (0.7, 0.8688, 0.8688),
                          (1.0, 1.0000, 1.0000)
                        ],
              'blue'  : [ (0.0, 0.0000, 0.0000),
                          (0.4, 0.5000, 0.5000),
                          (0.7, 0.6500, 0.6500),
                          (1.0, 1.0000, 1.0000)
                        ],
              'alpha' : [ (0.0, 1.0000, 1.0000),
                          (1.0, 1.0000, 1.0000)
                        ]
            }
    return ColorMapper.from_segment_map(_data, range=range, **traits)

def hive_matrix(range, **traits):
    _data = { 'red'   : [ (0.00, 0.0000, 0.0000),
                          (0.01, 1.0000, 1.0000),
                          (0.70, 0.8945, 0.8945),
                          (0.80, 0.4438, 0.4438), 
                          (0.87, 0.1796, 0.1796),
                          (1.00, 0.0000, 0.0000), 
                        ],
              'green' : [ (0.00, 0.0000, 0.0000),
                          (0.01, 1.0000, 1.0000),
                          (0.70, 0.6367, 0.6367),
                          (0.80, 0.2500, 0.2500),
                          (0.87, 0.1796, 0.1796),
                          (1.00, 0.0000, 0.0000),
                        ],
              'blue'  : [ (0.00, 0.0000, 0.0000),
                          (0.01, 1.0000, 1.0000),
                          (0.70, 0.3398, 0.3398),
                          (0.80, 0.4438, 0.4438),
                          (0.87, 0.1796, 0.1796),
                          (1.00, 0.0000, 0.0000),
                        ],
              'alpha' : [ (0.00, 1.0000, 1.0000),
                          (1.00, 1.0000, 1.0000)
                        ]
            }
    return ColorMapper.from_segment_map(_data, range=range, **traits)

# default colors
DEFAULT = {}
DEFAULT['container']  = 0xFFFFFF
DEFAULT['background'] = 0x000000
DEFAULT['gridcolor']  = 0x101010
DEFAULT['data 1']     = 0x0080FF
DEFAULT['data 2']     = 0x40B040
DEFAULT['data 3']     = 0xFFB0B0
DEFAULT['data 4']     = 0xFFB0B0
DEFAULT['fit 1']      = 0xFFA000
DEFAULT['fit 2']      = 0xFF6000
DEFAULT['fit 3']      = 0xFF6000
DEFAULT['fit 4']      = 0xFF6000
DEFAULT['colormap']   = hot
DEFAULT['scan']       = hot
DEFAULT['matrix']     = jet

# custom colors (change this)
custom = {}
custom['container']  = 0xF0F0F0
custom['background'] = 0x202020
custom['gridcolor']  = 0x101010
custom['data 1']     = 0xFFC575
custom['data 2']     = 0x61B750
custom['data 3']     = 0xFF6570
custom['data 4']     = 0x0080FF
custom['fit 1']      = 0x90724B
custom['fit 2']      = 0x3D6335
custom['fit 3']      = 0x9A454B
custom['fit 4']      = 0x14178D
custom['colormap']   = jet# hive
custom['scan']       = jet#hive
custom['matrix']     = jet#hive

# save colors
save = {}
save['container']  = 0xFFFFFF
save['background'] = 0xFFFFFF
save['gridcolor']  = 0xA0A0A0
save['data 1']     = 0x0000FF
save['data 2']     = 0x00FF00
save['data 3']     = 0xFF0000
save['data 4']     = 0xFF0000
save['fit 1']      = 0x0080FF
save['fit 2']      = 0xFFA000
save['fit 3']      = 0xFFA000
save['fit 4']      = 0xFFA000
save['colormap']   = hot
save['scan']       = hot
save['matrix']     = jet

# in order to change color scheme assign colors in custom dictionary
# and color dict to scheme.
scheme = custom