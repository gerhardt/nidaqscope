import logging, logging.handlers
import os
import inspect
import setup_params as setup

path = os.path.dirname(inspect.getfile(inspect.currentframe()))

# start the logger
logging_handler = logging.handlers.TimedRotatingFileHandler(path+'/log/diamond_log.txt', 'W6') # start new file every sunday, keeping all the old ones 
logging_handler.setFormatter(logging.Formatter("%(asctime)s - %(module)s.%(funcName)s - %(levelname)s - %(message)s"))
logging.getLogger().addHandler(logging_handler)
stream_handler=logging.StreamHandler()
stream_handler.setLevel(logging.ERROR) # console output
logging.getLogger().addHandler(stream_handler) # also log to stderr
logging.getLogger().setLevel(logging.INFO)
logging.getLogger().info('Starting logger.')

# start the JobManager
from tools import emod
emod.JobManager().start()

# start the CronDaemon
from tools import cron
cron.CronDaemon().start()

# define a shutdown function
from tools.utility import StoppableThread
import threading

def shutdown(timeout=1.0):
    """Terminate all threads."""
    cron.CronDaemon().stop()
    emod.JobManager().stop()
    for t in threading.enumerate():
        if isinstance(t, StoppableThread):
            t.stop(timeout=timeout)

import numpy as np

#########################################
# load hardware
#########################################
import hardware.nidaq
analogline=hardware.nidaq.AnalogInOut(AOChan=setup.analog_AO,
									  TickSource=setup.analog_tick,
									  AIChan=setup.analog_AI
									  )

#########################################
# load measurements
#########################################

import measurements.odmr

odmr = measurements.odmr.Analog()
odmr.edit_traits(analogline)