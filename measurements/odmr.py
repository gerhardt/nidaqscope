import numpy as np

from traits.api import SingletonHasTraits, Trait, Instance, Property, String, Range, Float, Int, Bool, Array, Enum, Button, on_trait_change, cached_property, Code, List, NO_COMPARE
from traitsui.api import View, Item, HGroup, VGroup, VSplit, Tabbed, EnumEditor, TextEditor, Group
from enable.api import Component, ComponentEditor
from chaco.api import ArrayPlotData, Plot, PlotLabel
from chaco.tools.api import PanTool
from chaco.tools.simple_zoom import SimpleZoom

from traitsui.file_dialog import save_file
from traitsui.menu import Action, Menu, MenuBar

import time
import threading
import logging

from tools.emod import ManagedJob
from tools.color import scheme

# from analysis.Analog_analysis import fit_Analog, n_lorentzians, n_gaussians

from tools.utility import GetSetItemsHandler, GetSetItemsMixin

class AnalogHandler( GetSetItemsHandler ):
    
    def saveLinePlot(self, info):
        filename = save_file(title='Save Line Plot')
        if filename is '':
            return
        else:
            if filename.find('.png')==-1:
                filename=filename+'.png'
            info.object.save_line_plot(filename)
    
    def saveMatrixPlot(self, info):
        filename = save_file(title='Save Matrix Plot')
        if filename is '':
            return
        else:
            if filename.find('.png')==-1:
                filename=filename+'.png'
            info.object.save_matrix_plot(filename)
    
    def saveAll(self, info):
        filename = save_file(title='Save All')
        if filename is '':
            return
        else:
            info.object.save_all(filename)


class Analog( ManagedJob, GetSetItemsMixin ):
    """Provides Analog measurements."""
    
    # starting and stopping
    keep_data = Bool(False) # helper variable to decide whether to keep existing data
    
    # measurement parameters
    power = Range(low=-100., high=25., value=-20, desc='Power [dBm]', label='Power [dBm]', mode='text', auto_set=False, enter_set=True)
    voltage_begin = Float(default_value=-10,    desc='Start Voltage [V]',    label='Begin [V]', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_str='%e'))
    voltage_end   = Float(default_value=10,    desc='Stop Voltage [V]',     label='End [V]', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_str='%e'))
    voltage_delta = Float(default_value=0.01,       desc='Voltage step [V]',     label='Delta [V]', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_str='%e'))
    t_pi  = Range(low=1., high=100000., value=50., desc='length of pi pulse [ns]', label='pi [ns]', mode='text', auto_set=False, enter_set=True)
    laser = Range(low=1., high=10000., value=300., desc='laser [ns]', label='laser [ns]', mode='text', auto_set=False, enter_set=True)
    wait  = Range(low=1., high=10000., value=1000., desc='wait [ns]', label='wait [ns]', mode='text', auto_set=False, enter_set=True)
    pulsed = Bool(True, label='pulsed')
    seconds_per_point = Range(low=20e-3, high=1, value=20e-3, desc='Seconds per point', label='Seconds per point', mode='text', auto_set=False, enter_set=True)
    stop_time = Range(low=1., value=np.inf, desc='Time after which the experiment stops by itself [s]', label='Stop time [s]', mode='text', auto_set=False, enter_set=True)
    n_lines = Range (low=1, high=10000, value=50, desc='Number of lines in Matrix', label='Matrix lines', mode='text', auto_set=False, enter_set=True)
    
    # control data fitting
    perform_fit = Bool(False, label='perform fit')
    fit_model = Enum(['gauss', 'lorentz'])
    number_of_resonances = Trait( 'auto', String('auto', auto_set=False, enter_set=True), Int(10000., desc='Number of Resonances used in fit', label='N', auto_set=False, enter_set=True))
    threshold = Range(low=-99, high=99., value=-50., desc='Threshold for detection of resonances [%]. The sign of the threshold specifies whether the resonances are negative or positive.', label='threshold [%]', mode='text', auto_set=False, enter_set=True)
    
    # fit result
    fit_parameters = Array(value=np.array((np.nan, np.nan, np.nan, np.nan)))
    fit_frequencies = Array(value=np.array((np.nan,)), label='v [V]') 
    fit_line_width = Array(value=np.array((np.nan,)), label='line_width [Hz]') 
    fit_contrast = Array(value=np.array((np.nan,)), label='contrast [%]')
    
    # measurement data    
    frequency = Array()
    counts = Array()
    counts_matrix = Array()
    run_time = Float(value=0.0, desc='Run time [s]', label='Run time [s]')
    
    # plotting
    line_label  = Instance( PlotLabel )
    line_data   = Instance( ArrayPlotData )
    matrix_data = Instance( ArrayPlotData )
    line_plot   = Instance( Plot, editor=ComponentEditor() )
    matrix_plot = Instance( Plot, editor=ComponentEditor() )
    
    baseline = Bool(False, label='baseline', desc='show baseline')
    
    # import/export
    import_button = Button(label='import')
    export_button = Button(label='export')
    peak          = Int(-1)
    
    def __init__(self, **kwargs):
        super(Analog, self).__init__(**kwargs)
        
        self._create_line_plot()
        self._create_matrix_plot()
        self.on_trait_change(self._update_line_data_index,      'frequency',            dispatch='ui')
        self.on_trait_change(self._update_line_data_value,      'counts',               dispatch='ui')
        self.on_trait_change(self._update_line_data_fit,        'fit_parameters',       dispatch='ui')
        self.on_trait_change(self._update_matrix_data_value,    'counts_matrix',        dispatch='ui')
        self.on_trait_change(self._update_matrix_data_index,    'n_lines,frequency',    dispatch='ui')
        self.on_trait_change(self._update_fit, 'counts,perform_fit,number_of_resonances,threshold', dispatch='ui')
    
    def _counts_matrix_default(self):
        return np.zeros( (self.n_lines, len(self.frequency)) )
    
    def _frequency_default(self):
        return np.arange(self.voltage_begin, self.voltage_end+self.voltage_delta, self.voltage_delta)
    
    def _counts_default(self):
        return np.zeros(self.frequency.shape)
    
    # data acquisition
    
    def apply_parameters(self):
        """Apply the current parameters and decide whether to keep previous data."""
        frequency = np.arange(self.voltage_begin, self.voltage_end+self.voltage_delta, self.voltage_delta)
        sequence = self.generate_sequence()
        
        if not self.keep_data or np.any(frequency != self.frequency):
            self.frequency = frequency
            self.counts = np.zeros(frequency.shape)
            self.run_time = 0.0
        
        self.sequence = sequence
        self.keep_data = True # when job manager stops and starts the job, data should be kept. Only new submission should clear data.
    
    # Sequence - PulseGenerator
    def generate_sequence(self):
        return 200 * [ (['laser'],self.laser), ([],self.wait), (['microwave'],self.t_pi) ] + [([],10000)]
    
    def _run(self):
                
        try:
            self.state='run'
            self.apply_parameters()
            
            if self.run_time >= self.stop_time:
                self.state='idle'
                return
            
            # if pulsed, turn on sequence
            
            n = len(self.frequency)
            
            self.counter.configure(n, self.seconds_per_point, DutyCycle=0.8)
            time.sleep(0.5)

            while self.run_time < self.stop_time:
                start_time = time.time()
                if threading.currentThread().stop_request.isSet():
                    break
                counts = self.counter.run()
                self.run_time += time.time() - start_time
                self.counts += counts
                self.counts_matrix = np.vstack( (counts, self.counts_matrix[:-1,:]) )
                self.trait_property_changed('counts', self.counts)
                
            self.state='idle'

            self.counter.clear()
        except:
            logging.getLogger().exception('Error in Analog.')
            self.state = 'error'
        finally:
            return
    # fitting
    def _update_fit(self):
        if self.perform_fit:
            N = self.number_of_resonances 
            if N != 'auto':
                N = int(N)
            try:
                p, dp = fit_Analog(self.frequency, self.counts, threshold=self.threshold*0.01, number_of_resonances=N, model=self.fit_model)
            except Exception:
                logging.getLogger().debug('Analog fit failed.', exc_info=True)
                p = np.nan*np.empty(4)
        else:
            p = np.nan*np.empty(4)
        self.fit_parameters = p
        self.fit_frequencies = p[1::3]
        self.fit_line_width = p[2::3]
        N = len(p)/3
        contrast = np.empty(N)
        c = p[0]
        pp = p[1:].reshape((N, 3))
        for i, pi in enumerate(pp):
            a = pi[2]
            g = pi[1]
            if self.fit_model == 'gauss':
                A = abs(a)
            elif self.fit_model == 'lorentz':
                A = np.abs(a/(np.pi*g))
            if a > 0:
                contrast[i] = 100*A/(A+c)
            else:
                contrast[i] = 100*A/c
        self.fit_contrast = [round(i) for i in contrast]
    
    
    # plotting
    def _create_line_plot(self):
        line_data = ArrayPlotData(frequency=np.array((0.,1.)), counts=np.array((0.,0.)), fit=np.array((0.,0.))) 
        line_plot = Plot(line_data, padding=8, padding_left=64, padding_bottom=32)
        line_plot.plot(('frequency','counts'), style='line', color=scheme['data 1'], line_width=2)
        line_plot.bgcolor = scheme['background']
        line_plot.x_grid = None
        line_plot.y_grid = None
        line_plot.index_axis.title = 'Frequency [MHz]'
        line_plot.value_axis.title = 'Fluorescence counts'
        line_plot.value_range.low = 0.0
        line_plot.tools.append(PanTool(line_plot))
        line_plot.overlays.append(SimpleZoom(line_plot, enable_wheel=False))
        self.line_data = line_data
        self.line_plot = line_plot
    
    def _create_matrix_plot(self):
        matrix_data = ArrayPlotData(image=np.zeros((2,2)))
        matrix_plot = Plot(matrix_data, padding=8, padding_left=64, padding_bottom=32)
        matrix_plot.index_axis.title = 'Frequency [MHz]'
        matrix_plot.value_axis.title = 'line #'
        matrix_plot.img_plot('image',
                             xbounds=(self.frequency[0],self.frequency[-1]),
                             ybounds=(0,self.n_lines),
                             colormap=scheme['matrix']
                            )
        self.matrix_data = matrix_data
        self.matrix_plot = matrix_plot
    
    def _baseline_changed(self):
        if self.baseline:
            self.line_plot.value_range.low = 0.0
        else:
            self.line_plot.value_range.low = 'auto'
    
    def _perform_fit_changed(self,new):
        plot = self.line_plot
        if new:
            plot.plot(('frequency','fit'), style='line', color=scheme['fit 1'], name='fit')
        else:
            plot.delplot('fit')
        plot.request_redraw()
    
    def _update_line_data_index(self):
        self.line_data.set_data('frequency', self.frequency*1e-6)
        self.counts_matrix = self._counts_matrix_default()
    
    def _update_line_data_value(self):
        self.line_data.set_data('counts', self.counts)
    
    def _update_line_data_fit(self):
        if not np.isnan(self.fit_parameters[0]):
            if self.fit_model == 'gauss':
                self.line_data.set_data('fit', n_gaussians(*self.fit_parameters)(self.frequency))
            elif self.fit_model == 'lorentz':
                self.line_data.set_data('fit', n_lorentzians(*self.fit_parameters)(self.frequency))
            
    def _update_matrix_data_value(self):
        self.matrix_data.set_data('image', self.counts_matrix)
    
    def _update_matrix_data_index(self):
        if self.n_lines > self.counts_matrix.shape[0]:
            self.counts_matrix = np.vstack( (self.counts_matrix,np.zeros((self.n_lines-self.counts_matrix.shape[0], self.counts_matrix.shape[1]))) )
        else:
            self.counts_matrix = self.counts_matrix[:self.n_lines]
        self.matrix_plot.components[0].index.set_data((self.frequency.min()*1e-6, self.frequency.max()*1e-6),(0.0,float(self.n_lines)))
    
    # import/export
    def _import_button_fired(self):
        self.t_pi = self.clipboard.get_pi()
        self.power = self.clipboard.get_power()
    
    def _export_button_fired(self):
        try:
            self.clipboard.set_freq(self.fit_frequencies[self.peak])
        except IndexError as e:
            return
        
    # saving data
    
    def save_line_plot(self, filename):
        self.save_figure(self.line_plot, filename)
    
    def save_matrix_plot(self, filename):
        self.save_figure(self.matrix_plot, filename)
    
    def save_all(self, filename):
        self.save_line_plot(filename+'_Analog_Line_Plot.png')
        self.save_matrix_plot(filename+'_Analog_Matrix_Plot.png')
        self.save(filename+'_Analog.pys')
    
    def submit(self):
        """Submit the job to the JobManager."""
        self.keep_data = False
        ManagedJob.submit(self)
    
    traits_view = View(VGroup(HGroup(Item('submit_button',   show_label=False),
                                     Item('remove_button',   show_label=False),
                                     Item('priority', enabled_when='state != "run"'),
                                     Item('state', style='readonly'),
                                     Item('run_time', style='readonly', format_str='%i'),
                                     Item('stop_time'),
                                     ),
                              VGroup(HGroup(Item('power', width=-40, enabled_when='state != "run"'),
                                            Item('voltage_begin', width=-80, enabled_when='state != "run"'),
                                            Item('voltage_end', width=-80, enabled_when='state != "run"'),
                                            Item('voltage_delta', width=-80, enabled_when='state != "run"'),
                                            Item('import_button', show_label=False),
                                            Item('export_button', show_label=False),
                                            Item('peak', width=-40),
                                            ),
                                     HGroup(Item('seconds_per_point', width=-40, enabled_when='state != "run"'),
                                            Item('pulsed', enabled_when='state != "run"'),
                                            Item('laser', width=-50, enabled_when='state != "run"'),
                                            Item('wait', width=-50, enabled_when='state != "run"'),
                                            Item('t_pi', width=-50, enabled_when='state != "run"'),
                                            ),
                                     HGroup(Item('baseline'),
                                            Item('perform_fit'),
                                            Item('fit_model'),
                                            Item('number_of_resonances', width=-60),
                                            Item('threshold', width=-60),
                                            Item('n_lines', width=-60),
                                            ),
                                     HGroup(Item('fit_contrast', style='readonly'),
                                            Item('fit_line_width', style='readonly'),
                                            Item('fit_frequencies', style='readonly'),
                                            ),
                                     ),
                              VSplit(Item('line_plot', show_label=False, resizable=True),
                                     Item('matrix_plot', show_label=False, resizable=True),
                                     ),
                              ),
                       title='Analog In/Out',
                       width =895,
                       height=1000,
                       x=1025,
                       y=0,
                       buttons=[],
                       resizable=True,
                       handler=AnalogHandler
                      )

    get_set_items = ['frequency', 'counts', 'counts_matrix',
                     'fit_parameters', 'fit_contrast', 'fit_line_width', 'fit_frequencies',
                     'perform_fit', 'run_time',
                     'power', 'voltage_begin', 'voltage_end', 'voltage_delta',
                     'laser', 'wait', 'pulsed', 't_pi',
                     'seconds_per_point', 'stop_time', 'n_lines',
                     'number_of_resonances', 'threshold',
                     '__doc__']
