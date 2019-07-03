# nidaqscope
The general idea is to implement a scope based on a NIDAQ-card for 
slow (<1kHz) scope applications, which can be e.g. used for evaluating 
an atomic line or Faraday filter.

see e.g.:

Faraday Filtering on the Cs-D1-Line for Quantum Hybrid Systems
https://dx.doi.org/10.1109/LPT.2018.2871770

or

Na-Faraday rotation filtering: The optimal point
http://dx.doi.org/10.1038/srep06552

Lines should be averaged and/or stacked, sucht that with each step another
paramter can be detuned.

Stuttgart, July 2019

