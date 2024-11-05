#
# ------------------------------------------------------------
# Copyright (c) All rights reserved
# SiLab, Institute of Physics, University of Bonn
# ------------------------------------------------------------
#

from basil.HL.RegisterHardwareLayer import RegisterHardwareLayer


class cmd(RegisterHardwareLayer):
    '''Implement RD53B command encoder configuration and timing interface driver.
    '''

    _registers = {'RESET': {'descr': {'addr': 0, 'size': 8, 'properties': ['writeonly']}},
                  'VERSION': {'descr': {'addr': 0, 'size': 8, 'properties': ['ro']}},
                  'START': {'descr': {'addr': 1, 'size': 1, 'offset': 7, 'properties': ['writeonly']}},
                  'READY': {'descr': {'addr': 2, 'size': 1, 'offset': 0, 'properties': ['ro']}},
                  'SYNCING': {'descr': {'addr': 2, 'size': 1, 'offset': 1, 'properties': ['ro']}},
                  'EXT_START_EN': {'descr': {'addr': 2, 'size': 1, 'offset': 2, 'properties': ['rw']}},
                  'EXT_TRIGGER_EN': {'descr': {'addr': 2, 'size': 1, 'offset': 3, 'properties': ['rw']}},
                  'OUTPUT_EN': {'descr': {'addr': 2, 'size': 1, 'offset': 4, 'properties': ['rw']}},
                  'SIZE': {'descr': {'addr': 3, 'size': 16}},
                  'REPETITIONS': {'descr': {'addr': 5, 'size': 16}},
                  'MEM_BYTES': {'descr': {'addr': 7, 'size': 16, 'properties': ['ro']}},
                  'AZ_VETO_CYCLES': {'descr': {'addr': 9, 'size': 16}},
                  'BYPASS_MODE_RESET': {'descr': {'addr': 11, 'size': 1, 'offset': 0, 'properties': ['wo']}},
                  'BYPASS_CDR': {'descr': {'addr': 11, 'size': 1, 'offset': 1, 'properties': ['wo']}},
                  'AUTO_SYNC': {'descr': {'addr': 11, 'size': 1, 'offset': 2, 'properties': ['wr']}}
                  }

    _require_version = "==2"

    def __init__(self, intf, conf):
        super(cmd, self).__init__(intf, conf)
        self._mem_offset = 16   # In bytes

    def init(self):
        super(cmd, self).init()
        self._mem_size = self.get_mem_size()

    def get_mem_size(self):
        return self.MEM_BYTES

    def get_cmd_size(self):
        return self.SIZE

    def reset(self):
        self.RESET = 0

    def start(self):
        self.START = 0

    def set_size(self, value):
        ''' CMD buffer size '''
        self.SIZE = value

    def get_size(self):
        ''' CMD buffer size '''
        return self.SIZE

    def set_repetitions(self, value):
        ''' CMD repetitions '''
        self.REPETITIONS = value

    def get_repetitions(self):
        ''' CMD repetitions '''
        return self.REPETITIONS

    def set_ext_trigger(self, ext_trigger_mode):
        ''' external trigger input enable '''
        self.EXT_TRIGGER_EN = ext_trigger_mode

    def get_ext_trigger(self):
        ''' external trigger input enable '''
        return self.EXT_TRIGGER_EN

    def set_ext_start(self, ext_start_mode):
        ''' external start input enable '''
        self.EXT_START_EN = ext_start_mode

    def get_ext_start(self):
        ''' external start input enable '''
        return self.EXT_START_EN

    def set_output_en(self, value):
        ''' CMD output driver. False=high impedance '''
        self.OUTPUT_EN = value

    def set_auto_sync(self, value):
        ''' Enables automatic sending of sync commands to prevent ITkPixV1 like chips from unlocking '''
        self.AUTO_SYNC = value

    def get_auto_sync(self):
        ''' Gets the status of the AUTO_SYNC register to enable automatic sending of sync commands to prevent ITkPixV1 like chips from unlocking '''
        return self.AUTO_SYNC

    # TODO: Bypass can be deleted
    def set_bypass_cdr(self, value):
        ''' CDR bypass mode (KC705+FMC_LPC). Enables the output drivers and sends cmd and serializer clock to the chip '''
        self.BYPASS_CDR = value

    def set_bypass_reset(self, value):
        ''' CDR bypass mode (KC705+FMC_LPC). Sends external CDR reset to the chip '''
        self.BYPASS_MODE_RESET = value

    def is_done(self):
        return self.READY
   
   # TODO: AZ can be deleted
    def get_az_veto_cycles(self):
        ''' Veto clock cycles in 1/160 MHz during AZ '''
        return self.AZ_VETO_CYCLES

    def set_az_veto_cycles(self, value):
        ''' Veto clock cycles in 1/160 MHz during AZ '''
        self.AZ_VETO_CYCLES = value

    def set_data(self, data, addr=0):
        if self._mem_size < len(data):
            raise ValueError('Size of data (%d bytes) is too big for memory (%d bytes)' % (len(data), self._mem_size))
        self._intf.write(self._conf['base_addr'] + self._mem_offset + addr, data)

    def get_data(self, size=None, addr=0):
        if size and self._mem_size < size:
            raise ValueError('Size is too big')
        if not size:
            return self._intf.read(self._conf['base_addr'] + self._mem_offset + addr, self._mem_size)
        else:
            return self._intf.read(self._conf['base_addr'] + self._mem_offset + addr, size)