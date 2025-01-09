#
# ------------------------------------------------------------
# Copyright (c) All rights reserved
# SiLab, Institute of Physics, University of Bonn
# ------------------------------------------------------------
#

import time
import threading
from tqdm import tqdm

from tjmonopix2.analysis import analysis, plotting
from tjmonopix2.system.scan_base import ScanBase

scan_configuration = {
    'start_column': 0,
    'stop_column': 224,
    'start_row': 0,
    'stop_row': 512,

    'scan_timeout': False,    # Timeout for scan after which the scan will be stopped, in seconds; if False no limit on scan time
    'max_triggers': 1000000,  # Number of maximum received triggers after stopping readout, if False no limit on received trigger

    'tot_calib_file': None    # path to ToT calibration file for charge to e⁻ conversion, if None no conversion will be done
}


class ExtTriggerScan(ScanBase):
    scan_id = 'ext_trigger_scan'

    stop_scan = threading.Event()

    def _configure(self, scan_timeout=False, max_triggers=1000, start_column=0, stop_column=512, start_row=0, stop_row=512, **_):
        self.log.info('External trigger scan needs TLU running!')

        if scan_timeout and max_triggers:
            self.log.warning('You should only use one of the stop conditions at a time.')

        self.chip.masks['enable'][start_column:stop_column, start_row:stop_row] = True
        self.chip.masks.apply_disable_mask()
        self.chip.masks.update()
        
        # Disable column 484 and 485:
        dcols_enable = [0] * 16
        for c in range(start_column, stop_column):
            dcols_enable[c // 32] |= (1 << ((c >> 1) & 15))
        for c in [484, 485]:  # List of disabled columns
            dcols_enable[c // 32] &= ~(1 << ((c >> 1) & 15))
        for i, v in enumerate(dcols_enable):
            self.chip._write_register(155 + i, v)  # EN_RO_CONF
            self.chip._write_register(171 + i, v)  # EN_BCID_CONF
            self.chip._write_register(187 + i, v)  # EN_RO_RST_CONF
            self.chip._write_register(203 + i, v)  # EN_FREEZE_CONF


        self.daq.configure_tlu_veto_pulse(veto_length=500)
        if max_triggers:
            # self.daq.configure_tlu_module(max_triggers=max_triggers)
            self.daq.configure_tlu_module(max_triggers=max_triggers, aidamode=True) # TODO: add in Testbench.yaml

    def _scan(self, scan_timeout=False, max_triggers=1000, **_):
        def timed_out():
            if scan_timeout:
                current_time = time.time()
                if current_time - start_time > scan_timeout:
                    self.log.info('Scan timeout was reached')
                    return True
            return False

        if scan_timeout:
            self.pbar = tqdm(total=scan_timeout, unit='')  # [s]
        elif max_triggers:
            self.pbar = tqdm(total=max_triggers, unit=' Triggers')
        start_time = time.time()

        with self.readout():
            self.stop_scan.clear()
            self.daq.enable_tlu_module()

            while not (self.stop_scan.is_set() or timed_out()):
                try:
                    if max_triggers:
                        triggers = self.daq.get_trigger_counter()
                    time.sleep(1)

                    # Update progress bar
                    try:
                        if scan_timeout:
                            self.pbar.update(1)
                        elif max_triggers:
                            self.pbar.update(self.daq.get_trigger_counter() - triggers)
                    except ValueError:
                        pass

                    # Stop scan if reached trigger limit
                    if max_triggers and triggers >= max_triggers:
                        self.stop_scan.set()
                        self.log.info('Trigger limit was reached: {0}'.format(max_triggers))

                except KeyboardInterrupt:  # React on keyboard interupt
                    self.stop_scan.set()
                    self.log.info('Scan was stopped due to keyboard interrupt')
        if scan_timeout or max_triggers:
            self.pbar.close()
        self.daq.disable_tlu_module()
        self.log.success('Scan finished')

    def _analyze(self):
        tot_calib_file = self.configuration['scan'].get('tot_calib_file', None)
        if tot_calib_file is not None:
            self.configuration['bench']['analysis']['cluster_hits'] = True

        with analysis.Analysis(raw_data_file=self.output_filename + '.h5', tot_calib_file=tot_calib_file, **self.configuration['bench']['analysis']) as a:
            a.analyze_data()

        if self.configuration['bench']['analysis']['create_pdf']:
            with plotting.Plotting(analyzed_data_file=a.analyzed_data_file) as p:
                p.create_standard_plots()


if __name__ == "__main__":
    with ExtTriggerScan(scan_config=scan_configuration) as scan:
        scan.start()
