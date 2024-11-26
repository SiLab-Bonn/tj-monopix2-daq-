#
# ------------------------------------------------------------
# Copyright (c) All rights reserved
# SiLab, Institute of Physics, University of Bonn
# ------------------------------------------------------------
#

from tjmonopix2.analysis import analysis
from tjmonopix2.scans.shift_and_inject import get_scan_loop_mask_steps, shift_and_inject
from tjmonopix2.system.scan_base import ScanBase
from tqdm import tqdm
from plotting_scurves import Plotting

scan_configuration = {
<<<<<<< Updated upstream
    'start_column': 225,
    'stop_column': 230,
    'start_row': 507,
=======
    'start_column': 480,
    'stop_column': 496,
    'start_row': 0,
>>>>>>> Stashed changes
    'stop_row': 512,

    'n_injections': 100,
    'VCAL_HIGH': 140,
    'VCAL_LOW_start': 139,
    'VCAL_LOW_stop': 1,
    'VCAL_LOW_step': -1
}

register_overrides = {
    'ITHR': 35,
    'IBIAS': 50,
    'VRESET': 143,
    'ICASN': 0,
    'VCASP': 93,
    "VCASC": 228,  # Default 228
    "IDB": 100,  # Default 100
    'ITUNE': 53,  # Default 53
    # 'MON_EN_VH': 0,
    # 'MON_EN_VL': 0,
    # 'OVR_EN_VH': 0,
    # 'OVR_EN_VL': 0,
    # Enable analog monitoring pixel
    'EN_PULSE_ANAMON_L': 1,
    'ANAMON_SFN_L': 0b0001,
    'ANAMON_SFP_L': 0b1000,
    'ANAMONIN_SFN1_L': 0b1000,
    'ANAMONIN_SFN2_L': 0b1000,
    'ANAMONIN_SFP_L': 0b1000,
    # Enable hitor
    'SEL_PULSE_EXT_CONF': 0
}


class ThresholdScan(ScanBase):
    scan_id = 'threshold_scan'

    def _configure(self, start_column=0, stop_column=512, start_row=0, stop_row=512, **_):
        self.chip.masks['enable'][start_column:stop_column, start_row:stop_row] = True
        self.chip.masks['injection'][start_column:stop_column, start_row:stop_row] = True
        self.chip.masks['tdac'][start_column:stop_column, start_row:stop_row] = 0b100

        #self.chip.masks['tdac'][start_column:stop_column, start_row:stop_row] = 0b100

        #self.chip.registers["VCASP"].write(40)

        #self.chip.registers["ICASN"].write(30)


        self.chip.masks.apply_disable_mask()
        self.chip.masks.update(force=True)

        for r in self.register_overrides:
            self.chip.registers[r].write(self.register_overrides[r])
        # Enable HITOR (active high) on all columns, all rows
        for i in range(512//16):
            self.chip._write_register(18+i, 0xffff)
            self.chip._write_register(50+i, 0xffff)

        self.daq.rx_channels['rx0']['DATA_DELAY'] = 14

        self.chip.registers["SEL_PULSE_EXT_CONF"].write(0)

    def _scan(self, n_injections=100, VCAL_HIGH=80, VCAL_LOW_start=80, VCAL_LOW_stop=40, VCAL_LOW_step=-1, **_):
        """
        Injects charges from VCAL_LOW_START to VCAL_LOW_STOP in steps of VCAL_LOW_STEP while keeping VCAL_HIGH constant.
        """

        self.chip.registers["VH"].write(VCAL_HIGH)
        vcal_low_range = range(VCAL_LOW_start, VCAL_LOW_stop, VCAL_LOW_step)

        pbar = tqdm(total=get_scan_loop_mask_steps(self) * len(vcal_low_range), unit='Mask steps')
        for scan_param_id, vcal_low in enumerate(vcal_low_range):
            self.chip.registers["VL"].write(vcal_low)

            self.store_scan_par_values(scan_param_id=scan_param_id, vcal_high=VCAL_HIGH, vcal_low=vcal_low)
            with self.readout(scan_param_id=scan_param_id):
                shift_and_inject(scan=self, n_injections=n_injections, pbar=pbar, scan_param_id=scan_param_id)
        pbar.close()
        self.log.success('Scan finished')

    def _analyze(self):
        with analysis.Analysis(raw_data_file=self.output_filename + '.h5', **self.configuration['bench']['analysis']) as a:
            a.analyze_data()

        if self.configuration['bench']['analysis']['create_pdf']:
            with Plotting(analyzed_data_file=a.analyzed_data_file) as p:
                p.create_standard_plots()


if __name__ == "__main__":
    with ThresholdScan(scan_config=scan_configuration, register_overrides=register_overrides) as scan:
        scan.start()
