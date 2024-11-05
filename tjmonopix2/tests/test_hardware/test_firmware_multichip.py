#
# ------------------------------------------------------------
# Copyright (c) All rights reserved
# SiLab, Institute of Physics, University of Bonn
# ------------------------------------------------------------
#

import unittest

from basil.utils.sim.utils import cocotb_compile_clean
from tjmonopix2.tests.test_hardware import utils as hw_utils


class FirmwareTest(unittest.TestCase):
    def setUp(self) -> None:
        conf = hw_utils.setup_cocotb()
        self.daq, self.duts = hw_utils.init_device(conf, chip_id=[0, 1, 2, 3])

    def tearDown(self) -> None:
        self.daq.close()
        cocotb_compile_clean()

    def test_register_rw(self) -> None:
        for dut_i, dut in enumerate(self.duts):
            print('Writing to chip with ID: %s' %dut.chip_id)
            dut.registers["VL"].write(38 + dut_i)
        for dut_i, dut in enumerate(self.duts):
            print('Reading chip with ID: %s' %dut.chip_id)
            reg = dut.registers["VL"].read()
            dut.write_command(dut.write_sync(write=False), repetitions=8)
            assert reg == 38 + dut_i

    def test_inj(self) -> None:
        for dut_i, dut in enumerate(self.duts):
            print('Enabling chip with ID: %s' %dut.chip_id)
            dut.registers["SEL_PULSE_EXT_CONF"].write(0)  # Use internal injection

            # Activate pixel (1, 128 + dut_i)
            dut.masks['enable'][1, 128 + dut_i] = True
            dut.masks['tdac'][1, 128 + dut_i] = 0b100
            dut.masks['injection'][1, 128 + dut_i] = True
            dut.masks.update()

        self.daq.reset_fifo()
        for dut_i, dut in enumerate(self.duts):
            print('Writing to chip with ID: %s' %dut.chip_id)
            dut.inject(PulseStartCnfg=0, PulseStopCnfg=8, repetitions=5 + dut_i)
            hw_utils.wait_for_sim(dut, repetitions=4*len(self.duts))

        data = self.daq["FIFO"].get_data()
        for dut_i, dut in enumerate(self.duts):
            print('Interpret data of chip with ID: %s' %dut.chip_id)
            hit, _ = dut.interpret_data(data)
            tot = (hit['te'] - hit['le']) & 0x7F
            assert hit['col'].tolist() == [1] * (5 + dut_i)
            assert hit['row'].tolist() == [128 + dut_i] * (5 + dut_i)
            assert tot.tolist() == [1] * (5 + dut_i)

if __name__ == "__main__":
    unittest.main()
