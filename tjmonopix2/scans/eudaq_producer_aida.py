import argparse
import socket
import threading
import time
import pyeudaq
import os
from tjmonopix2.scans.scan_ext_trigger import ExtTriggerScan
from tjmonopix2.system import logger


def host_reachable(host="192.168.10.23", port=24, timeout=20):
    try:
        socket.setdefaulttimeout(timeout)
        with socket.create_connection((host, port)):
            return True
    except (socket.timeout, socket.error):
        return False


class EudaqProducerAida(pyeudaq.Producer):
    scan_id = 'eudaq_scan' 

    def __init__(self, name, runctrl):
        pyeudaq.Producer.__init__(self, name, runctrl)
        self.log = logger.setup_derived_logger(self.__class__.__name__)
        self.is_running = 0
        self.scan = None
        self.conf = {}
        self.thread_scan = None
        self.reg_config = {}
        self.init_register_vals = {}
        self.BDAQBoardTimeout = 10

    def __del__(self):
         if self.is_running:
             self.scan.close()

    def DoInitialise(self):
        self.log.info("Initialization completed")

    def DoConfigure(self):
        self.log.info("Probing if power is up")
        if host_reachable("192.168.10.23", 24, self.BDAQBoardTimeout):
            try:
                self.log.info("Power is up")
                self.scan = ExtTriggerScan() 
                self.scan.init()    
            except Exception as e:
                raise e
            self.SetStatusTag("TriggerN", "0")
            self.log.info("Initialization completed")
        else:
            self.log.error("Initialization failed")
            raise RuntimeError("BDAQ board unreachable")

        eudaqConfig = self.GetConfiguration() 

        self.conf["start_column"] = int(eudaqConfig.Get("start_column", "0"))
        self.conf["stop_column"] = int(eudaqConfig.Get("stop_column", "512"))
        self.conf["start_row"] = int(eudaqConfig.Get("start_row", "0")) 
        self.conf["stop_row"] = int(eudaqConfig.Get("stop_row", "512"))
        self.conf["max_triggers"] = int(eudaqConfig.Get("max_triggers", "1000"))
        self.conf["run_nmb_zfill"] = int(eudaqConfig.Get("run_nmb_zfill", "6"))

        if eudaqConfig.Get("scan_timeout").lower() == "false":
            self.conf["scan_timeout"] = False
        elif eudaqConfig.Get("scan_timeout").lower() == "true":
            self.conf["scan_timeout"] = True

        if eudaqConfig.Get("chip_config_file").lower() == "none":
            self.conf["chip_config_file"] = None

        configurable_regs = ['VL', 'VH', 'ITHR', 'IBIAS', 'VCASP', 'ICASN', 'VRESET', 'VCLIP', 'IDB', 'IDEL', 'VCASC']
        for reg in configurable_regs:
            try:
                self.reg_config[reg] = eudaqConfig.Get(reg, None)
            except:
                pass

        self.scan.scan_config = self.conf
        print(self.reg_config)
        try:
            self.scan.configure()
            self.log.info("Configuration completed")
        except Exception as e:
            raise e

        for reg in self.reg_config.keys():            
            reg_val = self.reg_config[reg]
            reg_val = reg_val.replace(',', '.')
            if reg_val:
                reg_val_int = int(float(reg_val))
                reg_val_float = float(reg_val)

                if (reg_val_float - reg_val_int) > 0.001:
                    print('Contains, ', reg_val, ' in ', reg)
                    self.current_scan_register = reg

                print('After, repl ', reg_val, ' in ', reg)
                self.init_register_vals[reg] = self.scan.chip.registers[reg].read()

                if reg_val:
                    self.scan.chip.registers[reg].write(int(float(reg_val)))

        self.scan.chip.registers['SEL_PULSE_EXT_CONF'].write(0)

    def DoStatus(self):
        if self.scan is not None and self.scan.last_exception is not None:
            exception = self.scan.last_exception[1]
            self.scan.last_exception = None  # Clear exception
            raise exception
        #self.SetStatusTag('DataEventN'  ,'%d'%self.idev)


    def DoStartRun(self):
        try:
            self.scan.fifo_readout   # check if already configured
        except AttributeError:
            self.DoInitialise()
            self.DoConfigure()
        if not self.scan.scan_config["max_triggers"]:
            self.scan.daq.configure_tlu_module(max_triggers=False, aidamode=True)

        self.is_running = 1
        self.thread_scan = threading.Thread(target=self.scan.scan)
        self.thread_trigger = threading.Thread(target=self.SendTriggerNumber)
        self.thread_scan.start()
        self.thread_trigger.start()

    def DoStopRun(self):
        if self.is_running: 
            self.is_running = 0                    
            self.scan.stop_scan.set()
            self.thread_scan.join()
            self.thread_trigger.join()

            # rename output file to include the run number in the file name
            # much more convenient when doing analysis
            origFile = self.scan.output_filename + '.h5'

            self.scan.close()

            run_number = self.GetRunNumber()
            orig_base = os.path.basename(origFile)
            orig_dir = os.path.dirname(origFile)
            newFile = f'{orig_dir}/run{str(run_number).zfill(self.conf["run_nmb_zfill"])}_{orig_base}'
            os.rename(origFile, newFile)
            
            self.scan = None
            self.SetStatusTag("TriggerN", "0")
            self.log.info("Scan was stopped")

    def DoReset(self):
        if self.is_running:
            self.is_running=0
            self.scan.stop_scan.set()
            self.thread_scan.join()
            self.thread_trigger.join()
            self.SetStatusTag("TriggerN", "0")
        if self.scan is not None:
            self.scan.close()
        self.scan = None
        self.log.info("Reset completed")

    def DoTerminate(self):
        if self.is_running:
            self.is_running = 0
            self.scan.stop_scan.set()
            self.thread_scan.join()
            self.thread_trigger.join()
            self.scan.close()
        self.scan = None
        self.log.info("Terminated")

    def SendTriggerNumber(self):
        while self.is_running:
            self.SetStatusTag("TriggerN", str(self.scan.daq.get_trigger_counter()))
            time.sleep(1)


if __name__ == '__main__':
    # Parse program arguments
    parser = argparse.ArgumentParser(prog='EudaqProducerAida',
                                     description='Eudaq Producer Aida',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-r', metavar='address',
                        help='Destination address',
                        default='tcp://localhost:44000')
    parser.add_argument('-n', metavar='name',
                        help='Producer name',
                        default='EudaqProducerAida')

    args = parser.parse_args()

    producer = EudaqProducerAida(args.n, args.r)
    print(f'producer {args.n} connecting to runcontrol in {args.r}')
    producer.Connect()
    time.sleep(2)
    while producer.IsConnected():
        time.sleep(1)
