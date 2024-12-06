set_property SRC_FILE_INFO {cfile:/home/rasmus/git/tj-monopix2-daq/firmware/src/bdaq53_kx2.xdc rfile:../firmware/src/bdaq53_kx2.xdc id:1} [current_design]
set_property SRC_FILE_INFO {cfile:/home/rasmus/git/tj-monopix2-daq/firmware/src/SiTCP.xdc rfile:../firmware/src/SiTCP.xdc id:2} [current_design]
set_property src_info {type:XDC file:1 line:36 export:INPUT save:INPUT read:READ} [current_design]
set_max_delay -datapath_only -from [get_clocks CLK125PLLTX] -to [get_ports {rgmii_txd[*]}] 4.000
set_property src_info {type:XDC file:1 line:37 export:INPUT save:INPUT read:READ} [current_design]
set_max_delay -datapath_only -from [get_clocks CLK125PLLTX] -to [get_ports rgmii_tx_ctl] 4.000
set_property src_info {type:XDC file:1 line:38 export:INPUT save:INPUT read:READ} [current_design]
set_max_delay -datapath_only -from [get_clocks CLK125PLLTX90] -to [get_ports rgmii_txc] 4.000
set_property src_info {type:XDC file:2 line:3 export:INPUT save:INPUT read:READ} [current_design]
set_max_delay -datapath_only -from [get_pins -hier -filter {name =~ */GMII_RXBUF/cmpWrAddr*/C}] -to [get_pins -hier -filter {name =~ */GMII_RXBUF/smpWrStatusAddr*/D}] 5.500
set_property src_info {type:XDC file:2 line:4 export:INPUT save:INPUT read:READ} [current_design]
set_max_delay -datapath_only -from [get_pins -hier -filter {name =~ */GMII_TXBUF/orRdAct*/C}] -to [get_pins -hier -filter {name =~ */GMII_TXBUF/irRdAct*/D}] 5.500
set_property src_info {type:XDC file:2 line:5 export:INPUT save:INPUT read:READ} [current_design]
set_max_delay -datapath_only -from [get_pins -hier -filter {name =~ */GMII_TXBUF/muxEndTgl/C}] -to [get_pins -hier -filter {name =~ */GMII_TXBUF/rsmpMuxTrnsEnd*/D}] 5.500
set_property src_info {type:XDC file:2 line:7 export:INPUT save:INPUT read:READ} [current_design]
set_max_delay -datapath_only -from [get_pins -hier -filter {name =~ */SiTCP_INT_REG/regX10Data*/C}] -to [get_pins -hier -filter {name =~ */GMII_RXCNT/irMacFlowEnb/D}] 5.500
set_property src_info {type:XDC file:2 line:8 export:INPUT save:INPUT read:READ} [current_design]
set_max_delay -datapath_only -from [get_pins -hier -filter {name =~ */SiTCP_INT_REG/regX12Data*/C}] -to [get_pins -hier -filter {name =~ */GMII_RXCNT/muxMyMac*/D}] 5.500
set_property src_info {type:XDC file:2 line:9 export:INPUT save:INPUT read:READ} [current_design]
set_max_delay -datapath_only -from [get_pins -hier -filter {name =~ */SiTCP_INT_REG/regX13Data*/C}] -to [get_pins -hier -filter {name =~ */GMII_RXCNT/muxMyMac*/D}] 5.500
set_property src_info {type:XDC file:2 line:10 export:INPUT save:INPUT read:READ} [current_design]
set_max_delay -datapath_only -from [get_pins -hier -filter {name =~ */SiTCP_INT_REG/regX14Data*/C}] -to [get_pins -hier -filter {name =~ */GMII_RXCNT/muxMyMac*/D}] 5.500
set_property src_info {type:XDC file:2 line:11 export:INPUT save:INPUT read:READ} [current_design]
set_max_delay -datapath_only -from [get_pins -hier -filter {name =~ */SiTCP_INT_REG/regX15Data*/C}] -to [get_pins -hier -filter {name =~ */GMII_RXCNT/muxMyMac*/D}] 5.500
set_property src_info {type:XDC file:2 line:12 export:INPUT save:INPUT read:READ} [current_design]
set_max_delay -datapath_only -from [get_pins -hier -filter {name =~ */SiTCP_INT_REG/regX16Data*/C}] -to [get_pins -hier -filter {name =~ */GMII_RXCNT/muxMyMac*/D}] 5.500
set_property src_info {type:XDC file:2 line:13 export:INPUT save:INPUT read:READ} [current_design]
set_max_delay -datapath_only -from [get_pins -hier -filter {name =~ */SiTCP_INT_REG/regX17Data*/C}] -to [get_pins -hier -filter {name =~ */GMII_RXCNT/muxMyMac*/D}] 5.500
set_property src_info {type:XDC file:2 line:14 export:INPUT save:INPUT read:READ} [current_design]
set_max_delay -datapath_only -from [get_pins -hier -filter {name =~ */SiTCP_INT_REG/regX18Data*/C}] -to [get_pins -hier -filter {name =~ */GMII_RXCNT/muxMyIp*/D}] 5.500
set_property src_info {type:XDC file:2 line:15 export:INPUT save:INPUT read:READ} [current_design]
set_max_delay -datapath_only -from [get_pins -hier -filter {name =~ */SiTCP_INT_REG/regX19Data*/C}] -to [get_pins -hier -filter {name =~ */GMII_RXCNT/muxMyIp*/D}] 5.500
set_property src_info {type:XDC file:2 line:16 export:INPUT save:INPUT read:READ} [current_design]
set_max_delay -datapath_only -from [get_pins -hier -filter {name =~ */SiTCP_INT_REG/regX1AData*/C}] -to [get_pins -hier -filter {name =~ */GMII_RXCNT/muxMyIp*/D}] 5.500
set_property src_info {type:XDC file:2 line:17 export:INPUT save:INPUT read:READ} [current_design]
set_max_delay -datapath_only -from [get_pins -hier -filter {name =~ */SiTCP_INT_REG/regX1BData*/C}] -to [get_pins -hier -filter {name =~ */GMII_RXCNT/muxMyIp*/D}] 5.500
set_property src_info {type:XDC file:2 line:19 export:INPUT save:INPUT read:READ} [current_design]
set_max_delay -datapath_only -from [get_pins -hier -filter {name =~ */GMII_TXBUF/dlyBank0LastWrAddr*/C}] -to [get_pins -hier -filter {name =~ */GMII_TXBUF/rsmpBank0LastWrAddr*/D}] 5.500
set_property src_info {type:XDC file:2 line:20 export:INPUT save:INPUT read:READ} [current_design]
set_max_delay -datapath_only -from [get_pins -hier -filter {name =~ */GMII_TXBUF/dlyBank1LastWrAddr*/C}] -to [get_pins -hier -filter {name =~ */GMII_TXBUF/rsmpBank1LastWrAddr*/D}] 5.500
set_property src_info {type:XDC file:2 line:21 export:INPUT save:INPUT read:READ} [current_design]
set_max_delay -datapath_only -from [get_pins -hier -filter {name =~ */GMII_TXBUF/memRdReq*/C}] -to [get_pins -hier -filter {name =~ */GMII_TXBUF/irMemRdReq*/D}] 5.500
set_property src_info {type:XDC file:2 line:23 export:INPUT save:INPUT read:READ} [current_design]
set_max_delay -datapath_only -from [get_pins -hier -filter {name =~ */GMII_RXCNT/orMacTim*/C}] -to [get_pins -hier -filter {name =~ */GMII_TXCNT/irMacPauseTime*/D}] 5.500
set_property src_info {type:XDC file:2 line:24 export:INPUT save:INPUT read:READ} [current_design]
set_max_delay -datapath_only -from [get_pins -hier -filter {name =~ */GMII_RXCNT/orMacPause/C}] -to [get_pins -hier -filter {name =~ */GMII_TXCNT/irMacPauseExe_0/D}] 5.500
