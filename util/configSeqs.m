function seqs=configSeqs

seqUAV123_10fps_car14 = {struct('name','car14','path','.\seq\car14','startFrame',1,'endFrame',443,'nz',6,'ext','jpg','init_rect',[0,0,0,0])};
seqUAV123_10fps_group2_2 = {struct('name','group2_2','path','.\seq\group2\','startFrame',303,'endFrame',591,'nz',6,'ext','jpg','init_rect',[0,0,0,0])};
seqUAV123_10fps_wakeboard4 = {struct('name','wakeboard4','path','.\seq\wakeboard4\','startFrame',1,'endFrame',233,'nz',6,'ext','jpg','init_rect',[0,0,0,0])};

seqs = seqUAV123_10fps_group2_2; %seqUAV123, seqUAV123_10fps, seqUAV20L, TC128, OTB100
