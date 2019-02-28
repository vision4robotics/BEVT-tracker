%   This function runs the BEVT tracker on the video specified in 
%   "configSeqs".
%   This function borrowed from SRDCF, BACF paper. 
%   details of some parameters are not presented in the paper, you can
%   refer to SRDCF/CCOT/BACF paper for more details.

function results = run_BEVT(seq, video_path, lr)

% flag bit to set the function to be compatible with UAV123 dataset
benchmark_mode = 0;

% setup matconvnet
% this is CPU version
% if you want a GPU version, just set it to true and compile your
% matconvnet
global enableGPU;
enableGPU = false;

vl_setupnn();
vl_compilenn('enableGpu', false);

%   HOG feature parameters
hog_params.nDim   = 31;
params.video_path = video_path;
params.seq_name = seq.name;

%   Global feature parameters 
params.t_features = {
    ...struct('getFeature',@get_colorspace, 'fparams',grayscale_params),...  % Grayscale is not used as default
    struct('getFeature',@get_fhog,'fparams',hog_params),...
};
params.t_global.cell_size = 4;                  % Feature cell size
params.t_global.cell_selection_thresh = 0.75^2; % Threshold for reducing the cell size in low-resolution cases

%   Search region + extended background parameters
params.search_area_shape = 'square';    % the shape of the training/detection window: 'proportional', 'square' or 'fix_padding'
params.search_area_scale = 5;           % the size of the training/detection area proportional to the target size
params.filter_max_area   = 50^2;        % the size of the training/detection area in feature grid cells

%   Learning parameters
if(benchmark_mode)
    lr = 0.013;
end
params.learning_rate       = lr;        % learning rate
params.output_sigma_factor = 1/16;		% standard deviation of the desired correlation output (proportional to target)

%   Detection parameters
params.interpolate_response  = 4;        % correlation score interpolation strategy: 0 - off, 1 - feature grid, 2 - pixel grid, 4 - Newton's method
params.newton_iterations     = 50;           % number of Newton's iteration to maximize the detection scores
				% the weight of the standard (uniform) regularization, only used when params.use_reg_window == 0
%   Scale parameters
params.number_of_scales =  5;
params.scale_step       = 1.01;

%   size, position, frames initialization
params.wsize    = [seq.init_rect(1,4), seq.init_rect(1,3)];
params.init_pos = [seq.init_rect(1,2), seq.init_rect(1,1)] + floor(params.wsize/2);
params.s_frames = seq.s_frames;
params.no_fram  = seq.endFrame - seq.startFrame + 1;
params.seq_st_frame = seq.startFrame;
params.seq_en_frame = seq.endFrame;


%   ADMM parameters, # of iteration, and lambda- mu and betha are set in
%   the main function.
params.admm_iterations = 3;
params.betha = 20;
params.mumax = 10000;
%   ADMM parameter with no spatial punishment
params.admm_lambda = 0.01;

%   Debug and visualization
params.visualization = 1;

% response map
params.background_color = 0;
params.use_response_map = 1;
params.show_response_map = 0;
params.save_response_map = 0;
params.savedir = '.\reponse\';
params.response_map_lr = 0.010;

% Regularization window parameters
params.use_reg_window = 1;              % wather to use windowed regularization or not
params.reg_window_min = 0.1;			% the minimum value of the regularization window
params.reg_window_edge = 0.8;           % the impact of the spatial regularization (value at the target border), depends on the detection size and the feature dimensionality
params.reg_window_power = 3;            % the degree of the polynomial to use (e.g. 2 is a quadratic window)
params.reg_sparsity_threshold = 0.05;   % a relative threshold of which DFT coefficients that should be set to zero

%   Run the main function
results = BEVT_tracker(params);
