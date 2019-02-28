% This function runs BEVT tracker

function [results] = BEVT_tracker(params)

%   setting parameters for local use.
search_area_scale   = params.search_area_scale;
output_sigma_factor = params.output_sigma_factor;
learning_rate       = params.learning_rate;
filter_max_area     = params.filter_max_area;
nScales             = params.number_of_scales;
scale_step          = params.scale_step;
interpolate_response = params.interpolate_response;

% settings for admm
admm_iterations = params.admm_iterations;
betha = params.betha;
mumax = params.mumax;

features    = params.t_features;
video_path  = params.video_path;
s_frames    = params.s_frames;
pos         = floor(params.init_pos);
target_sz   = floor(params.wsize);

visualization  = params.visualization;
num_frames     = params.no_fram;
init_target_sz = target_sz;

% settings for feature ratio to the feature-cell size
featureRatio = params.t_global.cell_size;
search_area = prod(init_target_sz / featureRatio * search_area_scale);

% settings for response map
background_color = params.background_color;
response_map_lr = params.response_map_lr;
use_response_map = params.use_response_map;
show_response_map = params.show_response_map;
save_response_map = params.save_response_map;
savedir = params.savedir;

% when the number of cells are small, choose a smaller cell size
if isfield(params.t_global, 'cell_selection_thresh')
    if search_area < params.t_global.cell_selection_thresh * filter_max_area
        params.t_global.cell_size = min(featureRatio, max(1, ceil(sqrt(prod(init_target_sz * search_area_scale)/(params.t_global.cell_selection_thresh * filter_max_area)))));
        
        featureRatio = params.t_global.cell_size;
        search_area = prod(init_target_sz / featureRatio * search_area_scale);
    end
end

global_feat_params = params.t_global;

if search_area > filter_max_area
    currentScaleFactor = sqrt(search_area / filter_max_area);
else
    currentScaleFactor = 1.0;
end

% target size at the initial scale
base_target_sz = target_sz / currentScaleFactor;

% window size, taking padding into account
switch params.search_area_shape
    case 'proportional'
        sz = floor( base_target_sz * search_area_scale);     % proportional area, same aspect ratio as the target
    case 'square'
        sz = repmat(sqrt(prod(base_target_sz * search_area_scale)), 1, 2); % square area, ignores the target aspect ratio
    case 'fix_padding'
        sz = base_target_sz + sqrt(prod(base_target_sz * search_area_scale) + (base_target_sz(1) - base_target_sz(2))/4) - sum(base_target_sz)/2; % const padding
    otherwise
        error('Unknown "params.search_area_shape". Must be ''proportional'', ''square'' or ''fix_padding''');
end

% set the size to exactly match the cell size
sz = round(sz / featureRatio*1.2) * featureRatio;
use_sz = floor(sz/featureRatio);

% construct the label function- correlation output, 2D gaussian function,
% with a peak located upon the target
output_sigma = sqrt(prod(floor(base_target_sz/featureRatio))) * output_sigma_factor;
rg           = circshift(-floor((use_sz(1)-1)/2):ceil((use_sz(1)-1)/2), [0 -floor((use_sz(1)-1)/2)]);
cg           = circshift(-floor((use_sz(2)-1)/2):ceil((use_sz(2)-1)/2), [0 -floor((use_sz(2)-1)/2)]);
[rs, cs]     = ndgrid( rg,cg);
y            = exp(-0.5 * (((rs.^2 + cs.^2) / output_sigma^2)));
yf           = fft2(y); %   FFT of y.

if interpolate_response == 1
    interp_sz = use_sz * featureRatio;
else
    interp_sz = use_sz;
end

% construct cosine window
cos_window = single(hann(use_sz(1))*hann(use_sz(2))');

% Calculate feature dimension
try
    im = imread([video_path '/img/' s_frames{1}]);
catch
    try
        im = imread(s_frames{1});
    catch
        %disp([video_path '/' s_frames{1}])
        im = imread([video_path '/' s_frames{1}]);
    end
end
if size(im,3) == 3
    if all(all(im(:,:,1) == im(:,:,2)))
        colorImage = false;
    else
        colorImage = true;
    end
else
    colorImage = false;
end

% compute feature dimensionality
feature_dim = 0;
for n = 1:length(features)
    
    if ~isfield(features{n}.fparams,'useForColor')
        features{n}.fparams.useForColor = true;
    end
    
    if ~isfield(features{n}.fparams,'useForGray')
        features{n}.fparams.useForGray = true;
    end
    
    if (features{n}.fparams.useForColor && colorImage) || (features{n}.fparams.useForGray && ~colorImage)
        feature_dim = feature_dim + features{n}.fparams.nDim;
    end
end

if size(im,3) > 1 && colorImage == false
    im = im(:,:,1);
end

if nScales > 0
    scale_exp = (-floor((nScales-1)/2):ceil((nScales-1)/2));
    scaleFactors = scale_step .^ scale_exp;
    min_scale_factor = scale_step ^ ceil(log(max(5 ./ sz)) / log(scale_step));
    max_scale_factor = scale_step ^ floor(log(min([size(im,1) size(im,2)] ./ base_target_sz)) / log(scale_step));
end

if interpolate_response >= 3
    % Pre-computes the grid that is used for socre optimization
    ky = circshift(-floor((use_sz(1) - 1)/2) : ceil((use_sz(1) - 1)/2), [1, -floor((use_sz(1) - 1)/2)]);
    kx = circshift(-floor((use_sz(2) - 1)/2) : ceil((use_sz(2) - 1)/2), [1, -floor((use_sz(2) - 1)/2)])';
    newton_iterations = params.newton_iterations;
end

% initialize the projection matrix (x,y,h,w)
rect_position = zeros(num_frames, 4);
time = 0;

% convolutional feature extraction 
indLayers = [37, 28, 19];   % The CNN layers Conv5-4, Conv4-4, and Conv3-4 in VGG Net
nweights  = [0.25, 0.5, 1]; % Weights for combining correlation filter responses
numLayers = length(indLayers);

% allocate memory for multi-scale tracking
multires_pixel_template = zeros(sz(1), sz(2), size(im,3), nScales, 'uint8');
layer_response = zeros(use_sz(1), use_sz(2), numLayers);
layer_responsef_padded= zeros(use_sz(1), use_sz(2), numLayers);
response = zeros(use_sz(1), use_sz(2), nScales);
responsef_padded= zeros(use_sz(1), use_sz(2), nScales);
small_filter_sz = floor(base_target_sz/featureRatio);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% spatial punishment %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% create weight window
ref_window_power = params.reg_window_power;
    
% normalization factor
reg_scale = 0.5 * base_target_sz/featureRatio;

% construct grid
wrg = -(use_sz(1)-1)/2:(use_sz(1)-1)/2;
wcg = -(use_sz(2)-1)/2:(use_sz(2)-1)/2;
[wrs, wcs] = ndgrid(wrg, wcg);

% construct the regularization window
reg_window = (params.reg_window_edge - params.reg_window_min) * (abs(wrs/reg_scale(1)).^ref_window_power + abs(wcs/reg_scale(2)).^ref_window_power) + params.reg_window_min;
[x_max,y_max] = size(reg_window);
reg_window = (params.reg_window_edge - params.reg_window_min) * (abs(wrs/reg_scale(1)).^2 + abs(wcs/reg_scale(2)).^2) + params.reg_window_min;
reg_window_size = size(reg_window);
param_reg_window = reg_window' * reg_window;

% precalculate h*parameter
mu = 1;
T = prod(use_sz);
for ii = 1:admm_iterations
    param_reg_window_plus = 2 * param_reg_window/T + mu * eye(use_sz);
    param_reg_window_h(:,:,ii) = inv(param_reg_window_plus);
    mu = min(betha * mu, mumax);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% spatial punishment %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% in order for the tracker to learn the first frame
flag_learning = 1;

loop_frame = 1;
for frame = 1:numel(s_frames)
    %load image
    try
        im = imread([video_path '/img/' s_frames{frame}]);
    catch
        try
            im = imread([s_frames{frame}]);
        catch
            im = imread([video_path '/' s_frames{frame}]);
        end
    end
    if size(im,3) > 1 && colorImage == false
        im = im(:,:,1);
    end
    
    tic();
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% position detection %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %do not estimate translation and scaling on the first frame, since we
    %just want to initialize the tracker there
    if frame > 1
        for scale_ind = 1:nScales
            multires_pixel_template(:,:,:,scale_ind) = ...
                get_pixels(im, pos, round(sz*currentScaleFactor*scaleFactors(scale_ind)), sz);
            feat_t = get_CNN_features(multires_pixel_template(:,:,:,scale_ind),cos_window,indLayers);
            for ii = 1:numLayers
                xtf = fft2(feat_t{ii});
                responsef = permute(sum(bsxfun(@times, conj(model_g_f{ii}), xtf), 3), [1 2 4 3]);
                layer_responsef_padded(:,:,ii) = resizeDFT2(responsef, interp_sz);
            end
            responsef_padded(:,:,scale_ind)= sum(bsxfun(@times, layer_responsef_padded, permute(nweights,[3 1 2])), 3);
            response(:,:,scale_ind) = ifft2(responsef_padded(:,:,scale_ind), 'symmetric');
        end
        
        % find maximum peak
        if interpolate_response == 3
            error('Invalid parameter value for interpolate_response');
        elseif interpolate_response == 4
            [disp_row, disp_col, sind] = resp_newton(response, responsef_padded, newton_iterations, ky, kx, use_sz); % 思考下这里的kx以及ky是否需要修改一下，改成适合CNN的版本
        else
            [row, col, sind] = ind2sub(size(response), find(response == max(response(:)), 1));
            disp_row = mod(row - 1 + floor((interp_sz(1)-1)/2), interp_sz(1)) - floor((interp_sz(1)-1)/2);
            disp_col = mod(col - 1 + floor((interp_sz(2)-1)/2), interp_sz(2)) - floor((interp_sz(2)-1)/2);
        end
        
        if use_response_map == 1
            target_response = fftshift(response(:,:,sind));
            if show_response_map == 1
                [x_max,y_max] = size(response(:,:,sind));
                x = 1:x_max;
                y = 1:y_max;
                max_draw_response = max(target_response(:));
                [id_ymax,id_xmax]=find(target_response==max_draw_response);
                % draw response figure
                figure(response_fig_handle);
                hold off
                mesh(y,x,target_response);
                hold on
                plot3(id_xmax,id_ymax,max_draw_response,'k.','markersize',20)   %标记一个黑色的圆点
                text(id_xmax,id_ymax,max_draw_response,['  z=',num2str(max_draw_response)]);
            end
            if save_response_map == 1
                response_to_save(:,:,frame) = target_response;
            end
        end
            
        % calculate translation
        switch interpolate_response
            case 0
                translation_vec = round([disp_row, disp_col] * featureRatio * currentScaleFactor * scaleFactors(sind));
            case 1
                translation_vec = round([disp_row, disp_col] * currentScaleFactor * scaleFactors(sind));
            case 2
                translation_vec = round([disp_row, disp_col] * scaleFactors(sind));
            case 3
                translation_vec = round([disp_row, disp_col] * featureRatio * currentScaleFactor * scaleFactors(sind));
            case 4
                translation_vec = round([disp_row, disp_col] * featureRatio * currentScaleFactor * scaleFactors(sind));
        end
        
        % set the scale
        currentScaleFactor = currentScaleFactor * scaleFactors(sind);
        % adjust to make sure we are not to large or to small
        if currentScaleFactor < min_scale_factor
            currentScaleFactor = min_scale_factor;
        elseif currentScaleFactor > max_scale_factor
            currentScaleFactor = max_scale_factor;
        end
        
        % update position
        old_pos = pos;
        pos = pos + translation_vec;
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% position detection %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    target_sz = floor(base_target_sz * currentScaleFactor);

    % response map verifyer
    if use_response_map == 1
        if(frame == 1)
            score = 0;
            if save_response_map == 1
                score_to_save(frame) = score;
            end
        else
            score = exp(-norm(abs(model_response - target_response)));
            if(score <= 0.20)
                flag_learning = 0;
            else
                flag_learning = 1;
                if (score >= 0.50)
                    learning_rate = 0.019;
                else
                    learning_rate = 0.013;   
                end
                model_response = (1-response_map_lr) * model_response + response_map_lr * target_response;
            end
            if save_response_map == 1
                score_to_save(frame) = score;
            end
        end
    end

    if flag_learning == 1
        % extract training sample image region
        pixels = get_pixels(im,pos,round(sz*currentScaleFactor),sz);

        % extract features and do windowing
        feat = get_CNN_features(pixels,cos_window,indLayers);
        for ii = 1:numLayers
            xf = fft2(feat{ii});
            if (frame == 1)
                model_xf{ii} = xf;
            else
                model_xf{ii} = ((1 - learning_rate) * model_xf{ii}) + (learning_rate * xf);
            end

            % filter update of every Layer model
            g_f = single(zeros(size(xf)));
            h_f = g_f;
            l_f = g_f;
            mu  = 1;
            i = 1;
            
            size_gf = size(g_f);

            S_xx = sum(conj(model_xf{ii}) .* model_xf{ii}, 3);
            %   ADMM
            while (i <= admm_iterations)
                %   solve for G- please refer to the paper for more details
                B = S_xx + (T * mu);
                S_lx = sum(conj(model_xf{ii}) .* l_f, 3);
                S_hx = sum(conj(model_xf{ii}) .* h_f, 3);
                g_f = (((1/(T*mu)) * bsxfun(@times, yf, model_xf{ii})) - ((1/mu) * l_f) + h_f) - ...
                    bsxfun(@rdivide,(((1/(T*mu)) * bsxfun(@times, model_xf{ii}, (S_xx .* yf))) - ((1/mu) * bsxfun(@times, model_xf{ii}, S_lx)) + (bsxfun(@times, model_xf{ii}, S_hx))), B);

                %   solve for H
                
                %%% with spatial punishment %%%
                param_second = ifft2((mu*g_f) + l_f);
                h = zeros(use_sz(1),use_sz(2),size_gf(3));
                for iii = 1:size_gf(3)
                    h(:,:,iii) = param_reg_window_h(:,:,i) * param_second(:,:,iii);
                end
                [sx,sy,h] = get_subwindow_no_window(h, floor(use_sz/2) , small_filter_sz);
                t = single(zeros(use_sz(1), use_sz(2), size(h,3)));
                t(sx,sy,:) = h;
                h_f = fft2(t);
                %%% with spatial punishment %%%
                
                %%% no spatial punishment %%%
                %   solve for H
%                 h = (T/((mu*T)+ params.admm_lambda))* ifft2((mu*g_f) + l_f);
%                 [sx,sy,h] = get_subwindow_no_window(h, floor(use_sz/2) , small_filter_sz);
%                 t = single(zeros(use_sz(1), use_sz(2), size(h,3)));
%                 t(sx,sy,:) = h;
%                 h_f = fft2(t);
                %%% no spatial punishment %%%

                %   update L
                l_f = l_f + (mu * (g_f - h_f));

                %   update mu- betha = 10.
                mu = min(betha * mu, mumax);
                i = i+1;
            end
            model_g_f{ii} = g_f;
        end
        
        if use_response_map == 1
            if(frame == 1)
                % initialize ideal response
                for ii = 1:numLayers
                    xtf = fft2(feat{ii});
                    responsef = permute(sum(bsxfun(@times, conj(model_g_f{ii}), xtf), 3), [1 2 4 3]);
                    layer_responsef_padded(:,:,ii) = resizeDFT2(responsef, interp_sz);
                end
                model_responsef_padded(:,:)= sum(bsxfun(@times, layer_responsef_padded, permute(nweights,[3 1 2])), 3);
                model_response(:,:) = ifft2(model_responsef_padded(:,:), 'symmetric');
                model_response = fftshift(model_response);
                if show_response_map == 1
                    [x_max,y_max] = size(model_response);
                    x = 1:x_max;
                    y = 1:y_max;
                    max_model_response = max(model_response(:));
                    [id_ymax,id_xmax]=find(model_response==max_model_response);
                    % draw response figure
                    response_fig_handle = figure('Name','Reseponse');
                    mesh(y,x,model_response);
                    hold on
                    plot3(id_xmax,id_ymax,max_model_response,'k.','markersize',20)   %标记一个黑色的圆点
                    text(id_xmax,id_ymax,max_model_response,['  z=',num2str(max_model_response)]);
                end
                
                if save_response_map == 1
                    mkdir(savedir);
                    response_to_save(:,:,frame) = model_response;
                end
            end
        end
    end

    time = time + toc();
    %save position and calculate FPS
    rect_position(loop_frame,:) = [pos([2,1]) - floor(target_sz([2,1])/2), target_sz([2,1])];
    
    %visualization
    if visualization == 1
        rect_position_vis = [pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])];
        im_to_show = double(im)/255;
        if size(im_to_show,3) == 1
            im_to_show = repmat(im_to_show, [1 1 3]);
        end
        if frame == 1
            fig_handle = figure('Name', 'Tracking');
            imagesc(im_to_show);
            hold on;
            rectangle('Position',rect_position_vis, 'EdgeColor','g', 'LineWidth',2);
            text(10, 10, int2str(frame), 'color', [0 1 1]);
            hold off;
            axis off;axis image;set(gca, 'Units', 'normalized', 'Position', [0 0 1 1])
        else
            figure(fig_handle);
            imagesc(im_to_show);
            hold on;
            rectangle('Position',rect_position_vis, 'EdgeColor','g', 'LineWidth',2);
            text(20, 30, ['# Frame : ' int2str(loop_frame) ' / ' int2str(num_frames)], 'color', [1 0 0], 'BackgroundColor', [1 1 1], 'fontsize', 16);
            text(20, 100, ['FPS : ' num2str(1/(time/loop_frame))], 'color', [1 0 0], 'BackgroundColor', [1 1 1], 'fontsize', 16);
            text(20, 170, ['SCORE:' num2str(score)], 'color',[1,0,0],'BackgroundColor', [1 1 1], 'fontsize', 16);
            
            
            if background_color == 1
                resp_sz = round(sz*currentScaleFactor*scaleFactors(scale_ind));
                xs = floor(old_pos(2)) + (1:resp_sz(2)) - floor(resp_sz(2)/2);
                ys = floor(old_pos(1)) + (1:resp_sz(1)) - floor(resp_sz(1)/2);
                sc_ind = floor((nScales - 1)/2) + 1;
                resp_handle = imagesc(xs, ys, fftshift(response(:,:,sc_ind))); colormap hsv;
                alpha(resp_handle, 0.2);
            end
            hold off;
        end
        drawnow
    end
    loop_frame = loop_frame + 1;
end

%   save resutls.
if use_response_map == 1
    if save_response_map == 1
        save([savedir,params.seq_name,'.mat'],'response_to_save');
        save([savedir,params.seq_name,'.mat'],'score_to_save','-append');
    end
end
close all
fps = loop_frame / time;
results.type = 'rect';
results.res = rect_position;
results.fps = fps;
