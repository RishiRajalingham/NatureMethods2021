function parse_session_data(meta_data)

exclude_first_image_per_trial = 1;
NIMG = 2;
NSPK = 1;
nsites = size(meta_data,1);
results_all = cell(nsites,1);

for site_i = 1:nsites
    fn = meta_data{site_i,1};
    fn_splits = split(fn, '/');
    fn_date = fn_splits{length(fn_splits)-1};
    fn_site_index_splits = split(fn_splits{length(fn_splits)}, '_');
    fn_site_index = fn_site_index_splits{length(fn_site_index_splits)};
    fn_site_index = str2double(strrep(fn_site_index, '.mat', ''));
    fn_mwk = sprintf('../raw_data_2020/Yolo%s.mat', fn_date);
    fprintf(1, 'parsing %s .. \n', fn_mwk);
    dat = load(fn_mwk);
    
    site_index_oi = fn_site_index;
    laser_duration = meta_data{site_i,2};
    prepad = laser_duration - 100;
    allTrials = dat.allTrials;
    results_all{site_i} = get_site_data(allTrials, site_index_oi, prepad);
    results_all{site_i}.laser_duration = laser_duration;
end

save('dat/session_data.mat', 'results_all');

    function results = get_site_data(allTrials, site_index_oi, prepad)
        
        data_summary = get_session_data_base(allTrials);
        
        t_site_index = data_summary.site_index_per_trial == site_index_oi;
        
        X_raster = cell(NSPK, NIMG);
      
        postpad = 1200;
        time_ax = -(prepad-1):1:postpad;
       
        for spike_i = 1:NSPK
            spkt = data_summary.spike_times{spike_i};

            for im_i = 1:NIMG
                im_t = data_summary.image_times{im_i}(t_site_index,:);
                im_t = im_t(~isnan(im_t));
                
                X_raster_im_i = [];
                for ii = 1:length(im_t)
                    start_t = im_t(ii) - prepad;
                    end_t = im_t(ii) + postpad;
                    t = (spkt >= start_t) & (spkt <= end_t);
                    x_curr = histcounts(spkt(t), start_t:1:end_t);
                    X_raster_im_i = cat(1, X_raster_im_i,x_curr);
                end
                X_raster{spike_i, im_i} = X_raster_im_i;
            end
            
%          
        end
        results.X_raster = X_raster;
        results.time_ax = time_ax;
%         
        [mu, sig] = get_laser_onset_time(allTrials, t_site_index);
        results.laser_onset_mu = mu;
        results.laser_onset_sig = sig;
    end
    
    function [mu, sig] = get_laser_onset_time(allTrials, trials_oi)
        laser_onset_idx = strcmp(allTrials.codenames(:), 'stim_fhc');
        fhc_t = allTrials.time(trials_oi,laser_onset_idx,:);
        fhc_t_all = fhc_t(isfinite(fhc_t));

        t_img = allTrials.time_display{2}(trials_oi,:);
        t_img_all = t_img(isfinite(t_img));
        t_img_all = sort(t_img_all);

        x = [];
        for ii = 1:length(fhc_t_all)
            x = cat(1, x, t_img_all(find(t_img_all > fhc_t_all(ii), 1, 'first')) - fhc_t_all(ii));
        end
        mu = nanmean(x);
        sig = nanstd(x);
    end


    function session_data_summary = get_session_data_base(allTrials)
        site_index_idx = strcmp(allTrials.codenames(:), 'ml_v0');
        site_index_per_trial = squeeze(allTrials.dat(:,site_index_idx,1));
        
        sync_index_idx = strcmp(allTrials.codenames(:), 'sync');
        sync_time_per_trial = squeeze(allTrials.time(:,sync_index_idx,1));
        
        imshown_idx = strcmp(allTrials.codenames(:), 'firstImageShown');
        imshown_t = squeeze(allTrials.time(:,imshown_idx,:));
        imshown_x = squeeze(allTrials.dat(:,imshown_idx,:));
        
        % parse spike data
        mu_idx = strcmp(allTrials.codenames_spikes(:), 'Trig_bak1');
        su_idx = strcmp(allTrials.codenames_spikes(:), 'Acc_bak2');
        spike_times = cell(2,1);
        spike_times{1} = allTrials.spiketimes{mu_idx};
        spike_times{2} = allTrials.spiketimes{su_idx};
        
        % parse image presentation data
        im1_idx = strcmp(allTrials.codenames_display(:), 'im1');
        im2_idx = strcmp(allTrials.codenames_display(:), 'im2');
        image_times = cell(2,1);
        image_times{1} = allTrials.time_display{im1_idx};
        image_times{2} = allTrials.time_display{im2_idx};
        
        if exclude_first_image_per_trial
            for tr_i = 1:size(imshown_x,1)
                % find time at which first_image_shown is triggered, and
                % set all images shown after this to be OK.
                t_imshown = (imshown_x(tr_i,:) == 1);
                t_first_image_shown = nanmin(imshown_t(tr_i,t_imshown));
                if ~isempty(t_first_image_shown)
                    for im_idx = 1:2
                        t = image_times{im_idx}(tr_i,:) < t_first_image_shown;
                        image_times{im_idx}(tr_i,t) = nan;
                    end
                else
                    % if first_image_shown is never triggered, the first
                    % image was broken_fixation, so set it null.
                    for im_idx = 1:2
                        image_times{im_idx}(tr_i,:) = nan;
                    end
                end
            end
        end
        
        session_data_summary.site_index_per_trial = site_index_per_trial;
        session_data_summary.sync_time_per_trial = sync_time_per_trial;
        session_data_summary.spike_times = spike_times;
        session_data_summary.image_times = image_times;
        
    end

end
