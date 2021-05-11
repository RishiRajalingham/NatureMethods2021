function plot_summary_silencing_psth(meta_data, results_all)
addpath ../../utils/

%%
nsites = size(meta_data,1);
plot_data_all = cell(nsites,1);
for di = 1:nsites
    fn = meta_data{di,1};
    res = results_all{di};
    stim_duration = meta_data{di,2};
    figname = get_name_info(fn, stim_duration);
    
    keep = screen_data(res);
    if keep == 0
        continue;
    end
    
    time_info = get_time_info(res.time_ax, stim_duration);
    if isempty(time_info.t_to_plot)
        fprintf(1, 'time_info empty : %d \n', di);
        continue;
    end
    
    for s_i = 1 % multiunit only
        f = figure('Position', [680   624   377   474], 'Name', figname);
        x_raster_res = cell(1,2);
        for im_i = 1:2
            x_raster = res.X_raster{s_i, im_i} > 0;
            x_raster = remove_outlier_reps(x_raster);
            plot_psth(x_raster, time_info, im_i);
            plot_raster(x_raster, time_info, im_i);
            x_raster_res{im_i} = get_summary_to_save(x_raster, time_info);
        end
        plot_statistics(x_raster_res);
        saveas(f, sprintf('../fig_2020/%s_v2.pdf', figname));
    end
    
    plot_data_all_curr = [];
    plot_data_all_curr.fn = fn;
    plot_data_all_curr.figname = figname;
    plot_data_all_curr.stim_duration = stim_duration;
    plot_data_all_curr.x_raster_res = x_raster_res;
    plot_data_all{di} = plot_data_all_curr;
    
end

save('dat/summary_silencing_psth.mat', 'plot_data_all');

%% helper %%

    function figname = get_name_info(fn_, stim_duration_)
        fn_ = split(fn_, '/');
        fn_ = strrep(fn_{end}, '.mat', '');
        figname = sprintf('%s_%d', fn_, stim_duration_);
    end

    function time_info = get_time_info(time_ax, stim_duration_)
        t_orig = time_ax;
        if stim_duration_ == 200
            t_to_plot = t_orig >-100 & t_orig<300;
            t_on = -50+25; %appears to have a 25ms delay?
            t_off = t_on+stim_duration_;
        elseif stim_duration_ == 900
            t_to_plot = t_orig >-800 & t_orig < 300;
            t_on = -750+25;
            t_off = t_on+stim_duration_;
        else
            t_to_plot = [];
            t_on = [];
            t_off = [];
        end
        time_info.t_orig = t_orig;
        time_info.t_to_plot = t_to_plot;
        time_info.t_on = t_on;
        time_info.t_off = t_off;
    end

    function keep = screen_data(res_)
        ntr_1 = size(res_.X_raster{1,1}, 1);
        ntr_2 = size(res_.X_raster{1,2}, 1);
        min_reps = min(ntr_1,ntr_2);
        if min_reps < 5
            keep = 0;
        else
            
            t_visdrive = res_.time_ax>50 & res_.time_ax<150;
            t_baseline = res_.time_ax>-10 & res_.time_ax>50;
            fr_visdrive = nanmean(res_.X_raster{1,1}(:,t_visdrive),2);
            fr_baseline = nanmean(res_.X_raster{1,1}(:,t_baseline),2);
            keep = ttest2(fr_baseline, fr_visdrive);
        end
    end

    function x_raster_ = remove_outlier_reps(x_raster_)
        spikes_per_rep = sum(x_raster_, 2);
        t_keep = ~isoutlier(spikes_per_rep);
        x_raster_ = x_raster_(t_keep,:);
    end

    function plot_psth(x_raster, time_info, im_i)
        subplot(4,1,[3,4]); hold on;
        x_psth = smoothdata(x_raster, 2, 'movmean', 25) .* 1000;
        [x_mu,x_sem] = grpstats(x_psth, [], {'mean', 'sem'});
        
        t = time_info.t_orig(time_info.t_to_plot);
        x_mu = x_mu(time_info.t_to_plot);
        x_sem = x_sem(time_info.t_to_plot);
        
        if im_i == 1
            plot_section = t > -100;
            plot(t(plot_section), x_mu(plot_section), 'b-', 'linewidth', 2);
            plot(t(~plot_section), x_mu(~plot_section),'b--', 'linewidth', 0.5);
            [~, ~] = rr_boundedline(t(plot_section), x_mu(plot_section),...
                x_sem(plot_section), 'b');
        else
            plot(t, x_mu, 'r-', 'linewidth', 2);
            [~, ~] = rr_boundedline(t, x_mu, x_sem, 'r');
        end
        xline(0, 'linewidth', 2);
        xline(time_info.t_on, 'linewidth', 1, 'linestyle', '--');
        xline(time_info.t_off, 'linewidth', 1, 'linestyle', '--');
        axis tight;
    end

    function plot_raster(x_raster, time_info, im_i)
        ax1 = subplot(4,1,im_i); hold on;
        x_raster_to_plot = x_raster(:,time_info.t_to_plot);
        t = time_info.t_orig(time_info.t_to_plot);
        imagesc(x_raster_to_plot);
        set(gca, 'xtick', 50:50:size(x_raster_to_plot,2));
        set(gca, 'xticklabel', t(50:50:size(x_raster_to_plot,2)));
        xline(find(t==0), 'linewidth', 2);
        xline(find(t==time_info.t_on), 'linewidth', 1, 'linestyle', '--');
        xline(find(t==time_info.t_off), 'linewidth', 1, 'linestyle', '--');
        axis tight; axis off;
        
        if im_i == 1
            map = [1 1 1; 0 0 1.0];
        else
            map = [1 1 1; 1 0 0];
        end
        colormap(ax1, map)
    end

    function plot_statistics(x_raster_res_)
        x1 = x_raster_res_{1}.x_plot_psth;
        x2 = x_raster_res_{2}.x_plot_psth;
        t_p = x_raster_res_{1}.t_plot;
        
        p = ttest2(x1, x2);
        subplot(4,1,[3,4]); hold on;
        ylim auto;
        yax = ylim;
        yax = yax(2);
        for p_i = 1:length(p)
            if p(p_i) == 1
                plot(t_p(p_i), yax*1.1, 'k.');
            end
        end
        ylim auto;
    end

    function res_ = get_summary_to_save(x_raster, time_info)
        x_psth = smoothdata(x_raster, 2, 'movmean', 25) .* 1000;
        res_ = [];
        res_.t_plot = time_info.t_orig(time_info.t_to_plot);
        res_.x_plot = x_raster(:,time_info.t_to_plot);
        res_.x_plot_psth = x_psth(:,time_info.t_to_plot);
    end

    
end


