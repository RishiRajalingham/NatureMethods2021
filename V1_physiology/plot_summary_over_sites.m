function plot_summary_over_sites(plot_data_all)
nsites = length(plot_data_all);
col =  [0.11, 0.59, 0.1];
nboot = 100;
err_over_boot = 1;
normalize_opt = 'max_psth';



data_x = cell(2,1);
data_t = cell(2,1);
for i = 1:2
    data_x{i} = [];
    data_t{i} = [];
end

for i = 1:nsites
    if isempty(plot_data_all{i})
        continue
    end
    stim_dur = plot_data_all{i}.stim_duration;
    if stim_dur == 200
        stim_dur_idx = 1;
    else
        stim_dur_idx = 2;
    end
    
    x1 = plot_data_all{i}.x_raster_res{2}.x_plot_psth;
    x0 = plot_data_all{i}.x_raster_res{1}.x_plot_psth;
    mu_x1 = bootstrp(nboot, @nanmean, x1);
    mu_x0 = bootstrp(nboot, @nanmean, x0);
    
    del_x = mu_x1 - mu_x0;
    if strcmp(normalize_opt, 'max_psth')
        denom =mu_x1;
        denom2 = repmat(nanmax(denom, [], 2), 1,size(denom,2), 1);
        del_x = del_x ./ denom2;
    end
    data_x{stim_dur_idx} = cat(3, data_x{stim_dur_idx}, del_x);
    data_t{stim_dur_idx} = plot_data_all{i}.x_raster_res{1}.t_plot;
    
end


%%
figure;
for i = 1:2
    subplot(1,2,i); hold on;
    X = data_x{i};
    display(size(X))
    time = data_t{i};
    if i == 1
        t_on = -50+25; %appears to have a 25ms delay?
        t_off = t_on+200;
    else
        t_on = -750+25;
        t_off = t_on+900;
    end
    
    
    if err_over_boot
        mu = (nanmean(nanmean(X,3),1));
        sig = (nanstd(nanmean(X,3),1));
    else
        mu = (nanmean(nanmean(X,1),3));
        sig = (nansem(nanmean(X,1),3));
    end
    
    rr_boundedline(time, mu, sig, col);
    plot(time, mu, 'color', col, 'linewidth', 2);
    %     ylim([-0.1, 0.1])
    gridxy([t_on, t_off])
    %     plot(find(p < 0.05), 0, 'k*');
    yline(0); axis square;
end


%     figure,
%     for i = 1:2
%         subplot(1,2,i); hold on
%         [mu, sig]= grpstats(X{i}, [], {'mean', 'sem'});
%         for j = 1:size(X{i},1)
%             h = plot(T{i}, X{i}(j,:));
%             h.Color(4) = 0.1;
%
%         end
%         rr_boundedline(T{i}, mu, sig);
%         yline(0);
%     end
end