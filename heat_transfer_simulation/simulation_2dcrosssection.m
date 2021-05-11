% https://www.mathworks.com/help/pde/ug/heat-transfer-problem-with-temperature-dependent-properties.html#heatedBlockWithSlot-2

%%
thermalmodelT = createpde('thermal','transient');


w1 = 1.5 * 10^-3;   h1 = 7 * 10^-3;
w2 = 0.3 * 10^-3;   h2 = 5 * 10^-3;
w3 = 3 * 10^-3;   h3 = 7 * 10^-3;

im_w = 5 * 10^-3;
im_h = 10 * 10^-3;

T_BASELINE = 0;

r1 = [3 4 -w1/2 w1/2 w1/2 -w1/2  -h1/2 -h1/2 h1/2 h1/2];
r2 = [3 4 -w2/2 w2/2 w2/2 -w2/2  -h2/2 -h2/2 h2/2 h2/2];
r3 = [3 4 -w3/2-w1/2 w3/2-w1/2 w3/2-w1/2 -w3/2-w1/2  -h1/2 -h1/2 h1/2 h1/2];
gdm = [r1; r2; r3;]';
g = decsg(gdm,'R1+R2+R3',['R1'; 'R2'; 'R3']');
geometryFromEdges(thermalmodelT,g);

figure;
pdegplot(thermalmodelT,'EdgeLabels','on', 'SubdomainLabels','on'); 
title 'Block Geometry With Edge Labels Displayed'

%%
% silicone
thermalProperties(thermalmodelT,'ThermalConductivity', 0.2,... % 0.2 W/m·K
                                'MassDensity',1200,... % 1.2 g/cm3 or 1200 kg/m^3
                                'SpecificHeat',1300, ...  % 1050 to 1300 J/kg.K
                                'Face', 1); 
                    
% pcb
thermalProperties(thermalmodelT,'ThermalConductivity', 3, ... %3 W / m-K,... 
                                'MassDensity',1850,... % 1.850 g/cm3 or 1850 kg/m^3
                                'SpecificHeat',396, ...  % 396 J/kg-K
                                'Face', 3); 
                                
% brain
thermalProperties(thermalmodelT,'ThermalConductivity', 0.51,... % 0.2 W/m·K
                                'MassDensity',1100,... % 1.100 g/cm3 or 1200 kg/m^3
                                'SpecificHeat',3630, ...  % 3630  J/kg-K
                                'Face', 2); 
                                

%%
% 
% thermalProperties(thermalmodelT,'ThermalConductivity', 0.2,... % 0.2 W/m·K
%                                 'MassDensity',1100,... % 1.2 g/cm3 or 1200 kg/m^3
%                                 'SpecificHeat',1300); % 1050 to 1300 J/kg.K
                       


thermalBC(thermalmodelT,'Edge',1,'Temperature',@cross_section_boundary_condition);
thermalBC(thermalmodelT,'Edge',2,'Temperature',@cross_section_boundary_condition);
thermalBC(thermalmodelT,'Edge',9,'Temperature',@cross_section_boundary_condition);
thermalBC(thermalmodelT,'Edge',10,'Temperature',@cross_section_boundary_condition);
                            
msh = generateMesh(thermalmodelT,'Hmax',0.2 * 10^-3);
figure 
pdeplot(thermalmodelT); 
axis equal
title 'Block With Finite Element Mesh Displayed'

%%
tlist = 0:.1:5;
thermalIC(thermalmodelT, T_BASELINE);
R = solve(thermalmodelT,tlist);
T = R.Temperature;

getClosestNode = @(p,x,y) min((p(1,:) - x).^2 + (p(2,:) - y).^2);

[~,nid1] = getClosestNode( msh.Nodes, -w1/2, 0);
[~,nid2] = getClosestNode( msh.Nodes, -w2/2-0.5, 0);
% 
% h = figure;
% num_subplots = 4;
% plot_times = unique(floor(linspace(1,size(T,2),num_subplots)));
% h.Position = [1 1 2 1].*h.Position;
% for i = 1:length(plot_times)
%     subplot(2,2,i); 
% %     figure;
%     pdeplot(thermalmodelT,'XYData',T(:,plot_times(i)),'Contour','on','ColorMap','gray'); 
% %     title 'Temperature, Final Time, Transient Solution'
%     xlim([-im_w/2,im_w/2]);
%     ylim([-im_h/2,im_h/2])
% end
%%
figure; hold on;
axis equal
plot(tlist, T(nid1,:)); 
plot(tlist, T(nid2,:)); 
grid on
xlabel 'Time, seconds'
ylabel 'Temperature, degrees-Celsius'